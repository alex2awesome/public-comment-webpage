from __future__ import annotations

import hashlib
import json
import math
import random
import re
import textwrap
from collections import deque, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import dspy
import numpy as np
import pandas as pd
import time

from autometrics.aggregator.regression.ElasticNet import ElasticNet
from autometrics.aggregator.regression.Regression import Regression
from autometrics.dataset.Dataset import Dataset
from autometrics.generator.Generator import Generator
from autometrics.generator.LLMJudgeProposer import BasicLLMJudgeProposer
from autometrics.generator.RubricGenerator import RubricGenerator
from autometrics.generator.utils import format_prior_metrics_summary
from autometrics.metrics.Metric import Metric
from autometrics.metrics.MultiMetric import MultiMetric

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


@dataclass
class RuntimeConfig:
    """Container for knobs that control the scaffolding loop."""

    evaluation_metric: Union[str, Callable[[pd.Series, pd.Series], float]] = "pearson"
    classification_threshold: float = 0.5
    min_improvement: float = 0.01
    train_ratio: float = 0.7
    meta_ratio: float = 0.2
    baseline_metrics: int = 10
    metrics_per_iteration: int = 4
    rubric_metrics_per_iteration: int = 4
    max_metrics_iterations: int = 10
    max_rubric_iterations: int = 5
    initial_support: int = 4
    support_growth: float = 0.5
    rubric_support_growth: int = 2
    high_residual_top_k: int = 5
    enable_rubric_phase: bool = True
    top_k_bins: int = 4
    verbose: bool = False
    show_progress: bool = False
    improvement_window: int = 5
    burn_in_iterations: int = 1
    patience: int = 5
    log_prompts: bool = False
    max_new_metrics_per_iteration: int = 2
    max_metric_ceiling: int = 12
    allow_metric_retirement: bool = False
    retire_persistence: int = 2
    retire_importance_eps: float = 0.05
    log_residual_explanations: bool = False
    example_selection_strategy: str = "residual"


@dataclass
class PhaseState:
    """Tracks the currently accepted model/metric set for each phase."""

    metrics: List[Metric]
    model: Regression
    train_dataset: Dataset
    meta_dataset: Dataset
    train_predictions: pd.Series
    residuals: pd.Series
    meta_predictions: pd.Series
    performance: float
    support_budget: int
    importance: Sequence[Tuple[float, str]] = ()
    importance_map: Dict[str, float] = None
    phase: str = "metrics"


@dataclass
class ExampleSelectionResult:
    """Represents a batch of examples stitched together for the next prompt."""

    dataframe: pd.DataFrame
    selection_type: str
    context_note: str = ""
    formatter: Optional[Callable] = None


class ScaffoldingProposer(Generator):
    """
    Implements the residual-driven two-phase metric/rubric expansion plan outlined in the
    research notes. It is intentionally verbose so it can be debugged or extended.

    The proposer:
        • Generates an initial metric bank and baseline regression.
        • Iteratively mines residuals to request additional metrics.
        • Switches to rubric-backed metrics once scalar axes stop improving.
        • Returns the final accepted metric set so downstream code can persist it.
    """

    def __init__(
        self,
        name: str = "ScaffoldingProposer",
        description: str = "Residual-driven metric/rubric proposer",
        generator_llm=None,
        executor_class: Optional[type] = None,
        executor_kwargs: Optional[dict] = None,
        seed: Optional[int] = None,
        regression_class: type = ElasticNet,
        metrics_phase_generator_cls: type = BasicLLMJudgeProposer,
        rubric_phase_generator_cls: type = RubricGenerator,
        *,
        verbose: bool = False,
        show_progress: bool = False,
        truncate_chars: Optional[int] = None,
        run_name: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            generator_llm=generator_llm,
            executor_class=executor_class,
            executor_kwargs=executor_kwargs or {},
            truncate_chars=truncate_chars,
        )
        self.seed = seed
        self.regression_class = regression_class
        self.metrics_phase_generator_cls = metrics_phase_generator_cls
        self.rubric_phase_generator_cls = rubric_phase_generator_cls
        self.history: List[dict] = []
        self.final_state: Optional[PhaseState] = None
        self.artifacts_base_dir: Optional[Path] = None
        self.verbose = verbose
        self.show_progress = show_progress
        self._current_verbose = verbose
        self._current_progress = show_progress
        self.metric_low_importance_streak: Dict[str, int] = {}
        self._rejected_metrics: OrderedDict[str, Dict[str, str]] = OrderedDict()
        self._user_run_name = run_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        dataset: Dataset,
        target_measure: Optional[str] = None,
        n_metrics: int = 10,
        formatter: Optional[Callable] = None,
        **kwargs,
    ) -> List[Metric]:
        """
        Entry point used by Autometrics. Returns the final accepted metric instances.
        """
        if not target_measure:
            target_measure = dataset.get_target_columns()[0]

        runtime_config = self._build_config(n_metrics=n_metrics, overrides=kwargs)
        train_dataset, meta_dataset = self._create_splits(dataset, runtime_config)
        self.artifacts_base_dir = self._prepare_artifacts_base_dir(dataset, target_measure)
        self._current_verbose = bool(runtime_config.verbose or self.verbose)
        self._current_progress = bool(runtime_config.show_progress or self.show_progress)
        self._rejected_metrics = OrderedDict()
        self._log(
            "Starting scaffolding run with "
            f"{len(train_dataset.get_dataframe())} train rows / "
            f"{len(meta_dataset.get_dataframe())} meta rows."
        )

        initial_metrics = self._generate_initial_metrics(
            dataset=train_dataset,
            target_measure=target_measure,
            n_metrics=runtime_config.baseline_metrics,
            log_prompts=runtime_config.log_prompts,
        )

        if not initial_metrics:
            print("[Scaffolding] No initial metrics were generated, aborting.")
            return []

        # Ensure both splits contain the metric score columns
        self._ensure_metric_scores(train_dataset, initial_metrics)
        self._ensure_metric_scores(meta_dataset, initial_metrics)

        # `initial_state` captures baseline regression fit on the seed metric set plus bookkeeping (predictions, residuals, support limit, etc.)
        initial_support = max(
            1,
            min(
                runtime_config.initial_support,
                len(initial_metrics),
                runtime_config.max_metric_ceiling,
            ),
        )
        initial_state = self._fit_phase_state(
            train_dataset=train_dataset,
            meta_dataset=meta_dataset,
            metrics=initial_metrics,
            target_measure=target_measure,
            support_budget=initial_support,
            phase="metrics",
            iteration_index=-1,
            config=runtime_config,
            previous_metric_names=[],
            max_new_metrics=runtime_config.max_metric_ceiling,
            max_total_metrics=runtime_config.max_metric_ceiling,
        )

        self._update_retirement_trackers(initial_state, runtime_config)
        state = self._run_metrics_phase(
            state=initial_state,
            target_measure=target_measure,
            config=runtime_config,
        )

        if runtime_config.enable_rubric_phase:
            state = self._run_rubric_phase(
                state=state,
                target_measure=target_measure,
                config=runtime_config,
            )

        self.final_state = state
        return state.metrics if state else []

    # ------------------------------------------------------------------
    # Phase orchestration
    # ------------------------------------------------------------------
    def _run_metrics_phase(
        self,
        state: PhaseState,
        target_measure: str,
        config: RuntimeConfig,
    ) -> PhaseState:
        return self._run_phase(
            state=state,
            target_measure=target_measure,
            config=config,
            phase_name="metrics",
            generator_factory=self._instantiate_metrics_phase_generator,
            metrics_per_iter=config.metrics_per_iteration,
        support_growth=config.support_growth,
            max_iterations=config.max_metrics_iterations,
        )

    def _run_rubric_phase(
        self,
        state: PhaseState,
        target_measure: str,
        config: RuntimeConfig,
    ) -> PhaseState:
        if config.max_rubric_iterations <= 0:
            return state
        return self._run_phase(
            state=state,
            target_measure=target_measure,
            config=config,
            phase_name="rubric",
            generator_factory=self._instantiate_rubric_phase_generator,
            metrics_per_iter=config.rubric_metrics_per_iteration,
            support_growth=config.rubric_support_growth,
            max_iterations=config.max_rubric_iterations,
        )

    def _run_phase(
        self,
        state: PhaseState,
        target_measure: str,
        config: RuntimeConfig,
        *,
        phase_name: str,
        generator_factory: Callable[[], Generator],
        metrics_per_iter: int,
        support_growth: int,
        max_iterations: int,
    ) -> PhaseState:
        """
        Shared core loop for metrics-only and rubric phases.
        """
        iterations = min(max_iterations, 50)
        window_size = max(1, config.improvement_window)
        improvement_buffer = deque(maxlen=window_size)
        best_state = state
        best_performance = state.performance
        patience_counter = 0
        progress_bar = None
        if self._current_progress and tqdm is not None:
            progress_bar = tqdm(
                range(iterations),
                desc=f"[Scaffolding][{phase_name}] iterations",
                leave=False,
            )
            iterator = progress_bar
        else:
            iterator = range(iterations)

        for iteration in iterator:
            selection = self._select_example_batch(
                state=state,
                target_measure=target_measure,
                config=config,
                phase_name=phase_name,
                iteration=iteration,
            )
            if selection is None or selection.dataframe.empty:
                print(f"[Scaffolding][{phase_name}] No candidate examples available for selection strategy '{getattr(config, 'example_selection_strategy', 'residual')}'.")
                break

            residual_summary = ""
            residual_explanation = ""
            if selection.selection_type == "residual":
                residual_summary, residual_explanation = self._maybe_explain_residuals(
                    state=state,
                    residual_examples=selection.dataframe,
                    target_measure=target_measure,
                    config=config,
                    phase_name=phase_name,
                    iteration=iteration,
                )
                if residual_summary and self._current_verbose:
                    snippet = self._summarize_text(residual_summary, width=200)
                    self._log(f"Residual summary: {snippet}")

            candidate_dataset = self._clone_dataset_with_dataframe(
                template=state.train_dataset,
                dataframe=selection.dataframe,
            )
            generator = generator_factory()
            prior_metrics_payload = self._get_prior_metric_context(state.metrics)
            residual_note = selection.context_note or ""
            if residual_explanation:
                truncated_explanation = textwrap.shorten(
                    residual_explanation.replace("\n\n", "\n"),
                    width=800,
                    placeholder=" …",
                )
                residual_note = (
                    residual_note
                    + "\n\nResidual analysis summary:\n"
                    + truncated_explanation
                )
            generate_kwargs = {
                "dataset": candidate_dataset,
                "target_measure": target_measure,
                "n_metrics": metrics_per_iter,
                "prior_metrics": prior_metrics_payload,
                "residual_context_note": residual_note,
            }
            if selection.formatter:
                generate_kwargs["formatter"] = selection.formatter
            if config.log_prompts:
                generate_kwargs["log_prompts"] = True
            candidate_metrics = generator.generate(
                **generate_kwargs,
            )
            prompt_payloads = []
            get_payloads = getattr(generator, "get_last_prompt_payloads", None)
            if callable(get_payloads):
                try:
                    prompt_payloads = list(get_payloads())
                except Exception:
                    prompt_payloads = []
            combined_metrics, added_metrics = self._integrate_metrics(state, candidate_metrics)
            if not added_metrics:
                print(f"[Scaffolding][{phase_name}] Generator produced no unique metrics.")
                break

            prev_names = [metric.get_name() for metric in state.metrics]
            support_budget = self._determine_support_budget(
                candidate_count=len(combined_metrics),
                previous_count=len(state.metrics),
                config=config,
            )

            updated_state = self._fit_phase_state(
                train_dataset=state.train_dataset,
                meta_dataset=state.meta_dataset,
                metrics=combined_metrics,
                target_measure=target_measure,
                support_budget=support_budget,
                phase=phase_name,
                iteration_index=iteration,
                config=config,
                previous_metric_names=prev_names,
                max_new_metrics=config.max_new_metrics_per_iteration,
                max_total_metrics=config.max_metric_ceiling,
            )

            new_performance = updated_state.performance
            improvement_vs_current = new_performance - state.performance
            improvement_vs_best = new_performance - best_performance
            improvement_buffer.append(improvement_vs_current)
            rolling_avg = sum(improvement_buffer) / len(improvement_buffer)
            accepted = improvement_vs_best >= config.min_improvement
            iterations_completed = iteration + 1
            burn_in_reached = iterations_completed >= max(1, config.burn_in_iterations)

            if accepted:
                patience_counter = 0
                best_performance = new_performance
                best_state = updated_state
            else:
                if burn_in_reached:
                    patience_counter += 1

            if progress_bar is not None:
                progress_bar.set_postfix(
                    {
                        "iter": iteration,
                        "impr": f"{improvement_vs_current:.4f}",
                        "best_impr": f"{improvement_vs_best:.4f}",
                        "avg": f"{rolling_avg:.4f}",
                        "score": f"{updated_state.performance:.4f}",
                        "accepted": accepted,
                        "pat": patience_counter,
                    },
                    refresh=True,
                )
            artifact_info = self._persist_iteration_artifacts(
                phase_name=phase_name,
                iteration=iteration,
                state=updated_state,
                added_metrics=added_metrics,
                target_measure=target_measure,
                accepted=accepted,
                previous_metrics=state.metrics,
                prompt_payloads=prompt_payloads,
                residual_explanation=residual_explanation,
            )
            self.history.append(
                {
                    "phase": phase_name,
                    "iteration": iteration,
                    "accepted": accepted,
                    "improvement": improvement_vs_current,
                    "improvement_vs_current": improvement_vs_current,
                    "improvement_vs_best": improvement_vs_best,
                    "performance": updated_state.performance,
                    "added_metrics": [m.get_name() for m in added_metrics],
                    "rolling_avg": rolling_avg,
                    "patience_counter": patience_counter,
                    **(artifact_info or {}),
                }
            )

            if accepted:
                print(f"[Scaffolding][{phase_name}] Accepted iteration {iteration} (+{improvement_vs_best:.4f}).")
                state = updated_state
                self._update_retirement_trackers(state, config)
                self._prune_rejected_metrics(state.metrics)
            else:
                self._record_rejected_metrics(added_metrics)
                self._log(
                    f"Iteration {iteration} not accepted "
                    f"(Δ_best={improvement_vs_best:.4f}, patience={patience_counter}). "
                    f"Rolling average over last {len(improvement_buffer)} iteration(s) = {rolling_avg:.4f}."
                )

            if burn_in_reached and patience_counter >= max(1, config.patience):
                print(
                    f"[Scaffolding][{phase_name}] Patience limit reached "
                    f"({patience_counter} consecutive non-improvements after burn-in {config.burn_in_iterations}); stopping."
                )
                break
        if progress_bar is not None:
            progress_bar.close()
        state = best_state
        return state

    def _determine_support_budget(
        self,
        candidate_count: int,
        previous_count: int,
        config: RuntimeConfig,
    ) -> int:
        """
        Compute how many metrics may participate in the next regression fit.
        """
        ceiling = max(1, config.max_metric_ceiling)
        if previous_count <= 0:
            return max(1, min(candidate_count, ceiling))
        incremental = previous_count + max(0, config.max_new_metrics_per_iteration)
        incremental += max(0.0, config.support_growth)
        target = max(previous_count, math.ceil(incremental))
        return max(1, min(candidate_count, ceiling, target))

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _build_config(
        self,
        n_metrics: int,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> RuntimeConfig:
        """
        Merge defaults with caller overrides.
        """
        cfg = RuntimeConfig()
        cfg.baseline_metrics = max(1, n_metrics)
        if overrides:
            for key, value in overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        # Ensure ratios leave at least one row for both train and meta-dev
        total_ratio = cfg.train_ratio + cfg.meta_ratio
        if total_ratio > 0.99:
            scale = 0.99 / total_ratio
            cfg.train_ratio *= scale
            cfg.meta_ratio *= scale
        return cfg

    def _create_splits(
        self,
        dataset: Dataset,
        config: RuntimeConfig,
    ) -> Tuple[Dataset, Dataset]:
        """
        Deterministically split dataset into training and meta-dev pools without shuffling.
        """
        df = dataset.get_dataframe().reset_index(drop=True)
        if len(df) < 5:
            raise ValueError("ScaffoldingProposer requires at least 5 rows to create splits.")

        id_col = dataset.get_data_id_column()
        if id_col and id_col in df.columns:
            df = df.sort_values(by=id_col).reset_index(drop=True)

        train_size = max(1, int(len(df) * config.train_ratio))
        meta_size = max(1, int(len(df) * config.meta_ratio))
        if train_size + meta_size > len(df):
            meta_size = max(1, len(df) - train_size)

        train_df = df.iloc[:train_size].reset_index(drop=True)
        meta_df = df.iloc[train_size:train_size + meta_size].reset_index(drop=True)

        train_dataset = self._clone_dataset_with_dataframe(dataset, train_df)
        meta_dataset = self._clone_dataset_with_dataframe(dataset, meta_df)
        return train_dataset, meta_dataset

    def _clone_dataset_with_dataframe(self, template: Dataset, dataframe: pd.DataFrame) -> Dataset:
        """
        Create a shallow clone of the template Dataset but with a new dataframe.
        """
        return Dataset(
            dataframe=dataframe.copy(),
            target_columns=template.get_target_columns(),
            ignore_columns=template.get_ignore_columns(),
            metric_columns=list(template.get_metric_columns()),
            name=template.get_name(),
            data_id_column=template.get_data_id_column(),
            model_id_column=template.get_model_id_column(),
            input_column=template.get_input_column(),
            output_column=template.get_output_column(),
            reference_columns=template.get_reference_columns(),
            metrics=[],
            task_description=template.task_description,
        )

    def _prepare_artifacts_base_dir(self, dataset: Dataset, target_measure: str) -> Path:
        """
        Create a unique directory for this run so we do not clobber prior scaffolding artifacts.
        """
        safe_dataset = dataset.get_name().replace(" ", "_")
        safe_target = target_measure.replace(" ", "_")
        root = Path("scaffolding_runs") / safe_dataset / safe_target
        root.mkdir(parents=True, exist_ok=True)

        if self._user_run_name:
            slug = self._sanitize_run_name(self._user_run_name)
            unique_dir = root / slug
            if unique_dir.exists():
                raise FileExistsError(
                    f"Scaffolding run directory '{unique_dir}' already exists; provide a new --scaffolding-run-name."
                )
        else:
            run_id = hashlib.sha1()
            run_id.update(str(time.time()).encode("utf-8"))
            run_id.update(str(self.seed or random.random()).encode("utf-8"))
            unique_dir = root / f"run_{run_id.hexdigest()[:8]}"
        unique_dir.mkdir(parents=True, exist_ok=True)
        return unique_dir

    @staticmethod
    def _sanitize_run_name(name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
        slug = slug.strip("_")
        if not slug:
            raise ValueError("Provided scaffolding run name is empty after sanitization.")
        return slug

    def _generate_initial_metrics(
        self,
        dataset: Dataset,
        target_measure: str,
        n_metrics: int,
        log_prompts: bool = False,
    ) -> List[Metric]:
        """
        Use the metrics-phase generator on a small balanced slice of the data.
        """
        reference_df = self._select_reference_examples(
            dataframe=dataset.get_dataframe(),
            target_column=target_measure,
            per_bucket=5,
        )
        reference_dataset = self._clone_dataset_with_dataframe(dataset, reference_df)
        generator = self._instantiate_metrics_phase_generator()
        generate_kwargs = {
            "dataset": reference_dataset,
            "target_measure": target_measure,
            "n_metrics": n_metrics,
        }
        if log_prompts:
            generate_kwargs["log_prompts"] = True
        metrics = generator.generate(**generate_kwargs)
        self._log(f"Generated {len(metrics)} baseline metric(s).")
        return metrics

    # ------------------------------------------------------------------
    # Regression + scoring utilities
    # ------------------------------------------------------------------
    def _fit_phase_state(
        self,
        train_dataset: Dataset,
        meta_dataset: Dataset,
        metrics: List[Metric],
        target_measure: str,
        support_budget: int,
        phase: str,
        iteration_index: int,
        config: RuntimeConfig,
        *,
        previous_metric_names: Optional[Sequence[str]] = None,
        max_new_metrics: Optional[int] = None,
        max_total_metrics: Optional[int] = None,
    ) -> PhaseState:
        """
        Fit a sparse regression on the given metric set and compute bookkeeping artifacts.
        """
        # Ensure matrices are computed
        self._ensure_metric_scores(train_dataset, metrics)
        self._ensure_metric_scores(meta_dataset, metrics)

        regression = self.regression_class(
            name=f"{phase.capitalize()}Regression",
            description=f"{phase} sparse regression",
            input_metrics=metrics,
            dataset=train_dataset,
        )
        regression.learn(train_dataset, target_column=target_measure)
        important = regression.identify_important_metrics()
        importance_map = {name: abs(coef) for coef, name in important}
        previous_names = list(previous_metric_names or [])
        max_total = max(1, min(support_budget, max_total_metrics or support_budget))
        new_limit = None if not previous_names else max_new_metrics
        active_names = self._select_active_metric_names(
            importance=important,
            importance_map=importance_map,
            max_metrics=max_total,
            fallback_metrics=[metric.get_name() for metric in metrics],
            previous_metrics=previous_names,
            max_new_metrics=new_limit,
            config=config,
        )

        active_metrics = [
            metric for metric in metrics if metric.get_name() in active_names
        ]

        if len(active_metrics) != len(metrics):
            # Re-fit using the truncated support to keep coefficients aligned with the sparse set.
            regression = self.regression_class(
                name=f"{phase.capitalize()}Regression",
                description=f"{phase} sparse regression",
                input_metrics=active_metrics,
                dataset=train_dataset,
            )
            regression.learn(train_dataset, target_column=target_measure)

        train_preds = pd.Series(
            regression.predict(train_dataset, update_dataset=False),
            index=train_dataset.get_dataframe().index,
        )
        meta_preds = pd.Series(
            regression.predict(meta_dataset, update_dataset=False),
            index=meta_dataset.get_dataframe().index,
        )

        residuals = self._compute_residuals(
            dataset=train_dataset,
            target_column=target_measure,
            predictions=train_preds,
        )
        performance = self._evaluate_downstream_performance(
            dataset=meta_dataset,
            target_column=target_measure,
            predictions=meta_preds,
            config=config,
        )

        print(
            f"[Scaffolding][{phase}] iteration {iteration_index} "
            f"support={len(active_metrics)} score={performance:.4f}"
        )
        self._log(
            f"Refit {phase} regression (iteration {iteration_index}) with "
            f"{len(active_metrics)} active metric(s); performance={performance:.4f}"
        )

        return PhaseState(
            metrics=active_metrics,
            model=regression,
            train_dataset=train_dataset,
            meta_dataset=meta_dataset,
            train_predictions=train_preds,
            residuals=residuals,
            meta_predictions=meta_preds,
            performance=performance,
            support_budget=len(active_metrics),
            importance=important,
            importance_map=importance_map,
            phase=phase,
        )

    def _integrate_metrics(
        self,
        state: PhaseState,
        candidates: Sequence[Metric],
    ) -> Tuple[List[Metric], List[Metric]]:
        """
        Merge candidate metrics into the state and ensure both datasets are scored.
        Returns (combined_metrics, added_metrics).
        """
        added = self._dedupe_metrics(state.metrics, candidates)
        if not added:
            return state.metrics, []
        for dataset in (state.train_dataset, state.meta_dataset):
            self._ensure_metric_scores(dataset, added)
        return state.metrics + added, added

    def _ensure_metric_scores(self, dataset: Dataset, metrics: Sequence[Metric]) -> None:
        df = dataset.get_dataframe()
        existing = {metric.get_name(): metric for metric in dataset.get_metrics()}
        for metric in metrics:
            name = metric.get_name()
            if isinstance(metric, MultiMetric):
                dataset.add_metric(metric, update_dataset=True)
                df = dataset.get_dataframe()
                self._log(f"Scored multi-metric '{name}' on dataset '{dataset.get_name()}'.")
                continue
            if name not in existing:
                dataset.add_metric(metric, update_dataset=True)
                existing[name] = metric
                df = dataset.get_dataframe()
                self._log(f"Added and scored new metric '{name}' on dataset '{dataset.get_name()}'.")
            elif name not in df.columns:
                metric.predict(dataset, update_dataset=True)
                df = dataset.get_dataframe()
                self._log(f"Updated scores for metric '{name}' on dataset '{dataset.get_name()}'.")
            if name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(name)

    def _dedupe_metrics(
        self,
        existing: Sequence[Metric],
        candidates: Sequence[Metric],
    ) -> List[Metric]:
        existing_names = {metric.get_name() for metric in existing}
        unique: List[Metric] = []
        for metric in candidates:
            name = metric.get_name()
            if name in existing_names:
                continue
            existing_names.add(name)
            unique.append(metric)
        return unique

    def _select_active_metric_names(
        self,
        importance: Sequence[Tuple[float, str]],
        importance_map: Dict[str, float],
        max_metrics: int,
        fallback_metrics: Sequence[str],
        previous_metrics: Sequence[str],
        max_new_metrics: Optional[int],
        config: RuntimeConfig,
    ) -> List[str]:
        """
        Choose up to max_metrics metric names ordered by importance, honoring retirement limits.
        """
        ordered = [name for _coef, name in importance]
        if not ordered:
            ordered = list(fallback_metrics)
        ordered += [name for name in fallback_metrics if name not in ordered]
        prev_set = set(previous_metrics)
        protected = {
            name
            for name in prev_set
            if not self._can_retire_metric(name, importance_map, config)
        }
        available_slots = max(0, max_metrics - len(protected))
        new_limit = max_new_metrics if max_new_metrics is not None else available_slots
        new_limit = min(new_limit, available_slots) if new_limit is not None else available_slots
        new_count = 0
        selected: List[str] = []

        for name in ordered:
            if name in selected:
                continue
            is_new = name not in prev_set
            if name in protected:
                selected.append(name)
            elif is_new:
                if new_limit is None or new_count < new_limit:
                    selected.append(name)
                    new_count += 1
            else:
                if config.allow_metric_retirement:
                    selected.append(name)
            if len(selected) >= max_metrics:
                break

        if len(selected) < max_metrics:
            for name in ordered:
                if name in selected:
                    continue
                if len(selected) >= max_metrics:
                    break
                selected.append(name)
        return selected[:max_metrics]

    def _compute_residuals(
        self,
        dataset: Dataset,
        target_column: str,
        predictions: pd.Series,
    ) -> pd.Series:
        """
        Compute signed residuals (true - predicted).
        """
        df = dataset.get_dataframe()
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' missing from dataset.")
        truth = pd.Series(df[target_column]).reindex(predictions.index)
        residuals = truth - predictions
        return residuals

    def _evaluate_downstream_performance(
        self,
        dataset: Dataset,
        target_column: str,
        predictions: pd.Series,
        config: RuntimeConfig,
    ) -> float:
        """
        Evaluate the downstream scalar objective. Supports a few built-in metrics.
        """
        df = dataset.get_dataframe()
        truth = pd.Series(df[target_column]).reindex(predictions.index)
        metric = config.evaluation_metric
        if callable(metric):
            return float(metric(truth, predictions))
        method = str(metric or "pearson").lower()
        if method in {"pearson", "spearman"}:
            corr = truth.corr(predictions, method=method)
            return float(corr if not math.isnan(corr) else 0.0)
        if method == "rmse":
            return float(np.sqrt(np.mean(np.square(truth - predictions))))
        if method == "mae":
            return float(np.mean(np.abs(truth - predictions)))
        if method == "accuracy":
            threshold = config.classification_threshold
            preds = (predictions >= threshold).astype(int)
            return float(np.mean((truth.astype(int)) == preds))
        raise ValueError(f"Unknown evaluation metric '{metric}'.")

    # ------------------------------------------------------------------
    # High-residual selection utilities
    # ------------------------------------------------------------------
    def _select_example_batch(
        self,
        *,
        state: PhaseState,
        target_measure: str,
        config: RuntimeConfig,
        phase_name: str,
        iteration: int,
    ) -> Optional[ExampleSelectionResult]:
        strategy = str(getattr(config, "example_selection_strategy", "residual") or "residual").lower()
        if strategy == "residual":
            dataframe = self._select_high_residual_examples(
                dataset=state.train_dataset,
                residuals=state.residuals,
                target_column=target_measure,
                top_k=config.high_residual_top_k,
                bins=config.top_k_bins,
            )
            note = ""
            if not dataframe.empty and state.residuals.shape[0] > 0:
                note = (
                    f"Iteration {iteration} of the {phase_name} phase surfaces "
                    f"{len(dataframe)} high-residual example(s) that the current metrics score poorly. "
                    "Develop new metrics that address these failure modes without duplicating the existing axes."
                )
            return ExampleSelectionResult(
                dataframe=dataframe,
                selection_type="residual",
                context_note=note,
            )
        if strategy == "matching":
            return self._select_matching_pairs(
                dataset=state.train_dataset,
                predictions=state.train_predictions,
                target_column=target_measure,
                top_k=config.high_residual_top_k,
                phase_name=phase_name,
                iteration=iteration,
            )
        if strategy == "random":
            dataframe = self._select_random_examples(
                dataset=state.train_dataset,
                target_column=target_measure,
                top_k=config.high_residual_top_k,
                bins=config.top_k_bins,
                iteration=iteration,
            )
            note = (
                f"Iteration {iteration} of the {phase_name} phase uses a stratified random sample "
                f"of {len(dataframe)} example(s) to explore new parts of the corpus. Consider how fresh metrics "
                "might generalize beyond the prior residual set."
            )
            return ExampleSelectionResult(
                dataframe=dataframe,
                selection_type="random",
                context_note=note,
            )
        raise ValueError(
            f"Unknown example selection strategy '{strategy}'. "
            "Expected one of: 'residual', 'matching', or 'random'."
        )

    def _select_high_residual_examples(
        self,
        dataset: Dataset,
        residuals: pd.Series,
        target_column: str,
        top_k: int,
        bins: int,
    ) -> pd.DataFrame:
        """
        Stratify residuals by label stratum and grab the worst-scoring examples.
        """
        df = dataset.get_dataframe().copy()
        residual_series = residuals.reindex(df.index)
        df["_abs_residual"] = residual_series.abs()
        selected = self._stratified_select(
            dataframe=df,
            labels=df[target_column],
            per_group=top_k,
            bins=bins,
            sort_key="_abs_residual",
        )
        return selected.drop(columns=["_abs_residual"], errors="ignore")

    def _select_random_examples(
        self,
        dataset: Dataset,
        target_column: str,
        top_k: int,
        bins: int,
        iteration: int,
    ) -> pd.DataFrame:
        df = dataset.get_dataframe()
        if df.empty:
            return df
        random_seed = (self.seed or 0) + int(iteration or 0) + 1
        shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        selected = self._stratified_select(
            dataframe=shuffled,
            labels=shuffled[target_column],
            per_group=top_k,
            bins=bins,
            sort_key=None,
        )
        return selected.reset_index(drop=True)

    def _select_matching_pairs(
        self,
        dataset: Dataset,
        predictions: pd.Series,
        target_column: str,
        top_k: int,
        phase_name: str,
        iteration: int,
    ) -> ExampleSelectionResult:
        df = dataset.get_dataframe().copy()
        if df.empty:
            return ExampleSelectionResult(df, "matching", context_note="")
        prediction_series = predictions.reindex(df.index)
        df["_prediction"] = prediction_series
        df = df.dropna(subset=[target_column, "_prediction"])
        if df.empty:
            return ExampleSelectionResult(df, "matching", context_note="")

        df = df.reset_index(drop=True)
        neighbor_window = 5
        pairs: List[Tuple[float, pd.Series, pd.Series]] = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            label = row[target_column]
            for offset in range(1, neighbor_window + 1):
                other_idx = idx + offset
                if other_idx >= len(df):
                    break
                other = df.iloc[other_idx]
                other_label = other[target_column]
                if pd.isna(other_label) or label == other_label:
                    continue
                gap = abs(float(row["_prediction"]) - float(other["_prediction"]))
                pairs.append((gap, row, other))
                break

        if not pairs:
            empty_df = pd.DataFrame(columns=df.columns)
            return ExampleSelectionResult(
                dataframe=empty_df,
                selection_type="matching",
                context_note="No label-mismatched pairs with similar predictions were found.",
            )

        pairs.sort(key=lambda item: item[0])
        selected_pairs = pairs[: max(1, min(top_k, len(pairs)))]
        max_gap = max((gap for gap, _, _ in selected_pairs), default=1.0) or 1.0
        metric_columns = dataset.get_metric_columns() or []
        formatter = self._build_pair_formatter(column="pair_prompt")
        pair_rows: List[Dict[str, Any]] = []
        summary_lines: List[str] = []
        for pair_id, (gap, row_a, row_b) in enumerate(selected_pairs, start=1):
            prompt = self._compose_pair_prompt(
                pair_id=pair_id,
                row_a=row_a,
                row_b=row_b,
                dataset=dataset,
                target_column=target_column,
                metric_columns=metric_columns,
                prediction_gap=gap,
            )
            summary_lines.append(
                f"Pair {pair_id}: label {row_a[target_column]} vs {row_b[target_column]}, "
                f"pred {float(row_a['_prediction']):.3f} / {float(row_b['_prediction']):.3f}, gap {gap:.3f}"
            )
            pair_rows.append(
                {
                    target_column: float(max_gap - gap),
                    "pair_prompt": prompt,
                    "pair_prediction_gap": gap,
                    "pair_label_a": row_a[target_column],
                    "pair_label_b": row_b[target_column],
                    "pair_prediction_a": float(row_a["_prediction"]),
                    "pair_prediction_b": float(row_b["_prediction"]),
                }
            )

        pair_df = pd.DataFrame(pair_rows)
        context_note = (
            f"Iteration {iteration} of the {phase_name} phase surfaces {len(pair_rows)} matched pair(s) "
            "where the ensemble predicts nearly identical scores even though the labels differ. "
            "Each example in the dataset includes both comments plus their current metric outputs—propose new metrics that can cleanly separate these ambiguous pairs."
        )
        if summary_lines:
            context_note = context_note + "\nProblem pairs:\n" + "\n".join(f"- {line}" for line in summary_lines)
        return ExampleSelectionResult(
            dataframe=pair_df,
            selection_type="matching",
            context_note=context_note,
            formatter=formatter,
        )

    @staticmethod
    def _build_pair_formatter(column: str) -> Callable:
        def _formatter(row_tuple):
            _, row = row_tuple
            return str(row.get(column, ""))

        return _formatter

    def _compose_pair_prompt(
        self,
        pair_id: int,
        row_a: pd.Series,
        row_b: pd.Series,
        dataset: Dataset,
        target_column: str,
        metric_columns: Sequence[str],
        prediction_gap: float,
    ) -> str:
        input_column = dataset.get_input_column()
        output_column = dataset.get_output_column()
        text_a = self._shorten_text(row_a.get(input_column)) if input_column else ""
        text_b = self._shorten_text(row_b.get(input_column)) if input_column else ""
        output_a = self._shorten_text(row_a.get(output_column)) if output_column else ""
        output_b = self._shorten_text(row_b.get(output_column)) if output_column else ""
        metrics_summary: List[str] = []
        metric_limit = min(8, len(metric_columns))
        for metric_name in metric_columns[:metric_limit]:
            val_a = row_a.get(metric_name)
            val_b = row_b.get(metric_name)
            if pd.isna(val_a) and pd.isna(val_b):
                continue
            metrics_summary.append(
                f"{metric_name}: A={self._format_metric_value(val_a)} | B={self._format_metric_value(val_b)}"
            )
        metrics_block = "\n".join(metrics_summary) if metrics_summary else "Metric snapshots unavailable."
        label_a = row_a.get(target_column)
        label_b = row_b.get(target_column)
        pred_a = float(row_a.get("_prediction", 0.0))
        pred_b = float(row_b.get("_prediction", 0.0))
        prompt_parts = [
            f"Pair {pair_id}: the ensemble assigns {pred_a:.3f} vs {pred_b:.3f} despite labels {label_a} vs {label_b}.",
            f"Prediction gap: {prediction_gap:.3f}. The goal is to surface criteria that separate these cases.",
            "Example A:",
            f"  Label: {label_a} | Current output: {output_a or 'N/A'}",
            f"  Comment: {text_a or 'N/A'}",
            "Example B:",
            f"  Label: {label_b} | Current output: {output_b or 'N/A'}",
            f"  Comment: {text_b or 'N/A'}",
            "Metric snapshots (A vs B):",
            metrics_block,
        ]
        return "\n".join(prompt_parts)

    @staticmethod
    def _shorten_text(value: Any, width: int = 400) -> str:
        if value is None:
            return ""
        text = str(value)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        if len(text) <= width:
            return text
        return textwrap.shorten(text, width=width, placeholder=" …")

    @staticmethod
    def _format_metric_value(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (float, np.floating)):
            if math.isnan(value):
                return "NA"
            return f"{value:.3f}"
        return str(value)

    def _bin_labels(self, labels: pd.Series, bins: int) -> pd.Series:
        """
        Convert labels to coarse bins for stratified sampling.
        """
        unique_values = labels.dropna().unique()
        if len(unique_values) <= bins:
            return labels.fillna("missing").astype(str)
        quantiles = np.linspace(0, 1, num=bins + 1)
        return pd.qcut(labels, q=quantiles, duplicates="drop")

    def _stratified_select(
        self,
        dataframe: pd.DataFrame,
        labels: pd.Series,
        per_group: int,
        bins: int,
        sort_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generic helper to grab per-group samples, optionally ranked by a score column.
        """
        binned = self._bin_labels(labels, bins=bins)
        selections: List[pd.DataFrame] = []
        for _, group in dataframe.assign(_bin=binned).groupby("_bin"):
            if group.empty:
                continue
            if sort_key:
                subset = group.nlargest(per_group, sort_key)
            else:
                subset = group.head(per_group)
            selections.append(subset)
        if selections:
            result = pd.concat(selections, ignore_index=True)
        else:
            result = dataframe.head(per_group)
        return result.drop(columns=["_bin"], errors="ignore")

    def _update_retirement_trackers(
        self,
        state: PhaseState,
        config: RuntimeConfig,
    ) -> None:
        """
        Update low-importance streak counters after an accepted iteration.
        """
        if not state or not state.metrics:
            self.metric_low_importance_streak.clear()
            return
        active_names = {metric.get_name() for metric in state.metrics}
        for name in active_names:
            importance = 0.0
            if state.importance_map:
                importance = state.importance_map.get(name, 0.0)
            if importance <= config.retire_importance_eps:
                self.metric_low_importance_streak[name] = self.metric_low_importance_streak.get(name, 0) + 1
            else:
                self.metric_low_importance_streak[name] = 0
        for name in list(self.metric_low_importance_streak):
            if name not in active_names:
                self.metric_low_importance_streak.pop(name, None)

    def _can_retire_metric(
        self,
        name: str,
        importance_map: Dict[str, float],
        config: RuntimeConfig,
    ) -> bool:
        if not config.allow_metric_retirement:
            return False
        importance = importance_map.get(name, 0.0)
        if importance > config.retire_importance_eps:
            return False
        streak = self.metric_low_importance_streak.get(name, 0)
        return streak >= max(1, config.retire_persistence)

    # ------------------------------------------------------------------
    # Residual explanation utilities
    # ------------------------------------------------------------------
    def _maybe_explain_residuals(
        self,
        *,
        state: PhaseState,
        residual_examples: pd.DataFrame,
        target_measure: str,
        config: RuntimeConfig,
        phase_name: str,
        iteration: int,
    ) -> Optional[str]:
        if not config.log_residual_explanations:
            summary = self._format_residual_example_summary(
                state=state,
                residual_examples=residual_examples,
                target_measure=target_measure,
            )
            return (summary, None) if summary.strip() else (None, None)
        summary = self._format_residual_example_summary(
            state=state,
            residual_examples=residual_examples,
            target_measure=target_measure,
        )
        if not summary.strip():
            return (None, None)
        metrics_summary = format_prior_metrics_summary(self._get_prior_metric_context(state.metrics))
        context = (
            f"Dataset={state.train_dataset.get_name()} phase={phase_name} "
            f"iteration={iteration} target={target_measure}"
        )
        if not self.generator_llm:
            return (summary, None)
        explainer = ResidualFailureExplanationModule()
        try:
            with dspy.settings.context(lm=self.generator_llm):
                prediction = explainer(
                    residual_examples=summary,
                    metrics_summary=metrics_summary,
                    task_context=context,
                )
            explanation = getattr(prediction, "analysis", "") or ""
            explanation = explanation.strip()
            if explanation and len(explanation) > 600:
                explanation = self._summarize_text(explanation, width=600)
            return (summary, explanation or summary)
        except Exception as exc:  # pragma: no cover - best-effort logging
            warning = f"Residual explanation failed ({exc}); using fallback summary."
            return (summary, warning)

    def _format_residual_example_summary(
        self,
        *,
        state: PhaseState,
        residual_examples: pd.DataFrame,
        target_measure: str,
        max_examples: int = 5,
    ) -> str:
        df = residual_examples.copy()
        id_col = state.train_dataset.get_data_id_column()
        lines: List[str] = []
        for _, row in df.head(max_examples).iterrows():
            idx = row.name
            doc_id = row.get(id_col, f"row_{idx}")
            truth = row.get(target_measure, "NA")
            prediction = state.train_predictions.reindex([idx]).iloc[0] if idx in state.train_predictions.index else float("nan")
            residual = state.residuals.reindex([idx]).iloc[0] if idx in state.residuals.index else float("nan")
            pred_str = f"{prediction:.3f}" if isinstance(prediction, (float, int)) and not math.isnan(prediction) else "NA"
            resid_str = f"{residual:.3f}" if isinstance(residual, (float, int)) and not math.isnan(residual) else "NA"
            snippet = self._extract_text_snippet(row)
            lines.append(
                f"- ID {doc_id}: target={truth}, prediction={pred_str}, residual={resid_str}. Snippet: {snippet}"
            )
        return "\n".join(lines)

    def _extract_text_snippet(self, row: pd.Series) -> str:
        candidates = (
            "actual_comment",
            "responses_to_comments",
            "response_to_comment",
            "text",
            "call",
            "content",
        )
        for field in candidates:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                snippet = value.replace("\n", " ").strip()
                return textwrap.shorten(snippet, width=280, placeholder=" …")
        return "<no text available>"

    def _select_reference_examples(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        per_bucket: int,
    ) -> pd.DataFrame:
        """
        Grab a balanced handful of high/low examples per label bucket.
        """
        return self._stratified_select(
            dataframe=dataframe,
            labels=dataframe[target_column],
            per_group=per_bucket,
            bins=2,
        )

    def _persist_iteration_artifacts(
        self,
        *,
        phase_name: str,
        iteration: int,
        state: PhaseState,
        added_metrics: Sequence[Metric],
        target_measure: str,
        accepted: bool,
        previous_metrics: Sequence[Metric],
        prompt_payloads: Optional[Sequence[Dict[str, Any]]] = None,
        residual_explanation: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save metric definitions and score matrices for the iteration.
        """
        if not self.artifacts_base_dir:
            return {}

        iteration_dir = (
            self.artifacts_base_dir / phase_name / f"iteration_{iteration:02d}"
        )
        iteration_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = iteration_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        prompts_dir = iteration_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        explanation_path = None
        if residual_explanation:
            explanation_path = prompts_dir / "residual_explanation.json"
            try:
                with open(explanation_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "phase": phase_name,
                            "iteration": iteration,
                            "analysis": residual_explanation,
                        },
                        fh,
                        indent=2,
                        ensure_ascii=False,
                    )
            except Exception as exc:  # pragma: no cover
                print(f"[Scaffolding] Warning: failed to persist residual explanation: {exc}")
                explanation_path = None

        metrics_meta: List[Dict[str, str]] = []
        for metric in state.metrics:
            safe_name = metric.get_name().replace(" ", "_")
            metric_path = metrics_dir / f"{safe_name}.py"
            metric.save_python_code(str(metric_path))
            metrics_meta.append(
                {
                    "name": metric.get_name(),
                    "description": getattr(metric, "description", ""),
                    "path": str(metric_path),
                }
            )

        train_scores_path = iteration_dir / "train_scores.csv"
        meta_scores_path = iteration_dir / "meta_scores.csv"

        self._write_score_snapshot(
            dataset=state.train_dataset,
            metrics=state.metrics,
            target_column=target_measure,
            predictions=state.train_predictions,
            residuals=state.residuals,
            output_path=train_scores_path,
        )

        meta_residuals = self._compute_residuals(
            dataset=state.meta_dataset,
            target_column=target_measure,
            predictions=state.meta_predictions,
        )
        self._write_score_snapshot(
            dataset=state.meta_dataset,
            metrics=state.metrics,
            target_column=target_measure,
            predictions=state.meta_predictions,
            residuals=meta_residuals,
            output_path=meta_scores_path,
        )

        decisions: List[Dict[str, str]] = []
        previous_map = {
            metric.get_name(): getattr(metric, "description", "")
            for metric in previous_metrics
        }
        current_names = {metric.get_name() for metric in state.metrics}
        added_names = {metric.get_name() for metric in added_metrics}
        recorded: set[str] = set()

        for name, description in previous_map.items():
            status = "kept" if name in current_names else "eliminated"
            decisions.append(
                {
                    "name": name,
                    "status": status,
                    "description": description,
                }
            )
            recorded.add(name)

        for metric in added_metrics:
            name = metric.get_name()
            if name in recorded:
                continue
            status = "added_kept" if name in current_names else "added_eliminated"
            decisions.append(
                {
                    "name": name,
                    "status": status,
                    "description": getattr(metric, "description", ""),
                }
            )

        decisions_path = iteration_dir / "metric_decisions.json"
        try:
            with open(decisions_path, "w", encoding="utf-8") as fh:
                json.dump(decisions, fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[Scaffolding] Warning: failed to log metric decisions: {exc}")

        generation_prompt_path = prompts_dir / "generation_prompt.json"
        if prompt_payloads:
            try:
                with open(generation_prompt_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "phase": phase_name,
                            "iteration": iteration,
                            "prompt_attempts": list(prompt_payloads),
                        },
                        fh,
                        indent=2,
                        ensure_ascii=False,
                    )
            except Exception as exc:
                print(f"[Scaffolding] Warning: failed to persist prompt log: {exc}")
        else:
            generation_prompt_path = None

        for metric in added_metrics:
            prompt_context = getattr(metric, "generation_prompt_context", None)
            if not prompt_context:
                continue
            safe_name = metric.get_name().replace(" ", "_")
            prompt_path = prompts_dir / f"{safe_name}_prompt.json"
            try:
                with open(prompt_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "metric_name": metric.get_name(),
                            "description": getattr(metric, "description", ""),
                            "context": prompt_context,
                        },
                        fh,
                        indent=2,
                        ensure_ascii=False,
                    )
            except Exception as exc:
                print(f"[Scaffolding] Warning: failed to persist metric prompt for {metric.get_name()}: {exc}")

        metadata_path = iteration_dir / "metrics.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "phase": phase_name,
                        "iteration": iteration,
                        "accepted": accepted,
                        "metrics": metrics_meta,
                        "added_metrics": [m.get_name() for m in added_metrics],
                    },
                    fh,
                    indent=2,
                )
        except Exception as exc:
            print(f"[Scaffolding] Warning: failed to write metric metadata: {exc}")

        return {
            "train_scores_path": str(train_scores_path),
            "meta_scores_path": str(meta_scores_path),
            "metrics_dir": str(metrics_dir),
            "metrics_metadata_path": str(metadata_path),
            "metric_decisions_path": str(decisions_path),
            "prompts_dir": str(prompts_dir),
            "generation_prompt_path": str(generation_prompt_path) if generation_prompt_path else "",
            "residual_explanation_path": str(explanation_path) if explanation_path else "",
        }

    def _write_score_snapshot(
        self,
        *,
        dataset: Dataset,
        metrics: Sequence[Metric],
        target_column: str,
        predictions: pd.Series,
        residuals: pd.Series,
        output_path: Path,
    ) -> None:
        df = dataset.get_dataframe().copy()
        metric_cols = [m.get_name() for m in metrics]
        snapshot = pd.DataFrame()

        id_col = dataset.get_data_id_column()
        if id_col and id_col in df.columns:
            snapshot[id_col] = df[id_col]
        else:
            snapshot["row_index"] = df.index

        if target_column in df.columns:
            snapshot[target_column] = df[target_column]

        snapshot["prediction"] = predictions.reindex(df.index).values
        snapshot["residual"] = residuals.reindex(df.index).values

        for col in metric_cols:
            if col in df.columns:
                snapshot[col] = df[col]

        snapshot.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # Metric set bookkeeping
    # ------------------------------------------------------------------
    def _summarize_metrics(self, metrics: Sequence[Metric], *, status: str = "active") -> List[Dict[str, str]]:
        """
        Build a lightweight summary of metric names and descriptions for prompt context.
        """
        summaries: List[Dict[str, str]] = []
        for metric in metrics:
            name = metric.get_name() if hasattr(metric, "get_name") else getattr(metric, "name", "UnnamedMetric")
            if hasattr(metric, "get_description"):
                description = metric.get_description()
            else:
                description = getattr(metric, "description", "")
            summaries.append(
                {
                    "name": name,
                    "description": description or "",
                    "status": status,
                }
            )
        return summaries

    def _get_prior_metric_context(self, active_metrics: Sequence[Metric]) -> List[Dict[str, str]]:
        """
        Combine active and previously rejected metrics for downstream prompts.
        """
        summaries = self._summarize_metrics(active_metrics, status="active")
        summaries.extend(list(self._rejected_metrics.values()))
        return summaries

    def _log(self, message: str) -> None:
        """
        Emit a verbose log message when verbosity is enabled.
        """
        if self._current_verbose:
            print(f"[Scaffolding][debug] {message}")

    def _deduplicate_metrics(
        self,
        existing: Sequence[Metric],
        candidates: Sequence[Metric],
    ) -> List[Metric]:
        """
        Drop candidate metrics whose names already exist.
        """
        existing_names = {metric.get_name() for metric in existing}
        unique: List[Metric] = []
        for metric in candidates:
            if metric.get_name() in existing_names:
                continue
            existing_names.add(metric.get_name())
            unique.append(metric)
        return unique

    def _merge_metric_sets(
        self,
        current: Sequence[Metric],
        new_metrics: Sequence[Metric],
    ) -> List[Metric]:
        """
        Produce a combined, name-deduplicated list preserving order.
        """
        seen = set()
        merged: List[Metric] = []
        for metric in list(current) + list(new_metrics):
            name = metric.get_name()
            if name in seen:
                continue
            seen.add(name)
            merged.append(metric)
        return merged

    def _record_rejected_metrics(self, metrics: Sequence[Metric]) -> None:
        """
        Remember metrics that failed acceptance so we do not regenerate them blindly.
        """
        if not metrics:
            return
        for summary in self._summarize_metrics(metrics, status="rejected"):
            self._rejected_metrics[summary["name"]] = summary

    def _prune_rejected_metrics(self, active_metrics: Sequence[Metric]) -> None:
        """
        Remove entries from the rejected cache once a metric becomes active again.
        """
        if not active_metrics or not self._rejected_metrics:
            return
        for metric in active_metrics:
            self._rejected_metrics.pop(metric.get_name(), None)

    def _summarize_text(self, text: str, width: int = 400) -> str:
        """
        Compact helper to elide long residual explanations for logging/prompts.
        """
        cleaned = " ".join(text.split())
        return textwrap.shorten(cleaned, width=width, placeholder=" …")

    # ------------------------------------------------------------------
    # Generator instantiation helpers
    # ------------------------------------------------------------------
    def _instantiate_metrics_phase_generator(self) -> Generator:
        """
        Create the generator used during the metrics-only phase.
        """
        return self.metrics_phase_generator_cls(
            generator_llm=self.generator_llm,
            executor_class=self.executor_class,
            executor_kwargs=self.executor_kwargs,
            seed=self.seed,
            truncate_chars=self.formatter_truncate_chars,
        )

    def _instantiate_rubric_phase_generator(self) -> Generator:
        """
        Create the generator used once rubrics are required.
        """
        return self.rubric_phase_generator_cls(
            generator_llm=self.generator_llm,
            executor_class=self.executor_class,
            executor_kwargs=self.executor_kwargs,
            use_prometheus=False,
            seed=self.seed,
            truncate_chars=self.formatter_truncate_chars,
        )


class ResidualFailureExplanationSignature(dspy.Signature):
    """Prompt signature for explaining residual failure patterns."""

    residual_examples: str = dspy.InputField(
        desc="Bullet list of the highest-residual training examples with target/pred/residual details."
    )
    metrics_summary: str = dspy.InputField(
        desc="Summary of the currently accepted metrics (name + description)."
    )
    task_context: str = dspy.InputField(
        desc="High-level description of the dataset, phase, and iteration."
    )
    analysis: str = dspy.OutputField(
        desc="Short explanation of why these examples remain high-residual and guidance for future metrics."
    )


class ResidualFailureExplanationModule(dspy.Module):
    """DSPy wrapper that asks the LLM to reason about residual failures."""

    def __init__(self):
        super().__init__()
        self.explain = dspy.ChainOfThought(ResidualFailureExplanationSignature)

    def forward(
        self,
        residual_examples: str,
        metrics_summary: str,
        task_context: str,
    ):
        result = self.explain(
            residual_examples=residual_examples,
            metrics_summary=metrics_summary,
            task_context=task_context,
        )
        return dspy.Prediction(analysis=result.analysis)
