"""
Lightweight scriptized version of the EPA Autometrics notebook snippet so it can be
debugged or run under a standard Python interpreter.

Workflow:
    1. Load the EPA CSV and derive the `c` label denoting whether the response column
       contains text.
    2. Filter to rows that have a `call`, a long-form `actual_comment`, and pass the
       word-count bounds.
    3. Downsample to keep the label distribution balanced and sample the rows that
       will feed Autometrics.
    4. Run the Autometrics pipeline with rubric-based metric generation
       (default) or the residual scaffolding generator.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Optional

import dspy
import pandas as pd

from autometrics.aggregator.regression.ElasticNet import ElasticNet
from autometrics.autometrics import Autometrics
from autometrics.dataset.Dataset import Dataset

here = Path(__file__).parent.parent
repo_root = here / ".." / ".."

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate the EPA Autometrics workflow from the notebook."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default= repo_root / "notebooks" / "full_matched_comment_df__epa.csv",
        help="CSV containing the EPA comments.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of rows to draw for the Autometrics Dataset.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=500,
        help="Lower bound on `actual_comment` word count (exclusive).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=5_000,
        help="Upper bound on `actual_comment` word count (exclusive).",
    )
    parser.add_argument(
        "--downsample-random-state",
        type=int,
        default=42,
        help="Random seed used when balancing the dataset.",
    )
    parser.add_argument(
        "--sample-random-state",
        type=int,
        default=42,
        help="Random seed used for the final `.sample()` call.",
    )
    parser.add_argument(
        "--generator-model",
        default="openai/gpt-5-mini",
        help="Model ID for rubric generation prompts.",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-5-mini",
        help="Model ID for Autometrics execution.",
    )
    parser.add_argument(
        "--initial-core-metrics",
        type=int,
        default=10,
        help="Number of metrics in the initial core set before scaffolding begins.",
    )
    parser.add_argument(
        "--scaffolding-metrics-per-iteration",
        type=int,
        default=4,
        help="Number of new metrics the scaffolding loop requests each iteration.",
    )
    parser.add_argument(
        "--num-to-retrieve",
        type=int,
        default=5,
        help="Metrics to retrieve from the bank.",
    )
    parser.add_argument(
        "--max-new-metrics-per-iteration",
        type=int,
        default=2,
        help="Maximum number of new metrics that can be added to the core set per iteration.",
    )
    parser.add_argument(
        "--max-metric-ceiling",
        type=int,
        default=12,
        help="Upper bound on the number of core metrics retained after any iteration.",
    )
    parser.add_argument(
        "--allow-retire-old-metrics",
        action="store_true",
        help="Permit dropping previously accepted metrics when a new combination improves validation performance.",
    )
    parser.add_argument(
        "--retire-persistence",
        type=int,
        default=2,
        help="Number of consecutive accepted iterations a replacement metric must survive before an old metric can retire.",
    )
    parser.add_argument(
        "--retire-importance-eps",
        type=float,
        default=0.05,
        help="Minimum absolute importance/weight drop that an old metric must fall below before it is eligible for retirement.",
    )
    parser.add_argument(
        "--explain-residuals",
        action="store_true",
        help="Ask the generator LLM to explain the high-residual examples before proposing new metrics and log the responses.",
    )
    parser.add_argument(
        "--generated-metrics-dir",
        default="tutorial_metrics_epa_rubrics",
        help="Directory where generated rubric metrics will be written.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=("rubric", "scaffolding"),
        default="rubric",
        help="Metric generation strategy: rubric-driven or residual scaffolding.",
    )
    parser.add_argument(
        "--log-prompts",
        action="store_true",
        help="Log full prompts sent to generator LLMs (very verbose).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable tqdm progress bars and per-iteration logs.",
    )
    parser.add_argument(
        "--residual-top-k",
        type=int,
        default=5,
        help="Number of high-residual examples per label bin to feed into each metric-generation prompt.",
    )
    parser.add_argument(
        "--residual-bins",
        type=int,
        default=4,
        help="How many label bins to use when stratifying residuals (higher = more diverse residual sampling).",
    )
    parser.add_argument(
        "--resume-from-run",
        type=str,
        default=None,
        help="Optional run_id of a previous scaffolding run to resume (found under scaffolding_runs/.../run_<id>).",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional explicit path to a checkpoint JSON file to resume from.",
    )
    parser.add_argument(
        "--prompt-truncate-chars",
        type=int,
        default=12_000,
        help="Max characters from each input/output/reference copied into LLM prompts (<=0 disables truncation).",
    )
    parser.add_argument(
        "--scaffolding-run-name",
        type=str,
        default=None,
        help="Optional label for the scaffolding run directory (e.g., 'run_dec09_iter1'). Must be unique.",
    )
    return parser.parse_args()


def downsample_df(
    df: pd.DataFrame,
    label_col: str = "c",
    *,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Balance the dataset so each label value has the same number of rows.
    """
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not present in dataframe.")

    counts = df[label_col].value_counts(dropna=False)
    if counts.empty:
        return df.copy()

    min_count = int(counts.min())
    return (
        df.groupby(label_col, group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min_count,
                random_state=random_state,
            )
            if len(group) >= min_count
            else group
        )
        .reset_index(drop=True)
    )


def load_filtered_dataframe(
    csv_path: Path,
    *,
    min_words: int,
    max_words: int,
    downsample_state: Optional[int],
) -> pd.DataFrame:
    """
    Mirror the filtering logic from the notebook snippet before arriving at Autometrics.
    """
    epa_df = pd.read_csv(csv_path, index_col=0).assign(
        c=lambda frame: frame["comment_in_response"].notnull().astype(int),
    )
    filtered = (
        epa_df.loc[lambda frame: frame["call"].notnull()]
        .loc[lambda frame: frame["actual_comment"].notnull()]
        .loc[
            lambda frame: frame["actual_comment"]
            .str.split()
            .str.len()
            .gt(min_words)
        ]
        .loc[
            lambda frame: frame["actual_comment"]
            .str.split()
            .str.len()
            .lt(max_words)
        ]
    )
    return downsample_df(filtered, random_state=downsample_state)


def build_dataset(
    dataframe: pd.DataFrame,
    *,
    sample_size: int,
    sample_state: Optional[int],
) -> Dataset:
    sampled = dataframe.sample(sample_size, random_state=sample_state).reset_index(
        drop=True
    )
    return Dataset(
        dataframe=sampled,
        target_columns=["c"],
        ignore_columns=["index", "comment_in_response", "responses_to_comments"],
        metric_columns=[],
        name="MyCustomDataset_EPA",
        data_id_column="index",
        input_column="call",
        output_column="actual_comment",
        task_description=(
            "Rank candidate policy feedback drafts for follow-up: given a citizen’s "
            "submission, determine which responses merit escalation to agency officials."
        ),
    )


def sanitize_run_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    slug = slug.strip("_")
    if not slug:
        raise ValueError("Provided scaffolding run name is empty after sanitization.")
    return slug


def run_pipeline(args: argparse.Namespace) -> None:
    truncate_chars = args.prompt_truncate_chars if args.prompt_truncate_chars > 0 else None
    sanitized_run_name: Optional[str] = None
    if args.scaffolding_run_name:
        sanitized_run_name = sanitize_run_name(args.scaffolding_run_name)

    dataframe = load_filtered_dataframe(
        args.data_path,
        min_words=args.min_words,
        max_words=args.max_words,
        downsample_state=args.downsample_random_state,
    )

    if len(dataframe) < args.sample_size:
        raise ValueError(
            f"Filtered dataframe has only {len(dataframe)} rows; cannot sample "
            f"{args.sample_size}. Relax your filters or lower --sample-size."
        )

    dataset = build_dataset(
        dataframe,
        sample_size=args.sample_size,
        sample_state=args.sample_random_state,
    )
    target_measure = "c"

    run_root: Optional[Path] = None
    existing_run_dirs: set[str] = set()
    if sanitized_run_name:
        run_root = Path("scaffolding_runs") / dataset.get_name().replace(" ", "_") / target_measure
        run_root.mkdir(parents=True, exist_ok=True)
        expected_dir = run_root / sanitized_run_name
        if expected_dir.exists():
            raise FileExistsError(
                f"Scaffolding run directory '{expected_dir}' already exists; choose a new --scaffolding-run-name."
            )
        existing_run_dirs = {p.name for p in run_root.iterdir() if p.is_dir()}

    generator_llm = dspy.LM(args.generator_model)
    judge_llm = dspy.LM(args.judge_model)
    final_regression_cap = max(1, args.max_metric_ceiling)

    if args.generation_mode == "scaffolding":
        print("Running autometrics pipeline in scaffolding mode...")
        print("This will:")
        print(
            f"1. Run ScaffoldingProposer with {args.initial_core_metrics} baseline metric(s)"
        )
        print(
            f"2. Iterate on residuals, requesting {args.scaffolding_metrics_per_iteration} new metric(s) per loop"
        )
        print(f"3. Retrieve {args.num_to_retrieve} relevant metrics from the bank")
        print("4. Evaluate all metrics on your dataset")
        print(
            f"5. Allow up to {args.max_new_metrics_per_iteration} new metrics per round (ceiling {args.max_metric_ceiling})"
        )
        print(f"6. Select top {final_regression_cap} using ElasticNet regression")
        print("7. Create a final aggregated metric")
        metric_generation_configs = {
            "scaffolding": {
                "metrics_per_trial": args.initial_core_metrics,
                "generator_kwargs": {
                    "metrics_per_iteration": args.scaffolding_metrics_per_iteration,
                    "max_new_metrics_per_iteration": args.max_new_metrics_per_iteration,
                    "max_metric_ceiling": args.max_metric_ceiling,
                    "allow_metric_retirement": args.allow_retire_old_metrics,
                    "log_residual_explanations": args.explain_residuals,
                    "retire_persistence": args.retire_persistence,
                    "retire_importance_eps": args.retire_importance_eps,
                    "verbose": args.verbose,
                    "show_progress": args.verbose,
                    "high_residual_top_k": args.residual_top_k,
                    "top_k_bins": args.residual_bins,
                    "run_name": sanitized_run_name,
                },
            },
        }
        if args.log_prompts:
            metric_generation_configs["scaffolding"]["generator_kwargs"]["log_prompts"] = True
    else:
        print("Running autometrics pipeline...")
        print("This will:")
        print(f"1. Generate {args.initial_core_metrics} LLM judge metric(s)")
        print(f"2. Retrieve {args.num_to_retrieve} relevant metrics from the bank")
        print("3. Evaluate all metrics on your dataset")
        print(f"4. Select top {final_regression_cap} using ElasticNet regression")
        print("5. Create a final aggregated metric")
        metric_generation_configs = {
            "rubric_dspy": {
                "metrics_per_trial": args.initial_core_metrics,
                "generator_kwargs": {},
            },
            "llm_judge": {"metrics_per_trial": 0},
        }
        if args.log_prompts:
            metric_generation_configs["rubric_dspy"]["generator_kwargs"]["log_prompts"] = True

    autometrics = Autometrics(
        metric_generation_configs=metric_generation_configs,
        regression_strategy=ElasticNet,
        seed=42,
        generated_metrics_dir=args.generated_metrics_dir,
        formatter_truncate_chars=truncate_chars,
    )

    results = autometrics.run(
        dataset=dataset,
        target_measure=target_measure,
        generator_llm=generator_llm,
        judge_llm=judge_llm,
        num_to_retrieve=args.num_to_retrieve,
        num_to_regress=final_regression_cap,
        regenerate_metrics=True,
        resume_run_id=args.resume_from_run,
        resume_checkpoint_path=args.resume_checkpoint,
    )

    print("Pipeline complete! 🎉")
    if "final_metric" in results:
        print(f"Final metric name: {results['final_metric'].get_name()}")
    if sanitized_run_name and run_root:
        expected_dir = run_root / sanitized_run_name
        if not expected_dir.exists():
            current_dirs = [p for p in run_root.iterdir() if p.is_dir()]
            new_dirs = [p for p in current_dirs if p.name not in existing_run_dirs]
            candidate = None
            if len(new_dirs) == 1:
                candidate = new_dirs[0]
            elif new_dirs:
                candidate = max(new_dirs, key=lambda path: path.stat().st_mtime)
            if candidate:
                candidate.rename(expected_dir)
            else:
                print(
                    f"[Scaffolding] Warning: expected artifacts for run '{sanitized_run_name}' but could not locate a new directory under {run_root}."
                )
        if expected_dir.exists():
            print(f"[Scaffolding] Artifacts stored in: {expected_dir}")


if __name__ == "__main__":
    run_pipeline(parse_args())
