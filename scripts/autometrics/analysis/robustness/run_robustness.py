#!/usr/bin/env python3
"""
Robustness CSV runner: add metric scores as columns (in-place) for a chosen method.

Supported modes (choose one via --mode):
  - best_existing: run a specific existing metric class on the CSV
  - llm_judge: run an LLM-as-a-judge metric seeded by --seed
  - dna_eval: run DNAEval seeded by --seed
  - autometrics: reserved (to be implemented later)

If the target column already exists and is fully populated (no NaNs), the script skips recomputation.

This script reuses the generic CSV runner helpers and metric logic from existing analysis scripts.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import importlib.util
import inspect

# Ensure project root is importable
PROJECT_ROOT = "/nlp/scr2/nlp/personal-rm/autometrics"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Reuse the general CSV runner API
from analysis.robustness.run_metric_on_csv import (
    run_metric_or_aggregator_on_dataframe,
    import_by_path,
)

# Metric bank for existing metrics
from autometrics.metrics.MetricBank import all_metric_classes, build_metrics


def resolve_metric_class(name_or_path: str):
    """
    Resolve a metric class given either a full import path or a class name present in MetricBank.
    """
    if "." in name_or_path:
        return import_by_path(name_or_path)
    for cls in all_metric_classes:
        if cls.__name__ == name_or_path:
            return cls
    raise ValueError(f"Metric class not found: {name_or_path}")


# --- LLM Judge helpers (reuse custom classes defined in analysis scripts) ----
def create_llm_model(model_name: str, api_base: Optional[str], seed: int):
    """Create a dspy.LM like run_llm_judge_correlation does (minimal subset)."""
    import dspy
    temperature = 0.0001 * seed
    if model_name == "gpt4o_mini":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Please export OPENAI_API_KEY when using gpt4o_mini.")
        return dspy.LM("openai/gpt-4o-mini", api_key=api_key, temperature=temperature)
    elif model_name == "gpt5_mini":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Please export OPENAI_API_KEY when using gpt5_mini.")
        return dspy.LM("openai/gpt-5-mini", api_key=api_key, temperature=temperature)
    elif model_name == "qwen3_32b":
        base_url = api_base or "http://localhost:7410/v1"
        return dspy.LM("litellm_proxy/Qwen/Qwen3-32B", api_base=base_url, temperature=temperature, max_tokens=4096)
    elif model_name == "llama3_70b":
        base_url = api_base or "http://localhost:7410/v1"
        return dspy.LM("litellm_proxy/meta-llama/Llama-3.3-70B-Instruct", api_base=base_url, temperature=temperature)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_llm_judge_metric(
    prompt: str,
    score_range: Tuple[float, float],
    model_name: str,
    api_base: Optional[str],
    seed: int,
    with_references: bool,
    metric_name: Optional[str] = None,
    task_description: str = "",
):
    # Import the custom classes from the analysis script to avoid reimplementation
    from analysis.main_experiments.run_llm_judge_correlation import (
        CustomLLMJudgeMetricRefFree,
        CustomLLMJudgeMetricRefBased,
    )

    lm = create_llm_model(model_name, api_base, seed)
    base = metric_name or f"LLMJudge-{model_name}-seed{seed}"
    if with_references:
        return CustomLLMJudgeMetricRefBased(
            name=base,
            prompt=prompt,
            score_range=score_range,
            model=lm,
            task_description=task_description,
            seed=seed,
            model_name=model_name,
        )
    else:
        return CustomLLMJudgeMetricRefFree(
            name=base,
            prompt=prompt,
            score_range=score_range,
            model=lm,
            task_description=task_description,
            seed=seed,
            model_name=model_name,
        )


# --- DNAEval helper ----------------------------------------------------------
def build_dna_eval_metric(model_name: str, api_base: Optional[str], seed: int, task_description: Optional[str], metric_name: Optional[str]):
    # Use implementations from analysis/main_experiments
    from analysis.main_experiments.run_dna_eval import DNAEval, create_llm_model as dna_create_llm
    lm = dna_create_llm(model_name, api_base, seed)
    name = metric_name or f"DNAEval-{model_name}-seed{seed}"
    return DNAEval(name=name, model=lm, task_description=task_description)


def column_full(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().all() and (len(df[col]) > 0)


# --- Autometrics static regression loader ------------------------------------
def _infer_family_from_llm_model(llm_model: Optional[str]) -> Optional[str]:
    if not llm_model:
        return None
    token = llm_model.lower()
    if "qwen" in token:
        return "qwen"
    if "llama" in token:
        return "llama"
    if "gpt" in token or "openai" in token:
        return "openai"
    return None


def _find_static_regression_file(dataset: str, measure: str, seed: int, preferred_family: Optional[str]) -> str:
    """Search results/main_runs/autometrics for a matching static regression file.

    Expected layout examples:
      results/main_runs/autometrics/qwen/SimpEval_score/StaticRegression_SimpEval_score_seed42.py
      results/main_runs/autometrics/qwen/Primock57_time_sec/StaticRegression_Primock57_time_sec_seed42.py
    """
    base_dir = os.path.join(PROJECT_ROOT, "results", "main_runs", "autometrics")
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Autometrics results directory not found: {base_dir}")

    target_filename = f"StaticRegression_{dataset}_{measure}_seed{seed}.py"

    candidate_paths: List[Tuple[int, str]] = []
    # priority: preferred_family exact match gets priority 0, others get 1
    for fam in sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]):
        fam_dir = os.path.join(base_dir, fam)
        ds_dir = os.path.join(fam_dir, f"{dataset}_{measure}")
        fp = os.path.join(ds_dir, target_filename)
        if os.path.exists(fp):
            priority = 0 if preferred_family and fam == preferred_family else 1
            candidate_paths.append((priority, fp))

    if not candidate_paths:
        # Fallback: perform broader walk (in case structure differs)
        for fam in [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]:
            fam_dir = os.path.join(base_dir, fam)
            for root, _dirs, files in os.walk(fam_dir):
                if target_filename in files:
                    priority = 0 if preferred_family and fam == preferred_family else 1
                    candidate_paths.append((priority, os.path.join(root, target_filename)))

    if not candidate_paths:
        raise FileNotFoundError(
            f"Could not locate static regression file for {dataset}.{measure} seed={seed} under {base_dir}"
        )

    # Pick best priority then shortest path as tie-breaker
    candidate_paths.sort(key=lambda t: (t[0], len(t[1])))
    return candidate_paths[0][1]


def _import_module_from_path(file_path: str):
    module_name = f"autometrics_static_{abs(hash(file_path))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to construct import spec for: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _resolve_autometrics_aggregator_class(module, measure: str):
    """Prefer the canonical class name; otherwise scan for a GeneratedStaticRegressionAggregator subclass."""
    expected = f"Autometrics_Regression_{measure}_StaticRegression"
    if hasattr(module, expected):
        return getattr(module, expected)
    # Fallback: scan
    try:
        from autometrics.aggregator.generated.GeneratedRegressionMetric import GeneratedStaticRegressionAggregator
    except Exception:
        GeneratedStaticRegressionAggregator = None  # type: ignore
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name.endswith("_StaticRegression"):
            if GeneratedStaticRegressionAggregator is None:
                # No import available; best-effort by name
                return obj
            try:
                if issubclass(obj, GeneratedStaticRegressionAggregator):
                    return obj
            except Exception:
                continue
    raise ImportError("Failed to locate Autometrics static regression class in module")


def parse_score_range(range_str: str) -> Tuple[float, float]:
    s = range_str.strip()
    for ch in "[]()":
        s = s.replace(ch, "")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError("--score-range must be like '0,4' or '[0,4]'")
    lo = float(parts[0])
    hi = float("inf") if parts[1].lower() in ("inf", "infinity") else float(parts[1])
    return (lo, hi)

# --- Defaults from repo CSVs -------------------------------------------------
def _load_llm_judge_prompts_csv(csv_path: str) -> Dict[Tuple[str, str], Tuple[str, Tuple[float, float]]]:
    """Return mapping (dataset, measure) -> (prompt, (lo, hi))."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read LLM judge prompts from {csv_path}: {e}")
    mapping: Dict[Tuple[str, str], Tuple[str, Tuple[float, float]]] = {}
    for _, row in df.iterrows():
        ds = str(row['dataset'])
        ms = str(row['measure'])
        prompt = str(row['prompt'])
        sr = str(row['score_range'])
        mapping[(ds, ms)] = (prompt, parse_score_range(sr))
    return mapping


def _load_best_metrics_csv(csv_path: str) -> Dict[Tuple[str, str], str]:
    """Return mapping (dataset, measure) -> metric_class string."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read best metrics from {csv_path}: {e}")
    mapping: Dict[Tuple[str, str], str] = {}
    for _, row in df.iterrows():
        ds = str(row['dataset'])
        ms = str(row['measure'])
        cls_name = str(row['metric_class'])
        mapping[(ds, ms)] = cls_name
    return mapping


def _dataset_default_columns(dataset: str, measure: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """
    Return (input_column, output_column, reference_columns) defaults for known datasets.
    measure is unused for most, but included for future variations.
    """
    if not dataset:
        return None, None, None
    key = dataset.lower()
    if key.startswith("cogym"):
        # CoGym uses query -> formatted_conversation (process) or outcome (outcome)
        # We can't infer eval_type from dataset string reliably here, default to formatted_conversation;
        # callers for outcome tasks can override via args.
        return "query", "formatted_conversation", []
    if key == "helpsteer" or key == "helpsteer2":
        return "prompt", "response", []
    if key == "simpeval":
        return "original", "simple", ["ref1"]
    if key == "simpda":
        return "original", "simple", ["ref1"]
    if key == "primock57":
        return "transcript", "generated_note", ["human_note", "eval_note", "edited_note"]
    if key == "realhumaneval":
        # Our internal dataset wrapper constructs an "input" column; robustness CSV uses prefix/suffix
        # For robustness CSVs, we typically have: prefix_code, suffix_code, suggestion.
        # Prefer input->suggestion if present; otherwise default to prefix_code+suffix_code not supported here.
        return "input", "suggestion", []
    return None, None, None


def _auto_detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """Best-effort auto-detection of input/output/reference columns from CSV.
    Prefers unified robustness format: input, model_output, ref{n}.
    """
    cols = set(df.columns)
    # Detect references: ref1, ref2, ... (case-insensitive startswith)
    ref_cols: List[str] = []
    for c in df.columns:
        lc = str(c).lower()
        if lc.startswith("ref"):
            ref_cols.append(c)
    # Prefer unified names
    inp = "input" if "input" in cols else None
    out = "model_output" if "model_output" in cols else None
    # Fallbacks (kept minimal)
    if inp is None:
        for cand in ["original", "prompt", "query", "transcript"]:
            if cand in cols:
                inp = cand
                break
    if out is None:
        for cand in ["output", "response", "simple", "suggestion", "generated_note", "outcome", "formatted_conversation", "original_output"]:
            if cand in cols:
                out = cand
                break
    return inp, out, (ref_cols if ref_cols else None)


def main():
    parser = argparse.ArgumentParser(description="Run robustness metric on a CSV and write scores back.")
    parser.add_argument("--csv", required=True, help="Path to CSV to update in-place (or use --output to write elsewhere)")
    parser.add_argument("--mode", required=True, choices=["best_existing", "llm_judge", "dna_eval", "autometrics"], help="Which metric type to run")
    parser.add_argument("--input-column", default=None, help="Column with input/prompt (optional; auto-detected)")
    parser.add_argument("--output-column", default=None, help="Column with model output (optional; auto-detected)")
    parser.add_argument("--reference-columns", default=None, help="Comma-separated reference columns (optional)")
    parser.add_argument("--output", dest="output_csv", default=None, help="Optional output CSV path (default: overwrite input)")
    parser.add_argument("--metric-name", default=None, help="Override output column name for the metric")
    parser.add_argument("--dataset", default=None, help="Dataset key for defaults (e.g., SimpEval, HelpSteer)")
    parser.add_argument("--measure", default=None, help="Measure key for defaults (e.g., score, helpfulness)")
    parser.add_argument("--cache-dir", default=None, help="Optional AUTOMETRICS_CACHE_DIR override")

    # Best existing metric
    parser.add_argument("--metric-class", default=None, help="Metric class name or import path (best_existing mode; default pulls from results/best_metrics.csv when --dataset/--measure provided)")

    # LLM Judge
    parser.add_argument("--llm-model", default="gpt4o_mini", choices=["gpt4o_mini", "qwen3_32b", "llama3_70b", "gpt5_mini"], help="Model backend for LLM Judge / DNAEval")
    parser.add_argument("--api-base", default=None, help="API base for non-OpenAI-compatible backends")
    parser.add_argument("--llm-prompt", default=None, help="Judge prompt/axis text (default pulls from analysis/main_experiments/llm_judge_prompts.csv when --dataset/--measure provided)")
    parser.add_argument("--score-range", default="0,4", help="Score range, e.g., '0,4' or '0,inf' (default overridden by llm_judge_prompts.csv when available)")
    parser.add_argument("--task-description", default="", help="Optional task description for LLM Judge / DNAEval")

    # Seeding
    parser.add_argument("--seed", type=int, default=42, help="Seed for LLM-based metrics and cache")
    parser.add_argument("--force-recompute", action="store_true", help="Recompute even if target metric column already exists and is populated")
    # Autometrics specific (optional): allow users to hint the model family directory
    parser.add_argument("--autometrics-family", default=None, help="Optional model family subdir under results/main_runs/autometrics (e.g., qwen). Defaults to infer from --llm-model or search.")

    args = parser.parse_args()

    # Load CSV
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)

    # First try auto-detection from unified format
    a_in, a_out, a_refs = _auto_detect_columns(df)

    # Start from CLI args or auto-detected
    input_col = args.input_column or a_in
    output_col = args.output_column or a_out
    ref_cols: Optional[List[str]] = None
    if args.reference_columns:
        ref_cols = [c.strip() for c in args.reference_columns.split(",") if c.strip()]
    elif a_refs:
        ref_cols = a_refs

    # If still missing and dataset provided, fill with per-dataset defaults (legacy fallback)
    if args.dataset and (not input_col or not output_col or ref_cols is None):
        d_in, d_out, d_refs = _dataset_default_columns(args.dataset, args.measure)
        if not input_col and d_in:
            input_col = d_in
        if not output_col and d_out:
            output_col = d_out
        if ref_cols is None and d_refs:
            ref_cols = d_refs

    # Validate presence (at least input/output are required for all our metric modes)
    if not input_col or not output_col:
        raise ValueError("Could not determine input/output columns. Provide --input-column/--output-column or ensure the CSV has 'input' and 'model_output'.")

    # Cache dir
    if args.cache_dir:
        os.environ["AUTOMETRICS_CACHE_DIR"] = args.cache_dir

    # Determine metric instance per mode
    metric_instance = None
    metric_name = args.metric_name

    if args.mode == "best_existing":
        metric_class_name = args.metric_class
        # Default from results/best_metrics.csv
        if metric_class_name is None and args.dataset and args.measure:
            best_csv = os.path.join(PROJECT_ROOT, "results", "best_metrics.csv")
            best_map = _load_best_metrics_csv(best_csv)
            metric_class_name = best_map.get((args.dataset, args.measure))
            if metric_class_name is None:
                raise ValueError(f"No best metric found for {args.dataset}.{args.measure} in {best_csv}")
        if not metric_class_name:
            raise ValueError("--metric-class is required for best_existing mode when no --dataset/--measure provided")
        cls = resolve_metric_class(metric_class_name)
        # Prefer MetricBank factory to inherit defaults/GPU planning
        built = build_metrics(classes=[cls], cache_dir=args.cache_dir, seed=args.seed, use_cache=True)
        if not built:
            raise RuntimeError(f"Failed to instantiate metric class {cls.__name__}")
        metric_instance = built[0]
        if metric_name:
            try:
                metric_instance.name = metric_name
            except Exception:
                pass
        metric_name = metric_name or getattr(metric_instance, "name", cls.__name__)

    elif args.mode == "llm_judge":
        prompt = args.llm_prompt
        score_rng = None
        # Default from analysis/main_experiments/llm_judge_prompts.csv
        if args.dataset and args.measure:
            prompts_csv = os.path.join(PROJECT_ROOT, "analysis", "main_experiments", "llm_judge_prompts.csv")
            llm_map = _load_llm_judge_prompts_csv(prompts_csv)
            if (args.dataset, args.measure) in llm_map:
                default_prompt, default_range = llm_map[(args.dataset, args.measure)]
                if prompt is None:
                    prompt = default_prompt
                if args.score_range == "0,4":
                    score_rng = default_range
        if prompt is None:
            raise ValueError("No LLM judge prompt provided. Supply --llm-prompt or set --dataset/--measure present in llm_judge_prompts.csv")
        if score_rng is None:
            score_rng = parse_score_range(args.score_range)
        with_refs = bool(ref_cols and len(ref_cols) > 0)
        metric_instance = build_llm_judge_metric(
            prompt=prompt,
            score_range=score_rng,
            model_name=args.llm_model,
            api_base=args.api_base,
            seed=args.seed,
            with_references=with_refs,
            metric_name=args.metric_name,
            task_description=args.task_description,
        )
        metric_name = getattr(metric_instance, "name", args.metric_name or f"LLMJudge-{args.llm_model}-seed{args.seed}")

    elif args.mode == "dna_eval":
        metric_instance = build_dna_eval_metric(
            model_name=args.llm_model,
            api_base=args.api_base,
            seed=args.seed,
            task_description=args.task_description,
            metric_name=args.metric_name,
        )
        metric_name = getattr(metric_instance, "name", args.metric_name or f"DNAEval-{args.llm_model}-seed{args.seed}")

    elif args.mode == "autometrics":
        # Locate the appropriate exported static regression file
        preferred_family = args.autometrics_family or _infer_family_from_llm_model(args.llm_model)
        static_path = _find_static_regression_file(
            dataset=args.dataset,
            measure=args.measure,
            seed=args.seed,
            preferred_family=preferred_family,
        )
        module = _import_module_from_path(static_path)
        cls = _resolve_autometrics_aggregator_class(module, measure=args.measure)
        try:
            metric_instance = cls()
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Autometrics aggregator from {static_path}: {e}")
        metric_name = args.metric_name or getattr(metric_instance, "name", f"Autometrics_Regression_{args.measure}")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Skip if already full
    if metric_name and metric_name in df.columns and not args.force_recompute:
        if column_full(df, metric_name):
            print(f"✅ Column '{metric_name}' already present and fully populated. Skipping computation.")
            out_path = args.output_csv or args.csv
            if out_path != args.csv:
                df.to_csv(out_path, index=False)
            return
        else:
            missing = int(df[metric_name].isna().sum())
            total = len(df)
            print(f"🔁 Column '{metric_name}' already present but has {missing}/{total} missing. Recomputing column (full pass).")

    # Compute and write back
    df_out = run_metric_or_aggregator_on_dataframe(
        metric_or_aggregator=metric_instance,
        df=df,
        input_column=input_col,
        output_column=output_col,
        reference_columns=ref_cols,
        metric_name=metric_name,
        with_feedback=False,
        dependencies=None,
        init_params=None,
        cache_dir=args.cache_dir,
    )

    out_csv = args.output_csv or args.csv
    df_out.to_csv(out_csv, index=False)
    print(f"✅ Wrote results to: {out_csv}")


if __name__ == "__main__":
    main()


# Example Usage:

# python analysis/robustness/run_robustness.py --csv /nlp/scr2/nlp/personal-rm/autometrics/outputs/robustness/csvs/simpeval_score/original_subset.csv --mode llm_judge --dataset SimpEval --measure score --llm-model qwen3_32b --api-base http://sphinx3.stanford.edu:8544/v1 --seed 42 --metric-name llm_judge_qwen32b_s42
# python analysis/robustness/run_robustness.py --csv /nlp/scr2/nlp/personal-rm/autometrics/outputs/robustness/csvs/simpeval_score/original_subset.csv --mode dna_eval --dataset SimpEval --measure score --llm-model qwen3_32b --api-base http://sphinx3.stanford.edu:8544/v1 --seed 42 --metric-name dna_eval_qwen32b_s42
# python analysis/robustness/run_robustness.py --csv /nlp/scr2/nlp/personal-rm/autometrics/outputs/robustness/csvs/simpeval_score/original_subset.csv --mode best_existing --dataset SimpEval --measure score --seed 42 --metric-name best_existing_simpeval