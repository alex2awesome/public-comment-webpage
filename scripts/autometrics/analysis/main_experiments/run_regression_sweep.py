#!/usr/bin/env python3
"""
Quick regression-sweep runner for Autometrics.

Runs the Autometrics pipeline through Step 4 (metric evaluation) once, then
recomputes the regression selection and test correlations for multiple values
of num_to_regress (n) without regenerating/re-evaluating metrics each time.

Usage:
    python run_regression_sweep.py \
        --dataset ICLR \
        --target recommendation \
        --seed 42 \
        --output-dir results/main_runs/autometrics/gpt4omini/ICLR_recommendation_sweep \
        --n-values 1,3,5,10,15,20 \
        [--model-name openai/gpt-4o-mini] \
        [--api-base http://localhost:8000/v1]
"""

import os
import sys
import json
import argparse
from typing import Optional, Dict, Tuple, List

# Add repository root to path to import autometrics
sys.path.append('/nlp/scr2/nlp/personal-rm/autometrics')

import dspy

from autometrics.autometrics import Autometrics, HotellingPLS
from autometrics.aggregator.regression.Lasso import Lasso
from autometrics.aggregator.regression.PLS import PLS
from autometrics.dataset.Dataset import Dataset


def load_dataset(dataset_name: str) -> Dataset:
    """Load a dataset by name with persistent splits (minimal copy)."""
    if dataset_name == "ICLR":
        from autometrics.dataset.datasets.iclr.iclr import ICLR
        return ICLR()
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_unique_directories(dataset_name: str, target_name: str, seed: int) -> Tuple[str, str]:
    """Get unique cache and generated metrics directories for this experiment."""
    experiment_id = f"{dataset_name}_{target_name}_{seed}"
    cache_dir = f"./autometrics_cache_{experiment_id}"
    generated_metrics_dir = f"./generated_metrics_{experiment_id}"
    return cache_dir, generated_metrics_dir


def evaluate_regression_on_test(regression_metric, test_dataset: Dataset, target_measure: str) -> Tuple[Dict[str, float], Dict[str, Optional[float]]]:
    """Evaluate regression metric on test set and return correlation scores and p-values."""
    print("\U0001F4C8 Evaluating regression metric on test set (sweep)...")
    regression_metric.predict(test_dataset, update_dataset=True)

    try:
        df = test_dataset.get_dataframe()
        metric_name = regression_metric.get_name()
        pair_df = df[[target_measure, metric_name]].dropna()
        if len(pair_df) < 2:
            print("\u26A0\uFE0F Not enough valid pairs (<2) for correlation; returning zeros.")
            return {"pearson": 0.0, "spearman": 0.0, "kendall": 0.0}, {"pearson": None, "spearman": None, "kendall": None}

        from scipy.stats import pearsonr, spearmanr, kendalltau
        pr, pp = pearsonr(pair_df[target_measure], pair_df[metric_name])
        sr, sp = spearmanr(pair_df[target_measure], pair_df[metric_name])
        kr, kp = kendalltau(pair_df[target_measure], pair_df[metric_name])
        return {"pearson": float(pr), "spearman": float(sr), "kendall": float(kr)}, {"pearson": float(pp), "spearman": float(sp), "kendall": float(kp)}
    except Exception as e:
        print(f"\u26A0\uFE0F Warning: correlation computation failed: {e}")
        return {"pearson": 0.0, "spearman": 0.0, "kendall": 0.0}, {"pearson": None, "spearman": None, "kendall": None}


def format_model_name(model_name: str) -> str:
    """Match the main runner's local-proxy format for Qwen when using local api base."""
    if os.environ.get("OPENAI_API_BASE") and "localhost" in os.environ.get("OPENAI_API_BASE", ""):
        if model_name.startswith("Qwen/"):
            return f"litellm_proxy/{model_name}"
        elif "/" not in model_name and model_name.lower().startswith("qwen"):
            return "litellm_proxy/Qwen/Qwen3-32B"
    return model_name


def run_regression_sweep(
    dataset_name: str,
    target_name: str,
    seed: int,
    output_dir: str,
    n_values: List[int],
    generator_model_name: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    regression: str = "pls",
    lasso_alphas: Optional[List[float]] = None,
) -> Dict:
    # Prepare output directory and unique cache/metrics dirs
    os.makedirs(output_dir, exist_ok=True)
    cache_dir, generated_metrics_dir = get_unique_directories(dataset_name, target_name, seed)

    # Use unique cache to avoid interference
    os.environ["AUTOMETRICS_CACHE_DIR"] = cache_dir

    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base

    generator_model_name_base = (
        generator_model_name or os.environ.get("AUTOMETRICS_LM_GENERATOR") or "openai/gpt-4o-mini"
    )
    judge_model_name_base = (
        judge_model_name or os.environ.get("AUTOMETRICS_LM_JUDGE") or generator_model_name_base
    )

    generator_model_id = format_model_name(generator_model_name_base)
    judge_model_id = format_model_name(judge_model_name_base)

    api_key = os.environ.get("OPENAI_API_KEY", "None")
    if not api_key or api_key == "None":
        raise RuntimeError("OPENAI_API_KEY must be set for LLM-backed components.")

    print("\U0001F680 Starting regression sweep:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Target: {target_name}")
    print(f"   Seed: {seed}")
    print(f"   Generator LM: {generator_model_id}")
    print(f"   Judge LM: {judge_model_id}")
    print(f"   Output: {output_dir}")
    print(f"   n values: {n_values}")

    # LLMs
    if "Qwen" in generator_model_id:
        generator_llm = dspy.LM(generator_model_id, api_key=api_key, max_tokens=8192)
    else:
        generator_llm = dspy.LM(generator_model_id, api_key=api_key)

    if "Qwen" in judge_model_id:
        judge_llm = dspy.LM(judge_model_id, api_key=api_key, max_tokens=8192)
    else:
        judge_llm = dspy.LM(judge_model_id, api_key=api_key)

    # Load dataset and splits
    dataset_obj = load_dataset(dataset_name)
    train_dataset, val_dataset, test_dataset_base = dataset_obj.load_permanent_splits()
    print(f"   Train: {len(train_dataset.get_dataframe())} examples")
    print(f"   Val:   {len(val_dataset.get_dataframe())} examples")
    print(f"   Test:  {len(test_dataset_base.get_dataframe())} examples")

    # Build autometrics with fixed dirs
    autometrics = Autometrics(
        generated_metrics_dir=generated_metrics_dir,
        seed=seed,
    )

    # Steps 1-4 of the pipeline (generate -> retrieve -> evaluate) run once
    print("\n[Autometrics-Sweep] Step 1: Generating/Loading Metrics")
    generated_metrics = autometrics._generate_or_load_metrics(
        dataset=train_dataset,
        target_measure=target_name,
        generator_llm=generator_llm,
        judge_llm=judge_llm,
        regenerate_metrics=False,
        prometheus_api_base=None,
        model_save_dir=None,
    )
    print(f"[Autometrics-Sweep] Generated/Loaded {len(generated_metrics)} metrics")

    print("\n[Autometrics-Sweep] Step 2: Loading Metric Bank")
    metric_bank = autometrics._load_metric_bank(dataset=train_dataset)
    metric_bank = autometrics._merge_generated_with_bank(metric_bank, generated_metrics)

    print("[Autometrics-Sweep] Configuring retriever...")
    retriever_kwargs = autometrics.retriever_kwargs.copy()
    retriever_kwargs["metric_classes"] = metric_bank
    retriever_kwargs["model"] = generator_llm
    retriever_kwargs = autometrics._validate_and_adjust_retriever_config(
        retriever_kwargs, dataset=train_dataset, metric_bank=metric_bank, num_to_retrieve=30
    )
    retriever_instance = autometrics.retriever(**retriever_kwargs)

    print(f"\n[Autometrics-Sweep] Step 3: Retrieving Top 30 Metrics")
    retrieved_metrics = autometrics._retrieve_top_k_metrics(
        dataset=train_dataset,
        target_measure=target_name,
        k=30,
        retriever_instance=retriever_instance,
        metric_bank=metric_bank,
    )

    print(f"\n[Autometrics-Sweep] Step 4: Evaluating {len(retrieved_metrics)} Metrics on Dataset")
    successful_metric_instances = autometrics._evaluate_metrics_on_dataset(
        dataset=train_dataset,
        metric_classes=retrieved_metrics,
    )

    # Prepare fixed parts of regression kwargs
    # Choose regression strategy
    if regression.lower() in ("hotelling_pls", "hotelling", "hpls"):
        regression_strategy = HotellingPLS
    elif regression.lower() in ("lasso",):
        regression_strategy = Lasso
    elif regression.lower() in ("pls", "pls_regression"):
        regression_strategy = PLS
    else:
        raise ValueError(f"Unsupported regression strategy: {regression}")

    # Base kwargs depend on strategy
    base_regression_kwargs = autometrics.regression_kwargs.copy() if regression_strategy == HotellingPLS else {}
    if regression_strategy == HotellingPLS:
        base_regression_kwargs["random_state"] = seed
    if regression_strategy == Lasso and lasso_alphas and len(lasso_alphas) == 1:
        base_regression_kwargs["alpha"] = float(lasso_alphas[0])
    base_regression_kwargs["dataset"] = train_dataset

    # Run regression for each n
    sweep_results = {
        "dataset_name": dataset_name,
        "target_name": target_name,
        "seed": seed,
        "n_values": n_values,
        "regression": regression.lower(),
    }

    if regression_strategy == Lasso:
        alphas = lasso_alphas if lasso_alphas else [0.01]
        sweep_results["lasso_alphas"] = alphas
        sweep_results["per_alpha"] = {}

        for alpha in alphas:
            print(f"\n[Autometrics-Sweep] Lasso alpha={alpha}")
            alpha_key = f"{alpha}"
            sweep_results["per_alpha"][alpha_key] = {"per_n_results": {}}

            for n in n_values:
                print(f"\n[Autometrics-Sweep] Step 5: Regression Analysis for n={n}, alpha={alpha}")

                regression_kwargs = base_regression_kwargs.copy()
                regression_kwargs["alpha"] = float(alpha)

                regression_instance = regression_strategy(**regression_kwargs)

                regression_results = autometrics._regress_and_select_top_n(
                    dataset=train_dataset,
                    metric_instances=successful_metric_instances,
                    target_measure=target_name,
                    n=int(n),
                    regression_instance=regression_instance,
                )

                test_dataset = test_dataset_base.copy()
                try:
                    for metric in regression_results.get('top_metrics', []):
                        test_dataset.add_metric(metric, update_dataset=True)
                except Exception as _e:
                    print(f"\u26A0\uFE0F Warning: failed to precompute top metrics for test (n={n}, alpha={alpha}): {_e}")

                test_scores, test_p_values = evaluate_regression_on_test(
                    regression_results['regression_metric'], test_dataset, target_name
                )

                print("\n\u2705 Test correlations (n={}, alpha={})".format(n, alpha))
                for corr_type, score in test_scores.items():
                    print(f"   {corr_type.capitalize()}: {score:.4f}")

                sweep_results["per_alpha"][alpha_key]["per_n_results"][str(n)] = {
                    "test_scores": test_scores,
                    "test_p_values": test_p_values,
                    "top_metrics": [m.get_name() for m in regression_results.get('top_metrics', [])],
                    "importance_scores": [
                        (float(score), str(name)) for (score, name) in (regression_results.get('importance_scores') or [])
                    ],
                    "regression_metric_name": regression_results['regression_metric'].get_name() if regression_results.get('regression_metric') else None,
                }
    else:
        sweep_results["per_n_results"] = {}
        for n in n_values:
            print(f"\n[Autometrics-Sweep] Step 5: Regression Analysis for n={n}")

            regression_kwargs = base_regression_kwargs.copy()
            if regression_strategy == HotellingPLS:
                regression_kwargs["selection_mode"] = "top_n"
                regression_kwargs["top_n"] = int(n)

            regression_instance = regression_strategy(**regression_kwargs)

            regression_results = autometrics._regress_and_select_top_n(
                dataset=train_dataset,
                metric_instances=successful_metric_instances,
                target_measure=target_name,
                n=int(n),
                regression_instance=regression_instance,
            )

            test_dataset = test_dataset_base.copy()
            try:
                for metric in regression_results.get('top_metrics', []):
                    test_dataset.add_metric(metric, update_dataset=True)
            except Exception as _e:
                print(f"\u26A0\uFE0F Warning: failed to precompute top metrics for test (n={n}): {_e}")

            test_scores, test_p_values = evaluate_regression_on_test(
                regression_results['regression_metric'], test_dataset, target_name
            )

            print("\n\u2705 Test correlations (n={})".format(n))
            for corr_type, score in test_scores.items():
                print(f"   {corr_type.capitalize()}: {score:.4f}")

            sweep_results["per_n_results"][str(n)] = {
                "test_scores": test_scores,
                "test_p_values": test_p_values,
                "top_metrics": [m.get_name() for m in regression_results.get('top_metrics', [])],
                "importance_scores": [
                    (float(score), str(name)) for (score, name) in (regression_results.get('importance_scores') or [])
                ],
                "regression_metric_name": regression_results['regression_metric'].get_name() if regression_results.get('regression_metric') else None,
            }

    # Save combined sweep log
    log_file = os.path.join(output_dir, f"log_{seed}_sweep.json")
    with open(log_file, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n\u2705 Sweep complete. Saved: {log_file}")

    return sweep_results


def parse_args():
    parser = argparse.ArgumentParser(description="Autometrics regression sweep runner")
    parser.add_argument("--dataset", type=str, default="ICLR", help="Dataset name (default: ICLR)")
    parser.add_argument("--target", type=str, required=True, help="Target/measure name (e.g., recommendation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for sweep results")
    parser.add_argument("--n-values", type=str, default="1,3,5,10,15,20", help="Comma-separated list of n values to test")
    parser.add_argument("--model-name", dest="model_name", type=str, default=None, help="LLM model name (e.g., openai/gpt-4o-mini)")
    parser.add_argument("--api-base", dest="api_base", type=str, default=None, help="API base URL for OpenAI-compatible endpoints")
    parser.add_argument("--regression", type=str, default="pls", choices=["hotelling_pls", "lasso", "pls"], help="Regression strategy to use")
    parser.add_argument("--lasso-alpha", type=float, default=None, help="Single alpha for Lasso (deprecated by --lasso-alphas)")
    parser.add_argument("--lasso-alphas", type=str, default=None, help="Comma-separated list of alphas for Lasso sweep (e.g., 0.01,0.1,0.5,1.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("\u274C Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    n_values = [int(x.strip()) for x in args.n_values.split(',') if x.strip()]
    lasso_alphas = None
    if args.lasso_alphas:
        lasso_alphas = [float(x.strip()) for x in args.lasso_alphas.split(',') if x.strip()]
    elif args.lasso_alpha is not None:
        lasso_alphas = [float(args.lasso_alpha)]

    try:
        _ = run_regression_sweep(
            dataset_name=args.dataset,
            target_name=args.target,
            seed=args.seed,
            output_dir=args.output_dir,
            n_values=n_values,
            generator_model_name=args.model_name,
            judge_model_name=args.model_name,
            api_base=args.api_base,
            regression=args.regression,
            lasso_alphas=lasso_alphas,
        )
    except Exception as e:
        print(f"\n\U0001F4A5 Sweep failed: {e}")
        raise e


if __name__ == "__main__":
    main()


