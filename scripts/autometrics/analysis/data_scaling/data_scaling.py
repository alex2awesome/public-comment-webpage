#!/usr/bin/env python3
"""
Simple and elegant experiments file for running autometrics experiments.

This script:
1. Checks if experiment has already completed (score_[seed].txt exists with valid float)
2. If not, loads dataset with persistent splits and runs autometrics pipeline
3. Evaluates regression metric on test set to get final score
4. Saves results to output directory

Usage:
    python run_main_autometrics.py <dataset_name> <target_name> <seed> <output_dir>
"""

import os
import sys
import json
import dspy
import numpy as np
import argparse
from typing import Optional, Dict, Tuple

# Add autometrics to path
sys.path.append('/nlp/scr2/nlp/personal-rm/autometrics')

from autometrics.autometrics import Autometrics
from autometrics.dataset.Dataset import Dataset
from autometrics.autometrics import DEFAULT_GENERATOR_CONFIGS


def load_dataset(dataset_name: str) -> Dataset:
    """Load a dataset by name with persistent splits."""
    # Import the specific dataset class
    if dataset_name == "Primock57":
        from autometrics.dataset.datasets.primock57.primock57 import Primock57
        return Primock57()
    elif dataset_name == "HelpSteer":
        from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer
        return HelpSteer()
    elif dataset_name == "HelpSteer2":
        from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer2
        return HelpSteer2()
    elif dataset_name == "SummEval":
        from autometrics.dataset.datasets.summeval.summeval import SummEval
        return SummEval()
    elif dataset_name == "SimpDA":
        from autometrics.dataset.datasets.simplification.simplification import SimpDA
        return SimpDA()
    elif dataset_name == "SimpEval":
        from autometrics.dataset.datasets.simplification.simplification import SimpEval
        return SimpEval()
    elif dataset_name.startswith("CoGym"):
        from autometrics.dataset.datasets.cogym.cogym import (
            CoGymTabularOutcome, CoGymTabularProcess, 
            CoGymTravelOutcome, CoGymTravelProcess, 
            CoGymLessonOutcome, CoGymLessonProcess
        )
        if dataset_name == "CoGymTabularOutcome":
            return CoGymTabularOutcome()
        elif dataset_name == "CoGymTabularProcess":
            return CoGymTabularProcess()
        elif dataset_name == "CoGymTravelOutcome":
            return CoGymTravelOutcome()
        elif dataset_name == "CoGymTravelProcess":
            return CoGymTravelProcess()
        elif dataset_name == "CoGymLessonOutcome":
            return CoGymLessonOutcome()
        elif dataset_name == "CoGymLessonProcess":
            return CoGymLessonProcess()
    elif dataset_name.startswith("EvalGen"):
        # Use specific subclasses to preserve task descriptions and clear naming
        from autometrics.dataset.datasets.evalgen.evalgen import EvalGenProduct, EvalGenMedical
        if dataset_name == "EvalGenMedical":
            return EvalGenMedical()
        elif dataset_name == "EvalGenProduct":
            return EvalGenProduct()
    elif dataset_name == "RealHumanEval":
        from autometrics.dataset.datasets.realhumaneval.realhumaneval import RealHumanEval
        return RealHumanEval()
    elif dataset_name == "Design2Code":
        from autometrics.dataset.datasets.design2code.design2code import Design2Code
        return Design2Code()
    elif dataset_name == "AI_Researcher":
        from autometrics.dataset.datasets.airesearcher.ai_researcher import AI_Researcher
        return AI_Researcher()
    elif dataset_name == "ICLR":
        from autometrics.dataset.datasets.iclr.iclr import ICLR
        return ICLR()
    
    raise ValueError(f"Unknown dataset: {dataset_name}")


def check_experiment_completed(output_dir: str, seed: int) -> Optional[Dict[str, float]]:
    """Check if experiment has already completed and return scores if so."""
    # Check for all correlation types
    correlation_types = ['pearson', 'spearman', 'kendall']
    score_files = [os.path.join(output_dir, f"score_{corr_type}_{seed}.txt") for corr_type in correlation_types]
    log_file = os.path.join(output_dir, f"log_{seed}.json")
    
    # Check if all files exist
    if not (all(os.path.exists(f) for f in score_files) and os.path.exists(log_file)):
        return None
    
    # Try to read all scores
    try:
        scores = {}
        for corr_type, score_file in zip(correlation_types, score_files):
            with open(score_file, 'r') as f:
                score_str = f.read().strip()
                scores[corr_type] = float(score_str)
        
        print(f"✅ Experiment already completed with scores:")
        for corr_type, score in scores.items():
            print(f"   {corr_type.capitalize()}: {score:.4f}")
        return scores
    except (ValueError, IOError):
        print(f"⚠️  Score files exist but contain invalid data, re-running experiment")
        return None


def get_unique_directories(
    model_name: str,
    dataset_name: str,
    target_name: str,
    seed: int,
    train_size: Optional[int],
    generated_only: bool,
) -> tuple[str, str]:
    """Get unique cache and generated metrics directories for this data scaling experiment."""
    # Tokens describing scaling setup
    size_token = f"sz{train_size}" if train_size is not None else "szFull"
    mode_token = "genonly" if generated_only else "fullbank"

    # Create unique identifiers
    experiment_id = f"{model_name}_{dataset_name}_{target_name}_{seed}_{size_token}_{mode_token}"

    # Unique cache directory (renamed to scaling)
    cache_dir = f"./autometrics_cache_scaling/autometrics_cache_{experiment_id}"

    # Unique generated metrics directory (renamed to scaling)
    generated_metrics_dir = f"./generated_metrics_scaling/generated_metrics_{experiment_id}"

    return cache_dir, generated_metrics_dir


def evaluate_regression_on_test(regression_metric, test_dataset: Dataset, target_measure: str, successful_metric_instances) -> Tuple[Dict[str, float], Dict[str, Optional[float]]]:
    """Evaluate regression metric on test set and return correlation scores for all types.

    Also computes corresponding p-values for each correlation type, which the caller can log.
    Returns a tuple: (correlations: Dict[str, float], p_values: Dict[str, float]).
    """
    # Simply use predict() which will handle all dependencies automatically
    print(f"📈 Evaluating regression metric on test set...")
    regression_metric.predict(test_dataset, update_dataset=True)

    # Debug: validate data availability for correlation
    try:
        df = test_dataset.get_dataframe()
        metric_name = regression_metric.get_name()
        total_rows = len(df)
        missing_target = target_measure not in df.columns
        missing_metric = metric_name not in df.columns
        if missing_target or missing_metric:
            print(f"⚠️ Correlation debug: missing columns -> target_present={not missing_target}, metric_present={not missing_metric}")
            print(f"   Available columns sample: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
            # Fall back to zeros if we cannot compute correlations
            return {'pearson': 0.0, 'spearman': 0.0, 'kendall': 0.0}, {'pearson': None, 'spearman': None, 'kendall': None}

        # Count valid pairs for this specific (target, metric)
        pair_df = df[[target_measure, metric_name]].dropna()
        valid_pairs = len(pair_df)
        target_nans = int(df[target_measure].isna().sum())
        metric_nans = int(df[metric_name].isna().sum())
        print(f"🔎 Correlation debug: rows={total_rows}, valid_pairs={valid_pairs} for metric='{metric_name}' vs target='{target_measure}' (NaNs: target={target_nans}, metric={metric_nans})")
        if valid_pairs < 2:
            print("⚠️ Not enough valid pairs (<2). Skipping scipy correlation and returning zeros.")
            return {'pearson': 0.0, 'spearman': 0.0, 'kendall': 0.0}, {'pearson': None, 'spearman': None, 'kendall': None}
    except Exception as dbg_e:
        print(f"⚠️ Correlation debug: pre-check failed: {dbg_e}. Proceeding with guarded computation.")
    
    # Compute correlations directly between the target and the regression metric only
    from scipy.stats import pearsonr, spearmanr, kendalltau
    try:
        df = test_dataset.get_dataframe()
        metric_name = regression_metric.get_name()
        pair_df = df[[target_measure, metric_name]].dropna()
        if len(pair_df) < 2:
            print("⚠️ Not enough valid pairs (<2) after dropna for target vs regression. Returning zeros.")
            return {'pearson': 0.0, 'spearman': 0.0, 'kendall': 0.0}, {'pearson': None, 'spearman': None, 'kendall': None}
        # Quick sanity: report variance
        try:
            tgt_std = float(pair_df[target_measure].std())
            pred_std = float(pair_df[metric_name].std())
            print(f"🔎 Correlation debug: std(target)={tgt_std:.6f}, std(pred)={pred_std:.6f}")
        except Exception:
            pass
        pr, pp = pearsonr(pair_df[target_measure], pair_df[metric_name])
        sr, sp = spearmanr(pair_df[target_measure], pair_df[metric_name])
        kr, kp = kendalltau(pair_df[target_measure], pair_df[metric_name])
        return {'pearson': float(pr), 'spearman': float(sr), 'kendall': float(kr)}, {'pearson': float(pp), 'spearman': float(sp), 'kendall': float(kp)}
    except Exception as e:
        print(f"⚠️ Warning: Error computing direct correlations: {e}")
        return {'pearson': 0.0, 'spearman': 0.0, 'kendall': 0.0}, {'pearson': None, 'spearman': None, 'kendall': None}


def run_autometrics_experiment(
    dataset_name: str,
    target_name: str,
    seed: int,
    output_dir: str,
    generator_model_name: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    skip_mipro: bool = False,
    train_size: Optional[int] = None,
    generated_only: bool = False,
) -> Dict[str, float]:
    """Run a single autometrics experiment."""
    
    # Check if experiment already completed
    # Build scaling-aware output directory structure
    size_token = f"sz{train_size}" if train_size is not None else "szFull"
    mode_token = "genonly" if generated_only else "fullbank"
    derived_output_dir = os.path.join(output_dir, f"{dataset_name}_{target_name}_seed{seed}_{size_token}_{mode_token}")

    existing_scores = check_experiment_completed(derived_output_dir, seed)
    if existing_scores is not None:
        return existing_scores
    
    print(f"🚀 Starting data scaling experiment:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Target: {target_name}")
    print(f"   Seed: {seed}")
    print(f"   Output: {derived_output_dir}")
    if train_size is not None:
        print(f"   Train size (scaling): exactly {train_size} examples")
    print(f"   Mode: {'Generated-only' if generated_only else 'Generated + Metric Bank'}")
    
    # Create output directory
    os.makedirs(derived_output_dir, exist_ok=True)
    
    try:
        # Load dataset with persistent splits
        print(f"📊 Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        train_dataset, val_dataset, test_dataset = dataset.load_permanent_splits()

        # If requested train_size exceeds available size, gracefully skip and mark as completed with NaNs
        if train_size is not None:
            available_train = len(train_dataset.get_dataframe())
            if train_size > available_train:
                print(
                    f"⚠️ Requested train-size={train_size} exceeds available training examples ({available_train}). Skipping this run."
                )
                os.makedirs(derived_output_dir, exist_ok=True)
                # Write NaN scores so the orchestrator can treat as completed
                for corr_type in ['pearson', 'spearman', 'kendall']:
                    score_file = os.path.join(derived_output_dir, f"score_{corr_type}_{seed}.txt")
                    with open(score_file, 'w') as f:
                        f.write("nan")
                # Write a skip log for transparency
                skip_log = {
                    "dataset_name": dataset_name,
                    "target_name": target_name,
                    "seed": seed,
                    "train_size_requested": train_size,
                    "train_size_available": available_train,
                    "generated_only": generated_only,
                    "skipped": True,
                    "reason": "Requested train size exceeds available training examples"
                }
                with open(os.path.join(derived_output_dir, f"log_{seed}.json"), 'w') as f:
                    json.dump(skip_log, f, indent=2)
                return {"pearson": float('nan'), "spearman": float('nan'), "kendall": float('nan')}

        # For scaling experiments: cut training set to exactly N examples if specified
        if train_size is not None:
            train_dataset = train_dataset.get_subset(train_size, seed=seed)
        
        print(f"   Train: {len(train_dataset.get_dataframe())} examples")
        print(f"   Val: {len(val_dataset.get_dataframe())} examples")
        print(f"   Test: {len(test_dataset.get_dataframe())} examples")
        
        # Configure LLMs (CLI args take precedence; fall back to env vars; then defaults)
        print(f"🤖 Configuring LLMs...")
        if api_base:
            os.environ["OPENAI_API_BASE"] = api_base
        
        # Determine base model names
        generator_model_name_base = (
            generator_model_name
            or os.environ.get("AUTOMETRICS_LM_GENERATOR")
            or "openai/gpt-4o-mini"
        )
        judge_model_name_base = (
            judge_model_name
            or os.environ.get("AUTOMETRICS_LM_JUDGE")
            or generator_model_name_base
        )
        
        # Convert model names to proper litellm format when using an OpenAI-compatible endpoint
        def format_model_name(model_name: str) -> str:
            api_base_env = os.environ.get("OPENAI_API_BASE", "")
            if api_base_env:
                # If pointing to any OpenAI-compatible endpoint (local or remote),
                # ensure non-OpenAI providers like Qwen are routed via litellm_proxy
                if not model_name.startswith("litellm_proxy/"):
                    if model_name.startswith("Qwen/"):
                        return f"litellm_proxy/{model_name}"
                    elif "/" not in model_name and model_name.lower().startswith("qwen"):
                        return "litellm_proxy/Qwen/Qwen3-32B"
            return model_name
        
        generator_model_id = format_model_name(generator_model_name_base)
        judge_model_id = format_model_name(judge_model_name_base)

        # Get unique directories for this experiment (scaling-aware)
        cache_dir, generated_metrics_dir = get_unique_directories(
            generator_model_id, dataset_name, target_name, seed, train_size, generated_only
        )
        
        # Respect pre-set AUTOMETRICS_CACHE_DIR if provided (e.g., to share cache with ablations).
        # Otherwise, set to the unique main-run path.
        if not os.environ.get("AUTOMETRICS_CACHE_DIR"):
            os.environ["AUTOMETRICS_CACHE_DIR"] = cache_dir
        else:
            print(f"🔁 Using existing AUTOMETRICS_CACHE_DIR from environment: {os.environ['AUTOMETRICS_CACHE_DIR']}")
        
        print(f"   Generator LM: {generator_model_id}")
        print(f"   Judge LM: {judge_model_id}")
        
        # Create LLM instances with proper API key handling
        api_key = os.environ.get("OPENAI_API_KEY", "None")

        generator_llm = None
        judge_llm = None

        if "Qwen" in generator_model_id:
            generator_llm = dspy.LM(generator_model_id, api_key=api_key, max_tokens=8192)
        else:
            generator_llm = dspy.LM(generator_model_id, api_key=api_key)

        if "Qwen" in judge_model_id:
            judge_llm = dspy.LM(judge_model_id, api_key=api_key, max_tokens=8192)
        else:
            judge_llm = dspy.LM(judge_model_id, api_key=api_key)

        generator_configs = DEFAULT_GENERATOR_CONFIGS

        if skip_mipro:
            generator_configs = {
                "llm_judge": {"metrics_per_trial": 10, "description": "Basic LLM Judge"},
                "rubric_dspy": {"metrics_per_trial": 5, "description": "Rubric Generator (DSPy)"},
                "llm_judge_examples": {"metrics_per_trial": 1, "description": "LLM Judge (Example-Based)"},
            }
        
        # Create autometrics with unique directories
        print(f"🔧 Creating autometrics pipeline with full_bank_data_cutoff=None...")
        autometrics_kwargs = dict(
            generated_metrics_dir=generated_metrics_dir,
            metric_generation_configs=generator_configs,
            seed=seed,
            full_bank_data_cutoff=None,
        )
        if generated_only:
            autometrics_kwargs["metric_bank"] = []
        autometrics = Autometrics(**autometrics_kwargs)
        
        # Run autometrics pipeline on training data
        print(f"⚡ Running autometrics pipeline...")
        results = autometrics.run(
            dataset=train_dataset,
            target_measure=target_name,
            generator_llm=generator_llm,
            judge_llm=judge_llm,
            # Provide test set for report card sections and set HTML output path
            eval_dataset=test_dataset,
            report_output_path=os.path.join(derived_output_dir, f"report_{dataset_name}_{target_name}_{seed}_{size_token}_{mode_token}.html"),
            verbose=True,
        )
        
        # Get regression metric
        regression_metric = results['regression_metric']
        if regression_metric is None:
            raise ValueError("No regression metric generated")
        
        # Export the static regression Python file to output directory
        try:
            export_path = os.path.join(derived_output_dir, f"StaticRegression_{dataset_name}_{target_name}_seed{seed}_{size_token}_{mode_token}.py")
            regression_metric.export_python(export_path, inline_generated_metrics=True, name_salt=None)
            print(f"💾 Saved static regression to {export_path} ({os.path.getsize(export_path)} bytes)")
        except Exception as _exp_e:
            print(f"⚠️ Failed to export static regression: {_exp_e}")
        
        # Evaluate on test set
        print(f"📈 Evaluating regression metric on test set...")
        # Ensure constituent top metrics are computed on the test split before regression prediction
        try:
            for metric in results['top_metrics']:
                test_dataset.add_metric(metric, update_dataset=True)
        except Exception as _e:
            print(f"⚠️ Warning: failed to precompute top metrics on test: {_e}")
        test_scores, test_p_values = evaluate_regression_on_test(regression_metric, test_dataset, target_name, results['top_metrics'])
        
        print(f"✅ Test correlations:")
        for corr_type, score in test_scores.items():
            print(f"   {corr_type.capitalize()}: {score:.4f}")
        
        # Save scatter plot of predicted vs human scores (no recomputation)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            df = test_dataset.get_dataframe()
            metric_name = regression_metric.get_name()
            pair_df = df[[target_name, metric_name]].dropna()
            if len(pair_df) >= 2:
                plt.figure(figsize=(6, 6))
                plt.scatter(pair_df[metric_name], pair_df[target_name], s=10, alpha=0.6)
                plt.xlabel(metric_name)
                plt.ylabel(target_name)
                plt.title(f"Predicted vs Human ({dataset_name}, seed={seed})")
                vmin = float(min(pair_df[metric_name].min(), pair_df[target_name].min()))
                vmax = float(max(pair_df[metric_name].max(), pair_df[target_name].max()))
                plt.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1)
                plt.tight_layout()
                plot_path = os.path.join(derived_output_dir, f"scatter_pred_vs_target_{seed}_{size_token}_{mode_token}.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"📊 Saved scatter plot to {plot_path}")
            else:
                print("⚠️ Not enough data to plot scatter.")
        except Exception as _pe:
            print(f"⚠️ Failed to save scatter plot: {_pe}")
        
        # Save results
        print(f"💾 Saving results...")
        
        # Save individual score files for each correlation type
        for corr_type, score in test_scores.items():
            score_file = os.path.join(derived_output_dir, f"score_{corr_type}_{seed}.txt")
            with open(score_file, 'w') as f:
                f.write(f"{score}")
        
        # Save detailed log as JSON
        log_file = os.path.join(derived_output_dir, f"log_{seed}.json")
        log_data = {
            "dataset_name": dataset_name,
            "target_name": target_name,
            "seed": seed,
            "train_size": train_size,
            "generated_only": generated_only,
            "split_sizes": {
                "train": len(train_dataset.get_dataframe()),
                "val": len(val_dataset.get_dataframe()),
                "test": len(test_dataset.get_dataframe()),
            },
            "test_scores": test_scores,
            "test_p_values": test_p_values,
            "report_card": results['report_card'],
            "top_metrics": [m.get_name() for m in results['top_metrics']],
            "importance_scores": [(float(score), name) for score, name in results['importance_scores'][:10]],
            "generated_metrics_count": len(results['all_generated_metrics']),
            "retrieved_metrics_count": len(results['retrieved_metrics']),
            "pipeline_config": results['pipeline_config']
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✅ Experiment completed successfully!")
        return test_scores
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        raise e


def main():
    """Main function to run autometrics experiment."""
    parser = argparse.ArgumentParser(description="Run Autometrics experiment")
    parser.add_argument("dataset_name", type=str, help="Dataset name")
    parser.add_argument("target_name", type=str, help="Target/measure name")
    parser.add_argument("seed", type=int, help="Random seed")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--model-name", dest="model_name", type=str, default=None, help="LLM model name (e.g., openai/gpt-5-mini)")
    parser.add_argument("--api-base", dest="api_base", type=str, default=None, help="API base URL for OpenAI-compatible endpoints")
    parser.add_argument("--skip-mipro", action="store_true", help="Skip Mipro (typically used because MIPRO changes the lm temperature and new openai models do not support it)")
    parser.add_argument("--train-size", dest="train_size", type=int, default=None, help="If provided, subset the training set to exactly N examples")
    parser.add_argument("--generated-only", dest="generated_only", action="store_true", help="Use generated metrics only (empty metric bank)")
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    try:
        scores = run_autometrics_experiment(
            dataset_name=args.dataset_name,
            target_name=args.target_name,
            seed=args.seed,
            output_dir=args.output_dir,
            generator_model_name=args.model_name,
            judge_model_name=args.model_name,
            api_base=args.api_base,
            skip_mipro=args.skip_mipro,
            train_size=args.train_size,
            generated_only=args.generated_only,
        )
        print(f"\n🎉 Final test correlations:")
        for corr_type, score in scores.items():
            print(f"   {corr_type.capitalize()}: {score:.4f}")
    except Exception as e:
        print(f"\n💥 Experiment failed: {e}")
        raise e


if __name__ == "__main__":
    main()
