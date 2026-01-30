#!/usr/bin/env python3
"""
Best Static Metric Stability Analysis

This script tests the correlation stability of best-performing metrics across multiple random seeds.
It computes metrics once per seed and extracts all correlation functions (kendall, pearson, spearman) 
from the same run to avoid redundant computation.

Key features:
- Uses persistent test sets ONLY for reproducible results
- Computes each metric once per seed, extracts all correlations
- Resume capability for partial runs  
- Dataset-specific caching to reuse existing model caches
- Statistical analysis with 95% confidence intervals
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from scipy import stats
from collections import defaultdict
import traceback

# Add autometrics to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autometrics.dataset.Dataset import Dataset
from autometrics.experiments.correlation.correlation import CorrelationExperiment, correlation_func_from_name
from autometrics.metrics.MetricBank import build_metrics, all_metric_classes


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def load_dataset(dataset_name: str) -> Dataset:
    """Load a dataset by name."""
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
        from autometrics.dataset.datasets.evalgen.evalgen import EvalGen
        if dataset_name == "EvalGenMedical":
            return EvalGen('./autometrics/dataset/datasets/evalgen/medical.csv')
        elif dataset_name == "EvalGenProduct":
            return EvalGen('./autometrics/dataset/datasets/evalgen/product.csv')
    elif dataset_name == "RealHumanEval":
        from autometrics.dataset.datasets.realhumaneval.realhumaneval import RealHumanEval
        return RealHumanEval()
    elif dataset_name == "Design2Code":
        from autometrics.dataset.datasets.design2code.design2code import Design2Code
        return Design2Code()
    
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_metric_class(metric_class_name: str):
    """Get metric class by name."""
    for cls in all_metric_classes:
        if cls.__name__ == metric_class_name:
            return cls
    return None


def create_dataset_cache_dir(base_cache_dir: str, dataset_name: str) -> str:
    """Create dataset-specific cache directory matching correlation experiment paths."""
    # Use the same cache directory structure as all_correlation.sh
    cache_mapping = {
        "Primock57": "primock57",
        "HelpSteer": "helpsteer", 
        "HelpSteer2": "helpsteer2",
        "SummEval": "summeval",
        "SimpDA": "simpda",
        "SimpEval": "simeval",
        "CoGymTabularOutcome": "cogym_tabular_outcome",
        "CoGymTabularProcess": "cogym_tabular_process", 
        "CoGymTravelOutcome": "cogym_travel_outcome",
        "CoGymTravelProcess": "cogym_travel_process",
        "CoGymLessonOutcome": "cogym_lesson_outcome",
        "CoGymLessonProcess": "cogym_lesson_process",
        "EvalGenMedical": "evalgen_medical",
        "EvalGenProduct": "evalgen_product",
        "RealHumanEval": "real_human_eval",
        "Design2Code": "design2code"
    }
    
    cache_subdir = cache_mapping.get(dataset_name, dataset_name.lower())
    cache_dir = f"./.cache/{cache_subdir}"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def run_single_metric_seed(
    dataset_name: str,
    measure: str, 
    metric_name: str,
    metric_class_name: str,
    seed: int,
    correlation_funcs: Dict[str, Any],
    cache_dir: str,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Run a single metric on a single seed and return correlations for all correlation functions.
    
    Returns:
        Dict mapping correlation function name to correlation value
    """
    try:
        # Get the specific metric class
        metric_class = get_metric_class(metric_class_name)
        if not metric_class:
            raise ValueError(f"Unknown metric class: {metric_class_name}")
        
        # Build the specific metric we need
        metrics = build_metrics(
            classes=[metric_class],
            cache_dir=cache_dir,
            seed=seed
        )
        
        if not metrics:
            raise ValueError(f"Failed to build metric {metric_class_name}")
        
        metric_instance = metrics[0]
        
        # Load dataset and get test split
        dataset = load_dataset(dataset_name)
        _, _, test_dataset = dataset.load_permanent_splits()
        
        # Run correlation experiment with ALL correlation functions at once
        experiment = CorrelationExperiment(
            name=f"Stability Test - {dataset_name} - {measure} - {metric_name}",
            description=f"Testing correlation stability for {metric_name} on {dataset_name}",
            metrics=[metric_instance],
            output_dir=f"/tmp/stability_test_{seed}",
            dataset=test_dataset,
            correlation_funcs=correlation_funcs,
            seed=seed,
            should_split=False
        )
        
        # Run experiment and extract correlations for ALL functions
        all_correlations = experiment.run(print_results=False)
        
        # Extract correlations for all correlation functions
        correlations_by_func = {}
        
        for func_name, correlations_for_func in all_correlations.items():
            if measure not in correlations_for_func:
                raise ValueError(f"Measure {measure} not found in correlation results for {func_name}")
            
            df_corr = correlations_for_func[measure]
            
            # Find the correlation for our specific metric
            # The metric_name from CSV might be a submetric like "ROUGE-2-f1"
            metric_row = df_corr[df_corr['Metric'] == metric_name]
            
            if metric_row.empty:
                # List available metrics for debugging
                available_metrics = list(df_corr['Metric'].values)
                raise ValueError(f"Metric {metric_name} not found in correlation results for {func_name}. Available: {available_metrics}")
            
            correlations_by_func[func_name] = metric_row.iloc[0]['Correlation']
        
        return correlations_by_func
        
    except Exception as e:
        logger.error(f"Error running {metric_name} with seed {seed}: {str(e)}")
        raise


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, and 95% CI for a list of correlation values."""
    if not values or len(values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'ci_range': np.nan
        }
    
    values = np.array(values)
    valid_values = values[~np.isnan(values)]
    
    if len(valid_values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'ci_range': np.nan
        }
    
    mean_val = np.mean(valid_values)
    std_val = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
    
    # 95% confidence interval using t-distribution
    if len(valid_values) > 1:
        t_critical = stats.t.ppf(0.975, len(valid_values) - 1)
        margin_error = t_critical * (std_val / np.sqrt(len(valid_values)))
        ci_lower = mean_val - margin_error
        ci_upper = mean_val + margin_error
        ci_range = margin_error
    else:
        ci_lower = mean_val
        ci_upper = mean_val
        ci_range = 0.0
    
    return {
        'mean': mean_val,
        'std': std_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_range': ci_range
    }


def format_mean_ci(mean: float, ci_range: float) -> str:
    """Format mean ± CI for easy copying to papers."""
    if np.isnan(mean) or np.isnan(ci_range):
        return "N/A"
    return f"{mean:.4f} ± {ci_range:.4f}"


def save_results(results: Dict[str, Any], output_file: str, logger: logging.Logger):
    """Save results to CSV file."""
    rows = []
    
    for metric_key, metric_data in results.items():
        dataset, measure, metric_name, metric_class = metric_key
        
        # Get seed results
        seed_results = metric_data.get('seed_results', {})
        stats_results = metric_data.get('statistics', {})
        
        row = {
            'dataset': dataset,
            'measure': measure,
            'metric': metric_name,
            'metric_class': metric_class
        }
        
        # Add seed columns in order
        for seed in sorted(seed_results.keys()):
            row[f'seed_{seed}'] = seed_results[seed]
        
        # Add statistics
        row['mean'] = stats_results.get('mean', np.nan)
        row['std'] = stats_results.get('std', np.nan)
        row['ci_lower'] = stats_results.get('ci_lower', np.nan)
        row['ci_upper'] = stats_results.get('ci_upper', np.nan)
        row['ci_range'] = stats_results.get('ci_range', np.nan)
        row['mean_±_ci'] = format_mean_ci(stats_results.get('mean', np.nan), stats_results.get('ci_range', np.nan))
        
        # Add error info
        row['successful_runs'] = len([v for v in seed_results.values() if not np.isnan(v)])
        row['errors'] = metric_data.get('errors', '')
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")


def load_existing_results(output_file: str) -> Dict[str, Any]:
    """Load existing results if the file exists."""
    if not os.path.exists(output_file):
        return {}
    
    df = pd.read_csv(output_file)
    results = {}
    
    for _, row in df.iterrows():
        metric_key = (row['dataset'], row['measure'], row['metric'], row['metric_class'])
        
        # Extract seed results
        seed_results = {}
        for col in df.columns:
            if col.startswith('seed_'):
                seed = int(col.split('_')[1])
                value = row[col]
                if not pd.isna(value):
                    seed_results[seed] = value
        
        # Extract statistics
        statistics = {
            'mean': row.get('mean', np.nan),
            'std': row.get('std', np.nan),
            'ci_lower': row.get('ci_lower', np.nan),
            'ci_upper': row.get('ci_upper', np.nan),
            'ci_range': row.get('ci_range', np.nan)
        }
        
        results[metric_key] = {
            'seed_results': seed_results,
            'statistics': statistics,
            'errors': row.get('errors', '')
        }
    
    return results


def filter_metrics(metrics: List[Dict[str, Any]], dataset_filter: Optional[List[str]], measure_filter: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Filter metrics by dataset and/or measure."""
    filtered = metrics
    
    if dataset_filter:
        filtered = [m for m in filtered if m['dataset'] in dataset_filter]
    
    if measure_filter:
        filtered = [m for m in filtered if m['measure'] in measure_filter]
    
    return filtered


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test correlation stability of best metrics across seeds")
    parser.add_argument("--best-metrics-file", default="results/best_metrics.csv", help="Path to best metrics CSV")
    parser.add_argument("--output-file", default="results/main_runs/baselines/best_static_metrics.csv", help="Output file path")
    parser.add_argument("--cache-dir", default="./.cache/best_static_metric", help="Base cache directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Seeds to test")
    parser.add_argument("--correlation", default="all", help="Correlation functions: pearson, spearman, kendall, or 'all'")
    parser.add_argument("--dataset", nargs="*", help="Filter to specific datasets")
    parser.add_argument("--measure", nargs="*", help="Filter to specific measures")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    # Get correlation functions
    if args.correlation.lower() == "all":
        correlation_specs = ["kendall", "pearson", "spearman"]
    else:
        correlation_specs = [c.strip() for c in args.correlation.split(",")]
    
    correlation_funcs = {}
    for spec in correlation_specs:
        correlation_funcs[spec] = correlation_func_from_name(spec)
    
    logger.info(f"Using correlation functions: {list(correlation_funcs.keys())}")
    
    # Load and filter metrics
    df_metrics = pd.read_csv(args.best_metrics_file)
    metrics_to_test = []
    for _, row in df_metrics.iterrows():
        metrics_to_test.append({
            'dataset': row['dataset'],
            'measure': row['measure'], 
            'metric': row['metric'],
            'metric_class': row['metric_class']
        })
    
    metrics_to_test = filter_metrics(metrics_to_test, args.dataset, args.measure)
    logger.info(f"Testing {len(metrics_to_test)} metrics")
    
    # Load existing results for all correlation functions
    all_results = {}
    for corr_name in correlation_funcs:
        output_base = args.output_file.replace('.csv', '')
        output_file = f"{output_base}_{corr_name}.csv"
        all_results[corr_name] = {
            'output_file': output_file,
            'results': load_existing_results(output_file)
        }
        logger.info(f"Loaded existing results for {corr_name}: {len(all_results[corr_name]['results'])} metrics")
    
    # Process each metric ONCE with ALL correlation functions
    for i, metric_spec in enumerate(metrics_to_test, 1):
        dataset_name = metric_spec['dataset']
        measure = metric_spec['measure']
        metric_name = metric_spec['metric']
        metric_class = metric_spec['metric_class']
        
        metric_key = (dataset_name, measure, metric_name, metric_class)
        
        logger.info(f"\n[{i}/{len(metrics_to_test)}] Processing: {dataset_name} - {measure} - {metric_name}")
        
        # Create dataset-specific cache directory
        cache_dir = create_dataset_cache_dir(args.cache_dir, dataset_name)
        
        # Find what seeds we need to run across ALL correlation functions
        all_needed_seeds = set(args.seeds)
        for corr_name in correlation_funcs:
            existing_data = all_results[corr_name]['results'].get(metric_key, {'seed_results': {}})
            existing_seeds = set(existing_data['seed_results'].keys())
            all_needed_seeds = all_needed_seeds - existing_seeds
        
        needed_seeds = list(all_needed_seeds)
        
        if not needed_seeds:
            logger.info(f"  ✅ Already complete for all correlation functions")
            continue
        
        logger.info(f"  🔄 Running missing seeds: {needed_seeds}")
        
        # Run missing seeds ONCE with ALL correlation functions
        for seed in needed_seeds:
            try:
                logger.info(f"  Running seed {seed}...")
                
                # Run with ALL correlation functions at once - this is the key fix!
                all_correlations = run_single_metric_seed(
                    dataset_name, measure, metric_name, metric_class,
                    seed, correlation_funcs, cache_dir, logger  # Pass ALL correlation functions
                )
                
                # Update results for each correlation function
                for corr_name, correlation_value in all_correlations.items():
                    results_for_corr = all_results[corr_name]['results']
                    existing_data = results_for_corr.get(metric_key, {'seed_results': {}, 'errors': ''})
                    
                    # Add this seed's result
                    existing_data['seed_results'][seed] = correlation_value
                    
                    # Compute updated statistics (use absolute values for correlations)
                    correlation_values = list(existing_data['seed_results'].values())
                    statistics = compute_statistics([abs(v) for v in correlation_values if not pd.isna(v)])
                    
                    # Update results
                    results_for_corr[metric_key] = {
                        'seed_results': existing_data['seed_results'],
                        'statistics': statistics,
                        'errors': existing_data.get('errors', '')
                    }
                    
                    logger.info(f"    ✅ Seed {seed} {corr_name}: {correlation_value:.4f}")
                
            except Exception as e:
                error_msg = f"Seed {seed}: {str(e)}"
                logger.error(f"    ❌ {error_msg}")
                
                # Add error to all correlation functions
                for corr_name in correlation_funcs:
                    results_for_corr = all_results[corr_name]['results']
                    existing_data = results_for_corr.get(metric_key, {'seed_results': {}, 'errors': ''})
                    existing_errors = existing_data.get('errors', '')
                    new_errors = f"{existing_errors}; {error_msg}" if existing_errors else error_msg
                    
                    # Update with error (use absolute values for correlations)
                    correlation_values = list(existing_data['seed_results'].values())
                    statistics = compute_statistics([abs(v) for v in correlation_values if not pd.isna(v)]) if correlation_values else {}
                    
                    results_for_corr[metric_key] = {
                        'seed_results': existing_data['seed_results'],
                        'statistics': statistics,
                        'errors': new_errors
                    }
        
        # Save results after each metric for ALL correlation functions
        for corr_name in correlation_funcs:
            save_results(all_results[corr_name]['results'], all_results[corr_name]['output_file'], logger)
        
        # Report completion status
        completed_counts = {}
        for corr_name in correlation_funcs:
            results_for_corr = all_results[corr_name]['results']
            metric_data = results_for_corr.get(metric_key, {'seed_results': {}})
            completed_counts[corr_name] = len(metric_data['seed_results'])
        
        logger.info(f"  Completed: {completed_counts}")
    
    # Final summary
    for corr_name in correlation_funcs:
        logger.info(f"\n=== {corr_name.upper()} correlation completed ===")
        logger.info(f"Results saved to {all_results[corr_name]['output_file']}")
    
    logger.info("\nAll correlation analyses completed successfully!")


if __name__ == "__main__":
    main()