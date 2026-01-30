#!/usr/bin/env python3
"""
Experiment Utils for Metric Correlation Analysis
==============================================

This module provides shared functionality for running metric correlation experiments
across multiple seeds. It's designed to be reused for different types of experiments:

1. Best Static Metrics
2. LLM-as-a-Judge experiments
3. Custom metric evaluations

Key Features:
- Resume capability with existing results
- Multiple correlation support (pearson, spearman, kendall)
- Dataset-specific cache management
- Statistical analysis with confidence intervals
- Flexible metric instantiation and filtering

Usage:
------
from analysis.main_experiments.experiment_utils import (
    ExperimentRunner, 
    load_existing_results,
    compute_statistics,
    create_dataset_cache_dir
)

runner = ExperimentRunner(
    seeds=[42, 43, 44, 45, 46],
    correlation_funcs={"kendall": kendalltau},
    cache_base_dir="./.cache/my_experiment"
)

results = runner.run_experiment(metrics_to_test, output_file)
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from scipy.stats import t
from collections import defaultdict

# Import correlation functions
try:
    from scipy.stats import pearsonr, spearmanr, kendalltau
except ImportError:
    # Fallback definitions
    def _naive_corr(x, y):
        return np.corrcoef(x, y)[0, 1], np.nan
    pearsonr = _naive_corr
    spearmanr = _naive_corr
    kendalltau = _naive_corr

def correlation_func_from_name(name: str) -> Callable:
    """Convert correlation name to function."""
    name = name.lower()
    if name.startswith("pearson"):
        return pearsonr
    if name.startswith("spearman"):
        return spearmanr
    if name.startswith("kendall") or name.startswith("tau"):
        return kendalltau
    raise ValueError(f"Unknown correlation function '{name}'. Supported: pearson, spearman, kendall.")

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("experiment")

def create_dataset_cache_dir(base_cache_dir: str, dataset_name: str) -> str:
    """
    Create dataset-specific cache directory following the pattern from all_correlation.sh
    This reuses the existing caches from the correlation experiments.
    """
    # Use the exact same cache directory names as in all_correlation.sh
    dataset_to_cache = {
        'SimpDA': './.cache/simpda',
        'CoGymLessonOutcome': './.cache/cogym_lesson_outcome', 
        'CoGymLessonProcess': './.cache/cogym_lesson_process',
        'CoGymTabularOutcome': './.cache/cogym_tabular_outcome',
        'CoGymTabularProcess': './.cache/cogym_tabular_process',
        'CoGymTravelOutcome': './.cache/cogym_travel_outcome',
        'CoGymTravelProcess': './.cache/cogym_travel_process',
        'Design2Code': './.cache/design2code',
        'EvalGenMedical': './.cache/evalgen_medical',
        'EvalGenProduct': './.cache/evalgen_product',
        'HelpSteer': './.cache/helpsteer',
        'HelpSteer2': './.cache/helpsteer2',
        'Primock57': './.cache/primock57',
        'RealHumanEval': './.cache/real_human_eval',
        'SimpEval': './.cache/simeval',
        'SummEval': './.cache/summeval',
    }
    
    return dataset_to_cache.get(dataset_name, f"{base_cache_dir}/{dataset_name.lower()}")

def load_existing_results(output_file: str) -> pd.DataFrame:
    """Load existing results from output file if it exists."""
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            return df
        except Exception as e:
            print(f"Warning: Could not read existing results file {output_file}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def get_seed_columns(df: pd.DataFrame) -> List[str]:
    """Extract seed column names from DataFrame."""
    return [col for col in df.columns if col.startswith('seed_')]

def get_completed_seeds(row: pd.Series) -> List[int]:
    """Get list of completed seeds for a metric row."""
    completed = []
    seed_cols = get_seed_columns(pd.DataFrame([row]))
    for col in seed_cols:
        if pd.notna(row.get(col)):
            try:
                seed_num = int(col.replace('seed_', ''))
                completed.append(seed_num)
            except ValueError:
                continue
    return completed

def determine_seeds_to_run(
    existing_results_df: pd.DataFrame,
    target_seeds: List[int],
    dataset: str,
    measure: str,
    metric: str,
    metric_class: str
) -> Tuple[List[int], Optional[pd.Series]]:
    """
    Determine which seeds need to be run for a specific metric.
    
    Returns:
        Tuple of (seeds_to_run, existing_row_if_found)
    """
    if existing_results_df.empty:
        return target_seeds, None
    
    # Find existing row for this metric
    mask = (
        (existing_results_df['dataset'] == dataset) &
        (existing_results_df['measure'] == measure) &
        (existing_results_df['metric'] == metric) &
        (existing_results_df['metric_class'] == metric_class)
    )
    
    existing_rows = existing_results_df[mask]
    if existing_rows.empty:
        return target_seeds, None
    
    existing_row = existing_rows.iloc[0]
    completed_seeds = get_completed_seeds(existing_row)
    missing_seeds = [s for s in target_seeds if s not in completed_seeds]
    
    return missing_seeds, existing_row

def compute_statistics(correlations: List[float], target_seeds: List[int]) -> Dict[str, Any]:
    """
    Compute statistical measures for correlation values.
    
    Args:
        correlations: List of correlation values
        target_seeds: List of target seeds (for counting)
        
    Returns:
        Dictionary with statistical measures
    """
    valid_corrs = [c for c in correlations if not pd.isna(c)]
    num_successful = len(valid_corrs)
    
    if num_successful == 0:
        return {
            'mean_correlation': np.nan,
            'std_correlation': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'num_successful_runs': 0
        }
    
    mean_corr = np.mean(valid_corrs)
    
    if num_successful == 1:
        return {
            'mean_correlation': mean_corr,
            'std_correlation': 0.0,
            'ci_lower': mean_corr,
            'ci_upper': mean_corr,
            'num_successful_runs': num_successful
        }
    
    std_corr = np.std(valid_corrs, ddof=1)
    
    # 95% confidence interval using t-distribution
    alpha = 0.05
    df = num_successful - 1
    t_critical = t.ppf(1 - alpha/2, df)
    margin_error = t_critical * (std_corr / np.sqrt(num_successful))
    
    ci_lower = mean_corr - margin_error
    ci_upper = mean_corr + margin_error
    
    return {
        'mean_correlation': mean_corr,
        'std_correlation': std_corr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'num_successful_runs': num_successful
    }

def merge_results(
    existing_row: Optional[pd.Series],
    new_correlations: Dict[int, float],
    dataset: str,
    measure: str,
    metric: str,
    metric_class: str,
    target_seeds: List[int],
    errors: List[str]
) -> Dict[str, Any]:
    """
    Merge new correlation results with existing results.
    
    Returns:
        Complete result dictionary for this metric
    """
    # Start with existing data if available
    if existing_row is not None:
        result = existing_row.to_dict()
        # Update individual seed columns
        for seed, corr in new_correlations.items():
            result[f'seed_{seed}'] = corr
    else:
        # Create new result
        result = {
            'dataset': dataset,
            'measure': measure,
            'metric': metric,
            'metric_class': metric_class
        }
        # Initialize all seed columns
        for seed in target_seeds:
            result[f'seed_{seed}'] = new_correlations.get(seed, np.nan)
    
    # Collect all correlations for statistics
    all_correlations = []
    for seed in target_seeds:
        corr = result.get(f'seed_{seed}')
        if pd.notna(corr):
            all_correlations.append(corr)
    
    # Compute statistics
    stats = compute_statistics(all_correlations, target_seeds)
    result.update(stats)
    
    # Handle errors
    existing_errors = result.get('errors', '')
    if existing_errors and errors:
        result['errors'] = f"{existing_errors}; {'; '.join(errors)}"
    elif errors:
        result['errors'] = '; '.join(errors)
    elif 'errors' not in result:
        result['errors'] = ''
    
    return result

def format_mean_ci(row: pd.Series) -> str:
    """Format mean ± CI for easy copy-pasting into tables."""
    if pd.isna(row.get('mean_correlation')) or pd.isna(row.get('ci_lower')) or pd.isna(row.get('ci_upper')):
        return "N/A"
    mean_val = row['mean_correlation']
    ci_range = (row['ci_upper'] - row['ci_lower']) / 2
    return f"{mean_val:.4f} ± {ci_range:.4f}"

def save_results_with_formatting(results: List[Dict[str, Any]], output_file: str, existing_results_df: pd.DataFrame):
    """
    Save results with proper formatting and preservation of existing data.
    """
    # Create DataFrame from processed results
    new_results_df = pd.DataFrame(results)
    
    # Preserve existing results that weren't processed
    if not existing_results_df.empty:
        processed_ids = set()
        for result in results:
            metric_id = (result['dataset'], result['measure'], result['metric'], result['metric_class'])
            processed_ids.add(metric_id)
        
        preserved_results = []
        for _, existing_row in existing_results_df.iterrows():
            existing_id = (existing_row['dataset'], existing_row['measure'], existing_row['metric'], existing_row['metric_class'])
            if existing_id not in processed_ids:
                preserved_results.append(existing_row.to_dict())
        
        if preserved_results:
            final_results = preserved_results + results
            results_df = pd.DataFrame(final_results)
        else:
            results_df = new_results_df
    else:
        results_df = new_results_df
    
    # Add formatted column
    results_df['mean_±_ci'] = results_df.apply(format_mean_ci, axis=1)
    
    # Fix column ordering: metadata, seeds in ascending order, then statistics
    base_columns = ['dataset', 'measure', 'metric', 'metric_class']
    
    # Get all seed columns and sort them numerically
    seed_columns = [col for col in results_df.columns if col.startswith('seed_')]
    seed_columns.sort(key=lambda x: int(x.replace('seed_', '')))
    
    # Statistics columns in logical order
    stats_columns = [
        'mean_correlation', 'std_correlation', 'ci_lower', 'ci_upper', 
        'num_successful_runs', 'errors', 'mean_±_ci'
    ]
    
    # Build final column order
    final_columns = base_columns + seed_columns + [col for col in stats_columns if col in results_df.columns]
    
    # Reorder DataFrame columns
    results_df = results_df.reindex(columns=final_columns)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save results
    results_df.to_csv(output_file, index=False)

class ExperimentRunner:
    """
    Base class for running metric stability experiments across multiple seeds.
    
    This class handles the common workflow:
    1. Load existing results
    2. Determine which metrics/seeds need to be run
    3. Run experiments with proper caching
    4. Compute statistics and save results
    5. Generate summary reports
    """
    
    def __init__(
        self,
        seeds: List[int],
        correlation_funcs: Dict[str, Callable],
        cache_base_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        self.seeds = seeds
        self.correlation_funcs = correlation_funcs
        self.cache_base_dir = cache_base_dir
        self.logger = logger or logging.getLogger("experiment")
    
    def run_single_metric_seed(
        self,
        dataset_name: str,
        measure: str,
        metric_instance: Any,
        seed: int,
        correlation_func: Callable,
        correlation_name: str
    ) -> float:
        """
        Run a single metric on a single seed and return correlation.
        
        This method should be overridden by subclasses to implement
        the specific metric evaluation logic.
        """
        raise NotImplementedError("Subclasses must implement run_single_metric_seed")
    
    def instantiate_metric(
        self,
        metric_class: type,
        seed: int,
        dataset_name: str,
        **kwargs
    ) -> Any:
        """
        Instantiate a metric with proper caching.
        
        This method should be overridden by subclasses to implement
        specific metric instantiation logic.
        """
        raise NotImplementedError("Subclasses must implement instantiate_metric")
    
    def run_experiment(
        self,
        metrics_to_test: List[Dict[str, Any]],
        output_base: str,
        dataset_filter: Optional[List[str]] = None,
        measure_filter: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run the complete experiment across all correlation functions.
        
        Args:
            metrics_to_test: List of metric specifications
            output_base: Base path for output files
            dataset_filter: Optional dataset filter
            measure_filter: Optional measure filter
            
        Returns:
            Dictionary mapping correlation names to results
        """
        all_results = {}
        
        # Apply filters once
        filtered_metrics = self._apply_filters(metrics_to_test, dataset_filter, measure_filter)
        
        # Cache to store computed correlations across all correlation functions
        correlation_cache = {}
        
        for correlation_name, correlation_func in self.correlation_funcs.items():
            self.logger.info(f"\n=== Processing {correlation_name.upper()} correlation ===")
            
            # Create correlation-specific output file
            if len(self.correlation_funcs) > 1:
                current_output_file = f"{output_base}_{correlation_name}.csv"
                self.logger.info(f"Output file: {current_output_file}")
            else:
                current_output_file = f"{output_base}.csv"
            
            results = self._run_single_correlation(
                filtered_metrics,
                current_output_file,
                correlation_name,
                correlation_func,
                correlation_cache
            )
            
            all_results[correlation_name] = results
        
        return all_results
    
    def _apply_filters(
        self,
        metrics_to_test: List[Dict[str, Any]],
        dataset_filter: Optional[List[str]],
        measure_filter: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Apply dataset and measure filters to metrics list."""
        filtered = metrics_to_test
        
        if dataset_filter:
            original_count = len(filtered)
            filtered = [m for m in filtered if m['dataset'] in dataset_filter]
            self.logger.info(f"  Filtered: {original_count} → {len(filtered)} metrics")
        
        if measure_filter:
            original_count = len(filtered)
            filtered = [m for m in filtered if m['measure'] in measure_filter]
            self.logger.info(f"  Filtered: {original_count} → {len(filtered)} metrics")
        
        return filtered
    
    def _run_single_correlation(
        self,
        metrics_to_test: List[Dict[str, Any]],
        output_file: str,
        correlation_name: str,
        correlation_func: Callable,
        correlation_cache: Dict[Tuple[str, str, str, int], Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Run a single correlation experiment across all metrics.
        
        Args:
            metrics_to_test: List of metric specifications
            output_file: Output file for results
            correlation_name: Name of the correlation function
            correlation_func: Correlation function to use
            correlation_cache: Cache to store computed correlations
            
        Returns:
            List of results for this correlation
        """
        # Load existing results
        existing_results_df = load_existing_results(output_file)
        if not existing_results_df.empty:
            self.logger.info(f"Found existing results with {len(existing_results_df)} metrics")
            existing_seeds = get_seed_columns(existing_results_df)
            self.logger.info(f"  Existing seeds: {[s.replace('seed_', '') for s in existing_seeds]}")
        else:
            self.logger.info("No existing results found, starting fresh")
        
        # Process each metric
        results = []
        skipped_count = 0
        resumed_count = 0
        new_runs_count = 0
        
        for i, metric_spec in enumerate(metrics_to_test, 1):
            dataset = metric_spec['dataset']
            measure = metric_spec['measure']
            metric = metric_spec['metric']
            metric_class = metric_spec['metric_class']
            
            self.logger.info(f"\n[{i}/{len(metrics_to_test)}] Checking: {dataset} - {measure} - {metric}")
            
            # Determine which seeds to run
            seeds_to_run, existing_row = determine_seeds_to_run(
                existing_results_df, self.seeds, dataset, measure, metric, metric_class
            )
            
            if not seeds_to_run:
                self.logger.info(f"  ✅ Skipping: All seeds completed successfully")
                completed_seeds = get_completed_seeds(existing_row)
                self.logger.info(f"     Completed seeds: {completed_seeds}")
                skipped_count += 1
                # Still add to results for final statistics
                results.append(existing_row.to_dict())
                continue
            
            if existing_row is not None:
                self.logger.info(f"  🔄 Resuming: Running missing seeds {seeds_to_run}")
                resumed_count += 1
            else:
                self.logger.info(f"  🆕 New metric: Running seeds {seeds_to_run}")
                new_runs_count += 1
            
            # Run the metric for missing seeds with caching
            result = self._run_metric_with_seeds_cached(
                metric_spec, seeds_to_run, existing_row, correlation_name, correlation_func, 
                output_file, existing_results_df, results, correlation_cache
            )
            # Update results list (result already added during seed processing)
            # Remove old result if it exists and add the final result
            results = [r for r in results if not (
                r['dataset'] == dataset and r['measure'] == measure and 
                r['metric'] == metric and r['metric_class'] == metric_class
            )]
            results.append(result)
        
        # Save final results
        save_results_with_formatting(results, output_file, existing_results_df)
        
        # Log summary
        self._log_summary(results, skipped_count, resumed_count, new_runs_count, correlation_name)
        
        return results

    def _log_summary(
        self,
        results: List[Dict[str, Any]],
        skipped_count: int,
        resumed_count: int,
        new_runs_count: int,
        correlation_name: str
    ):
        """Log experiment summary statistics."""
        self.logger.info(f"\n=== RESUME SUMMARY for {correlation_name.upper()} ===")
        self.logger.info(f"Skipped (already complete): {skipped_count}")
        self.logger.info(f"Resumed (partial completion): {resumed_count}")
        self.logger.info(f"New metrics: {new_runs_count}")
        self.logger.info(f"Total processed: {len(results)}")
        
        self.logger.info(f"\n=== FINAL SUMMARY for {correlation_name.upper()} ===")
        self.logger.info(f"Total metrics in results: {len(results)}")
        
        # Calculate success statistics
        results_df = pd.DataFrame(results)
        successful_metrics = results_df[results_df['num_successful_runs'] > 0]
        fully_successful = results_df[results_df['num_successful_runs'] == len(self.seeds)]
        
        self.logger.info(f"Metrics with at least 1 successful run: {len(successful_metrics)}")
        self.logger.info(f"Metrics with all {len(self.seeds)} successful runs: {len(fully_successful)}")
        
        if len(successful_metrics) > 0:
            avg_std = successful_metrics['std_correlation'].mean()
            self.logger.info(f"Average standard deviation of correlations: {avg_std:.4f}")
            
            # Find most and least stable metrics
            most_stable = successful_metrics.loc[successful_metrics['std_correlation'].idxmin()]
            least_stable = successful_metrics.loc[successful_metrics['std_correlation'].idxmax()]
            
            self.logger.info(f"Most stable metric (lowest std): {most_stable['metric']} (std={most_stable['std_correlation']:.4f})")
            if len(successful_metrics) > 1:
                self.logger.info(f"Least stable metric (highest std): {least_stable['metric']} (std={least_stable['std_correlation']:.4f})")

    def _run_metric_with_seeds_cached(
        self,
        metric_spec: Dict[str, Any],
        seeds_to_run: List[int],
        existing_row: Optional[pd.Series],
        correlation_name: str,
        correlation_func: Callable,
        output_file: str,
        existing_results_df: pd.DataFrame,
        all_results: List[Dict[str, Any]],
        correlation_cache: Dict[Tuple[str, str, str, int], Dict[str, float]]
    ) -> Dict[str, Any]:
        """Run a metric across multiple seeds with correlation caching."""
        dataset = metric_spec['dataset']
        measure = metric_spec['measure']
        metric = metric_spec['metric']
        metric_class_name = metric_spec['metric_class']
        
        new_correlations = {}
        errors = []
        
        for seed in seeds_to_run:
            self.logger.info(f"  Running seed {seed}...")
            
            # Create cache key for this metric and seed
            cache_key = (dataset, measure, metric, seed)
            
            # Check if we have cached results for this metric and seed
            if cache_key in correlation_cache:
                # Use cached correlation for this specific correlation function
                if correlation_name in correlation_cache[cache_key]:
                    cached_correlation = correlation_cache[cache_key][correlation_name]
                    new_correlations[seed] = cached_correlation
                    self.logger.debug(f"    Used cached correlation {cached_correlation:.4f}")
                    
                    # Save intermediate progress after using cache
                    temp_result = merge_results(
                        existing_row, new_correlations, dataset, measure, metric, metric_class_name, 
                        self.seeds, errors
                    )
                    
                    # Update the results list with current progress
                    updated_results = [r for r in all_results if not (
                        r['dataset'] == dataset and r['measure'] == measure and 
                        r['metric'] == metric and r['metric_class'] == metric_class_name
                    )]
                    updated_results.append(temp_result)
                    
                    # Save intermediate results after using cache
                    save_results_with_formatting(updated_results, output_file, existing_results_df)
                    self.logger.debug(f"    💾 Saved progress after seed {seed} (cached)")
                    continue
            
            # Compute correlation if not cached
            try:
                # run_single_metric_seed now returns ALL correlations for this seed
                all_correlations_for_seed = self.run_single_metric_seed(
                    dataset, measure, metric_spec, seed
                )
                
                # Extract the specific correlation we need for this function
                correlation = all_correlations_for_seed[correlation_name]
                new_correlations[seed] = correlation
                self.logger.debug(f"    Found correlation {correlation:.4f}")
                
                # Cache ALL correlation results from this computation
                correlation_cache[cache_key] = all_correlations_for_seed
                
                # Save intermediate progress after each seed
                temp_result = merge_results(
                    existing_row, new_correlations, dataset, measure, metric, metric_class_name, 
                    self.seeds, errors
                )
                
                # Update the results list with current progress
                updated_results = [r for r in all_results if not (
                    r['dataset'] == dataset and r['measure'] == measure and 
                    r['metric'] == metric and r['metric_class'] == metric_class_name
                )]
                updated_results.append(temp_result)
                
                # Save intermediate results after each seed
                save_results_with_formatting(updated_results, output_file, existing_results_df)
                self.logger.debug(f"    💾 Saved progress after seed {seed}")
                
            except Exception as e:
                error_msg = f"Seed {seed}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(f"    ❌ {error_msg}")
                new_correlations[seed] = np.nan
        
        # Merge with existing results
        result = merge_results(
            existing_row, new_correlations, dataset, measure, metric, metric_class_name, 
            self.seeds, errors
        )
        
        # Log completion status
        successful_runs = result['num_successful_runs']
        total_target = len(self.seeds)
        self.logger.info(f"  Completed: {successful_runs}/{total_target} successful runs, mean={result.get('mean_correlation', 'N/A'):.4f}")
        
        if existing_row is not None:
            self.logger.info(f"  ✅ Updated existing metric")
        else:
            self.logger.info(f"  ✅ Added new metric")
        
        return result 