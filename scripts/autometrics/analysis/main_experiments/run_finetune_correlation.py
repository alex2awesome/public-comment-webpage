#!/usr/bin/env python3
"""
Fine-tuned Metric Correlation Stability Analysis

This script tests the correlation stability of fine-tuned ModernBERT metrics across multiple 
random seeds. It trains metrics using FinetuneGenerator on the persistent trainset and 
measures their correlation with human annotations on the persistent testset.

The script follows the same patterns as run_llm_judge_correlation.py but handles:
- Fine-tuning ModernBERT models with PEFT/LoRA for each dataset and measure
- Training on persistent trainset, evaluation on persistent testset
- Model storage and caching across seeds
- Reference-free vs reference-based metrics based on dataset

Example usage:
    python run_finetune_correlation.py --dataset HelpSteer SimpEval
    python run_finetune_correlation.py --correlation kendall --model-save-dir /custom/path
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from collections import defaultdict
import traceback
import torch
from pathlib import Path

# Add autometrics to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autometrics.dataset.Dataset import Dataset
from autometrics.experiments.correlation.correlation import CorrelationExperiment, correlation_func_from_name
from autometrics.generator.FinetuneGenerator import FinetuneGenerator
from autometrics.metrics.generated.GeneratedFinetunedMetric import (
    GeneratedRefFreeFinetunedMetric,
    GeneratedRefBasedFinetunedMetric
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Get logger for this script
    logger = logging.getLogger(__name__)
    
    # Suppress verbose logging from dependencies when not in verbose mode
    if not verbose:
        # Transformers can be very verbose
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('transformers.trainer').setLevel(logging.WARNING)
        logging.getLogger('transformers.modeling_utils').setLevel(logging.WARNING)
        logging.getLogger('transformers.tokenization_utils').setLevel(logging.WARNING)
        
        # PEFT and model loading
        logging.getLogger('peft').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('torch.nn').setLevel(logging.WARNING)
        
        # Cache-related logging
        logging.getLogger('diskcache').setLevel(logging.WARNING)
        logging.getLogger('diskcache.core').setLevel(logging.WARNING)
        
        # HTTP and API related
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        
        # ML libraries
        logging.getLogger('sklearn').setLevel(logging.WARNING)
        logging.getLogger('datasets').setLevel(logging.WARNING)
        logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
        
        # Autometrics internals
        logging.getLogger('autometrics').setLevel(logging.WARNING)
        logging.getLogger('autometrics.metrics').setLevel(logging.WARNING)
        logging.getLogger('autometrics.metrics.Metric').setLevel(logging.WARNING)
        logging.getLogger('autometrics.experiments').setLevel(logging.WARNING)
        
        # General noise suppression
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        logging.getLogger('concurrent.futures').setLevel(logging.WARNING)
        
        # Set root logger to WARNING to catch anything else
        logging.getLogger().setLevel(logging.WARNING)
        
        # But keep our script's logger at INFO so we still see our progress messages
        logger.setLevel(logging.INFO)
        
        # Also make sure we see warnings and errors from our specific logger
        if logger.level > logging.WARNING:
            logger.setLevel(logging.WARNING)
    
    return logger


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
    
    raise ValueError(f"Unknown dataset: {dataset_name}")


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


def get_available_datasets_measures() -> List[Tuple[str, str]]:
    """Get all available dataset-measure combinations for fine-tuning."""
    datasets_measures = []
    
    # Datasets with their target measures
    dataset_configs = {
        "HelpSteer": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
        "HelpSteer2": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
        "SimpDA": ["fluency", "meaning", "simplicity"], 
        "SimpEval": ["score"],  # SimpEval only has 'score' as target column
        "SummEval": ["coherence", "consistency", "fluency", "relevance"],
        "Primock57": ["inc_plus_omi", "incorrect", "omissions", "time_sec"],
        "CoGymTabularOutcome": ["outcomeRating"],
        "CoGymTabularProcess": ["agentRating", "communicationRating"],
        "CoGymTravelOutcome": ["outcomeRating"],
        "CoGymTravelProcess": ["agentRating", "communicationRating"],
        "CoGymLessonOutcome": ["outcomeRating"],
        "CoGymLessonProcess": ["agentRating", "communicationRating"],
        "EvalGenMedical": ["grade"],
        "EvalGenProduct": ["grade"],
        "RealHumanEval": ["accepted"]
        # Design2Code removed: it's a PairwiseDataset with binary targets, not suitable for regression fine-tuning
    }
    
    for dataset_name, measures in dataset_configs.items():
        for measure in measures:
            datasets_measures.append((dataset_name, measure))
    
    return datasets_measures


def create_finetune_metric(
    dataset_name: str,
    measure: str,
    seed: int,
    model_save_dir: Optional[str] = None,
    logger: logging.Logger = None
) -> Optional:
    """
    Create a fine-tuned metric for a dataset-measure combination using FinetuneGenerator.
    
    Returns:
        Fine-tuned metric instance or None if training failed
    """
    try:
        if logger:
            logger.info(f"Creating fine-tuned metric for {dataset_name}.{measure} with seed {seed}")
        
        # Load dataset and get train split
        dataset_instance = load_dataset(dataset_name)
        train_dataset, _, _ = dataset_instance.load_permanent_splits()
        
        if logger:
            logger.info(f"Loaded dataset {dataset_name}: train_size={len(train_dataset.get_dataframe())}")
        
        # Create FinetuneGenerator with seed-specific configuration
        generator = FinetuneGenerator(
            name=f"FinetuneGenerator_{dataset_name}_{measure}_seed{seed}",
            seed=seed,
            model_save_dir=model_save_dir,
            num_train_epochs=3,  # Conservative epochs for stability
            batch_size=16,
            learning_rate=5e-5
        )
        
        # Generate the fine-tuned metric
        if logger:
            logger.info(f"Starting fine-tuning for {dataset_name}.{measure}...")
        
        start_time = time.time()
        metrics = generator.generate(
            dataset=train_dataset,
            target_measure=measure,
            n_metrics=1  # Only generate one metric per seed
        )
        training_time = time.time() - start_time
        
        if metrics and len(metrics) > 0:
            metric = metrics[0]
            if logger:
                logger.info(f"Fine-tuning completed in {training_time:.1f}s: {metric.name}")
            return metric
        else:
            if logger:
                logger.error(f"FinetuneGenerator returned no metrics for {dataset_name}.{measure}")
            return None
            
    except Exception as e:
        if logger:
            logger.error(f"Failed to create fine-tuned metric for {dataset_name}.{measure}: {str(e)}")
            logger.debug(traceback.format_exc())
        return None


def run_single_metric_seed(
    dataset_name: str,
    measure: str,
    metric_instance,
    seed: int,
    correlation_funcs: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Dict[str, float]]:
    """
    Run a single fine-tuned metric on a single seed and return correlations and p-values for all correlation functions.
    
    Returns:
        Dict mapping correlation function name to dict of {'correlation': float, 'p_value': float}
    """
    try:
        # Load dataset and get test split (we evaluate on test, trained on train)
        dataset = load_dataset(dataset_name)
        _, _, test_dataset = dataset.load_permanent_splits()
        
        if logger:
            logger.info(f"Evaluating on test set: {len(test_dataset.get_dataframe())} examples")
        
        # Run correlation experiment with ALL correlation functions at once
        experiment = CorrelationExperiment(
            name=f"Finetune Test - {dataset_name} - {measure} - {metric_instance.name}",
            description=f"Testing fine-tuned metric correlation for {metric_instance.name} on {dataset_name}",
            metrics=[metric_instance],
            output_dir=f"/tmp/finetune_test_{seed}",
            dataset=test_dataset,
            correlation_funcs=correlation_funcs,
            seed=seed,
            should_split=False
        )
        
        # Run experiment and extract correlations for ALL functions
        all_correlations = experiment.run(print_results=False)
        
        # Extract correlations and p-values for all correlation functions
        results_by_func = {}
        
        for func_name, correlations_for_func in all_correlations.items():
            if measure not in correlations_for_func:
                raise ValueError(f"Measure {measure} not found in correlation results for {func_name}")
            
            df_corr = correlations_for_func[measure]
            
            # Find the correlation for our specific metric
            metric_row = df_corr[df_corr['Metric'] == metric_instance.name]
            if metric_row.empty:
                raise ValueError(f"Metric {metric_instance.name} not found in correlation results")
            
            correlation = metric_row.iloc[0]['Correlation']
            p_value = metric_row.iloc[0]['P-value']
            results_by_func[func_name] = {'correlation': correlation, 'p_value': p_value}
        
        if logger:
            logger.debug(f"Seed {seed} results: {results_by_func}")
        return results_by_func
        
    except Exception as e:
        if logger:
            logger.error(f"Error running seed {seed}: {str(e)}")
            logger.debug(traceback.format_exc())
        raise


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistical measures for correlation values."""
    valid_values = [v for v in values if not pd.isna(v)]
    n = len(valid_values)
    
    if n == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'ci_range': np.nan,
            'num_successful_runs': 0
        }
    
    mean_val = np.mean(valid_values)
    
    if n == 1:
        return {
            'mean': mean_val,
            'std': 0.0,
            'ci_lower': mean_val,
            'ci_upper': mean_val,
            'ci_range': 0.0,
            'num_successful_runs': n
        }
    
    std_val = np.std(valid_values, ddof=1)
    
    # 95% confidence interval using t-distribution
    alpha = 0.05
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_value * std_val / np.sqrt(n)
    
    return {
        'mean': mean_val,
        'std': std_val,
        'ci_lower': mean_val - margin_error,
        'ci_upper': mean_val + margin_error,
        'ci_range': margin_error,
        'num_successful_runs': n
    }


def format_mean_ci(mean: float, ci_range: float) -> str:
    """Format mean ± CI for easy copying to papers."""
    if np.isnan(mean) or np.isnan(ci_range):
        return "N/A"
    return f"{mean:.4f} ± {ci_range:.4f}"


def sort_columns_for_output(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame columns in a logical order with seed columns grouped and sorted numerically."""
    if df.empty:
        return df
    
    # Base columns that should come first
    base_columns = ['dataset', 'measure', 'metric', 'metric_class', 'num_successful_runs', 'errors']
    
    # Extract seed columns and sort them numerically
    correlation_columns = []
    p_value_columns = []
    
    for col in df.columns:
        if col.startswith('seed_') and col.endswith('_correlation'):
            correlation_columns.append(col)
        elif col.startswith('seed_') and col.endswith('_p_value'):
            p_value_columns.append(col)
    
    # Sort by seed number (extract number from column name)
    def extract_seed_number(col_name):
        return int(col_name.split('_')[1])
    
    correlation_columns.sort(key=extract_seed_number)
    p_value_columns.sort(key=extract_seed_number)
    
    # Statistics columns
    stats_columns = [
        'mean_correlation', 'std_correlation', 'ci_lower_correlation', 'ci_upper_correlation',
        'mean_p_value', 'std_p_value', 'ci_lower_p_value', 'ci_upper_p_value'
    ]
    
    # Construct final column order
    final_columns = []
    for col in base_columns:
        if col in df.columns:
            final_columns.append(col)
    
    final_columns.extend(correlation_columns)
    final_columns.extend(p_value_columns)
    
    for col in stats_columns:
        if col in df.columns:
            final_columns.append(col)
    
    # Add any remaining columns not covered above
    for col in df.columns:
        if col not in final_columns:
            final_columns.append(col)
    
    return df[final_columns]


def save_results(results: Dict[str, Any], output_file: str, logger: logging.Logger):
    """Save results to CSV file with properly sorted columns."""
    try:
        df = pd.DataFrame(results)
        df = sort_columns_for_output(df)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {e}")
        raise


def load_existing_results(output_file: str) -> Dict[str, Any]:
    """Load existing results from output file if it exists."""
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            return df.to_dict('records')
        except Exception as e:
            logging.warning(f"Could not read existing results file {output_file}: {e}")
    return []


def merge_with_existing_results(new_results: List[Dict], output_file: str, logger: logging.Logger) -> List[Dict]:
    """
    Merge new results with existing results, preserving all seed columns.
    
    Args:
        new_results: List of new result dictionaries
        output_file: Path to existing results file
        logger: Logger instance
        
    Returns:
        List of merged result dictionaries
    """
    if not os.path.exists(output_file):
        logger.info(f"No existing results file found at {output_file}")
        return new_results
    
    try:
        # Load existing results
        existing_df = pd.read_csv(output_file)
        logger.info(f"Loaded {len(existing_df)} existing results from {output_file}")
        
        # Convert new results to DataFrame
        new_df = pd.DataFrame(new_results)
        
        if existing_df.empty:
            logger.info("Existing file is empty, using new results")
            return new_results
        
        # Find matching rows (same dataset, measure, metric)
        merge_keys = ['dataset', 'measure', 'metric']
        merged_results = []
        
        for _, new_row in new_df.iterrows():
            # Find existing row with matching keys
            mask = True
            for key in merge_keys:
                mask = mask & (existing_df[key] == new_row[key])
            
            matching_rows = existing_df[mask]
            
            if len(matching_rows) > 0:
                # Merge with existing row
                existing_row = matching_rows.iloc[0].copy()
                merged_row = existing_row.to_dict()
                
                # Add new seed columns from new_row
                for col in new_row.index:
                    if col.startswith('seed_') and col.endswith(('_correlation', '_p_value')):
                        merged_row[col] = new_row[col]
                    elif col in ['num_successful_runs', 'errors', 'mean_correlation', 'std_correlation', 
                                'ci_lower_correlation', 'ci_upper_correlation', 'mean_±_ci', 'mean_p_value', 'std_p_value',
                                'ci_lower_p_value', 'ci_upper_p_value']:
                        # These will be recalculated after merging all seeds
                        pass
                
                # Collect all correlation and p-value data for recalculation
                all_correlations = []
                all_p_values = []
                all_errors = []
                
                for col in merged_row.keys():
                    if col.startswith('seed_') and col.endswith('_correlation'):
                        val = merged_row[col]
                        if pd.notna(val):
                            all_correlations.append(val)
                    elif col.startswith('seed_') and col.endswith('_p_value'):
                        val = merged_row[col]
                        if pd.notna(val):
                            all_p_values.append(val)
                
                # Parse errors from both old and new
                if pd.notna(existing_row.get('errors', '')) and existing_row['errors']:
                    all_errors.extend(existing_row['errors'].split('; '))
                if pd.notna(new_row.get('errors', '')) and new_row['errors']:
                    all_errors.extend(new_row['errors'].split('; '))
                
                # Recalculate statistics (use absolute values for correlations)
                if all_correlations:
                    corr_stats = compute_statistics([abs(c) for c in all_correlations if not pd.isna(c)])
                    merged_row.update({
                        'num_successful_runs': corr_stats['num_successful_runs'],
                        'mean_correlation': corr_stats['mean'],
                        'std_correlation': corr_stats['std'],
                        'ci_lower_correlation': corr_stats['ci_lower'],
                        'ci_upper_correlation': corr_stats['ci_upper'],
                        'mean_±_ci': format_mean_ci(corr_stats['mean'], corr_stats['ci_range'])
                    })
                
                if all_p_values:
                    pval_stats = compute_statistics(all_p_values)
                    merged_row.update({
                        'mean_p_value': pval_stats['mean'],
                        'std_p_value': pval_stats['std'],
                        'ci_lower_p_value': pval_stats['ci_lower'],
                        'ci_upper_p_value': pval_stats['ci_upper']
                    })
                
                merged_row['errors'] = '; '.join(all_errors) if all_errors else ''
                merged_results.append(merged_row)
                logger.info(f"Merged results for {new_row['dataset']}.{new_row['measure']} - now has {len(all_correlations)} seeds")
            else:
                # No existing row, add as new
                merged_results.append(new_row.to_dict())
                logger.info(f"Added new result for {new_row['dataset']}.{new_row['measure']}")
        
        # Add any existing rows that don't have new data
        for _, existing_row in existing_df.iterrows():
            mask = True
            for key in merge_keys:
                mask = mask & (new_df[key] == existing_row[key])
            
            if len(new_df[mask]) == 0:
                # No new data for this row, keep existing
                merged_results.append(existing_row.to_dict())
                logger.info(f"Preserved existing result for {existing_row['dataset']}.{existing_row['measure']}")
        
        # Sort columns before returning
        merged_df = pd.DataFrame(merged_results)
        merged_df = sort_columns_for_output(merged_df)
        
        logger.info(f"Merge complete: {len(merged_results)} total results")
        return merged_df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error merging with existing results: {e}")
        logger.info("Using new results only")
        return new_results


def main():
    parser = argparse.ArgumentParser(
        description="Test correlation stability of fine-tuned ModernBERT metrics across multiple seeds"
    )
    
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds to test"
    )
    parser.add_argument(
        "--correlation",
        default="all",
        help="Correlation function(s): 'pearson', 'spearman', 'kendall', or 'all' for all three"
    )
    parser.add_argument(
        "--dataset",
        nargs="*",
        help="Filter to specific datasets (e.g., HelpSteer SimpEval)"
    )
    parser.add_argument(
        "--measure",
        nargs="*",
        help="Filter to specific measures (e.g., helpfulness fluency)"
    )
    parser.add_argument(
        "--model-save-dir",
        default=None,
        help="Directory to save fine-tuned models (default: auto-detected)"
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output CSV file path (default: auto-generated based on correlation)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Handle correlation types
    if args.correlation.lower() == "all":
        correlation_types = ["kendall", "pearson", "spearman"]
    else:
        correlation_types = [args.correlation.lower()]
    
    # Validate correlation types
    valid_correlations = {"pearson", "spearman", "kendall"}
    for corr_type in correlation_types:
        if corr_type not in valid_correlations:
            logging.error(f"Unknown correlation function: {corr_type}")
            return 1
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("Starting fine-tuned metric correlation stability analysis")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Correlations: {correlation_types}")
    logger.info(f"Model save directory: {args.model_save_dir or 'auto-detected'}")
    
    # Check for GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        logger.info(f"Current GPU: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA not available - training will be slow on CPU!")
    
    # Create base output directory and sub-results directory
    os.makedirs("results/main_runs/baselines", exist_ok=True)
    os.makedirs("results/main_runs/baselines/finetune_sub_results", exist_ok=True)
    
    # Get all available dataset-measure combinations
    all_dataset_measures = get_available_datasets_measures()
    
    # Filter by dataset if specified
    if args.dataset:
        allowed_datasets = set(args.dataset)
        all_dataset_measures = [(d, m) for d, m in all_dataset_measures if d in allowed_datasets]
        logger.info(f"Filtered to datasets: {args.dataset}")
    
    # Filter by measure if specified
    if args.measure:
        allowed_measures = set(args.measure)
        all_dataset_measures = [(d, m) for d, m in all_dataset_measures if m in allowed_measures]
        logger.info(f"Filtered to measures: {args.measure}")
    
    if not all_dataset_measures:
        logger.error("No dataset-measure combinations to process after filtering")
        return 1
    
    logger.info(f"Processing {len(all_dataset_measures)} dataset-measure combinations")
    
    # Group by dataset for dataset-specific output files
    datasets_measures = {}
    for dataset_name, measure in all_dataset_measures:
        if dataset_name not in datasets_measures:
            datasets_measures[dataset_name] = []
        datasets_measures[dataset_name].append(measure)
    
    # Process each correlation type
    for correlation_type in correlation_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {correlation_type.upper()} correlation analysis")
        logger.info(f"{'='*60}")
        
        # Get correlation function
        correlation_func = correlation_func_from_name(correlation_type)
        correlation_funcs = {correlation_type: correlation_func}
        
        # All results across datasets for final merging
        all_results = []
        dataset_output_files = []
        
        # Process each dataset separately
        for dataset_name, measures in datasets_measures.items():
            logger.info(f"\n--- Processing dataset: {dataset_name} ---")
            
            # Dataset-specific output file in sub-results directory
            dataset_output_file = f"results/main_runs/baselines/finetune_sub_results/finetune_{correlation_type}_{dataset_name}.csv"
            dataset_output_files.append(dataset_output_file)
            
            logger.info(f"Dataset output file: {dataset_output_file}")
            
            # Results storage for this dataset
            dataset_results = []
            
            # Process each measure for this dataset
            for measure in measures:
                logger.info(f"Processing {dataset_name}.{measure}")
                
                # Create metric name
                metric_name_template = f"Finetune-{dataset_name}-{measure}"
                
                # Run correlation for each seed
                correlations = []
                p_values = []
                errors = []
                
                for seed in args.seeds:
                    try:
                        logger.info(f"  Running seed {seed}...")
                        
                        # Create fine-tuned metric for this seed
                        metric = create_finetune_metric(
                            dataset_name=dataset_name,
                            measure=measure,
                            seed=seed,
                            model_save_dir=args.model_save_dir,
                            logger=logger
                        )
                        
                        if metric is None:
                            raise ValueError("Failed to create fine-tuned metric")
                        
                        # Run the correlation
                        corr_results = run_single_metric_seed(
                            dataset_name=dataset_name,
                            measure=measure,
                            metric_instance=metric,
                            seed=seed,
                            correlation_funcs=correlation_funcs,
                            logger=logger
                        )
                        
                        correlation = corr_results[correlation_type]['correlation']
                        p_value = corr_results[correlation_type]['p_value']
                        correlations.append(correlation)
                        p_values.append(p_value)
                        logger.info(f"    Correlation: {correlation:.4f}, p-value: {p_value:.4f}")
                        
                    except Exception as e:
                        error_msg = f"Seed {seed}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(f"    Failed: {error_msg}")
                        correlations.append(np.nan)
                        p_values.append(np.nan)
                
                # Compute statistics for correlations and p-values (use absolute values for correlations)
                corr_stats = compute_statistics([abs(c) for c in correlations if not pd.isna(c)])
                pval_stats = compute_statistics(p_values)
                
                # Determine metric class based on dataset
                try:
                    dataset_instance = load_dataset(dataset_name)
                    reference_columns = dataset_instance.get_reference_columns()
                    has_references = reference_columns is not None and len(reference_columns) > 0
                    metric_class_name = "GeneratedRefBasedFinetunedMetric" if has_references else "GeneratedRefFreeFinetunedMetric"
                except:
                    metric_class_name = "GeneratedRefFreeFinetunedMetric"  # Default fallback
                
                # Create result row
                result = {
                    'dataset': dataset_name,
                    'measure': measure,
                    'metric': f"{metric_name_template}-ModernBERT",
                    'metric_class': metric_class_name,
                    'num_successful_runs': corr_stats['num_successful_runs'],
                    'errors': '; '.join(errors) if errors else ''
                }
                
                # Add individual seed results for correlations
                for i, seed in enumerate(args.seeds):
                    result[f'seed_{seed}_correlation'] = correlations[i] if i < len(correlations) else np.nan
                    result[f'seed_{seed}_p_value'] = p_values[i] if i < len(p_values) else np.nan
                
                # Add correlation statistics
                result.update({
                    'mean_correlation': corr_stats['mean'],
                    'std_correlation': corr_stats['std'],
                    'ci_lower_correlation': corr_stats['ci_lower'],
                    'ci_upper_correlation': corr_stats['ci_upper'],
                    'mean_±_ci': format_mean_ci(corr_stats['mean'], corr_stats['ci_range'])
                })
                
                # Add p-value statistics
                result.update({
                    'mean_p_value': pval_stats['mean'],
                    'std_p_value': pval_stats['std'],
                    'ci_lower_p_value': pval_stats['ci_lower'],
                    'ci_upper_p_value': pval_stats['ci_upper']
                })
                
                dataset_results.append(result)
                logger.info(f"  Completed: mean_corr={corr_stats['mean']:.4f}, "
                            f"CI=[{corr_stats['ci_lower']:.4f}, {corr_stats['ci_upper']:.4f}], "
                            f"mean_p_val={pval_stats['mean']:.4f}")
            
            # Save dataset-specific results
            if dataset_results:
                save_results(dataset_results, dataset_output_file, logger)
                all_results.extend(dataset_results)  # Add to combined results
                logger.info(f"Dataset {dataset_name} results saved to {dataset_output_file}")
            else:
                logger.warning(f"No results generated for dataset {dataset_name}")
        
        # Create merged output file
        if args.output_file:
            merged_output_file = args.output_file
        else:
            merged_output_file = f"results/main_runs/baselines/finetune_{correlation_type}.csv"
        
        # Save merged results
        if all_results:
            merged_results = merge_with_existing_results(all_results, merged_output_file, logger)
            save_results(merged_results, merged_output_file, logger)
            
            # Print summary for this correlation type
            print(f"\nSummary of Fine-tuned Metric {correlation_type.upper()} Correlation Results:")
            print(f"Seeds: {args.seeds}")
            print(f"Merged results saved to: {merged_output_file}")
            print(f"Dataset-specific results saved to:")
            for dataset_file in dataset_output_files:
                print(f"  {dataset_file}")
            print(f"\nTop 5 most stable correlations:")
            
            # Sort by mean correlation (absolute value)
            df_results = pd.DataFrame(merged_results)
            df_results['abs_mean'] = df_results['mean_correlation'].abs()
            df_top = df_results.nlargest(5, 'abs_mean')
            
            for _, row in df_top.iterrows():
                mean_corr = row['mean_correlation']
                std_corr = row['std_correlation']
                ci_width = (row['ci_upper_correlation'] - row['ci_lower_correlation']) / 2
                print(f"  {row['dataset']}.{row['measure']}: "
                      f"{mean_corr:.4f} ± {ci_width:.4f} "
                      f"(95% CI: [{row['ci_lower_correlation']:.4f}, {row['ci_upper_correlation']:.4f}])")
            
        else:
            logger.error(f"No results generated for {correlation_type}")
    
    logger.info("Fine-tuned metric correlation analysis completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 