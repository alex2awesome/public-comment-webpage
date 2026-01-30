#!/usr/bin/env python3
"""
Merge LLM Judge Correlation Results

This script merges individual dataset-specific CSV files from the sub_results directory
into final consolidated files for each model and correlation type.

Usage:
    python scripts/llm_judge/merge_llm_judge_results.py --model gpt4o_mini --correlation all
    python scripts/llm_judge/merge_llm_judge_results.py --model qwen3_32b --correlation kendall
"""

import os
import sys
import argparse
import pandas as pd
import logging
from pathlib import Path
import glob

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def merge_dataset_files(model: str, correlation: str, sub_results_dir: str, output_dir: str, logger: logging.Logger) -> bool:
    """Merge all dataset-specific files for a given model and correlation type."""
    
    # Find all matching dataset files
    pattern = f"llm_judge_{model}_{correlation}_*.csv"
    search_path = os.path.join(sub_results_dir, pattern)
    dataset_files = glob.glob(search_path)
    
    if not dataset_files:
        logger.warning(f"No dataset files found matching pattern: {pattern}")
        return False
    
    logger.info(f"Found {len(dataset_files)} dataset files for {model}_{correlation}")
    
    # Read and combine all files
    all_data = []
    for file_path in sorted(dataset_files):
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                all_data.append(df)
                dataset_name = os.path.basename(file_path).replace(f"llm_judge_{model}_{correlation}_", "").replace(".csv", "")
                logger.info(f"  Added {len(df)} rows from {dataset_name}")
            else:
                logger.warning(f"  Empty file: {file_path}")
        except Exception as e:
            logger.error(f"  Failed to read {file_path}: {e}")
    
    if not all_data:
        logger.error(f"No valid data found for {model}_{correlation}")
        return False
    
    # Merge all data
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by dataset and measure for consistent output
    merged_df = merged_df.sort_values(['dataset', 'measure'], na_position='last')
    
    # Save merged file
    output_file = os.path.join(output_dir, f"llm_judge_{model}_{correlation}.csv")
    merged_df.to_csv(output_file, index=False)
    
    logger.info(f"Merged {len(merged_df)} total rows into {output_file}")
    
    # Print summary statistics
    logger.info(f"Summary for {model}_{correlation}:")
    logger.info(f"  Total metrics: {len(merged_df)}")
    logger.info(f"  Datasets: {sorted(merged_df['dataset'].unique())}")
    logger.info(f"  Mean correlation: {merged_df['mean_correlation'].mean():.4f}")
    logger.info(f"  Top 3 correlations:")
    
    # Show top correlations
    merged_df['abs_mean'] = merged_df['mean_correlation'].abs()
    top_3 = merged_df.nlargest(3, 'abs_mean')
    for _, row in top_3.iterrows():
        ci_width = (row['ci_upper_correlation'] - row['ci_lower_correlation']) / 2
        logger.info(f"    {row['dataset']}.{row['measure']}: {row['mean_correlation']:.4f} ± {ci_width:.4f}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Merge LLM judge correlation results from individual dataset files"
    )
    
    parser.add_argument(
        "--model",
        default="gpt4o_mini",
        choices=["gpt4o_mini", "qwen3_32b", "llama3_70b"],
        help="LLM model to merge results for"
    )
    parser.add_argument(
        "--correlation",
        default="all",
        help="Correlation function(s): 'pearson', 'spearman', 'kendall', or 'all' for all three"
    )
    parser.add_argument(
        "--sub-results-dir",
        default="results/main_runs/baselines/llm_judge_sub_results",
        help="Directory containing individual dataset result files"
    )
    parser.add_argument(
        "--output-dir",
        default="results/main_runs/baselines",
        help="Directory to save merged result files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Handle correlation types
    if args.correlation.lower() == "all":
        correlation_types = ["kendall", "pearson", "spearman"]
    else:
        correlation_types = [args.correlation.lower()]
    
    # Validate correlation types
    valid_correlations = {"pearson", "spearman", "kendall"}
    for corr_type in correlation_types:
        if corr_type not in valid_correlations:
            logger.error(f"Unknown correlation function: {corr_type}")
            return 1
    
    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.sub_results_dir):
        logger.error(f"Sub-results directory does not exist: {args.sub_results_dir}")
        return 1
    
    logger.info(f"Starting merge for model: {args.model}")
    logger.info(f"Correlation types: {correlation_types}")
    logger.info(f"Input directory: {args.sub_results_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    success_count = 0
    
    # Process each correlation type
    for correlation_type in correlation_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Merging {correlation_type.upper()} results")
        logger.info(f"{'='*50}")
        
        success = merge_dataset_files(
            model=args.model,
            correlation=correlation_type,
            sub_results_dir=args.sub_results_dir,
            output_dir=args.output_dir,
            logger=logger
        )
        
        if success:
            success_count += 1
        else:
            logger.error(f"Failed to merge {correlation_type} results")
    
    if success_count == len(correlation_types):
        logger.info(f"\nSuccessfully merged all {success_count} correlation types!")
        return 0
    else:
        logger.error(f"\nOnly {success_count}/{len(correlation_types)} correlation types merged successfully")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 