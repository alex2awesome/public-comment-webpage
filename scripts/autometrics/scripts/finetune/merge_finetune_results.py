#!/usr/bin/env python3
"""
Merge Fine-tuned Metric Correlation Results

This script merges individual dataset-specific CSV files from the sub_results directory
into final consolidated files for each correlation type.

Usage:
    python scripts/finetune/merge_finetune_results.py --correlation all
    python scripts/finetune/merge_finetune_results.py --correlation kendall
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

def merge_dataset_files(correlation: str, sub_results_dir: str, output_dir: str, logger: logging.Logger) -> bool:
    """Merge all dataset-specific files for a given correlation type."""
    
    # Find all matching dataset files
    pattern = f"finetune_{correlation}_*.csv"
    search_path = os.path.join(sub_results_dir, pattern)
    dataset_files = glob.glob(search_path)
    
    if not dataset_files:
        logger.warning(f"No dataset files found matching pattern: {pattern}")
        return False
    
    logger.info(f"Found {len(dataset_files)} dataset files for {correlation}")
    
    # Read and combine all files
    all_data = []
    for file_path in sorted(dataset_files):
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                all_data.append(df)
                dataset_name = os.path.basename(file_path).replace(f"finetune_{correlation}_", "").replace(".csv", "")
                logger.info(f"  Added {len(df)} rows from {dataset_name}")
            else:
                logger.warning(f"  Empty file: {file_path}")
        except Exception as e:
            logger.error(f"  Failed to read {file_path}: {e}")
    
    if not all_data:
        logger.error(f"No valid data found for {correlation}")
        return False
    
    # Merge all data
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by dataset and measure for consistent output
    merged_df = merged_df.sort_values(['dataset', 'measure'], na_position='last')
    
    # Save merged file
    output_file = os.path.join(output_dir, f"finetune_{correlation}.csv")
    merged_df.to_csv(output_file, index=False)
    
    logger.info(f"Merged {len(merged_df)} total rows into {output_file}")
    
    # Print summary statistics
    logger.info(f"Summary for {correlation}:")
    logger.info(f"  Total metrics: {len(merged_df)}")
    if 'mean_correlation' in merged_df.columns:
        mean_abs_corr = merged_df['mean_correlation'].abs().mean()
        logger.info(f"  Mean |correlation|: {mean_abs_corr:.4f}")
    if 'num_successful_runs' in merged_df.columns:
        successful_runs = merged_df['num_successful_runs'].sum()
        total_possible = len(merged_df) * 5  # Assuming 5 seeds
        logger.info(f"  Successful runs: {successful_runs}/{total_possible}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Merge fine-tuned metric correlation results from individual dataset files"
    )
    
    parser.add_argument(
        "--correlation",
        default="all",
        help="Correlation function(s): 'pearson', 'spearman', 'kendall', or 'all' for all three"
    )
    parser.add_argument(
        "--sub-results-dir",
        default="results/main_runs/baselines/finetune_sub_results",
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
    
    logger.info(f"Starting merge for correlation types: {correlation_types}")
    logger.info(f"Input directory: {args.sub_results_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Merge results for each correlation type
    success_count = 0
    for correlation_type in correlation_types:
        logger.info(f"\n--- Merging {correlation_type} results ---")
        
        if merge_dataset_files(correlation_type, args.sub_results_dir, args.output_dir, logger):
            success_count += 1
            logger.info(f"✓ Successfully merged {correlation_type} results")
        else:
            logger.error(f"✗ Failed to merge {correlation_type} results")
    
    # Final summary
    logger.info(f"\n=== MERGE SUMMARY ===")
    logger.info(f"Successfully merged: {success_count}/{len(correlation_types)} correlation types")
    
    if success_count > 0:
        logger.info(f"Merged files saved to:")
        for corr_type in correlation_types:
            output_file = os.path.join(args.output_dir, f"finetune_{corr_type}.csv")
            if os.path.exists(output_file):
                logger.info(f"  {output_file}")
        
        logger.info(f"\nIndividual dataset files available in:")
        logger.info(f"  {args.sub_results_dir}")
        
        return 0
    else:
        logger.error("No correlation types were successfully merged")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 