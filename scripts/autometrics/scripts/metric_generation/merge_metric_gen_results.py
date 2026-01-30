#!/usr/bin/env python3
#SBATCH --job-name=merge_metric_gen
#SBATCH --output=scripts/metric_generation/logs/merge_metric_gen.out
#SBATCH --error=scripts/metric_generation/logs/merge_metric_gen.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=sphinx

"""
Merge Metric Generation Benchmark Results

This script merges individual dataset-specific CSV files into consolidated
final results files for the metric generation benchmark.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

# Add autometrics to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def merge_results(input_dir: str, output_dir: str, correlation_type: str, logger: logging.Logger):
    """Merge individual CSV files into consolidated results."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Pattern for input files
    pattern = f"metric_generation_benchmark_*_{correlation_type}.csv"
    
    # Find all result files
    result_files = list(input_path.glob(pattern))
    
    if not result_files:
        logger.warning(f"No result files found matching pattern: {pattern}")
        return
    
    logger.info(f"Found {len(result_files)} result files to merge")
    
    # Read and combine all files
    all_results = []
    
    for file_path in result_files:
        logger.info(f"Reading: {file_path}")
        try:
            df = pd.read_csv(file_path)
            all_results.append(df)
            logger.info(f"  Loaded {len(df)} records")
        except Exception as e:
            logger.error(f"  Error reading {file_path}: {e}")
            continue
    
    if not all_results:
        logger.error("No valid result files found")
        return
    
    # Combine all results
    logger.info("Merging results...")
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Remove duplicates (keep the most recent/complete one)
    logger.info(f"Combined dataset has {len(combined_df)} records")
    
    # Sort by key columns for consistent ordering
    sort_columns = ['dataset', 'measure', 'generator_type']
    combined_df = combined_df.sort_values(sort_columns)
    
    # Remove duplicates based on unique combination of dataset, measure, generator_type
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['dataset', 'measure', 'generator_type'], keep='last')
    after_dedup = len(combined_df)
    
    if before_dedup != after_dedup:
        logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
    
    # Create output filename
    output_file = output_path / f"metric_generation_benchmark_final_{correlation_type}.csv"
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save merged results
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Saved merged results to: {output_file}")
    logger.info(f"Final dataset contains {len(combined_df)} records")
    
    # Print summary statistics
    print_summary_stats(combined_df, logger)
    
    return output_file


def print_summary_stats(df: pd.DataFrame, logger: logging.Logger):
    """Print summary statistics for the merged results."""
    
    logger.info("\n" + "="*60)
    logger.info("METRIC GENERATION BENCHMARK SUMMARY")
    logger.info("="*60)
    
    # Dataset coverage
    datasets = df['dataset'].nunique()
    measures = df.groupby('dataset')['measure'].nunique().sum()
    generator_types = df['generator_type'].nunique()
    
    logger.info(f"Coverage:")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Dataset-measure combinations: {measures}")
    logger.info(f"  Generator types: {generator_types}")
    logger.info(f"  Total experiments: {len(df)}")
    
    # Generator performance summary
    logger.info(f"\nGenerator Performance (by mean correlation):")
    generator_stats = df.groupby(['generator_type', 'generator_description'])['mean_correlation'].agg(['mean', 'count']).round(4)
    generator_stats = generator_stats.sort_values('mean', ascending=False)
    
    for (gen_type, description), stats in generator_stats.iterrows():
        logger.info(f"  {description}: {stats['mean']:.4f} (n={stats['count']})")
    
    # Dataset difficulty (by correlation variance)
    logger.info(f"\nDataset Difficulty (by correlation variance):")
    dataset_stats = df.groupby('dataset')['mean_correlation'].agg(['mean', 'std', 'count']).round(4)
    dataset_stats = dataset_stats.sort_values('std', ascending=False)
    
    for dataset, stats in dataset_stats.iterrows():
        logger.info(f"  {dataset}: μ={stats['mean']:.4f}, σ={stats['std']:.4f} (n={stats['count']})")
    
    # Top performing combinations
    logger.info(f"\nTop 10 Generator-Dataset Combinations:")
    top_combinations = df.nlargest(10, 'mean_correlation')[['dataset', 'measure', 'generator_description', 'mean_correlation', 'mean_±_ci']]
    
    for _, row in top_combinations.iterrows():
        logger.info(f"  {row['generator_description']} on {row['dataset']}.{row['measure']}: {row['mean_±_ci']}")
    
    # Success rate analysis
    logger.info(f"\nSuccess Rate Analysis:")
    total_possible = len(df) * 5  # 5 seeds per experiment
    total_successful = df['num_successful_runs'].sum()
    success_rate = total_successful / total_possible * 100
    
    logger.info(f"  Total possible runs: {total_possible}")
    logger.info(f"  Successful runs: {total_successful}")
    logger.info(f"  Overall success rate: {success_rate:.1f}%")
    
    # Generator success rates
    logger.info(f"\nGenerator Success Rates:")
    for gen_type in df['generator_type'].unique():
        subset = df[df['generator_type'] == gen_type]
        possible = len(subset) * 5
        successful = subset['num_successful_runs'].sum()
        rate = successful / possible * 100 if possible > 0 else 0
        logger.info(f"  {subset.iloc[0]['generator_description']}: {rate:.1f}% ({successful}/{possible})")


def main():
    parser = argparse.ArgumentParser(description="Merge metric generation benchmark results")
    
    parser.add_argument(
        "--input-dir",
        default="results/ablations/metric_generation",
        help="Directory containing individual result files"
    )
    parser.add_argument(
        "--output-dir", 
        default="results/ablations/metric_generation",
        help="Directory to save merged results"
    )
    parser.add_argument(
        "--correlation",
        default="kendall",
        choices=["pearson", "spearman", "kendall"],
        help="Correlation type to merge"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    logger.info("Starting Metric Generation Benchmark Results Merge")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Correlation type: {args.correlation}")
    
    # Merge results
    output_file = merge_results(args.input_dir, args.output_dir, args.correlation, logger)
    
    if output_file:
        logger.info(f"\n{'='*60}")
        logger.info("MERGE COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*60}")
        logger.info(f"Merged results saved to: {output_file}")
        logger.info("Ready for analysis and publication!")
    else:
        logger.error("Merge failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 