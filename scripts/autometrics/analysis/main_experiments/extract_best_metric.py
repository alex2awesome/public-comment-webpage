#!/usr/bin/env python3
"""
Script to extract the best performing metric from correlation CSV files.
Loops through outputs/correlation/.../correlation_{measureName}.csv files
and extracts the top-performing metric, saving results to a CSV in results folder.
"""

import argparse
import os
import pandas as pd
from pathlib import Path
import glob
from typing import List, Dict, Any


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract best performing metrics from correlation CSV files"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/correlation",
        help="Input directory containing correlation results (default: outputs/correlation)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="results",
        help="Output directory for results CSV (default: results)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="best_metrics.csv",
        help="Output filename (default: best_metrics.csv)"
    )
    
    parser.add_argument(
        "--correlation-type",
        type=str,
        choices=["kendall", "spearman", "pearson"],
        default="kendall",
        help="Correlation type to analyze (default: kendall)"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        help="Specific datasets to analyze (default: all datasets)"
    )
    
    parser.add_argument(
        "--measures",
        type=str, 
        nargs="*",
        help="Specific measures to analyze (default: all measures)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def find_correlation_files(input_dir: str, correlation_type: str, 
                          datasets: List[str] = None, measures: List[str] = None) -> List[Dict[str, str]]:
    """
    Find all correlation CSV files matching the criteria.
    
    Returns:
        List of dictionaries with keys: 'dataset', 'measure', 'filepath'
    """
    files = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Get all dataset directories
    dataset_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        
        # Filter by specified datasets if provided
        if datasets and dataset_name not in datasets:
            continue
            
        correlation_dir = dataset_dir / correlation_type
        if not correlation_dir.exists():
            continue
            
        # Find all correlation CSV files
        correlation_files = glob.glob(str(correlation_dir / "correlation_*.csv"))
        
        for filepath in correlation_files:
            filename = Path(filepath).name
            # Extract measure name from filename: correlation_{measure}.csv
            if filename.startswith("correlation_") and filename.endswith(".csv"):
                measure_name = filename[12:-4]  # Remove "correlation_" and ".csv"
                
                # Filter by specified measures if provided
                if measures and measure_name not in measures:
                    continue
                    
                files.append({
                    'dataset': dataset_name,
                    'measure': measure_name,
                    'filepath': filepath
                })
    
    return files


def extract_best_metric(filepath: str) -> Dict[str, Any]:
    """
    Extract the best performing metric from a correlation CSV file.
    
    Returns:
        Dictionary with metric information
    """
    try:
        df = pd.read_csv(filepath)
        
        if df.empty:
            return None
            
        # Get the first row (should be the best performing metric)
        best_row = df.iloc[0]
        
        return {
            'metric': best_row['Metric'],
            'metric_class': best_row['Metric_Class'],
            'correlation': best_row['Correlation'],
            'p_value': best_row['P-value']
        }
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def main():
    """Main function."""
    args = parse_args()
    
    if args.verbose:
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Output file: {args.output_file}")
        print(f"Correlation type: {args.correlation_type}")
        print(f"Datasets: {args.datasets if args.datasets else 'All'}")
        print(f"Measures: {args.measures if args.measures else 'All'}")
        print()
    
    # Find all correlation files
    correlation_files = find_correlation_files(
        args.input_dir, 
        args.correlation_type,
        args.datasets,
        args.measures
    )
    
    if not correlation_files:
        print("No correlation files found matching the criteria.")
        return
    
    if args.verbose:
        print(f"Found {len(correlation_files)} correlation files to process")
    
    # Extract best metrics from each file
    results = []
    
    for file_info in correlation_files:
        if args.verbose:
            print(f"Processing {file_info['dataset']}/{file_info['measure']}")
            
        best_metric = extract_best_metric(file_info['filepath'])
        
        if best_metric:
            results.append({
                'dataset': file_info['dataset'],
                'measure': file_info['measure'],
                'metric': best_metric['metric'],
                'metric_class': best_metric['metric_class'],
                f'{args.correlation_type}_correlation': best_metric['correlation'],
                'p_value': best_metric['p_value']
            })
    
    if not results:
        print("No valid results found.")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by dataset and measure for consistent output
    results_df = results_df.sort_values(['dataset', 'measure'])
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    output_path = output_dir / args.output_file
    results_df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")
    print(f"Processed {len(results)} dataset-measure combinations")
    
    if args.verbose:
        print("\nSample of results:")
        print(results_df.head())


if __name__ == "__main__":
    main()

# Example usage:
# python analysis/main_experiments/extract_best_metric.py