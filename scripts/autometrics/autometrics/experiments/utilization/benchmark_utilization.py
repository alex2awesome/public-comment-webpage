#!/usr/bin/env python3
"""
Benchmark Utilizer Script - Run utilization benchmarks for all metrics in the MetricBank

This script automatically runs utilization benchmarks for all metrics in the
MetricBank, saving results as it goes and allowing for interrupted runs to
be resumed without repeating work.
"""

import os
import sys
import time
import argparse
import logging
import importlib
import pandas as pd
import glob
import platform
import json
import traceback
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from autometrics.metrics.MetricBank import (
    build_reference_based_metrics,
    build_reference_free_metrics,
)
from autometrics.experiments.utilization.utilization import UtilizationExperiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_utilizer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Default benchmark settings
DEFAULT_OUTPUT_DIR = "outputs/utilization"
DEFAULT_NUM_EXAMPLES = 50
DEFAULT_BURN_IN = 5
DEFAULT_LENGTHS = ["short", "medium", "long"]
DEFAULT_USE_SYNTHETIC = True

# Custom JSON encoder to handle numpy dtypes and other special types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle numpy dtype
        if hasattr(obj, 'dtype'):
            return str(obj.dtype)
        # Handle any other types with a __str__ method
        try:
            return str(obj)
        except:
            pass
        
        return super().default(obj)

def safe_json_dump(obj, file_path):
    """Safely dump object to JSON file with custom encoder."""
    try:
        with open(file_path, 'w') as f:
            json.dump(obj, f, cls=CustomJSONEncoder, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error serializing to JSON: {str(e)}")
        return False

def safe_json_dumps(obj):
    """Safely convert object to JSON string with custom encoder."""
    try:
        return json.dumps(obj, cls=CustomJSONEncoder)
    except Exception as e:
        logger.error(f"Error serializing to JSON: {str(e)}")
        # Create a simplified version with only strings
        simplified = {}
        for k, v in obj.items() if isinstance(obj, dict) else []:
            try:
                simplified[str(k)] = str(v)
            except:
                simplified[str(k)] = "UNSERIALIZABLE"
        return json.dumps(simplified)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run utilization benchmarks for all metrics in the MetricBank"
    )
    
    parser.add_argument(
        "--output-dir", 
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store outputs (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--num-examples", 
        type=int, 
        default=DEFAULT_NUM_EXAMPLES,
        help=f"Number of examples to test per length category (default: {DEFAULT_NUM_EXAMPLES})"
    )
    
    parser.add_argument(
        "--burn-in", 
        type=int, 
        default=DEFAULT_BURN_IN,
        help=f"Number of burn-in samples to run (default: {DEFAULT_BURN_IN})"
    )
    
    parser.add_argument(
        "--lengths", 
        default=",".join(DEFAULT_LENGTHS),
        help=f"Comma-separated list of length categories (default: {','.join(DEFAULT_LENGTHS)})"
    )
    
    parser.add_argument(
        "--skip-reference-based",
        action="store_true",
        help="Skip reference-based metrics"
    )
    
    parser.add_argument(
        "--skip-reference-free",
        action="store_true",
        help="Skip reference-free metrics"
    )
    
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-run of already completed metrics"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--clean-partial-results",
        action="store_true",
        help="Clean partial results before running metrics (removes incomplete metric folders)"
    )
    
    # Cache directory for metric artifacts (mirrors benchmark_correlation.py)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom diskcache directory for metric caching (overrides Metric default)"
    )
    
    return parser.parse_args()

def metric_has_complete_results(metric_name: str, output_dir: str, lengths: List[str]) -> bool:
    """
    Check if a metric has complete utilization results.
    
    Args:
        metric_name: Name of the metric
        output_dir: Base output directory
        lengths: List of length categories to check
        
    Returns:
        True if the metric has complete results, False otherwise
    """
    # For metrics used with synthetic data, we need to check if results exist for each length
    metric_dir = os.path.join(output_dir, "synthetic", metric_name)
    
    # If the metric directory doesn't exist at all, clearly incomplete
    if not os.path.exists(metric_dir):
        return False
        
    # Check for the summary.csv file in each length category directory
    for length in lengths:
        summary_path = os.path.join(metric_dir, length, "summary.csv")
        if not os.path.exists(summary_path):
            return False
    
    # Also check if raw_data.csv exists for each length (ensures complete run)
    for length in lengths:
        raw_data_path = os.path.join(metric_dir, length, "raw_data.csv")
        if not os.path.exists(raw_data_path):
            return False
            
    return True

def clean_partial_metric_results(metric_name: str, output_dir: str):
    """
    Clean partial results for a metric.
    
    Args:
        metric_name: Name of the metric
        output_dir: Base output directory
    """
    metric_dir = os.path.join(output_dir, "synthetic", metric_name)
    
    if os.path.exists(metric_dir):
        print(f"Cleaning partial results for {metric_name}...")
        shutil.rmtree(metric_dir)
        print(f"Removed directory: {metric_dir}")

def safely_import_metric(metric_class_path: str) -> Optional[Any]:
    """
    Safely import a metric class without crashing if dependencies are missing.
    
    Args:
        metric_class_path: Fully qualified path to the metric class (e.g., 'autometrics.metrics.reference_based.BLEU.BLEU')
        
    Returns:
        The metric instance or None if import failed
    """
    try:
        module_path, class_name = metric_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        MetricClass = getattr(module, class_name)
        return MetricClass()
    except (ImportError, AttributeError, Exception) as e:
        logger.warning(f"Could not import {metric_class_path}: {str(e)}")
        return None

def get_metric_class_path(metric) -> str:
    """Get the fully qualified class path for a metric instance."""
    return f"{metric.__class__.__module__}.{metric.__class__.__name__}"

def detect_hardware_info() -> Dict[str, Any]:
    """
    Detect hardware information, especially GPU details.
    
    Returns:
        Dictionary containing hardware information
    """
    hw_info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "timestamp": datetime.now().isoformat(),
        "gpus": []
    }
    
    # Try to get CUDA/GPU information if available
    try:
        import torch
        if torch.cuda.is_available():
            hw_info["cuda_version"] = torch.version.cuda
            hw_info["gpu_count"] = torch.cuda.device_count()
            hw_info["gpus"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3)  # in GB
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        hw_info["gpu_info_source"] = "torch_not_available"
        
        # Try using pynvml if torch is not available
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            hw_info["gpu_count"] = device_count
            hw_info["gpu_info_source"] = "pynvml"
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                
                hw_info["gpus"].append({
                    "index": i,
                    "name": name.decode('utf-8') if isinstance(name, bytes) else name,
                    "memory_total": info.total / (1024**3)  # in GB
                })
            
            pynvml.nvmlShutdown()
        except (ImportError, Exception):
            hw_info["gpu_info_source"] = "none_available"
    
    return hw_info

def aggregate_results(output_dir: str, hw_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Aggregate results from all metrics into a single DataFrame.
    
    Args:
        output_dir: Base output directory
        hw_info: Hardware information dictionary
        
    Returns:
        DataFrame containing aggregated results
    """
    all_summaries = []
    
    # Find all summary.csv files in the output directory
    synthetic_dir = os.path.join(output_dir, "synthetic")
    pattern = os.path.join(synthetic_dir, "*", "*", "summary.csv")
    summary_files = glob.glob(pattern)
    
    print(f"Found {len(summary_files)} summary files to aggregate")
    
    for file_path in summary_files:
        # Extract metric name and length category from path
        # Path format: {output_dir}/synthetic/{metric_name}/{length}/summary.csv
        parts = file_path.split(os.sep)
        metric_name = parts[-3]
        length = parts[-2]
        
        print(f"Processing summary for {metric_name} ({length})")
        
        try:
            df = pd.read_csv(file_path)
            # Add columns for metric name and length
            df['metric'] = metric_name
            df['length'] = length
            
            # Add hardware info columns
            df['system'] = hw_info.get('system', 'unknown')
            df['cpu_count'] = hw_info.get('cpu_count', 0)
            df['gpu_count'] = hw_info.get('gpu_count', 0)
            
            # Add GPU model if available (take the first GPU)
            if hw_info.get('gpus') and len(hw_info['gpus']) > 0:
                df['gpu_model'] = hw_info['gpus'][0].get('name', 'unknown')
            else:
                df['gpu_model'] = 'none'
                
            all_summaries.append(df)
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {str(e)}")
            print(f"ERROR: Could not read {file_path}: {str(e)}")
    
    if not all_summaries:
        logger.warning("No summary files found to aggregate")
        print("WARNING: No summary files found to aggregate")
        return pd.DataFrame()
        
    # Combine all summaries
    combined_df = pd.concat(all_summaries, ignore_index=True)
    
    # Reorder columns to put metric and length first
    cols = ['metric', 'length', 'system', 'cpu_count', 'gpu_count', 'gpu_model'] + [
        col for col in combined_df.columns if col not in [
            'metric', 'length', 'system', 'cpu_count', 'gpu_count', 'gpu_model'
        ]
    ]
    combined_df = combined_df[cols]
    
    return combined_df

def log_error_for_metric(metric_name: str, error: Exception, traceback_str: str, output_dir: str):
    """
    Log detailed error information for a failed metric.
    
    Args:
        metric_name: Name of the metric that failed
        error: Exception that was raised
        traceback_str: Traceback as a string
        output_dir: Base output directory
    """
    # Create errors directory if it doesn't exist
    errors_dir = os.path.join(output_dir, "errors")
    os.makedirs(errors_dir, exist_ok=True)
    
    # Create an error log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_file = os.path.join(errors_dir, f"{metric_name}_error_{timestamp}.log")
    
    # Write the error information to the file
    with open(error_file, 'w') as f:
        f.write(f"ERROR IN METRIC: {metric_name}\n")
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write(f"ERROR TYPE: {type(error).__name__}\n")
        f.write(f"ERROR MESSAGE: {str(error)}\n")
        f.write("\nTRACEBACK:\n")
        f.write(traceback_str)
    
    # Also maintain a summary file of all errors
    summary_file = os.path.join(errors_dir, "failed_metrics.csv")
    
    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.isfile(summary_file)
    
    # Append to the summary file
    with open(summary_file, 'a') as f:
        if not file_exists:
            f.write("metric,timestamp,error_type,error_message,error_log_file\n")
        
        # Escape any commas in the error message
        error_msg = str(error).replace(',', ';')
        f.write(f"{metric_name},{timestamp},{type(error).__name__},{error_msg},{error_file}\n")

def patch_utils():
    """
    Patch the utilization.py and isolated_runner.py to handle serialization issues.
    This makes runtime modifications to avoid changing the core code files.
    """
    try:
        # Patch the isolated_runner module if it's available
        from autometrics.experiments.utilization import isolated_runner
        
        # Store the original json.dumps function
        original_dumps = json.dumps
        
        # Replace with our safe version
        def safe_dumps(obj, *args, **kwargs):
            try:
                return original_dumps(obj, *args, **kwargs, cls=CustomJSONEncoder)
            except Exception as e:
                print(f"Error serializing to JSON: {str(e)}")
                # Create a simplified version with only strings
                simplified = {}
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        try:
                            simplified[str(k)] = str(v)
                        except:
                            simplified[str(k)] = "UNSERIALIZABLE"
                
                return original_dumps(simplified, *args, **kwargs)
        
        # Replace json.dumps in the isolated_runner module
        isolated_runner.json.dumps = safe_dumps
        
        print("Successfully patched isolated_runner.py for better JSON serialization")
    except ImportError:
        print("Could not patch isolated_runner.py - module not found")
    except Exception as e:
        print(f"Error patching modules: {str(e)}")

def run_benchmark_for_metric(metric, args):
    """
    Run the utilization benchmark for a single metric.
    
    Args:
        metric: The metric instance to benchmark
        args: Command-line arguments
        
    Returns:
        True if successful, False if failed
    """
    metric_name = metric.get_name()
    print(f"\n{'='*80}")
    print(f"STARTING BENCHMARK FOR: {metric_name}")
    print(f"{'='*80}")
    logger.info(f"Starting benchmark for {metric_name}")
    
    # Clean partial results if requested
    if args.clean_partial_results:
        clean_partial_metric_results(metric_name, args.output_dir)
    
    # Get constructor parameters for debugging
    constructor_params = {}
    if hasattr(metric, '_init_params'):
        constructor_params = metric._init_params.copy()
    
    try:
        # Try to pre-validate metric parameters
        filtered_params = {}
        if hasattr(metric, '_init_params'):
            import inspect
            if hasattr(metric.__class__, '__init__'):
                sig = inspect.signature(metric.__class__.__init__)
                allowed_params = set(sig.parameters.keys()) - {'self'}
                
                # Filter params to only include those the constructor accepts
                for k, v in metric._init_params.items():
                    if k in allowed_params:
                        filtered_params[k] = v
                
                print(f"Filtered constructor params for {metric_name}: {filtered_params}")
    except Exception as e:
        print(f"Error during pre-validation of parameters: {str(e)}")
    
    try:
        # Create the experiment
        print(f"Configuring experiment for {metric_name}...")
        experiment = UtilizationExperiment(
            name=f"{metric_name} Utilization Benchmark",
            description=f"Resource utilization benchmark for {metric_name}",
            metrics=[metric],
            output_dir=args.output_dir,
            num_examples=args.num_examples,
            num_burn_in=args.burn_in,
            lengths=args.lengths.split(','),
            use_synthetic=DEFAULT_USE_SYNTHETIC,
            seed=args.seed,
            measure_import_costs=True,
            use_isolated_trials=True,
            use_deterministic_examples=True
        )
        
        # Run the experiment
        print(f"Running benchmark for {metric_name}...")
        print(f"Testing with {args.num_examples} examples per length category: {args.lengths}")
        print(f"Using {args.burn_in} burn-in samples")
        
        # Monitor for errors in the experiment
        experiment_error = None
        experiment_traceback = None
        
        try:
            experiment.run(print_results=True)  # Always print results for each metric
        except Exception as e:
            experiment_error = e
            experiment_traceback = traceback.format_exc()
            print(f"Error during experiment run: {str(e)}")
            
            # Continue with saving whatever results were generated
            print("Attempting to save partial results...")
        
        # Save the results
        print(f"Saving results for {metric_name}...")
        try:
            experiment.save_results()
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            if experiment_error is None:  # Only update if we don't already have an error
                experiment_error = e
                experiment_traceback = traceback.format_exc()
        
        # If there was an error during the experiment, report it now
        if experiment_error is not None:
            print(f"\n{'!'*80}")
            print(f"BENCHMARK ENCOUNTERED ERRORS FOR: {metric_name}")
            print(f"Error: {str(experiment_error)}")
            print(f"See error log for details")
            print(f"{'!'*80}\n")
            
            # Log detailed error information to a separate file
            log_error_for_metric(metric_name, experiment_error, experiment_traceback, args.output_dir)
            return False
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPLETED SUCCESSFULLY FOR: {metric_name}")
        print(f"{'='*80}\n")
        logger.info(f"Benchmark completed for {metric_name}")
        return True
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        
        # Log the error to both console and file
        error_msg = f"ERROR benchmarking {metric_name}: {str(e)}"
        logger.error(error_msg)
        print(f"\n{'!'*80}")
        print(f"BENCHMARK FAILED FOR: {metric_name}")
        print(f"Error: {str(e)}")
        print(f"Constructor params: {constructor_params}")
        print(f"See error log for details")
        print(f"{'!'*80}\n")
        
        # Log detailed error information to a separate file
        log_error_for_metric(metric_name, e, traceback_str, args.output_dir)
        
        return False

def main():
    """Main function to run the benchmarks."""
    args = parse_args()
    
    # Apply runtime patches to handle serialization issues
    patch_utils()
    
    # Start with hardware detection
    print("\nDetecting hardware information...")
    hw_info = detect_hardware_info()
    
    # Print hardware information
    print("\nHARDWARE INFORMATION:")
    print(f"System: {hw_info.get('system', 'unknown')}")
    print(f"Processor: {hw_info.get('processor', 'unknown')}")
    print(f"CPU Count: {hw_info.get('cpu_count', 'unknown')}")
    
    if hw_info.get('gpus'):
        print(f"GPU Count: {len(hw_info['gpus'])}")
        for i, gpu in enumerate(hw_info['gpus']):
            print(f"  GPU {i}: {gpu.get('name', 'unknown')} "
                  f"({gpu.get('memory_total', 0):.2f} GB)")
    else:
        print("No GPUs detected")
    
    # Save hardware info to a file
    hw_info_path = os.path.join(args.output_dir, "hardware_info.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    safe_json_dump(hw_info, hw_info_path)
    
    print(f"Hardware information saved to {hw_info_path}")
    
    # Log the start of the benchmarking process
    logger.info("Starting metric utilization benchmarking")
    print("\nSTARTING METRIC UTILIZATION BENCHMARKING")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of examples: {args.num_examples}")
    print(f"Burn-in samples: {args.burn_in}")
    print(f"Length categories: {args.lengths}")
    
    # Create errors directory for potential failures
    errors_dir = os.path.join(args.output_dir, "errors")
    os.makedirs(errors_dir, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Collect metrics to benchmark using the new builder helpers
    # ------------------------------------------------------------------
    metrics_to_benchmark: List[Any] = []

    common_factory_kwargs = {
        "cache_dir": args.cache_dir,
        "seed": args.seed,
    }

    if not args.skip_reference_based:
        ref_based_metrics = build_reference_based_metrics(**common_factory_kwargs)
        metrics_to_benchmark.extend(ref_based_metrics)
        logger.info(f"Including {len(ref_based_metrics)} reference-based metrics")
        print(f"Including {len(ref_based_metrics)} reference-based metrics:")
        for m in ref_based_metrics:
            print(f"  - {m.get_name()}")

    if not args.skip_reference_free:
        ref_free_metrics = build_reference_free_metrics(**common_factory_kwargs)
        metrics_to_benchmark.extend(ref_free_metrics)
        logger.info(f"Including {len(ref_free_metrics)} reference-free metrics")
        print(f"Including {len(ref_free_metrics)} reference-free metrics:")
        for m in ref_free_metrics:
            print(f"  - {m.get_name()}")
    
    print(f"\nTotal metrics to process: {len(metrics_to_benchmark)}")
    logger.info(f"Total metrics to process: {len(metrics_to_benchmark)}")
    
    # Initialize counters
    total_metrics = len(metrics_to_benchmark)
    completed_metrics = 0
    skipped_metrics = 0
    failed_metrics = 0
    
    # Track metrics for the final report
    results_tracker = {
        "completed": [],
        "skipped": [],
        "failed": []
    }
    
    # Process each metric
    for i, metric in enumerate(metrics_to_benchmark, 1):
        metric_name = metric.get_name()
        print(f"\nProcessing metric {i}/{total_metrics}: {metric_name}")
        logger.info(f"Processing metric {i}/{total_metrics}: {metric_name}")
        
        # Check if this metric already has complete results
        if not args.force_rerun and metric_has_complete_results(
            metric_name, args.output_dir, args.lengths.split(',')
        ):
            print(f"Skipping {metric_name} - results already exist")
            logger.info(f"Skipping {metric_name} - results already exist")
            skipped_metrics += 1
            results_tracker["skipped"].append(metric_name)
            continue
        
        # Run the benchmark
        success = run_benchmark_for_metric(metric, args)
        
        if success:
            completed_metrics += 1
            results_tracker["completed"].append(metric_name)
        else:
            failed_metrics += 1
            results_tracker["failed"].append(metric_name)
        
        # Log progress
        progress_msg = f"Progress: {i}/{total_metrics} metrics processed"
        status_msg = f"Status: {completed_metrics} completed, {skipped_metrics} skipped, {failed_metrics} failed"
        
        print("\n" + "-" * 50)
        print(progress_msg)
        print(status_msg)
        print("-" * 50 + "\n")
        
        logger.info(progress_msg)
        logger.info(status_msg)
    
    # Aggregate results
    print("\nAggregating results from all metrics...")
    logger.info("Aggregating results from all metrics")
    combined_results = aggregate_results(args.output_dir, hw_info)
    
    # Save the aggregated results
    if not combined_results.empty:
        aggregate_path = os.path.join(args.output_dir, "aggregated_results.csv")
        combined_results.to_csv(aggregate_path, index=False)
        print(f"Aggregated results saved to {aggregate_path}")
        logger.info(f"Aggregated results saved to {aggregate_path}")
    else:
        print("WARNING: No results to aggregate")
        logger.warning("No results to aggregate")
    
    # Save the benchmark summary
    summary = {
        "total_metrics": total_metrics,
        "completed_metrics": completed_metrics,
        "skipped_metrics": skipped_metrics,
        "failed_metrics": failed_metrics,
        "completed_list": results_tracker["completed"],
        "skipped_list": results_tracker["skipped"],
        "failed_list": results_tracker["failed"]
    }
    
    summary_df = pd.DataFrame({
        "Category": ["Total Metrics", "Completed", "Skipped", "Failed"],
        "Count": [total_metrics, completed_metrics, skipped_metrics, failed_metrics]
    })
    
    summary_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Save detailed lists
    detailed_summary = pd.DataFrame({
        "Metric": (
            [f"COMPLETED: {m}" for m in results_tracker["completed"]] +
            [f"SKIPPED: {m}" for m in results_tracker["skipped"]] +
            [f"FAILED: {m}" for m in results_tracker["failed"]]
        )
    })
    detailed_summary_path = os.path.join(args.output_dir, "benchmark_detailed_summary.csv")
    detailed_summary.to_csv(detailed_summary_path, index=False)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Total metrics: {total_metrics}")
    print(f"Completed metrics: {completed_metrics}")
    print(f"Skipped metrics (already had results): {skipped_metrics}")
    print(f"Failed metrics: {failed_metrics}")
    
    if failed_metrics > 0:
        print("\nFailed metrics:")
        for metric in results_tracker["failed"]:
            print(f"  - {metric}")
        print(f"\nSee error logs in {errors_dir} for details")
    
    print(f"\nAggregated results saved to {os.path.join(args.output_dir, 'aggregated_results.csv')}")
    print(f"Benchmark summary saved to {os.path.join(args.output_dir, 'benchmark_summary.csv')}")
    
    logger.info("\nBENCHMARK COMPLETE")
    logger.info(f"Total metrics: {total_metrics}")
    logger.info(f"Completed metrics: {completed_metrics}")
    logger.info(f"Skipped metrics: {skipped_metrics}")
    logger.info(f"Failed metrics: {failed_metrics}")
    
    return 0

if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    duration_hours = duration_minutes / 60
    
    print(f"\nTotal runtime: {duration_minutes:.2f} minutes ({duration_hours:.2f} hours)")
    logger.info(f"Total runtime: {duration_minutes:.2f} minutes ({duration_hours:.2f} hours)")
    
    sys.exit(exit_code)