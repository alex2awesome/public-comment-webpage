#!/usr/bin/env python3
"""
Command-line utility to run the utilization experiment.

This script provides a convenient interface to run the utilization experiment 
with various configuration options.
"""

import os
import sys
import argparse
from typing import List
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from autometrics.experiments.utilization.utilization import UtilizationExperiment


def load_metric(metric_name: str):
    """Dynamically load a metric from the metrics package based on its name."""
    if metric_name == "BLEU":
        from autometrics.metrics.reference_based.BLEU import BLEU
        return BLEU()
    elif metric_name == "ROUGE":
        from autometrics.metrics.reference_based.ROUGE import ROUGE
        return ROUGE()
    elif metric_name == "BERTScore":
        from autometrics.metrics.reference_based.BERTScore import BERTScore
        return BERTScore()
    elif metric_name == "SARI":
        from autometrics.metrics.reference_based.SARI import SARI
        return SARI()
    elif metric_name == "LENS":
        from autometrics.metrics.reference_based.LENS import LENS
        return LENS()
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def load_dataset(dataset_name: str):
    """Dynamically load a dataset based on its name."""
    if dataset_name == "SimpDA":
        from autometrics.dataset.datasets.simplification.simplification import SimpDA
        return SimpDA()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run utilization experiments to measure resource usage of metrics"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="outputs/utilization",
        help="Directory to store outputs (default: outputs/utilization)"
    )
    
    parser.add_argument(
        "--num-examples", 
        type=int, 
        default=30,
        help="Number of examples to test per length category (default: 30)"
    )
    
    parser.add_argument(
        "--burn-in", 
        type=int, 
        default=5,
        help="Number of burn-in samples to run (default: 5)"
    )
    
    parser.add_argument(
        "--metrics", 
        default="BLEU,ROUGE,BERTScore",
        help="Comma-separated list of metric names to test (default: BLEU,ROUGE,BERTScore)"
    )
    
    parser.add_argument(
        "--lengths", 
        default="short,medium,long",
        help="Comma-separated list of length categories (default: short,medium,long)"
    )
    
    # Create mutually exclusive group for data source
    data_source = parser.add_mutually_exclusive_group(required=False)
    
    data_source.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real dataset"
    )
    
    data_source.add_argument(
        "--dataset",
        default="SimpDA",
        help="Dataset to use if not using synthetic data (default: SimpDA)"
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
        "--measure-import-costs",
        action="store_true",
        help="Measure import and construction costs in a clean process (default: False)"
    )
    
    parser.add_argument(
        "--skip-import-costs",
        action="store_true",
        help="Skip measuring import and construction costs"
    )
    
    parser.add_argument(
        "--use-isolated-trials",
        action="store_true",
        help="Run each length category in a separate process for clean memory measurements (default: True)"
    )
    
    parser.add_argument(
        "--no-isolated-trials",
        action="store_true",
        help="Disable isolated trials (not recommended as it may lead to memory accumulation between trials)"
    )
    
    parser.add_argument(
        "--deterministic-examples",
        action="store_true",
        help="Use deterministic text examples for more consistent memory measurements"
    )
    
    return parser.parse_args()


def main():
    """Run the utilization experiment with command-line arguments."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Parse parameters
    output_dir = args.output_dir
    num_examples = args.num_examples
    num_burn_in = args.burn_in
    metric_names = args.metrics.split(',')
    lengths = args.lengths.split(',')
    use_synthetic = args.synthetic
    dataset_name = args.dataset
    seed = args.seed
    
    # Load metrics
    metrics = []
    for metric_name in metric_names:
        try:
            metric = load_metric(metric_name)
            metrics.append(metric)
            logging.info(f"Loaded metric: {metric_name}")
        except (ValueError, ImportError) as e:
            logging.warning(f"Could not load metric {metric_name}: {e}")
    
    if not metrics:
        logging.error("No valid metrics specified")
        return 1
    
    # Load dataset if not using synthetic data
    dataset = None
    if not use_synthetic:
        try:
            dataset = load_dataset(dataset_name)
            logging.info(f"Using dataset: {dataset_name}")
        except (ValueError, ImportError) as e:
            logging.warning(f"Could not load dataset {dataset_name}: {e}")
            logging.warning("Falling back to synthetic data")
            use_synthetic = True
    
    # Create experiment directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle import cost measurement setting
    # Default is to measure import costs unless explicitly skipped
    measure_import_costs = not args.skip_import_costs
    if args.measure_import_costs:
        measure_import_costs = True
    
    # Use isolated trials if specified
    use_isolated_trials = not args.no_isolated_trials  # Default to True unless explicitly disabled
    
    # Use deterministic examples if specified
    use_deterministic_examples = args.deterministic_examples
    
    # Create and run the experiment
    experiment = UtilizationExperiment(
        name="Metric Utilization Benchmark",
        description=f"Resource utilization benchmark for {', '.join(m.get_name() for m in metrics)}",
        metrics=metrics,
        output_dir=output_dir,
        dataset=dataset,
        num_examples=num_examples,
        num_burn_in=num_burn_in,
        lengths=lengths,
        use_synthetic=use_synthetic,
        seed=seed,
        measure_import_costs=measure_import_costs,
        use_isolated_trials=use_isolated_trials,
        use_deterministic_examples=use_deterministic_examples
    )
    
    print(f"Starting utilization experiment with {len(metrics)} metrics")
    print(f"Testing {num_examples} examples {'per length category' if use_synthetic else 'from dataset'}")
    print(f"With {num_burn_in} burn-in samples")
    print(f"Using {'synthetic' if use_synthetic else 'real'} data")
    print(f"Output directory: {output_dir}")
    
    experiment.run(print_results=True)
    experiment.save_results()
    
    print(f"Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 