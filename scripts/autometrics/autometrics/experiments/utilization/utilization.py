#!/usr/bin/env python3
# File: utilization/utilization.py

import os
import time
import statistics
import psutil
import random
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import traceback
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pynvml
    HAS_PYNVML = True
    pynvml.nvmlInit()
except (ImportError, pynvml.NVMLError):
    HAS_PYNVML = False

try:
    import nltk
    nltk.download('words')
    from nltk.corpus import words as nltk_words
    NLTK_WORDS = set(w.lower() for w in nltk_words.words() if 3 <= len(w) <= 10 and w.isalpha())
    if not NLTK_WORDS:  # Fallback if NLTK data not downloaded
        raise ImportError("NLTK words not available")
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

from autometrics.experiments.experiment import Experiment
from autometrics.experiments.results import TabularResult, JSONResult, FigureResult, TextResult
from autometrics.metrics.Metric import Metric


class ResourceTracker:
    """Tracks and records resource usage during metric execution."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.end_time = None
        self.start_memory_usage = 0
        self.peak_memory_usage = 0
        self.start_gpu_memory = 0
        self.peak_gpu_memory = 0
        self.disk_before = None
        self.disk_after = None
        self.device_count = 0
        
        # Use the resource utilities for consistent measurements
        from autometrics.experiments.utilization.resource import snap
        self.snap_func = snap
        
        if HAS_TORCH and torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
            torch.cuda.reset_peak_memory_stats()
        
        if HAS_PYNVML:
            try:
                self.device_count = pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError:
                pass
    
    def start(self):
        """Start tracking resources."""
        # Force garbage collection for more consistent measurements
        import gc
        gc.collect()
        
        # Capture full state using our snapshot utility
        self.start_snapshot = self.snap_func("start_tracking")
        
        # Extract values for backwards compatibility
        self.start_memory_usage = self.start_snapshot["cpu_ram_mb"]
        self.peak_memory_usage = self.start_memory_usage
        self.start_gpu_memory = self.start_snapshot["gpu_ram_mb"]
        self.peak_gpu_memory = self.start_gpu_memory
        self.disk_before = self.start_snapshot["disk_used_mb"] * 1024 * 1024  # Convert back to bytes for compatibility
        self.start_time = time.time()
        
        return self
    
    def stop(self):
        """Stop tracking resources and record final metrics."""
        self.end_time = time.time()
        # Sleep a tiny bit to ensure allocations are registered
        time.sleep(0.01)
        
        # Take end snapshot
        self.end_snapshot = self.snap_func("end_tracking")
        
        # Extract values for backwards compatibility
        self.peak_memory_usage = self.end_snapshot["cpu_ram_mb"]
        self.peak_gpu_memory = self.end_snapshot["gpu_ram_mb"]
        self.disk_after = self.end_snapshot["disk_used_mb"] * 1024 * 1024  # Convert back to bytes
        
        return self
    
    def get_results(self):
        """Return the recorded metrics as a dictionary."""
        # Calculate deltas directly to match original implementation behavior
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        duration_ms = duration * 1000
        
        # Calculate incremental RAM usage
        incremental_ram = max(0, self.peak_memory_usage - self.start_memory_usage)
        
        # Calculate incremental GPU memory usage
        incremental_gpu = max(0, self.peak_gpu_memory - self.start_gpu_memory)
        
        # Calculate disk usage change
        disk_delta = (self.disk_after - self.disk_before) / (1024 * 1024) if self.disk_after and self.disk_before else 0
        
        return {
            'duration_milliseconds': duration_ms,
            'cpu_ram_mb': incremental_ram,
            'gpu_ram_mb': incremental_gpu,
            'disk_usage_change_mb': disk_delta,
            'baseline_cpu_ram_mb': self.start_memory_usage,
            'baseline_gpu_ram_mb': self.start_gpu_memory,
            'total_cpu_ram_mb': self.peak_memory_usage,
            'total_gpu_ram_mb': self.peak_gpu_memory
        }


@contextmanager
def track_resources():
    """Context manager for tracking resource usage."""
    tracker = ResourceTracker().start()
    try:
        yield tracker
    finally:
        tracker.stop()


def generate_deterministic_text(length_category: str, seed: int = 42) -> Tuple[str, str, List[str]]:
    """Generate deterministic synthetic text samples for testing.
    
    Args:
        length_category: "short", "medium", or "long"
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing (input_text, output_text, reference_texts)
    """
    import random
    
    # Simple word list to avoid external dependencies
    word_list = (
        "time person year way day thing man world life hand part child eye woman "
        "place work week case point government company number group problem fact"
    ).split()
    
    # Set random seed for deterministic generation
    rng = random.Random(seed)
    
    # Determine text lengths based on category
    if length_category == "short":
        input_len = 12
        output_len = 15
        ref_count = 2
        ref_lens = [10, 14]
    elif length_category == "medium":
        input_len = 90
        output_len = 100
        ref_count = 2
        ref_lens = [95, 105]
    else:  # long
        input_len = 260
        output_len = 280
        ref_count = 2
        ref_lens = [275, 285]
    
    # Generate texts using consistent random seed
    def rand_text(length):
        return " ".join(rng.choice(word_list) for _ in range(length)) + "."
    
    input_text = rand_text(input_len)
    output_text = rand_text(output_len)
    reference_texts = [rand_text(ref_len) for ref_len in ref_lens]
    
    return input_text, output_text, reference_texts


def generate_synthetic_text(length_category: str) -> Tuple[str, str, List[str]]:
    """Generate synthetic text samples for testing.
    
    Args:
        length_category: "short", "medium", or "long"
        
    Returns:
        Tuple containing (input_text, output_text, reference_texts)
    """
    if HAS_NLTK:
        # Use NLTK words if available
        vocab = list(NLTK_WORDS)
        if len(vocab) > 1000:  # Limit to a reasonable subset for efficiency
            vocab = random.sample(vocab, 1000)
    else:
        # Fallback vocabulary
        vocab = [
            "apple", "banana", "cat", "dog", "elephant", "frog", "guitar", 
            "house", "igloo", "jungle", "koala", "lemon", "mango", "night",
            "orange", "penguin", "queen", "rabbit", "strawberry", "tiger",
            "umbrella", "violet", "whale", "xylophone", "yellow", "zebra",
            "book", "chair", "desk", "ear", "flower", "garden", "hat", 
            "island", "jacket", "key", "lamp", "mountain", "notebook", 
            "ocean", "phone", "quilt", "river", "sun", "tree", "university",
            "village", "window", "xerox", "yard", "zebra", "airplane", 
            "bicycle", "computer", "dictionary", "engine", "forest"
        ]
    
    if length_category == "short":
        input_len = random.randint(3, 10)
        output_len = random.randint(3, 10)
        ref_count = random.randint(1, 3)
        ref_lens = [random.randint(3, 10) for _ in range(ref_count)]
    elif length_category == "medium":
        input_len = random.randint(80, 120)
        output_len = random.randint(80, 120)
        ref_count = random.randint(1, 3)
        ref_lens = [random.randint(80, 120) for _ in range(ref_count)]
    else:  # long
        input_len = random.randint(800, 1200)
        output_len = random.randint(800, 1200)
        ref_count = random.randint(1, 3)
        ref_lens = [random.randint(800, 1200) for _ in range(ref_count)]
    
    input_text = " ".join(random.choices(vocab, k=input_len))
    output_text = " ".join(random.choices(vocab, k=output_len))
    reference_texts = [" ".join(random.choices(vocab, k=ref_len)) for ref_len in ref_lens]
    
    return input_text, output_text, reference_texts


def measure_current_memory():
    """Get the current memory usage of the process."""
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    gpu_memory = 0
    if HAS_TORCH and torch.cuda.is_available():
        try:
            gpu_memory = sum([
                torch.cuda.memory_allocated(i) / (1024 * 1024) 
                for i in range(torch.cuda.device_count())
            ]) if torch.cuda.device_count() > 0 else 0
        except Exception:
            pass
    elif HAS_PYNVML:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory += info.used / (1024 * 1024)
        except Exception:
            pass
            
    return {
        'cpu_ram_mb': cpu_memory,
        'gpu_ram_mb': gpu_memory
    }


class UtilizationExperiment(Experiment):
    """Experiment class to measure resource utilization of metrics."""
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        metrics: List[Metric], 
        output_dir: str,
        dataset=None,
        num_examples: int = 30,
        num_burn_in: int = 5,
        lengths: List[str] = None,
        use_synthetic: bool = True,
        seed: int = 42,
        measure_import_costs: bool = True,  # Whether to measure import costs in separate process
        use_isolated_trials: bool = True,  # Whether to run each length category in a separate process
        use_deterministic_examples: bool = False,  # Whether to use deterministic text examples
        **kwargs
    ):
        # Measure initial memory footprint before loading anything
        self.initial_memory = measure_current_memory()
        
        # Create a minimal dataset if none provided for the parent constructor
        if dataset is None and use_synthetic:
            from autometrics.dataset.Dataset import Dataset
            class MinimalDataset(Dataset):
                def __init__(self): pass
                def get_splits(self, seed=42): return self, self, self
            dataset = MinimalDataset()
            
        # Update output directory based on synthetic vs real data
        self.use_synthetic = use_synthetic  # Store for potential future use
        if use_synthetic:
            output_dir = os.path.join(output_dir, "synthetic")
            
        super().__init__(name, description, metrics, output_dir, dataset, seed, False, **kwargs)
        
        self.num_examples = num_examples
        self.num_burn_in = num_burn_in
        self.lengths = lengths or ["short", "medium", "long"]
        self.measure_import_costs = measure_import_costs
        self.use_isolated_trials = use_isolated_trials
        self.use_deterministic_examples = use_deterministic_examples
        self.results = {}
        
    def run(self, print_results: bool = False):
        """Run the utilization experiment."""
        # Prepare a string to capture all terminal output for the text summary
        summary_output = []
        
        def log(message):
            if print_results:
                print(message)
            summary_output.append(message)
        
        log(f"Running utilization experiment with {len(self.metrics)} metrics")
        if self.use_synthetic:
            log(f"Testing {self.num_examples} synthetic examples per length category: {', '.join(self.lengths)}")
        else:
            log(f"Testing using real data from provided dataset")
        
        # Print initial memory footprint
        log(f"Initial memory footprint (before experiment):")
        log(f"  CPU RAM: {self.initial_memory['cpu_ram_mb']:.2f} MB")
        log(f"  GPU RAM: {self.initial_memory['gpu_ram_mb']:.2f} MB")
        
        # Store all results
        all_results = {
            "metrics": [],
            "results_by_metric": {},
            "initial_memory": self.initial_memory,
            "import_costs": {}  # New section for import/construction costs
        }
        
        # For real dataset, we don't use length categories
        categories_to_test = self.lengths if self.use_synthetic else ["dataset"]
        
        # Import the metric profiler if needed
        if self.measure_import_costs:
            try:
                from autometrics.experiments.utilization.metric_profiler import measure_metric_phases
                from autometrics.experiments.utilization.resource import calc_delta
            except ImportError:
                log("⚠️ Warning: metric_profiler not found. Import costs will not be measured.")
                self.measure_import_costs = False
        
        # Run experiment for each metric
        for metric in self.metrics:
            metric_name = metric.get_name()
            log(f"Testing metric: {metric_name}")
            
            # Measure import costs if enabled (using separate process)
            if self.measure_import_costs:
                log(f"  Measuring import and construction costs in clean process...")
                
                # Generate the full class path for the metric (best effort)
                try:
                    metric_module = metric.__class__.__module__
                    metric_class = metric.__class__.__name__
                    metric_class_path = f"{metric_module}.{metric_class}"
                    
                    # Create sample test data (the first example from the first category)
                    category = categories_to_test[0]
                    sample_data = self._get_test_examples(category)[0]
                    
                    # Get constructor arguments from the metric instance if possible
                    constructor_kwargs = {}
                    if hasattr(metric, '_init_params'):
                        # Filter out excluded parameters that don't affect the metric
                        constructor_kwargs = {k: v for k, v in metric._init_params.items() 
                                            if not hasattr(metric, '_excluded_params') or 
                                            k not in metric._excluded_params}
                    
                    # Add use_cache=False to avoid any caching during import cost measurement
                    if hasattr(metric.__class__, '__init__'):
                        import inspect
                        # Only include parameters that the constructor actually accepts
                        sig = inspect.signature(metric.__class__.__init__)
                        allowed_params = set(sig.parameters.keys()) - {'self'}
                        constructor_kwargs = {k: v for k, v in constructor_kwargs.items() 
                                             if k in allowed_params}
                        
                        # Only add use_cache if the constructor accepts it
                        if 'use_cache' in allowed_params:
                            constructor_kwargs['use_cache'] = False
                    
                    # Measure phases
                    try:
                        # Log the constructor kwargs for debugging
                        log(f"    Constructor kwargs: {constructor_kwargs}")
                        
                        checkpoints = measure_metric_phases(
                            metric_class_path=metric_class_path,
                            constructor_kwargs=constructor_kwargs,
                            sample_data=sample_data
                        )
                        
                        if not checkpoints:
                            log(f"  ⚠️ Warning: No checkpoints returned from metric profiler")
                            # Still continue with the experiment without import costs
                        else:
                            # Calculate deltas between checkpoints
                            import_costs = []
                            for i in range(len(checkpoints) - 1):
                                before = checkpoints[i]
                                after = checkpoints[i+1]
                                delta = calc_delta(before, after)
                                import_costs.append(delta)
                            
                            # Store the results
                            all_results["import_costs"][metric_name] = {
                                "checkpoints": checkpoints,
                                "deltas": import_costs
                            }
                            
                            # Save as tabular result
                            if import_costs:
                                import_df = pd.DataFrame(import_costs)
                                self.results[f"{metric_name}/import_costs"] = TabularResult(import_df)
                            
                            for delta in import_costs:
                                log(f"    {delta['phase']}:")
                                log(f"      CPU RAM: {delta['cpu_ram_mb']:.2f} MB")
                                log(f"      GPU RAM: {delta['gpu_ram_mb']:.2f} MB")
                                log(f"      Duration: {delta['duration_milliseconds']:.2f} ms")
                    
                    except Exception as e:
                        log(f"  ⚠️ Error measuring import costs: {str(e)}")
                        # Get full traceback for debugging
                        import traceback
                        log(f"  Traceback: {traceback.format_exc()}")
                        # Continue with the experiment, but without import costs
                
                except Exception as e:
                    log(f"  ⚠️ Error setting up import cost measurement: {str(e)}")
                    import traceback
                    log(f"  Traceback: {traceback.format_exc()}")
                    # Continue with the experiment
            
            # Disable cache for this metric if possible
            original_cache_state = None
            try:
                original_cache_state = metric.use_cache
                metric.use_cache = False
            except AttributeError:
                pass
            
            metric_results = {"name": metric_name, "results": {}}
            
            # For each length category (if synthetic) or just once (if real data)
            for category in categories_to_test:
                if self.use_synthetic:
                    log(f"  Testing with {category} inputs/outputs")
                else:
                    log(f"  Testing with real dataset examples")
                
                # Track metadata for results
                result_prefix = f"{metric_name}/{category}" if self.use_synthetic else f"{metric_name}"
                
                # Check for model loading methods
                has_model_methods = hasattr(metric, '_load_model') and hasattr(metric, '_unload_model')
                model_was_loaded = False
                if has_model_methods:
                    model_was_loaded = hasattr(metric, 'model') and metric.model is not None
                
                # Run trials in isolated subprocess if requested and supported
                if self.use_isolated_trials and self.use_synthetic:
                    log(f"  Running in isolated subprocess for clean memory measurements...")
                    
                    # Generate the full class path for the metric
                    metric_module = metric.__class__.__module__
                    metric_class = metric.__class__.__name__
                    metric_class_path = f"{metric_module}.{metric_class}"
                    
                    # Get constructor arguments from the metric instance if possible
                    constructor_kwargs = {}
                    if hasattr(metric, '_init_params'):
                        # Filter out excluded parameters
                        constructor_kwargs = {k: v for k, v in metric._init_params.items() 
                                            if not hasattr(metric, '_excluded_params') or 
                                            k not in metric._excluded_params}
                    
                    # Import the isolated runner
                    from autometrics.experiments.utilization.isolated_runner import run_isolated_category
                    
                    # Run the isolated category
                    length_results = run_isolated_category(
                        metric_class_path=metric_class_path,
                        constructor_kwargs=constructor_kwargs,
                        category=category,
                        num_examples=self.num_examples,
                        num_burn_in=self.num_burn_in,
                        use_synthetic=self.use_synthetic,
                        seed=self.seed,
                        use_deterministic_examples=self.use_deterministic_examples
                    )
                    
                    if not length_results:
                        log(f"  ⚠️ Error: Failed to run isolated trials for {category}")
                        continue
                else:
                    # Get test examples
                    test_examples = self._get_test_examples(category)
                    length_results = []
                    
                    # If the metric has load/unload methods, unload and reload the model
                    if has_model_methods:
                        # Unload model to ensure we measure full memory usage
                        try:
                            metric._unload_model()
                        except Exception as e:
                            log(f"  ⚠️ Warning: Failed to unload model: {str(e)}")

                        try:
                            metric._load_model()
                        except Exception as e:
                            log(f"  ⚠️ Warning: Failed to load model: {str(e)}")
                    
                    # Force garbage collection before starting
                    import gc
                    gc.collect()
                    
                    # Perform burn-in samples first
                    for i in range(min(self.num_burn_in, len(test_examples))):
                        input_text, output_text, reference_texts = test_examples[i]
                        try:
                            metric.calculate(input_text, output_text, reference_texts)
                        except Exception as e:
                            log(f"  ⚠️ Error during burn-in: {str(e)}")
                    
                    # Now run the actual experiments
                    for i in range(min(self.num_examples, len(test_examples))):
                        input_text, output_text, reference_texts = test_examples[i]
                        
                        # Force garbage collection between trials for more consistent measurements
                        if i > 0 and i % 5 == 0:  # Every 5 trials
                            gc.collect()
                        
                        try:
                            with track_resources() as tracker:
                                result = metric.calculate(input_text, output_text, reference_texts)
                            
                            resources = tracker.get_results()
                            length_results.append(resources)
                        except Exception as e:
                            log(f"  ⚠️ Error on example {i}: {str(e)}")
                            traceback.print_exc()
                        
                        # Unload the model if it was not loaded initially and has unload methods
                        if not model_was_loaded and has_model_methods:
                            try:
                                metric._unload_model()
                            except Exception as e:
                                log(f"  ⚠️ Warning: Failed to unload model: {str(e)}")
                                
                # Calculate statistics
                if length_results:
                    stats = self._calculate_statistics(length_results, all_results["import_costs"].get(metric_name, {}).get("deltas", []))
                    metric_results["results"][category] = {
                        "raw_data": length_results,
                        "summary": stats
                    }
                    
                    # Save raw data
                    raw_df = pd.DataFrame(length_results)
                    raw_df.insert(0, 'example_id', range(len(raw_df)))
                    self.results[f"{result_prefix}/raw_data"] = TabularResult(raw_df)
                    
                    # Save summary
                    summary_df = pd.DataFrame([stats])
                    self.results[f"{result_prefix}/summary"] = TabularResult(summary_df)
                    
                    # Generate plots
                    self._generate_resource_plots(raw_df, result_prefix)
            
            # Restore cache setting
            if original_cache_state is not None:
                try:
                    metric.use_cache = original_cache_state
                except AttributeError:
                    pass
            
            # Final unload of the model if it was not loaded initially
            if not model_was_loaded and has_model_methods:
                try:
                    metric._unload_model()
                except Exception as e:
                    log(f"  ⚠️ Warning: Failed to unload model: {str(e)}")

            all_results["metrics"].append(metric_name)
            all_results["results_by_metric"][metric_name] = metric_results
        
        # Save combined results if we have multiple metrics
        if len(self.metrics) > 1:
            self._generate_comparison_plots(all_results)
        
        # Save full results JSON
        self.results["full_results"] = JSONResult(all_results)
        
        log("\nUtilization Experiment Results Summary:")
        
        # First print import costs for each metric if available
        if all_results.get("import_costs") and len(all_results["import_costs"]) > 0:
            log("\n=== IMPORT AND CONSTRUCTION COSTS ===")
            for metric_name in all_results["metrics"]:
                if metric_name in all_results["import_costs"]:
                    import_costs = all_results["import_costs"][metric_name]["deltas"]
                    log(f"\n{metric_name} import and construction:")
                    total_cpu = 0
                    total_gpu = 0
                    for delta in import_costs:
                        phase = delta["phase"]
                        cpu = delta["cpu_ram_mb"]
                        gpu = delta["gpu_ram_mb"]
                        duration = delta["duration_milliseconds"]
                        log(f"  {phase}:")
                        log(f"    CPU RAM: {cpu:.2f} MB")
                        log(f"    GPU RAM: {gpu:.2f} MB")
                        log(f"    Duration: {duration:.2f} ms")
                        total_cpu += cpu
                        total_gpu += gpu
                    log(f"  TOTAL IMPORT COSTS:")
                    log(f"    CPU RAM: {total_cpu:.2f} MB")
                    log(f"    GPU RAM: {total_gpu:.2f} MB")
        
        log("\n=== RUNTIME COSTS ===")
        
        # Create a summary dataframe for all metrics and categories
        summary_rows = []
        
        for metric_name in all_results["metrics"]:
            for category, data in all_results["results_by_metric"][metric_name]["results"].items():
                summary = data["summary"]
                log(f"\n{metric_name} - {category}:")
                log(f"  Duration (ms): {summary['mean_duration_milliseconds']:.2f} " +
                    f"±{summary['ci_upper_duration'] - summary['mean_duration_milliseconds']:.2f}")
                
                # Print both incremental and total memory usage
                incremental_ram = summary['mean_cpu_ram_mb']
                baseline_ram = summary['mean_baseline_cpu_ram_mb']
                total_ram = summary['mean_total_cpu_ram_mb']
                
                log(f"  CPU RAM - Baseline: {baseline_ram:.2f} MB")
                log(f"  CPU RAM - Used by metric: {incremental_ram:.2f} MB " +
                    f"±{summary['ci_upper_cpu_ram'] - summary['mean_cpu_ram_mb']:.2f}")
                log(f"  CPU RAM - Total: {total_ram:.2f} MB")
                
                # Always print GPU RAM stats, even if zero
                incremental_gpu = summary['mean_gpu_ram_mb']
                baseline_gpu = summary['mean_baseline_gpu_ram_mb']
                total_gpu = summary['mean_total_gpu_ram_mb']
                
                log(f"  GPU RAM - Baseline: {baseline_gpu:.2f} MB")
                log(f"  GPU RAM - Used by metric: {incremental_gpu:.2f} MB " +
                    f"±{summary['ci_upper_gpu_ram'] - summary['mean_gpu_ram_mb']:.2f}")
                log(f"  GPU RAM - Total: {total_gpu:.2f} MB")
                
                # Print combined import + runtime if available
                if "total_with_import_cpu_ram_mb" in summary:
                    log(f"  CPU RAM - With Import: {summary['total_with_import_cpu_ram_mb']:.2f} MB (import: {summary['import_cpu_ram_mb']:.2f} MB + runtime: {incremental_ram:.2f} MB)")
                
                if "total_with_import_gpu_ram_mb" in summary:
                    log(f"  GPU RAM - With Import: {summary['total_with_import_gpu_ram_mb']:.2f} MB (import: {summary['import_gpu_ram_mb']:.2f} MB + runtime: {incremental_gpu:.2f} MB)")
                
                # Add to summary rows for CSV
                summary_rows.append({
                    'metric': metric_name,
                    'category': category,
                    'duration_ms': summary['mean_duration_milliseconds'],
                    'duration_ci': summary['ci_upper_duration'] - summary['mean_duration_milliseconds'],
                    'cpu_ram_mb_baseline': baseline_ram,
                    'cpu_ram_mb_incremental': incremental_ram, 
                    'cpu_ram_mb_total': total_ram,
                    'cpu_ram_ci': summary['ci_upper_cpu_ram'] - summary['mean_cpu_ram_mb'],
                    'gpu_ram_mb_baseline': baseline_gpu,
                    'gpu_ram_mb_incremental': incremental_gpu,
                    'gpu_ram_mb_total': total_gpu,
                    'gpu_ram_ci': summary['ci_upper_gpu_ram'] - summary['mean_gpu_ram_mb'],
                    'import_cpu_ram_mb': summary.get('import_cpu_ram_mb', 0),
                    'import_gpu_ram_mb': summary.get('import_gpu_ram_mb', 0),
                    'total_with_import_cpu_ram_mb': summary.get('total_with_import_cpu_ram_mb', incremental_ram),
                    'total_with_import_gpu_ram_mb': summary.get('total_with_import_gpu_ram_mb', incremental_gpu)
                })
        
        # Save the collected summary text as a TextResult
        self.results["summary_text"] = TextResult("\n".join(summary_output))
        
        # Save the summary data as a CSV
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            self.results["complete_summary"] = TabularResult(summary_df)

    def _get_test_examples(self, category):
        """Get test examples for the experiment.
        
        Args:
            category: The category name ("short", "medium", "long", or "dataset")
            
        Returns:
            List of (input_text, output_text, reference_texts) tuples
        """
        examples = []
        
        if self.use_synthetic:
            # Generate synthetic examples based on length category
            for i in range(self.num_examples + self.num_burn_in):
                if self.use_deterministic_examples:
                    # Use different seed for each example, but derived from the main seed
                    example_seed = self.seed + i
                    examples.append(generate_deterministic_text(category, example_seed))
                else:
                    # Use the original non-deterministic generation
                    examples.append(generate_synthetic_text(category))
        else:
            # Use examples from dataset without filtering by length
            df = self.test_dataset.get_dataframe()
            
            # If dataset is too large, take a subset
            if len(df) > (self.num_examples + self.num_burn_in):
                test_dataset = self.test_dataset.get_subset(self.num_examples + self.num_burn_in, self.seed)
                # Use random seed for consistent subsetting
                df = test_dataset.get_dataframe()
            
            # Convert each row to the expected format
            for _, row in df.iterrows():
                input_text = row[self.test_dataset.get_input_column()]
                output_text = row[self.test_dataset.get_output_column()]
                references = [row[ref_col] for ref_col in self.test_dataset.get_reference_columns()]
                examples.append((input_text, output_text, references))
        
        return examples
    
    def _calculate_statistics(self, results, import_costs=None):
        """Calculate statistics from a list of resource measurements.
        
        Args:
            results: List of resource measurement dictionaries
            import_costs: Optional import cost data to include in the calculations
        """
        durations = [r["duration_milliseconds"] for r in results]
        cpu_ram = [r["cpu_ram_mb"] for r in results]
        gpu_ram = [r["gpu_ram_mb"] for r in results]
        disk_usage = [r["disk_usage_change_mb"] for r in results]
        baseline_cpu_ram = [r["baseline_cpu_ram_mb"] for r in results]
        baseline_gpu_ram = [r["baseline_gpu_ram_mb"] for r in results]
        total_cpu_ram = [r["total_cpu_ram_mb"] for r in results]
        total_gpu_ram = [r["total_gpu_ram_mb"] for r in results]
        
        # Add import costs if provided
        import_cpu_ram = 0
        import_gpu_ram = 0
        import_duration = 0
        if import_costs:
            # Sum all import and construction costs
            for cost in import_costs:
                if cost.get("phase", "").startswith("start→") or cost.get("phase", "").startswith("after_import→"):
                    import_cpu_ram += cost.get("cpu_ram_mb", 0)
                    import_gpu_ram += cost.get("gpu_ram_mb", 0)
                    import_duration += cost.get("duration_milliseconds", 0)
        
        mean_duration = statistics.mean(durations)
        mean_cpu_ram = statistics.mean(cpu_ram)
        mean_gpu_ram = statistics.mean(gpu_ram)
        mean_disk = statistics.mean(disk_usage)
        mean_baseline_cpu_ram = statistics.mean(baseline_cpu_ram)
        mean_baseline_gpu_ram = statistics.mean(baseline_gpu_ram)
        mean_total_cpu_ram = statistics.mean(total_cpu_ram)
        mean_total_gpu_ram = statistics.mean(total_gpu_ram)
        
        # Calculate 95% confidence intervals
        if len(durations) > 1:
            std_duration = statistics.stdev(durations)
            std_cpu_ram = statistics.stdev(cpu_ram)
            std_gpu_ram = statistics.stdev(gpu_ram)
            std_disk = statistics.stdev(disk_usage)
            std_baseline_cpu_ram = statistics.stdev(baseline_cpu_ram)
            std_baseline_gpu_ram = statistics.stdev(baseline_gpu_ram)
            std_total_cpu_ram = statistics.stdev(total_cpu_ram)
            std_total_gpu_ram = statistics.stdev(total_gpu_ram)
            
            n = len(durations)
            # Use t-distribution with n-1 degrees of freedom for small samples
            # For simplicity, approximating with 1.96 * std/sqrt(n) for 95% CI
            margin_duration = 1.96 * std_duration / (n ** 0.5)
            margin_cpu_ram = 1.96 * std_cpu_ram / (n ** 0.5)
            margin_gpu_ram = 1.96 * std_gpu_ram / (n ** 0.5)
            margin_disk = 1.96 * std_disk / (n ** 0.5)
            margin_baseline_cpu_ram = 1.96 * std_baseline_cpu_ram / (n ** 0.5)
            margin_baseline_gpu_ram = 1.96 * std_baseline_gpu_ram / (n ** 0.5)
            margin_total_cpu_ram = 1.96 * std_total_cpu_ram / (n ** 0.5)
            margin_total_gpu_ram = 1.96 * std_total_gpu_ram / (n ** 0.5)
        else:
            margin_duration = margin_cpu_ram = margin_gpu_ram = margin_disk = 0
            margin_baseline_cpu_ram = margin_baseline_gpu_ram = margin_total_cpu_ram = margin_total_gpu_ram = 0
        
        stats = {
            "mean_duration_milliseconds": mean_duration,
            "ci_lower_duration": max(0, mean_duration - margin_duration),
            "ci_upper_duration": mean_duration + margin_duration,
            
            "mean_cpu_ram_mb": mean_cpu_ram,
            "ci_lower_cpu_ram": max(0, mean_cpu_ram - margin_cpu_ram),
            "ci_upper_cpu_ram": mean_cpu_ram + margin_cpu_ram,
            
            "mean_gpu_ram_mb": mean_gpu_ram,
            "ci_lower_gpu_ram": max(0, mean_gpu_ram - margin_gpu_ram),
            "ci_upper_gpu_ram": mean_gpu_ram + margin_gpu_ram,
            
            "mean_disk_usage_change_mb": mean_disk,
            "ci_lower_disk": mean_disk - margin_disk,
            "ci_upper_disk": mean_disk + margin_disk,
            
            "mean_baseline_cpu_ram_mb": mean_baseline_cpu_ram,
            "ci_lower_baseline_cpu_ram": max(0, mean_baseline_cpu_ram - margin_baseline_cpu_ram),
            "ci_upper_baseline_cpu_ram": mean_baseline_cpu_ram + margin_baseline_cpu_ram,
            
            "mean_baseline_gpu_ram_mb": mean_baseline_gpu_ram,
            "ci_lower_baseline_gpu_ram": max(0, mean_baseline_gpu_ram - margin_baseline_gpu_ram),
            "ci_upper_baseline_gpu_ram": mean_baseline_gpu_ram + margin_baseline_gpu_ram,
            
            "mean_total_cpu_ram_mb": mean_total_cpu_ram,
            "ci_lower_total_cpu_ram": max(0, mean_total_cpu_ram - margin_total_cpu_ram),
            "ci_upper_total_cpu_ram": mean_total_cpu_ram + margin_total_cpu_ram,
            
            "mean_total_gpu_ram_mb": mean_total_gpu_ram,
            "ci_lower_total_gpu_ram": max(0, mean_total_gpu_ram - margin_total_gpu_ram),
            "ci_upper_total_gpu_ram": mean_total_gpu_ram + margin_total_gpu_ram,
        }
        
        # Include import costs if provided
        if import_costs:
            stats["import_cpu_ram_mb"] = import_cpu_ram
            stats["import_gpu_ram_mb"] = import_gpu_ram
            stats["import_duration_ms"] = import_duration
            
            # Calculate total cost (import + runtime)
            stats["total_with_import_cpu_ram_mb"] = import_cpu_ram + mean_cpu_ram
            stats["total_with_import_gpu_ram_mb"] = import_gpu_ram + mean_gpu_ram
            stats["total_with_import_duration_ms"] = import_duration + mean_duration
        
        return stats
    
    def _generate_resource_plots(self, df, prefix):
        """Generate plots for a single metric/category."""
        # Time series plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = df['example_id']
        ax.plot(x, df['duration_milliseconds'], 'o-', label='Duration (ms)')
        ax.set_xlabel('Example')
        ax.set_ylabel('Duration (milliseconds)')
        ax.set_title('Runtime Performance')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save_figure(fig, f"{prefix}/duration_timeseries")
        
        # Memory usage plot - incremental vs baseline
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the baseline memory (which should be constant across runs)
        baseline_memory = df['baseline_cpu_ram_mb'].mean()
        
        # Plot stacked bar chart to show baseline and incremental memory
        ax.bar(x, df['baseline_cpu_ram_mb'], label='Baseline RAM', alpha=0.5, color='lightgray')
        ax.bar(x, df['cpu_ram_mb'], bottom=df['baseline_cpu_ram_mb'], 
               label='Additional RAM used by metric', color='blue', alpha=0.7)
        
        # If GPU memory was used, plot it similarly
        if df['gpu_ram_mb'].sum() > 0:
            ax2 = ax.twinx()
            ax2.plot(x, df['gpu_ram_mb'], 'r--', label='GPU RAM')
            ax2.set_ylabel('GPU Memory (MB)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax.legend()
            
        ax.set_xlabel('Example')
        ax.set_ylabel('CPU Memory (MB)')
        ax.set_title('Memory Utilization')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save_figure(fig, f"{prefix}/memory_timeseries")
        
        # Histogram of duration - but only if we have enough different values
        fig, ax = plt.subplots(figsize=(10, 6))
        durations = df['duration_milliseconds'].values
        if len(durations) > 1 and not np.all(durations == durations[0]):
            # Only create a histogram if we have at least 2 different values
            bin_count = min(20, max(1, len(df) // 2))
            ax.hist(durations, bins=bin_count, alpha=0.7)
        else:
            # If all values are identical, just make a single bar
            ax.bar([durations[0]], [len(durations)], alpha=0.7, width=durations[0]*0.1 if durations[0] > 0 else 0.1)
        
        ax.axvline(df['duration_milliseconds'].mean(), color='red', linestyle='dashed', linewidth=2, 
                  label=f'Mean: {df["duration_milliseconds"].mean():.2f} ms')
        ax.set_xlabel('Duration (milliseconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Duration Distribution')
        ax.legend()
        fig.tight_layout()
        self._save_figure(fig, f"{prefix}/duration_histogram")
        
        # Histogram of memory usage - but only if we have enough different values
        fig, ax = plt.subplots(figsize=(10, 6))
        memory_values = df['cpu_ram_mb'].values
        if len(memory_values) > 1 and not np.all(memory_values == memory_values[0]):
            # Only create a histogram if we have at least 2 different values
            bin_count = min(10, max(1, len(df) // 3))
            ax.hist(memory_values, bins=bin_count, alpha=0.7, 
                    label=f'Mean: {df["cpu_ram_mb"].mean():.2f} MB')
        else:
            # If all values are identical, just make a single bar
            ax.bar([memory_values[0]], [len(memory_values)], alpha=0.7, 
                   width=memory_values[0]*0.1 if memory_values[0] > 0 else 0.1,
                   label=f'Mean: {df["cpu_ram_mb"].mean():.2f} MB')
            
        ax.axvline(df['cpu_ram_mb'].mean(), color='red', linestyle='dashed', linewidth=2)
        ax.set_xlabel('CPU RAM Usage (MB)')
        ax.set_ylabel('Frequency')
        ax.set_title('Memory Usage Distribution')
        ax.legend()
        fig.tight_layout()
        self._save_figure(fig, f"{prefix}/memory_histogram")
    
    def _generate_comparison_plots(self, all_results):
        """Generate comparison plots across all metrics."""
        if not self.use_synthetic:
            # For real datasets, create summary comparison across metrics
            summary_data = []
            for metric_name in all_results["metrics"]:
                data = all_results["results_by_metric"][metric_name]["results"]["dataset"]["summary"]
                
                # Basic data from runtime measurements
                metric_data = {
                    "metric": metric_name,
                    "duration_ms": data["mean_duration_milliseconds"],
                    "duration_ci": data["ci_upper_duration"] - data["mean_duration_milliseconds"],
                    "cpu_ram_mb": data["mean_cpu_ram_mb"],  # This is incremental RAM during calculation
                    "cpu_ram_ci": data["ci_upper_cpu_ram"] - data["mean_cpu_ram_mb"],
                    "gpu_ram_mb": data["mean_gpu_ram_mb"],
                    "gpu_ram_ci": data["ci_upper_gpu_ram"] - data["mean_gpu_ram_mb"],
                    "baseline_cpu_ram_mb": data.get("mean_baseline_cpu_ram_mb", 0),  # Baseline during calculation
                }
                
                # Add import cost data if available
                if "import_costs" in all_results and metric_name in all_results["import_costs"]:
                    import_costs = all_results["import_costs"][metric_name]["deltas"]
                    
                    # Extract costs by phase
                    for delta in import_costs:
                        phase = delta["phase"]
                        phase_key = phase.replace("→", "_to_").replace(" ", "_").lower()
                        
                        metric_data[f"import_{phase_key}_cpu_mb"] = delta["cpu_ram_mb"]
                        metric_data[f"import_{phase_key}_gpu_mb"] = delta["gpu_ram_mb"]
                        metric_data[f"import_{phase_key}_duration_ms"] = delta["duration_milliseconds"]
                        
                        # For phases related to model loading, track specially
                        if "first_call" in phase:
                            metric_data["model_load_cpu_mb"] = delta["cpu_ram_mb"]
                            metric_data["model_load_gpu_mb"] = delta["gpu_ram_mb"]
                
                summary_data.append(metric_data)
            
            df = pd.DataFrame(summary_data)
            self.results["metric_comparison/summary"] = TabularResult(df)
            
            # Generate a more detailed import costs breakdown if available
            if "import_costs" in all_results and all_results["import_costs"]:
                import_cost_data = []
                for metric_name, cost_data in all_results["import_costs"].items():
                    for delta in cost_data["deltas"]:
                        import_cost_data.append({
                            "metric": metric_name,
                            "phase": delta["phase"],
                            "cpu_ram_mb": delta["cpu_ram_mb"],
                            "gpu_ram_mb": delta["gpu_ram_mb"],
                            "duration_ms": delta["duration_milliseconds"],
                            "baseline_cpu_ram_mb": delta["baseline_cpu_ram_mb"],
                            "total_cpu_ram_mb": delta["total_cpu_ram_mb"]
                        })
                
                if import_cost_data:
                    import_df = pd.DataFrame(import_cost_data)
                    self.results["metric_comparison/import_costs"] = TabularResult(import_df)
                    
                    # Create a stacked bar chart for import/construction/model loading costs
                    self._generate_import_cost_plots(import_df)
            
            # Bar chart comparison
            fig, ax = plt.subplots(figsize=(12, 8))
            metrics = df["metric"].values
            x_pos = np.arange(len(metrics))
            
            ax.bar(x_pos, df["duration_ms"], yerr=df["duration_ci"], capsize=10)
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Duration (milliseconds)')
            ax.set_title('Average Runtime by Metric')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            fig.tight_layout()
            self._save_figure(fig, "metric_comparison/duration")
            
            # Box and whisker plot for timing
            if len(all_results["metrics"]) > 1:
                # Collect all timing data across all metrics
                timing_data = []
                metric_labels = []
                for metric_name in all_results["metrics"]:
                    raw_data = all_results["results_by_metric"][metric_name]["results"]["dataset"]["raw_data"]
                    durations = [d["duration_milliseconds"] for d in raw_data]
                    timing_data.append(durations)
                    metric_labels.append(metric_name)
                
                # Create box plot
                fig, ax = plt.subplots(figsize=(12, 8))
                box = ax.boxplot(timing_data, patch_artist=True, labels=metric_labels)
                
                # Add some styling
                for patch in box['boxes']:
                    patch.set_facecolor('lightblue')
                
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Duration (milliseconds)')
                ax.set_title('Runtime Distribution by Metric')
                ax.set_xticklabels(metric_labels, rotation=45, ha='right')
                plt.grid(True, axis='y', alpha=0.3)
                fig.tight_layout()
                self._save_figure(fig, "metric_comparison/duration_boxplot")
            
            # Memory comparison - stacked bar to show baseline and incremental
            fig, ax = plt.subplots(figsize=(12, 8))
            x_pos = np.arange(len(metrics))
            
            # Plot baseline memory
            ax.bar(x_pos, df["baseline_cpu_ram_mb"], label='Baseline RAM', alpha=0.5, color='lightgray')
            
            # Plot incremental memory on top
            ax.bar(x_pos, df["cpu_ram_mb"], bottom=df["baseline_cpu_ram_mb"],
                   yerr=df["cpu_ram_ci"], capsize=10, label='Incremental RAM', color='blue', alpha=0.7)
            
            # If any GPU memory was used, add it as a separate set of bars
            if df["gpu_ram_mb"].sum() > 0:
                ax2 = ax.twinx()
                ax2.bar(x_pos + 0.2, df["gpu_ram_mb"], width=0.3, yerr=df["gpu_ram_ci"], 
                        capsize=10, color='red', alpha=0.7, label='GPU RAM')
                ax2.set_ylabel('GPU Memory (MB)', color='r')
                
                # Combine legends
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper left')
            else:
                ax.legend()
                
            ax.set_xlabel('Metrics')
            ax.set_ylabel('CPU Memory Usage (MB)')
            ax.set_title('Memory Usage by Metric')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            fig.tight_layout()
            self._save_figure(fig, "metric_comparison/memory")
        else:
            # For synthetic data, compare within each length category
            for length in self.lengths:
                summary_data = []
                for metric_name in all_results["metrics"]:
                    if length in all_results["results_by_metric"][metric_name]["results"]:
                        data = all_results["results_by_metric"][metric_name]["results"][length]["summary"]
                        summary_data.append({
                            "metric": metric_name,
                            "length": length,
                            "duration_ms": data["mean_duration_milliseconds"],
                            "duration_ci": data["ci_upper_duration"] - data["mean_duration_milliseconds"],
                            "cpu_ram_mb": data["mean_cpu_ram_mb"],
                            "cpu_ram_ci": data["ci_upper_cpu_ram"] - data["mean_cpu_ram_mb"],
                            "gpu_ram_mb": data["mean_gpu_ram_mb"],
                            "gpu_ram_ci": data["ci_upper_gpu_ram"] - data["mean_gpu_ram_mb"],
                        })
                
                if not summary_data:
                    continue
                    
                df = pd.DataFrame(summary_data)
                self.results[f"metric_comparison/{length}/summary"] = TabularResult(df)
                
                # Bar chart comparison - duration
                fig, ax = plt.subplots(figsize=(12, 8))
                metrics = df["metric"].values
                x_pos = np.arange(len(metrics))
                
                ax.bar(x_pos, df["duration_ms"], yerr=df["duration_ci"], capsize=10)
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Duration (milliseconds)')
                ax.set_title(f'Average Runtime by Metric ({length})')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(metrics, rotation=45, ha='right')
                fig.tight_layout()
                self._save_figure(fig, f"metric_comparison/{length}/duration")
                
                # Box and whisker plot for timing
                if len(all_results["metrics"]) > 1:
                    # Collect all timing data across all metrics
                    timing_data = []
                    metric_labels = []
                    for metric_name in all_results["metrics"]:
                        if length in all_results["results_by_metric"][metric_name]["results"]:
                            raw_data = all_results["results_by_metric"][metric_name]["results"][length]["raw_data"]
                            durations = [d["duration_milliseconds"] for d in raw_data]
                            timing_data.append(durations)
                            metric_labels.append(metric_name)
                    
                    # Create box plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    box = ax.boxplot(timing_data, patch_artist=True, labels=metric_labels)
                    
                    # Add some styling
                    for patch in box['boxes']:
                        patch.set_facecolor('lightblue')
                    
                    ax.set_xlabel('Metrics')
                    ax.set_ylabel('Duration (milliseconds)')
                    ax.set_title(f'Runtime Distribution by Metric ({length})')
                    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
                    plt.grid(True, axis='y', alpha=0.3)
                    fig.tight_layout()
                    self._save_figure(fig, f"metric_comparison/{length}/duration_boxplot")
                
                # Bar chart comparison - memory
                fig, ax = plt.subplots(figsize=(12, 8))
                width = 0.35
                
                ax.bar(x_pos - width/2, df["cpu_ram_mb"], width, yerr=df["cpu_ram_ci"], capsize=10, label='CPU RAM')
                if df["gpu_ram_mb"].sum() > 0:
                    ax.bar(x_pos + width/2, df["gpu_ram_mb"], width, yerr=df["gpu_ram_ci"], capsize=10, label='GPU RAM')
                    
                ax.set_xlabel('Metrics')
                ax.set_ylabel('Memory Usage (MB)')
                ax.set_title(f'Average Memory Usage by Metric ({length})')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(metrics, rotation=45, ha='right')
                ax.legend()
                fig.tight_layout()
                self._save_figure(fig, f"metric_comparison/{length}/memory")
            
            # Also create a comparison across all lengths for each metric
            for metric_name in all_results["metrics"]:
                length_data = []
                metric_results = all_results["results_by_metric"][metric_name]["results"]
                
                for length in self.lengths:
                    if length in metric_results:
                        data = metric_results[length]["summary"]
                        length_data.append({
                            "metric": metric_name,
                            "length": length,
                            "duration_ms": data["mean_duration_milliseconds"],
                            "duration_ci": data["ci_upper_duration"] - data["mean_duration_milliseconds"],
                            "cpu_ram_mb": data["mean_cpu_ram_mb"],
                            "cpu_ram_ci": data["ci_upper_cpu_ram"] - data["mean_cpu_ram_mb"],
                            "gpu_ram_mb": data["mean_gpu_ram_mb"],
                            "gpu_ram_ci": data["ci_upper_gpu_ram"] - data["mean_gpu_ram_mb"],
                        })
                
                if not length_data:
                    continue
                    
                df = pd.DataFrame(length_data)
                self.results[f"length_comparison/{metric_name}/summary"] = TabularResult(df)
                
                # Bar chart comparison - duration vs length
                fig, ax = plt.subplots(figsize=(10, 6))
                lengths = df["length"].values
                x_pos = np.arange(len(lengths))
                
                ax.bar(x_pos, df["duration_ms"], yerr=df["duration_ci"], capsize=10)
                ax.set_xlabel('Text Length')
                ax.set_ylabel('Duration (milliseconds)')
                ax.set_title(f'Runtime vs Text Length ({metric_name})')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(lengths)
                fig.tight_layout()
                self._save_figure(fig, f"length_comparison/{metric_name}/duration")
                
                # Box plot comparison - duration vs length
                # Collect timing data across all lengths
                timing_data = []
                length_labels = []
                for length in self.lengths:
                    if length in metric_results:
                        raw_data = metric_results[length]["raw_data"]
                        durations = [d["duration_milliseconds"] for d in raw_data]
                        timing_data.append(durations)
                        length_labels.append(length)
                
                # Create box plot
                fig, ax = plt.subplots(figsize=(10, 6))
                box = ax.boxplot(timing_data, patch_artist=True, labels=length_labels)
                
                # Add some styling
                for patch in box['boxes']:
                    patch.set_facecolor('lightblue')
                
                ax.set_xlabel('Text Length')
                ax.set_ylabel('Duration (milliseconds)')
                ax.set_title(f'Runtime Distribution vs Text Length ({metric_name})')
                plt.grid(True, axis='y', alpha=0.3)
                fig.tight_layout()
                self._save_figure(fig, f"length_comparison/{metric_name}/duration_boxplot")
                
                # Bar chart comparison - memory vs length
                fig, ax = plt.subplots(figsize=(10, 6))
                width = 0.35
                
                ax.bar(x_pos - width/2, df["cpu_ram_mb"], width, yerr=df["cpu_ram_ci"], capsize=10, label='CPU RAM')
                if df["gpu_ram_mb"].sum() > 0:
                    ax.bar(x_pos + width/2, df["gpu_ram_mb"], width, yerr=df["gpu_ram_ci"], capsize=10, label='GPU RAM')
                    
                ax.set_xlabel('Text Length')
                ax.set_ylabel('Memory Usage (MB)')
                ax.set_title(f'Memory Usage vs Text Length ({metric_name})')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(lengths)
                ax.legend()
                fig.tight_layout()
                self._save_figure(fig, f"length_comparison/{metric_name}/memory")

    def _generate_import_cost_plots(self, import_df):
        """Generate visualization of import, construction, and model loading costs."""
        # Create a pivot table to restructure the data for the stacked bar chart
        pivot_df = import_df.pivot_table(
            index="metric", 
            columns="phase", 
            values=["cpu_ram_mb", "gpu_ram_mb", "duration_ms"],
            aggfunc="mean"
        ).fillna(0)
        
        # CPU Memory Usage Plot
        if "cpu_ram_mb" in pivot_df:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get all phases for CPU RAM
            phases = pivot_df["cpu_ram_mb"].columns.tolist()
            metrics = pivot_df.index.tolist()
            x_pos = np.arange(len(metrics))
            bottom = np.zeros(len(metrics))
            
            # Plot each phase as a stacked bar
            for phase in phases:
                values = pivot_df["cpu_ram_mb"][phase].values
                ax.bar(x_pos, values, bottom=bottom, label=phase)
                bottom += values
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('CPU Memory (MB)')
            ax.set_title('Memory Usage by Phase')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            fig.tight_layout()
            self._save_figure(fig, "metric_comparison/memory_by_phase")
        
        # Duration Plot
        if "duration_ms" in pivot_df:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get all phases for duration
            phases = pivot_df["duration_ms"].columns.tolist()
            metrics = pivot_df.index.tolist()
            x_pos = np.arange(len(metrics))
            bottom = np.zeros(len(metrics))
            
            # Plot each phase as a stacked bar
            for phase in phases:
                values = pivot_df["duration_ms"][phase].values
                ax.bar(x_pos, values, bottom=bottom, label=phase)
                bottom += values
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Duration (ms)')
            ax.set_title('Time by Phase')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            fig.tight_layout()
            self._save_figure(fig, "metric_comparison/duration_by_phase")
        
        # If GPU memory was used, create a GPU memory usage plot
        if "gpu_ram_mb" in pivot_df and pivot_df["gpu_ram_mb"].sum().sum() > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get all phases for GPU RAM
            phases = pivot_df["gpu_ram_mb"].columns.tolist()
            metrics = pivot_df.index.tolist()
            x_pos = np.arange(len(metrics))
            bottom = np.zeros(len(metrics))
            
            # Plot each phase as a stacked bar
            for phase in phases:
                values = pivot_df["gpu_ram_mb"][phase].values
                ax.bar(x_pos, values, bottom=bottom, label=phase)
                bottom += values
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('GPU Memory (MB)')
            ax.set_title('GPU Memory Usage by Phase')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            fig.tight_layout()
            self._save_figure(fig, "metric_comparison/gpu_memory_by_phase")
        
        # Create a combined bar chart showing import vs. runtime memory usage
        fig, ax = plt.subplots(figsize=(12, 8))
        metrics = import_df["metric"].unique()
        x_pos = np.arange(len(metrics))
        bar_width = 0.35
        
        # Calculate total import costs vs runtime costs
        import_costs = []
        runtime_costs = []
        
        for metric in metrics:
            # Sum all phases for this metric to get total import memory
            metric_import = import_df[import_df["metric"] == metric]["cpu_ram_mb"].sum()
            import_costs.append(metric_import)
            
            # Get runtime memory from comparison data
            try:
                runtime_data = self.results[f"{metric}/summary"].dataframe
                runtime_mem = runtime_data["mean_cpu_ram_mb"].values[0]
                runtime_costs.append(runtime_mem)
            except (KeyError, IndexError):
                runtime_costs.append(0)
        
        # Plot the bars
        ax.bar(x_pos - bar_width/2, import_costs, bar_width, label='Import & Construction')
        ax.bar(x_pos + bar_width/2, runtime_costs, bar_width, label='Runtime')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('CPU Memory (MB)')
        ax.set_title('Import vs Runtime Memory Usage')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        fig.tight_layout()
        self._save_figure(fig, "metric_comparison/import_vs_runtime")

    def _save_figure(self, fig, result_path):
        """Save a figure as a result and close it to prevent memory leaks."""
        self.results[result_path] = FigureResult(fig)
        plt.close(fig)  # Close the figure to prevent memory leaks


def run_isolated_trials(metric_class_path, constructor_kwargs, category, num_examples, num_burn_in, use_synthetic, seed=42, use_deterministic_examples=False):
    """
    Run trials for a single category in an isolated subprocess.
    
    Args:
        metric_class_path: Fully qualified path to the metric class
        constructor_kwargs: Parameters to pass to the metric constructor
        category: Length category ('short', 'medium', 'long') or 'dataset' for real data
        num_examples: Number of examples to test
        num_burn_in: Number of burn-in examples
        use_synthetic: Whether to use synthetic data
        seed: Random seed
        use_deterministic_examples: Whether to use deterministic text examples
        
    Returns:
        List of resource measurement dictionaries
    """
    import os
    import sys
    import json
    import importlib
    import traceback
    import gc
    import time
    from contextlib import contextmanager
    
    from autometrics.experiments.utilization.resource import snap
    
    # Function to get current memory use
    def get_memory():
        return snap("memory_check")
    
    @contextmanager
    def memory_checkpoint(label):
        """Context manager to track memory usage before and after a block of code."""
        # Force garbage collection before taking measurements
        gc.collect()
        time.sleep(0.1)  # Small delay to allow memory to stabilize
        
        # Take memory measurement before
        mem_before = get_memory()
        print(f"[{label}] Before: CPU RAM = {mem_before['cpu_ram_mb']:.2f} MB, GPU RAM = {mem_before['gpu_ram_mb']:.2f} MB")
        
        # Execute the block
        yield
        
        # Force garbage collection after execution
        gc.collect()
        time.sleep(0.1)  # Small delay to allow memory to stabilize
        
        # Take memory measurement after
        mem_after = get_memory()
        print(f"[{label}] After: CPU RAM = {mem_after['cpu_ram_mb']:.2f} MB, GPU RAM = {mem_after['gpu_ram_mb']:.2f} MB")
        print(f"[{label}] Delta: CPU RAM = {mem_after['cpu_ram_mb'] - mem_before['cpu_ram_mb']:.2f} MB, "
              f"GPU RAM = {mem_after['gpu_ram_mb'] - mem_before['gpu_ram_mb']:.2f} MB")
    
    # Create checkpoint list
    results = []
    
    try:
        # Initial memory state (record as the true baseline)
        baseline_mem = get_memory()
        print(f"Initial baseline memory: CPU RAM = {baseline_mem['cpu_ram_mb']:.2f} MB, GPU RAM = {baseline_mem['gpu_ram_mb']:.2f} MB")
        
        # Import the module
        with memory_checkpoint("module_import"):
            module_path, class_name = metric_class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
        
        # Construct the metric
        with memory_checkpoint("metric_construction"):
            MetricClass = getattr(module, class_name)
            metric = MetricClass(**constructor_kwargs)
        
        # Generate test examples
        with memory_checkpoint("generate_examples"):
            if use_synthetic:
                # Import locally to minimize memory usage in subprocess
                if use_deterministic_examples:
                    from autometrics.experiments.utilization.utilization import generate_deterministic_text
                    
                    test_examples = []
                    for i in range(num_examples + num_burn_in):
                        # Use a different seed for each example, but derived from the main seed
                        example_seed = seed + i
                        test_examples.append(generate_deterministic_text(category, example_seed))
                else:
                    from autometrics.experiments.utilization.utilization import generate_synthetic_text
                    
                    test_examples = []
                    for _ in range(num_examples + num_burn_in):
                        test_examples.append(generate_synthetic_text(category))
            else:
                # For real datasets, we need to access the dataset
                # This is trickier in a subprocess - we could serialize the dataset data
                # For now, we'll just return an error
                return {"error": "Real dataset not supported in isolated mode"}
        
        # Create resource tracker
        from autometrics.experiments.utilization.utilization import ResourceTracker
        
        # Force garbage collection before starting burn-in
        gc.collect()
        
        # Perform burn-in samples first
        with memory_checkpoint("burn_in_phase"):
            for i in range(min(num_burn_in, len(test_examples))):
                input_text, output_text, reference_texts = test_examples[i]
                try:
                    metric.calculate(input_text, output_text, reference_texts)
                except Exception as e:
                    return {"error": f"Error during burn-in: {str(e)}", "traceback": traceback.format_exc()}
        
        # Reset baseline memory after burn-in
        # This ensures we're only measuring incremental memory from the actual trials
        post_burnin_mem = get_memory()
        print(f"Post burn-in memory (new baseline): CPU RAM = {post_burnin_mem['cpu_ram_mb']:.2f} MB, GPU RAM = {post_burnin_mem['gpu_ram_mb']:.2f} MB")
        print(f"Memory increase during burn-in: CPU RAM = {post_burnin_mem['cpu_ram_mb'] - baseline_mem['cpu_ram_mb']:.2f} MB, GPU RAM = {post_burnin_mem['gpu_ram_mb'] - baseline_mem['gpu_ram_mb']:.2f} MB")
        
        # Now run the actual trials
        for i in range(min(num_examples, len(test_examples) - num_burn_in)):
            input_text, output_text, reference_texts = test_examples[i + num_burn_in]
            
            # Force garbage collection between trials for more consistent measurements
            if i > 0:
                gc.collect()
                time.sleep(0.05)  # Small delay to allow memory to stabilize
            
            try:
                tracker = ResourceTracker().start()
                result = metric.calculate(input_text, output_text, reference_texts)
                tracker.stop()
                
                resources = tracker.get_results()
                
                # Use post-burnin memory as the true baseline
                # This gives more consistent measurements of incremental memory usage
                resources['baseline_cpu_ram_mb'] = post_burnin_mem['cpu_ram_mb']
                resources['baseline_gpu_ram_mb'] = post_burnin_mem['gpu_ram_mb']
                resources['total_cpu_ram_mb'] = resources['baseline_cpu_ram_mb'] + resources['cpu_ram_mb']
                resources['total_gpu_ram_mb'] = resources['baseline_gpu_ram_mb'] + resources['gpu_ram_mb']
                
                results.append(resources)
                print(f"Trial {i+1} completed: CPU RAM = {resources['cpu_ram_mb']:.2f} MB, Duration = {resources['duration_milliseconds']:.2f} ms")
            except Exception as e:
                return {"error": f"Error on example {i}: {str(e)}", "traceback": traceback.format_exc()}
        
        # Clean up resources - this is important for accurate memory reporting
        with memory_checkpoint("cleanup"):
            # Try unloading model if the metric has that capability
            if hasattr(metric, '_unload_model') and callable(getattr(metric, '_unload_model')):
                try:
                    metric._unload_model()
                except Exception:
                    pass
            
            # Explicitly delete the metric
            del metric
            del module
            # Force garbage collection
            gc.collect()
        
        # Final memory state
        final_mem = get_memory()
        print(f"Final memory: CPU RAM = {final_mem['cpu_ram_mb']:.2f} MB, GPU RAM = {final_mem['gpu_ram_mb']:.2f} MB")
        print(f"Total memory delta: CPU RAM = {final_mem['cpu_ram_mb'] - baseline_mem['cpu_ram_mb']:.2f} MB, GPU RAM = {final_mem['gpu_ram_mb'] - baseline_mem['gpu_ram_mb']:.2f} MB")
        
        return results
        
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "traceback": traceback.format_exc()}


def main():
    """Example usage of the UtilizationExperiment."""
    from autometrics.metrics.reference_based.BLEU import BLEU
    from autometrics.metrics.reference_based.ROUGE import ROUGE
    from autometrics.metrics.reference_based.LENS import LENS
    
    # Example metrics - including one that supports model loading/unloading
    metrics = [BLEU(), ROUGE()]
    
    # Optionally add a LENS metric if available
    try:
        lens = LENS()
        # LENS has _load_model and _unload_model methods which will be used
        # by the experiment to properly measure GPU memory usage
        metrics.append(lens)
        print("Added LENS metric which supports model unloading/loading")
    except (ImportError, Exception) as e:
        print(f"LENS metric not available: {str(e)}")
    
    print("\n--- Running experiment with synthetic data ---")
    # Create and run the experiment with synthetic data
    # Output will go to outputs/utilization/synthetic
    experiment = UtilizationExperiment(
        name="Metric Utilization Experiment",
        description="Measuring resource usage of metrics on synthetic data",
        metrics=metrics,
        output_dir="outputs/utilization",  # Will be appended with /synthetic
        num_examples=10,  # Fewer examples for demonstration
        num_burn_in=2,
        use_synthetic=True
    )
    
    experiment.run(print_results=True)
    experiment.save_results()
    
    print("\n--- Running experiment with real dataset ---")
    # Example with real dataset
    # Output will go to outputs/utilization
    try:
        from autometrics.dataset.datasets.simplification.simplification import SimpDA
        dataset = SimpDA()
        
        experiment_real = UtilizationExperiment(
            name="Metric Utilization on Real Data",
            description="Measuring resource usage of metrics on real data",
            metrics=metrics,
            output_dir="outputs/utilization", 
            dataset=dataset,
            num_examples=10,
            num_burn_in=2,
            use_synthetic=False
        )
        
        experiment_real.run(print_results=True)
        experiment_real.save_results()
        print("\nExperiments complete. Results saved to:")
        print("  - Synthetic data: outputs/utilization/synthetic")
        print("  - Real data: outputs/utilization/real_data")
    except ImportError:
        print("SimpDA dataset not available for demonstration")
        print("\nSynthetic data experiment complete. Results saved to:")
        print("  - outputs/utilization/synthetic")


if __name__ == "__main__":
    main()

