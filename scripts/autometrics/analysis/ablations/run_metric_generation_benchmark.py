#!/usr/bin/env python3
"""
Metric Generation Benchmark Experiment

This script benchmarks different metric generation approaches by generating metrics
using different generators and evaluating their correlation with human annotations.

The benchmark follows this methodology:
1. Uses persistent train sets for metric generation (training data for generators)
2. Uses persistent validation sets for evaluation (to measure correlation)
3. Generates different numbers of metrics per generator type:
   - LLMJudgeProposer: 10 metrics per trial (5 trials = 50 total)
   - RubricGenerator (Prometheus): 10 metrics per trial (5 trials = 50 total)
   - RubricGenerator (DSPy): 10 metrics per trial (5 trials = 50 total)
   - GEvalJudgeProposer: 10 metrics per trial (5 trials = 50 total)
   - CodeGenerator: 10 metrics per trial (5 trials = 50 total)
   - OptimizedJudgeProposer: 1 metric per trial (5 trials = 5 total)
   - FinetuneGenerator: 1 metric per trial (5 trials = 5 total)
   - LLMJudgeExampleProposer: 1 metric per trial (5 trials = 5 total)

For generators with 10 metrics per trial, correlation results are averaged together.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import json
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from pathlib import Path
import dspy
from collections import defaultdict
import requests
import signal
from datetime import datetime
import litellm

# Add autometrics to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autometrics.dataset.Dataset import Dataset
from autometrics.experiments.correlation.correlation import CorrelationExperiment, correlation_func_from_name

# Generators
from autometrics.generator.LLMJudgeProposer import BasicLLMJudgeProposer
from autometrics.generator.LLMJudgeExampleProposer import LLMJudgeExampleProposer
from autometrics.generator.OptimizedJudgeProposer import OptimizedJudgeProposer
from autometrics.generator.GEvalJudgeProposer import GEvalJudgeProposer
from autometrics.generator.CodeGenerator import CodeGenerator
from autometrics.generator.RubricGenerator import RubricGenerator
from autometrics.generator.FinetuneGenerator import FinetuneGenerator


# Exit codes for different failure types
EXIT_CODE_SERVER_FAILURE = 42
EXIT_CODE_GENERAL_ERROR = 1
EXIT_CODE_SUCCESS = 0


class DetailedErrorInfo:
    """Enhanced error tracking with context preservation."""
    def __init__(self, operation: str, context: dict = None):
        self.operation = operation
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
        
    def log_error(self, logger: logging.Logger, error: Exception, additional_context: dict = None):
        """Log detailed error information with full context and traceback."""
        combined_context = {**self.context}
        if additional_context:
            combined_context.update(additional_context)
            
        logger.error(f"{'='*60}")
        logger.error(f"DETAILED ERROR REPORT")
        logger.error(f"{'='*60}")
        logger.error(f"Operation: {self.operation}")
        logger.error(f"Timestamp: {self.timestamp}")
        logger.error(f"Error Type: {type(error).__name__}")
        logger.error(f"Error Message: {str(error)}")
        logger.error(f"Context: {json.dumps(combined_context, indent=2, default=str)}")
        logger.error(f"Full Traceback:")
        logger.error(traceback.format_exc())
        logger.error(f"{'='*60}")
        
    def create_error_dict(self, error: Exception, additional_context: dict = None) -> dict:
        """Create a structured error dictionary for storage/analysis."""
        combined_context = {**self.context}
        if additional_context:
            combined_context.update(additional_context)
            
        return {
            'operation': self.operation,
            'timestamp': self.timestamp,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': combined_context,
            'traceback': traceback.format_exc()
        }


def enhanced_exception_handler(operation: str, context: dict = None):
    """Decorator for enhanced exception handling with detailed logging."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            error_info = DetailedErrorInfo(operation, context)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get logger from args if available
                logger = None
                for arg in args:
                    if hasattr(arg, '__class__') and 'logger' in arg.__class__.__name__.lower():
                        logger = arg
                        break
                if not logger:
                    logger = logging.getLogger(__name__)
                
                error_info.log_error(logger, e)
                raise  # Re-raise the original exception
        return wrapper
    return decorator


# Global tracking for server failures
class ServerFailureTracker:
    def __init__(self, failure_threshold: int = 5, time_window: int = 300):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.failure_times = []
        self.time_window = time_window  # 5 minutes
        self.last_health_check = 0
        
    def record_failure(self):
        """Record a server failure and check if we should give up."""
        current_time = time.time()
        self.failure_times.append(current_time)
        
        # Clean old failures outside time window
        self.failure_times = [t for t in self.failure_times if current_time - t <= self.time_window]
        
        self.failure_count = len(self.failure_times)
        return self.failure_count >= self.failure_threshold
    
    def reset(self):
        """Reset failure tracking after successful operations."""
        self.failure_count = 0
        self.failure_times = []

# Global tracker instance
server_tracker = ServerFailureTracker()


def is_server_failure_error(error_str: str) -> bool:
    """Check if an error indicates server failure."""
    server_failure_indicators = [
        "InternalServerError: Litellm_proxyException",
        "Connection error",
        "ConnectionError",
        "Connection refused",
        "connection refused",
        "server.*down",
        "timeout",
        "503",
        "502", 
        "500",
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectionResetError",
        "RemoteDisconnected"
    ]
    
    error_lower = error_str.lower()
    return any(indicator.lower() in error_lower for indicator in server_failure_indicators)


def check_server_health(api_base: str, timeout: int = 10) -> bool:
    """Check if the server is healthy by making a health check request."""
    if not api_base:
        return True  # Can't check health for OpenAI API
        
    try:
        # Try multiple health check endpoints
        health_endpoints = [
            f"{api_base.rstrip('/')}/models",
            f"{api_base.rstrip('/')}/health",
            f"{api_base.rstrip('/')}/v1/models",
            f"{api_base.rstrip('/')}/v1/health"
        ]
        
        for endpoint in health_endpoints:
            try:
                response = requests.get(endpoint, timeout=timeout)
                if response.status_code == 200:
                    logging.info(f"✅ Server health check passed: {endpoint}")
                    return True
            except Exception as e:
                logging.debug(f"Health check failed for {endpoint}: {e}")
                continue
        
        logging.warning(f"❌ All server health check endpoints failed for {api_base}")
        return False
    except Exception as e:
        logging.warning(f"❌ Server health check failed: {e}")
        return False


def log_server_status(logger: logging.Logger, api_base: str, operation: str = "Unknown"):
    """Log detailed server status information."""
    if not api_base:
        logger.info(f"🔍 Server Status Check ({operation}): No API base - skipping health check")
        return True
    
    logger.info(f"🔍 Server Status Check ({operation}): {api_base}")
    
    # Check server health
    is_healthy = check_server_health(api_base)
    
    if is_healthy:
        logger.info(f"✅ Server is healthy and responding")
        
        # Try to get additional server info
        try:
            models_url = f"{api_base.rstrip('/')}/models"
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                if 'data' in models_data:
                    model_names = [model.get('id', 'unknown') for model in models_data['data']]
                    logger.info(f"📋 Available models: {model_names}")
                else:
                    logger.info(f"📋 Server responded but no model data found")
            else:
                logger.warning(f"⚠️  Server responded with status {response.status_code}")
        except Exception as e:
            logger.debug(f"Could not get detailed server info: {e}")
    else:
        logger.error(f"❌ Server is NOT responding - this may cause metric generation failures!")
        
        # Try to get more diagnostic info
        try:
            # Try a simple ping to see if server is reachable at all
            import socket
            from urllib.parse import urlparse
            
            parsed_url = urlparse(api_base)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.warning(f"⚠️  Server port {port} is reachable but API endpoints are not responding")
                logger.warning(f"   This suggests the server may be running but the API is down")
            else:
                logger.error(f"❌ Server port {port} is not reachable - server may be completely down")
                
        except Exception as e:
            logger.error(f"❌ Could not diagnose server connectivity: {e}")
    
    return is_healthy


def handle_server_failure(logger: logging.Logger, api_base: str):
    """Handle server failure by manual requeue then exit."""
    logger.error("=" * 60)
    logger.error("CRITICAL: Server failure detected!")
    logger.error(f"Server failure threshold exceeded ({server_tracker.failure_count} failures)")
    logger.error(f"API Base: {api_base}")
    logger.error(f"Time: {datetime.now().isoformat()}")
    logger.error("=" * 60)
    
    # Try to check server health one more time
    if api_base and not check_server_health(api_base):
        logger.error("Server health check failed - server appears to be down")
    else:
        logger.warning("Server health check passed - may be intermittent issues")
    
    # Manually requeue the job since --requeue doesn't work with exit codes
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        logger.info(f"Attempting to requeue job {slurm_job_id} due to server failure")
        try:
            import subprocess
            result = subprocess.run(['scontrol', 'requeue', slurm_job_id], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info(f"Successfully requeued job {slurm_job_id}")
            else:
                logger.error(f"Failed to requeue job {slurm_job_id}: {result.stderr}")
        except Exception as e:
            logger.error(f"Error requeuing job {slurm_job_id}: {e}")
    else:
        logger.warning("SLURM_JOB_ID not found - cannot requeue job")
    
    logger.info("Exiting after requeue attempt")
    sys.exit(EXIT_CODE_SERVER_FAILURE)


def setup_signal_handlers(logger: logging.Logger):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        logger.info("Exiting gracefully - server cleanup will be handled by SLURM job termination")
        sys.exit(EXIT_CODE_GENERAL_ERROR)
    
    # Handle common termination signals
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger(__name__)
    
    # Suppress verbose logging from dependencies when not in verbose mode
    if not verbose:
        # DSPy can be very verbose
        logging.getLogger('dspy').setLevel(logging.WARNING)
        
        # Cache-related logging
        logging.getLogger('diskcache').setLevel(logging.WARNING)
        logging.getLogger('diskcache.core').setLevel(logging.WARNING)
        
        # HTTP and API related
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        
        # Model and ML related
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('tokenizers').setLevel(logging.WARNING)
        
        # Autometrics internals
        logging.getLogger('autometrics').setLevel(logging.WARNING)
        logging.getLogger('autometrics.metrics').setLevel(logging.WARNING)
        logging.getLogger('autometrics.metrics.Metric').setLevel(logging.WARNING)
        logging.getLogger('autometrics.experiments').setLevel(logging.WARNING)
        
        # General noise suppression
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        logging.getLogger('concurrent.futures').setLevel(logging.WARNING)
        
        # Cache key creation logging (various libraries might do this)
        logging.getLogger('cache').setLevel(logging.WARNING)
        logging.getLogger('caching').setLevel(logging.WARNING)
        
        # General third-party library noise
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('plotly').setLevel(logging.WARNING)
        
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
        from autometrics.dataset.datasets.evalgen.evalgen import EvalGen
        if dataset_name == "EvalGenMedical":
            return EvalGen('./autometrics/dataset/datasets/evalgen/medical.csv')
        elif dataset_name == "EvalGenProduct":
            return EvalGen('./autometrics/dataset/datasets/evalgen/product.csv')
    elif dataset_name == "RealHumanEval":
        from autometrics.dataset.datasets.realhumaneval.realhumaneval import RealHumanEval
        return RealHumanEval()
    
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_available_datasets_measures() -> List[Tuple[str, str]]:
    """Get all available dataset-measure combinations for metric generation benchmark."""
    datasets_measures = []
    
    # Datasets with their target measures
    dataset_configs = {
        "HelpSteer": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
        "HelpSteer2": ["helpfulness", "correctness", "coherence", "complexity", "verbosity"],
        "SimpDA": ["fluency", "meaning", "simplicity"], 
        "SimpEval": ["score"],
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
    }
    
    for dataset_name, measures in dataset_configs.items():
        for measure in measures:
            datasets_measures.append((dataset_name, measure))
    
    return datasets_measures


def create_llm_model(model_name: str, api_base: Optional[str] = None, seed: int = 42):
    """Create LLM model instance based on model name with unique cache busting per seed."""
    
    temperature = 0.0001 * seed

    if model_name == "gpt4o_mini":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Please export OPENAI_API_KEY before running with gpt4o_mini.")
        model = dspy.LM("openai/gpt-4o-mini", api_key=api_key, temperature=temperature)
    
    elif model_name == "qwen3_32b":
        # Use provided api_base or default to localhost (for local server)
        base_url = api_base or "http://localhost:7410/v1"
        # FIXED: Use the full API base with /v1 like the working demo
        # The server only responds to /v1 endpoints, not root endpoints
        if not base_url.endswith('/v1'):
            base_url = f"{base_url.rstrip('/')}/v1"
        # FIXED: Include api_key="None" like the working demo configuration
        model = dspy.LM("litellm_proxy/Qwen/Qwen3-32B", api_base=base_url, api_key="None", temperature=temperature, max_tokens=4096)
    
    elif model_name == "prometheus":
        # Use provided api_base or default to permanent prometheus server
        from prometheus_eval.litellm import LiteLLM
        base_url = api_base or "http://future-hgx-1:7420/v1"
        model = LiteLLM(
            "litellm_proxy/Unbabel/M-Prometheus-14B",
            api_base=base_url
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def create_generator(
    generator_type: str, 
    generator_llm: dspy.LM, 
    judge_llm: dspy.LM, 
    seed: int, 
    model_save_dir: Optional[str] = None,
    api_base: Optional[str] = None
):
    """Create generator instance based on type with seed-specific configuration."""
    
    if generator_type == "llm_judge":
        return BasicLLMJudgeProposer(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm, "seed": seed},
            seed=seed,
        )
    
    elif generator_type == "llm_judge_examples":
        return LLMJudgeExampleProposer(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm, "seed": seed},
            seed=seed,
            max_optimization_samples=100  # Limit for faster processing
        )
    
    elif generator_type == "llm_judge_optimized":
        return OptimizedJudgeProposer(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm, "seed": seed},
            auto_mode="medium",
            num_threads=16,
            eval_function_name='inverse_distance',
            seed=seed
        )
    
    elif generator_type == "geval":
        return GEvalJudgeProposer(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm, "seed": seed},
            seed=seed,
        )
    
    elif generator_type == "codegen":
        return CodeGenerator(
            generator_llm=generator_llm,
            executor_kwargs={"seed": seed},
            seed=seed,
        )
    
    elif generator_type == "rubric_prometheus":
        judge_llm_prometheus = create_llm_model("prometheus", api_base="http://future-hgx-1:7420/v1")
        return RubricGenerator(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm_prometheus, "seed": seed},
            use_prometheus=True,
            seed=seed,
        )
    
    elif generator_type == "rubric_dspy":
        return RubricGenerator(
            generator_llm=generator_llm,
            executor_kwargs={"model": judge_llm, "seed": seed},
            use_prometheus=False,
            seed=seed,
        )
    
    elif generator_type == "finetune":
        return FinetuneGenerator(
            generator_llm=generator_llm,
            model_save_dir=model_save_dir,
            seed=seed
        )
    
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def save_generated_metrics(metrics: List, generator_type: str, dataset_name: str, measure: str, seed: int, output_dir: str):
    """Save generated metrics to organized directory structure."""
    
    # Create output directory structure
    generator_dir = Path(output_dir) / "generated_metrics" / generator_type / f"seed_{seed}" / dataset_name
    generator_dir.mkdir(parents=True, exist_ok=True)
    
    metric_paths = []
    
    for i, metric in enumerate(metrics):
        # Create clean filename
        safe_dataset_name = dataset_name.replace(" ", "_").replace("/", "_")
        safe_measure_name = measure.replace(" ", "_").replace("/", "_")
        metric_filename = f"{safe_dataset_name}_{safe_measure_name}_{generator_type}_seed{seed}_metric{i+1:02d}.py"
        
        metric_path = generator_dir / metric_filename
        
        # Save metric as standalone Python file
        metric.save_python_code(str(metric_path))
        metric_paths.append(str(metric_path))
    
    return metric_paths


def run_correlation_evaluation(
    metrics: List, 
    val_dataset: Dataset, 
    measure: str, 
    correlation_funcs: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, List[float]]:
    """Run correlation evaluation for a list of metrics on validation dataset using all correlation functions."""
    
    # Results dictionary: correlation_func_name -> list of correlations for each metric
    all_correlations = {}
    
    logger.debug(f"Starting correlation evaluation with {len(metrics)} metrics")
    logger.debug(f"Metrics: {[m.name if m else 'None' for m in metrics]}")
    
    # Filter out None metrics
    valid_metrics = [m for m in metrics if m is not None]
    if len(valid_metrics) != len(metrics):
        logger.warning(f"Filtered out {len(metrics) - len(valid_metrics)} None metrics")
    
    if not valid_metrics:
        logger.error("No valid metrics to evaluate")
        # Return empty results for all correlation functions
        return {corr_name: [] for corr_name in correlation_funcs}
    
    for i, metric in enumerate(valid_metrics):
        try:
            logger.debug(f"Processing metric {i+1}/{len(valid_metrics)}: {metric.name}")
            
            # Run correlation experiment for single metric with ALL correlation functions
            experiment = CorrelationExperiment(
                name=f"MetricGen Eval - {metric.name}",
                description=f"Evaluating generated metric: {metric.name}",
                metrics=[metric],
                output_dir=f"/tmp/metricgen_{hash(metric.name)}",
                dataset=val_dataset,
                correlation_funcs=correlation_funcs,
                should_split=False
            )
            
            logger.debug(f"Running correlation experiment for {metric.name}")
            results = experiment.run(print_results=False)
            logger.debug(f"Experiment results keys: {list(results.keys())}")
            
            # Extract correlations for each correlation function
            for corr_name, correlations_for_func in results.items():
                if corr_name not in all_correlations:
                    all_correlations[corr_name] = []
                
                logger.debug(f"Processing {corr_name} results for {metric.name}")
                logger.debug(f"Available measures in results: {list(correlations_for_func.keys())}")
                
                if measure not in correlations_for_func:
                    logger.error(f"Measure {measure} not found in {corr_name} results")
                    all_correlations[corr_name].append(np.nan)
                    continue
                
                df_corr = correlations_for_func[measure]
                logger.debug(f"Correlation DataFrame shape: {df_corr.shape}")
                logger.debug(f"Available metrics in DataFrame: {list(df_corr['Metric'].values)}")
                
                # Find the correlation for our specific metric
                metric_row = df_corr[df_corr['Metric'] == metric.name]
                
                if metric_row.empty:
                    logger.error(f"Metric {metric.name} not found in correlation results for {corr_name}")
                    all_correlations[corr_name].append(np.nan)
                else:
                    correlation = metric_row.iloc[0]['Correlation']
                    logger.debug(f"Found correlation for {metric.name} ({corr_name}): {correlation}")
                    all_correlations[corr_name].append(correlation)
                
        except Exception as e:
            # Enhanced error logging for individual metric evaluation
            metric_error_info = DetailedErrorInfo(
                operation="single_metric_correlation",
                context={
                    'metric_name': metric.name if metric else 'Unknown',
                    'metric_index': i,
                    'correlation_funcs': list(correlation_funcs.keys()),
                    'measure': measure,
                    'val_dataset_size': len(val_dataset.get_dataframe()) if val_dataset else 'unknown'
                }
            )
            metric_error_info.log_error(logger, e, {
                'operation_details': 'individual_metric_correlation_phase',
                'metric_type': type(metric).__name__ if metric else 'None'
            })
            
            # Add NaN for all correlation functions for this metric
            for corr_name in correlation_funcs:
                if corr_name not in all_correlations:
                    all_correlations[corr_name] = []
                all_correlations[corr_name].append(np.nan)
    
    logger.debug(f"Final correlation results: {all_correlations}")
    return all_correlations


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


def save_results(results: List[Dict], output_file: str, logger: logging.Logger):
    """Save results to CSV file with properly sorted columns."""
    try:
        if not results:
            logger.warning(f"No results to save to {output_file}")
            return
            
        df = pd.DataFrame(results)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {e}")
        raise


def load_existing_results(output_file: str) -> List[Dict]:
    """Load existing results from output file if it exists."""
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            return df.to_dict('records')
        except Exception as e:
            logging.warning(f"Could not read existing results file {output_file}: {e}")
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark metric generation approaches across multiple seeds and datasets"
    )
    
    parser.add_argument(
        "--generator-model",
        default="gpt4o_mini",
        choices=["gpt4o_mini", "qwen3_32b"],
        help="LLM model to use for metric generation"
    )
    parser.add_argument(
        "--judge-model",
        default="gpt4o_mini",
        choices=["gpt4o_mini", "qwen3_32b", "prometheus"],
        help="LLM model to use for judging/evaluation"
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL for the model (e.g., http://localhost:7410/v1)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/ablations/metric_generation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model-save-dir",
        default="/sphinx/u/salt-checkpoints/autometrics/models",
        help="Directory to save fine-tuned models"
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
        "--generator",
        nargs="*",
        choices=["llm_judge", "llm_judge_examples", "llm_judge_optimized", "geval", 
                "codegen", "rubric_prometheus", "rubric_dspy", "finetune"],
        help="Filter to specific generator types"
    )
    parser.add_argument(
        "--skip-generators",
        nargs="*",
        choices=["llm_judge", "llm_judge_examples", "llm_judge_optimized", "geval", 
                "codegen", "rubric_prometheus", "rubric_dspy", "finetune"],
        help="Filter to specific generator types to skip"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    parser.add_argument(
        "--per-measure-files",
        action="store_true",
        help="If set, write per-measure output files for HelpSteer/HelpSteer2 (for parallel safety). Backwards compatible."
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers(logger)
    
    logger.info("Starting Metric Generation Benchmark")
    logger.info(f"Generator Model: {args.generator_model}")
    logger.info(f"Judge Model: {args.judge_model}")
    logger.info(f"API Base: {args.api_base}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Correlation: {args.correlation}")
    
    # Initial server health check if using local server
    if args.api_base and "localhost" in args.api_base:
        logger.info("Performing initial server health check...")
        log_server_status(logger, args.api_base, "Initial Startup")
        if not check_server_health(args.api_base):
            logger.error(f"Initial server health check failed for {args.api_base}")
            logger.error("Please ensure the server is running before starting the benchmark")
            return EXIT_CODE_SERVER_FAILURE
        else:
            logger.info("Initial server health check passed")

    output_dir = f"{args.output_dir}/{args.generator_model}"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/sub_results", exist_ok=True)
    
    # Get correlation functions
    if args.correlation.lower() == "all":
        correlation_specs = ["kendall", "pearson", "spearman"]
    else:
        correlation_specs = [c.strip() for c in args.correlation.split(",")]
    
    correlation_funcs = {}
    for spec in correlation_specs:
        correlation_funcs[spec] = correlation_func_from_name(spec)
    
    print(f"Using correlation functions: {list(correlation_funcs.keys())}")
    
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
    
    # Define generator configurations
    generator_configs = {
        "llm_judge": {"metrics_per_trial": 10, "description": "Basic LLM Judge"},
        "rubric_prometheus": {"metrics_per_trial": 10, "description": "Rubric Generator (Prometheus)"},
        "rubric_dspy": {"metrics_per_trial": 10, "description": "Rubric Generator (DSPy)"},
        "geval": {"metrics_per_trial": 10, "description": "G-Eval"},
        "codegen": {"metrics_per_trial": 10, "description": "Code Generation"},
        "llm_judge_optimized": {"metrics_per_trial": 1, "description": "LLM Judge (MIPROv2-Optimized)"},
        "finetune": {"metrics_per_trial": 1, "description": "Fine-tuned ModernBERT"},
        "llm_judge_examples": {"metrics_per_trial": 1, "description": "LLM Judge (Example-Based)"},
    }
    
    # Filter generators if specified
    if args.generator:
        generator_configs = {k: v for k, v in generator_configs.items() if k in args.generator}
        logger.info(f"Filtered to generators: {args.generator}")
    
    # Filter generators if specified
    if args.skip_generators:
        generator_configs = {k: v for k, v in generator_configs.items() if k not in args.skip_generators}
        logger.info(f"Skipped generators: {args.skip_generators}")
    
    # Patch: determine if we want per-measure output files for HelpSteer/HelpSteer2
    def get_dataset_output_file(dataset_name, measure, correlation_type):
        if args.per_measure_files and dataset_name in ("HelpSteer", "HelpSteer2"):
            return f"{output_dir}/sub_results/metric_generation_benchmark_{args.generator_model}_{args.judge_model}_{correlation_type}_{dataset_name}_{measure}.csv"
        else:
            return f"{output_dir}/sub_results/metric_generation_benchmark_{args.generator_model}_{args.judge_model}_{correlation_type}_{dataset_name}.csv"
    
    # Process each correlation type
    for correlation_type in correlation_funcs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {correlation_type.upper()} correlation analysis")
        logger.info(f"{'='*60}")
        
        # Get correlation function
        correlation_func = correlation_funcs[correlation_type]
        single_correlation_funcs = {correlation_type: correlation_func}
        
        # All results across datasets for final merging
        all_results_for_correlation = []
        dataset_output_files = []
        
        # Load existing merged results for this correlation function
        output_base = f"{output_dir}/metric_generation_benchmark_{args.generator_model}_{args.judge_model}"
        merged_output_file = f"{output_base}_{correlation_type}.csv"
        existing_merged_results = load_existing_results(merged_output_file)
        logger.info(f"Loaded {len(existing_merged_results)} existing {correlation_type} results from {merged_output_file}")

        # Group dataset-measure combinations for dataset-specific files
        dataset_measures_by_dataset = {}
        for dataset_name, measure in all_dataset_measures:
            if dataset_name not in dataset_measures_by_dataset:
                dataset_measures_by_dataset[dataset_name] = []
            dataset_measures_by_dataset[dataset_name].append((dataset_name, measure))

        # Process each dataset separately
        for dataset_name, dataset_measures in dataset_measures_by_dataset.items():
            logger.info(f"\n--- Processing dataset: {dataset_name} ---")
            
            # Periodic server health check before each dataset
            if args.api_base and "localhost" in args.api_base:
                logger.info(f"🔍 Pre-dataset server health check for {dataset_name}...")
                log_server_status(logger, args.api_base, f"Before {dataset_name}")
                if not check_server_health(args.api_base):
                    logger.warning(f"Server health check failed before processing {dataset_name}")
                    if server_tracker.record_failure():
                        handle_server_failure(logger, args.api_base)
            
            # For per-measure files, process each measure separately
            for dataset_name, measure in dataset_measures:
                dataset_output_file = get_dataset_output_file(dataset_name, measure, correlation_type)
                dataset_output_files.append(dataset_output_file)
                logger.info(f"Dataset output file: {dataset_output_file}")
            
            # Load existing dataset-specific results
            existing_dataset_results = load_existing_results(dataset_output_file)
            dataset_results = existing_dataset_results.copy()
            logger.info(f"Loaded {len(existing_dataset_results)} existing dataset results from {dataset_output_file}")

            # Process each dataset-measure combination for this dataset
            for dataset_name, measure in dataset_measures:
                logger.info(f"\nProcessing: {dataset_name} - {measure}")
                
                # Enhanced dataset loading with detailed error tracking
                dataset_error_info = DetailedErrorInfo(
                    operation="dataset_loading",
                    context={
                        'dataset_name': dataset_name,
                        'measure': measure,
                        'correlation_type': correlation_type
                    }
                )
                
                try:
                    # Load dataset
                    dataset = load_dataset(dataset_name)
                    task_description = dataset.get_task_description()
                    
                    logger.info(f"Dataset loaded: {dataset_name}")
                    logger.info(f"Task: {task_description}")
                    logger.info(f"Target measure: {measure}")
                    
                    # Process each generator type
                    for generator_type, config in generator_configs.items():
                        logger.info(f"\n--- Generator: {config['description']} ---")
                        
                        # Server health check before each generator (especially important for long-running ones)
                        if args.api_base and "localhost" in args.api_base:
                            logger.info(f"🔍 Pre-generator server health check for {generator_type}...")
                            log_server_status(logger, args.api_base, f"Before {generator_type}")
                            
                            # For long-running generators, be more strict about server health
                            long_running_generators = ["llm_judge_examples", "llm_judge_optimized", "finetune"]
                            if generator_type in long_running_generators:
                                if not check_server_health(args.api_base):
                                    logger.error(f"❌ Server health check failed before long-running generator {generator_type}")
                                    logger.error(f"   Skipping {generator_type} to avoid wasting time on failing server")
                                    continue
                        
                        metrics_per_trial = config["metrics_per_trial"]
                        
                        # Check existing results for this dataset-measure-generator combination
                        existing_result = None
                        for result in dataset_results:
                            if (result.get('dataset') == dataset_name and 
                                result.get('measure') == measure and 
                                result.get('generator_type') == generator_type):
                                existing_result = result
                                break
                        
                        # Collect seed results
                        seed_correlations = []
                        seed_errors = []
                        
                        for seed in args.seeds:
                            # Check if we already have this seed's result
                            seed_key = f'seed_{seed}_avg_correlation'
                            
                            if existing_result and seed_key in existing_result and not pd.isna(existing_result[seed_key]):
                                seed_correlations.append(existing_result[seed_key])
                                logger.info(f"  Seed {seed}: Using cached result")
                                continue
                            
                            logger.info(f"  Running seed {seed}...")
                            
                            try:
                                # Create models with seed-specific temperature
                                generator_llm = create_llm_model(args.generator_model, args.api_base, seed)
                                judge_llm = create_llm_model(args.judge_model, args.api_base, seed)
                                
                                # Create generator
                                generator = create_generator(
                                    generator_type, 
                                    generator_llm, 
                                    judge_llm, 
                                    seed,
                                    args.model_save_dir,
                                    args.api_base
                                )
                                
                                # Get persistent splits - use TRAIN for generation, VAL for evaluation
                                train_dataset, val_dataset, _ = dataset.load_permanent_splits()
                                logger.info(f"    Using persistent splits - Train: {len(train_dataset.get_dataframe())} examples, Val: {len(val_dataset.get_dataframe())} examples")
                                
                                # Generate metrics using TRAINING data
                                logger.info(f"    Generating {config['metrics_per_trial']} metrics using training split...")
                                # Enhanced metric generation with detailed error tracking
                                error_info = DetailedErrorInfo(
                                    operation="metric_generation",
                                    context={
                                        'generator_type': generator_type,
                                        'dataset_name': dataset_name,
                                        'measure': measure,
                                        'seed': seed,
                                        'metrics_per_trial': config['metrics_per_trial']
                                    }
                                )
                                
                                try:
                                    logger.debug(f"    Creating generator: {generator_type}")
                                    logger.debug(f"    Generator config: {config}")
                                    logger.debug(f"    Seed being passed: {seed}")
                                    
                                    metrics = generator.generate(
                                        dataset=train_dataset, 
                                        target_measure=measure, 
                                        n_metrics=config['metrics_per_trial']
                                    )
                                    logger.debug(f"    Generated {len(metrics) if metrics else 0} metrics: {[m.name if m else 'None' for m in (metrics or [])]}")
                                    
                                    # Check for None metrics or empty list
                                    if not metrics:
                                        logger.error(f"    No metrics generated for seed {seed}")
                                        raise ValueError("Generator returned empty metrics list")
                                        
                                    none_metrics = [i for i, m in enumerate(metrics) if m is None]
                                    if none_metrics:
                                        logger.error(f"    Found None metrics at indices: {none_metrics}")
                                        raise ValueError(f"Generator produced None metrics at indices: {none_metrics}")
                                        
                                except Exception as e:
                                    # Enhanced error logging with full context
                                    error_info.log_error(logger, e, {
                                        'operation_details': 'metric_generation_phase',
                                        'generator_instance': str(generator),
                                        'train_dataset_size': len(train_dataset.get_dataframe()) if train_dataset else 'unknown'
                                    })
                                    
                                    # Also log the full traceback for immediate visibility
                                    logger.error(f"    Full traceback for metric generation error:")
                                    logger.error(traceback.format_exc())
                                    
                                    # Check if this is a server failure
                                    if is_server_failure_error(str(e)):
                                        logger.warning(f"    Detected server failure error: {str(e)}")
                                        if server_tracker.record_failure():
                                            handle_server_failure(logger, args.api_base)
                                    
                                    # Record this as a seed error and continue
                                    error_msg = f"Seed {seed}: {str(e)}"
                                    seed_errors.append(error_msg)
                                    seed_correlations.append(np.nan)
                                    continue

                                # Evaluate correlations using VALIDATION data
                                logger.info(f"    Evaluating correlations on validation split...")
                                correlation_error_info = DetailedErrorInfo(
                                    operation="correlation_evaluation",
                                    context={
                                        'generator_type': generator_type,
                                        'dataset_name': dataset_name,
                                        'measure': measure,
                                        'seed': seed,
                                        'correlation_type': correlation_type,
                                        'num_metrics': len(metrics) if metrics else 0,
                                        'metric_names': [m.name if m else 'None' for m in (metrics or [])]
                                    }
                                )
                                
                                try:
                                    logger.debug(f"    Metrics for correlation: {[m.name if m else 'None' for m in metrics]}")
                                    logger.debug(f"    Validation dataset size: {len(val_dataset.get_dataframe())}")
                                    logger.debug(f"    Target measure: {measure}")
                                    logger.debug(f"    Correlation function: {correlation_type}")
                                    
                                    all_correlations = run_correlation_evaluation(
                                        metrics, val_dataset, measure, single_correlation_funcs, logger
                                    )
                                    logger.debug(f"    Correlation results: {all_correlations}")
                                    
                                    # Save generated metrics
                                    metric_paths = save_generated_metrics(
                                        metrics, generator_type, dataset_name, measure, seed, output_dir
                                    )
                                    logger.info(f"    Saved metrics to: {len(metric_paths)} files")
                                    for path in metric_paths:
                                        logger.info(f"    {path}")
                                    
                                    # Post-generation server health check
                                    if args.api_base and "localhost" in args.api_base:
                                        logger.info(f"🔍 Post-generation server health check for {generator_type}...")
                                        log_server_status(logger, args.api_base, f"After {generator_type}")
                                    
                                    # Reset failure tracker on successful operation
                                    server_tracker.reset()
                                    
                                except Exception as e:
                                    # Enhanced error logging with full context
                                    correlation_error_info.log_error(logger, e, {
                                        'operation_details': 'correlation_evaluation_phase',
                                        'val_dataset_size': len(val_dataset.get_dataframe()) if val_dataset else 'unknown',
                                        'correlation_funcs': list(single_correlation_funcs.keys()),
                                        'metrics_generated': len(metrics) if metrics else 0
                                    })
                                    
                                    # Also log the full traceback for immediate visibility
                                    logger.error(f"    Full traceback for correlation evaluation error:")
                                    logger.error(traceback.format_exc())
                                    
                                    # Check if this is a server failure
                                    if is_server_failure_error(str(e)):
                                        logger.warning(f"    Detected server failure error in correlation: {str(e)}")
                                        if server_tracker.record_failure():
                                            handle_server_failure(logger, args.api_base)
                                    
                                    # Record this as a seed error and continue
                                    error_msg = f"Seed {seed}: Correlation evaluation failed: {str(e)}"
                                    seed_errors.append(error_msg)
                                    seed_correlations.append(np.nan)
                                    continue
                                
                                # Process results for this correlation function
                                correlations = all_correlations[correlation_type]
                                
                                # For generators with multiple metrics, average the correlations
                                if metrics_per_trial > 1:
                                    valid_correlations = [c for c in correlations if not pd.isna(c)]
                                    if valid_correlations:
                                        avg_correlation = np.mean([abs(c) for c in valid_correlations])
                                        logger.info(f"    Average {correlation_type} correlation: {avg_correlation:.4f} (from {len(valid_correlations)}/{len(correlations)} valid)")
                                    else:
                                        avg_correlation = np.nan
                                        logger.warning(f"    No valid {correlation_type} correlations from {len(correlations)} metrics")
                                else:
                                    avg_correlation = abs(correlations[0]) if correlations and not pd.isna(correlations[0]) else np.nan
                                    logger.info(f"    {correlation_type} correlation: {avg_correlation:.4f}")
                                
                                seed_correlations.append(avg_correlation)
                                
                            except Exception as e:
                                error_msg = f"Seed {seed}: {str(e)}"
                                seed_errors.append(error_msg)
                                logger.error(f"    Error: {error_msg}")
                                logger.error(f"    Full traceback:")
                                logger.error(traceback.format_exc())
                                
                                # Check if this is a server failure
                                if is_server_failure_error(str(e)):
                                    logger.warning(f"    Detected server failure error in seed processing: {str(e)}")
                                    if server_tracker.record_failure():
                                        handle_server_failure(logger, args.api_base)
                                
                                seed_correlations.append(np.nan)
                        
                        # Compute statistics
                        stats_result = compute_statistics(seed_correlations)
                        
                        # Create result record
                        result = {
                            'dataset': dataset_name,
                            'measure': measure,
                            'generator_type': generator_type,
                            'generator_description': config['description'],
                            'metrics_per_trial': metrics_per_trial,
                            'num_successful_runs': stats_result['num_successful_runs'],
                            'errors': '; '.join(seed_errors) if seed_errors else ''
                        }
                        
                        # Add individual seed results
                        for i, seed in enumerate(args.seeds):
                            result[f'seed_{seed}_avg_correlation'] = seed_correlations[i] if i < len(seed_correlations) else np.nan
                        
                        # Add statistics
                        result.update({
                            'mean_correlation': stats_result['mean'],
                            'std_correlation': stats_result['std'],
                            'ci_lower_correlation': stats_result['ci_lower'],
                            'ci_upper_correlation': stats_result['ci_upper'],
                            'mean_±_ci': format_mean_ci(stats_result['mean'], stats_result['ci_range'])
                        })
                        
                        # Update existing results or add new result
                        if existing_result:
                            # Update existing result in dataset_results
                            for i, res in enumerate(dataset_results):
                                if (res.get('dataset') == dataset_name and 
                                    res.get('measure') == measure and 
                                    res.get('generator_type') == generator_type):
                                    dataset_results[i] = result
                                    break
                            logger.info(f"Updated existing result for {generator_type}")
                        else:
                            dataset_results.append(result)
                            logger.info(f"Added new result for {generator_type}")
                        
                        # Save results immediately after each generator completes
                        try:
                            save_results(dataset_results, dataset_output_file, logger)
                            logger.info(f"Saved intermediate results to {dataset_output_file}")
                        except Exception as e:
                            logger.warning(f"Failed to save intermediate results: {e}")
                        
                        logger.info(f"Generator {generator_type} completed: mean={stats_result['mean']:.4f}, "
                                  f"CI=[{stats_result['ci_lower']:.4f}, {stats_result['ci_upper']:.4f}]")
                
                except Exception as e:
                    # Enhanced error logging for dataset processing
                    dataset_error_info.log_error(logger, e, {
                        'operation_details': 'dataset_processing_main_loop',
                        'generators_attempted': list(generator_configs.keys()),
                        'task_description': task_description[:200] + '...' if 'task_description' in locals() and len(task_description) > 200 else getattr(locals().get('task_description'), 'task_description', 'unknown')
                    })
                    continue
            
            # Save dataset-specific results
            if dataset_results:
                save_results(dataset_results, dataset_output_file, logger)
                all_results_for_correlation.extend(dataset_results)  # Add to combined results
                logger.info(f"Dataset {dataset_name} results saved to {dataset_output_file}")
            else:
                logger.warning(f"No results generated for dataset {dataset_name}")
        
        # Save merged results for this correlation function
        if all_results_for_correlation:
            # Patch: merge all per-measure files if present, else fall back to per-dataset file
            merged_results = existing_merged_results.copy()
            per_measure_files = []
            for dataset_name, dataset_measures in dataset_measures_by_dataset.items():
                for _, measure in dataset_measures:
                    pm_file = get_dataset_output_file(dataset_name, measure, correlation_type)
                    if os.path.exists(pm_file):
                        per_measure_files.append(pm_file)
            if per_measure_files:
                logger.info(f"Merging from per-measure files: {per_measure_files}")
                for pm_file in per_measure_files:
                    pm_results = load_existing_results(pm_file)
                    for result in pm_results:
                        found = False
                        for i, existing in enumerate(merged_results):
                            if (existing.get('dataset') == result['dataset'] and 
                                existing.get('measure') == result['measure'] and 
                                existing.get('generator_type') == result['generator_type']):
                                merged_results[i] = result  # Update existing
                                found = True
                                break
                        if not found:
                            merged_results.append(result)  # Add new
                # Warn if old per-dataset file also exists
                for dataset_name in dataset_measures_by_dataset:
                    old_file = get_dataset_output_file(dataset_name, dataset_measures_by_dataset[dataset_name][0][1], correlation_type)
                    if os.path.exists(old_file) and old_file not in per_measure_files:
                        logger.warning(f"Both per-measure and per-dataset files exist for {dataset_name}. Using per-measure files.")
            else:
                logger.info(f"No per-measure files found, using per-dataset files.")
                for dataset_name in dataset_measures_by_dataset:
                    old_file = get_dataset_output_file(dataset_name, dataset_measures_by_dataset[dataset_name][0][1], correlation_type)
                    if os.path.exists(old_file):
                        old_results = load_existing_results(old_file)
                        for result in old_results:
                            found = False
                            for i, existing in enumerate(merged_results):
                                if (existing.get('dataset') == result['dataset'] and 
                                    existing.get('measure') == result['measure'] and 
                                    existing.get('generator_type') == result['generator_type']):
                                    merged_results[i] = result  # Update existing
                                    found = True
                                    break
                            if not found:
                                merged_results.append(result)  # Add new
            save_results(merged_results, merged_output_file, logger)
            
            # Print summary for this correlation type
            logger.info(f"\nSummary of Metric Generation {correlation_type.upper()} Results:")
            logger.info(f"Generator Model: {args.generator_model}")
            logger.info(f"Judge Model: {args.judge_model}")
            logger.info(f"Seeds: {args.seeds}")
            logger.info(f"Merged results saved to: {merged_output_file}")
            logger.info(f"Dataset-specific results saved to:")
            for dataset_file in dataset_output_files:
                logger.info(f"  {dataset_file}")
            
            # Show top performers
            logger.info(f"\nTop 5 performing generator-dataset combinations ({correlation_type}):")
            df_results = pd.DataFrame(merged_results)
            if not df_results.empty and 'mean_correlation' in df_results.columns:
                df_top = df_results.nlargest(5, 'mean_correlation')
                
                for _, row in df_top.iterrows():
                    logger.info(f"  {row['generator_description']} on {row['dataset']}.{row['measure']}: "
                               f"{row['mean_correlation']:.4f} ± {row['ci_upper_correlation'] - row['ci_lower_correlation']:.4f}")
        
        else:
            logger.error(f"No results generated for {correlation_type}")

    logger.info("Metric Generation Benchmark completed successfully!")
    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    # Add custom pricing for Prometheus model (128k context)
    litellm.model_cost["litellm_proxy/Unbabel/M-Prometheus-14B"] = {
        "max_tokens": 128000,
        "input_cost_per_token": 0.0,  # Free since it's our own server
        "output_cost_per_token": 0.0,  # Free since it's our own server
        "max_input_tokens": 120000,
        "max_output_tokens": 8000
    }
    sys.exit(main()) 