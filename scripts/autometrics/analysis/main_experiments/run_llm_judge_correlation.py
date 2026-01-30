#!/usr/bin/env python3
"""
LLM as a Judge Correlation Stability Analysis

This script tests the correlation stability of LLM as a Judge metrics across multiple 
random seeds. It reads LLM judge prompts from a CSV file, creates custom LLM judge 
metrics, and measures their correlation with human annotations across different datasets.

The script follows the same patterns as run_best_static_metric.py but handles:
- Custom score ranges per prompt
- LLM API calls with retry logic
- Different model backends (GPT-4o-mini, Qwen3-32B, GPT-5-mini, etc.)
- Reference-free vs reference-based metrics based on dataset

Example usage:
    python run_llm_judge_correlation.py --model gpt4o_mini --dataset HelpSteer SimpEval
    python run_llm_judge_correlation.py --model qwen3_32b --api-base http://localhost:7410/v1
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import time
import ast
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
from collections import defaultdict
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import uuid

# Add autometrics to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autometrics.dataset.Dataset import Dataset
from autometrics.experiments.correlation.correlation import CorrelationExperiment, correlation_func_from_name
from autometrics.metrics.Metric import Metric
from autometrics.metrics.generated.GeneratedRefFreeMetric import GeneratedRefFreeMetric
from autometrics.metrics.generated.GeneratedRefBasedMetric import GeneratedRefBasedMetric

# LLM imports
import dspy


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
        # Use specific subclasses to preserve task descriptions and clear naming
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
    elif dataset_name == "ICLR":
        from autometrics.dataset.datasets.iclr.iclr import ICLR
        return ICLR()
    
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
        "Design2Code": "design2code",
        "ICLR": "iclr"
    }
    
    cache_subdir = cache_mapping.get(dataset_name, dataset_name.lower())
    cache_dir = f"./.cache/{cache_subdir}"
    os.makedirs(cache_dir, exist_ok=True)
    logging.debug(f"Created/using cache dir at {cache_dir} for dataset {dataset_name}")
    return cache_dir


# Custom DSPy Signatures for LLM Judge with flexible ranges
class LLMJudgeSignatureRefFree(dspy.Signature):
    """Given the task description and evaluation axis, rate the output text."""
    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    axis: str = dspy.InputField(desc="The evaluation axis/rubric.")
    input_text: str = dspy.InputField(desc="The input text.")
    output_text: str = dspy.InputField(desc="The output text to rate.")
    score_range: str = dspy.InputField(desc="The valid range for the score (e.g., '0-4', '1-5').")
    score: float = dspy.OutputField(desc="Numerical score in the specified range.")


class LLMJudgeSignatureRefBased(dspy.Signature):
    """Given the task description and evaluation axis, rate the output text using reference text."""
    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    axis: str = dspy.InputField(desc="The evaluation axis/rubric.")
    input_text: str = dspy.InputField(desc="The input text.")
    reference_text: str = dspy.InputField(desc="The reference text to compare against.")
    output_text: str = dspy.InputField(desc="The output text to rate.")
    score_range: str = dspy.InputField(desc="The valid range for the score (e.g., '0-4', '1-5').")
    score: float = dspy.OutputField(desc="Numerical score in the specified range.")


class CustomLLMJudgeMetricRefFree(GeneratedRefFreeMetric):
    """Custom reference-free LLM as a Judge metric with configurable score ranges and prompts."""
    
    def __init__(
        self, 
        name: str,
        prompt: str, 
        score_range: Tuple[float, float],
        model: dspy.LM,
        task_description: str = "",
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(
            name=name,
            description=f"LLM Judge metric for {name}",
            **kwargs
        )
        self.axis = prompt  # Store as axis to match parent class expectations
        self.prompt = prompt  # Keep for our own use
        self.score_range = score_range
        self.model = model
        self.task_description = task_description
        self.max_retries = max_retries
        self.seed = kwargs.get("seed")
        self.model_name = kwargs.get("model_name")
        
        # Create DSPy module
        self._judge_module = dspy.ChainOfThought(LLMJudgeSignatureRefFree)
        
        # Exclude heavy objects from cache key
        self.exclude_from_cache_key("model", "_judge_module")
    
    def _format_score_range(self) -> str:
        """Format score range for the LLM prompt."""
        min_val, max_val = self.score_range
        if np.isinf(max_val):
            return f"{min_val}-∞"
        else:
            return f"{min_val}-{max_val}"
    
    def _call_llm_with_retry(self, input_text: str, output_text: str) -> float:
        """Call LLM with retry logic and range validation."""
        logging.debug(
            f"[RefFree:{self.name}] Starting LLM call with prompt len={len(self.prompt)} "
            f"input_len={len(str(input_text))} output_len={len(str(output_text))} "
            f"score_range={self.score_range} max_retries={self.max_retries}"
        )
        # Inject a benign cache-busting token for GPT-5
        axis = self.prompt
        task_description = self.task_description
        model_name_lower = (self.model_name or "").lower()
        use_cache_bust = ("gpt5" in model_name_lower) or ("gpt-5" in model_name_lower) or (model_name_lower == "gpt5_mini")
        cache_token = ""
        if use_cache_bust:
            cache_token = f" [cache-id:{self.seed}]" if self.seed is not None else " [cache-id]"
            # Prefer adding to task_description to minimize effect on rubric
            task_description = f"{task_description}\n(Ignore this line; cache identifier){cache_token}"
        logging.debug(f"[RefFree:{self.name}] cache_bust={'ON' if use_cache_bust else 'OFF'} token='{cache_token}' model_name='{self.model_name}'")
        for attempt in range(self.max_retries):
            try:
                # Temporarily suppress verbose logging during inference
                loggers_to_quiet = ['dspy', 'openai', 'httpx', 'httpcore']
                original_levels = {}
                for logger_name in loggers_to_quiet:
                    logger_obj = logging.getLogger(logger_name)
                    original_levels[logger_name] = logger_obj.level
                    logger_obj.setLevel(logging.ERROR)
                
                try:
                    logging.debug(f"[RefFree:{self.name}] Attempt {attempt+1}/{self.max_retries} - invoking model {type(self.model).__name__}")
                    with dspy.settings.context(lm=self.model):
                        result = self._judge_module(
                            task_description=task_description,
                            axis=axis,
                            input_text=input_text,
                            output_text=output_text,
                            score_range=self._format_score_range(),
                        )
                        
                        # Extract and validate score
                        score = float(result.score)
                        clamped = self._validate_score(score)
                        logging.debug(f"[RefFree:{self.name}] Attempt {attempt+1} succeeded with raw score={score} clamped={clamped}")
                        return clamped
                finally:
                    # Restore original logging levels
                    for logger_name, level in original_levels.items():
                        logging.getLogger(logger_name).setLevel(level)
                    
            except Exception as e:
                logging.warning(f"LLM call attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logging.error(f"All {self.max_retries} attempts failed for {self.name}")
                    return np.nan
                time.sleep(2 ** attempt)  # exponential backoff
        
        return np.nan
    
    def _validate_score(self, score: float) -> float:
        """Validate and clamp score to the specified range."""
        min_val, max_val = self.score_range
        if np.isinf(max_val):
            return max(min_val, score)
        return max(min_val, min(max_val, score))
    
    def _calculate_impl(self, input_text: str, output_text: str, references=None, **kwargs) -> float:
        """Calculate metric for a single input-output pair."""
        return self._call_llm_with_retry(input_text, output_text)
    
    def _calculate_batched_impl(self, inputs: List[str], outputs: List[str], references=None, **kwargs) -> List[float]:
        """Calculate metric for a batch of input-output pairs with parallel processing."""
        max_workers = min(8, len(inputs))  # Limit concurrent requests to avoid rate limits
        
        def process_single(pair_idx_input_output):
            idx, input_text, output_text = pair_idx_input_output
            return idx, self._call_llm_with_retry(input_text, output_text)
        
        results = [np.nan] * len(inputs)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single, (i, inp, out)): i 
                for i, (inp, out) in enumerate(zip(inputs, outputs))
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    idx, score = future.result()
                    results[idx] = score
                except Exception as e:
                    logging.warning(f"Failed to process item {futures[future]}: {e}")
                    raise e
        
        return results
    
    def _generate_metric_card(self, author_model=None, **kwargs) -> str:
        """Generate a simple metric card for this LLM judge metric."""
        prompt_text = getattr(self, 'prompt', 'N/A')
        if len(prompt_text) > 100:
            prompt_text = prompt_text[:100] + "..."
        return f"LLM as a Judge metric for {self.name} using {prompt_text}"
    
    def _generate_python_code(self, include_metric_card=True, **kwargs) -> str:
        """Generate Python code to recreate this metric."""
        return f"# Generated metric code for {self.name}\n# This is a placeholder implementation"


class CustomLLMJudgeMetricRefBased(GeneratedRefBasedMetric):
    """Custom reference-based LLM as a Judge metric with configurable score ranges and prompts."""
    
    def __init__(
        self, 
        name: str,
        prompt: str, 
        score_range: Tuple[float, float],
        model: dspy.LM,
        task_description: str = "",
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(
            name=name,
            description=f"LLM Judge metric for {name}",
            **kwargs
        )
        self.axis = prompt  # Store as axis to match parent class expectations
        self.prompt = prompt  # Keep for our own use
        self.score_range = score_range
        self.model = model
        self.task_description = task_description
        self.max_retries = max_retries
        self.seed = kwargs.get("seed")
        self.model_name = kwargs.get("model_name")
        
        # Create DSPy module
        self._judge_module = dspy.ChainOfThought(LLMJudgeSignatureRefBased)
        
        # Exclude heavy objects from cache key
        self.exclude_from_cache_key("model", "_judge_module")
    
    def _format_score_range(self) -> str:
        """Format score range for the LLM prompt."""
        min_val, max_val = self.score_range
        if np.isinf(max_val):
            return f"{min_val}-∞"
        else:
            return f"{min_val}-{max_val}"
    
    def _call_llm_with_retry(self, input_text: str, output_text: str, references: Optional[List[str]] = None) -> float:
        """Call LLM with retry logic and range validation."""
        logging.debug(
            f"[RefBased:{self.name}] Starting LLM call with prompt len={len(self.prompt)} "
            f"input_len={len(str(input_text))} output_len={len(str(output_text))} "
            f"ref_len={(len(references[0]) if references and len(references)>0 and references[0] is not None else 0)} "
            f"score_range={self.score_range} max_retries={self.max_retries}"
        )
        # Inject a benign cache-busting token for GPT-5
        model_name_lower = (self.model_name or "").lower()
        use_cache_bust = ("gpt5" in model_name_lower) or ("gpt-5" in model_name_lower) or (model_name_lower == "gpt5_mini")
        logging.debug(f"[RefBased:{self.name}] cache_bust={'ON' if use_cache_bust else 'OFF'} model_name='{self.model_name}'")
        for attempt in range(self.max_retries):
            try:
                # Temporarily suppress verbose logging during inference
                loggers_to_quiet = ['dspy', 'openai', 'httpx', 'httpcore']
                original_levels = {}
                for logger_name in loggers_to_quiet:
                    logger_obj = logging.getLogger(logger_name)
                    original_levels[logger_name] = logger_obj.level
                    logger_obj.setLevel(logging.ERROR)
                
                try:
                    logging.debug(f"[RefBased:{self.name}] Attempt {attempt+1}/{self.max_retries} - invoking model {type(self.model).__name__}")
                    with dspy.settings.context(lm=self.model):
                        # Use first reference if multiple are provided
                        reference_text = references[0] if references and len(references) > 0 else ""
                        if use_cache_bust:
                            cache_token = f" [cache-id:{self.seed}]" if self.seed is not None else " [cache-id]"
                            reference_text = f"{reference_text}\n(Ignore this line; cache identifier){cache_token}"
                            logging.debug(f"[RefBased:{self.name}] using cache token '{cache_token}'")
                        
                        result = self._judge_module(
                            task_description=self.task_description,
                            axis=self.prompt,
                            input_text=input_text,
                            reference_text=reference_text,
                            output_text=output_text,
                            score_range=self._format_score_range(),
                        )
                        
                        # Extract and validate score
                        score = float(result.score)
                        clamped = self._validate_score(score)
                        logging.debug(f"[RefBased:{self.name}] Attempt {attempt+1} succeeded with raw score={score} clamped={clamped}")
                        return clamped
                finally:
                    # Restore original logging levels
                    for logger_name, level in original_levels.items():
                        logging.getLogger(logger_name).setLevel(level)
                    
            except Exception as e:
                logging.warning(f"LLM call attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logging.error(f"All {self.max_retries} attempts failed for {self.name}")
                    return np.nan
                time.sleep(2 ** attempt)  # exponential backoff
        
        return np.nan
    
    def _validate_score(self, score: float) -> float:
        """Validate and clamp score to the specified range."""
        min_val, max_val = self.score_range
        if np.isinf(max_val):
            return max(min_val, score)
        return max(min_val, min(max_val, score))
    
    def _calculate_impl(self, input_text: str, output_text: str, references=None, **kwargs) -> float:
        """Calculate metric for a single input-output pair."""
        return self._call_llm_with_retry(input_text, output_text, references)
    
    def _calculate_batched_impl(self, inputs: List[str], outputs: List[str], references=None, **kwargs) -> List[float]:
        """Calculate metric for a batch of input-output pairs with parallel processing."""
        max_workers = min(8, len(inputs))  # Limit concurrent requests to avoid rate limits
        
        def process_single(pair_idx_input_output_ref):
            idx, input_text, output_text, ref_for_this_item = pair_idx_input_output_ref
            return idx, self._call_llm_with_retry(input_text, output_text, ref_for_this_item)
        
        results = [np.nan] * len(inputs)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {}
            for i, (inp, out) in enumerate(zip(inputs, outputs)):
                ref_for_this_item = references[i] if references and i < len(references) else None
                future = executor.submit(process_single, (i, inp, out, ref_for_this_item))
                futures[future] = i
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    idx, score = future.result()
                    results[idx] = score
                except Exception as e:
                    logging.warning(f"Failed to process item {futures[future]}: {e}")
                    raise e
        
        return results
    
    def _generate_metric_card(self, author_model=None, **kwargs) -> str:
        """Generate a simple metric card for this LLM judge metric."""
        prompt_text = getattr(self, 'prompt', 'N/A')
        if len(prompt_text) > 100:
            prompt_text = prompt_text[:100] + "..."
        return f"LLM as a Judge metric for {self.name} using {prompt_text}"
    
    def _generate_python_code(self, include_metric_card=True, **kwargs) -> str:
        """Generate Python code to recreate this metric."""
        return f"# Generated metric code for {self.name}\n# This is a placeholder implementation"


def parse_score_range(range_str: str) -> Tuple[float, float]:
    """Parse score range string into tuple of floats.
    
    Examples:
        "[0,4]" -> (0.0, 4.0)
        "[0,inf)" -> (0.0, float('inf'))
        "(1,5]" -> (1.0, 5.0)
    """
    try:
        logging.debug(f"Parsing score range from string: {range_str}")
        # Clean up the string - remove any brackets/parentheses and split on comma
        cleaned = range_str.strip()
        for char in ['[', ']', '(', ')']:
            cleaned = cleaned.replace(char, '')
        
        # Split on comma
        parts = cleaned.split(',')
        if len(parts) != 2:
            raise ValueError(f"Expected two parts separated by comma")
        
        # Parse each part
        min_val = float(parts[0].strip())
        max_part = parts[1].strip()
        
        if max_part.lower() == 'inf':
            max_val = float('inf')
        else:
            max_val = float(max_part)
        
        logging.debug(f"Parsed score range: ({min_val}, {max_val})")
        return (min_val, max_val)
        
    except Exception as e:
        logging.error(f"Failed to parse score range '{range_str}': {e}")
        return (0.0, 5.0)  # Default fallback


def load_llm_judge_prompts(csv_file: str) -> Dict[str, Dict[str, Any]]:
    """Load LLM judge prompts from CSV file.
    
    Returns:
        Dict mapping "dataset_measure" to prompt info
    """
    df = pd.read_csv(csv_file)
    logging.debug(f"Loaded prompts CSV '{csv_file}' with shape {df.shape} and columns {list(df.columns)}")
    prompts = {}
    
    for _, row in df.iterrows():
        key = f"{row['dataset']}_{row['measure']}"
        logging.debug(f"Processing prompt row for key={key} dataset={row['dataset']} measure={row['measure']} score_range={row['score_range']}")
        prompts[key] = {
            'dataset': row['dataset'],
            'measure': row['measure'], 
            'prompt': row['prompt'],
            'score_range': parse_score_range(row['score_range'])
        }
    
    logging.info(f"Loaded {len(prompts)} LLM judge prompts from {csv_file}")
    try:
        datasets_list = sorted(set(df['dataset'].tolist()))
        measures_list = sorted(set(df['measure'].tolist()))
        logging.debug(f"Prompts cover datasets={datasets_list} measures={measures_list}")
    except Exception:
        pass
    return prompts


def create_llm_model(model_name: str, api_base: Optional[str] = None, seed: int = 42) -> dspy.LM:
    """Create LLM model instance based on model name with unique cache busting per seed."""
    
    temperature = 0.0001 * seed
    logging.debug(f"Creating LLM model name={model_name} seed={seed} temperature={temperature} api_base={api_base}")

    if model_name == "gpt4o_mini":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Please export OPENAI_API_KEY before running with gpt4o_mini.")
        model = dspy.LM("openai/gpt-4o-mini", api_key=api_key, temperature=temperature)

    elif model_name == "gpt5_mini":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Please export OPENAI_API_KEY before running with gpt5mini.")
        model = dspy.LM("openai/gpt-5-mini", api_key=api_key, temperature=temperature)
    
    elif model_name == "qwen3_32b":
        # Use provided api_base or default to localhost (for local server)
        base_url = api_base or "http://localhost:7410/v1"
        logging.debug(f"Using base_url={base_url} for qwen3_32b")
        model = dspy.LM("litellm_proxy/Qwen/Qwen3-32B", api_base=base_url, temperature=temperature, max_tokens=4096) # Raise the max_tokens for Qwen since it's a reasoning model
    
    elif model_name == "llama3_70b":
        # Use provided api_base or default
        base_url = api_base or "http://localhost:7410/v1"
        logging.debug(f"Using base_url={base_url} for llama3_70b")
        model = dspy.LM("litellm_proxy/meta-llama/Llama-3.3-70B-Instruct", api_base=base_url, temperature=temperature)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

    logging.debug(f"Created model object of type {type(model).__name__} for {model_name}")
    return model


def determine_metric_class(dataset: Dataset) -> type:
    """Determine whether to use reference-based or reference-free metrics based on dataset."""
    reference_columns = dataset.get_reference_columns()
    has_references = reference_columns is not None and len(reference_columns) > 0
    
    if has_references:
        logging.debug(f"Dataset has references ({len(reference_columns)}). Using CustomLLMJudgeMetricRefBased")
        return CustomLLMJudgeMetricRefBased
    else:
        logging.debug("Dataset has no references. Using CustomLLMJudgeMetricRefFree")
        return CustomLLMJudgeMetricRefFree


def run_single_metric_seed(
    dataset_name: str,
    measure: str, 
    metric_instance: Metric,
    seed: int,
    correlation_funcs: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Dict[str, float]]:
    """
    Run a single metric on a single seed and return correlations and p-values for all correlation functions.
    
    Returns:
        Dict mapping correlation function name to dict of {'correlation': float, 'p_value': float}
    """
    try:
        # Load dataset and get test split
        dataset = load_dataset(dataset_name)
        _, _, test_dataset = dataset.load_permanent_splits()
        try:
            df = test_dataset.get_dataframe()
            logging.debug(f"Loaded test split for {dataset_name} with shape {df.shape} and target columns {getattr(test_dataset, 'target_columns', 'unknown')}")
        except Exception as e:
            logging.debug(f"Could not introspect test split dataframe: {e}")
        
        # Run correlation experiment with ALL correlation functions at once
        experiment = CorrelationExperiment(
            name=f"LLM Judge Test - {dataset_name} - {measure} - {metric_instance.name}",
            description=f"Testing LLM judge correlation for {metric_instance.name} on {dataset_name}",
            metrics=[metric_instance],
            output_dir=f"/tmp/llm_judge_test_{seed}",
            dataset=test_dataset,
            correlation_funcs=correlation_funcs,
            seed=seed,
            should_split=False
        )
        logging.debug(f"Initialized CorrelationExperiment with metrics={[m.name for m in [metric_instance]]} corr_funcs={list(correlation_funcs.keys())} seed={seed}")
        
        # Run experiment and extract correlations for ALL functions
        all_correlations = experiment.run(print_results=False)
        logging.debug(f"Experiment run complete for seed={seed}. Available corr func results={list(all_correlations.keys())}")
        
        # Extract correlations and p-values for all correlation functions
        results_by_func = {}
        
        for func_name, correlations_for_func in all_correlations.items():
            if measure not in correlations_for_func:
                raise ValueError(f"Measure {measure} not found in correlation results for {func_name}")
            
            df_corr = correlations_for_func[measure]
            logging.debug(f"Seed {seed} - {func_name} correlation table for {measure} has shape {df_corr.shape}")
            
            # Find the correlation for our specific metric
            metric_row = df_corr[df_corr['Metric'] == metric_instance.name]
            if metric_row.empty:
                raise ValueError(f"Metric {metric_instance.name} not found in correlation results")
            
            correlation = metric_row.iloc[0]['Correlation']
            p_value = metric_row.iloc[0]['P-value']
            results_by_func[func_name] = {'correlation': correlation, 'p_value': p_value}

        logger.debug(f"Seed {seed} results: {results_by_func}")
        return results_by_func
        
    except Exception as e:
        logger.error(f"Error running seed {seed}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistical measures for correlation values."""
    logging.debug(f"Computing statistics over {len(values)} values (nan filtered)")
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
    logging.debug(f"Stats: n={n} mean={mean_val}")
    
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
    
    result_stats = {
        'mean': mean_val,
        'std': std_val,
        'ci_lower': mean_val - margin_error,
        'ci_upper': mean_val + margin_error,
        'ci_range': margin_error,
        'num_successful_runs': n
    }
    logging.debug(f"Stats computed: {result_stats}")
    return result_stats


def format_mean_ci(mean: float, ci_range: float) -> str:
    """Format mean ± CI for easy copying to papers."""
    if np.isnan(mean) or np.isnan(ci_range):
        return "N/A"
    formatted = f"{mean:.4f} ± {ci_range:.4f}"
    logging.debug(f"Formatted mean±CI: {formatted}")
    return formatted


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
    logging.debug(f"Sorting output columns: {len(correlation_columns)} corr seed cols, {len(p_value_columns)} pval seed cols")
    
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
    
    sorted_df = df[final_columns]
    logging.debug(f"Final output columns order has {len(final_columns)} columns")
    return sorted_df


def save_results(results: Dict[str, Any], output_file: str, logger: logging.Logger):
    """Save results to CSV file with properly sorted columns."""
    try:
        df = pd.DataFrame(results)
        df = sort_columns_for_output(df)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        logging.debug(f"Saved results with shape {df.shape} to {output_file}. Columns={list(df.columns)}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {e}")
        raise


def load_existing_results(output_file: str) -> Dict[str, Any]:
    """Load existing results from output file if it exists."""
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            logging.debug(f"Loaded existing results from {output_file} with shape {df.shape}")
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
        logging.debug(f"Existing results columns: {list(existing_df.columns)}")
        
        # Convert new results to DataFrame
        new_df = pd.DataFrame(new_results)
        logging.debug(f"New results rows: {len(new_df)} columns: {list(new_df.columns)}")
        
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
        logging.debug(f"Merged results final shape: {merged_df.shape}")
        
        logger.info(f"Merge complete: {len(merged_results)} total results")
        return merged_df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error merging with existing results: {e}")
        logger.info("Using new results only")
        return new_results


def main():
    parser = argparse.ArgumentParser(
        description="Test correlation stability of LLM as a Judge metrics across multiple seeds"
    )
    
    parser.add_argument(
        "--llm-judge-file",
        default="analysis/main_experiments/llm_judge_prompts.csv",
        help="CSV file containing LLM judge prompts"
    )
    parser.add_argument(
        "--model",
        default="gpt4o_mini",
        choices=["gpt4o_mini", "qwen3_32b", "llama3_70b", "gpt5_mini"],
        help="LLM model to use for judging"
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL for the model (e.g., http://localhost:7410/v1)"
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output CSV file path (default: auto-generated based on model and correlation)"
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
    
    logger.info("Starting LLM as a Judge correlation stability analysis")
    logger.info(f"Model: {args.model}")
    logger.info(f"API Base: {args.api_base}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Correlations: {correlation_types}")
    
    # Create base output directory and sub-results directory
    os.makedirs("results/main_runs/baselines", exist_ok=True)
    os.makedirs("results/main_runs/baselines/llm_judge_sub_results", exist_ok=True)
    
    # Load LLM judge prompts
    try:
        prompts = load_llm_judge_prompts(args.llm_judge_file)
    except Exception as e:
        logger.error(f"Failed to load prompts from {args.llm_judge_file}: {e}")
        return 1
    
    # Filter by dataset if specified
    if args.dataset:
        allowed_datasets = set(args.dataset)
        prompts = {k: v for k, v in prompts.items() if v['dataset'] in allowed_datasets}
        logger.info(f"Filtered to {len(prompts)} prompts for datasets: {args.dataset}")
    
    if not prompts:
        logger.error("No prompts to process after filtering")
        return 1
    
    # Group prompts by dataset for dataset-specific output files
    prompts_by_dataset = {}
    for key, prompt_info in prompts.items():
        dataset_name = prompt_info['dataset']
        if dataset_name not in prompts_by_dataset:
            prompts_by_dataset[dataset_name] = {}
        prompts_by_dataset[dataset_name][key] = prompt_info
    
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
        for dataset_name, dataset_prompts in prompts_by_dataset.items():
            logger.info(f"\n--- Processing dataset: {dataset_name} ---")
            
            # Dataset-specific output file in sub-results directory
            dataset_output_file = f"results/main_runs/baselines/llm_judge_sub_results/llm_judge_{args.model}_{correlation_type}_{dataset_name}.csv"
            dataset_output_files.append(dataset_output_file)
            
            logger.info(f"Dataset output file: {dataset_output_file}")
            
            # Results storage for this dataset
            dataset_results = []
            
            # Process each LLM judge prompt for this dataset
            for key, prompt_info in dataset_prompts.items():
                dataset_name = prompt_info['dataset']
                measure = prompt_info['measure']
                prompt_text = prompt_info['prompt']
                score_range = prompt_info['score_range']
                
                logger.info(f"Processing {dataset_name}.{measure}")
                
                # Get dataset instance to determine metric class and task description
                try:
                    dataset_instance = load_dataset(dataset_name)
                except Exception as e:
                    logger.warning(f"Failed to load dataset {dataset_name}: {e}")
                    continue
                
                # Determine metric class based on dataset references
                metric_class = determine_metric_class(dataset_instance)
                
                # Get task description from dataset
                task_description = dataset_instance.get_task_description() or f"Evaluate {measure} for {dataset_name} dataset"
                
                # Create metric name
                base_metric_name = f"LLMJudge-{args.model}-{measure}"
                
                # Run correlation for each seed
                correlations = []
                p_values = []
                errors = []
                
                for seed in args.seeds:
                    try:
                        logger.info(f"  Running seed {seed}...")
                        
                        # Create seed-specific model for cache busting
                        seed_model = create_llm_model(args.model, args.api_base, seed)
                        
                        # Use a unique metric name per seed to avoid dataset-level caching
                        metric_name_seeded = f"{base_metric_name}-seed{seed}"
                        
                        # Create fresh metric instance for this seed
                        metric = metric_class(
                            name=metric_name_seeded,
                            prompt=prompt_text,
                            score_range=score_range,
                            model=seed_model,
                            task_description=task_description,
                            seed=seed,
                            model_name=args.model
                        )
                        logging.debug(f"Constructed metric {metric_class.__name__} for seed={seed} with name={metric_name_seeded} score_range={score_range}")
                        
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
                        logging.debug(f"Seed {seed} results stored. Running totals: {len([c for c in correlations if not pd.isna(c)])} successful correlations")
                        
                    except Exception as e:
                        error_msg = f"Seed {seed}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(f"    Failed: {error_msg}")
                        correlations.append(np.nan)
                        p_values.append(np.nan)
                
                # Compute statistics for correlations and p-values (use absolute values for correlations)
                corr_stats = compute_statistics([abs(c) for c in correlations if not pd.isna(c)])
                pval_stats = compute_statistics(p_values)
                
                # Create result row
                result = {
                    'dataset': dataset_name,
                    'measure': measure,
                    'metric': base_metric_name,
                    'metric_class': metric_class.__name__,
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
            merged_output_file = f"results/main_runs/baselines/llm_judge_{args.model}_{correlation_type}.csv"
        
        # Save merged results
        if all_results:
            merged_results = merge_with_existing_results(all_results, merged_output_file, logger)
            save_results(merged_results, merged_output_file, logger)
            
            # Print summary for this correlation type
            print(f"\nSummary of LLM Judge {correlation_type.upper()} Correlation Results:")
            print(f"Model: {args.model}")
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
    
    logger.info("LLM as a Judge correlation analysis completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 