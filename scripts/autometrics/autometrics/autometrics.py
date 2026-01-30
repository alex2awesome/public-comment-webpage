# autometrics/autometrics.py
from autometrics.recommend.LLMRec import LLMRec
from autometrics.recommend.ColBERT import ColBERT
from autometrics.recommend.BM25 import BM25
from autometrics.recommend.PipelinedRec import PipelinedRec
from autometrics.metrics.MetricBank import all_metric_classes
from autometrics.aggregator.regression.PLS import PLS
from autometrics.aggregator.regression.HotellingPLS import HotellingPLS
from autometrics.dataset.Dataset import Dataset
from autometrics.metrics.Metric import Metric
from autometrics.recommend.MetricRecommender import MetricRecommender
from typing import List, Type, Optional, Dict, Any, Union
import dspy
import os
import importlib.util
import inspect
from autometrics.metrics.MultiMetric import MultiMetric

## This is the main file for the complete Autometrics pipeline.
#
# Pipeline Steps:
# 1. Generate metrics using the Metric Generation Configs and augment the metric bank with the new metrics
# 2. Retrieve the top K most relevant metrics from the metric bank using the Retriever
# 3. Evaluate the top K metrics on the dataset
# 4. Regress to get the top N metrics using the Regression Strategy
# 5. Generate a report card with the top N metrics and the single regression metric
# 6. Return the top N metrics, the single regression metric, and the report card

# =============================
# Autometrics Pipeline Scaffold
# =============================

def _detect_gpu_availability() -> bool:
    """
    Detect if GPU is available for use.
    
    Returns:
        True if CUDA is available and functional, False otherwise
    """
    try:
        # Use the existing GPU detection utility from the codebase
        from autometrics.metrics.utils.gpu_allocation import is_cuda_available
        return is_cuda_available()
    except ImportError:
        # Fallback to direct torch check if the utility is not available
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

def _get_default_retriever_config() -> dict:
    """
    Get the default retriever configuration based on hardware availability.
    
    Returns:
        Dictionary with retriever configuration optimized for the current hardware
    """
    gpu_available = _detect_gpu_availability()
    
    if gpu_available:
        # Use ColBERT + LLMRec pipeline for GPU environments
        print("[Autometrics] GPU detected - using ColBERT + LLMRec pipeline for optimal performance")
        return {
            "recommenders": [ColBERT, LLMRec],
            "top_ks": [60, 30],  # ColBERT gets more, LLMRec narrows down
            "index_paths": [None, None],  # Use default paths
            "force_reindex": False
        }
    else:
        # Use BM25 + LLMRec pipeline for CPU-only environments
        print("[Autometrics] No GPU detected - using BM25 + LLMRec pipeline for CPU-optimized performance")
        return {
            "recommenders": [BM25, LLMRec],
            "top_ks": [60, 30],  # BM25 gets more, LLMRec narrows down
            "index_paths": [None, None],  # Use default paths
            "force_reindex": False
        }

def _get_cache_dir() -> str:
    """
    Get the cache directory from environment variable AUTOMETRICS_CACHE_DIR,
    with fallback to "./autometrics_cache" if not set.
    
    Returns:
        Cache directory path as string
    """
    return os.environ.get("AUTOMETRICS_CACHE_DIR", "./autometrics_cache")

# Default configurations
FULL_GENERATOR_CONFIGS = {
    "llm_judge": {"metrics_per_trial": 10, "description": "Basic LLM Judge"},
    "rubric_prometheus": {"metrics_per_trial": 10, "description": "Rubric Generator (Prometheus)"},
    "rubric_dspy": {"metrics_per_trial": 10, "description": "Rubric Generator (DSPy)"},
    "geval": {"metrics_per_trial": 10, "description": "G-Eval"},
    "codegen": {"metrics_per_trial": 10, "description": "Code Generation"},
    "llm_judge_optimized": {"metrics_per_trial": 1, "description": "LLM Judge (MIPROv2-Optimized)"},
    "finetune": {"metrics_per_trial": 1, "description": "Fine-tuned ModernBERT"},
    "llm_judge_examples": {"metrics_per_trial": 1, "description": "LLM Judge (Example-Based)"},
}

DEFAULT_GENERATOR_CONFIGS  = {
    "llm_judge": {"metrics_per_trial": 10, "description": "Basic LLM Judge"},
    "rubric_dspy": {"metrics_per_trial": 5, "description": "Rubric Generator (DSPy)"},
    "llm_judge_optimized": {"metrics_per_trial": 1, "description": "LLM Judge (MIPROv2-Optimized)"},
    "llm_judge_examples": {"metrics_per_trial": 1, "description": "LLM Judge (Example-Based)"},
}

# Dynamic default retriever configuration based on hardware
DEFAULT_RETRIEVER_KWARGS = _get_default_retriever_config()

DEFAULT_REGRESSION_KWARGS = {
}

class Autometrics:
    """
    Main Autometrics pipeline orchestrator.
    This class ties together metric generation, retrieval, evaluation, regression, and reporting.
    """
    def __init__(
        self,
        metric_generation_configs: Optional[dict] = DEFAULT_GENERATOR_CONFIGS,
        retriever: Type[MetricRecommender] = PipelinedRec,
        retriever_kwargs: dict = DEFAULT_RETRIEVER_KWARGS,
        regression_strategy: Type = PLS,
        regression_kwargs: dict = DEFAULT_REGRESSION_KWARGS,
        metric_bank: Union[List[Type[Metric]], str] = all_metric_classes,
        generated_metrics_dir: Optional[str] = None,
        merge_generated_with_bank: bool = False,
        seed: int = 42,
        allowed_failed_metrics: int = 0,
        full_bank_data_cutoff: Optional[int] = 100,
        # New parameters for metric priors (no defaults - users must explicitly set if desired)
        metric_priors: Optional[List[Type[Metric]]] = None,
        generated_metric_priors: Optional[Dict[str, int]] = None,
        # Parallelization configuration
        enable_parallel_evaluation: bool = True,
        max_parallel_workers: int = 20,
        # Drop policy for generated metrics with negative coefficients in final regression
        drop_generated_negative_coefficients: bool = True,
        formatter_truncate_chars: Optional[int] = None,
    ):
        """
        Initialize the Autometrics pipeline.
        Args:
            metric_generation_configs: Dict of generator configs (defaults to DEFAULT_GENERATOR_CONFIGS)
            retriever: Retriever class to use (defaults to PipelinedRec)
            retriever_kwargs: Keyword arguments to pass to retriever constructor (defaults to PipelinedRec defaults)
                Note: The default retriever configuration automatically adapts to hardware:
                - GPU environments: Uses ColBERT + LLMRec pipeline for optimal performance
                - CPU-only environments: Uses BM25 + LLMRec pipeline for CPU-optimized performance
            regression_strategy: Regression aggregator class (defaults to PLS)
            regression_kwargs: Keyword arguments to pass to regression strategy constructor (dataset is automatically added during construction)
            metric_bank: Either:
                - List of metric classes (e.g., [BLEU, SARI, LevenshteinDistance]) - RECOMMENDED
                - Path to directory containing generated metric Python files (all files in the directory must be valid metrics)
                - Defaults to all_metric_classes from MetricBank (automatically switches to reference_free if no reference columns)
            generated_metrics_dir: Where to save generated metrics (defaults to "generated_metrics/{run_id}")
            merge_generated_with_bank: If True, save generated metrics directly to metric_bank directory
            seed: Random seed for reproducibility (defaults to 42)
            allowed_failed_metrics: Maximum number of metrics that can fail evaluation before raising an exception (default: 0)
            full_bank_data_cutoff: If not None (default: 100), and the dataset size is
                less than or equal to this value, and the default metric bank
                (i.e., `metric_bank is all_metric_classes`) is in use, the pipeline
                will ignore the full metric bank and use only generated metrics.
                If set to None, this behavior is disabled and the metric bank is
                used as-is regardless of dataset size. This setting does not modify
                the value stored in `self.metric_bank`; the decision is applied at
                runtime inside `run()` based on the current dataset size.
            metric_priors: List of metric classes that should be included upfront (e.g., [LDLRewardModel, BLEU])
                These metrics will be evaluated on the dataset before retrieval and included in the final regression
                No defaults - users must explicitly set if desired
            generated_metric_priors: Dict mapping generator types to number of metrics to generate as priors
                (e.g., {"llm_judge_optimized": 1, "llm_judge_examples": 1})
                These generated metrics will be included upfront alongside metric_priors
                No defaults - users must explicitly set if desired
            enable_parallel_evaluation: Whether to use parallel execution for metric evaluation (default: True)
                This can significantly speed up evaluation for network-bound metrics (LLM-based metrics)
            max_parallel_workers: Maximum number of parallel workers for metric evaluation (default: 20)
                For network-bound operations, this can be higher than CPU cores
        """
        # Store configs with meaningful defaults (no more None antipattern!)
        self.metric_generation_configs = metric_generation_configs
        self.retriever = retriever
        self.retriever_kwargs = retriever_kwargs
        self.regression_strategy = regression_strategy
        self.regression_kwargs = regression_kwargs
        self.metric_bank = metric_bank
        self.generated_metrics_dir = generated_metrics_dir
        self.merge_generated_with_bank = merge_generated_with_bank
        self.seed = seed
        self.allowed_failed_metrics = allowed_failed_metrics
        self.full_bank_data_cutoff = full_bank_data_cutoff
        
        # Store prior configurations (no defaults - users must explicitly set if desired)
        self.metric_priors = metric_priors or []
        self.generated_metric_priors = generated_metric_priors or {}
        
        # Store parallelization configuration
        self.enable_parallel_evaluation = enable_parallel_evaluation
        self.max_parallel_workers = max_parallel_workers
        self.drop_generated_negative_coefficients = drop_generated_negative_coefficients
        self.formatter_truncate_chars = formatter_truncate_chars

    def run(
        self,
        dataset: Dataset,
        target_measure: str,
        generator_llm: dspy.LM,
        judge_llm: dspy.LM,
        num_to_retrieve: int = 30,
        num_to_regress: Union[int, str] = 5,
        regenerate_metrics: bool = False,
        prometheus_api_base: Optional[str] = None,
        model_save_dir: Optional[str] = None,
        eval_dataset: Optional[Dataset] = None,
        report_output_path: Optional[str] = None,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the full Autometrics pipeline.
        Args:
            dataset: The dataset to evaluate on
            target_measure: The specific target measure/column to evaluate (e.g., "helpfulness", "fluency")
            generator_llm: LLM for metric generation
            judge_llm: LLM for metric evaluation
            num_to_retrieve: Number of metrics to retrieve from the bank (default: 30)
            num_to_regress: Number of metrics to select via regression (default: 5)
            regenerate_metrics: If True, force regeneration of metrics even if files exist
            prometheus_api_base: API base for Prometheus models (required for rubric_prometheus)
            model_save_dir: Directory to save fine-tuned models (defaults to "/finetunes")
            kwargs: Additional config for generators, retrievers, etc.
        Returns:
            Dict with keys: 'top_metrics', 'regression_metric', 'report_card', 'all_generated_metrics', etc.
        """
        print(f"[Autometrics] Starting pipeline for {dataset.get_name()} - {target_measure}")
        print(f"[Autometrics] Configuration: retrieve={num_to_retrieve}, regress={num_to_regress}, regenerate={regenerate_metrics}")
        
        # 0. Process metric priors (if any)
        prior_metrics = []
        if self.metric_priors or self.generated_metric_priors:
            print("\n[Autometrics] Step 0: Processing Metric Priors")
            prior_metrics = self._process_metric_priors(
                dataset, target_measure, generator_llm, judge_llm, regenerate_metrics, prometheus_api_base, model_save_dir
            )
            print(f"[Autometrics] Processed {len(prior_metrics)} prior metrics")
        
        # 1. Generate metrics (or load from disk if available)
        print("\n[Autometrics] Step 1: Generating/Loading Metrics")
        generated_metrics = self._generate_or_load_metrics(
            dataset, target_measure, generator_llm, judge_llm, regenerate_metrics, prometheus_api_base, model_save_dir
        )
        print(f"[Autometrics] Generated/Loaded {len(generated_metrics)} metrics")
        
        # 2. Load metric bank and configure retriever
        print("\n[Autometrics] Step 2: Loading Metric Bank")
        metric_bank = self._load_metric_bank(dataset)
        
        print(f"[Autometrics] Loaded {len(metric_bank)} metrics in bank")

        # 2.1 Apply small-dataset override to use generated metrics only
        # Do not overwrite self.metric_bank; only adjust the local metric_bank for this run
        try:
            dataset_size = len(dataset.get_dataframe())
        except Exception:
            dataset_size = None
        if (
            self.full_bank_data_cutoff is not None
            and dataset_size is not None
            and dataset_size <= self.full_bank_data_cutoff
            and (self.metric_bank is all_metric_classes)
        ):
            print(
                f"[Autometrics] Dataset size ({dataset_size}) <= cutoff ({self.full_bank_data_cutoff}) with default bank — using generated metrics only"
            )
            metric_bank = []

        # 2.5 Merge generated metrics with metric bank
        print("\n[Autometrics]  Merging Generated Metrics with Metric Bank")
        metric_bank = self._merge_generated_with_bank(metric_bank, generated_metrics)
        
        # Configure retriever with metric bank and model
        print("[Autometrics] Configuring retriever...")
        retriever_kwargs = self.retriever_kwargs.copy()
        retriever_kwargs["metric_classes"] = metric_bank
        retriever_kwargs["model"] = generator_llm  # Use generator LLM for LLMRec
        
        # Generate dynamic index paths and validate retrieval hyperparameters
        retriever_kwargs = self._validate_and_adjust_retriever_config(retriever_kwargs, dataset, metric_bank, num_to_retrieve)
        
        retriever_instance = self.retriever(**retriever_kwargs)
        
        # Construct regression strategy with dataset-specific kwargs
        print("[Autometrics] Configuring regression strategy...")
        regression_kwargs = self.regression_kwargs.copy()
        
        # Route selection behavior for HotellingPLS based on num_to_regress (only when explicitly used)
        if self.regression_strategy == HotellingPLS:
            regression_kwargs["random_state"] = self.seed
            if isinstance(num_to_regress, int) and num_to_regress > 0:
                regression_kwargs["selection_mode"] = "top_n"
                regression_kwargs["top_n"] = num_to_regress
            else:
                regression_kwargs["selection_mode"] = "alpha"
                regression_kwargs.pop("top_n", None)
        # Add dataset to kwargs since all regression classes accept dataset parameter
        regression_kwargs["dataset"] = dataset
        regression_instance = self.regression_strategy(**regression_kwargs)
        
        # 3. Retrieve top-K metrics using retriever
        print(f"\n[Autometrics] Step 3: Retrieving Top {num_to_retrieve} Metrics")
        retrieved_metrics = self._retrieve_top_k_metrics(dataset, target_measure, num_to_retrieve, retriever_instance, metric_bank)
        print(f"[Autometrics] Retrieved {len(retrieved_metrics)} metrics")
        
        # 4. Combine priors with retrieved metrics and evaluate
        all_metrics_to_evaluate = prior_metrics + retrieved_metrics
        print(f"\n[Autometrics] Step 4: Evaluating {len(all_metrics_to_evaluate)} Metrics on Dataset ({len(prior_metrics)} priors + {len(retrieved_metrics)} retrieved)")
        successful_metric_instances = self._evaluate_metrics_on_dataset(dataset, all_metrics_to_evaluate)
        
        # 5. Regress to get top-N metrics using regression strategy
        if self.regression_strategy == HotellingPLS:
            if regression_kwargs.get("selection_mode") == "alpha":
                print(f"\n[Autometrics] Step 5: Regression Analysis (Auto variable selection via Hotelling T²)")
            else:
                print(f"\n[Autometrics] Step 5: Regression Analysis (Selecting Top {num_to_regress} via Hotelling T²)")
        else:
            print(f"\n[Autometrics] Step 5: Regression Analysis (Selecting Top {num_to_regress} via {self.regression_strategy.__name__})")
        regression_results = self._regress_and_select_top_n(
            dataset, successful_metric_instances, target_measure, num_to_regress, regression_instance
        )

        print(f"[Autometrics] Found top {len(regression_results['top_metrics'])} metrics.")
        try:
            top_metric_names = []
            for metric in regression_results['top_metrics']:
                if hasattr(metric, '__name__'):
                    top_metric_names.append(metric.__name__)
                elif hasattr(metric, 'get_name') and callable(getattr(metric, 'get_name')):
                    top_metric_names.append(metric.get_name())
                else:
                    top_metric_names.append(type(metric).__name__)
            print(f"[Autometrics] Top metrics: {top_metric_names}")
        except Exception:
            # Fallback to safe repr if anything unexpected happens
            print(f"[Autometrics] Top metrics: {[type(m).__name__ for m in regression_results['top_metrics']]}")
        # Note: generated-negative drop policy is applied inside regression selection

        # 6. Generate a report card
        print("\n[Autometrics] Step 6: Generating Report Card")
        report_card = self._generate_report_card(
            regression_results['top_metrics'],
            regression_results['regression_metric'],
            dataset,
            target_measure
        )

        # 6b. Generate HTML report card artifact
        try:
            from autometrics.util.report_card import generate_metric_report_card
            html_artifacts = generate_metric_report_card(
                regression_metric=regression_results['regression_metric'],
                metrics=regression_results['top_metrics'],
                target_measure=target_measure,
                eval_dataset=eval_dataset,
                train_dataset=dataset,
                lm=generator_llm,
                output_path=report_output_path or os.path.join("artifacts", f"report_{dataset.get_name().replace(' ', '_')}_{target_measure.replace(' ', '_')}_{self.seed}.html"),
                verbose=verbose,
            )
            report_card_html = html_artifacts.get('html', '')
            report_card_path = html_artifacts.get('path')
            # Pull Kendall tau and Pearson r for regression if available
            regression_kendall_tau = None
            regression_pearson_r = None
            try:
                arts = html_artifacts.get('artifacts') or {}
                regression_kendall_tau = arts.get('kendall_tau_regression')
                regression_pearson_r = arts.get('pearson_r_regression')
            except Exception:
                pass
            print(f"[Autometrics] Report card HTML generated{f' at {report_card_path}' if report_card_path else ''}")
        except Exception as e:
            print(f"[Autometrics] Warning: Failed to generate HTML report card: {e}")
            report_card_html = ""
            report_card_path = None
            regression_kendall_tau = None
            regression_pearson_r = None
        
        # 7. Return results
        print("\n[Autometrics] Pipeline Complete!")
        
        return {
            'top_metrics': regression_results['top_metrics'],
            'regression_metric': regression_results['regression_metric'],
            'report_card': report_card,
            'report_card_html': report_card_html,
            'report_card_path': report_card_path,
            'regression_kendall_tau_eval': regression_kendall_tau,
            'regression_pearson_r_eval': regression_pearson_r,
            'all_generated_metrics': generated_metrics,
            'prior_metrics': prior_metrics,
            'retrieved_metrics': retrieved_metrics,
            'importance_scores': regression_results['importance_scores'],
            'dataset': dataset,
            'target_measure': target_measure,
            'pipeline_config': {
                'num_to_retrieve': num_to_retrieve,
                'num_to_regress': num_to_regress,
                'regenerate_metrics': regenerate_metrics,
                'generator_configs': self.metric_generation_configs,
                'metric_priors': [m.__name__ for m in self.metric_priors] if self.metric_priors else None,
                'generated_metric_priors': self.generated_metric_priors,
                'full_bank_data_cutoff': self.full_bank_data_cutoff,
            }
        }

    # =============================
    # Internal Helper Methods
    # =============================

    def _load_metric_bank(self, dataset: Optional[Dataset] = None) -> List[Type[Metric]]:
        """
        Load metric classes from metric_bank.
        - If metric_bank is a list: return it directly
        - If metric_bank is a directory: load all classes from the directory (assumes all files are valid metrics)
        - If metric_bank is None: return all_metric_classes
        
        If dataset is provided and has no reference columns, automatically switch to reference_free metrics
        when using the default all_metric_classes.
        """
        if isinstance(self.metric_bank, list):
            # Special case: if metric_bank is all_metric_classes and dataset has no reference columns,
            # switch to reference_free metrics
            if (self.metric_bank is all_metric_classes and 
                dataset is not None and 
                not self._dataset_has_reference_columns(dataset)):
                print("[Autometrics] Dataset has no reference columns - switching to reference_free metrics")
                from autometrics.metrics.MetricBank import reference_free_metric_classes
                return reference_free_metric_classes
            return self.metric_bank
        elif isinstance(self.metric_bank, str):
            return self._load_metrics_from_directory(self.metric_bank)
        else:
            return all_metric_classes

    def _process_metric_priors(
        self,
        dataset: Dataset,
        target_measure: str,
        generator_llm: dspy.LM,
        judge_llm: dspy.LM,
        regenerate_metrics: bool = False,
        prometheus_api_base: Optional[str] = None,
        model_save_dir: Optional[str] = None
    ) -> List[Type[Metric]]:
        """
        Process metric priors - both regular metrics and generated metrics.
        Returns a list of metric classes that should be included upfront.
        """
        prior_metrics = []
        
        # Add regular metric priors
        if self.metric_priors:
            print(f"[Autometrics] Adding {len(self.metric_priors)} regular metric priors")
            prior_metrics.extend(self.metric_priors)
        
        # Generate and add generated metric priors
        if self.generated_metric_priors:
            print(f"[Autometrics] Generating {sum(self.generated_metric_priors.values())} prior metrics using generators: {list(self.generated_metric_priors.keys())}")
            
            # Determine output directory for generated prior metrics
            if self.merge_generated_with_bank and isinstance(self.metric_bank, str):
                output_dir = self.metric_bank
            else:
                # Create a unique run ID based on dataset, target measure, and seed
                run_id = f"{dataset.get_name()}_{target_measure}_{self.seed or 42}"
                output_dir = self.generated_metrics_dir or f"generated_metrics/{run_id}"
            
            # Process each generator configuration for priors
            for generator_type, metrics_per_trial in self.generated_metric_priors.items():
                # Check if metrics already exist for this generator
                safe_dataset_name = dataset.get_name().replace(" ", "_").replace("/", "_")
                safe_measure_name = target_measure.replace(" ", "_").replace("/", "_")
                generator_dir = os.path.join(output_dir, "generated_metrics", safe_dataset_name, safe_measure_name, f"seed_{self.seed or 42}", generator_type)
                
                if not regenerate_metrics and os.path.exists(generator_dir):
                    # Try to load existing metrics
                    try:
                        existing_metrics = self._load_metrics_from_directory(generator_dir)
                        if len(existing_metrics) >= metrics_per_trial:
                            print(f"[Autometrics] Loaded {len(existing_metrics)} existing prior metrics for {generator_type}")
                            prior_metrics.extend(existing_metrics[:metrics_per_trial])
                            continue
                    except Exception as e:
                        print(f"[Autometrics] Warning: Failed to load existing prior metrics for {generator_type}: {e}")
                
                # Generate new prior metrics
                print(f"[Autometrics] Generating {metrics_per_trial} prior metrics using {generator_type}...")
                
                # Create generator based on type
                generator = self._create_generator(
                    generator_type, generator_llm, judge_llm, self.seed, prometheus_api_base, model_save_dir
                )
                
                # Generate metrics
                metrics = generator.generate(
                    dataset=dataset,
                    target_measure=target_measure,
                    n_metrics=metrics_per_trial
                )
                
                if not metrics:
                    print(f"[Autometrics] Warning: No prior metrics generated for {generator_type}")
                    continue
                
                # Save generated metrics
                metric_paths = self._save_generated_metrics(
                    metrics, generator_type, dataset.get_name(), target_measure, 
                    self.seed or 42, output_dir
                )
                
                print(f"[Autometrics] Saved {len(metric_paths)} prior metrics for {generator_type}")
                
                # Load the saved metrics as classes
                saved_metrics = self._load_metrics_from_directory(generator_dir)
                prior_metrics.extend(saved_metrics[:metrics_per_trial])
        
        return prior_metrics

    def _load_metrics_from_directory(self, directory: str) -> List[Type[Metric]]:
        """
        Load all metric classes from a directory.
        Assumes all Python files in the directory are valid metric classes.
        Uses importlib to dynamically import each .py file and collects all classes that inherit from Metric.
        Returns a list of metric classes (not instances).
        """
        metric_classes = []
        directory = os.path.abspath(directory)
        for filename in os.listdir(directory):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_path = os.path.join(directory, filename)
                module_name = os.path.splitext(filename)[0]
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    print(f"[Autometrics] Warning: Failed to import {filename}: {e}")
                    continue
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Only include classes defined in this module (not imported ones)
                    if obj.__module__ == module.__name__:
                        # Only include subclasses of Metric (but not Metric itself)
                        if issubclass(obj, Metric) and obj is not Metric:
                            metric_classes.append(obj)
        return metric_classes

    def _merge_generated_with_bank(self, metric_bank: List[Type[Metric]], generated_metrics: List[Type[Metric]]) -> List[Type[Metric]]:
        """
        Merge generated metrics with the existing metric bank.
        
        Args:
            metric_bank: List of existing metric classes (built-in metrics)
            generated_metrics: List of newly generated metric classes
            
        Returns:
            Combined list of metric classes with built-in metrics first, followed by generated metrics
            
        Note:
            - Both inputs are lists of metric classes (not instances)
            - Generated metrics are appended to the end of the metric bank
            - Duplicates are handled by checking metric names (generated metrics typically have unique names)
        """
        if not generated_metrics:
            # No generated metrics to merge, return metric bank as-is
            return metric_bank
        
        if not metric_bank:
            # No existing metric bank, return generated metrics
            return generated_metrics
        
        # Create a set of existing metric names for duplicate checking
        existing_metric_names = {metric.__name__ for metric in metric_bank}
        
        # Filter out any generated metrics that have the same name as existing metrics
        unique_generated_metrics = []
        for metric in generated_metrics:
            if metric.__name__ not in existing_metric_names:
                unique_generated_metrics.append(metric)
            else:
                print(f"[Autometrics] Warning: Skipping generated metric '{metric.__name__}' - name already exists in metric bank")
        
        # Combine metric bank with unique generated metrics
        combined_metrics = metric_bank + unique_generated_metrics
        
        print(f"[Autometrics] Merged {len(unique_generated_metrics)} unique generated metrics with {len(metric_bank)} existing metrics")
        
        return combined_metrics

    def _generate_or_load_metrics(
        self, 
        dataset: Dataset, 
        target_measure: str,
        generator_llm: dspy.LM, 
        judge_llm: dspy.LM, 
        regenerate_metrics: bool = False,
        prometheus_api_base: Optional[str] = None,
        model_save_dir: Optional[str] = None
    ) -> List[Type[Metric]]:
        """
        Generate new metrics using configured generators, or load from disk if available.
        Save generated metrics as Python files in generated_metrics_dir.
        Return a list of metric classes (not instances).
        """
        
        all_generated_metrics = []
        
        # Determine output directory for generated metrics
        if self.merge_generated_with_bank and isinstance(self.metric_bank, str):
            output_dir = self.metric_bank
        else:
            # Create a unique run ID based on dataset, target measure, and seed
            run_id = f"{dataset.get_name()}_{target_measure}_{self.seed or 42}"
            output_dir = self.generated_metrics_dir or f"generated_metrics/{run_id}"
        
        # Process each generator configuration
        for generator_type, config in self.metric_generation_configs.items():
            metrics_per_trial = config.get("metrics_per_trial", 1)
            generator_kwargs = dict(config.get("generator_kwargs", {}))
            scaffolding_run_name = generator_kwargs.pop("run_name", None)
            
            # Check if metrics already exist for this generator
            # Directory structure: dataset_name/measure/seed/generator_type
            safe_dataset_name = dataset.get_name().replace(" ", "_").replace("/", "_")
            safe_measure_name = target_measure.replace(" ", "_").replace("/", "_")
            generator_dir = os.path.join(output_dir, "generated_metrics", safe_dataset_name, safe_measure_name, f"seed_{self.seed or 42}", generator_type)
            
            if not regenerate_metrics and os.path.exists(generator_dir):
                # Try to load existing metrics
                try:
                    existing_metrics = self._load_metrics_from_directory(generator_dir)
                    if len(existing_metrics) >= metrics_per_trial:
                        print(f"[Autometrics] Loaded {len(existing_metrics)} existing metrics for {generator_type}")
                        all_generated_metrics.extend(existing_metrics)
                        continue
                except Exception as e:
                    print(f"[Autometrics] Warning: Failed to load existing metrics for {generator_type}: {e}")
            
            # Generate new metrics
            print(f"[Autometrics] Generating {metrics_per_trial} metrics using {generator_type}...")
            
            # Create generator based on type
            generator = self._create_generator(
                generator_type,
                generator_llm,
                judge_llm,
                self.seed,
                prometheus_api_base,
                model_save_dir,
                scaffolding_run_name=scaffolding_run_name,
            )
            
            # Generate metrics
            generate_kwargs = {
                "dataset": dataset,
                "target_measure": target_measure,
                "n_metrics": metrics_per_trial,
            }
            if generator_kwargs:
                generate_kwargs.update(generator_kwargs)

            metrics = generator.generate(**generate_kwargs)
            
            if not metrics:
                print(f"[Autometrics] Warning: No metrics generated for {generator_type}")
                continue
            
            # Save generated metrics
            metric_paths = self._save_generated_metrics(
                metrics, generator_type, dataset.get_name(), target_measure, 
                self.seed or 42, output_dir
            )
            
            print(f"[Autometrics] Saved {len(metric_paths)} metrics for {generator_type}")
            
            # Load the saved metrics as classes
            saved_metrics = self._load_metrics_from_directory(generator_dir)
            all_generated_metrics.extend(saved_metrics)
        
        return all_generated_metrics
    
    def _create_generator(
        self,
        generator_type: str,
        generator_llm: dspy.LM,
        judge_llm: dspy.LM,
        seed: int,
        prometheus_api_base: Optional[str] = None,
        model_save_dir: Optional[str] = None,
        scaffolding_run_name: Optional[str] = None,
    ):
        """Create generator instance based on type."""
        from autometrics.generator.LLMJudgeProposer import BasicLLMJudgeProposer
        from autometrics.generator.LLMJudgeExampleProposer import LLMJudgeExampleProposer
        from autometrics.generator.OptimizedJudgeProposer import OptimizedJudgeProposer
        from autometrics.generator.GEvalJudgeProposer import GEvalJudgeProposer
        from autometrics.generator.CodeGenerator import CodeGenerator
        from autometrics.generator.RubricGenerator import RubricGenerator
        from autometrics.generator.ScaffoldingProposer import ScaffoldingProposer
        from autometrics.generator.FinetuneGenerator import FinetuneGenerator
        
        if generator_type == "llm_judge":
            return BasicLLMJudgeProposer(
                generator_llm=generator_llm,
                executor_kwargs={"model": judge_llm, "seed": seed},
                seed=seed,
                truncate_chars=self.formatter_truncate_chars,
            )
        elif generator_type == "llm_judge_examples":
            return LLMJudgeExampleProposer(
                generator_llm=generator_llm,
                executor_kwargs={"model": judge_llm, "seed": seed},
                seed=seed,
                max_optimization_samples=100,
                truncate_chars=self.formatter_truncate_chars,
            )
        elif generator_type == "llm_judge_optimized":
            return OptimizedJudgeProposer(
                generator_llm=generator_llm,
                executor_kwargs={"model": judge_llm, "seed": seed},
                auto_mode="medium",
                num_threads=16,
                eval_function_name='inverse_distance',
                seed=seed,
                truncate_chars=self.formatter_truncate_chars,
            )
        elif generator_type == "geval":
            return GEvalJudgeProposer(
                generator_llm=generator_llm,
                executor_kwargs={"model": judge_llm, "seed": seed},
                seed=seed,
                truncate_chars=self.formatter_truncate_chars,
            )
        elif generator_type == "codegen":
            return CodeGenerator(
                generator_llm=generator_llm,
                executor_kwargs={"seed": seed},
                seed=seed,
                truncate_chars=self.formatter_truncate_chars,
            )
        elif generator_type == "rubric_prometheus":
            # Require api_base for prometheus
            if not prometheus_api_base:
                raise ValueError("prometheus_api_base is required for rubric_prometheus generator")
            
            # Create Prometheus judge LLM
            from prometheus_eval.litellm import LiteLLM
            judge_llm_prometheus = LiteLLM(
                "litellm_proxy/Unbabel/M-Prometheus-14B",
                api_base=prometheus_api_base
            )
            return RubricGenerator(
                generator_llm=generator_llm,
                executor_kwargs={"model": judge_llm_prometheus, "seed": seed},
                use_prometheus=True,
                seed=seed,
                truncate_chars=self.formatter_truncate_chars,
            )
        elif generator_type == "rubric_dspy":
            return RubricGenerator(
                generator_llm=generator_llm,
                executor_kwargs={"model": judge_llm, "seed": seed},
                use_prometheus=False,
                seed=seed,
                truncate_chars=self.formatter_truncate_chars,
            )
        elif generator_type == "finetune":
            # Use provided model_save_dir or default
            save_dir = model_save_dir or "/finetunes"
            return FinetuneGenerator(
                generator_llm=generator_llm,
                model_save_dir=save_dir,
                seed=seed,
                truncate_chars=self.formatter_truncate_chars,
            )
        elif generator_type == "scaffolding":
            return ScaffoldingProposer(
                generator_llm=generator_llm,
                executor_class=None,
                executor_kwargs={"model": judge_llm, "seed": seed},
                seed=seed,
                truncate_chars=self.formatter_truncate_chars,
                run_name=scaffolding_run_name,
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
    
    def _save_generated_metrics(self, metrics: List, generator_type: str, dataset_name: str, measure: str, seed: int, output_dir: str):
        """Save generated metrics to organized directory structure."""
        from pathlib import Path
        import re as _re
        
        def _infer_exported_class_name(metric_obj) -> str:
            base = getattr(metric_obj, 'name', metric_obj.__class__.__name__)
            safe = str(base).replace(' ', '_').replace('-', '_')
            mod = metric_obj.__class__.__module__
            if mod.endswith('GeneratedLLMJudgeMetric'):
                return f"{safe}_LLMJudge"
            if mod.endswith('GeneratedExampleRubric'):
                return f"{safe}_ExampleRubric"
            if mod.endswith('GeneratedOptimizedJudge'):
                return f"{safe}_OptimizedJudge"
            # Fallback: parse class name from generated code
            try:
                code = metric_obj._generate_python_code(include_metric_card=False)
                m = _re.search(r"class\s+([A-Za-z_]\w*)\(", code)
                if m:
                    return m.group(1)
            except Exception:
                pass
            return safe
        
        # Create output directory structure: dataset_name/measure/seed/generator_type
        safe_dataset_name = dataset_name.replace(" ", "_").replace("/", "_")
        safe_measure_name = measure.replace(" ", "_").replace("/", "_")
        generator_dir = Path(output_dir) / "generated_metrics" / safe_dataset_name / safe_measure_name / f"seed_{seed}" / generator_type
        generator_dir.mkdir(parents=True, exist_ok=True)
        
        metric_paths = []
        
        for i, metric in enumerate(metrics):
            if metric is None:
                continue
                
            # Create clean filename
            metric_filename = f"{safe_dataset_name}_{safe_measure_name}_{generator_type}_seed{seed}_metric{i+1:02d}.py"
            
            metric_path = generator_dir / metric_filename
            
            # Save metric as standalone Python file
            metric.save_python_code(str(metric_path))
            metric_paths.append(str(metric_path))

            # Attach helpful attributes for downstream import-by-path
            try:
                setattr(metric, "_python_file_path", str(metric_path))
                setattr(metric, "_exported_class_name", _infer_exported_class_name(metric))
            except Exception:
                pass
        
        return metric_paths

    def _retrieve_top_k_metrics(self, dataset: Dataset, target_measure: str, k: int, retriever_instance: MetricRecommender, metric_bank: List[Type[Metric]]) -> List[Type[Metric]]:
        """
        Use the retriever to get the top-K most relevant metrics for the dataset and target.
        Return a list of metric classes.
        """
        if not metric_bank:
            print("[Autometrics] Warning: No metrics in metric bank")
            return []
        
        print(f"[Autometrics] Retrieving top {k} metrics from {len(metric_bank)} available metrics")
        
        # Use the retriever to get top-K metrics
        retrieved_metrics = retriever_instance.recommend(
            dataset=dataset,
            target_measurement=target_measure,
            k=k
        )
        
        # Filter out None results
        retrieved_metrics = [m for m in retrieved_metrics if m is not None]
        
        print(f"[Autometrics] Retrieved {len(retrieved_metrics)} metrics")
        for i, metric in enumerate(retrieved_metrics):
            print(f"  {i+1}. {metric.__name__}")
        
        return retrieved_metrics

    def _evaluate_metrics_on_dataset(self, dataset: Dataset, metric_classes: List[Type[Metric]]) -> List[Metric]:
        """
        Add each metric to the dataset and compute its values using parallel execution.
        Returns the list of successfully evaluated metric instances.
        
        This method uses a two-phase approach to avoid race conditions:
        1. Phase 1: Parallel execution of metric.predict() on dataset copies
        2. Phase 2: Sequential aggregation of results back to the original dataset
        """
        if not metric_classes:
            print("[Autometrics] No metrics to evaluate")
            return []
        
        # Use MetricBank to properly instantiate metrics with GPU allocation and caching
        from autometrics.metrics.MetricBank import build_metrics
        
        successful_metrics = []
        failed_metrics = []
        
        # Build metrics with proper configuration
        print(f"[Autometrics] Building {len(metric_classes)} metrics with GPU allocation...")
        metrics = build_metrics(
            classes=metric_classes,
            cache_dir=_get_cache_dir(),
            seed=self.seed,
            use_cache=True,
            gpu_buffer_ratio=0.10
        )
        print(f"[Autometrics] Built {len([m for m in metrics if m is not None])} valid metrics")
        
        # Filter out None metrics
        valid_metrics = [(i, metric) for i, metric in enumerate(metrics) if metric is not None]
        
        if not valid_metrics:
            print("[Autometrics] No valid metrics to evaluate")
            return []
        
        # Check for device_map="auto" metrics that should be forced to sequential
        auto_device_map_metrics = []
        for i, metric in valid_metrics:
            # Check if this metric uses device_map="auto" by inspecting its attributes
            if hasattr(metric, 'device_map') and metric.device_map == "auto":
                auto_device_map_metrics.append((i, metric))
                print(f"  🔄 {metric.get_name()} uses device_map='auto' - will be evaluated sequentially")
            elif hasattr(metric, 'load_kwargs') and isinstance(metric.load_kwargs, dict):
                if metric.load_kwargs.get('device_map') == "auto":
                    auto_device_map_metrics.append((i, metric))
                    print(f"  🔄 {metric.get_name()} uses load_kwargs device_map='auto' - will be evaluated sequentially")
        
        # Also check if any metrics were allocated with device_map="auto" by the GPU allocator
        # We need to check the actual allocation that was applied during build_metrics
        for i, metric in valid_metrics:
            metric_name = metric.get_name()
            # Check if this metric was allocated with device_map="auto" by looking at its actual device_map attribute
            if hasattr(metric, 'device_map') and metric.device_map == "auto":
                if (i, metric) not in auto_device_map_metrics:  # Avoid duplicates
                    auto_device_map_metrics.append((i, metric))
                    print(f"  🔄 {metric_name} was allocated with device_map='auto' by GPU allocator - will be evaluated sequentially")
        
        # Separate auto device_map metrics from regular metrics
        regular_metrics = [(i, metric) for i, metric in valid_metrics if (i, metric) not in auto_device_map_metrics]
        
        successful_metrics = []
        
        # Phase 1: Evaluate regular metrics in parallel (if enabled and we have multiple)
        if self.enable_parallel_evaluation and len(regular_metrics) > 1:
            print(f"[Autometrics] Evaluating {len(regular_metrics)} regular metrics using parallel execution...")
            successful_metrics.extend(self._evaluate_metrics_parallel(dataset, metric_classes, regular_metrics))
        elif len(regular_metrics) > 0:
            print(f"[Autometrics] Evaluating {len(regular_metrics)} regular metrics using sequential execution...")
            successful_metrics.extend(self._evaluate_metrics_sequential(dataset, metric_classes, regular_metrics))
        
        # Phase 2: Evaluate device_map="auto" metrics sequentially (if any)
        if len(auto_device_map_metrics) > 0:
            print(f"[Autometrics] Evaluating {len(auto_device_map_metrics)} device_map='auto' metrics sequentially...")
            print(f"[Autometrics] device_map='auto' metrics: [{', '.join(m.get_name() for _, m in auto_device_map_metrics)}]")
            successful_metrics.extend(self._evaluate_metrics_sequential(dataset, metric_classes, auto_device_map_metrics))
        
        return successful_metrics
    
    def _evaluate_metrics_sequential(self, dataset: Dataset, metric_classes: List[Type[Metric]], valid_metrics: List[tuple]) -> List[Metric]:
        """
        Sequential evaluation of metrics with memory management.
        """
        import gc
        import torch
        
        successful_metrics = []
        failed_metrics = []
        
        for idx, (i, metric) in enumerate(valid_metrics):
            try:
                print(f"  Evaluating {idx+1}/{len(valid_metrics)}: {metric.get_name()}")
                
                # Add metric to dataset and compute values
                dataset.add_metric(metric, update_dataset=True)
                
                print(f"    ✓ {metric.get_name()} computed successfully")
                successful_metrics.append(metric)
                
                # Unload the metric after successful evaluation to free GPU memory
                # This is especially important for large models like LDLReward27B (27B params)
                if hasattr(metric, '_unload_model') and callable(getattr(metric, '_unload_model')):
                    metric._unload_model()
                    print(f"    🔄 Unloaded {metric.get_name()} to free GPU memory")
                elif hasattr(metric, '_unload_models') and callable(getattr(metric, '_unload_models')):
                    metric._unload_models()
                    print(f"    🔄 Unloaded {metric.get_name()} to free GPU memory")
                else:
                    # For metrics without explicit unload methods, clear heavy attributes
                    # Preserve dspy.LM to avoid losing LLM configuration for export/reuse
                    for attr in ['model', 'tokenizer', 'qg', 'qa']:
                        if not hasattr(metric, attr):
                            continue
                        if attr == 'model' and isinstance(getattr(metric, 'model', None), dspy.LM):
                            print(f"    🔄 Skipping clearing dspy.LM for {metric.get_name()}")
                            continue
                        setattr(metric, attr, None)
                    if hasattr(metric, 'model') and getattr(metric, 'model', None) is None:
                        print(f"    🔄 Cleared model for {metric.get_name()} (not dspy.LM)")
                    print(f"    🔄 Cleared heavy attributes (preserved dspy.LM if present) for {metric.get_name()}")
                
                # Force garbage collection and CUDA cache clearing
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ✗ Error evaluating {metric.get_name()}: {e}")
                failed_metrics.append(metric_classes[i])
                
                # Try to unload the failed metric as well
                try:
                    if hasattr(metric, '_unload_model') and callable(getattr(metric, '_unload_model')):
                        metric._unload_model()
                    elif hasattr(metric, '_unload_models') and callable(getattr(metric, '_unload_models')):
                        metric._unload_models()
                    else:
                        for attr in ['model', 'tokenizer', 'qg', 'qa']:
                            if hasattr(metric, attr):
                                setattr(metric, attr, None)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as unload_error:
                    print(f"    ⚠ Warning: Failed to unload {metric.get_name()} after error: {unload_error}")
                
                # Check if we've exceeded the allowed failed metrics threshold
                if len(failed_metrics) > self.allowed_failed_metrics:
                    print(f"Exceeded allowed failed metrics limit ({self.allowed_failed_metrics}). Failed metrics: {[m.__name__ for m in failed_metrics]}")
                    raise e
                continue
        
        print(f"[Autometrics] Sequential evaluation complete. Dataset now has {len(dataset.get_metrics())} metrics")
        return successful_metrics
    
    def _evaluate_metrics_parallel(self, dataset: Dataset, metric_classes: List[Type[Metric]], valid_metrics: List[tuple]) -> List[Metric]:
        """
        Parallel evaluation of metrics using a two-phase approach with smart memory management.
        
        This method implements a fallback strategy for GPU memory errors:
        1. First attempts parallel execution for all metrics
        2. If any metric fails with GPU memory issues (CUDA OOM or CUBLAS allocation failure), adds it to a sequential queue
        3. Runs failed metrics one at a time, unloading other metrics between runs
        """
        import concurrent.futures
        import threading
        from typing import Tuple, Optional
        import gc
        import torch
        
        successful_metrics = []
        failed_metrics = []
        cuda_oom_metrics = []  # Metrics that failed due to CUDA OOM
        
        # Phase 1: Parallel execution of metric.predict() on dataset copies
        def evaluate_single_metric(metric_info: Tuple[int, Metric]) -> Tuple[int, Metric, Optional[dict], Optional[Exception]]:
            """
            Evaluate a single metric on a dataset copy.
            Returns: (original_index, metric_instance, results_dict, exception_if_any)
            """
            original_index, metric = metric_info
            
            try:
                # Debug: Show metric device info before prediction
                print(f"[Parallel] {metric.get_name()} - checking device before predict...")
                if hasattr(metric, 'model') and metric.model is not None:
                    try:
                        if hasattr(metric.model, 'device'):
                            print(f"[Parallel] {metric.get_name()} model device: {metric.model.device}")
                        elif hasattr(metric.model, 'hf_device_map'):
                            print(f"[Parallel] {metric.get_name()} model hf_device_map: {metric.model.hf_device_map}")
                        else:
                            print(f"[Parallel] {metric.get_name()} model has no device info")
                    except Exception as e:
                        print(f"[Parallel] {metric.get_name()} could not determine model device: {e}")
                
                # Create a deep copy of the dataset to avoid race conditions
                dataset_copy = dataset.copy()
                
                # Execute the expensive predict operation
                results = metric.predict(dataset_copy, update_dataset=True)
                
                # Extract the computed values from the dataset copy
                metric_name = metric.get_name()
                if isinstance(metric, MultiMetric):
                    # For MultiMetrics, get all submetric values
                    submetric_names = metric.get_submetric_names()
                    results_dict = {}
                    for submetric_name in submetric_names:
                        if submetric_name in dataset_copy.get_dataframe().columns:
                            results_dict[submetric_name] = dataset_copy.get_dataframe()[submetric_name].tolist()
                else:
                    # For regular metrics, get the single metric value
                    if metric_name in dataset_copy.get_dataframe().columns:
                        results_dict = {metric_name: dataset_copy.get_dataframe()[metric_name].tolist()}
                    else:
                        results_dict = {}
                
                print(f"    ✓ {metric.get_name()} computed successfully (parallel)")
                return (original_index, metric, results_dict, None)
                
            except Exception as e:
                # Enhanced error handling for GPU-related issues
                error_msg = str(e)
                if "meta tensor" in error_msg.lower() or "device" in error_msg.lower():
                    print(f"    ⚠ GPU device conflict for {metric.get_name()}: {error_msg}")
                    print(f"    🔧 This may be due to device mapping conflicts in parallel execution")
                elif "truncation_strategy" in error_msg.lower():
                    print(f"    ⚠ Tokenizer parameter conflict for {metric.get_name()}: {error_msg}")
                    print(f"    🔧 This is due to deprecated tokenizer parameters")
                elif "cuda out of memory" in error_msg.lower() or "cublas_status_alloc_failed" in error_msg.lower():
                    print(f"    💾 GPU memory issue for {metric.get_name()}: {error_msg}")
                    print(f"    🔄 Will retry this metric sequentially after unloading others")
                else:
                    print(f"    ✗ Error evaluating {metric.get_name()} (parallel): {error_msg}")
                
                return (original_index, metric, None, e)
        
        # Determine optimal number of workers
        # Our GPU allocation system already handles device conflicts properly
        max_workers = min(len(valid_metrics), self.max_parallel_workers)
        
        # Execute metrics in parallel
        evaluation_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_metric = {
                executor.submit(evaluate_single_metric, metric_info): metric_info 
                for metric_info in valid_metrics
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_metric):
                metric_info = future_to_metric[future]
                try:
                    result = future.result()
                    evaluation_results.append(result)
                except Exception as e:
                    # This shouldn't happen since we handle exceptions in evaluate_single_metric
                    print(f"    ✗ Unexpected error in parallel execution: {e}")
                    original_index, metric = metric_info
                    evaluation_results.append((original_index, metric, None, e))
        
        # Sort results by original index to maintain order
        evaluation_results.sort(key=lambda x: x[0])
        
        # Phase 2: Sequential aggregation of results back to the original dataset
        print("[Autometrics] Aggregating results back to original dataset...")
        
        for original_index, metric, results_dict, exception in evaluation_results:
            if exception is not None:
                # Check if this is a GPU memory error (CUDA OOM or CUBLAS allocation failure)
                error_msg = str(exception)
                if "cuda out of memory" in error_msg.lower() or "cublas_status_alloc_failed" in error_msg.lower():
                    # Add to GPU memory error queue for sequential retry
                    cuda_oom_metrics.append((original_index, metric, metric_classes[original_index]))
                    print(f"    📋 Added {metric.get_name()} to GPU memory error retry queue")
                else:
                    # Other types of errors - add to failed metrics
                    failed_metrics.append(metric_classes[original_index])
                    
                    # Check if we've exceeded the allowed failed metrics threshold
                    if len(failed_metrics) > self.allowed_failed_metrics:
                        print(f"Exceeded allowed failed metrics limit ({self.allowed_failed_metrics}). Failed metrics: {[m.__name__ for m in failed_metrics]}")
                        raise exception
                continue
            
            try:
                # Add metric to the original dataset (this modifies shared state, so it's sequential)
                dataset.add_metric(metric, update_dataset=False)  # Don't trigger predict again
                
                # Manually update the dataframe with the computed results
                if results_dict:
                    df = dataset.get_dataframe()
                    for column_name, values in results_dict.items():
                        if len(values) == len(df):
                            df[column_name] = values
                        else:
                            print(f"    ⚠ Warning: Length mismatch for {column_name} ({len(values)} vs {len(df)})")
                    dataset.set_dataframe(df)
                
                successful_metrics.append(metric)
                print(f"    ✓ {metric.get_name()} aggregated successfully")
                
            except Exception as e:
                print(f"    ✗ Error aggregating {metric.get_name()}: {e}")
                failed_metrics.append(metric_classes[original_index])
                
                # Check if we've exceeded the allowed failed metrics threshold
                if len(failed_metrics) > self.allowed_failed_metrics:
                    print(f"Exceeded allowed failed metrics limit ({self.allowed_failed_metrics}). Failed metrics: {[m.__name__ for m in failed_metrics]}")
                    raise e
        
        # Phase 3: Sequential retry of CUDA OOM metrics
        if cuda_oom_metrics:
            print(f"\n[Autometrics] Retrying {len(cuda_oom_metrics)} CUDA OOM metrics sequentially...")
            
            # First, unload all currently loaded metrics to free up GPU memory
            print("[Autometrics] Unloading all metrics to free GPU memory...")
            self._unload_all_metrics(successful_metrics)
            
            # Now try each CUDA OOM metric one at a time
            for original_index, metric, metric_class in cuda_oom_metrics:
                print(f"\n[Autometrics] Retrying {metric.get_name()} sequentially...")
                
                try:
                    # Create a fresh dataset copy for this metric
                    dataset_copy = dataset.copy()
                    
                    # Execute the metric
                    results = metric.predict(dataset_copy, update_dataset=True)
                    
                    # Extract the computed values
                    metric_name = metric.get_name()
                    if isinstance(metric, MultiMetric):
                        # For MultiMetrics, get all submetric values
                        submetric_names = metric.get_submetric_names()
                        results_dict = {}
                        for submetric_name in submetric_names:
                            if submetric_name in dataset_copy.get_dataframe().columns:
                                results_dict[submetric_name] = dataset_copy.get_dataframe()[submetric_name].tolist()
                    else:
                        # For regular metrics, get the single metric value
                        if metric_name in dataset_copy.get_dataframe().columns:
                            results_dict = {metric_name: dataset_copy.get_dataframe()[metric_name].tolist()}
                        else:
                            results_dict = {}
                    
                    # Add metric to the original dataset
                    dataset.add_metric(metric, update_dataset=False)
                    
                    # Manually update the dataframe with the computed results
                    if results_dict:
                        df = dataset.get_dataframe()
                        for column_name, values in results_dict.items():
                            if len(values) == len(df):
                                df[column_name] = values
                            else:
                                print(f"    ⚠ Warning: Length mismatch for {column_name} ({len(values)} vs {len(df)})")
                        dataset.set_dataframe(df)
                    
                    successful_metrics.append(metric)
                    print(f"    ✓ {metric.get_name()} retry successful!")
                    
                except Exception as e:
                    print(f"    ✗ {metric.get_name()} retry failed: {e}")
                    failed_metrics.append(metric_class)
                    
                    # Check if we've exceeded the allowed failed metrics threshold
                    if len(failed_metrics) > self.allowed_failed_metrics:
                        print(f"Exceeded allowed failed metrics limit ({self.allowed_failed_metrics}). Failed metrics: {[m.__name__ for m in failed_metrics]}")
                        raise e
                
                finally:
                    # Always unload this metric after processing to free memory for the next one
                    self._unload_metric(metric)
                    # Force garbage collection to free up memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        print(f"[Autometrics] Parallel evaluation complete. Dataset now has {len(dataset.get_metrics())} metrics")
        print(f"[Autometrics] Successfully evaluated {len(successful_metrics)} metrics, {len(failed_metrics)} failed")
        
        return successful_metrics
    
    def _regress_and_select_top_n(
        self, 
        dataset: Dataset, 
        metric_instances: List[Metric], 
        target_measure: str,
        n: Union[int, str],
        regression_instance: Any # Added regression_instance parameter
    ) -> Dict[str, Any]:
        """
        Fit the regression strategy on the dataset using the given metrics.
        Select the top-N most important metrics, refit regression, and return results.
        
        Uses a hybrid dataset approach:
        - Creates a copy for safe experimentation during metric selection
        - Adds the final regression metric to the original dataset for user access
        
        Returns a dict with keys: 'top_metrics', 'regression_metric', etc.
        """
        if not metric_instances:
            print("[Autometrics] No metrics for regression")
            return {
                'top_metrics': [],
                'regression_metric': None,
                'importance_scores': [],
                'final_regression_metric': None
            }
        
        # HotellingPLS fast-path: train once, optionally drop generated-negatives, and return trained instance
        if hasattr(regression_instance, "get_selected_columns") and isinstance(regression_instance, HotellingPLS):
            regression_dataset = dataset.copy()
            regression_instance.input_metrics = metric_instances
            regression_instance.learn(regression_dataset, target_column=target_measure)

            selected_cols = set(regression_instance.get_selected_columns())

            # Normalize importance pairs to (score, name)
            raw_pairs = regression_instance.identify_important_metrics() or []
            importance_pairs = []
            for a, b in raw_pairs:
                if isinstance(a, (int, float)) and not isinstance(b, (int, float)):
                    importance_pairs.append((float(a), b))
                elif isinstance(b, (int, float)) and not isinstance(a, (int, float)):
                    importance_pairs.append((float(b), a))
                else:
                    try:
                        importance_pairs.append((float(a), str(b)))
                    except Exception:
                        try:
                            importance_pairs.append((float(b), str(a)))
                        except Exception:
                            continue

            # Map selected columns back to metric instances
            top_metrics = []
            for metric in metric_instances:
                if isinstance(metric, MultiMetric):
                    if any(sub in selected_cols for sub in metric.get_submetric_names()):
                        top_metrics.append(metric)
                else:
                    if metric.get_name() in selected_cols:
                        top_metrics.append(metric)

            # Optionally drop generated metrics with negative coefficients and retrain
            if self.drop_generated_negative_coefficients:
                try:
                    kept_metrics = self._select_metrics_after_dropping_generated_negatives(
                        regression_metric=regression_instance,
                        selected_metrics=top_metrics
                    )
                except Exception:
                    kept_metrics = top_metrics
            else:
                kept_metrics = top_metrics

            if kept_metrics != top_metrics:
                try:
                    new_reg = type(regression_instance)(
                        name=getattr(regression_instance, 'name', None),
                        description=getattr(regression_instance, 'description', None),
                        input_metrics=kept_metrics,
                        dataset=regression_dataset,
                        selection_mode=getattr(regression_instance, 'selection_mode', 'alpha'),
                        top_n=getattr(regression_instance, 'top_n', None),
                        random_state=getattr(regression_instance, 'random_state', self.seed),
                    )
                    new_reg.learn(regression_dataset, target_column=target_measure)

                    try:
                        new_importance_pairs = new_reg.identify_important_metrics() or []
                    except Exception:
                        new_importance_pairs = []

                    new_selected_cols = set(new_reg.get_selected_columns())
                    new_top_metrics = []
                    for metric in kept_metrics:
                        if isinstance(metric, MultiMetric):
                            if any(sub in new_selected_cols for sub in metric.get_submetric_names()):
                                new_top_metrics.append(metric)
                        else:
                            if metric.get_name() in new_selected_cols:
                                new_top_metrics.append(metric)

                    dataset.add_metric(new_reg, update_dataset=True)
                    return {
                        'top_metrics': new_top_metrics,
                        'regression_metric': new_reg,
                        'importance_scores': new_importance_pairs,
                        'final_regression_metric': new_reg,
                        'all_metrics_importance': new_importance_pairs
                    }
                except Exception:
                    pass

            # Add the trained regression metric to the original dataset
            dataset.add_metric(regression_instance, update_dataset=True)

            return {
                'top_metrics': top_metrics,
                'regression_metric': regression_instance,
                'importance_scores': importance_pairs,
                'final_regression_metric': regression_instance,
                'all_metrics_importance': importance_pairs
            }

        # Legacy path (e.g., plain PLS): ensure n is a positive int
        if not isinstance(n, int) or n <= 0:
            n = 5
        print(f"[Autometrics] Running regression to select top {n} metrics from {len(metric_instances)} candidates...")
        
        # Get metric column names from the dataset
        metric_columns = dataset.get_metric_columns()
        if not metric_columns:
            print("[Autometrics] Warning: No metric columns found in dataset")
            return {
                'top_metrics': [],
                'regression_metric': None,
                'importance_scores': [],
                'final_regression_metric': None
            }
        
        # Create a copy of the dataset for regression (safe experimentation)
        regression_dataset = dataset.copy()
        
        # Fit regression on all metrics to identify importance
        print("  Fitting regression on all metrics to identify importance...")
        regression_instance.input_metrics = metric_instances
        regression_instance.learn(regression_dataset, target_column=target_measure)
        
        # Get importance scores for all metrics
        importance_pairs = regression_instance.identify_important_metrics()
        
        # Sort by importance (absolute value of coefficient/importance)
        importance_pairs.sort(key=lambda x: abs(x[0]), reverse=True)
        
        print("  Metric importance scores:")
        for i, (score, metric_name) in enumerate(importance_pairs):
            print(f"    {i+1}. {metric_name}: {score:.4f}")
        
        # Select top-N most important metrics
        top_n_metric_names = [metric_name for _, metric_name in importance_pairs[:n]]
        top_n_metrics = []
        
        # Find the corresponding metric instances
        for metric_name in top_n_metric_names:
            for metric in metric_instances:
                # Handle both regular metrics and MultiMetrics
                if isinstance(metric, MultiMetric):
                    # Prefer exact submetric name match first (e.g., "SARI_F" in ["SARI_P", "SARI_F"]) and
                    # then fallback to stripped-name match when the prefix is the metric base name (e.g., "F" in ["F", "P"]).
                    metric_base_name = metric.get_name()
                    submetric_names = set(metric.get_submetric_names() or [])

                    matched = False
                    # Full-name match
                    if metric_name in submetric_names:
                        matched = True
                    else:
                        # Prefixed name -> bare submetric fallback
                        if metric_name.startswith(metric_base_name + "_"):
                            submetric_name = metric_name[len(metric_base_name) + 1:]
                            if submetric_name in submetric_names:
                                matched = True

                    if matched:
                        if metric not in top_n_metrics:
                            top_n_metrics.append(metric)
                        break
                else:
                    # For regular metrics, check the metric name
                    if metric.get_name() == metric_name:
                        if metric not in top_n_metrics:
                            top_n_metrics.append(metric)
                        break
        
        print(f"  Selected top {len(top_n_metrics)} metrics: {[m.get_name() for m in top_n_metrics]}")
        
        # Create final regression metric using only top-N metrics
        print("  Creating final regression metric with top-N metrics...")
        final_regression = type(regression_instance)(
            name=f"Autometrics_Regression_{target_measure}",
            description=f"Regression aggregator for {target_measure} using top {n} metrics",
            dataset=regression_dataset,  # Pass the dataset to avoid None error
            input_metrics=top_n_metrics  # Pass input_metrics directly to avoid dataset.get_metrics() call
        )
        
        final_regression.learn(regression_dataset, target_column=target_measure)
        
        # After training the final regression, optionally drop generated metrics with negative coefficients
        # and refit once on the kept metrics (N may shrink as a result)
        if self.drop_generated_negative_coefficients:
            try:
                kept_metrics = self._select_metrics_after_dropping_generated_negatives(
                    regression_metric=final_regression,
                    selected_metrics=top_n_metrics
                )
            except Exception as e:
                print(f"  ⚠ Warning: Drop policy failed, continuing without dropping: {e}")
                kept_metrics = top_n_metrics
            
            if kept_metrics != top_n_metrics:
                top_n_metrics = kept_metrics
                try:
                    final_regression = type(regression_instance)(
                        name=f"Autometrics_Regression_{target_measure}",
                        description=f"Regression aggregator for {target_measure} using top {len(top_n_metrics)} metrics",
                        dataset=regression_dataset,
                        input_metrics=top_n_metrics
                    )
                    final_regression.learn(regression_dataset, target_column=target_measure)
                except Exception as e:
                    print(f"  ⚠ Warning: Refit after drop failed, continuing with original final regression: {e}")
        
        # Add the final regression metric to the original dataset for user access
        # This ensures users get the final regression result in their original dataset
        dataset.add_metric(final_regression, update_dataset=True)
        
        return {
            'top_metrics': top_n_metrics,
            'regression_metric': final_regression,
            'importance_scores': importance_pairs,
            'final_regression_metric': final_regression,
            'all_metrics_importance': importance_pairs
        }

    def _generate_report_card(
        self, 
        top_metrics: List[Metric], 
        regression_metric: Any, 
        dataset: Dataset,
        target_measure: str
    ) -> str:
        """
        Generate a report card summarizing the results.
        For now, this is a simple summary with the top metrics and their importance.
        """
        report = f"""
# Autometrics Report Card

## Dataset Information
- **Dataset**: {dataset.get_name()}
- **Target Measure**: {target_measure}
- **Dataset Size**: {len(dataset.get_dataframe())} examples

## Top Metrics Selected
"""
        
        if top_metrics:
            for i, metric in enumerate(top_metrics):
                # Handle both regular metrics and MultiMetrics
                if isinstance(metric, MultiMetric):
                    # For MultiMetrics, show the metric name and submetrics
                    submetric_names = metric.get_submetric_names()
                    report += f"- **{i+1}.** {metric.get_name()} (MultiMetric: {', '.join(submetric_names)})\n"
                else:
                    # For regular metrics, show the metric name
                    report += f"- **{i+1}.** {metric.get_name()}\n"
        else:
            report += "- No metrics selected\n"
        
        report += f"""
## Regression Aggregator
- **Type**: {type(regression_metric).__name__ if regression_metric else 'None'}
- **Name**: {regression_metric.get_name() if regression_metric else 'None'}
- **Description**: {regression_metric.get_description() if regression_metric else 'None'}

## Summary
The Autometrics pipeline successfully identified the most relevant metrics for evaluating {target_measure} on the {dataset.get_name()} dataset. The selected metrics can be used individually or combined through the regression aggregator for comprehensive evaluation.
"""

        #  Enrich report with Hotelling T² selection details
        if regression_metric and hasattr(regression_metric, "get_selected_columns"):
            try:
                selected_vars = len(getattr(regression_metric, "get_selected_columns")())
            except Exception:
                selected_vars = None
            report += "\n## Hotelling T² Selection\n"
            if selected_vars is not None:
                report += f"- Selected variables: {selected_vars}\n"
            if hasattr(regression_metric, "A_star_"):
                report += f"- A*: {getattr(regression_metric, 'A_star_', None)}\n"
            if hasattr(regression_metric, "alpha_star_"):
                report += f"- alpha*: {getattr(regression_metric, 'alpha_star_', None)}\n"
            if hasattr(regression_metric, "t2_cutoff_") and getattr(regression_metric, "t2_cutoff_", None) is not None:
                try:
                    report += f"- T² cutoff: {float(getattr(regression_metric, 't2_cutoff_', 0.0)):.4f}\n"
                except Exception:
                    pass
        
        return report

    def _validate_and_adjust_retriever_config(self, retriever_kwargs: dict, dataset: Dataset, metric_bank: List[Type[Metric]], num_to_retrieve: int) -> dict:
        """
        Validate and adjust retriever configuration to ensure compatibility with num_to_retrieve.
        
        This method:
        1. Generates dynamic index paths based on metric bank content to avoid stale indices
        2. Validates that top_ks are compatible with num_to_retrieve
        3. Adjusts top_ks if necessary to ensure the final k matches num_to_retrieve
        """
        import hashlib
        from platformdirs import user_data_dir

        print(f"[Autometrics] Validating and adjusting retriever config for {dataset.get_name()} with {len(metric_bank)} metrics to retrieve {num_to_retrieve} metrics")
        
        print(metric_bank)
        print(sorted([cls.__name__ for cls in metric_bank]))
        print(''.join(sorted([cls.__name__ for cls in metric_bank])).encode())
        print(hashlib.md5(''.join(sorted([cls.__name__ for cls in metric_bank])).encode()).hexdigest()[:8])

        # Create a hash of the metric bank for cache busting
        metric_names = sorted([cls.__name__ for cls in metric_bank])
        metric_hash = hashlib.md5(''.join(metric_names).encode()).hexdigest()[:8]

        # Include document mode in index namespacing to separate description-only vs full-card indices
        doc_mode_suffix = "desc" if retriever_kwargs.get("use_description_only", False) else "card"
        
        # Get dataset name for path
        dataset_name = dataset.get_name().replace(" ", "_").replace("/", "_")
        
        # Handle PipelinedRec case
        if "index_paths" in retriever_kwargs:
            index_paths = retriever_kwargs["index_paths"]
            recommenders = retriever_kwargs.get("recommenders", [])
            top_ks = retriever_kwargs.get("top_ks", [])
            
            # Validate and adjust top_ks
            if top_ks:
                # Ensure the final k matches num_to_retrieve
                if top_ks[-1] != num_to_retrieve:
                    print(f"[Autometrics] Adjusting final top_k from {top_ks[-1]} to {num_to_retrieve} to match num_to_retrieve")
                    top_ks[-1] = num_to_retrieve
                
                # Ensure all previous steps are >= final k
                for i in range(len(top_ks) - 1):
                    if top_ks[i] < top_ks[-1]:
                        print(f"[Autometrics] Adjusting top_k[{i}] from {top_ks[i]} to {top_ks[-1]} to ensure pipeline compatibility")
                        top_ks[i] = top_ks[-1]
                
                retriever_kwargs["top_ks"] = top_ks
            
            # Generate dynamic index paths
            new_index_paths = []
            for i, path in enumerate(index_paths):
                # Only replace if path is explicitly None
                if path is None and i < len(recommenders):
                    # Get retriever type name
                    retriever_type = recommenders[i].__name__.lower()
                    # Generate dynamic path
                    dynamic_path = os.path.join(
                        user_data_dir("autometrics"),
                        f"{retriever_type}_{dataset_name}_{metric_hash}_{doc_mode_suffix}"
                    )
                    new_index_paths.append(dynamic_path)
                else:
                    # Keep existing path (even if it's a string)
                    new_index_paths.append(path)
            
            retriever_kwargs["index_paths"] = new_index_paths
            
        # Handle single retriever case
        elif "index_path" in retriever_kwargs and retriever_kwargs["index_path"] is None:
            # For single retrievers, we can get the type from self.retriever
            retriever_type = self.retriever.__name__.lower()
            dynamic_path = os.path.join(
                user_data_dir("autometrics"),
                f"{retriever_type}_{dataset_name}_{metric_hash}_{doc_mode_suffix}"
            )
            retriever_kwargs["index_path"] = dynamic_path
        # If index_path is not None, keep the existing path unchanged
        
        return retriever_kwargs

    def _dataset_has_reference_columns(self, dataset: Dataset) -> bool:
        """
        Helper to check if a dataset has any reference columns.
        Returns True if the dataset has reference columns, False otherwise.
        """
        reference_columns = dataset.get_reference_columns()
        return reference_columns is not None and len(reference_columns) > 0

    def _unload_all_metrics(self, metrics: List[Metric]):
        """
        Unload all metrics to free up GPU memory.
        This is called before retrying CUDA OOM metrics sequentially.
        """
        print(f"[Autometrics] Unloading {len(metrics)} metrics...")
        
        for metric in metrics:
            self._unload_metric(metric)
        
        # Force garbage collection and CUDA cache clearing
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[Autometrics] All metrics unloaded")
    
    def _unload_metric(self, metric: Metric):
        """
        Unload a single metric to free up GPU memory.
        Checks for _unload_model method and calls it if available.
        """
        try:
            # Check if the metric has an _unload_model method
            if hasattr(metric, '_unload_model') and callable(getattr(metric, '_unload_model')):
                metric._unload_model()
                print(f"    🔄 Unloaded {metric.get_name()}")
            elif hasattr(metric, '_unload_models') and callable(getattr(metric, '_unload_models')):
                # Some metrics use _unload_models (plural)
                metric._unload_models()
                print(f"    🔄 Unloaded {metric.get_name()}")
            else:
                # For metrics without explicit unload methods, try to clear heavy attributes
                # Preserve dspy.LM to avoid losing LLM configuration for export/reuse
                for attr in ['model', 'tokenizer', 'qg', 'qa']:
                    if not hasattr(metric, attr):
                        continue
                    try:
                        import dspy
                        if attr == 'model' and isinstance(getattr(metric, 'model', None), dspy.LM):
                            print(f"    🔄 Skipping clearing dspy.LM for {metric.get_name()}")
                            continue
                    except Exception:
                        # If type check fails, fall back to clearing
                        pass
                    setattr(metric, attr, None)
                if hasattr(metric, 'model') and getattr(metric, 'model', None) is None:
                    print(f"    🔄 Cleared model for {metric.get_name()} (not dspy.LM)")
                print(f"    🔄 Cleared heavy attributes (preserved dspy.LM if present) for {metric.get_name()}")
        except Exception as e:
            print(f"    ⚠ Warning: Failed to unload {metric.get_name()}: {e}")

    # =============================
    # Post-Regression Drop Helpers
    # =============================

    def _is_generated_metric_class(self, metric_cls: Type[Metric]) -> bool:
        """
        Best-effort detection if a metric class is a generated metric.
        Priority: subclass check against known generated base classes; fallback to module path heuristic.
        """
        try:
            from autometrics.metrics.generated.GeneratedLLMJudgeMetric import GeneratedLLMJudgeMetric
        except Exception:
            GeneratedLLMJudgeMetric = None  # type: ignore
        try:
            from autometrics.metrics.generated.GeneratedExampleRubric import GeneratedExampleRubric
        except Exception:
            GeneratedExampleRubric = None  # type: ignore
        try:
            from autometrics.metrics.generated.GeneratedOptimizedJudge import GeneratedOptimizedJudge
        except Exception:
            GeneratedOptimizedJudge = None  # type: ignore
        try:
            from autometrics.metrics.generated.GeneratedGEvalMetric import GeneratedGEvalMetric
        except Exception:
            GeneratedGEvalMetric = None  # type: ignore
        try:
            from autometrics.metrics.generated.GeneratedPrometheus import GeneratedPrometheus
        except Exception:
            GeneratedPrometheus = None  # type: ignore
        try:
            from autometrics.metrics.generated.GeneratedFinetunedMetric import GeneratedFinetunedMetric
        except Exception:
            GeneratedFinetunedMetric = None  # type: ignore
        try:
            from autometrics.metrics.generated.GeneratedCodeMetric import GeneratedCodeMetric
        except Exception:
            GeneratedCodeMetric = None  # type: ignore
        try:
            from autometrics.metrics.generated.GeneratedRefBasedMetric import GeneratedRefBasedMetric
        except Exception:
            GeneratedRefBasedMetric = None  # type: ignore
        try:
            from autometrics.metrics.generated.GeneratedRefFreeMetric import GeneratedRefFreeMetric
        except Exception:
            GeneratedRefFreeMetric = None  # type: ignore

        generated_bases = [
            b for b in [
                GeneratedLLMJudgeMetric,
                GeneratedExampleRubric,
                GeneratedOptimizedJudge,
                GeneratedGEvalMetric,
                GeneratedPrometheus,
                GeneratedFinetunedMetric,
                GeneratedCodeMetric,
                GeneratedRefBasedMetric,
                GeneratedRefFreeMetric,
            ] if b is not None
        ]

        try:
            if any(issubclass(metric_cls, b) for b in generated_bases):
                return True
        except Exception:
            pass

        # Fallback heuristic: module path contains ".metrics.generated."
        try:
            module_path = getattr(metric_cls, "__module__", "") or ""
            if ".metrics.generated." in module_path:
                return True
        except Exception:
            pass
        return False

    def _select_metrics_after_dropping_generated_negatives(self, regression_metric: Any, selected_metrics: List[Metric]) -> List[Metric]:
        """
        Return a filtered list with generated metrics removed if their coefficients are negative.
        Existing metrics are preserved. If all selected metrics are generated and all have
        negative coefficients, return the original list (do not drop any).
        """
        try:
            model = getattr(regression_metric, 'model', None)
            if model is None or not hasattr(model, 'coef_'):
                return selected_metrics

            # Feature names used by the fitted regression (preferred order)
            try:
                feature_names = list(regression_metric.get_selected_columns())
            except Exception:
                try:
                    feature_names = list(regression_metric.get_input_columns())
                except Exception:
                    return selected_metrics

            if not feature_names:
                return selected_metrics

            coef = getattr(model, 'coef_', None)
            if coef is None:
                return selected_metrics

            # Normalize to 1D array of length n_features
            import numpy as _np
            coef_arr = _np.array(coef)
            if coef_arr.ndim == 2 and coef_arr.shape[0] == 1:
                coef_vec = coef_arr[0]
            else:
                coef_vec = coef_arr
            if coef_vec.shape[0] != len(feature_names):
                # Shape mismatch; safer to abort
                return selected_metrics

            name_to_idx = {name: i for i, name in enumerate(feature_names)}

            # Map metrics to their feature indices
            def _features_for_metric(metric: Metric) -> List[int]:
                try:
                    if isinstance(metric, MultiMetric):
                        cols = metric.get_submetric_names() or []
                    else:
                        cols = [metric.get_name()]
                except Exception:
                    cols = []
                return [name_to_idx[c] for c in cols if c in name_to_idx]

            metric_indices = {m: _features_for_metric(m) for m in selected_metrics}

            # Determine which generated metrics have any negative coefficient
            generated_negative_metrics = []
            for m, idxs in metric_indices.items():
                if not idxs:
                    continue
                try:
                    is_generated = self._is_generated_metric_class(type(m))
                except Exception:
                    is_generated = False
                if not is_generated:
                    continue
                # Any negative among this metric's coefficients?
                try:
                    if any(float(coef_vec[i]) < 0.0 for i in idxs):
                        generated_negative_metrics.append(m)
                except Exception:
                    continue

            if not generated_negative_metrics:
                return selected_metrics

            # Sanity check: if all selected metrics are generated and all have negative coeffs, keep all
            try:
                all_selected_generated = all(self._is_generated_metric_class(type(m)) for m in selected_metrics if m is not None)
                if all_selected_generated and len(generated_negative_metrics) == len([m for m in selected_metrics if m is not None]):
                    print("[Autometrics] All selected metrics are generated with negative coefficients; skipping drop policy")
                    return selected_metrics
            except Exception:
                pass
            # Build and return filtered list
            kept = [m for m in selected_metrics if m not in generated_negative_metrics]
            try:
                if len(kept) < len(selected_metrics):
                    dropped_metric_names = []
                    for m in selected_metrics:
                        if m in generated_negative_metrics:
                            try:
                                dropped_metric_names.append(m.get_name())
                            except Exception:
                                dropped_metric_names.append(type(m).__name__)
                    print(f"[Autometrics] Dropping generated metrics with negative coefficients: {dropped_metric_names}")
            except Exception:
                pass
            return kept

        except Exception as e:
            print(f"[Autometrics] Warning: Error applying drop policy: {e}")
            return selected_metrics

# =============================
# End of Autometrics Pipeline Scaffold
# ============================= 
