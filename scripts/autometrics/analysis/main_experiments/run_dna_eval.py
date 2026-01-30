from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric
from typing import ClassVar, List, Dict, Any, Optional, Tuple
import dspy
import os
import sys
import argparse
import logging
import time
import pandas as pd
import numpy as np
from scipy import stats

# Ensure repository root is on path when invoked directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autometrics.dataset.Dataset import Dataset
from autometrics.experiments.correlation.correlation import CorrelationExperiment, correlation_func_from_name

# ================================
# DSPy Signatures
# ================================

class MetricGen(dspy.Signature):
    """You are a helpful assistant in evaluating the quality of the outputs for a given instruction.

Please propose three concise questions about whether a potential output is a good output for a given instruction. Another assistant will evaluate different aspects of the output by answering all the questions.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.

# Requirements for Your Output:
(1) The questions should **specifically** target the given instruction instead of some general standards, so the questions may revolve around key points of the instruction. 
(2) You should directly give the questions without any other words.
(3) Questions are presented from most important to least important."""
    task_description: str = dspy.InputField(desc="The task description.")
    input: str = dspy.InputField(desc="The input text.")
    questions: List[str] = dspy.OutputField(desc="The questions.")

class EvalMetric(dspy.Signature):
    """You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to evaluate the output for the given instruction.

Score the given output for the given instruction on a scale of 1 to 10 by considering only the given aspect.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction, then consider its helpfulness, accuracy, level of detail, harmlessness, etc.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.
(3) You should avoid any potential bias and your judgment should be as objective as possible."""
    task_description: str = dspy.InputField(desc="The task description.")
    input: str = dspy.InputField(desc="The input text.")
    output: str = dspy.InputField(desc="The output text.")
    question: str = dspy.InputField(desc="The question to evaluate the output on.")
    score: float = dspy.OutputField(desc="The score.")

class WeightGen(dspy.Signature):
    """You are a helpful assistant in evaluating the quality of the outputs for a given instruction.

Please propose respective importance weightage for three aspects in evaluating the output.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the output honestly/precisely/closely executes the instruction.
(2) Outputs should NOT contain more/less than what the instruction asks for, as such outputs do NOT precisely execute the instruction.

# Requirements for Your Output:
(1) The weightages should be in percentage form and sum up to 100%.  Do not include the percentage sign.
(2) You should directly give the weightages without any other words."""
    task_description: str = dspy.InputField(desc="The task description.")
    input: str = dspy.InputField(desc="The input text.")
    questions: List[str] = dspy.InputField(desc="The questions to evaluate the output on.")
    weights: List[float] = dspy.OutputField(desc="The weights for how important each question is.")

# ================================
# DNAProgram
# ================================

class DNAProgram(dspy.Module):
    def __init__(self):
        super(DNAProgram, self).__init__()
        self.metric_gen = dspy.ChainOfThought(MetricGen)
        self.eval_metric = dspy.ChainOfThought(EvalMetric)
        self.weight_gen = dspy.ChainOfThought(WeightGen)
        
    def forward(self, task_description, input, output):
        questions = self.metric_gen(task_description=task_description, input=input).questions
        scores = []
        for question in questions:
            scores.append(self.eval_metric(task_description=task_description, input=input, output=output, question=question).score)
        weights = self.weight_gen(task_description=task_description, input=input, questions=questions).weights
        weights = [weight / sum(weights) for weight in weights] # Normalize weights to sum to 1
        score = sum(scores[i] * weights[i] for i in range(len(scores)))
        return score

# ================================
# DNAEval Metric
# ================================

DEFAULT_MODEL = dspy.LM("openai/gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY"))

class DNAEval(ReferenceFreeMetric):
    """"""
    
    # Resource usage statistics (in megabytes)
    gpu_mem: ClassVar[float] = 0.0  # in MB
    cpu_mem: ClassVar[float] = 0.0  # in MB
    description: ClassVar[str] = "Decompose and Aggregate framework for using LLM-as-a-judge in an effective and interpretable way."

    def __init__(self, name="DNAEvalRefFree", description="Decompose and Aggregate framework for using LLM-as-a-judge in an effective and interpretable way.", model=DEFAULT_MODEL, task_description=None, **kwargs):
        self.model = model
        self.task_description = task_description
        super().__init__(name, description, **kwargs)

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        program = DNAProgram()

        dspy_inputs = [dspy.Example(input=input, output=output, task_description=self.task_description).with_inputs('input', 'output', 'task_description') for input, output in zip(inputs, outputs)]
        with dspy.settings.context(lm=self.model):
            results = program.batch(dspy_inputs)

        def safe_float(r):
            try:
                return float(r)
            except Exception:
                return 0.0

        return [safe_float(r) for r in results]

    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Calculate the metric
        """
        program = DNAProgram()

        with dspy.settings.context(lm=self.model):
            result = program(input=input, output=output, task_description=self.task_description)
        try:
            return float(result)
        except Exception:
            return 0.0

# ================================
# Experiment Runner Utilities
# ================================


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    logger = logging.getLogger(__name__)

    if not verbose:
        # Quiet noisy libs
        logging.getLogger('dspy').setLevel(logging.WARNING)
        logging.getLogger('diskcache').setLevel(logging.WARNING)
        logging.getLogger('diskcache.core').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('tokenizers').setLevel(logging.WARNING)
        logging.getLogger('autometrics').setLevel(logging.WARNING)
        logging.getLogger('autometrics.metrics').setLevel(logging.WARNING)
        logging.getLogger('autometrics.experiments').setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.WARNING)

        logger.setLevel(logging.INFO)
        if logger.level > logging.WARNING:
            logger.setLevel(logging.WARNING)

    return logger


def load_dataset(dataset_name: str) -> Dataset:
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


def get_available_datasets_measures() -> List[Tuple[str, str]]:
    datasets_measures: List[Tuple[str, str]] = []
    dataset_configs: Dict[str, List[str]] = {
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
        "RealHumanEval": ["accepted"],
        "Design2Code": ["score"],
        "ICLR": ["recommendation"]
    }

    for dataset_name, measures in dataset_configs.items():
        for measure in measures:
            datasets_measures.append((dataset_name, measure))
    return datasets_measures


def create_llm_model(model_name: str, api_base: Optional[str] = None, seed: int = 42) -> dspy.LM:
    temperature = 0.0001 * seed
    if model_name == "gpt4o_mini":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Please export OPENAI_API_KEY before running with gpt4o_mini.")
        return dspy.LM("openai/gpt-4o-mini", api_key=api_key, temperature=temperature)
    elif model_name == "gpt5_mini":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Please export OPENAI_API_KEY before running with gpt5_mini.")
        return dspy.LM("openai/gpt-5-mini", api_key=api_key, temperature=temperature)
    elif model_name == "qwen3_32b":
        base_url = api_base or "http://localhost:7510/v1"
        return dspy.LM("litellm_proxy/Qwen/Qwen3-32B", api_base=base_url, temperature=temperature, max_tokens=4096)
    elif model_name == "llama3_70b":
        base_url = api_base or "http://localhost:7510/v1"
        return dspy.LM("litellm_proxy/meta-llama/Llama-3.3-70B-Instruct", api_base=base_url, temperature=temperature)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_statistics(values: List[float]) -> Dict[str, float]:
    valid_values = [v for v in values if not pd.isna(v)]
    n = len(valid_values)
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "ci_range": np.nan, "num_successful_runs": 0}
    mean_val = np.mean(valid_values)
    if n == 1:
        return {"mean": mean_val, "std": 0.0, "ci_lower": mean_val, "ci_upper": mean_val, "ci_range": 0.0, "num_successful_runs": n}
    std_val = np.std(valid_values, ddof=1)
    alpha = 0.05
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_value * std_val / np.sqrt(n)
    return {"mean": mean_val, "std": std_val, "ci_lower": mean_val - margin_error, "ci_upper": mean_val + margin_error, "ci_range": margin_error, "num_successful_runs": n}


def format_mean_ci(mean: float, ci_range: float) -> str:
    if np.isnan(mean) or np.isnan(ci_range):
        return "N/A"
    return f"{mean:.4f} ± {ci_range:.4f}"


def sort_columns_for_output(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base_columns = ['dataset', 'measure', 'metric', 'metric_class', 'num_successful_runs', 'errors']
    correlation_columns = []
    p_value_columns = []
    for col in df.columns:
        if col.startswith('seed_') and col.endswith('_correlation'):
            correlation_columns.append(col)
        elif col.startswith('seed_') and col.endswith('_p_value'):
            p_value_columns.append(col)
    def extract_seed_number(col_name):
        return int(col_name.split('_')[1])
    correlation_columns.sort(key=extract_seed_number)
    p_value_columns.sort(key=extract_seed_number)
    stats_columns = [
        'mean_correlation', 'std_correlation', 'ci_lower_correlation', 'ci_upper_correlation',
        'mean_p_value', 'std_p_value', 'ci_lower_p_value', 'ci_upper_p_value'
    ]
    final_columns: List[str] = []
    for col in base_columns:
        if col in df.columns:
            final_columns.append(col)
    final_columns.extend(correlation_columns)
    final_columns.extend(p_value_columns)
    for col in stats_columns:
        if col in df.columns:
            final_columns.append(col)
    for col in df.columns:
        if col not in final_columns:
            final_columns.append(col)
    return df[final_columns]


def save_results(results: Dict[str, Any], output_file: str, logger: logging.Logger):
    try:
        df = pd.DataFrame(results)
        df = sort_columns_for_output(df)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_file}: {e}")
        raise


def merge_with_existing_results(new_results: List[Dict], output_file: str, logger: logging.Logger) -> List[Dict]:
    if not os.path.exists(output_file):
        logger.info(f"No existing results file found at {output_file}")
        return new_results
    try:
        existing_df = pd.read_csv(output_file)
        logger.info(f"Loaded {len(existing_df)} existing results from {output_file}")
        new_df = pd.DataFrame(new_results)
        if existing_df.empty:
            logger.info("Existing file is empty, using new results")
            return new_results
        merge_keys = ['dataset', 'measure', 'metric']
        merged_results: List[Dict[str, Any]] = []
        for _, new_row in new_df.iterrows():
            mask = True
            for key in merge_keys:
                mask = mask & (existing_df[key] == new_row[key])
            matching_rows = existing_df[mask]
            if len(matching_rows) > 0:
                existing_row = matching_rows.iloc[0].copy()
                merged_row = existing_row.to_dict()
                for col in new_row.index:
                    if col.startswith('seed_') and col.endswith(('_correlation', '_p_value')):
                        merged_row[col] = new_row[col]
                all_correlations: List[float] = []
                all_p_values: List[float] = []
                all_errors: List[str] = []
                for col, val in merged_row.items():
                    if isinstance(col, str) and col.startswith('seed_') and col.endswith('_correlation') and pd.notna(val):
                        all_correlations.append(val)
                    if isinstance(col, str) and col.startswith('seed_') and col.endswith('_p_value') and pd.notna(val):
                        all_p_values.append(val)
                if pd.notna(existing_row.get('errors', '')) and existing_row['errors']:
                    all_errors.extend(str(existing_row['errors']).split('; '))
                if pd.notna(new_row.get('errors', '')) and new_row['errors']:
                    all_errors.extend(str(new_row['errors']).split('; '))
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
                logger.info(f"Merged results for {new_row['dataset']}.{new_row['measure']}")
            else:
                merged_results.append(new_row.to_dict())
                logger.info(f"Added new result for {new_row['dataset']}.{new_row['measure']}")
        for _, existing_row in existing_df.iterrows():
            mask = True
            for key in merge_keys:
                mask = mask & (new_df[key] == existing_row[key])
            if len(new_df[mask]) == 0:
                merged_results.append(existing_row.to_dict())
        merged_df = pd.DataFrame(merged_results)
        merged_df = sort_columns_for_output(merged_df)
        return merged_df.to_dict('records')
    except Exception as e:
        logger.error(f"Error merging with existing results: {e}")
        logger.info("Using new results only")
        return new_results


def run_single_metric_seed(
    dataset_name: str,
    measure: str,
    metric_instance: DNAEval,
    seed: int,
    correlation_funcs: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Dict[str, float]]:
    try:
        dataset = load_dataset(dataset_name)
        _, _, test_dataset = dataset.load_permanent_splits()
        experiment = CorrelationExperiment(
            name=f"DNAEval Test - {dataset_name} - {measure} - {metric_instance.name}",
            description=f"Testing DNAEval correlation for {metric_instance.name} on {dataset_name}",
            metrics=[metric_instance],
            output_dir=f"/tmp/dna_eval_test_{seed}",
            dataset=test_dataset,
            correlation_funcs=correlation_funcs,
            seed=seed,
            should_split=False
        )
        all_correlations = experiment.run(print_results=False)
        results_by_func: Dict[str, Dict[str, float]] = {}
        for func_name, correlations_for_func in all_correlations.items():
            if measure not in correlations_for_func:
                raise ValueError(f"Measure {measure} not found in correlation results for {func_name}")
            df_corr = correlations_for_func[measure]
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
        raise


def main():
    parser = argparse.ArgumentParser(description="DNAEval correlation stability experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt4o_mini"],
        help="LLM models to use (e.g., gpt4o_mini qwen3_32b)"
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL for non-OpenAI models (e.g., http://localhost:7510/v1)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/main_runs/baselines",
        help="Base output directory for results"
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
        help="Correlation function(s): 'pearson', 'spearman', 'kendall', or 'all'"
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
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    if args.correlation.lower() == "all":
        correlation_types = ["kendall", "pearson", "spearman"]
    else:
        correlation_types = [args.correlation.lower()]
    valid_correlations = {"pearson", "spearman", "kendall"}
    for ct in correlation_types:
        if ct not in valid_correlations:
            print(f"Unknown correlation function: {ct}")
            return 1

    logger = setup_logging(args.verbose)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "dna_eval_sub_results"), exist_ok=True)

    all_dataset_measures = get_available_datasets_measures()
    if args.dataset:
        allowed_datasets = set(args.dataset)
        all_dataset_measures = [(d, m) for d, m in all_dataset_measures if d in allowed_datasets]
        logger.info(f"Filtered to datasets: {args.dataset}")
    if args.measure:
        allowed_measures = set(args.measure)
        all_dataset_measures = [(d, m) for d, m in all_dataset_measures if m in allowed_measures]
        logger.info(f"Filtered to measures: {args.measure}")
    if not all_dataset_measures:
        logger.error("No dataset-measure combinations to process after filtering")
        return 1

    logger.info("Starting DNAEval correlation stability experiments")
    logger.info(f"Models: {args.models}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Correlations: {correlation_types}")

    # Group measures by dataset for dataset-specific output files
    datasets_measures: Dict[str, List[str]] = {}
    for dataset_name, measure in all_dataset_measures:
        datasets_measures.setdefault(dataset_name, []).append(measure)

    for correlation_type in correlation_types:
        correlation_func = correlation_func_from_name(correlation_type)
        correlation_funcs = {correlation_type: correlation_func}

        for model_name in args.models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {correlation_type.upper()} for model {model_name}")
            logger.info(f"{'='*60}")

            all_results: List[Dict[str, Any]] = []
            dataset_output_files: List[str] = []

            for dataset_name, measures in datasets_measures.items():
                logger.info(f"\n--- Processing dataset: {dataset_name} ---")
                dataset_output_file = os.path.join(
                    args.output_dir,
                    "dna_eval_sub_results",
                    f"dna_eval_{model_name}_{correlation_type}_{dataset_name}.csv"
                )
                dataset_output_files.append(dataset_output_file)
                dataset_results: List[Dict[str, Any]] = []

                # Obtain dataset for task description
                try:
                    dataset_instance = load_dataset(dataset_name)
                except Exception as e:
                    logger.warning(f"Failed to load dataset {dataset_name}: {e}")
                    continue

                for measure in measures:
                    base_metric_name = f"DNAEval-{model_name}-{measure}"
                    correlations: List[float] = []
                    p_values: List[float] = []
                    errors: List[str] = []

                    for seed in args.seeds:
                        try:
                            logger.info(f"  Running seed {seed}...")
                            seed_model = create_llm_model(model_name, args.api_base, seed)
                            task_description = dataset_instance.get_task_description() or f"Evaluate {measure} for {dataset_name} dataset"
                            metric = DNAEval(
                                name=f"{base_metric_name}-seed{seed}",
                                model=seed_model,
                                task_description=task_description
                            )
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
                        except Exception as e:
                            error_msg = f"Seed {seed}: {str(e)}"
                            errors.append(error_msg)
                            logger.error(f"    Failed: {error_msg}")
                            correlations.append(np.nan)
                            p_values.append(np.nan)

                    corr_stats = compute_statistics([abs(c) for c in correlations if not pd.isna(c)])
                    pval_stats = compute_statistics(p_values)

                    result: Dict[str, Any] = {
                        'dataset': dataset_name,
                        'measure': measure,
                        'metric': base_metric_name,
                        'metric_class': DNAEval.__name__,
                        'num_successful_runs': corr_stats['num_successful_runs'],
                        'errors': '; '.join(errors) if errors else ''
                    }
                    for i, seed in enumerate(args.seeds):
                        result[f'seed_{seed}_correlation'] = correlations[i] if i < len(correlations) else np.nan
                        result[f'seed_{seed}_p_value'] = p_values[i] if i < len(p_values) else np.nan
                    result.update({
                        'mean_correlation': corr_stats['mean'],
                        'std_correlation': corr_stats['std'],
                        'ci_lower_correlation': corr_stats['ci_lower'],
                        'ci_upper_correlation': corr_stats['ci_upper'],
                        'mean_±_ci': format_mean_ci(corr_stats['mean'], corr_stats['ci_range'])
                    })
                    result.update({
                        'mean_p_value': pval_stats['mean'],
                        'std_p_value': pval_stats['std'],
                        'ci_lower_p_value': pval_stats['ci_lower'],
                        'ci_upper_p_value': pval_stats['ci_upper']
                    })
                    dataset_results.append(result)
                    logger.info(
                        f"  Completed: mean_corr={corr_stats['mean']:.4f}, CI=[{corr_stats['ci_lower']:.4f}, {corr_stats['ci_upper']:.4f}], mean_p_val={pval_stats['mean']:.4f}"
                    )

                if dataset_results:
                    save_results(dataset_results, dataset_output_file, logger)
                    all_results.extend(dataset_results)
                    logger.info(f"Dataset {dataset_name} results saved to {dataset_output_file}")
                else:
                    logger.warning(f"No results generated for dataset {dataset_name}")

            merged_output_file = os.path.join(
                args.output_dir,
                f"dna_eval_{model_name}_{correlation_type}.csv"
            )
            if all_results:
                merged_results = merge_with_existing_results(all_results, merged_output_file, logger)
                save_results(merged_results, merged_output_file, logger)

                print(f"\nSummary of DNAEval {correlation_type.upper()} Correlation Results:")
                print(f"Models: {model_name}")
                print(f"Seeds: {args.seeds}")
                print(f"Merged results saved to: {merged_output_file}")
                print(f"Dataset-specific results saved to:")
                for dataset_file in dataset_output_files:
                    print(f"  {dataset_file}")
                print(f"\nTop 5 most stable correlations:")
                df_results = pd.DataFrame(merged_results)
                df_results['abs_mean'] = df_results['mean_correlation'].abs()
                df_top = df_results.nlargest(5, 'abs_mean')
                for _, row in df_top.iterrows():
                    mean_corr = row['mean_correlation']
                    ci_width = (row['ci_upper_correlation'] - row['ci_lower_correlation']) / 2
                    print(f"  {row['dataset']}.{row['measure']}: {mean_corr:.4f} ± {ci_width:.4f} (95% CI: [{row['ci_lower_correlation']:.4f}, {row['ci_upper_correlation']:.4f}])")
            else:
                logger.error(f"No results generated for {correlation_type}")

    logger.info("DNAEval correlation analysis completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())