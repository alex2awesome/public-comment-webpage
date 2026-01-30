from autometrics.generator.Generator import Generator
import dspy
from typing import Optional, Callable, List, Any
import random
import time
import re
import pandas as pd

# Utilities for dataset formatting and processing
from autometrics.util.normalize import find_distinct_quintiles_with_min_max, map_to_bucket

# Import optimization utilities that will now run in the Generator
from dspy.evaluate import Evaluate

# Import evaluation functions
def exact_match_rounded(x, y):
    return int(round(x) == round(y))

def inverse_distance(x, y):
    if x == y:
        return 1
    return 1 / (abs(x - y) + 1)

def get_wrapped_metric(metric_func):
    def wrapped_metric(example, pred, trace=None):
        return metric_func(example.score, pred.score)
    return wrapped_metric

# Utilities to avoid duplication and enable reuse across generators
from autometrics.generator.utils import (
    get_good_bad_examples,
    generate_axes_of_variation,
    smart_limit_examples_for_context,
    get_max_context_tokens,
    is_context_length_error,
)

class LLMAsAJudgeSignature(dspy.Signature):
    """Given an input text, the task description that the model was trying to follow, and a measure to rate the text on, return a score on this measure."""
    text = dspy.InputField(desc="The input text that we want to rate.")
    task_description = dspy.InputField(desc="A description of the task that the model was trying to solve when it generated the text. Could be left blank if not available.")
    measure = dspy.InputField(desc="The measure that we want to rate the text on.")
    suggested_range = dspy.InputField(desc="The suggested range of possible values for the measure.")
    score = dspy.OutputField(desc="The score that the text should receive on this measure.")

class LLMAsAJudge(dspy.Module):
    def __init__(self):
        super(LLMAsAJudge, self).__init__()
        self.generate_score = dspy.ChainOfThought(LLMAsAJudgeSignature)

    def forward(self, text, measure, suggested_range=(1,5), task_description=None):
        if task_description is None:
            task_description = "None"
        suggested_range_str = f"{suggested_range[0]} to {suggested_range[1]}"
        score = self.generate_score(task_description=task_description, text=text, measure=measure, suggested_range=suggested_range_str).score
        # Convert the string score to a float by stripping any additional text and converting to a float
        if '\n' in score:
            score = score.split('\n')[0]
        try:
            score = float(score.strip())
        except:
            score = 0.0

        return dspy.Prediction(text=text, measure=measure, score=score)

class LLMJudgeExampleProposer(Generator):
    """Generate *LLM-as-a-judge* metrics with optimized examples by proposing axes of variation
    and selecting diverse examples across score ranges.

    The class conforms to the new *Generator* interface which includes an
    optional *generator_llm* (used here to generate axes) as well as the
    ability to specify a custom executor class. The executor class is automatically
    determined based on whether the dataset has reference columns.
    
    This differs from BasicLLMJudgeProposer by:
    1. **Example Selection**: Uses quintile-based bucketing to select diverse examples
    2. **Optimization**: Performs few-shot optimization with selected examples
    3. **Evaluation**: Uses evaluation functions to optimize example selection
    """

    def __init__(
        self,
        name: str = "LLMJudgeExampleProposer",
        description: str = "Propose LLM-as-a-Judge measures with optimized examples based on the dataset and task description",
        generator_llm: Optional[dspy.LM] = None,
        executor_class: type | None = None,
        executor_kwargs: dict | None = None,
        attempts: int = 5,
        examples_per_range: int = 2,
        seed: int = 42,
        eval_function_name: str = 'inverse_distance',
        custom_eval_function: Optional[Callable] = None,
        max_workers: int = 16,
        max_optimization_samples: int = 100,  # Limit dataset size for optimization
        truncate_chars: Optional[int] = None,
    ):

        super().__init__(
            name=name,
            description=description,
            generator_llm=generator_llm,
            executor_class=executor_class,
            executor_kwargs=executor_kwargs or {},
            truncate_chars=truncate_chars,
        )

        self.attempts = attempts
        self.examples_per_range = examples_per_range  
        self.seed = seed
        self.eval_function_name = eval_function_name
        self.custom_eval_function = custom_eval_function
        self.max_workers = max_workers
        self.max_optimization_samples = max_optimization_samples

        # Set up evaluation function
        if custom_eval_function is not None:
            self.eval_function = custom_eval_function
        elif eval_function_name == 'exact_match_rounded':
            self.eval_function = exact_match_rounded
        elif eval_function_name == 'inverse_distance':
            self.eval_function = inverse_distance
        else:
            self.eval_function = inverse_distance  # Default fallback

        # Set random seed for reproducible example selection
        random.seed(seed)

        # Guarantee attribute is a dictionary for ** expansion later
        if self.executor_kwargs is None:
            self.executor_kwargs = {}

        if executor_kwargs and 'model' in executor_kwargs:
            judge_model = executor_kwargs['model']
            if judge_model and hasattr(judge_model, 'name'):
                self.judge_model_name = judge_model.name
            elif judge_model and hasattr(judge_model, 'model'):
                if hasattr(judge_model.model, 'name'):
                    self.judge_model_name = judge_model.model.name
                else:
                    self.judge_model_name = judge_model.model.split('/')[-1]
        else:
            self.judge_model_name = "UnknownLLM"

        # Keep a reference to judge_model for executor_kwargs convenience
        self.judge_model = executor_kwargs.get('model') if executor_kwargs else None

    def _determine_executor_class(self, dataset):
        """Determine whether to use reference-based or reference-free metrics based on dataset."""
        reference_columns = dataset.get_reference_columns()
        has_references = reference_columns is not None and len(reference_columns) > 0
        
        # Import here to avoid circular imports
        from autometrics.metrics.generated.GeneratedExampleRubric import (
            GeneratedRefBasedExampleRubricMetric,
            GeneratedRefFreeExampleRubricMetric
        )
        
        if has_references:
            return GeneratedRefBasedExampleRubricMetric
        else:
            return GeneratedRefFreeExampleRubricMetric
    
    def _prepare_dataset_bucketted(self, dataset, target_column, task_description, metric_name, formatter, suggested_range=(1,5)):
        """Prepare dataset by bucketing examples into quintiles for diverse example selection."""
        buckets = [[] for _ in range(5)]
        dspy_dataset = []

        # Filter out NaN values first to prevent optimization issues
        df = dataset.get_dataframe()
        df_clean = df.dropna(subset=[target_column])
        
        if len(df_clean) == 0:
            print(f"Warning: No valid (non-NaN) values found in target column '{target_column}'")
            return buckets, dspy_dataset
            
        print(f"Filtered dataset: {len(df)} -> {len(df_clean)} examples (removed {len(df) - len(df_clean)} NaN values)")
        
        # Limit dataset size for optimization to speed up processing
        if len(df_clean) > self.max_optimization_samples:
            df_clean = df_clean.sample(n=self.max_optimization_samples, random_state=self.seed)
            print(f"Limited optimization dataset to {self.max_optimization_samples} samples for faster processing (seed={self.seed})")
        
        # Create a simple dataset-like object for quintile calculation with clean data
        class CleanDataset:
            def __init__(self, df):
                self.df = df
            def get_dataframe(self):
                return self.df
        
        clean_dataset = CleanDataset(df_clean)
        quintiles = find_distinct_quintiles_with_min_max(clean_dataset, target_column)

        for i, row in df_clean.iterrows():
            # Skip NaN values (double check)
            if pd.isna(row[target_column]):
                continue
                
            bucket_idx = map_to_bucket(row[target_column], quintiles)
            
            # Format the row using the dataset's input/output structure
            formatted_example = {
                'text': formatter((i, row)),  # Pass tuple as expected by formatter
                'task_description': task_description,
                'measure': metric_name,  # Use 'measure' to match DSPy signature
                'suggested_range': suggested_range,
                'score': row[target_column]
            }
            
            buckets[bucket_idx].append(formatted_example)

            dspy_dataset.append(
                dspy.Example(
                    text=formatter((i, row)),  # Pass tuple as expected by formatter
                    task_description=task_description,
                    measure=metric_name,  # Use 'measure' to match DSPy signature
                    suggested_range=suggested_range,
                    score=row[target_column]
                ).with_inputs('text', 'task_description', 'measure', 'suggested_range')
            )

        return buckets, dspy_dataset

    def _optimize_examples(self, buckets, trainset, judge_model, seed):
        """Optimize example selection using multiple attempts and evaluation - RUNS IN GENERATOR."""
        if not buckets or not trainset or not self.eval_function:
            print("Warning: Cannot optimize examples - missing buckets, trainset, or eval function")
            return []

        print(f"Optimizing examples with {self.attempts} attempts, {self.examples_per_range} examples per range (seed={seed})...")

        # Set temperature based on seed for cache busting
        temperature = 0.0001 * seed
        
        # Create a program for optimization
        program = LLMAsAJudge()
        
        # Create evaluator
        evaluate = Evaluate(
            devset=trainset, 
            metric=get_wrapped_metric(self.eval_function), 
            num_threads=min(16, self.max_workers), 
            display_progress=True, 
            display_table=False
        )

        # Helper to run evaluate with rate-limit wait-and-retry
        def _run_evaluate_with_rate_limit(_program):
            retries_left = 3
            while True:
                try:
                    return evaluate(_program)
                except Exception as e:
                    msg = str(e)
                    if (
                        'RateLimitError' in msg
                        or 'Rate limit' in msg
                        or 'rate limit' in msg
                        or '429' in msg
                        or 'Too Many Requests' in msg
                        or 'rate_limit_exceeded' in msg
                        or 'quota' in msg
                        or 'exceeded your current quota' in msg
                    ) and retries_left > 0:
                        wait_seconds = 30.0
                        try:
                            m = re.search(r"Please try again in\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds|s|sec|seconds)\b", msg, re.IGNORECASE)
                            if m:
                                _val = float(m.group(1))
                                _unit = m.group(2).lower()
                                wait_seconds = _val / 1000.0 if _unit.startswith('m') else _val
                            else:
                                m = re.search(r"Retry-After\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds|s|sec|seconds)?", msg, re.IGNORECASE)
                                if m:
                                    _val = float(m.group(1))
                                    _unit = (m.group(2) or 's').lower()
                                    wait_seconds = _val / 1000.0 if _unit.startswith('m') else _val
                        except Exception:
                            pass
                        wait_seconds = max(0.0, min(wait_seconds, 30.0))
                        retries_left -= 1
                        time.sleep(wait_seconds)
                        continue
                    raise

        # Smart limit examples for context window constraints
        # Get model name for token estimation
        model_name = "gpt-3.5-turbo"  # Default
        if self.judge_model and hasattr(self.judge_model, 'model'):
            model_name = self.judge_model.model
        elif self.generator_llm and hasattr(self.generator_llm, 'model'):
            model_name = self.generator_llm.model
        
        # Estimate total examples we would use
        total_examples = sum(len(bucket) for bucket in buckets if bucket)
        target_examples = min(total_examples, self.attempts * self.examples_per_range * len(buckets))
        
        # Extract example texts for token estimation
        example_texts = []
        for bucket in buckets:
            for example in bucket:
                if hasattr(example, 'text'):
                    example_texts.append(example.text)
                elif hasattr(example, 'input') and hasattr(example.input, 'text'):
                    example_texts.append(example.input.text)
                else:
                    example_texts.append(str(example))
        
        # Smart limit examples
        limited_examples = smart_limit_examples_for_context(
            example_texts,
            "Task description for optimization",  # We'll use actual task description in optimization
            model_name,
            target_examples=target_examples,
            dspy_overhead_tokens=4096,
            output_tokens=2048,
            safety_margin=2000
        )
        
        # Calculate how many examples per bucket we can use
        max_examples_per_bucket = max(1, len(limited_examples) // len(buckets)) if buckets else 1
        adjusted_examples_per_range = min(self.examples_per_range, max_examples_per_bucket)
        
        if adjusted_examples_per_range < self.examples_per_range:
            print(f"Adjusted examples per range from {self.examples_per_range} to {adjusted_examples_per_range} due to context constraints")
        
        # Generate different demo sets with adjusted limits
        random.seed(seed)  # Reseed for consistent behavior
        demosets = [[] for _ in range(self.attempts)]
        for i in range(self.attempts):
            for bucket in buckets:
                if bucket:  # Only sample from non-empty buckets
                    sample = random.sample(bucket, min(adjusted_examples_per_range, len(bucket)))
                    demosets[i].extend(sample)

        # Find the best demo set
        best_score = -100000
        best_examples = []
        
        # Set DSPy context with temperature for cache busting
        with dspy.settings.context(lm=judge_model, temperature=temperature):
            for i, demoset in enumerate(demosets):
                if not demoset:  # Skip empty demo sets
                    continue
                    
                try:
                    new_program = program.deepcopy()
                    
                    # Use the correct DSPy path we discovered
                    new_program.generate_score.predict.demos = demoset
                    
                    score = _run_evaluate_with_rate_limit(new_program)
                    
                    print(f"Attempt {i+1}/{self.attempts}: Score = {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_examples = demoset
                        print(f"New best score: {best_score:.4f}")
                except Exception as e:
                    error_str = str(e)
                    if is_context_length_error(error_str):
                        print(f"Context length error in attempt {i+1}: {error_str}")
                        # Try with fewer examples if this is a context length error
                        if len(demoset) > 1:
                            reduced_demoset = demoset[:len(demoset)//2]
                            try:
                                new_program = program.deepcopy()
                                new_program.generate_score.predict.demos = reduced_demoset
                                score = _run_evaluate_with_rate_limit(new_program)
                                print(f"  Reduced attempt {i+1}: Score = {score:.4f} with {len(reduced_demoset)} examples")
                                if score > best_score:
                                    best_score = score
                                    best_examples = reduced_demoset
                                    print(f"  New best score: {best_score:.4f}")
                            except Exception as e2:
                                print(f"  Reduced attempt also failed: {e2}")
                    else:
                        print(f"Non-context error in attempt {i+1}: {e}")
                    continue

        if best_examples:
            print(f"Optimization complete. Best score: {best_score:.4f}, using {len(best_examples)} examples")
        else:
            print("Warning: Optimization failed, using no examples")

        return best_examples

    def generate(self, dataset, target_measure: Optional[str] = None, n_metrics: int = 5, formatter: Optional[Callable] = None, **kwargs):
        """
        Generate new example-based metrics based on the dataset and task description.
        Automatically detects if the dataset has references and uses the appropriate metric class.
        
        This Generator does the example optimization and passes optimized examples to the Executor.
        The Executor only needs to execute with the pre-optimized examples.
        """

        task_description = dataset.get_task_description()

        formatter = self._resolve_formatter(dataset, formatter)
        
        # Step-1: Determine the appropriate executor class based on dataset
        if self.executor_class is None:
            dynamic_executor_class = self._determine_executor_class(dataset)
        else:
            dynamic_executor_class = self.executor_class
        
        # Step-2: Get target measure and prepare dataset -------------------------
        target_column = target_measure or dataset.get_target_columns()[0]
        suggested_range = (
            dataset.get_dataframe()[target_column].min().item(), 
            dataset.get_dataframe()[target_column].max().item()
        )

        # Step-3: Create metrics directly from target measure (no axis generation needed)
        # For example-based optimization, we work directly with the target measure
        metric_description = f"Example-based llm as a judge metric for '{target_column}'."
        if task_description:
            metric_description = f"Example-based llm as a judge metric for '{target_column}'.  The original task description for the task we are evaluating is: {task_description}"

        # Create a single metric based on the target measure
        metric_name = f"{target_column}_{self.judge_model_name}_examples"

        # Step-4: Prepare the buckets and optimize examples IN THE GENERATOR --------
        train_buckets, trainset = self._prepare_dataset_bucketted(
            dataset,
            target_column,
            task_description,
            target_column,  # Use target column name as the metric name
            formatter,
            suggested_range
        )

        # Step-5: OPTIMIZATION HAPPENS IN GENERATOR - NOT EXECUTOR ------------------
        print(f"🔍 Starting example optimization with {len(train_buckets)} buckets...")
        optimized_examples = self._optimize_examples(
            train_buckets, 
            trainset, 
            self.judge_model, 
            self.seed
        )

        # Step-6: Create the simplified metric with pre-optimized examples -----------
        
        # Validate and reconcile seed values
        executor_kwargs = self.executor_kwargs.copy()
        if self.seed is not None:
            if 'seed' in executor_kwargs and executor_kwargs['seed'] != self.seed:
                print(f"Warning: Seed mismatch detected. Proposer seed ({self.seed}) differs from executor_kwargs seed ({executor_kwargs['seed']}). Using proposer seed.")
            executor_kwargs['seed'] = self.seed
        elif 'seed' not in executor_kwargs:
            # No seed provided anywhere, that's fine
            pass
        
        metric = dynamic_executor_class(
            name=metric_name,
            description=metric_description,
            axis=f"Score for {target_column}",  # Simple axis description
            task_description=task_description,
            train_dataset=dataset,
            target_column=target_column,
            suggested_range=suggested_range,
            optimized_examples=optimized_examples,  # Pass pre-optimized examples
            # Pass optimization metadata for metric card generation
            attempts=self.attempts,
            examples_per_range=self.examples_per_range,
            eval_function_name=self.eval_function_name,
            metric_card_author_model=self.generator_llm,
            **executor_kwargs,
        )

        # TODO:  THIS IS IMPORTANT!!  RIGHT NOW WE ARE RETURNING A DUPLICATE METRIC TO ACHIEVE N_METRICS.  WE NEED TO FIX THIS IF A USER EVER WANTS TO GENERATE MORE THAN ONE METRIC.
        if n_metrics > 1:
            print("--------------------------------")
            print(f"WARNING!!!  GENERATOR {self.name} ONLY RETURNS DUPLICATES OF THE SAME METRIC.  THIS IS NOT A PROBLEM IF YOU ARE ONLY GENERATING ONE METRIC.  IF YOU ARE GENERATING MORE THAN ONE METRIC, THIS IS CURRENTLY NOT SUPPORTED.")
            print("--------------------------------")

        # Return list with one metric (or multiple if n_metrics > 1)
        # For now, we return the same metric since example-based optimization
        # works on the target measure directly
        return [metric] * min(n_metrics, 1)  # Only return 1 metric for now

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return self.__str__() 
