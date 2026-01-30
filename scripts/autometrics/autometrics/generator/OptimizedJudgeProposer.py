from autometrics.generator.Generator import Generator
from autometrics.metrics.generated.GeneratedOptimizedJudge import (
    GeneratedRefFreeOptimizedJudge,
    GeneratedRefBasedOptimizedJudge
)
import dspy
from dspy.teleprompt import MIPROv2
from typing import Optional, Callable, List
import platformdirs
import os
import uuid
import tempfile
import re
import random

# Utilities to avoid duplication and enable reuse across generators
from autometrics.generator.utils import (
    get_good_bad_examples,
    generate_axes_of_variation,
    smart_limit_examples_for_context,
    get_max_context_tokens,
)



def exact_match_rounded(x, y):
    """Evaluation function for MIPROv2 optimization - exact match on rounded values."""
    return int(round(x) == round(y))

# Possible improvement: Change the scaling based on the range of the target column
# 1 / (5 * (1 - abs((x - y) / (max - min))))
# This would be a less harsh function for wide ranges of values
def inverse_distance(x, y):
    """Evaluation function for MIPROv2 optimization - inverse distance metric."""
    if x == y:
        return 1
    return 1 / (abs(x - y) + 1)


def get_wrapped_metric(metric_func):
    """Wrap an evaluation function for use with MIPROv2."""
    def wrapped_metric(example, pred, trace=None):
        return metric_func(example.score, pred.score)
    return wrapped_metric


def parse_formatted_text(formatted_text):
    """Parse formatted text to extract input and output."""
    # The formatted text should be in a pattern like:
    # «Input (query): «...»\nOutput (outcome): «...»»
    try:
        # Find the input section
        input_start = formatted_text.find('«Input')
        if input_start == -1:
            # Fallback: try to split on common patterns
            parts = formatted_text.split('\n')
            if len(parts) >= 2:
                return parts[0], '\n'.join(parts[1:])
            return formatted_text, ""
        
        # Extract input
        input_section = formatted_text[input_start:]
        output_start = input_section.find('Output')
        
        if output_start == -1:
            return formatted_text, ""
        
        # Parse input 
        input_part = input_section[:output_start]
        # Remove the outer formatting
        input_match = re.search(r'«Input[^:]*:[^«]*«([^»]*)»', input_part)
        if input_match:
            input_text = input_match.group(1).strip()
        else:
            input_text = input_part.strip()
        
        # Parse output
        output_part = input_section[output_start:]
        output_match = re.search(r'Output[^:]*:[^«]*«([^»]*)»', output_part)
        if output_match:
            output_text = output_match.group(1).strip()
        else:
            output_text = output_part.strip()
        
        return input_text, output_text
        
    except Exception as e:
        print(f"Warning: Failed to parse formatted text: {e}")
        # Fallback to using the entire text as output
        return "", formatted_text


def prepare_dataset_for_optimization(dataset, target_column, task_description, axis_description, formatter, suggested_range=(1, 5)):
    """Prepare dataset for MIPROv2 optimization."""
    dspy_dataset = []
    
    # Check if dataset has references
    reference_columns = dataset.get_reference_columns()
    has_references = reference_columns is not None and len(reference_columns) > 0

    for i, row in dataset.get_dataframe().iterrows():
        # Filter out rows with NaN target values that can break MIPROv2
        target_value = row[target_column]
        if target_value is None or (hasattr(target_value, '__iter__') and any(x != x for x in [target_value])) or (not hasattr(target_value, '__iter__') and target_value != target_value):
            print(f"Warning: Skipping row {i} with NaN target value: {target_value}")
            continue
            
        formatted_text = formatter((i, row))
        input_text, output_text = parse_formatted_text(formatted_text)
        
        if has_references:
            # Use reference column (try first available reference column)
            reference_col = reference_columns[0]
            reference_text = row[reference_col] if reference_col in row else ""
            
            dspy_dataset.append(
                dspy.Example(
                    task_description=task_description,
                    axis=axis_description,
                    input_text=input_text,
                    reference_text=reference_text,
                    output_text=output_text,
                    suggested_range=suggested_range,
                    score=target_value
                ).with_inputs('task_description', 'axis', 'input_text', 'reference_text', 'output_text', 'suggested_range')
            )
        else:
            dspy_dataset.append(
                dspy.Example(
                    task_description=task_description,
                    axis=axis_description,
                    input_text=input_text,
                    output_text=output_text,
                    suggested_range=suggested_range,
                    score=target_value
                ).with_inputs('task_description', 'axis', 'input_text', 'output_text', 'suggested_range')
            )

    print(f"📊 Prepared {len(dspy_dataset)} examples for optimization (filtered out NaN values)")
    return dspy_dataset


class OptimizedJudgeProposer(Generator):
    """Generate *Optimized LLM-as-a-judge* metrics by proposing axes of variation and optimizing prompts.

    This class uses MIPROv2 to optimize prompts for each proposed evaluation axis,
    resulting in high-quality LLM judge metrics. The class conforms to the new
    *Generator* interface which includes an optional *generator_llm* (used here to
    generate axes) as well as the ability to specify a custom executor class.
    """

    def __init__(
        self,
        name: str = "OptimizedJudgeProposer",
        description: str = "Propose optimized LLM-as-a-Judge measures with MIPROv2-optimized prompts",
        generator_llm: Optional[dspy.LM] = None,
        executor_class: type | None = None,
        executor_kwargs: dict | None = None,
        eval_function_name: str = 'inverse_distance',
        custom_eval_function: Optional[Callable] = None,
        auto_mode: str = "medium",
        num_threads: int = 64,
        max_bootstrapped_demos: int = 8,
        max_labeled_demos: int = 8,
        max_train_set_size: int = 200, # To limit training time
        seed: Optional[int] = None,
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

        # Guarantee attribute is a dictionary for ** expansion later
        if self.executor_kwargs is None:
            self.executor_kwargs = {}

        # Setup evaluation function for optimization
        if custom_eval_function is not None:
            self.eval_function = custom_eval_function
        elif eval_function_name == 'exact_match_rounded':
            self.eval_function = exact_match_rounded
        elif eval_function_name == 'inverse_distance':
            self.eval_function = inverse_distance
        else:
            self.eval_function = inverse_distance  # Default fallback

        # MIPROv2 optimization settings
        self.auto_mode = auto_mode
        self.num_threads = num_threads
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_train_set_size = max_train_set_size

        # Seed for reproducible variation in metric generation
        self.seed = seed

        # Extract judge model info for naming
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

        # Setup directory for saving optimized prompts
        self.prompts_dir = os.path.join(platformdirs.user_data_dir("autometrics"), "optimized_prompts")
        os.makedirs(self.prompts_dir, exist_ok=True)

    def _determine_executor_class(self, dataset):
        """Determine whether to use reference-based or reference-free metrics based on dataset."""
        reference_columns = dataset.get_reference_columns()
        has_references = reference_columns is not None and len(reference_columns) > 0
        
        if has_references:
            return GeneratedRefBasedOptimizedJudge
        else:
            return GeneratedRefFreeOptimizedJudge
    
    def _preprocess_dataset(self, dataset, target_measure, formatter: Optional[Callable] = None):
        formatter = self._resolve_formatter(dataset, formatter)

        df = dataset.get_dataframe()
        if not target_measure:
            target_measure = dataset.get_target_columns()[0]

        good_examples, bad_examples = get_good_bad_examples(df, target_measure)

        good_examples_formatted = [formatter(row) for row in good_examples.iterrows()]
        bad_examples_formatted = [formatter(row) for row in bad_examples.iterrows()]

        return good_examples_formatted, bad_examples_formatted

    def _optimize_prompt_for_target(self, target_column: str, dataset, task_description: str, formatter: Callable):
        """Optimize a prompt for a specific target column using MIPROv2."""
        
        print(f"🔧 Optimizing prompt for target column: {target_column}")
        
        # Calculate suggested range from data
        target_column_data = dataset.get_dataframe()[target_column]

        # Filter out NaN values before calculating range
        target_column_clean = target_column_data.dropna()

        if len(target_column_clean) == 0:
            print(f"Warning: No valid (non-NaN) values found in target column '{target_column}'. Using default range (1, 5).")
            suggested_range = (1, 5)
        elif len(target_column_clean.unique()) == 1:
            # All values are the same - create a small range around the single value
            single_value = target_column_clean.iloc[0]
            suggested_range = (single_value - 0.5, single_value + 0.5)
            print(f"Warning: All values in target column '{target_column}' are the same ({single_value}). Using range {suggested_range}.")
        else:
            suggested_range = (target_column_clean.min().item(), target_column_clean.max().item())

        print(f"📊 Calculated suggested_range for '{target_column}': {suggested_range} (from {len(target_column_clean)} valid values)")
        
        # Create a descriptive axis for optimization
        axis_description = f"Evaluate the output on the '{target_column}' criterion (may or may not be super descriptive). Rate from around {suggested_range[0]} to {suggested_range[1]}."
        
        # Prepare training dataset for optimization  
        train_set = prepare_dataset_for_optimization(
            dataset,
            target_column,
            task_description,
            axis_description,  # Use descriptive axis as metric name
            formatter,
            suggested_range
        )

        if train_set is None:
            train_set = []
        
        # Smart limit examples for context window constraints
        if len(train_set) > 0:
            # Extract example texts for token estimation
            example_texts = []
            for example in train_set:
                if hasattr(example, 'text'):
                    example_texts.append(example.text)
                elif hasattr(example, 'input') and hasattr(example.input, 'text'):
                    example_texts.append(example.input.text)
                else:
                    # Fallback: convert example to string
                    example_texts.append(str(example))
            
            # Get model name for token estimation
            model_name = "gpt-3.5-turbo"  # Default
            if self.judge_model and hasattr(self.judge_model, 'model'):
                model_name = self.judge_model.model
            elif self.generator_llm and hasattr(self.generator_llm, 'model'):
                model_name = self.generator_llm.model
            
            # Smart limit examples
            limited_examples = smart_limit_examples_for_context(
                example_texts,
                task_description,
                model_name,
                target_examples=min(self.max_bootstrapped_demos + self.max_labeled_demos, len(train_set)),
                dspy_overhead_tokens=4096,
                output_tokens=2048,
                safety_margin=2000
            )
            
            # If we need to limit examples, recreate train_set with limited examples
            if len(limited_examples) < len(train_set):
                print(f"Limiting MIPROv2 prompt to {len(limited_examples)} examples due to context constraints")
                
                # Update optimization parameters to match limited examples
                self.max_bootstrapped_demos = min(self.max_bootstrapped_demos, len(limited_examples) // 2)
                self.max_labeled_demos = min(self.max_labeled_demos, len(limited_examples) // 2)
                print(f"Adjusted optimization params: bootstrapped={self.max_bootstrapped_demos}, labeled={self.max_labeled_demos}")
        
        # Initialize DSPy signature and module based on reference type
        reference_columns = dataset.get_reference_columns()
        has_references = reference_columns is not None and len(reference_columns) > 0
        
        if has_references:
            from autometrics.metrics.generated.GeneratedOptimizedJudge import _OptimizedJudgeSignatureRefBased as SignatureClass
        else:
            from autometrics.metrics.generated.GeneratedOptimizedJudge import _OptimizedJudgeSignatureRefFree as SignatureClass
            
        # Create the base program
        base_program = dspy.ChainOfThought(SignatureClass)
        
        # Set up MIPROv2 optimizer
        teleprompter = MIPROv2(
            metric=get_wrapped_metric(self.eval_function),
            auto=self.auto_mode,
            num_threads=self.num_threads,
        )
        
        # Optimize the program with proper DSPy context
        print(f"📈 Running MIPROv2 optimization...")
        # Use the judge model for optimization since it will be doing the actual scoring
        optimization_model = self.judge_model if self.judge_model else self.generator_llm
        
        # Set temperature based on seed for reproducible variation
        if self.seed is not None:
            temperature = 0.0001 * self.seed
            print(f"🎲 Using seed-based temperature: {temperature} (seed: {self.seed})")
            # Create a copy of the model with the seed-based temperature
            if hasattr(optimization_model, 'kwargs'):
                temp_kwargs = optimization_model.kwargs.copy()
                temp_kwargs['temperature'] = temperature
                optimization_model_with_temp = type(optimization_model)(
                    model=optimization_model.model, **temp_kwargs
                )
            else:
                optimization_model_with_temp = optimization_model
        else:
            optimization_model_with_temp = optimization_model

        if self.max_train_set_size is not None and len(train_set) > self.max_train_set_size:
            print(f"Warning: Train set size ({len(train_set)}) exceeds max_train_set_size ({self.max_train_set_size}). Truncating train set to reduce training time.")
            random.seed(self.seed)
            random.shuffle(train_set)
            train_set = train_set[:self.max_train_set_size]
            
        print(f"🔧 Starting MIPROv2 optimization with:")
        print(f"  - Model: {optimization_model_with_temp}")
        print(f"  - Train set size: {len(train_set)}")
        print(f"  - Auto mode: {self.auto_mode}")
        print(f"  - Num threads: {self.num_threads}")
        print(f"  - Max bootstrapped demos: {self.max_bootstrapped_demos}")
        print(f"  - Max labeled demos: {self.max_labeled_demos}")
        
        with dspy.settings.context(lm=optimization_model_with_temp):
            optimized_program = teleprompter.compile(
                base_program,
                trainset=train_set,
                max_bootstrapped_demos=self.max_bootstrapped_demos,
                max_labeled_demos=self.max_labeled_demos,
                requires_permission_to_run=False,
            )
            
        print(f"🎯 MIPROv2 optimization completed. Result type: {type(optimized_program)}")
        
        # Save optimized prompt
        unique_id = str(uuid.uuid4())[:8]
        clean_target_name = target_column.replace(" ", "_").replace("-", "_")
        dataset_name = getattr(dataset, 'name', 'UnknownDataset')
        seed_suffix = f"_seed{self.seed}" if self.seed is not None else ""
        prompt_filename = f"{dataset_name}_{clean_target_name}_{self.judge_model_name}_{unique_id}_optimized{seed_suffix}.json"
        prompt_path = os.path.join(self.prompts_dir, prompt_filename)
        
        optimized_program.save(prompt_path)
        print(f"💾 Saved optimized prompt to: {prompt_path}")
        
        return {
            'optimized_program': optimized_program,
            'prompt_path': prompt_path,
            'suggested_range': suggested_range,
            'target_column': target_column
        }

    def generate(self, dataset, target_measure: Optional[str] = None, n_metrics: int = 5, formatter: Optional[Callable] = None, **kwargs):
        """
        Generate optimized metrics directly for the target measure using MIPROv2.
        Unlike other generators, this doesn't generate multiple axes but creates one
        optimized metric per target column specified by n_metrics.
        """

        task_description = dataset.get_task_description()

        if not task_description:
            task_description = "Respond to the user's query."

        formatter = self._resolve_formatter(dataset, formatter)

        # Track whether the caller explicitly requested a target.
        explicit_target_requested = target_measure is not None

        if not target_measure:
            target_measure = dataset.get_target_columns()[0]
        
        # Step-1: Determine the appropriate executor class based on dataset
        if self.executor_class is None:
            dynamic_executor_class = self._determine_executor_class(dataset)
        else:
            dynamic_executor_class = self.executor_class
        
        # Step-2: Generate metrics for target measure(s) -----------------------------
        new_metrics = []
        target_columns = dataset.get_target_columns()

        # Validate explicit target and build the list of targets to generate for
        if explicit_target_requested and target_measure not in target_columns:
            raise ValueError(
                f"target_measure '{target_measure}' not found in dataset target columns: {target_columns}"
            )

        if explicit_target_requested:
            targets_to_generate = [target_measure] * n_metrics
        else:
            targets_to_generate = target_columns[: min(n_metrics, len(target_columns))]

        for i, current_target in enumerate(targets_to_generate):
            print(f"\n🎯 Optimizing metric for target: {current_target}")

            # Optimize prompt for this target measure
            print(f"🚀 Calling _optimize_prompt_for_target with:")
            print(f"  - target_column: {current_target}")
            print(f"  - dataset: {dataset.get_name()}")
            print(f"  - task_description: {task_description[:100]}...")

            optimization_result = self._optimize_prompt_for_target(
                target_column=current_target,
                dataset=dataset,
                task_description=task_description,
                formatter=formatter,
            )

            print(f"📊 Optimization result type: {type(optimization_result)}")
            print(f"📊 Optimization result: {optimization_result}")

            # Create metric name with dataset name to avoid collisions
            dataset_name = getattr(dataset, 'name', 'UnknownDataset')
            seed_suffix = f"_seed{self.seed}" if self.seed is not None else ""
            base_metric_name = f"{dataset_name}_{current_target}_{self.judge_model_name}_optimized{seed_suffix}"
            # Add index suffix only when explicitly generating multiple for the same target
            metric_name = (
                f"{base_metric_name}_{i+1}" if explicit_target_requested and n_metrics > 1 else base_metric_name
            )

            # Create a descriptive axis for the target column
            axis_description = (
                f"Evaluate the output on the '{current_target}' criterion (may or may not be super descriptive). "
                f"Rate from around {optimization_result['suggested_range'][0]} to {optimization_result['suggested_range'][1]}."
            )

            # Validate and reconcile seed values
            executor_kwargs = self.executor_kwargs.copy()
            if self.seed is not None:
                if 'seed' in executor_kwargs and executor_kwargs['seed'] != self.seed:
                    print(
                        f"Warning: Seed mismatch detected. Proposer seed ({self.seed}) differs from executor_kwargs seed ({executor_kwargs['seed']}). Using proposer seed."
                    )
                executor_kwargs['seed'] = self.seed
            elif 'seed' not in executor_kwargs:
                # No seed provided anywhere, that's fine
                pass

            new_metrics.append(
                dynamic_executor_class(
                    name=metric_name,
                    description=f"MIPROv2-optimized LLM judge for {current_target} on {dataset_name}",
                    axis=axis_description,
                    task_description=task_description,
                    optimized_prompt_path=optimization_result['prompt_path'],
                    suggested_range=optimization_result['suggested_range'],
                    metric_card_author_model=self.generator_llm,
                    **executor_kwargs,
                )
            )

            print(f"✅ Created optimized metric: {metric_name}")

        print(f"\n🎉 Generated {len(new_metrics)} optimized LLM judge metrics!")
        return new_metrics

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return self.__str__() 
