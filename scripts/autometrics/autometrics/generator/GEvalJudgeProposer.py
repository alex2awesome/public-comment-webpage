from autometrics.generator.Generator import Generator
from autometrics.metrics.generated.GeneratedGEvalMetric import (
    GeneratedRefFreeGEvalMetric,
    GeneratedRefBasedGEvalMetric
)
import dspy
from typing import Optional, Callable

# Utilities to avoid duplication and enable reuse across generators
from autometrics.generator.utils import (
    get_good_bad_examples,
    generate_axes_of_variation,
    build_metric_name_from_axis,
)


class GEvalJudgeProposer(Generator):
    """Generate *G-Eval* metrics by proposing evaluation criteria and auto-generating evaluation steps.
    """

    def __init__(
        self,
        name: str = "GEvalJudgeProposer",
        description: str = "Propose G-Eval measures based on the dataset and task description",
        generator_llm: Optional[dspy.LM] = None,
        executor_class: type | None = None,
        executor_kwargs: dict | None = None,
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

        # Store seed for temperature-based cache busting
        self.seed = seed

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
        
        if has_references:
            return GeneratedRefBasedGEvalMetric
        else:
            return GeneratedRefFreeGEvalMetric
    
    # Get Good and Bad Examples formatted properly
    def _preprocess_dataset(self, dataset, target_measure, formatter: Optional[Callable] = None):  # type: ignore[override]
        formatter = self._resolve_formatter(dataset, formatter)

        df = dataset.get_dataframe()
        if not target_measure:
            target_measure = dataset.get_target_columns()[0]

        good_examples, bad_examples = get_good_bad_examples(df, target_measure)

        good_examples_formatted = [formatter(row) for row in good_examples.iterrows()]
        bad_examples_formatted = [formatter(row) for row in bad_examples.iterrows()]

        return good_examples_formatted, bad_examples_formatted
    
    def generate(self, dataset, target_measure: Optional[str] = None, n_metrics: int = 5, formatter: Optional[Callable] = None, **kwargs):
        """
        Generate new G-Eval metrics based on the dataset and task description.
        Automatically detects if the dataset has references and uses the appropriate metric class.
        """

        task_description = dataset.get_task_description()

        formatter = self._resolve_formatter(dataset, formatter)
        
        # Step-1: Determine the appropriate executor class based on dataset
        if self.executor_class is None:
            dynamic_executor_class = self._determine_executor_class(dataset)
        else:
            dynamic_executor_class = self.executor_class
        
        # Step-2: Prepare / cache dataset & formatter ---------------------------------
        good_examples_formatted, bad_examples_formatted = self._preprocess_dataset(dataset, target_measure, formatter)

        # Step-3: Ask the language model to propose axes -----------------------------
        axes = generate_axes_of_variation(
            task_description=task_description,
            good_examples=good_examples_formatted,
            bad_examples=bad_examples_formatted,
            generator_llm=self.generator_llm,
            target_name=target_measure,
            num_axes_to_generate=n_metrics,
            seed=self.seed,
        )

        axes = axes[:n_metrics] if n_metrics else axes

        # Step-4: Wrap each axis in the appropriate GEval metric ------------------
        new_metrics = []
        for axis in axes:
            metric_name = build_metric_name_from_axis(axis, suffix=f"_{self.judge_model_name}_geval")
            # Validate and reconcile seed values
            executor_kwargs = self.executor_kwargs.copy()
            if self.seed is not None:
                if 'seed' in executor_kwargs and executor_kwargs['seed'] != self.seed:
                    print(f"Warning: Seed mismatch detected. Proposer seed ({self.seed}) differs from executor_kwargs seed ({executor_kwargs['seed']}). Using proposer seed.")
                executor_kwargs['seed'] = self.seed
            elif 'seed' not in executor_kwargs:
                # No seed provided anywhere, that's fine
                pass
            
            new_metrics.append(
                dynamic_executor_class(
                    name=metric_name,
                    description=f"G-Eval metric for {axis}",
                    evaluation_criteria=axis,
                    task_description=task_description,
                    metric_card_author_model=self.generator_llm,
                    **executor_kwargs,
                )
            )

        return new_metrics

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return self.__str__() 
