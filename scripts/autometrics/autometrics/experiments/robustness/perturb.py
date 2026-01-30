import dspy
from typing import Union, Literal, Callable, List
from autometrics.util.format import get_default_formatter
from autometrics.dataset.Dataset import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class GeneratePerturbationStrategies(dspy.Signature):
    """You will be given:  
• A Task description  
• A Dimension to prioritize when perturbing outputs  
• The Example Input, optional Example Reference, and Example Output  

Instructions:  
Your primary focus should be on degrading performance along the specified Dimension.  
1. Begin with a rich reasoning paragraph (3–5 sentences) that explores a variety of ways to subtly degrade model outputs. Do **not** reference the specific example.  
2. Under the heading **Strategies:**, list **1–3** numbered, high-level perturbation strategies.  
   - Each strategy should be a short phrase (5–15 words) naming the category of change, followed by one concise sentence of abstract explanation.  
   - Do **not** include concrete rewrites, instance-specific examples, or example sentences.  
"""
    task: str = dspy.InputField(description="The task that the model was originally trying to complete")
    example_sets: list[str] = dspy.InputField(description="Example inputs, outputs, and (optionally) references showcasing the model's performance on the task")
    dimension: str = dspy.InputField(description="The dimension to prioritize for the perturbation (this should be the aspect of the model output that is most impacted by the perturbation)")
    perturbation_strategies: list[str] = dspy.OutputField(description="A list of perturbation strategies that can be used to test the robustness of the model")


class PerturbWorse(dspy.Signature):
    """You will be given:  
    • A Task description  
    • A Dimension to prioritize when perturbing outputs  
    • The Example Input, optional Example Reference, and Model Output  
    • A perturbation_strength value ("subtle" or "obvious")  
    • A list of perturbation_strategies to apply  

Instructions:  
Your goal is to apply each strategy to the Model Output and produce a degraded version that specifically harms performance along the given Dimension, using the specified strength.  
Under the heading **Perturbed Outputs:**, return exactly one perturbed output per strategy.  
    - For **subtle** strength, introduce minimal distortion.  
    - For **obvious** strength, introduce more pronounced degradation.  
Do **not** include any reasoning, explanations, or examples -- only the perturbed text."""
    task: str = dspy.InputField(description="The task that the model was originally trying to complete")
    dimension: str = dspy.InputField(description="The dimension to prioritize for the perturbation (this should be the aspect of the model output that is most impacted by the perturbation)")
    input: str = dspy.InputField(description="The input provided to the model")
    references: Union[list[str], None] = dspy.InputField(description="The references of good outputs (may be None)")
    model_output: str = dspy.InputField(description="The output produced by the model")
    perturbation_strength: Literal["subtle", "obvious"] = dspy.InputField(description="The strength of the perturbation (subtle or obvious)")
    perturbation_strategies: list[str] = dspy.InputField(description="The perturbation strategies to use")
    perturbed_outputs: list[str] = dspy.OutputField(description="Perturbed text that is worse than the original model output.  Produce one perturbed output per strategy.")

class PerturbSame(dspy.Signature):
    """You will be given:
    • A Task description  
    • A Dimension to preserve when perturbing outputs  
    • The Example Input, optional Example Reference, and Model Output  
    • A perturbation_strength value ("subtle" or "obvious")  

Instructions:
Apply a perturbation to the Model Output that **maintains** performance on the specified Dimension.
Under the heading **Perturbed Output:** return exactly one string:
    - For **subtle** strength, apply a minimal change that does not impair the target Dimension.
    - For **obvious** strength, apply a more noticeable change that still keeps the target Dimension intact.
Some examples of types of perturbations would include: rephrasing, reordering, replacing words with synonyms, stylistic changes, etc. that do not impair the target Dimension.
If any change would harm the specified Dimension, simply return the original Model Output.
After producing your original plan/reasoning do **not** include any more reasoning, explanations, or examples -- only the perturbed text."""
    task: str = dspy.InputField(description="The task that the model was originally trying to complete")
    input: str = dspy.InputField(description="The input provided to the model")
    references: Union[list[str], None] = dspy.InputField(description="The references of good outputs (may be None)")
    model_output: str = dspy.InputField(description="The output produced by the model")
    perturbation_strength: Literal["subtle", "obvious"] = dspy.InputField(description="The strength of the perturbation (subtle or obvious)")
    dimension: str = dspy.InputField(description="The aspect of the model output that MUST be preserved in quality")
    perturbed_output: str = dspy.OutputField(description="Perturbed text that preserves performance along the given Dimension.")

class ProducePerturbations(dspy.Module):
    def __init__(self, num_examples: int = 3, formatter: Callable = None, max_workers: int = None):
        self.generate_perturbation_strategies: GeneratePerturbationStrategies = dspy.ChainOfThought(GeneratePerturbationStrategies)
        self.perturb_worse: PerturbWorse = dspy.Predict(PerturbWorse)
        self.perturb_same: PerturbSame = dspy.ChainOfThought(PerturbSame)
        self.num_examples = num_examples
        self.formatter = formatter
        self.max_workers = max_workers if max_workers is not None else 1

    def forward(self, task: str, dimension: str, dataset: Dataset):
        if self.formatter is None:
            self.formatter = get_default_formatter(dataset)

        df = dataset.get_dataframe()
        sampled_rows = df.sample(self.num_examples) if self.num_examples < len(df) else df
        formatted_rows = [self.formatter(row) for row in sampled_rows.iterrows()]

        perturbation_strategies = self.generate_perturbation_strategies(
            task=task,
            dimension=dimension,
            example_sets=formatted_rows
        ).perturbation_strategies

        input_col = dataset.get_input_column()
        output_col = dataset.get_output_column()
        ref_cols = dataset.get_reference_columns()
        records = df.to_dict('records')

        overall_worse_subtle, overall_worse_obvious = [], []
        overall_same_subtle, overall_same_obvious   = [], []

        def _process(record):
            inp = record[input_col]
            refs = [record[c] for c in ref_cols]
            out = record[output_col]

            res_worse_subtle = self.perturb_worse(
                task=task,
                dimension=dimension,
                input=inp,
                references=refs,
                model_output=out,
                perturbation_strength="subtle",
                perturbation_strategies=perturbation_strategies
            ).perturbed_outputs

            res_worse_obvious = self.perturb_worse(
                task=task,
                dimension=dimension,
                input=inp,
                references=refs,
                model_output=out,
                perturbation_strength="obvious",
                perturbation_strategies=perturbation_strategies
            ).perturbed_outputs

            res_same_subtle = self.perturb_same(
                task=task,
                dimension=dimension,
                input=inp,
                references=refs,
                model_output=out,
                perturbation_strength="subtle"
            ).perturbed_output

            res_same_obvious = self.perturb_same(
                task=task,
                dimension=dimension,
                input=inp,
                references=refs,
                model_output=out,
                perturbation_strength="obvious"
            ).perturbed_output

            return res_worse_subtle, res_worse_obvious, res_same_subtle, res_same_obvious

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for worse_subtle, worse_obvious, same_subtle, same_obvious in tqdm(
                executor.map(_process, records),
                total=len(records),
                desc="Perturbing examples"
            ):
                overall_worse_subtle.append(worse_subtle)
                overall_worse_obvious.append(worse_obvious)
                overall_same_subtle.append(same_subtle)
                overall_same_obvious.append(same_obvious)

        return {
            "perturbed_worse_subtle": overall_worse_subtle,
            "perturbed_worse_obvious": overall_worse_obvious,
            "perturbed_same_subtle": overall_same_subtle,
            "perturbed_same_obvious": overall_same_obvious,
            "strategies": perturbation_strategies
        }