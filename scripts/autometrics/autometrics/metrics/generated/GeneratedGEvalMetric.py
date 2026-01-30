import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from tqdm import tqdm
import dspy

from autometrics.metrics.generated.utils.utils import generate_llm_constructor_code
from autometrics.metrics.generated.utils.metric_card import generate_further_reading
from autometrics.metrics.generated.utils.metric_card import MetricCardBuilder
from autometrics.metrics.generated.GeneratedRefFreeMetric import GeneratedRefFreeMetric
from autometrics.metrics.generated.GeneratedRefBasedMetric import GeneratedRefBasedMetric

__all__ = ["GeneratedRefFreeGEvalMetric", "GeneratedRefBasedGEvalMetric"]

# G-Eval prompt template
G_EVAL_PROMPT = """You are evaluating the output of a model.  Here is the task description: {task_description}

Evaluation Criteria:

{evaluation_criteria} (Score 1 lowest - 5 highest)

Evaluation Steps:

{evaluation_steps}

Example:


Source Text:

{source}

References (if available):
{references}

Model Output:

{model_output}


Evaluation Form (scores ONLY):

- {evaluation_criteria_name}:"""

# DSPy signature for evaluation steps generation
class GenerateEvaluationStepsSignature(dspy.Signature):
    """Given a task description and evaluation criteria please come up with evaluation steps that will lead to an accurate score.  Note that the task description is the prompt that was provided to the model."""
    task_description = dspy.InputField(desc="A description of the task that the model was trying to solve when it generated the text.  Could be left blank if not available.")
    evaluation_criteria = dspy.InputField(desc="The criteria that the model output should be evaluated on.")
    evaluation_steps = dspy.OutputField(desc="The steps that will be used to evaluate the model output. Please number the steps.")

class GenerateEvaluationSteps(dspy.Module):
    def __init__(self):
        self.generate_evaluation_steps = dspy.ChainOfThought(GenerateEvaluationStepsSignature)

    def forward(self, task_description, evaluation_criteria):
        evaluation_steps = self.generate_evaluation_steps(task_description=task_description, evaluation_criteria=evaluation_criteria).evaluation_steps
        return evaluation_steps

def find_score_token_index(logprobs_content, possible_scores=[1, 2, 3, 4, 5]):
    """
    Find the index of the last token that contains a score (1-5) by searching backwards.
    
    Args:
        logprobs_content: List of ChatCompletionTokenLogprob objects
        possible_scores: List of possible score values
        
    Returns:
        int: Index of the token containing the score, or None if not found
    """
    score_tokens = [str(score) for score in possible_scores]
    
    # Search backwards through the tokens
    for i in range(len(logprobs_content) - 1, -1, -1):
        token = logprobs_content[i].token
        # Check if this token is exactly one of our score tokens
        if token in score_tokens:
            return i
    
    return None

def extract_score_probabilities(logprobs_content, possible_scores=[1, 2, 3, 4, 5]):
    """
    Convert logprobs to probability distribution over possible scores.
    Finds the last token containing a score (1-5) and extracts probabilities from there.
    
    Args:
        logprobs_content: List of ChatCompletionTokenLogprob objects from the API response
        possible_scores: List of possible score values (default [1, 2, 3, 4, 5])
    
    Returns:
        dict: Mapping from score to probability, or None if no score token found
    """
    # Find the index of the last score token
    score_token_index = find_score_token_index(logprobs_content, possible_scores)
    
    if score_token_index is None:
        print("Warning: No score token (1-5) found in the model output!")
        return None
    
    # Get the top logprobs for the score token
    top_logprobs = logprobs_content[score_token_index].top_logprobs
    
    # Convert possible scores to strings since tokens are strings
    score_tokens = [str(score) for score in possible_scores]
    
    # Extract logprobs for score tokens, defaulting to very low probability for missing scores
    score_logprobs = {}
    for score_token in score_tokens:
        # Find the logprob for this score token
        found = False
        for top_logprob in top_logprobs:
            if top_logprob.token == score_token:
                score_logprobs[score_token] = top_logprob.logprob
                found = True
                break
        
        # If not found, assign a very low logprob (effectively zero probability)
        if not found:
            score_logprobs[score_token] = -20.0  # Very low logprob
    
    # Convert logprobs to probabilities using exp
    score_probs = {token: np.exp(logprob) for token, logprob in score_logprobs.items()}
    
    # Normalize to sum to 1 (softmax normalization)
    total_prob = sum(score_probs.values())
    normalized_probs = {token: prob / total_prob for token, prob in score_probs.items()}
    
    # Convert back to integer keys
    result = {int(token): prob for token, prob in normalized_probs.items()}
    
    return result

def GEval(formatted_prompt, source, references, model_output, dspy_lm):
    """Perform G-Eval scoring with logprobs."""
    # Replace {user_query} placeholder with the actual source text if present
    formatted_prompt = formatted_prompt.replace("{user_query}", source)
    prompt = formatted_prompt.format(source=source, references=references, model_output=model_output)

    results = dspy_lm.forward(prompt=prompt, logprobs=True, top_logprobs=5)

    # Extract score probabilities using the smarter approach
    score_probs = extract_score_probabilities(results.choices[0].logprobs.content)

    weighted_score = 0
    
    if score_probs is not None:
        weighted_score = sum([score * prob for score, prob in score_probs.items()])
    else:
        print("Could not extract score probabilities - no valid score token found.")
    
    return weighted_score

# Base mixin for shared G-Eval functionality
class _GEvalMetricMixin:
    """Shared functionality for both reference-free and reference-based G-Eval metrics."""

    DEFAULT_MAX_WORKERS = 32

    def __init__(
        self,
        name: str,
        description: str,
        evaluation_criteria: str,
        model: dspy.LM,
        task_description: Optional[str] = None,
        evaluation_steps: Optional[str] = None,
        auto_generate_steps: bool = True,
        possible_scores: List[int] = [1, 2, 3, 4, 5],
        criteria_generation_model: Optional[dspy.LM] = None,
        metric_card: Optional[str] = None,
        metric_card_author_model: Optional[dspy.LM] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        is_reference_based: bool = False,
        **kwargs,
    ):
        self.evaluation_criteria = evaluation_criteria
        self.task_description = task_description or "No task description provided"
        self.model = model
        self.model_str = str(getattr(model, "model", model))
        self.evaluation_steps = evaluation_steps
        self.auto_generate_steps = auto_generate_steps
        self.possible_scores = possible_scores
        self.criteria_generation_model = criteria_generation_model or model
        self.max_workers = max_workers
        self.is_reference_based = kwargs.get("is_reference_based", is_reference_based)

        if metric_card_author_model is None:
            metric_card_author_model = model if isinstance(model, dspy.LM) else None

        if metric_card == "provided":
            self.metric_card = self.__doc__
            metric_card = self.metric_card

        # Generate evaluation steps if needed
        if self.evaluation_steps is None and self.auto_generate_steps:
            self._generate_evaluation_steps()
        elif self.evaluation_steps is None:
            self.evaluation_steps = "1. Read the source text and model output carefully.\n2. Compare against the evaluation criteria.\n3. Assign a score from 1-5."

        # Create the formatted prompt template
        self.formatted_prompt = G_EVAL_PROMPT.format(
            task_description=self.task_description,
            evaluation_criteria=self.evaluation_criteria,
            evaluation_steps=self.evaluation_steps,
            evaluation_criteria_name=self.evaluation_criteria.split(":")[0] if ":" in self.evaluation_criteria else self.evaluation_criteria,
            source="{source}",
            references="{references}",
            model_output="{model_output}"
        )

        # Initialize parent with shared parameters
        super().__init__(
            name,
            description,
            metric_card=metric_card,
            metric_card_author_model=metric_card_author_model,
            evaluation_criteria=evaluation_criteria,
            model_str=str(getattr(model, "model", model)),
            task_description=self.task_description,
            evaluation_steps=self.evaluation_steps,
            auto_generate_steps=auto_generate_steps,
            possible_scores=possible_scores,
            **kwargs,
        )

        # Exclude heavy objects from cache key
        self.exclude_from_cache_key("model", "criteria_generation_model")

    def _generate_evaluation_steps(self):
        """Generate evaluation steps if auto_generate_steps is True."""
        try:
            with dspy.settings.context(lm=self.criteria_generation_model):
                generator = GenerateEvaluationSteps()
                self.evaluation_steps = generator.forward(
                    task_description=self.task_description,
                    evaluation_criteria=self.evaluation_criteria
                )
        except Exception as e:
            print(f"Warning: Could not auto-generate evaluation steps: {e}")
            self.evaluation_steps = "1. Read the source text and model output carefully.\n2. Compare against the evaluation criteria.\n3. Assign a score from 1-5."

    def _call_geval(self, input_text: str, output_text: str, references: Optional[str] = None) -> float:
        """Call G-Eval scoring function."""
        input_text = str(input_text) if input_text is not None else ""
        output_text = str(output_text) if output_text is not None else ""
        
        # Prepare references string
        if references is None:
            references_str = "None provided"
        elif isinstance(references, list):
            references_str = "\n".join([f"Reference {i+1}: {str(ref)}" for i, ref in enumerate(references) if ref is not None])
        else:
            references_str = str(references)
        
        score = GEval(
            formatted_prompt=self.formatted_prompt,
            source=input_text,
            references=references_str,
            model_output=output_text,
            dspy_lm=self.model
        )
        
        return score

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        del kwargs  # pragma: no cover
        results: List[float] = [0.0] * len(outputs)

        # Fail-fast if workers=1
        if self.max_workers == 1:
            return [self._call_geval(i, o, r) for i, o, r in zip(inputs, outputs, references or [None] * len(outputs))]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._call_geval, i, o, r): idx 
                for idx, (i, o, r) in enumerate(zip(inputs, outputs, references or [None] * len(outputs)))
            }
            
            # Collect results with progress bar
            with tqdm(total=len(futures), desc="Processing G-Eval") as pbar:
                for future in as_completed(futures):
                    index = futures[future]
                    # FIXED: Let errors propagate naturally instead of catching and returning 0.0
                    # This allows the cache to distinguish between failures and valid results
                    results[index] = future.result()
                    pbar.update(1)
        
        return results

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _generate_python_code(self, include_metric_card: bool = True) -> str:
        """Export a standalone python file that re-creates this metric."""
        class_name = "GeneratedRefBasedGEvalMetric" if self.is_reference_based else "GeneratedRefFreeGEvalMetric"
        code = f"""# Auto-generated G-Eval metric file for {self.name}
import dspy
import os
from autometrics.metrics.generated.GeneratedGEvalMetric import {class_name}
from typing import ClassVar

DEFAULT_MODEL = {generate_llm_constructor_code(self.model)}

class {self.name.replace(" ", "_").replace("-", "_")}_GEval({class_name}):
    \"\"\"{self.metric_card if include_metric_card else ""}\"\"\"

    description: ClassVar[str] = {json.dumps(self.description)}

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name={json.dumps(self.name)},
            description={json.dumps(self.description)},
            evaluation_criteria={json.dumps(self.evaluation_criteria)},
            model=model,
            task_description={json.dumps(self.task_description)},
            evaluation_steps={json.dumps(self.evaluation_steps)},
            auto_generate_steps={self.auto_generate_steps},
            possible_scores={self.possible_scores},
            metric_card={json.dumps("provided" if include_metric_card else "None")},
            max_workers={self.max_workers},
        )

    def __repr__(self):
        return f"{self.name.replace(' ', '_').replace('-', '_')}_GEval(model={generate_llm_constructor_code(self.model).replace('\"', "\\\"")})"

"""
        return code
    def _serialize(self) -> dict:
        """Serialize the metric to a dictionary for in-memory operations."""
        return {
            "name": self.name,
            "description": self.description,
            "evaluation_criteria": self.evaluation_criteria,
            "model": generate_llm_constructor_code(self.model),  # Stores model constructor code
            "task_description": self.task_description,
            "evaluation_steps": self.evaluation_steps,
            "auto_generate_steps": self.auto_generate_steps,
            "possible_scores": list(self.possible_scores),  # Explicitly convert to list
            "metric_card": self.metric_card,
            "max_workers": self.max_workers,
            "is_reference_based": self.is_reference_based,
        }

    @classmethod
    def _deserialize(cls, data: dict):
        """Deserialize a dictionary to create a metric instance."""
        # Convert model constructor code string back to model instance
        model_code = data.pop("model")
        # This assumes the model code string can be safely evaluated to create a model
        model = eval(model_code)
        data["model"] = model
        
        # Ensure possible_scores is a list
        data["possible_scores"] = list(data["possible_scores"])
        
        return cls(**data)
    
    # ------------------------------------------------------------------
    # Metric-card helpers
    # ------------------------------------------------------------------

    def _metric_details_template(self, *, reference_based: bool) -> str:
        """Return the *Metric Details* section for ref-free / ref-based G-Eval.

        Parameters
        ----------
        reference_based : bool
            If True, emit the reference-based variant; otherwise emit the
            reference-free variant.
        """
        kind = "reference-based" if reference_based else "reference-free"
        ref_flag = "Yes" if reference_based else "No"
        input_req = "Yes (plus reference)" if reference_based else "Yes"

        # --- Header & description ----------------------------------------
        lines = [
            f"**{self.name}** is a **{kind}** G-Eval metric that uses logprobs to evaluate system outputs along evaluation criteria.",
            f"In this case the criteria is `{self.evaluation_criteria}`.",
            "",
            "G-Eval differs from regular LLM-as-a-Judge by:",
            "",
            "1. **Generating evaluation steps** automatically based on criteria",
            "2. **Using logprobs** to extract token-level probabilities for scores 1-5",
            "3. **Computing weighted scores** from the probability distribution",
            "",
            "The prompt supplies:",
            "",
            "1. **Task description** *d*",
            f"2. **Evaluation criteria** `{self.evaluation_criteria}`",
            "3. **Auto-generated evaluation steps**",
            "4. **Input text** *x*",
        ]
        if reference_based:
            lines.append("5. **Reference text** *r*")
            lines.append("6. **Output text** *y*")
        else:
            lines.append("5. **Output text** *y*")

        # --- Scoring sentence --------------------------------------------
        lines.extend(
            [
                "",
                r"Unlike regular LLM judging, G-Eval extracts logprobs for score tokens "
                r"$\{1,2,3,4,5\}$ and computes $\hat{s} = \sum_{i=1}^{5} i \cdot P(i|\text{prompt})$; "
                "higher = better adherence to the criteria.",
                "",
                "- **Metric Type:** G-Eval (probabilistic LLM scoring)",
                "- **Range:** 1-5 (continuous, weighted by logprobs)",
                "- **Higher is Better?:** Yes",
                f"- **Reference-Based?:** {ref_flag}",
                f"- **Input-Required?:** {input_req}",
                "",
                "### Evaluation Steps",
                "",
                f"The following evaluation steps are {'auto-generated' if self.auto_generate_steps else 'pre-defined'} for this metric.  Note they just describe a process to the model, but there is no reason to expect the model actually follows these steps in completing the evaluation.  The evaluation steps are:",
                "",
            ]
        )
        
        # Add the actual evaluation steps with proper formatting
        if self.evaluation_steps:
            step_lines = [line.strip() for line in self.evaluation_steps.split('\n') if line.strip()]
            for i, step in enumerate(step_lines):
                # Remove existing numbering if present and add consistent numbering
                step_text = step
                if step_text.startswith(f"{i+1}."):
                    step_text = step_text[len(f"{i+1}."):].strip()
                elif step_text.startswith(f"{i+1}"):
                    step_text = step_text[len(f"{i+1}"):].strip()
                lines.append(f"{i+1}. {step_text}")
        
        lines.extend(
            [
                "",
                "### Formal Definition",
                "",
                r"Let $f _{\theta}$ be the LLM and",
            ]
        )

        if reference_based:
            lines.append(
                r"$\pi _{\text{G-RB}}(d,\{criteria\},\{steps\},x,r,y)$ construct the G-Eval "
                "prompt."
            )
        else:
            lines.append(
                r"$\pi _{\text{G-RF}}(d,\{criteria\},\{steps\},x,y)$ construct the G-Eval "
                "prompt."
            )

        lines.extend(
            [
                "",
                "$$",
                r"P(s|\text{prompt}) = \frac{\exp(\text{logprob}(s))}{\sum_{i=1}^{5} \exp(\text{logprob}(i))}",
                "$$",
                "",
                "$$",
                r"\hat{s} = \sum_{s=1}^{5} s \cdot P(s|\text{prompt})",
                "$$",
                "",
                r"The metric value is "
                + (
                    r"$\operatorname{G-Eval}^{\text{RB}}_{\{criteria\}}"
                    r"(d,x,r,y)=\hat{s}$."
                    if reference_based
                    else r"$\operatorname{G-Eval}^{\text{RF}}_{\{criteria\}}"
                    r"(d,x,y)=\hat{s}$."
                ),
                "",
                "### Inputs and Outputs",
                "- **Inputs:**",
                "  - **Task description** *d*",
                f"  - **Evaluation criteria** `{self.evaluation_criteria}`",
                "  - **Auto-generated evaluation steps**",
                "  - **Input text** *x*",
            ]
        )
        if reference_based:
            lines.append("  - **Reference text** *r*")
        lines.append("  - **Output text** *y*")
        lines.extend(
            [
                "- **Outputs:**",
                "  - Weighted score "
                r"$\hat{s} \in [1,5]$ (continuous)",
            ]
        )

        return "\n".join(lines)
    
    def generate_metric_details_ref_free(self) -> str:
        """Metric-details section for the **reference-free** variant."""
        return self._metric_details_template(reference_based=False)

    def generate_metric_details_ref_based(self) -> str:
        """Metric-details section for the **reference-based** variant."""
        return self._metric_details_template(reference_based=True)

    def generate_intended_use(self):
        class IntendedUseSignature(dspy.Signature):
            """Given the task description, and an evaluation criteria, consider a G-Eval metric that is evaluating the text along this criteria.  Your task is to generate the domain, a list of tasks, and a set of circumstances where G-Eval is best suited to be used as well as where it should not be used.  Note that you are generating the intended use for G-Eval, not the intended use for the task!!"""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            criteria: str = dspy.InputField(desc="The evaluation criteria.")
            model_name: str = dspy.InputField(desc="The name of the model that is being used for G-Eval.")
            domain: str = dspy.OutputField(desc="The domain of the task.  Some examples are: Text Generation, Code Generation, Discourse, etc.")
            tasks: List[str] = dspy.OutputField(desc="A list of tasks that G-Eval is best suited to be used for.  Some examples are: Travel Planning, Code Review, Machine Translation, Dialogue Response Generation, etc.")
            best_suited_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where G-Eval is best suited to be used.  This can describe properties of the task, data, environment, etc. that would lead to successful evaluation when using G-Eval on this criteria. (approximately one sentence each)")
            not_recommended_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where G-Eval is not recommended to be used.  This can describe properties of the task, data, environment, etc. that would lead to unsuccessful evaluation when using G-Eval on this criteria. (approximately one sentence each)")

        with dspy.settings.context(lm=self.model):
            outputs = dspy.ChainOfThought(IntendedUseSignature)(
                task_description=self.task_description,
                criteria=self.evaluation_criteria,
                model_name=str(getattr(self.model, "model", self.model)),
            )
        
        return f"""- **Domain:** {outputs.domain}
- **Tasks:** {"\n  - " + "\n  - ".join(outputs.tasks)}
- **Best Suited For:** {"\n  - " + "\n  - ".join(outputs.best_suited_for_circumstances)}
- **Not Recommended For:** {"\n  - " + "\n  - ".join(outputs.not_recommended_for_circumstances)}"""

    def generate_metric_implementation(self):
        ref_type = "reference-based" if self.is_reference_based else "reference-free"
        return f"""### Reference Implementations

- **Libraries/Packages:**
  - [AutoMetrics G-Eval ({ref_type})](https://github.com/XenonMolecule/autometrics/blob/main/autometrics/metrics/generated/GeneratedGEvalMetric.py)
  - [Original G-Eval Paper](https://arxiv.org/abs/2303.16634)

### Computational Complexity

- **Efficiency:**
  - Requires a single LLM call per input-output pair with logprobs=True.
  - AutoMetrics does parallel calls on batched inputs.
  - G-Eval step generation is cached and reused across evaluations.

- **Scalability:**
  - Performance is linear in the number of input-output pairs.
  - Performance depends on the underlying LLM model and the dataset size.
  - Logprob extraction adds minimal overhead compared to regular LLM calls."""

    def generate_known_limitations(self):
        class KnownLimitationsSignature(dspy.Signature):
            """Given the task description, and an evaluation criteria, consider a G-Eval metric that is evaluating the text along this criteria.  Your task is to generate a list of biases, task misalignment risks, and failure cases that could be present in this evaluation.  Especially consider the criteria and how it is aligned or misaligned with BOTH this task and other tasks that G-Eval may be used for.  Note that you are generating the known limitations for G-Eval, not the known limitations for the task!!"""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            criteria: str = dspy.InputField(desc="The evaluation criteria.")
            model_name: str = dspy.InputField(desc="The name of the model that is being used for G-Eval.")
            biases: List[str] = dspy.OutputField(desc="A list of biases the could be present in this evaluation (approximately one sentence each).")
            task_misalignment_risks: List[str] = dspy.OutputField(desc="A list of ways in which this evaluation could be misaligned with the task (approximately one sentence each).")
            failure_cases: List[str] = dspy.OutputField(desc="A list of failure cases that could occur in this evaluation (approximately one sentence each).")

        with dspy.settings.context(lm=self.model):
            outputs = dspy.ChainOfThought(KnownLimitationsSignature)(
                task_description=self.task_description,
                criteria=self.evaluation_criteria,
                model_name=str(getattr(self.model, "model", self.model)),
            )
        
        return f"""- **Biases:** {"\n  - " + "\n  - ".join(outputs.biases)}
- **Task Misalignment Risks:** {"\n  - " + "\n  - ".join(outputs.task_misalignment_risks)}
- **Failure Cases:** {"\n  - " + "\n  - ".join(outputs.failure_cases)}"""

    def generature_further_reading(self):
        return generate_further_reading(self) + "\n  - [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)\n  - [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://openreview.net/pdf?id=uccHPGDlao)"

    def _generate_metric_card(self, author_model: Optional[dspy.LM] = None):
        """Produce a metric card via a custom builder."""
        
        class GEvalMetricCardBuilder(MetricCardBuilder):
            def metric_details(self) -> str:
                if self.metric.is_reference_based:
                    return self.metric.generate_metric_details_ref_based()
                else:
                    return self.metric.generate_metric_details_ref_free()
            
            def intended_use(self) -> str:
                return self.metric.generate_intended_use()
            
            def metric_implementation(self) -> str:
                return self.metric.generate_metric_implementation()
            
            def known_limitations(self) -> str:
                return self.metric.generate_known_limitations()
            
            def further_reading(self) -> str:
                return self.metric.generature_further_reading()

        with dspy.settings.context(lm=author_model or self.model):
            builder = GEvalMetricCardBuilder(self)
            return builder.build()


class GeneratedRefFreeGEvalMetric(_GEvalMetricMixin, GeneratedRefFreeMetric):
    """Reference-free metric that leverages G-Eval to evaluate outputs along evaluation criteria.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    evaluation_criteria    The textual criteria used for evaluation (e.g. "Clarity: How clear is …")
    model           A *dspy.LM* instance (or wrapper exposing .model attribute) used for evaluation
    task_description Optional task context passed to G-Eval
    evaluation_steps Pre-defined evaluation steps (optional)
    auto_generate_steps Whether to auto-generate evaluation steps if not provided
    possible_scores List of possible score values (default [1,2,3,4,5])
    criteria_generation_model Separate model for generating evaluation criteria (optional, defaults to main model)
    metric_card_author_model  LLM used to generate the metric-card (defaults to *model*)
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = False
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del references, kwargs  # pragma: no cover
        return self._call_geval(input, output)


class GeneratedRefBasedGEvalMetric(_GEvalMetricMixin, GeneratedRefBasedMetric):
    """Reference-based metric that leverages G-Eval to evaluate outputs along evaluation criteria using reference text.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    evaluation_criteria    The textual criteria used for evaluation (e.g. "Clarity: How clear is …")
    model           A *dspy.LM* instance (or wrapper exposing .model attribute) used for evaluation
    task_description Optional task context passed to G-Eval
    evaluation_steps Pre-defined evaluation steps (optional)
    auto_generate_steps Whether to auto-generate evaluation steps if not provided
    possible_scores List of possible score values (default [1,2,3,4,5])
    criteria_generation_model Separate model for generating evaluation criteria (optional, defaults to main model)
    metric_card_author_model  LLM used to generate the metric-card (defaults to *model*)
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = True
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del kwargs  # pragma: no cover
        return self._call_geval(input, output, references) 