import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import re
import time

import dspy

from autometrics.metrics.generated.utils.utils import generate_llm_constructor_code
from autometrics.metrics.generated.utils.metric_card import generate_further_reading
from autometrics.metrics.generated.utils.metric_card import MetricCardBuilder
from autometrics.metrics.generated.GeneratedRefFreeMetric import GeneratedRefFreeMetric
from autometrics.metrics.generated.GeneratedRefBasedMetric import GeneratedRefBasedMetric
from autometrics.metrics.Metric import MetricResult

__all__ = ["GeneratedRefFreeLLMJudgeMetric", "GeneratedRefBasedLLMJudgeMetric"]


# DSPy signatures for reference-free and reference-based metrics
class _LLMJudgeSignatureRefFree(dspy.Signature):
    """Given the task description, and an evaluation axis, rate the output text along the axis. It may be helpful to use the input text as context."""

    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
    input_text: str = dspy.InputField(desc="The text that was input to the model to produce the output text.")
    output_text: str = dspy.InputField(desc="The text that was produced by the model (this is the text that we want to rate).")
    score: int = dspy.OutputField(desc="A numerical score 1-5.")


class _LLMJudgeSignatureRefBased(dspy.Signature):
    """Given the task description, and an evaluation axis, rate the output text along the axis using the reference text as guidance."""

    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
    input_text: str = dspy.InputField(desc="The text that was input to the model to produce the output text.")
    reference_text: str = dspy.InputField(desc="The reference text to compare against.")
    output_text: str = dspy.InputField(desc="The text that was produced by the model (this is the text that we want to rate).")
    score: int = dspy.OutputField(desc="A numerical score 1-5.")


# Base mixin for shared LLM judge functionality
class _LLMJudgeMetricMixin:
    """Shared functionality for both reference-free and reference-based LLM judge metrics."""

    DEFAULT_MAX_WORKERS = 32
    # These judges use DSPy ChainOfThought and can provide reasoning
    has_feedback = True

    def __init__(
        self,
        name: str,
        description: str,
        axis: str,
        model: dspy.LM,
        task_description: Optional[str] = None,
        metric_card: Optional[str] = None,
        metric_card_author_model: Optional[dspy.LM] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        is_reference_based: bool = False,
        rubric: Optional[dict] = None,
        **kwargs,
    ):
        self.axis = axis
        self.task_description = task_description or "None"
        self.model = model
        # Capture reconstructable model constructor code for later export, before any cleanup
        try:
            from autometrics.metrics.generated.utils.utils import generate_llm_constructor_code as _gen_llm_code
            self._model_ctor_code = _gen_llm_code(model) if model is not None else None
        except Exception:
            self._model_ctor_code = None
        self.model_str = str(getattr(model, "model", model))
        self.max_workers = max_workers
        self.is_reference_based = kwargs.get("is_reference_based", is_reference_based)
        self.rubric = rubric  # Store the rubric if provided

        if metric_card_author_model is None:
            metric_card_author_model = model if isinstance(model, dspy.LM) else None

        if metric_card == "provided":
            self.metric_card = self.__doc__
            metric_card = self.metric_card

        # Initialize parent with shared parameters
        super().__init__(
            name,
            description,
            metric_card=metric_card,
            metric_card_author_model=metric_card_author_model,
            axis=axis,
            model_str=str(getattr(model, "model", model)),
            task_description=self.task_description,
            **kwargs,
        )

        # Exclude heavy objects from cache key
        self.exclude_from_cache_key("model")

        # Prepare the DSPy module based on reference type
        signature = _LLMJudgeSignatureRefBased if is_reference_based else _LLMJudgeSignatureRefFree
        self._judge_module = dspy.ChainOfThought(signature)

    def _has_structured_rubric(self) -> bool:
        """Check if this metric has a structured rubric vs. just an axis."""
        return (hasattr(self, 'rubric') and 
                isinstance(getattr(self, 'rubric'), dict) and 
                'score1_description' in getattr(self, 'rubric', {}))

    def _format_rubric_as_markdown(self) -> List[str]:
        """Format the rubric as a markdown table."""
        lines = [
            "| Score | Description |",
            "|-------|-------------|"
        ]
        
        # Add score descriptions
        for i in range(1, 6):
            score_key = f"score{i}_description"
            if score_key in self.rubric:
                description = self.rubric[score_key]
                # Format bullet points properly for markdown
                # Replace bullet points with proper markdown and add line breaks
                if description.startswith("- "):
                    # Split by bullet points and rejoin with proper markdown formatting
                    bullets = [bullet.strip() for bullet in description.split("- ") if bullet.strip()]
                    formatted_description = "• " + "<br/>• ".join(bullets)
                else:
                    bullets = [bullet.strip() for bullet in description.split("- ") if bullet.strip()]
                    formatted_description = "<br/>• ".join(bullets)
                lines.append(f"| {i} | {formatted_description} |")
            else:
                lines.append(f"| {i} | N/A |")
        
        return lines

    def _call_llm(self, input_text: str, output_text: str, references: Optional[str] = None) -> MetricResult:
        input_text = str(input_text) if input_text is not None else ""
        output_text = str(output_text) if output_text is not None else ""
        if references is not None:
            if isinstance(references, list):
                references = [str(ref) if ref is not None else "" for ref in references]
                reference_text = references[0] if references else ""
            else:
                reference_text = str(references)
        else:
            reference_text = None

        def _invoke(task_desc: str):
            with dspy.settings.context(lm=self.model):
                if self.is_reference_based and reference_text is not None:
                    return self._judge_module(
                        task_description=task_desc,
                        axis=self.axis,
                        input_text=input_text,
                        reference_text=reference_text,
                        output_text=output_text,
                        lm=self.model,
                    )
                else:
                    return self._judge_module(
                        task_description=task_desc,
                        axis=self.axis,
                        input_text=input_text,
                        output_text=output_text,
                        lm=self.model,
                    )

        rate_limit_retries_left = 3
        while True:
            try:
                pred = _invoke(self.task_description)
                try:
                    score_val = float(pred.score)
                except Exception:
                    try:
                        score_val = float(str(pred.score).strip())
                    except Exception:
                        score_val = 0.0
                feedback = getattr(pred, 'reasoning', '')
                return MetricResult(score=score_val, feedback=feedback)
            except Exception as e:
                msg = str(e)
                if (
                    'RateLimitError' in msg or 'Rate limit' in msg or 'rate limit' in msg or '429' in msg
                    or 'Too Many Requests' in msg or 'rate_limit_exceeded' in msg or 'quota' in msg
                    or 'exceeded your current quota' in msg
                ) and rate_limit_retries_left > 0:
                    wait_seconds = 30.0
                    try:
                        m = re.search(r"Please try again in\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds|s|sec|seconds)\b", msg, re.IGNORECASE)
                        if m:
                            _val = float(m.group(1)); _unit = m.group(2).lower(); wait_seconds = _val / 1000.0 if _unit.startswith('m') else _val
                        else:
                            m = re.search(r"Retry-After\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds|s|sec|seconds)?", msg, re.IGNORECASE)
                            if m:
                                _val = float(m.group(1)); _unit = (m.group(2) or 's').lower(); wait_seconds = _val / 1000.0 if _unit.startswith('m') else _val
                    except Exception:
                        pass
                    wait_seconds = max(0.0, min(wait_seconds, 30.0))
                    rate_limit_retries_left -= 1
                    time.sleep(wait_seconds)
                    continue
                needs_retry = ('Adapter JSONAdapter failed to parse' in msg or 'ContextWindowExceededError' in msg or 'response was truncated' in msg or 'exceeding max_tokens' in msg)
                if not needs_retry:
                    raise
                retry_task_desc = (self.task_description or '') + ' '
                pred = _invoke(retry_task_desc)
                score_val = float(pred.score)
                feedback = getattr(pred, 'reasoning', '')
                return MetricResult(score=score_val, feedback=feedback)

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        del kwargs  # pragma: no cover
        results: List[float] = [0.0] * len(outputs)

        # Fail-fast if workers=1
        if self.max_workers == 1:
            return [self._call_llm(i, o, r) for i, o, r in zip(inputs, outputs, references or [None] * len(outputs))]

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_llm, i, o, r): idx 
                for idx, (i, o, r) in enumerate(zip(inputs, outputs, references or [None] * len(outputs)))
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                mr = fut.result()
                results[idx] = mr.score
        return results

    def _calculate_with_feedback_impl(self, input, output, references=None, **kwargs):
        del kwargs  # pragma: no cover
        return self._call_llm(input, output, references)

    def _calculate_batched_with_feedback_impl(self, inputs, outputs, references=None, **kwargs):
        del kwargs  # pragma: no cover
        results: List[MetricResult] = [MetricResult(0.0, "")] * len(outputs)
        # Fail-fast if workers=1
        if self.max_workers == 1:
            return [self._call_llm(i, o, r) for i, o, r in zip(inputs, outputs, references or [None] * len(outputs))]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_llm, i, o, r): idx 
                for idx, (i, o, r) in enumerate(zip(inputs, outputs, references or [None] * len(outputs)))
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return results

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _generate_python_code(self, include_metric_card: bool = True) -> str:
        """Export a standalone python file that re-creates this metric."""
        class_name = "GeneratedRefBasedLLMJudgeMetric" if self.is_reference_based else "GeneratedRefFreeLLMJudgeMetric"
        

        _default_model_code = generate_llm_constructor_code(self.model)

        code = f"""# Auto-generated metric file for {self.name}
import dspy
import os
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import {class_name}
from typing import ClassVar

DEFAULT_MODEL = {_default_model_code}

class {self.name.replace(" ", "_").replace("-", "_")}_LLMJudge({class_name}):
    \"\"\"{self.metric_card if include_metric_card else ""}\"\"\"

    description: ClassVar[str] = {json.dumps(self.description)}

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name={json.dumps(self.name)},
            description={json.dumps(self.description)},
            axis={json.dumps(self.axis)},
            model=model,
            task_description={json.dumps(self.task_description)},
            metric_card={json.dumps("provided" if include_metric_card else "None")},
            max_workers={self.max_workers},
        )

    def __repr__(self):
        return f"{self.name.replace(' ', '_').replace('-', '_')}_LLMJudge(model={generate_llm_constructor_code(self.model).replace("\"", "\\\"")})"

"""
        return code
    
    def _serialize(self) -> dict:
        """Serialize the metric to a dictionary for in-memory operations."""
        return {
            "name": self.name,
            "description": self.description,
            "axis": self.axis,
            "model": generate_llm_constructor_code(self.model),
            "task_description": self.task_description,
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
        # You may want additional validation here
        model = eval(model_code)
        data["model"] = model
        
        return cls(**data)
    
    # ------------------------------------------------------------------
    # Metric-card helpers
    # ------------------------------------------------------------------

    def _metric_details_template(self, *, reference_based: bool) -> str:
        """Return the *Metric Details* section for ref-free / ref-based judges.

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
            f"**{self.name}** is a **{kind}** LLM-as-a-Judge metric that prompts an LLM to rate a system output along a single, run-time-specified evaluation axis.",
            f"In this case the axis is `{self.description}`.",
            "",
            "The prompt supplies:",
            "",
            "1. **Task description** *d*",
            f"2. **{'Rubric' if self._has_structured_rubric() else 'Axis rubric'}** `{self.description}`",
            "3. **Input text** *x*",
        ]
        if reference_based:
            lines.append("4. **Reference text** *r*")
            lines.append("5. **Output text** *y*")
        else:
            lines.append("4. **Output text** *y*")

        # --- Scoring sentence --------------------------------------------
        lines.extend(
            [
                "",
                r"Greedy decoding (temperature = 0) yields an integer score "
                r"$\hat{s}\!\in\!\{1,2,3,4,5\}$; higher = better adherence "
                "to the axis.",
                "",
                "- **Metric Type:** LLM as a Judge",
                "- **Range:** 1-5 (1 = worst, 5 = best)",
                "- **Higher is Better?:** Yes",
                f"- **Reference-Based?:** {ref_flag}",
                f"- **Input-Required?:** {input_req}",
                "",
                "### Formal Definition",
                "",
                r"Let $f _{\\theta}$ be the LLM and",
            ]
        )

        if reference_based:
            lines.append(
                r"$\pi _{\text{RB}}(d,\{axis\},x,r,y)$ construct the textual "
                "prompt."
            )
        else:
            lines.append(
                r"$\pi _{\text{RF}}(d,\{axis\},x,y)$ construct the textual "
                "prompt."
            )

        lines.extend(
            [
                "",
                "$$",
                r"\hat{s} \;=\; \operatorname*{arg\,max}\limits_{s \in "
                r"\{1,\dots,5\}} "
                r"f _{\theta}\!\bigl("
                r"s \,\bigl|\, "
                + (r"\pi _{\text{RB}}(d,\{axis\},x,r,y)"
                   if reference_based
                   else r"\pi _{\text{RF}}(d,\{axis\},x,y)")
                + r"\bigr)",
                "$$",
                "",
                r"The metric value is "
                + (
                    r"$\operatorname{LJ}^{\text{RB}}_{\{axis\}}"
                    r"(d,x,r,y)=\hat{s}$."
                    if reference_based
                    else r"$\operatorname{LJ}^{\text{RF}}_{\{axis\}}"
                    r"(d,x,y)=\hat{s}$."
                ),
                "",
            ]
        )

        # Add rubric details section if we have a structured rubric
        if self._has_structured_rubric():
            rubric = getattr(self, 'rubric', {})
            lines.extend([
                "### Rubric Details",
                "",
                f"**Criteria:** {rubric.get('criteria', 'N/A')}",
                "",
                "#### Scoring Rubric",
                "",
            ])
            lines.extend(self._format_rubric_as_markdown())
            lines.append("")

        lines.extend([
                "### Inputs and Outputs",
                "- **Inputs:**",
                "  - **Task description** *d*",
                f"  - **{'Rubric' if self._has_structured_rubric() else 'Axis rubric'}** `{self.description}`",
                "  - **Input text** *x*",
            ])
        if reference_based:
            lines.append("  - **Reference text** *r*")
        lines.append("  - **Output text** *y*")
        lines.extend(
            [
                "- **Outputs:**",
                "  - Scalar score "
                r"$\hat{s} \in \{1,2,3,4,5\}$",
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
            """Given the task description, and an evaluation axis, consider an LLM Judge that is evaluating the text along this axis.  Your task is to generate the domain, a list of tasks, and a set of circumstances where the LLM Judge is best suited to be used as well as where it should not be used.  Note that you are generating the intended use for the LLM Judge, not the intended use for the task!!"""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
            model_name: str = dspy.InputField(desc="The name of the model that is being used as the LLM Judge.")
            domain: str = dspy.OutputField(desc="The domain of the task.  Some examples are: Text Generation, Code Generation, Discourse, etc.")
            tasks: List[str] = dspy.OutputField(desc="A list of tasks that the LLM Judge is best suited to be used for.  Some examples are: Travel Planning, Code Review, Machine Translation, Dialogue Response Generation, etc.")
            best_suited_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where the LLM Judge is best suited to be used.  This can describe properties of the task, data, environment, etc. that would lead to successful evaluation when using the LLM Judge on this axis. (approximately one sentence each)")
            not_recommended_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where the LLM Judge is not recommended to be used.  This can describe properties of the task, data, environment, etc. that would lead to unsuccessful evaluation when using the LLM Judge on this axis. (approximately one sentence each)")

        with dspy.settings.context(lm=self.model):
            outputs = dspy.ChainOfThought(IntendedUseSignature)(
                task_description=self.task_description,
                axis=self.axis,
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
  - [AutoMetrics LLM as a Judge ({ref_type})](https://github.com/XenonMolecule/autometrics/blob/main/autometrics/metrics/generated/GeneratedLLMJudgeMetric.py)

### Computational Complexity

- **Efficiency:**
  - Requires a single LLM call per input-output pair.
  - AutoMetrics does parallel calls on batched inputs.

- **Scalability:**
  - Performance is linear in the number of input-output pairs.
  - Performance depends on the underlying LLM model and the dataset size.  Additional consideration would include whether or not the LLM is a reasoning model."""

    def generate_known_limitations(self):
        class KnownLimitationsSignature(dspy.Signature):
            """Given the task description, and an evaluation axis, consider an LLM Judge that is evaluating the text along this axis.  Your task is to generate a list of biases, task misalignment risks, and failure cases that could be present in this evaluation.  Especially consider the axis and how it is aligned or misaligned with BOTH this task and other tasks that the LLM Judge may be used for.  Note that you are generating the known limitations for the LLM Judge, not the known limitations for the task!!"""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
            model_name: str = dspy.InputField(desc="The name of the model that is being used as the LLM Judge.")
            biases: List[str] = dspy.OutputField(desc="A list of biases the could be present in this evaluation (approximately one sentence each).")
            task_misalignment_risks: List[str] = dspy.OutputField(desc="A list of ways in which this evaluation could be misaligned with the task (approximately one sentence each).")
            failure_cases: List[str] = dspy.OutputField(desc="A list of failure cases that could occur in this evaluation (approximately one sentence each).")

        with dspy.settings.context(lm=self.model):
            outputs = dspy.ChainOfThought(KnownLimitationsSignature)(
                task_description=self.task_description,
                axis=self.axis,
                model_name=str(getattr(self.model, "model", self.model)),
            )
        
        return f"""- **Biases:** {"\n  - " + "\n  - ".join(outputs.biases)}
- **Task Misalignment Risks:** {"\n  - " + "\n  - ".join(outputs.task_misalignment_risks)}
- **Failure Cases:** {"\n  - " + "\n  - ".join(outputs.failure_cases)}"""

    def generature_further_reading(self):
        return generate_further_reading(self) + "\n  - [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://openreview.net/pdf?id=uccHPGDlao)"

    def _generate_metric_card(self, author_model: Optional[dspy.LM] = None):
        """Produce a metric card via a custom builder."""
        
        class LLMJudgeMetricCardBuilder(MetricCardBuilder):
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
            builder = LLMJudgeMetricCardBuilder(self)
            return builder.build()


class GeneratedRefFreeLLMJudgeMetric(_LLMJudgeMetricMixin, GeneratedRefFreeMetric):
    """Reference-free metric that leverages an LLM to judge outputs along a textual axis.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    axis            The textual axis/rubric used for judgement (e.g. "*Clarity*: How clear is …")
    model           A *dspy.LM* instance (or wrapper exposing .model attribute) used for judging
    task_description Optional task context passed to the judge
    metric_card_author_model  LLM used to generate the metric-card (defaults to *model*)
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = False
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del references, kwargs  # pragma: no cover
        return self._call_llm(input, output).score


class GeneratedRefBasedLLMJudgeMetric(_LLMJudgeMetricMixin, GeneratedRefBasedMetric):
    """Reference-based metric that leverages an LLM to judge outputs along a textual axis using reference text.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    axis            The textual axis/rubric used for judgement (e.g. "*Clarity*: How clear is …")
    model           A *dspy.LM* instance (or wrapper exposing .model attribute) used for judging
    task_description Optional task context passed to the judge
    metric_card_author_model  LLM used to generate the metric-card (defaults to *model*)
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = True
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del kwargs  # pragma: no cover
        return self._call_llm(input, output, references).score