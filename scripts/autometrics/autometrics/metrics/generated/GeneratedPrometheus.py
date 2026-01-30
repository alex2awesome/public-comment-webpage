import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any
import math
import pandas as pd
from tqdm import tqdm
import os

import dspy

from prometheus_eval.litellm import LiteLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from IPython.display import display, HTML

from autometrics.metrics.generated.utils.utils import generate_llm_constructor_code
from autometrics.metrics.generated.utils.metric_card import generate_further_reading
from autometrics.metrics.generated.utils.metric_card import MetricCardBuilder
from autometrics.metrics.generated.GeneratedRefFreeMetric import GeneratedRefFreeMetric
from autometrics.metrics.generated.GeneratedRefBasedMetric import GeneratedRefBasedMetric

__all__ = ["GeneratedRefFreePrometheusMetric", "GeneratedRefBasedPrometheusMetric"]


# Base mixin for shared Prometheus functionality
class _PrometheusMetricMixin:
    """Shared functionality for both reference-free and reference-based Prometheus metrics."""

    DEFAULT_MAX_WORKERS = 32

    def __init__(
        self,
        name: str,
        description: str,
        rubric: Dict[str, str],
        model: Optional[Any] = None,
        judge: Optional[PrometheusEval] = None,
        task_description: Optional[str] = None,
        judge_api_base: str = "http://jagupard37:8000/v1",
        metric_card: Optional[str] = None,
        metric_card_author_model: Optional[dspy.LM] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        is_reference_based: bool = False,
        **kwargs,
    ):
        # Ensure LITELLM_PROXY_API_KEY is set to 'None' if not already set
        if 'LITELLM_PROXY_API_KEY' not in os.environ:
            os.environ['LITELLM_PROXY_API_KEY'] = 'None'

        self.rubric = rubric
        self.task_description = task_description or "No task description provided"
        self.judge_api_base = judge_api_base
        self.max_workers = max_workers
        self.is_reference_based = kwargs.get("is_reference_based", is_reference_based)

        # Handle model and judge initialization
        if judge is not None:
            # If judge is provided, use it and extract model info
            self.judge = judge
            self.model = getattr(judge, 'model', None)
            self.model_str = "litellm_proxy/Unbabel/M-Prometheus-14B"  # Use clean model name
            self.judge_api_base = str(getattr(judge, 'api_base', judge_api_base))
        else:
            # No judge provided, need to create one
            # Check if model is a DSPy LM (which doesn't work with PrometheusEval) 
            if model is not None and hasattr(model, '__class__') and 'dspy' in str(type(model)):
                print(f"DEBUG: Detected DSPy LM model, creating default Prometheus LiteLLM model instead")
                # Don't try to convert DSPy LM, just use default Prometheus model
                model = None
            
            # Initialize default model if not provided or if DSPy LM was detected
            if model is None:
                print(f"DEBUG: Creating default LiteLLM model with api_base={judge_api_base}")
                try:
                    # Use correct LiteLLM constructor: LiteLLM(name, api_base)
                    model = LiteLLM('litellm_proxy/Unbabel/M-Prometheus-14B', api_base=judge_api_base)
                    print(f"DEBUG: Successfully created default LiteLLM model")
                except Exception as e:
                    print(f"ERROR: Failed to create default LiteLLM model: {e}")
                    raise RuntimeError(f"Could not create Prometheus model: {e}")
            
            self.model = model
            self.model_str = "litellm_proxy/Unbabel/M-Prometheus-14B"  # Use clean model name
            
            # Create PrometheusEval with the model
            try:
                self.judge = PrometheusEval(model=self.model, absolute_grade_template=ABSOLUTE_PROMPT)
                print(f"DEBUG: Successfully created PrometheusEval")
            except Exception as e:
                print(f"ERROR: Failed to create PrometheusEval with model: {e}")
                raise RuntimeError(f"Could not create PrometheusEval: {e}")

        if metric_card_author_model is None:
            metric_card_author_model = dspy.settings.lm if hasattr(dspy.settings, 'lm') else None

        if metric_card == "provided":
            self.metric_card = self.__doc__
            metric_card = self.metric_card

        # Initialize parent with shared parameters
        try:
            super().__init__(
                name,
                description,
                metric_card=metric_card,
                metric_card_author_model=metric_card_author_model,
                rubric=rubric,
                model_str=self.model_str,
                task_description=self.task_description,
                judge_api_base=judge_api_base,
                **kwargs,
            )
        except AssertionError as e:
            if "No LM is loaded" in str(e):
                raise RuntimeError(
                    "DSPy metric card generation failed: No LM is loaded. "
                    "To fix this, either:\n"
                    "1. Pass a valid LM as metric_card_author_model parameter, or\n"
                    "2. Set dspy.settings.lm to a valid LM instance, or\n"
                    "3. Use metric_card='provided' to skip automatic generation"
                ) from e
            else:
                raise

        # Exclude heavy objects from cache key
        self.exclude_from_cache_key("model", "judge")

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
                    formatted_description = description
                lines.append(f"| {i} | {formatted_description} |")
            else:
                lines.append(f"| {i} | N/A |")
        
        return lines

    def _call_prometheus(self, input_text: str, output_text: str, references: Optional[str] = None) -> float:
        input_text = str(input_text) if input_text is not None else ""
        output_text = str(output_text) if output_text is not None else ""
        if references is not None:
            if isinstance(references, list):
                references = "\n".join([str(ref) for ref in references if ref is not None])
            else:
                references = str(references)
        
        # Format the rubric for Prometheus evaluation
        rubric_formatted = f"""**Criteria:** {self.rubric['criteria']}

**Score 1:** {self.rubric['score1_description']}

**Score 2:** {self.rubric['score2_description']}

**Score 3:** {self.rubric['score3_description']}

**Score 4:** {self.rubric['score4_description']}

**Score 5:** {self.rubric['score5_description']}"""

        # Include task description if available
        input_with_task = input_text
        if self.task_description and self.task_description != "No task description provided":
            input_with_task = f"Task: {self.task_description}\n\nInput: {input_text}"

        reference = references if self.is_reference_based else None
        
        # Call Prometheus evaluation - remove use_tqdm to avoid parameter conflict
        feedback, score = self.judge.single_absolute_grade(
            instruction=input_with_task,
            response=output_text,
            rubric=rubric_formatted,
            reference_answer=reference
        )

        return float(score)

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        del kwargs  # pragma: no cover
        results: List[float] = [0.0] * len(outputs)

        # Fail-fast if workers=1
        if self.max_workers == 1:
            for i, (inp, out, ref) in enumerate(zip(inputs, outputs, references or [None] * len(outputs))):
                # FIXED: Let errors propagate naturally instead of catching and returning 0.0
                # This allows the cache to distinguish between failures and valid results
                results[i] = self._call_prometheus(inp, out, ref)
            return results

        # Use ThreadPoolExecutor to process each item in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._call_prometheus, i, o, r): idx 
                for idx, (i, o, r) in enumerate(zip(inputs, outputs, references or [None] * len(outputs)))
            }

            # Initialize tqdm progress bar
            with tqdm(total=len(futures), desc="Processing Prometheus Evaluation") as pbar:
                for future in as_completed(futures):
                    index = futures[future]
                    # FIXED: Let errors propagate naturally instead of catching and returning 0.0
                    # This allows the cache to distinguish between failures and valid results
                    results[index] = future.result()
                    pbar.update(1)

        return results

    # ------------------------------------------------------------------
    # Rubric display helpers
    # ------------------------------------------------------------------

    def display_rubric(self, rubric=None, metric_title=None):
        """
        Display the rubric in a tabular format for Jupyter Notebooks with enhanced text visibility.
        
        Parameters:
        rubric (dict, optional): A dictionary containing the rubric criteria and score descriptions.
                    If None, uses self.rubric.
        metric_title (str, optional): Title of the metric to display above the rubric table.
        
        Returns:
        None: Displays the rubric as a table in Jupyter Notebook.
        """
        if rubric is None:
            rubric = self.rubric
            
        pd.set_option('display.max_colwidth', None)

        # Create a pandas DataFrame to hold the rubric
        rubric_df = pd.DataFrame({
            "Criteria": [rubric.get("criteria", "N/A")],
            "Score 1": [rubric.get("score1_description", "N/A")],
            "Score 2": [rubric.get("score2_description", "N/A")],
            "Score 3": [rubric.get("score3_description", "N/A")],
            "Score 4": [rubric.get("score4_description", "N/A")],
            "Score 5": [rubric.get("score5_description", "N/A")]
        })

        # Apply custom CSS to ensure proper text wrapping and visibility
        styled_rubric = rubric_df.style.set_table_styles(
            [{
                'selector': 'td',
                'props': [('white-space', 'pre-wrap'), ('word-wrap', 'break-word')]
            }]
        )

        # Display the title if provided
        if metric_title:
            display(HTML(f"<h3 style='text-align: left;'>{metric_title}</h3>"))

        # Display the styled DataFrame
        display(styled_rubric)

    def display(self):
        """
        Display the metric rubric in a tabular format for Jupyter Notebooks.
        """
        self.display_rubric(metric_title=self.name)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _generate_python_code(self, include_metric_card: bool = True) -> str:
        """Export a standalone python file that re-creates this metric."""
        class_name = "GeneratedRefBasedPrometheusMetric" if self.is_reference_based else "GeneratedRefFreePrometheusMetric"
        
        # Clean up the class name: Much simpler now that names are clean from source
        clean_metric_name = (self.name
                           .replace(" ", "_")   # Replace spaces
                           .replace("-", "_")   # Replace hyphens  
                           .replace("/", "_"))  # Fix any remaining slashes
        
        generated_class_name = f"{clean_metric_name}_Metric"
        
        # Extract actual API base from the model or use default
        actual_api_base = self.judge_api_base
        if hasattr(self.model, 'api_base') and self.model.api_base:
            actual_api_base = self.model.api_base
        elif hasattr(self.model, 'base_url') and self.model.base_url:
            actual_api_base = self.model.base_url
        
        # Generate the model constructor code  
        model_name = "litellm_proxy/Unbabel/M-Prometheus-14B"
        model_constructor = f'LiteLLM("{model_name}", api_base="{actual_api_base}")'
        
        code = f"""# Auto-generated Prometheus metric file for {clean_metric_name}
import os
from prometheus_eval.litellm import LiteLLM
from autometrics.metrics.generated.GeneratedPrometheus import {class_name}
from typing import ClassVar

DEFAULT_MODEL = {model_constructor}

class {generated_class_name}({class_name}):
    \"\"\"{self.metric_card if include_metric_card else ""}\"\"\"

    description: ClassVar[str] = {json.dumps(self.description)}
    
    def __init__(self, model=DEFAULT_MODEL, judge_api_base="{actual_api_base}"):
        super().__init__(
            name={json.dumps(self.name)},
            description={json.dumps(self.description)},
            rubric={json.dumps(self.rubric)},
            model=model,
            task_description={json.dumps(self.task_description)},
            judge_api_base=judge_api_base,
            metric_card={json.dumps("provided" if include_metric_card else "None")},
            max_workers={self.max_workers},
        )

    def __repr__(self):
        return f"{generated_class_name}(model=LiteLLM('{model_name}', api_base='{actual_api_base}'))"

"""
        return code

    def _serialize(self) -> dict:
        """Serialize the metric to a dictionary for in-memory operations."""
        return {
            "name": self.name,
            "description": self.description,
            "rubric": self.rubric,
            "model_str": self.model_str,
            "task_description": self.task_description,
            "judge_api_base": self.judge_api_base,
            "metric_card": self.metric_card,
            "max_workers": self.max_workers,
            "is_reference_based": self.is_reference_based,
        }

    @classmethod
    def _deserialize(cls, data: dict):
        """Deserialize a dictionary to create a metric instance."""
        # Recreate the model from string representation
        model = LiteLLM(data.pop("model_str"), api_base=data["judge_api_base"], api_key="None")
        data["model"] = model
        
        return cls(**data)
    
    # ------------------------------------------------------------------
    # Metric-card helpers
    # ------------------------------------------------------------------

    def _metric_details_template(self, *, reference_based: bool) -> str:
        """Return the *Metric Details* section for ref-free / ref-based Prometheus metrics."""
        kind = "reference-based" if reference_based else "reference-free"
        ref_flag = "Yes" if reference_based else "No"
        input_req = "Yes (plus reference)" if reference_based else "Yes"

        # --- Header & description ----------------------------------------
        lines = [
            f"**{self.name}** is a **{kind}** Prometheus-based metric that uses detailed rubrics to evaluate system outputs.",
            f"The evaluation is performed using Prometheus-Eval with the following criteria: `{self.rubric.get('criteria', 'N/A')}`.",
            "",
            "Prometheus-Eval differs from regular LLM-as-a-Judge by:",
            "",
            "1. **Using specialized training** on evaluation tasks",
            "2. **Detailed rubrics** with score-specific descriptions",
            "3. **Consistent scoring** across different types of content",
            "",
            "The rubric provides:",
            "",
            "- **Clear criteria** for evaluation",
            "- **Score-specific descriptions** for scores 1-5",
            "- **Detailed guidelines** for consistent evaluation",
            "",
            "- **Metric Type:** Prometheus-based (rubric-guided LLM scoring)",
            "- **Range:** 1-5 (integer scores)",
            "- **Higher is Better?:** Yes",
            f"- **Reference-Based?:** {ref_flag}",
            f"- **Input-Required?:** {input_req}",
            "",
            "### Rubric Details",
            "",
            f"**Criteria:** {self.rubric.get('criteria', 'N/A')}",
            "",
            "#### Scoring Rubric",
            "",
        ]
        
        # Add rubric as markdown table
        lines.extend(self._format_rubric_as_markdown())
        lines.append("")
        
        lines.extend([
            "### Inputs and Outputs",
            "- **Inputs:**",
            "  - **Task description** (if provided)",
            "  - **Input text** *x*",
        ])
        if reference_based:
            lines.append("  - **Reference text** *r*")
        lines.append("  - **Output text** *y*")
        lines.extend([
            "- **Outputs:**",
            "  - Integer score 1-5 based on rubric evaluation",
        ])

        return "\n".join(lines)
    
    def generate_metric_details_ref_free(self) -> str:
        """Metric-details section for the **reference-free** variant."""
        return self._metric_details_template(reference_based=False)

    def generate_metric_details_ref_based(self) -> str:
        """Metric-details section for the **reference-based** variant."""
        return self._metric_details_template(reference_based=True)

    def generate_intended_use(self):
        class IntendedUseSignature(dspy.Signature):
            """Given the task description, and a rubric, consider a Prometheus-based metric that is evaluating the text using this rubric.  Your task is to generate the domain, a list of tasks, and a set of circumstances where Prometheus evaluation with this rubric is best suited to be used as well as where it should not be used."""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            criteria: str = dspy.InputField(desc="The evaluation criteria.")
            rubric_summary: str = dspy.InputField(desc="A summary of the rubric scoring guidelines.")
            domain: str = dspy.OutputField(desc="The domain of the task. Some examples are: Text Generation, Code Generation, Discourse, etc.")
            tasks: List[str] = dspy.OutputField(desc="A list of tasks that Prometheus evaluation is best suited to be used for.")
            best_suited_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where Prometheus evaluation is best suited to be used. (approximately one sentence each)")
            not_recommended_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where Prometheus evaluation is not recommended to be used. (approximately one sentence each)")

        # Create rubric summary
        rubric_summary = f"Criteria: {self.rubric.get('criteria', 'N/A')}\n"
        for i in range(1, 6):
            score_desc = self.rubric.get(f'score{i}_description', 'N/A')
            rubric_summary += f"Score {i}: {score_desc[:100]}{'...' if len(score_desc) > 100 else ''}\n"

        with dspy.settings.context(lm=self.metric_card_author_model):
            outputs = dspy.ChainOfThought(IntendedUseSignature)(
                task_description=self.task_description,
                criteria=self.rubric.get('criteria', 'N/A'),
                rubric_summary=rubric_summary,
            )
        
        return f"""- **Domain:** {outputs.domain}
- **Tasks:** {"\n  - " + "\n  - ".join(outputs.tasks)}
- **Best Suited For:** {"\n  - " + "\n  - ".join(outputs.best_suited_for_circumstances)}
- **Not Recommended For:** {"\n  - " + "\n  - ".join(outputs.not_recommended_for_circumstances)}"""

    def generate_metric_implementation(self):
        ref_type = "reference-based" if self.is_reference_based else "reference-free"
        return f"""### Reference Implementations

- **Libraries/Packages:**
  - [AutoMetrics Prometheus ({ref_type})](https://github.com/XenonMolecule/autometrics/blob/main/autometrics/metrics/generated/GeneratedPrometheus.py)
  - [Prometheus-Eval](https://github.com/prometheus-eval/prometheus-eval)

### Computational Complexity

- **Efficiency:**
  - Requires a single LLM call per input-output pair using Prometheus model.
  - AutoMetrics does parallel calls on batched inputs.
  - Rubric formatting is cached and reused across evaluations.

- **Scalability:**
  - Performance is linear in the number of input-output pairs.
  - Performance depends on the Prometheus model and the dataset size.
  - Specialized training makes Prometheus efficient for evaluation tasks."""

    def generate_known_limitations(self):
        class KnownLimitationsSignature(dspy.Signature):
            """Given the task description, and a rubric, consider a Prometheus-based metric that evaluates text using this rubric. Your task is to generate a list of biases, task misalignment risks, and failure cases that could be present in this evaluation."""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            criteria: str = dspy.InputField(desc="The evaluation criteria.")
            rubric_summary: str = dspy.InputField(desc="A summary of the rubric scoring guidelines.")
            biases: List[str] = dspy.OutputField(desc="A list of biases that could be present in this evaluation (approximately one sentence each).")
            task_misalignment_risks: List[str] = dspy.OutputField(desc="A list of ways in which this evaluation could be misaligned with the task (approximately one sentence each).")
            failure_cases: List[str] = dspy.OutputField(desc="A list of failure cases that could occur in this evaluation (approximately one sentence each).")

        # Create rubric summary
        rubric_summary = f"Criteria: {self.rubric.get('criteria', 'N/A')}\n"
        for i in range(1, 6):
            score_desc = self.rubric.get(f'score{i}_description', 'N/A')
            rubric_summary += f"Score {i}: {score_desc[:100]}{'...' if len(score_desc) > 100 else ''}\n"

        with dspy.settings.context(lm=self.metric_card_author_model):
            outputs = dspy.ChainOfThought(KnownLimitationsSignature)(
                task_description=self.task_description,
                criteria=self.rubric.get('criteria', 'N/A'),
                rubric_summary=rubric_summary,
            )
        
        return f"""- **Biases:** {"\n  - " + "\n  - ".join(outputs.biases)}
- **Task Misalignment Risks:** {"\n  - " + "\n  - ".join(outputs.task_misalignment_risks)}
- **Failure Cases:** {"\n  - " + "\n  - ".join(outputs.failure_cases)}"""

    def generature_further_reading(self):
        return generate_further_reading(self) + "\n  - [Prometheus: Inducing Fine-grained Evaluation Capability in Language Models](https://arxiv.org/abs/2310.08491)\n  - [Prometheus-Eval Library](https://github.com/prometheus-eval/prometheus-eval)"

    def _generate_metric_card(self, author_model: Optional[dspy.LM] = None):
        """Produce a metric card via a custom builder."""
        
        class PrometheusMetricCardBuilder(MetricCardBuilder):
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

        with dspy.settings.context(lm=author_model or self.metric_card_author_model):
            builder = PrometheusMetricCardBuilder(self)
            return builder.build()


class GeneratedRefFreePrometheusMetric(_PrometheusMetricMixin, GeneratedRefFreeMetric):
    """Reference-free metric that leverages Prometheus-Eval to evaluate outputs using detailed rubrics.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    rubric          Dictionary containing rubric criteria and score descriptions
    model           A Prometheus model instance (LiteLLM) used for evaluation
    judge           Optional PrometheusEval instance (created automatically if not provided)
    task_description Optional task context passed to Prometheus
    judge_api_base  API base URL for the Prometheus model
    metric_card_author_model  LLM used to generate the metric-card (defaults to model)
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = False
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del references, kwargs  # pragma: no cover
        return self._call_prometheus(input, output)


class GeneratedRefBasedPrometheusMetric(_PrometheusMetricMixin, GeneratedRefBasedMetric):
    """Reference-based metric that leverages Prometheus-Eval to evaluate outputs using detailed rubrics and reference text.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    rubric          Dictionary containing rubric criteria and score descriptions
    model           A Prometheus model instance (LiteLLM) used for evaluation
    judge           Optional PrometheusEval instance (created automatically if not provided)
    task_description Optional task context passed to Prometheus
    judge_api_base  API base URL for the Prometheus model
    metric_card_author_model  LLM used to generate the metric-card (defaults to model)
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = True
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del kwargs  # pragma: no cover
        return self._call_prometheus(input, output, references) 