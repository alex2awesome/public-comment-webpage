import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
import re
import time

import dspy

from autometrics.metrics.generated.utils.utils import generate_llm_constructor_code
from autometrics.metrics.generated.utils.metric_card import generate_further_reading
from autometrics.metrics.generated.utils.metric_card import MetricCardBuilder
from autometrics.metrics.generated.utils.dspy_inspection import (
    inspect_dspy_program_from_path,
    load_optimized_program_from_embedded_data,
    format_examples_as_markdown_table
)
from autometrics.metrics.generated.GeneratedRefFreeMetric import GeneratedRefFreeMetric
from autometrics.metrics.generated.GeneratedRefBasedMetric import GeneratedRefBasedMetric
from autometrics.metrics.Metric import MetricResult

__all__ = ["GeneratedRefFreeOptimizedJudge", "GeneratedRefBasedOptimizedJudge"]


# DSPy signatures for reference-free and reference-based optimized judges
class _OptimizedJudgeSignatureRefFree(dspy.Signature):
    """Given the task description, and an evaluation target, rate the output text along the target. It may be helpful to use the input text as context. Use the conversation history for examples and guidance."""

    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    axis: str = dspy.InputField(desc="The evaluation axis / rubric / criterion (may or may not be super descriptive).")
    input_text: str = dspy.InputField(desc="The text that was input to the model to produce the output text.")
    output_text: str = dspy.InputField(desc="The text that was produced by the model (this is the text that we want to rate).")
    suggested_range: Tuple[float, float] = dspy.InputField(desc="The suggested range of possible values for the metric.")
    score: float = dspy.OutputField(desc="A numerical score which is within or close to the suggested range.")


class _OptimizedJudgeSignatureRefBased(dspy.Signature):
    """Given the task description, and an evaluation target, rate the output text along the target using the reference text as guidance. Use the conversation history for examples and guidance."""

    task_description: str = dspy.InputField(desc="Brief description of the underlying task.")
    axis: str = dspy.InputField(desc="The evaluation axis / rubric / criterion (may or may not be super descriptive).")
    input_text: str = dspy.InputField(desc="The text that was input to the model to produce the output text.")
    reference_text: str = dspy.InputField(desc="The reference text to compare against.")
    output_text: str = dspy.InputField(desc="The text that was produced by the model (this is the text that we want to rate).")
    suggested_range: Tuple[float, float] = dspy.InputField(desc="The suggested range of possible values for the metric.")
    score: float = dspy.OutputField(desc="A numerical score which is within or close to the suggested range.")


# Base mixin for shared optimized judge functionality
class _OptimizedJudgeMetricMixin:
    """Shared functionality for both reference-free and reference-based optimized LLM judge metrics."""

    DEFAULT_MAX_WORKERS = 32

    def __init__(
        self,
        name: str,
        description: str,
        axis: str,
        model: dspy.LM,
        task_description: Optional[str] = None,
        optimized_prompt_path: Optional[str] = None,
        suggested_range: Tuple[float, float] = (1, 5),
        metric_card: Optional[str] = None,
        metric_card_author_model: Optional[dspy.LM] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        is_reference_based: bool = False,
        **kwargs,
    ):
        self.axis = axis
        self.task_description = task_description or "None"
        self.model = model
        self.model_str = str(getattr(model, "model", model))
        self.max_workers = max_workers
        self.is_reference_based = kwargs.get("is_reference_based", is_reference_based)
        self.optimized_prompt_path = optimized_prompt_path
        self.suggested_range = suggested_range

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
            optimized_prompt_path=optimized_prompt_path,
            suggested_range=suggested_range,
            **kwargs,
        )

        # Exclude heavy objects from cache key
        self.exclude_from_cache_key("model", "_optimized_module")

        # Load the optimized DSPy module
        self._optimized_module = None
        self._load_optimized_module()
        
        # Cache for optimized prompt inspection
        self._prompt_data = None
        self._prompt_examples = None
        self._prompt_instructions = None

    def _load_optimized_module(self):
        """Load the optimized DSPy module from the saved path."""
        print(f"Loading optimized module from path: {self.optimized_prompt_path}")
        
        if self.optimized_prompt_path and os.path.exists(self.optimized_prompt_path):
            try:
                print(f"Optimized prompt file exists, size: {os.path.getsize(self.optimized_prompt_path)} bytes")
                
                # Create a temporary module to load the optimized prompt
                signature = _OptimizedJudgeSignatureRefBased if self.is_reference_based else _OptimizedJudgeSignatureRefFree
                temp_module = dspy.ChainOfThought(signature)
                
                print(f"Loading optimized prompt...")
                temp_module.load(self.optimized_prompt_path)
                self._optimized_module = temp_module
                
                print(f"✅ Successfully loaded optimized prompt from: {self.optimized_prompt_path}")
                
                # Try to inspect what was loaded
                try:
                    if hasattr(self._optimized_module, 'predict') and hasattr(self._optimized_module.predict, 'demos'):
                        demos = self._optimized_module.predict.demos
                        print(f"✅ Loaded module has {len(demos)} demos")
                        if demos:
                            print(f"✅ First demo keys: {demos[0].__dict__.keys() if hasattr(demos[0], '__dict__') else 'No __dict__'}")
                    else:
                        print("⚠️  Loaded module structure doesn't match expected format")
                except Exception as inspect_e:
                    print(f"⚠️  Could not inspect loaded module: {inspect_e}")
                    
            except Exception as e:
                print(f"❌ Failed to load optimized prompt from {self.optimized_prompt_path}: {e}")
                print(f"❌ Exception type: {type(e)}")
                import traceback
                traceback.print_exc()
                print("📝 Falling back to base signature...")
                signature = _OptimizedJudgeSignatureRefBased if self.is_reference_based else _OptimizedJudgeSignatureRefFree
                self._optimized_module = dspy.ChainOfThought(signature)
        else:
            if self.optimized_prompt_path:
                print(f"❌ Optimized prompt file does not exist: {self.optimized_prompt_path}")
            else:
                print("❌ No optimized prompt path provided")
            # Fallback to base signature if no optimized prompt path provided
            signature = _OptimizedJudgeSignatureRefBased if self.is_reference_based else _OptimizedJudgeSignatureRefFree
            self._optimized_module = dspy.ChainOfThought(signature)
            print("📝 Using base signature (no optimization)")

    # Optimized judges use DSPy CoT; enable feedback
    has_feedback = True

    def _call_optimized_llm(self, input_text: str, output_text: str, references: Optional[str] = None) -> MetricResult:
        input_text = str(input_text) if input_text is not None else ""
        output_text = str(output_text) if output_text is not None else ""
        if references is not None:
            if isinstance(references, list):
                references = [str(ref) if ref is not None else "" for ref in references]
                # Use first reference if multiple are provided
                reference_text = references[0] if references else ""
            else:
                reference_text = str(references)
        else:
            reference_text = None

        def _invoke(task_desc: str):
            with dspy.settings.context(lm=self.model):
                if self.is_reference_based and reference_text is not None:
                    return self._optimized_module(
                        task_description=task_desc,
                        axis=self.axis,
                        input_text=input_text,
                        reference_text=reference_text,
                        output_text=output_text,
                        suggested_range=self.suggested_range,
                        lm=self.model,
                    )
                else:
                    return self._optimized_module(
                        task_description=task_desc,
                        axis=self.axis,
                        input_text=input_text,
                        output_text=output_text,
                        suggested_range=self.suggested_range,
                        lm=self.model,
                    )

        # Convert score to float, handling various string formats
        # Attempt with special handling for parser/ctx and rate limit
        rate_limit_retries_left = 3
        while True:
            try:
                pred = _invoke(self.task_description)
                # Parse score
                score = pred.score
                try:
                    if isinstance(score, str):
                        if '\n' in score:
                            score = score.split('\n')[0]
                        m = re.search(r'\d+\.?\d*', score.strip())
                        score_val = float(m.group()) if m else 0.0
                    else:
                        score_val = float(score)
                except Exception:
                    score_val = 0.0
                feedback = getattr(pred, 'reasoning', '')
                return MetricResult(score=score_val, feedback=feedback)
            except Exception as e:
                msg = str(e)
                # Handle rate limit errors with wait-and-retry
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
                needs_retry = (
                    'Adapter JSONAdapter failed to parse' in msg
                    or 'ContextWindowExceededError' in msg
                    or 'response was truncated' in msg
                    or 'exceeding max_tokens' in msg
                )
                if not needs_retry:
                    raise
                retry_task_desc = (self.task_description or "") + " "
                score = _invoke(retry_task_desc)
                break

        # Normalize score
        try:
            if isinstance(score, str):
                # Handle newlines and extra text
                if '\n' in score:
                    score = score.split('\n')[0]
                # Extract number from string
                score_match = re.search(r'\d+\.?\d*', score.strip())
                if score_match:
                    score = float(score_match.group())
                else:
                    score = 0.0
            score = float(score)
        except (ValueError, TypeError):
            score = 0.0

        feedback = getattr(pred, 'reasoning', '')
        return MetricResult(score=score, feedback=feedback)

    def _call_optimized_llm_with_feedback(self, input_text: str, output_text: str, references: Optional[str] = None) -> MetricResult:
        # Delegate to unified call
        return self._call_optimized_llm(input_text, output_text, references)
        
        

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        del kwargs  # pragma: no cover
        results: List[float] = [0.0] * len(outputs)

        # Fail-fast if workers=1
        if self.max_workers == 1:
            return [self._call_optimized_llm(i, o, r).score for i, o, r in zip(inputs, outputs, references or [None] * len(outputs))]

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_optimized_llm, i, o, r): idx 
                for idx, (i, o, r) in enumerate(zip(inputs, outputs, references or [None] * len(outputs)))
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result().score
        return results

    def _calculate_with_feedback_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del kwargs  # pragma: no cover
        return self._call_optimized_llm(input, output, references)

    def _calculate_batched_with_feedback_impl(self, inputs, outputs, references=None, **kwargs):
        del kwargs  # pragma: no cover
        results: List[MetricResult] = [MetricResult(0.0, "")] * len(outputs)
        if self.max_workers == 1:
            return [self._call_optimized_llm(i, o, r) for i, o, r in zip(inputs, outputs, references or [None] * len(outputs))]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._call_optimized_llm, i, o, r): idx 
                for idx, (i, o, r) in enumerate(zip(inputs, outputs, references or [None] * len(outputs)))
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return results

    # ------------------------------------------------------------------
    # Optimized prompt inspection
    # ------------------------------------------------------------------

    def _inspect_optimized_prompt(self):
        """Inspect the optimized prompt to extract examples and instructions."""
        if self._prompt_data is not None:
            return  # Already cached
        
        try:
            if self.optimized_prompt_path and os.path.exists(self.optimized_prompt_path):
                # Load from file path
                self._prompt_data, self._prompt_examples, self._prompt_instructions = inspect_dspy_program_from_path(
                    self.optimized_prompt_path
                )
                print(f"✅ Extracted {len(self._prompt_examples)} examples from optimized prompt")
            else:
                # Fallback: empty data
                self._prompt_data = None
                self._prompt_examples = []
                self._prompt_instructions = "No optimized prompt available."
        except Exception as e:
            print(f"⚠️  Failed to inspect optimized prompt: {e}")
            self._prompt_data = None
            self._prompt_examples = []
            self._prompt_instructions = "Failed to load optimized prompt."

    def _get_optimized_examples_markdown(self, max_examples: int = 3) -> List[str]:
        """Get the optimized examples formatted as markdown table."""
        self._inspect_optimized_prompt()
        
        if not self._prompt_examples:
            return ["*No optimized examples available.*"]
        
        return format_examples_as_markdown_table(self._prompt_examples, max_examples)

    def _get_optimized_instructions(self) -> str:
        """Get the optimized prompt instructions."""
        self._inspect_optimized_prompt()
        
        return self._prompt_instructions or "No optimized instructions available."

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _generate_python_code(self, include_metric_card: bool = True) -> str:
        """Export a standalone python file that re-creates this metric."""
        class_name = "GeneratedRefBasedOptimizedJudge" if self.is_reference_based else "GeneratedRefFreeOptimizedJudge"
        
        # Copy the optimized prompt file content for embedding
        prompt_data = ""
        if self.optimized_prompt_path and os.path.exists(self.optimized_prompt_path):
            try:
                with open(self.optimized_prompt_path, 'r') as f:
                    prompt_data = f.read()
            except Exception as e:
                print(f"Warning: Could not read optimized prompt file: {e}")

        code = f"""# Auto-generated optimized metric file for {self.name}
import dspy
import os
import tempfile
import json
import atexit
from autometrics.metrics.generated.GeneratedOptimizedJudge import {class_name}
from typing import ClassVar

DEFAULT_MODEL = {generate_llm_constructor_code(self.model)}

# Embedded optimized prompt data
OPTIMIZED_PROMPT_DATA = {json.dumps(prompt_data)}

class {self.name.replace(" ", "_").replace("-", "_")}_OptimizedJudge({class_name}):
    \"\"\"{self.metric_card if include_metric_card else ""}\"\"\"

    description: ClassVar[str] = {json.dumps(self.description)}

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        # Create persistent temporary file for optimized prompt
        temp_prompt_path = None
        if OPTIMIZED_PROMPT_DATA.strip():
            try:
                print("Creating temporary file for optimized prompt...")
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
                # Parse the JSON string back to JSON data
                import json
                json_data = json.loads(OPTIMIZED_PROMPT_DATA)
                temp_file.write(json.dumps(json_data, indent=2))
                temp_file.flush()
                temp_file.close()
                temp_prompt_path = temp_file.name
                print(f"Created temporary file: {{temp_prompt_path}}")
                
                # Register cleanup function
                def cleanup_temp_file():
                    try:
                        if os.path.exists(temp_prompt_path):
                            os.unlink(temp_prompt_path)
                            print(f"Cleaned up temporary file: {{temp_prompt_path}}")
                    except Exception as e:
                        print(f"Warning: Could not clean up temporary file: {{e}}")
                        
                atexit.register(cleanup_temp_file)
                
            except Exception as e:
                print(f"Error creating temporary file for optimized prompt: {{e}}")
                temp_prompt_path = None
        else:
            print("No optimized prompt data available")
        
        super().__init__(
            name={json.dumps(self.name)},
            description={json.dumps(self.description)},
            axis={json.dumps(self.axis)},
            model=model,
            task_description={json.dumps(self.task_description)},
            optimized_prompt_path=temp_prompt_path,
            suggested_range={self.suggested_range},
            metric_card={json.dumps("provided" if include_metric_card else "None")},
            max_workers={self.max_workers},
        )

    def __repr__(self):
        return f"{self.name.replace(' ', '_').replace('-', '_')}_OptimizedJudge(model={generate_llm_constructor_code(self.model).replace("\"", "\\\"")})"

"""
        return code
    
    def _serialize(self) -> dict:
        """Serialize the metric to a dictionary for standalone Python generation."""
        # Read prompt data for serialization
        prompt_data = ""
        if self.optimized_prompt_path and os.path.exists(self.optimized_prompt_path):
            try:
                with open(self.optimized_prompt_path, 'r') as f:
                    prompt_data = f.read()
                print(f"🔧 Read prompt data ({len(prompt_data)} chars) for serialization")
            except Exception as e:
                print(f"❌ Error reading optimized prompt file: {e}")
                prompt_data = ""
        
        return {
            "name": self.name,
            "description": self.description,
            "axis": self.axis,
            "model": generate_llm_constructor_code(self.model),
            "task_description": self.task_description,
            "optimized_prompt_data": prompt_data,  # Store data instead of path
            "suggested_range": self.suggested_range,
            "metric_card": self.metric_card,
            "max_workers": self.max_workers,
            "is_reference_based": self.is_reference_based,
        }

    @classmethod
    def _deserialize(cls, data: dict):
        """Deserialize a dictionary to create a metric instance."""
        # Convert model constructor code string back to model instance
        model_code = data.pop("model")
        model = eval(model_code)
        
        # Handle optimized prompt data
        prompt_data = data.pop("optimized_prompt_data", "")
        temp_prompt_path = None
        if prompt_data.strip():
            print(f"🔍 Deserializing optimized prompt data ({len(prompt_data)} chars)")
            
            import tempfile
            import json
            
            try:
                # Basic validation - ensure it's valid JSON
                json.loads(prompt_data)
                print(f"✅ Prompt data is valid JSON")
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(prompt_data)
                    temp_prompt_path = f.name
                    
                print(f"✅ Created temporary file: {temp_prompt_path}")
                
            except json.JSONDecodeError as je:
                print(f"❌ Invalid JSON in prompt data: {je}")
                print(f"❌ First 200 chars: {prompt_data[:200]}...")
                print("❌ Will use base signature instead")
                temp_prompt_path = None
                
            except Exception as e:
                print(f"❌ Error creating temp file: {e}")
                temp_prompt_path = None
        
        data["model"] = model
        data["optimized_prompt_path"] = temp_prompt_path
        
        return cls(**data)

    # ------------------------------------------------------------------
    # Metric-card helpers
    # ------------------------------------------------------------------

    def _metric_details_template(self, *, reference_based: bool) -> str:
        """Return the *Metric Details* section for ref-free / ref-based optimized judges.

        Parameters
        ----------
        reference_based : bool
            If True, emit the reference-based variant; otherwise emit the
            reference-free variant.
        """
        kind = "reference-based" if reference_based else "reference-free"
        ref_flag = "Yes" if reference_based else "No"
        input_req = "Yes (plus reference)" if reference_based else "Yes"

        # Get optimized prompt information
        self._inspect_optimized_prompt()
        num_examples = len(self._prompt_examples) if self._prompt_examples else 0

        # --- Header & description ----------------------------------------
        lines = [
            f"**{self.name}** is a **{kind}** Optimized LLM-as-a-Judge metric that uses MIPROv2-optimized prompts to rate system outputs.",
            f"The evaluation axis is: `{self.axis}`.",
            "",
            "Optimized LLM judging differs from standard LLM-as-a-Judge by:",
            "",
            "1. **Using MIPROv2 optimization** to automatically improve prompts based on training data",
            "2. **Learning from examples** to develop better evaluation strategies",
            "3. **Optimizing demonstrations** to improve few-shot performance",
            "",
            f"This metric was optimized with {num_examples} examples and includes an optimized instruction set.",
            "",
            "### Optimized Prompt Instructions",
            "",
            f"The optimized prompt uses the following instructions:",
            "",
            f"```",
            f"{self._get_optimized_instructions()}",
            f"```",
            "",
            "### Optimized Examples",
            "",
            f"The final optimized prompt includes {num_examples} carefully selected examples:" if num_examples > 0 else "No examples were found in the optimized prompt.",
        ]

        # Add examples table if available
        if num_examples > 0:
            lines.append("")
            lines.extend(self._get_optimized_examples_markdown(max_examples=3))

        lines.extend([
            "",
            "### Evaluation Process",
            "",
            "The evaluation follows this process:",
            "",
            "1. **Task description** *d*",
            f"2. **Evaluation axis** `{self.axis}`",
            "3. **Optimized prompt instructions** (shown above)",
            "4. **Optimized examples** (if available)",
            "5. **Input text** *x*",
        ])
        if reference_based:
            lines.append("6. **Reference text** *r*")
            lines.append("7. **Output text** *y*")
        else:
            lines.append("6. **Output text** *y*")

        lines.extend([
            "",
            r"The LLM follows the optimized instructions and examples to assign scores "
            r"$\hat{s}\!\in\!\{1,2,3,4,5\}$ within the suggested range; higher = better adherence to the axis.",
            "",
            "- **Metric Type:** Optimized LLM as a Judge",
            "- **Range:** Variable (depends on suggested range, typically 1-5)",
            "- **Higher is Better?:** Yes",
            f"- **Reference-Based?:** {ref_flag}",
            f"- **Input-Required?:** {input_req}",
            "- **Optimization Method:** MIPROv2",
            "",
            "### Optimization Details",
            "",
            "This metric was optimized using **MIPROv2** (Multi-stage Instruction Proposer & Optimizer v2).",
            "The optimization process:",
            "",
            "1. **Analyzed training examples** to understand the evaluation task",
            "2. **Generated candidate prompts** with different strategies",
            "3. **Tested prompts** against ground truth scores",
            "4. **Selected best-performing** prompt configuration",
            "",
            f"- **Optimized Examples**: {num_examples}",
            f"- **Score Range**: {self.suggested_range[0]} to {self.suggested_range[1]}",
            f"- **Prompt Location**: `{self.optimized_prompt_path}`",
            "",
            "### Formal Definition",
            "",
            r"Let $f _{\\theta}$ be the LLM with optimized prompting strategy $\pi^*$ and",
        ])

        if reference_based:
            lines.append(
                r"$\pi^* _{\text{RB}}(d,\{axis\},x,r,y)$ construct the optimized "
                "prompt."
            )
        else:
            lines.append(
                r"$\pi^* _{\text{RF}}(d,\{axis\},x,y)$ construct the optimized "
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
                + (r"\pi^* _{\text{RB}}(d,\{axis\},x,r,y)"
                   if reference_based
                   else r"\pi^* _{\text{RF}}(d,\{axis\},x,y)")
                + r"\bigr)",
                "$$",
                "",
                r"The metric value is "
                + (
                    r"$\operatorname{OptLJ}^{\text{RB}}_{\{axis\}}"
                    r"(d,x,r,y)=\hat{s}$."
                    if reference_based
                    else r"$\operatorname{OptLJ}^{\text{RF}}_{\{axis\}}"
                    r"(d,x,y)=\hat{s}$."
                ),
                "",
                "### Inputs and Outputs",
                "- **Inputs:**",
                "  - **Task description** *d*",
                f"  - **Evaluation axis** `{self.axis}`",
                "  - **Optimized instructions** (embedded in prompt)",
                "  - **Optimized examples** (embedded in prompt)",
                "  - **Input text** *x*",
            ]
        )
        if reference_based:
            lines.append("  - **Reference text** *r*")
        lines.append("  - **Output text** *y*")
        lines.extend(
            [
                "- **Outputs:**",
                "  - Scalar score "
                rf"$\hat{{s}} \in [{self.suggested_range[0]}, {self.suggested_range[1]}]$",
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
            """Given the task description, and an evaluation axis, consider an Optimized LLM Judge (using MIPROv2) that is evaluating the text along this axis.  Your task is to generate the domain, a list of tasks, and a set of circumstances where the Optimized LLM Judge is best suited to be used as well as where it should not be used.  Note that you are generating the intended use for the Optimized LLM Judge, not the intended use for the task!!"""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
            model_name: str = dspy.InputField(desc="The name of the model that is being used as the LLM Judge.")
            domain: str = dspy.OutputField(desc="The domain of the task.  Some examples are: Text Generation, Code Generation, Discourse, etc.")
            tasks: List[str] = dspy.OutputField(desc="A list of tasks that the Optimized LLM Judge is best suited to be used for.  Some examples are: Travel Planning, Code Review, Machine Translation, Dialogue Response Generation, etc.")
            best_suited_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where the Optimized LLM Judge is best suited to be used.  This can describe properties of the task, data, environment, etc. that would lead to successful evaluation when using the Optimized LLM Judge on this axis. (approximately one sentence each)")
            not_recommended_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where the Optimized LLM Judge is not recommended to be used.  This can describe properties of the task, data, environment, etc. that would lead to unsuccessful evaluation when using the Optimized LLM Judge on this axis. (approximately one sentence each)")

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
  - [AutoMetrics Optimized LLM as a Judge ({ref_type})](https://github.com/XenonMolecule/autometrics/blob/main/autometrics/metrics/generated/GeneratedOptimizedJudge.py)
  - [MIPROv2 Optimization](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/mipro_optimizer_v2.py)

### Computational Complexity

- **Efficiency:**
  - Requires a single LLM call per input-output pair (same as basic LLM judge).
  - AutoMetrics does parallel calls on batched inputs.
  - **Optimization overhead** is incurred once during metric generation.

- **Scalability:**
  - Performance is linear in the number of input-output pairs.
  - Performance depends on the underlying LLM model and the dataset size.
  - Optimized prompts may achieve better quality/cost tradeoffs."""

    def generate_known_limitations(self):
        class KnownLimitationsSignature(dspy.Signature):
            """Given the task description, and an evaluation axis, consider an Optimized LLM Judge (using MIPROv2) that is evaluating the text along this axis.  Your task is to generate a list of biases, task misalignment risks, and failure cases that could be present in this evaluation.  Especially consider the axis and how it is aligned or misaligned with BOTH this task and other tasks that the Optimized LLM Judge may be used for.  Note that you are generating the known limitations for the Optimized LLM Judge, not the known limitations for the task!!"""
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
        
        # Add optimization-specific limitations
        optimization_limitations = [
            "Optimization is based on training data which may not represent all use cases or edge scenarios",
            "MIPROv2 optimization may overfit to the specific training examples and generalize poorly to different domains",
            "Optimized prompts may be sensitive to changes in the underlying LLM model or version",
        ]
        
        all_biases = outputs.biases + optimization_limitations[:1]
        all_risks = outputs.task_misalignment_risks + optimization_limitations[1:2]
        all_failures = outputs.failure_cases + optimization_limitations[2:]
        
        return f"""- **Biases:** {"\n  - " + "\n  - ".join(all_biases)}
- **Task Misalignment Risks:** {"\n  - " + "\n  - ".join(all_risks)}
- **Failure Cases:** {"\n  - " + "\n  - ".join(all_failures)}"""

    def generature_further_reading(self):
        return generate_further_reading(self) + "\n  - [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://openreview.net/pdf?id=uccHPGDlao)\n  - [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://aclanthology.org/2024.emnlp-main.525.pdf)"

    def _generate_metric_card(self, author_model: Optional[dspy.LM] = None):
        """Produce a metric card via a custom builder."""
        
        class OptimizedJudgeMetricCardBuilder(MetricCardBuilder):
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
            builder = OptimizedJudgeMetricCardBuilder(self)
            return builder.build()


class GeneratedRefFreeOptimizedJudge(_OptimizedJudgeMetricMixin, GeneratedRefFreeMetric):
    """Reference-free metric that leverages an optimized LLM (via MIPROv2) to judge outputs along a textual axis.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    axis            The textual axis/rubric used for judgement (e.g. "*Clarity*: How clear is …")
    model           A *dspy.LM* instance (or wrapper exposing .model attribute) used for judging
    task_description Optional task context passed to the judge
    optimized_prompt_path Path to the MIPROv2-optimized prompt file
    suggested_range Tuple of (min, max) score range from training data
    metric_card_author_model  LLM used to generate the metric-card (defaults to *model*)
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = False
        # Initialize cache variables first
        self._prompt_data = None
        self._prompt_examples = None
        self._prompt_instructions = None
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del references, kwargs  # pragma: no cover
        return self._call_optimized_llm(input, output).score


class GeneratedRefBasedOptimizedJudge(_OptimizedJudgeMetricMixin, GeneratedRefBasedMetric):
    """Reference-based metric that leverages an optimized LLM (via MIPROv2) to judge outputs along a textual axis using reference text.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    axis            The textual axis/rubric used for judgement (e.g. "*Clarity*: How clear is …")
    model           A *dspy.LM* instance (or wrapper exposing .model attribute) used for judging
    task_description Optional task context passed to the judge
    optimized_prompt_path Path to the MIPROv2-optimized prompt file
    suggested_range Tuple of (min, max) score range from training data
    metric_card_author_model  LLM used to generate the metric-card (defaults to *model*)
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = True
        # Initialize cache variables first
        self._prompt_data = None
        self._prompt_examples = None
        self._prompt_instructions = None
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del kwargs  # pragma: no cover
        return self._call_optimized_llm(input, output, references).score 