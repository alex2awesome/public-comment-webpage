import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any
import math
import time
from tqdm import tqdm

import dspy
from dspy.evaluate import Evaluate

from autometrics.metrics.generated.utils.utils import generate_llm_constructor_code, smart_truncate_text
from autometrics.metrics.generated.utils.metric_card import generate_further_reading
from autometrics.metrics.generated.utils.metric_card import MetricCardBuilder
from autometrics.metrics.generated.GeneratedRefFreeMetric import GeneratedRefFreeMetric
from autometrics.metrics.generated.GeneratedRefBasedMetric import GeneratedRefBasedMetric
from autometrics.metrics.Metric import MetricResult

__all__ = ["GeneratedRefFreeExampleRubricMetric", "GeneratedRefBasedExampleRubricMetric"]


# Evaluation functions
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


# DSPy signature for example-based scoring
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

    def forward(self, text, measure, suggested_range=(1,5), task_description=None, lm=None, temperature=None):
        if task_description is None:
            task_description = "None"
        suggested_range_str = f"{suggested_range[0]} to {suggested_range[1]}"

        # Compute effective LM (prefer explicit arg, fallback to global)
        effective_lm = lm or dspy.settings.lm

        ctx_kwargs = { 'temperature': temperature }
        if effective_lm is not None:
            ctx_kwargs['lm'] = effective_lm

        with dspy.settings.context(**ctx_kwargs):
            raw = self.generate_score(
                task_description=task_description,
                text=text,
                measure=measure,
                suggested_range=suggested_range_str,
                lm=effective_lm
            )
            score = raw.score
        
        # Convert the string score to a float by stripping any additional text and converting to a float
        if '\n' in score:
            score = score.split('\n')[0]
        try:
            score = float(score.strip())
        except:
            score = 0.0

        return dspy.Prediction(text=text, measure=measure, score=score, reasoning=getattr(raw, 'reasoning', ''))


def grade_row(row, axis, llm, formatter, task_description, program, suggested_range=(1,5), temperature=None):
    """Helper function to grade a single row"""
    if llm is None:
        llm = dspy.settings.lm

    with dspy.settings.context(lm=llm, temperature=temperature):
        return program(formatter(row), axis, suggested_range=suggested_range, task_description=task_description, lm=llm, temperature=temperature).score


# Base mixin for shared Example Rubric functionality
class _ExampleRubricMetricMixin:
    """Shared functionality for both reference-free and reference-based example rubric metrics."""

    DEFAULT_MAX_WORKERS = 32
    # Example-based judges use DSPy ChainOfThought and can provide reasoning
    has_feedback = True

    def __init__(
        self,
        name: str,
        description: str,
        axis: str,
        model: dspy.LM,
        task_description: Optional[str] = None,
        train_dataset: Optional[Any] = None,
        target_column: Optional[str] = None,
        suggested_range: tuple = (1, 5),
        seed: int = 42,
        optimized_examples: Optional[List] = None,  # Pre-optimized examples from Generator
        load_prompt: Optional[str] = None,
        output_prompt_path: Optional[str] = None,
        metric_card: Optional[str] = None,
        metric_card_author_model: Optional[dspy.LM] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        is_reference_based: bool = False,
        # Metadata about optimization (for metric cards) - passed from Generator
        attempts: int = 5,
        examples_per_range: int = 2,
        eval_function_name: str = 'inverse_distance',
        custom_eval_function: Optional[Any] = None,
        debug_mode: bool = False,  # Enable verbose debug output for troubleshooting
        **kwargs,
    ):
        # Store core attributes
        self.axis = axis
        self.model = model
        self.model_str = str(getattr(model, "model", model))
        self.task_description = task_description or "No task description provided"
        self.train_dataset = train_dataset
        self.target_column = target_column
        self.suggested_range = suggested_range
        self.seed = seed
        self.max_workers = max_workers
        self.is_reference_based = kwargs.get("is_reference_based", is_reference_based)
        self.debug_mode = debug_mode
        
        # Store optimization metadata for metric card generation (even though optimization was done in Generator)
        self.attempts = attempts
        self.examples_per_range = examples_per_range
        self.eval_function_name = eval_function_name
        self.optimize = False  # Should always be False for Executor
        self.optimized_examples = optimized_examples or []  # Store the original examples
        
        # Helper method for debug printing
        def debug_print(*args, **kwargs):
            if self.debug_mode:
                print(*args, **kwargs)
        self.debug_print = debug_print
        
        # Add eval_function for backward compatibility with tests
        if custom_eval_function is not None:
            self.eval_function = custom_eval_function
        elif eval_function_name == 'exact_match_rounded':
            self.eval_function = exact_match_rounded
        elif eval_function_name == 'inverse_distance':
            self.eval_function = inverse_distance
        else:
            self.eval_function = inverse_distance  # Default fallback

        if metric_card_author_model is None:
            metric_card_author_model = model if isinstance(model, dspy.LM) else None

        if metric_card == "provided":
            self.metric_card = self.__doc__
            metric_card = self.metric_card

        # Initialize the DSPy program
        self.program = LLMAsAJudge()
        # Capture reconstructable model constructor code for later export, before any cleanup
        try:
            from autometrics.metrics.generated.utils.utils import generate_llm_constructor_code as _gen_llm_code
            self._model_ctor_code = _gen_llm_code(model) if model is not None else None
        except Exception:
            self._model_ctor_code = None
        
        # Debug: Verify the program was created correctly
        self.debug_print(f"🔍 Example Judge program created: {type(self.program)}")
        self.debug_print(f"🔍 Example Judge program attributes: {[attr for attr in dir(self.program) if not attr.startswith('_')]}")
        
        if hasattr(self.program, 'generate_score'):
            self.debug_print(f"🔍 Example Judge program.generate_score: {type(self.program.generate_score)}")
            if hasattr(self.program.generate_score, 'predict'):
                self.debug_print(f"🔍 Example Judge program.generate_score.predict: {type(self.program.generate_score.predict)}")
                initial_demos = getattr(self.program.generate_score.predict, 'demos', [])
                self.debug_print(f"🔍 Example Judge initial demo count: {len(initial_demos)}")
            else:
                self.debug_print("❌ Example Judge program.generate_score missing predict attribute")
        else:
            self.debug_print("❌ Example Judge program missing generate_score attribute")

        # Load pre-optimized examples or load from prompt
        if load_prompt is not None:
            self.program.load(load_prompt)
            self.debug_print(f"Loaded program from {load_prompt}")
        elif optimized_examples:
            # Load optimized examples if provided (from Generator)
            self._load_optimized_examples(optimized_examples)
            self.debug_print(f"Loaded {len(optimized_examples)} pre-optimized examples from Generator")

        # Remove optimization-related parameters from kwargs before passing to parent
        parent_kwargs = {k: v for k, v in kwargs.items() if k not in ['attempts', 'examples_per_range', 'eval_function_name']}
        
        # Initialize parent with shared parameters
        super().__init__(
            name=name,
            description=description,
            model=model,
            task_description=task_description,
            train_dataset=train_dataset,
            target_column=target_column,
            suggested_range=suggested_range,
            seed=seed,
            load_prompt=load_prompt,
            output_prompt_path=output_prompt_path,
            metric_card=metric_card,
            metric_card_author_model=metric_card_author_model,
            max_workers=max_workers,
            is_reference_based=is_reference_based,
            **parent_kwargs,
        )

        # Exclude heavy objects from cache key
        self.exclude_from_cache_key("model", "train_dataset", "program")

    def _load_optimized_examples(self, optimized_examples):
        """Load pre-optimized examples into the DSPy program."""
        if not optimized_examples:
            self.debug_print("No optimized examples provided")
            return
            
        try:
            self.debug_print(f"Loading {len(optimized_examples)} optimized examples...")
            self.debug_print(f"Example data type: {type(optimized_examples)}")
            if optimized_examples:
                self.debug_print(f"First example type: {type(optimized_examples[0])}")
                
                # With the fixed serialization, these should all be proper DSPy Example objects
                first_example = optimized_examples[0]
                if hasattr(first_example, '_store'):
                    self.debug_print(f"✅ First example is a proper DSPy Example with _store attribute")
                    self.debug_print(f"✅ Example fields: {list(first_example._store.keys())}")
                elif hasattr(first_example, '__dict__'):
                    self.debug_print(f"✅ First example has __dict__: {first_example.__dict__.keys()}")
                else:
                    self.debug_print(f"❌ First example type not recognized: {type(first_example)}")
                    
            # Directly assign the examples to the DSPy program
            # Since they should now be properly constructed DSPy Example objects
            self.program.generate_score.predict.demos = optimized_examples
            
            self.debug_print(f"✅ Successfully loaded {len(optimized_examples)} optimized examples into DSPy program")
            self.debug_print(f"✅ Verification: {len(self.program.generate_score.predict.demos)} examples now in program.generate_score.predict.demos")
            
            if optimized_examples:
                first_demo = self.program.generate_score.predict.demos[0]
                self.debug_print(f"🔍 First demo type: {type(first_demo)}")
                if hasattr(first_demo, '_store'):
                    self.debug_print(f"🔍 First demo fields: {list(first_demo._store.keys())}")
                elif hasattr(first_demo, '__dict__'):
                    self.debug_print(f"🔍 First demo attributes: {list(first_demo.__dict__.keys())}")
                    
            self.debug_print("Loaded optimized examples from Generator")
            
        except Exception as e:
            self.debug_print(f"❌ Error loading optimized examples: {e}")
            import traceback
            traceback.print_exc()
            self.debug_print("Continuing without optimized examples")

    def _format_examples_as_markdown(self, max_examples: int = 3) -> List[str]:
        """Format the first few examples as a markdown table."""
        lines = []
        
        try:
            # Get examples from the DSPy program
            if hasattr(self.program, 'generate_score') and hasattr(self.program.generate_score, 'predict'):
                examples = self.program.generate_score.predict.demos
                if not examples:
                    return ["*No examples available.*"]
                
                # Take the first few examples
                sample_examples = examples[:max_examples]
                
                # Create markdown table header (removed Notes column)
                lines.extend([
                    "| Input Text | Score |",
                    "|------------|-------|"
                ])
                
                # Add each example as a table row
                for i, example in enumerate(sample_examples):
                    # Extract fields from the example
                    text = example.get('text', 'N/A')
                    score = example.get('score', 'N/A')
                    
                    # Allow longer text for better readability, intelligently avoiding breaking markdown links
                    text = smart_truncate_text(str(text), 400)
                    
                    # Escape pipe characters in text for markdown table
                    text = str(text).replace("|", "\\|").replace("\n", " ")
                    
                    lines.append(f"| {text} | {score} |")
                
                # Add note if there are more examples
                if len(examples) > max_examples:
                    lines.append("")
                    lines.append(f"*Showing {max_examples} of {len(examples)} total examples.*")
                
                return lines
                
        except Exception as e:
            return [f"*Could not extract examples: {e}*"]
        
        return ["*No examples available.*"]

    def _call_llm_judge(self, input_text: str, output_text: str, references: Optional[str] = None) -> MetricResult:
        """Call the LLM judge and return MetricResult(score, feedback).

        If context length is exceeded, drop the longest demo and retry. Never
        modify self.program or its demos; always work on a deepcopy.
        """
        
        input_text = str(input_text) if input_text is not None else ""
        output_text = str(output_text) if output_text is not None else ""
        if references is not None:
            if isinstance(references, list):
                references = [str(ref) if ref is not None else "" for ref in references]
            else:
                references = str(references)
        
        self.debug_print(f"🔍 Example Judge Debug - Processing:")
        self.debug_print(f"   Input: {input_text[:100]}...")
        self.debug_print(f"   Output: {output_text[:100]}...")
        self.debug_print(f"   References: {references}")
        self.debug_print(f"   Axis: {self.axis}")
        self.debug_print(f"   Program type: {type(self.program)}")
        
        # For example-based metrics, we format the full text using the dataset's formatter
        if self.train_dataset:
            # Reconstruct a row-like object for formatting
            row = {
                self.train_dataset.get_input_column(): input_text,
                self.train_dataset.get_output_column(): output_text,
            }
            
            # Add references if this is a reference-based metric
            if self.is_reference_based and references is not None:
                reference_columns = self.train_dataset.get_reference_columns()
                if reference_columns:
                    if isinstance(references, list):
                        for i, ref in enumerate(references):
                            if i < len(reference_columns):
                                row[reference_columns[i]] = ref
                    else:
                        row[reference_columns[0]] = references
            
            from autometrics.util.format import get_default_formatter
            formatter = get_default_formatter(self.train_dataset)
            formatted_text = formatter((0, row))
        else:
            # Fallback formatting
            if self.is_reference_based and references:
                formatted_text = f"Input: {input_text}\nReference: {references}\nOutput: {output_text}"
            else:
                formatted_text = f"Input: {input_text}\nOutput: {output_text}"

        self.debug_print(f"🔍 Example Judge Formatted text: {formatted_text[:200]}...")

        # Set temperature based on seed for cache busting
        temperature = 0.0001 * self.seed
        
        self.debug_print(f"🔍 Example Judge calling DSPy program with temperature: {temperature}")

        # --- Begin context length error handling logic ---
        import copy
        # Start with a shallow reference to self.program for performance
        program = self.program
        demos = []
        if hasattr(program, 'generate_score') and hasattr(program.generate_score, 'predict'):
            orig_demos = getattr(program.generate_score.predict, 'demos', [])
            demos = list(orig_demos)  # shallow copy for removal
        else:
            orig_demos = []
        deepcopied = False  # Track if we've switched to a deepcopy
        # Preserve original task description and use a local variable for cache-busting
        original_task_desc = self.task_description
        attempt_task_desc = original_task_desc
        
        attempts_left = 2  # first try + one retry
        rate_limit_retries_left = 3  # allow a few retries on rate limit
        # Compute effective LM for this call
        _eff_lm = self.model or dspy.settings.lm
        _ctx_kwargs = { 'temperature': temperature }
        if _eff_lm is not None:
            _ctx_kwargs['lm'] = _eff_lm

        with dspy.settings.context(**_ctx_kwargs):
            while True:
                try:
                    # Set the current demos for this attempt (on the current program)
                    if hasattr(program, 'generate_score') and hasattr(program.generate_score, 'predict'):
                        program.generate_score.predict.demos = demos
                        demos_count = len(demos)
                        self.debug_print(f"🔍 Example Judge program has {demos_count} demos loaded ({'deepcopy' if deepcopied else 'shallow'})")
                        if demos_count == 0:
                            self.debug_print("⚠️  Example Judge WARNING: No demos loaded - this will likely cause poor scoring!")
                    else:
                        self.debug_print(f"❌ Example Judge program missing demos attribute ({'deepcopy' if deepcopied else 'shallow'})")
                    
                    # Keep the same effective LM within nested context
                    with dspy.settings.context(**_ctx_kwargs):
                        self.debug_print(f"🔍 Example Judge DSPy context set with model: {self.model}")
                        self.debug_print(f"🔍 Example Judge current DSPy settings LM: {dspy.settings.lm}")
                        result = program(
                            text=formatted_text,
                            measure=self.axis,
                            suggested_range=self.suggested_range,
                            task_description=attempt_task_desc,
                            lm=_eff_lm,
                            temperature=temperature,
                        )
                    self.debug_print(f"🔍 Example Judge DSPy result: {result}")
                    self.debug_print(f"🔍 Example Judge result type: {type(result)}")
                    if hasattr(result, 'score'):
                        self.debug_print(f"🔍 Example Judge result.score: {result.score}")
                        self.debug_print(f"🔍 Example Judge result.score type: {type(result.score)}")
                        score_value = float(result.score)
                        self.debug_print(f"🔍 Example Judge final score: {score_value}")
                        # Check model history to verify LLM was actually called
                        if hasattr(self.model, 'history') and score_value == 0.0:
                            history_len = len(self.model.history)
                            self.debug_print(f"⚠️  Example Judge Model history length: {history_len}")
                            if history_len == 0:
                                self.debug_print("🚨 CRITICAL: Model history is empty - LLM was never called!")
                            else:
                                self.debug_print(f"🔍 Example Judge Last call: {self.model.history[-1]}")
                        feedback = getattr(result, 'reasoning', '')
                        return MetricResult(score=score_value, feedback=feedback)
                    else:
                        self.debug_print(f"❌ Example Judge result has no 'score' attribute")
                        self.debug_print(f"❌ Example Judge result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                        raise ValueError(f"DSPy result missing score attribute: {result}")
                except Exception as e:
                    error_str = str(e)
                    # Check for context length error (robust to different error types)
                    if "ContextWindowExceededError" in error_str or "context length" in error_str or "input is longer than the model's context length" in error_str:
                        self.debug_print(f"⚠️  Context length error detected: {e}")
                        if len(demos) == 0:
                            self.debug_print("🚨 No demos left to drop, cannot proceed.")
                            raise
                        # On first context error, switch to a deepcopy of the program and demos
                        if not deepcopied:
                            self.debug_print("⚠️  Switching to deepcopy of program and demos due to context error.")
                            program = copy.deepcopy(self.program)
                            if hasattr(program, 'generate_score') and hasattr(program.generate_score, 'predict'):
                                demos = list(program.generate_score.predict.demos)
                            else:
                                demos = []
                            deepcopied = True
                        # Find the longest demo (by text length)
                        def demo_length(demo):
                            # Try to get the 'text' field, fallback to str
                            if hasattr(demo, 'get'):
                                return len(str(demo.get('text', demo)))
                            elif hasattr(demo, 'text'):
                                return len(str(demo.text))
                            else:
                                return len(str(demo))
                        longest_idx = max(range(len(demos)), key=lambda i: demo_length(demos[i]))
                        self.debug_print(f"⚠️  Dropping demo #{longest_idx} (length={demo_length(demos[longest_idx])}) and retrying...")
                        demos.pop(longest_idx)
                        continue
                    # Handle rate limit errors by waiting then retrying
                    elif (
                        'RateLimitError' in error_str
                        or 'Rate limit' in error_str
                        or 'rate limit' in error_str
                        or '429' in error_str
                        or 'Too Many Requests' in error_str
                        or 'rate_limit_exceeded' in error_str
                        or 'quota' in error_str
                        or 'exceeded your current quota' in error_str
                    ) and rate_limit_retries_left > 0:
                        # Try to parse suggested wait time like: "Please try again in 27.875s" or "Please try again in 27875ms"
                        wait_seconds = 30.0
                        try:
                            import re as _re
                            m = _re.search(r"Please try again in\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds|s|sec|seconds)\b", error_str, _re.IGNORECASE)
                            if m:
                                _val = float(m.group(1))
                                _unit = m.group(2).lower()
                                wait_seconds = _val / 1000.0 if _unit.startswith('m') else _val
                            else:
                                m = _re.search(r"Retry-After\s*:?\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds|s|sec|seconds)?", error_str, _re.IGNORECASE)
                                if m:
                                    _val = float(m.group(1))
                                    _unit = (m.group(2) or 's').lower()
                                    wait_seconds = _val / 1000.0 if _unit.startswith('m') else _val
                        except Exception:
                            pass
                        # Cap to maximum of 30 seconds
                        wait_seconds = max(0.0, min(wait_seconds, 30.0))
                        rate_limit_retries_left -= 1
                        self.debug_print(f"⏳ Rate limit hit. Waiting {wait_seconds:.2f}s before retrying... ({3 - rate_limit_retries_left}/3)")
                        time.sleep(wait_seconds)
                        continue
                    # Handle truncation/adapter parse failures by retrying once with a cache-bust
                    elif (
                        'Adapter JSONAdapter failed to parse' in error_str
                        or 'response was truncated' in error_str
                        or 'exceeding max_tokens' in error_str
                    ) and attempts_left > 1:
                        attempts_left -= 1
                        self.debug_print("⚠️  Truncation/parse issue detected. Retrying once with cache-busting space in task_description...")
                        # Cache-bust by appending a space to task_description for this call
                        import copy as _copy
                        program = _copy.deepcopy(program) if not deepcopied else program
                        deepcopied = True
                        if hasattr(program, 'generate_score') and hasattr(program.generate_score, 'predict'):
                            program.generate_score.predict.demos = demos
                        # Temporarily modify local attempt_task_desc (do not mutate self)
                        attempt_task_desc = (original_task_desc or "") + " "
                        continue
                    else:
                        self.debug_print(f"🚨 CRITICAL EXCEPTION in Example Judge DSPy call:")
                        self.debug_print(f"   Exception: {e}")
                        self.debug_print(f"   Exception Type: {type(e).__name__}")
                        self.debug_print(f"   Program: {program}")
                        self.debug_print(f"   Model: {self.model}")
                        import traceback
                        traceback.print_exc()
                        raise

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        del kwargs  # pragma: no cover
        results: List[float] = [0.0] * len(outputs)

        # Fail-fast if workers=1
        if self.max_workers == 1:
            for i, (inp, out, ref) in enumerate(zip(inputs, outputs, references or [None] * len(outputs))):
                # Return only the numeric score on legacy path
                results[i] = self._call_llm_judge(inp, out, ref).score
            return results

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._call_llm_judge, i, o, r): idx 
                for idx, (i, o, r) in enumerate(zip(inputs, outputs, references or [None] * len(outputs)))
            }
            
            # Collect results with progress bar
            with tqdm(total=len(futures), desc="Processing Example Rubric Evaluation") as pbar:
                for future in as_completed(futures):
                    index = futures[future]
                    # Map MetricResult to score for legacy path
                    results[index] = future.result().score
                    pbar.update(1)
        
        return results

    def _calculate_with_feedback_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del kwargs  # pragma: no cover
        return self._call_llm_judge(input, output, references)

    def _calculate_batched_with_feedback_impl(self, inputs, outputs, references=None, **kwargs):
        del kwargs  # pragma: no cover
        results: List[MetricResult] = [MetricResult(0.0, "")] * len(outputs)
        if self.max_workers == 1:
            return [self._call_llm_judge(i, o, r) for i, o, r in zip(inputs, outputs, references or [None] * len(outputs))]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._call_llm_judge, i, o, r): idx 
                for idx, (i, o, r) in enumerate(zip(inputs, outputs, references or [None] * len(outputs)))
            }
            with tqdm(total=len(futures), desc="Processing Example Rubric Evaluation") as pbar:
                for future in as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                    pbar.update(1)
        
        return results

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _generate_python_code(self, include_metric_card: bool = True) -> str:
        """Export a standalone python file that re-creates this metric."""
        class_name = "GeneratedRefBasedExampleRubricMetric" if self.is_reference_based else "GeneratedRefFreeExampleRubricMetric"
        
        # CRITICAL FIX: Better example extraction with debugging
        examples_data = []
        print(f"🔍 DEBUG: Extracting examples for serialization...")
        print(f"🔍 DEBUG: Program type: {type(self.program)}")
        
        try:
            # Try multiple paths to find examples
            if hasattr(self.program, 'generate_score'):
                print(f"🔍 DEBUG: Found generate_score: {type(self.program.generate_score)}")
                if hasattr(self.program.generate_score, 'predict'):
                    print(f"🔍 DEBUG: Found predict: {type(self.program.generate_score.predict)}")
                    if hasattr(self.program.generate_score.predict, 'demos'):
                        examples_data = self.program.generate_score.predict.demos
                        print(f"🔍 DEBUG: Found demos: {len(examples_data)} examples")
                    else:
                        print(f"🔍 DEBUG: No demos attribute in predict")
                else:
                    print(f"🔍 DEBUG: No predict attribute in generate_score")
            else:
                print(f"🔍 DEBUG: No generate_score attribute in program")
                
        except Exception as e:
            print(f"🔍 DEBUG: Exception during example extraction: {e}")
        
        # Convert examples to proper serializable format
        serialized_examples = []
        if examples_data:
            print(f"🔍 DEBUG: Serializing {len(examples_data)} examples...")
            
            for i, example in enumerate(examples_data):
                try:
                    # CRITICAL FIX: Handle different example types
                    example_dict = None
                    input_keys = []
                    
                    # Preferred: DSPy Example object stores fields in _store
                    if hasattr(example, '_store') and isinstance(getattr(example, '_store', None), dict):
                        print(f"🔍 DEBUG: Example {i} has _store; extracting fields")
                        example_dict = dict(example._store)
                        # Known input fields for this signature
                        preferred_inputs = ["task_description", "text", "measure", "suggested_range"]
                        input_keys = [k for k in preferred_inputs if k in example_dict]
                        if not input_keys:
                            # Fallback to any non-output keys
                            input_keys = [k for k in example_dict.keys() if k != 'score']
                    elif hasattr(example, '__dict__'):
                        # DSPy Example object (may have limited public attrs)
                        print(f"🔍 DEBUG: Example {i} is DSPy Example object (no _store)")
                        example_dict = {k: v for k, v in example.__dict__.items() if not k.startswith('_')}
                        if hasattr(example, '_input_keys'):
                            input_keys = list(example._input_keys)
                        elif hasattr(example, 'inputs'):
                            try:
                                input_keys = list(example.inputs())
                            except Exception:
                                input_keys = [k for k in example_dict.keys() if k != 'score']
                        else:
                            input_keys = [k for k in example_dict.keys() if k != 'score']
                    elif isinstance(example, dict):
                        # Plain dictionary
                        print(f"🔍 DEBUG: Example {i} is dictionary")
                        example_dict = dict(example)
                        input_keys = [k for k in example_dict.keys() if k != 'score']
                    else:
                        # Try to convert to dict if possible
                        print(f"🔍 DEBUG: Example {i} is {type(example)}, trying to convert")
                        try:
                            example_dict = dict(example)
                            input_keys = [k for k in example_dict.keys() if k != 'score']
                            print(f"🔍 DEBUG: Successfully converted example {i} to dict")
                        except Exception as convert_e:
                            print(f"🔍 DEBUG: Failed to convert example {i} to dict: {convert_e}")
                            continue
                    
                    if example_dict is not None:
                        # Coerce non-serializable values to strings for safety
                        clean_dict = {}
                        for k, v in example_dict.items():
                            if isinstance(v, (str, int, float, bool)) or v is None:
                                clean_dict[k] = v
                            else:
                                try:
                                    clean_dict[k] = str(v)
                                except Exception:
                                    clean_dict[k] = None
                        # Ensure output field exists (score)
                        if 'score' not in clean_dict:
                            # Leave absent rather than fabricate
                            pass
                        # Determine inputs
                        if not input_keys:
                            input_keys = [k for k in clean_dict.keys() if k != 'score']
                        input_keys_args = ', '.join(f'"{key}"' for key in input_keys)
                        # Use a Python literal via repr for safe embedding
                        py_literal = repr(clean_dict)
                        serialized_example = (
                            f"dspy.Example({py_literal})"
                            f".with_inputs({input_keys_args})"
                        )
                        serialized_examples.append(serialized_example)
                        print(f"🔍 DEBUG: Serialized example {i}")
                    else:
                        print(f"🔍 DEBUG: Could not extract data from example {i}")
                        
                except Exception as e:
                    print(f"🔍 DEBUG: Failed to serialize example {i}: {e}")
                    continue
        else:
            print(f"🔍 DEBUG: No examples found for serialization!")
        
        # Format examples for the Python file
        examples_list_str = "[]"
        if serialized_examples:
            print(f"🔍 DEBUG: Formatting {len(serialized_examples)} serialized examples")
            # Multi-line format for readability
            examples_list_str = "[\n    " + ",\n    ".join(serialized_examples) + "\n]"
        else:
            print(f"🔍 DEBUG: Using empty examples list")
        
        # Generate the code
        code = f'''# Auto-generated Example Rubric metric file for {self.name}
import dspy
from autometrics.metrics.generated.GeneratedExampleRubric import {class_name}
from typing import ClassVar
import os

DEFAULT_MODEL = {generate_llm_constructor_code(self.model)}

# Optimized examples data (properly serialized DSPy Examples)
OPTIMIZED_EXAMPLES = {examples_list_str}

class {self.name.replace(" ", "_").replace("-", "_")}_ExampleRubric({class_name}):
    \"\"\"{self.metric_card.replace('"""', '\\"\\"\\"') if include_metric_card else ""}\"\"\"

    description: ClassVar[str] = {json.dumps(self.description)}

    def __init__(self, model: dspy.LM = DEFAULT_MODEL):
        super().__init__(
            name={json.dumps(self.name)},
            description={json.dumps(self.description)},
            axis={json.dumps(self.axis)},
            model=model,
            task_description={json.dumps(self.task_description)},
            suggested_range={self.suggested_range},
            seed={self.seed},
            optimized_examples=OPTIMIZED_EXAMPLES,
            metric_card={json.dumps("provided" if include_metric_card else "None")},
            max_workers={self.max_workers},
            is_reference_based={self.is_reference_based},
        )

    def __repr__(self):
        return f"{self.name.replace(' ', '_').replace('-', '_')}_ExampleRubric(model={generate_llm_constructor_code(self.model).replace('\"', '\\\\')})"

'''
        return code

    def _serialize(self) -> dict:
        """Serialize the metric to a dictionary for in-memory operations."""
        # Extract examples if available
        examples_data = []
        try:
            # Use the correct DSPy path we discovered through testing
            if hasattr(self.program, 'generate_score') and hasattr(self.program.generate_score, 'predict'):
                examples_data = self.program.generate_score.predict.demos
        except Exception as e:
            print(f"Warning: Could not extract examples during serialization: {e}")
        
        return {
            "name": self.name,
            "description": self.description,
            "axis": self.axis,
            "model": generate_llm_constructor_code(self.model),
            "task_description": self.task_description,
            "target_column": self.target_column,
            "suggested_range": self.suggested_range,
            "seed": self.seed,
            "optimized_examples": self.optimized_examples,  # Store the original examples
            "examples_data": examples_data,  # Store extracted examples for reference
            "metric_card": self.metric_card,
            "max_workers": self.max_workers,
            "is_reference_based": self.is_reference_based,
            "debug_mode": self.debug_mode,
            # Optimization metadata
            "attempts": self.attempts,
            "examples_per_range": self.examples_per_range,
            "eval_function_name": self.eval_function_name,
        }

    @classmethod
    def _deserialize(cls, data: dict):
        """Deserialize a dictionary to create a metric instance."""
        # Convert model constructor code string back to model instance
        model_code = data.pop("model")
        model = eval(model_code)
        data["model"] = model
        
        # Extract examples data
        examples_data = data.pop("examples_data", [])
        # Note: We can't use debug_print here since the instance isn't created yet
        # We'll use regular print for deserialization messages
        
        # Create the instance
        instance = cls(**data)
        
        # Load the examples if available
        if examples_data:
            try:
                instance.debug_print(f"Loading {len(examples_data)} examples during deserialization...")
                instance.debug_print(f"Examples data type: {type(examples_data)}")
                
                # Ensure we have a program with the right structure
                if not hasattr(instance.program, 'generate_score'):
                    instance.debug_print("Warning: deserialized program does not have generate_score attribute")
                    return instance
                    
                if not hasattr(instance.program.generate_score, 'predict'):
                    instance.debug_print("Warning: deserialized program.generate_score does not have predict attribute")
                    return instance
                
                # Use the correct DSPy path we discovered through testing
                instance.program.generate_score.predict.demos = examples_data
                instance.debug_print(f"✅ Successfully loaded {len(examples_data)} examples during deserialization")
                
                # Verify the examples were loaded
                loaded_demos = getattr(instance.program.generate_score.predict, 'demos', [])
                instance.debug_print(f"✅ Verification: {len(loaded_demos)} examples now in deserialized program")
                
            except Exception as e:
                instance.debug_print(f"❌ Error: Could not load examples during deserialization: {e}")
                instance.debug_print(f"❌ Exception type: {type(e)}")
                import traceback
                traceback.print_exc()
        else:
            instance.debug_print("No examples_data to load during deserialization")
        
        return instance
    
    # ------------------------------------------------------------------
    # Metric-card helpers
    # ------------------------------------------------------------------

    def _metric_details_template(self, *, reference_based: bool) -> str:
        """Return the *Metric Details* section for ref-free / ref-based example rubrics."""
        kind = "reference-based" if reference_based else "reference-free"
        ref_flag = "Yes" if reference_based else "No"
        input_req = "Yes (plus reference)" if reference_based else "Yes"

        # Count examples if available
        num_examples = 0
        try:
            # Use the correct DSPy path we discovered through testing
            if hasattr(self.program, 'generate_score') and hasattr(self.program.generate_score, 'predict'):
                num_examples = len(self.program.generate_score.predict.demos)
        except Exception:
            num_examples = 0

        lines = [
            f"**{self.name}** is a **{kind}** example-based LLM-as-a-Judge metric that uses optimized few-shot examples to evaluate system outputs.",
            f"The evaluation axis is: `{self.axis}`.",
            "",
            "Example-based LLM judging differs from standard LLM-as-a-Judge by:",
            "",
            "1. **Example Selection**: Uses quintile-based bucketing to select diverse examples across score ranges",
            "2. **Few-Shot Optimization**: Optimizes example selection through multiple attempts and evaluation",
            "3. **Consistent Scoring**: Examples provide concrete scoring patterns for the LLM to follow",
            "",
            f"This metric was optimized using {self.attempts} attempts with {self.examples_per_range} examples per score range.",
            "",
            "### Optimized Examples",
            "",
            f"The final optimized prompt includes {num_examples} carefully selected examples." if num_examples > 0 else "No examples were loaded for this metric.",
        ]
        
        # Add examples table if available
        if num_examples > 0:
            lines.append("")
            lines.extend(self._format_examples_as_markdown(max_examples=3))
        
        lines.extend([
            "",
            "### Evaluation Process",
            "",
            "The evaluation follows this process:",
            "",
            "1. **Task description** *d*",
            f"2. **Evaluation axis** `{self.axis}`",
            "3. **Optimized examples** showing score patterns",
            "4. **Input text** *x*",
        ])
        if reference_based:
            lines.append("5. **Reference text** *r*")
            lines.append("6. **Output text** *y*")
        else:
            lines.append("5. **Output text** *y*")

        lines.extend([
            "",
            r"The LLM follows the example patterns to assign scores "
            r"$\hat{s}\!\in\!\{1,2,3,4,5\}$ within the suggested range; higher = better adherence to the axis.",
            "",
            "- **Metric Type:** Example-based LLM as a Judge",
            "- **Range:** Variable (depends on suggested range, typically 1-5)",
            "- **Higher is Better?:** Yes",
            f"- **Reference-Based?:** {ref_flag}",
            f"- **Input-Required?:** {input_req}",
            "",
            "### Example Optimization Details",
            "",
            f"- **Optimization Attempts**: {self.attempts}",
            f"- **Examples per Score Range**: {self.examples_per_range}",
            f"- **Evaluation Function**: {self.eval_function_name}",
            f"- **Score Range**: {self.suggested_range[0]} to {self.suggested_range[1]}",
            f"- **Random Seed**: {self.seed} (for reproducible example selection)",
            "",
            "### Inputs and Outputs",
            "- **Inputs:**",
            "  - **Task description** *d*",
            f"  - **Evaluation axis** `{self.axis}`",
            "  - **Optimized examples** (embedded in prompt)",
            "  - **Input text** *x*",
        ])
        if reference_based:
            lines.append("  - **Reference text** *r*")
        lines.append("  - **Output text** *y*")
        lines.extend([
            "- **Outputs:**",
            f"  - Scalar score within range {self.suggested_range[0]}-{self.suggested_range[1]}",
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
            """Given the task description, evaluation axis, and example-based optimization details, consider an example-based LLM Judge. Generate the domain, tasks, and circumstances where this optimized example-based evaluation is best suited."""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
            model_name: str = dspy.InputField(desc="The name of the model that is being used as the LLM Judge.")
            optimization_details: str = dspy.InputField(desc="Details about the example optimization process.")
            domain: str = dspy.OutputField(desc="The domain of the task. Some examples are: Text Generation, Code Generation, Discourse, etc.")
            tasks: List[str] = dspy.OutputField(desc="A list of tasks that example-based LLM Judge is best suited for.")
            best_suited_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where example-based LLM Judge is best suited to be used.")
            not_recommended_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where example-based LLM Judge is not recommended.")

        optimization_details = f"Optimized with {self.attempts} attempts, {self.examples_per_range} examples per range, using {self.eval_function_name} evaluation function"

        with dspy.settings.context(lm=self.metric_card_author_model):
            outputs = dspy.ChainOfThought(IntendedUseSignature)(
                task_description=self.task_description,
                axis=self.axis,
                model_name=str(getattr(self.model, "model", self.model)),
                optimization_details=optimization_details,
            )
        
        return f"""- **Domain:** {outputs.domain}
- **Tasks:** {"\n  - " + "\n  - ".join(outputs.tasks)}
- **Best Suited For:** {"\n  - " + "\n  - ".join(outputs.best_suited_for_circumstances)}
- **Not Recommended For:** {"\n  - " + "\n  - ".join(outputs.not_recommended_for_circumstances)}"""

    def generate_metric_implementation(self):
        ref_type = "reference-based" if self.is_reference_based else "reference-free"
        return f"""### Reference Implementations

- **Libraries/Packages:**
  - [AutoMetrics Example-based LLM Judge ({ref_type})](https://github.com/XenonMolecule/autometrics/blob/main/autometrics/metrics/generated/GeneratedExampleRubric.py)
  - [DSPy Few-shot Optimization](https://dspy-docs.vercel.app/)

### Computational Complexity

- **Efficiency:**
  - Requires a single LLM call per input-output pair (same as basic LLM judge).
  - AutoMetrics does parallel calls on batched inputs.
  - One-time optimization cost during metric creation.

- **Scalability:**
  - Performance is linear in the number of input-output pairs.
  - Performance depends on the underlying LLM model and the dataset size.
  - Example optimization improves consistency but doesn't affect runtime complexity."""

    def generate_known_limitations(self):
        class KnownLimitationsSignature(dspy.Signature):
            """Given the task description, evaluation axis, and example-based optimization details, consider an example-based LLM Judge. Generate biases, task misalignment risks, and failure cases."""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            axis: str = dspy.InputField(desc="The evaluation axis / rubric.")
            model_name: str = dspy.InputField(desc="The name of the model that is being used as the LLM Judge.")
            optimization_details: str = dspy.InputField(desc="Details about the example optimization process.")
            biases: List[str] = dspy.OutputField(desc="A list of biases that could be present in this evaluation.")
            task_misalignment_risks: List[str] = dspy.OutputField(desc="A list of ways in which this evaluation could be misaligned with the task.")
            failure_cases: List[str] = dspy.OutputField(desc="A list of failure cases that could occur in this evaluation.")

        optimization_details = f"Optimized with {self.attempts} attempts, {self.examples_per_range} examples per range, using {self.eval_function_name} evaluation function"

        with dspy.settings.context(lm=self.metric_card_author_model):
            outputs = dspy.ChainOfThought(KnownLimitationsSignature)(
                task_description=self.task_description,
                axis=self.axis,
                model_name=str(getattr(self.model, "model", self.model)),
                optimization_details=optimization_details,
            )
        
        return f"""- **Biases:** {"\n  - " + "\n  - ".join(outputs.biases)}
- **Task Misalignment Risks:** {"\n  - " + "\n  - ".join(outputs.task_misalignment_risks)}
- **Failure Cases:** {"\n  - " + "\n  - ".join(outputs.failure_cases)}"""

    def generature_further_reading(self):
        return generate_further_reading(self) + "\n  - [Few-Shot Learning with DSPy](https://dspy-docs.vercel.app/docs/building-blocks/optimizers)\n  - [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://openreview.net/pdf?id=uccHPGDlao)"

    def _generate_metric_card(self, author_model: Optional[dspy.LM] = None):
        """Produce a metric card via a custom builder."""
        
        class ExampleRubricMetricCardBuilder(MetricCardBuilder):
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
            builder = ExampleRubricMetricCardBuilder(self)
            return builder.build()


class GeneratedRefFreeExampleRubricMetric(_ExampleRubricMetricMixin, GeneratedRefFreeMetric):
    """Reference-free metric that leverages example-based LLM judging with optimized few-shot examples.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    axis            The textual axis/rubric used for judgement (e.g. "*Clarity*: How clear is …")
    model           A *dspy.LM* instance used for judging
    task_description Optional task context passed to the judge
    train_dataset   Training dataset used for example selection
    target_column   Column name containing target scores for optimization
    train_buckets   Pre-computed quintile buckets of examples (optional)
    trainset        Pre-computed DSPy training set (optional)
    suggested_range Tuple of (min, max) suggested score range
    attempts        Number of optimization attempts (default: 5)
    examples_per_range Number of examples to select from each quintile (default: 2)
    seed            Random seed for reproducible example selection (default: 42)
    eval_function_name Name of evaluation function ('inverse_distance' or 'exact_match_rounded')
    custom_eval_function Custom evaluation function (optional)
    load_prompt     Path to load pre-optimized prompt (optional)
    optimize        Whether to perform example optimization (default: True)
    output_prompt_path Path to save optimized prompt (optional)
    metric_card_author_model LLM used to generate the metric-card (defaults to *model*)
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = False
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del references, kwargs  # pragma: no cover
        return self._call_llm_judge(input, output).score


class GeneratedRefBasedExampleRubricMetric(_ExampleRubricMetricMixin, GeneratedRefBasedMetric):
    """Reference-based metric that leverages example-based LLM judging with optimized few-shot examples using reference text.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    axis            The textual axis/rubric used for judgement (e.g. "*Clarity*: How clear is …")
    model           A *dspy.LM* instance used for judging
    task_description Optional task context passed to the judge
    train_dataset   Training dataset used for example selection
    target_column   Column name containing target scores for optimization
    train_buckets   Pre-computed quintile buckets of examples (optional)
    trainset        Pre-computed DSPy training set (optional)
    suggested_range Tuple of (min, max) suggested score range
    attempts        Number of optimization attempts (default: 5)
    examples_per_range Number of examples to select from each quintile (default: 2)
    seed            Random seed for reproducible example selection (default: 42)
    eval_function_name Name of evaluation function ('inverse_distance' or 'exact_match_rounded')
    custom_eval_function Custom evaluation function (optional)
    load_prompt     Path to load pre-optimized prompt (optional)
    optimize        Whether to perform example optimization (default: True)
    output_prompt_path Path to save optimized prompt (optional)
    metric_card_author_model LLM used to generate the metric-card (defaults to *model*)
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = True
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del kwargs  # pragma: no cover
        return self._call_llm_judge(input, output, references).score 