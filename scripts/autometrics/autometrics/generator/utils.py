from typing import Callable, List, Mapping, Optional, Sequence
import re
import dspy
import litellm
import math
import logging

logger = logging.getLogger(__name__)

# Reusable helper -----------------------------------------------------------

def get_good_bad_examples(df, target_column: str, num_examples: int = 5, flip: bool = False):
    """Return two dataframes: examples with *highest* target values and *lowest* target values.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the examples.
    target_column : str
        Column in *df* that contains the numeric quality signal we want to sort by.
    num_examples : int, optional
        How many good / bad examples to return, by default 5.
    flip : bool, optional
        If *True* the meaning of good/bad is flipped (good becomes lowest values), by default False.
    """
    good_examples = df.sort_values(by=target_column, ascending=False).head(num_examples)
    bad_examples = df.sort_values(by=target_column, ascending=True).head(num_examples)

    if flip:
        return bad_examples, good_examples

    return good_examples, bad_examples


def truncate_examples_if_needed(examples: List[str], max_length_per_example: int = 2000) -> List[str]:
    """Truncate examples if they're too long to help with context length issues."""
    truncated = []
    for example in examples:
        if len(example) > max_length_per_example:
            truncated.append(example[:max_length_per_example] + "...")
        else:
            truncated.append(example)
    return truncated


def is_context_length_error(error_str: str) -> bool:
    """Check if an error indicates context length exceeded."""
    context_error_indicators = [
        "context length",
        "maximum context length",
        "context window",
        "too many tokens",
        "token limit",
        "exceeds maximum",
        "exceeds the maximum",
        "maximum allowed length",
        "input length",
        "allow-auto-truncate",
        "input is too long",
        "context_length_exceeded"
    ]
    error_lower = error_str.lower()
    return any(indicator in error_lower for indicator in context_error_indicators)


def count_tokens_with_litellm(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using litellm.token_counter with fallback."""
    try:
        return litellm.token_counter(model=model_name, text=text)
    except Exception as e:
        print(f"Warning: litellm.token_counter failed with {e}, using fallback estimation")
        # Fallback: rough estimation (1.3 tokens per word on average)
        return int(len(text.split()) * 1.3)


def get_max_context_tokens(model_name: str = "gpt-3.5-turbo") -> int:
    """Get max context window for model, preferring input tokens over general max."""
    try:
        # First try to get detailed model cost info which may have input/output breakdown
        model_cost_info = litellm.model_cost.get(model_name)
        if model_cost_info:
            # Prefer max_input_tokens if available (for models like GPT-4o-mini)
            if 'max_input_tokens' in model_cost_info:
                max_input = model_cost_info['max_input_tokens']
                print(f"Using max_input_tokens for {model_name}: {max_input}")
                return max_input
            elif 'max_tokens' in model_cost_info:
                max_general = model_cost_info['max_tokens']
                print(f"Using max_tokens for {model_name}: {max_general}")
                return max_general
        
        # Fall back to litellm.get_max_tokens
        max_tokens = litellm.get_max_tokens(model=model_name)
        if max_tokens is not None:
            return max_tokens
            
        # Default to Qwen3's context length (40960) as a reasonable modern default
        print(f"Warning: Could not get max tokens for {model_name}, using default 40960 (Qwen3 context)")
        return 40960
        
    except Exception as e:
        print(f"Warning: litellm token limit lookup failed with {e}, using default 40960")
        return 40960


def estimate_dspy_prompt_tokens(
    task_description: str,
    examples: List[str],
    model_name: str = "gpt-3.5-turbo",
    dspy_overhead_tokens: int = 4096,
    output_tokens: int = 2048,
    safety_margin: int = 2000
) -> dict:
    """
    Estimate total tokens for a DSPy prompt including examples.
    
    Returns:
        dict with token estimates and recommendations
    """
    # Count base prompt tokens
    base_prompt = f"Task: {task_description}\n\nExamples:\n"
    base_tokens = count_tokens_with_litellm(base_prompt, model_name)
    
    # Count example tokens
    example_tokens = 0
    for example in examples:
        example_tokens += count_tokens_with_litellm(example, model_name)
    
    # Add DSPy formatting overhead (30% for DSPy structure)
    total_example_tokens = int(example_tokens * 1.3)
    
    # Total estimated tokens
    total_estimated = base_tokens + total_example_tokens
    
    # Get model limits
    max_context = get_max_context_tokens(model_name)
    available_tokens = max_context - dspy_overhead_tokens - output_tokens - safety_margin
    
    # Calculate how many examples we can fit
    tokens_per_example = total_example_tokens / len(examples) if examples else 0
    max_examples = int((available_tokens - base_tokens) / tokens_per_example) if tokens_per_example > 0 else 0
    
    return {
        'base_tokens': base_tokens,
        'example_tokens': total_example_tokens,
        'total_estimated': total_estimated,
        'max_context': max_context,
        'available_tokens': available_tokens,
        'tokens_per_example': tokens_per_example,
        'max_examples': max_examples,
        'fits_in_context': total_estimated <= available_tokens,
        'model_name': model_name
    }


def smart_limit_examples_for_context(
    examples: List[str],
    task_description: str,
    model_name: str = "gpt-3.5-turbo",
    target_examples: int = 8,
    dspy_overhead_tokens: int = 4096,
    output_tokens: int = 2048,
    safety_margin: int = 2000
) -> List[str]:
    """
    Intelligently limit examples to fit within context window.
    
    Strategy:
    1. Try target number of examples
    2. If too long, try fewer examples
    3. If still too long, truncate examples
    4. If still too long, return minimal set
    """
    print(f"Smart limiting examples for {model_name}: target={target_examples}, total={len(examples)}")
    
    # Estimate tokens for target number of examples
    target_examples_list = examples[:target_examples]
    estimate = estimate_dspy_prompt_tokens(
        task_description, target_examples_list, model_name,
        dspy_overhead_tokens, output_tokens, safety_margin
    )
    
    print(f"Token estimate: base={estimate['base_tokens']}, examples={estimate['example_tokens']}, "
          f"total={estimate['total_estimated']}, available={estimate['available_tokens']}")
    
    if estimate['fits_in_context']:
        print(f"✅ Target {target_examples} examples fit in context")
        return target_examples_list
    
    # Try fewer examples
    for num_examples in range(target_examples, 1, -2):
        if num_examples > len(examples):
            continue
            
        test_examples = examples[:num_examples]
        estimate = estimate_dspy_prompt_tokens(
            task_description, test_examples, model_name,
            dspy_overhead_tokens, output_tokens, safety_margin
        )
        
        if estimate['fits_in_context']:
            print(f"✅ {num_examples} examples fit in context")
            return test_examples
    
    # Try truncated examples
    for num_examples in [2, 1]:
        if num_examples > len(examples):
            continue
            
        test_examples = truncate_examples_if_needed(examples[:num_examples], 1500)
        estimate = estimate_dspy_prompt_tokens(
            task_description, test_examples, model_name,
            dspy_overhead_tokens, output_tokens, safety_margin
        )
        
        if estimate['fits_in_context']:
            print(f"✅ {num_examples} truncated examples fit in context")
            return test_examples
    
    # Last resort: minimal examples
    minimal_examples = truncate_examples_if_needed(examples[:1], 1000) if examples else []
    print(f"⚠️ Using minimal examples: {len(minimal_examples)}")
    return minimal_examples


def _extract_axis_display_name(axis: str) -> str:
    """
    Pull a human-readable title from an axis string that may include bold markers.
    """
    name_part = axis.split(":")[0].strip()
    if "**" in name_part:
        candidate = name_part.split("**")[1]
    elif "*" in name_part:
        candidate = name_part.split("*")[1]
    else:
        candidate = name_part
    candidate = candidate.replace("*", "").strip()
    if len(candidate) > 80:
        logger.warning(f"Axis {axis} is too long and we weren't able to truncate: {candidate}. I'm hard-truncating it now to 80 characters")
        candidate = candidate[:80]
    return candidate


def build_metric_name_from_axis(
    axis: str,
    suffix: str = "",
    *,
    max_length: Optional[int] = None,
) -> str:
    """
    Generate a sanitized metric name from an axis description and optional suffix.
    """
    base = _extract_axis_display_name(axis)
    combined = f"{base}{suffix}"
    combined = combined.replace(",", " ")
    sanitized = re.sub(r"[/:.\s(){}\[\]]+", "_", combined)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "Metric"
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip("_")
    return sanitized


def format_prior_metrics_summary(prior_metrics: Optional[Sequence[Mapping[str, str]]]) -> str:
    """
    Convert prior metric dicts into a prompt-ready summary, including rejected metrics.
    """
    if not prior_metrics:
        return (
            "Active metrics currently influencing residuals:\n"
            "- <none>. This iteration is starting from scratch."
        )

    active_lines: List[str] = []
    rejected_lines: List[str] = []
    for metric in prior_metrics:
        name = metric.get("name", "Unnamed Metric")
        description = metric.get("description") or "No description supplied."
        status = (metric.get("status") or "active").lower()
        line = f"- {name}: {description}"
        if status == "rejected":
            rejected_lines.append(
                f"{line} (rejected earlier; not used when generating the current residuals)"
            )
        else:
            active_lines.append(line)

    sections: List[str] = []
    if active_lines:
        sections.append(
            "Active metrics currently influencing residuals:\n" + "\n".join(active_lines)
        )
    else:
        sections.append(
            "Active metrics currently influencing residuals:\n"
            "- <none>. Residuals above were produced without any accepted metrics."
        )

    if rejected_lines:
        sections.append(
            "Previously rejected metrics (excluded from scoring so far):\n"
            + "\n".join(rejected_lines)
        )

    return "\n\n".join(sections)

# ---------------------------------------------------------------------------
# DSPy module for generating axes of variation
# ---------------------------------------------------------------------------

class GenerateAxisOfVariationSignature(dspy.Signature):
    """Given a task description, a target metric, and good/bad examples, generate a list of axes of variation which could be used to explain the differences between the good and bad examples.  These axes of variation will be used as measures to evaluate the model's performance, so they should be informative and useful for the model to improve on."""

    task_description: str = dspy.InputField(desc="A description of the overall task the model is trying to solve.")
    target_name: Optional[str] = dspy.InputField(desc="Optional hint of the target metric/column we care about. Could be 'None' or something generic like 'quality' or 'score'.")
    good_examples: List[str] = dspy.InputField(desc="A list of examples with *high* quality according to the target metric.")
    bad_examples: List[str] = dspy.InputField(desc="A list of examples with *low* quality according to the target metric.")
    num_axes_to_generate: int = dspy.InputField(desc="The number of axes of variation to generate.")
    prior_metrics_context: str = dspy.InputField(desc="Summary of previously accepted metrics (name + description). Provide an empty string if no prior metrics exist.")
    residual_note: str = dspy.InputField(desc="Additional instructions about residual failure cases that new metrics should address. Provide an empty string if not applicable.")
    axes_of_variation: List[str] = dspy.OutputField(desc="An ordered list (most-important first) describing possible axes of variation. Please bold the name of the axis of variation (e.g. **Axes Name**), and ALSO include a brief sentence-long explanation of the axis of variation. (e.g. **Axes Name** Brief Explanation).  Please include exactly `num_axes_to_generate` axes of variation in the output.  Avoid special characters since they sometimes mess up the parsing.")


class GenerateAxisOfVariation(dspy.Module):
    """DSPy module wrapping a Chain-of-Thought call to generate axes of variation."""

    def __init__(self):
        super().__init__()
        self.generate_axes = dspy.ChainOfThought(GenerateAxisOfVariationSignature)

    def forward(
        self,
        task_description: str,
        good_examples: List[str],
        bad_examples: List[str],
        target_name: Optional[str] = None,
        num_axes_to_generate: int = 5,
        prior_metrics_context: str = "",
        residual_note: str = "",
    ):
        if not target_name:
            target_name = "None"
        response = self.generate_axes(
            task_description=task_description,
            target_name=target_name,
            good_examples=good_examples,
            bad_examples=bad_examples,
            num_axes_to_generate=num_axes_to_generate,
            prior_metrics_context=prior_metrics_context or "",
            residual_note=residual_note or "",
        ).axes_of_variation

        # Clean up each axis string in the list
        axes = [axis.strip() for axis in response]
        # Remove any empty strings
        axes = [axis for axis in axes if axis]
        # If first item starts with a number, remove it
        if axes and axes[0].startswith("1."):
            axes[0] = axes[0][2:].strip()

        return dspy.Prediction(axes_of_variation=axes)


# ---------------------------------------------------------------------------
# Convenience wrapper that downstream code can call directly
# ---------------------------------------------------------------------------

def generate_axes_of_variation(
    task_description: str,
    good_examples: List[str],
    bad_examples: List[str],
    generator_llm: Optional[dspy.LM] = None,
    target_name: Optional[str] = None,
    num_axes_to_generate: int = 5,
    seed: Optional[int] = None,
    prior_metrics_context: Optional[str] = None,
    residual_note: Optional[str] = None,
    prompt_logger: Optional[Callable[[dict], None]] = None,
) -> List[str]:
    """Generate a ranked list of textual axes of variation.
    
    Implements fallback for context length errors by trying fewer examples and truncation.
    """
    # Set temperature based on seed for cache busting
    temperature = 0.0001 * seed if seed is not None else None
    
    def try_generate_with_examples(
        good_ex: List[str],
        bad_ex: List[str],
        attempt_label: str,
    ) -> List[str]:
        """Helper to try generation with given examples."""
        if prompt_logger:
            prompt_logger(
                {
                    "attempt": attempt_label,
                    "task_description": task_description,
                    "target_name": target_name,
                    "num_axes": num_axes_to_generate,
                    "good_examples": good_ex,
                    "bad_examples": bad_ex,
                    "prior_metrics_context": prior_metrics_context,
                    "residual_note": residual_note,
                }
            )
        if generator_llm is not None:
            if temperature is not None:
                with dspy.settings.context(lm=generator_llm, temperature=temperature):
                    axes_pred = GenerateAxisOfVariation()(
                        task_description,
                        good_ex,
                        bad_ex,
                        target_name,
                        num_axes_to_generate,
                        prior_metrics_context or "",
                        residual_note or "",
                    )
            else:
                with dspy.settings.context(lm=generator_llm):
                    axes_pred = GenerateAxisOfVariation()(
                        task_description,
                        good_ex,
                        bad_ex,
                        target_name,
                        num_axes_to_generate,
                        prior_metrics_context or "",
                        residual_note or "",
                    )
        else:
            if temperature is not None:
                with dspy.settings.context(temperature=temperature):
                    axes_pred = GenerateAxisOfVariation()(
                        task_description,
                        good_ex,
                        bad_ex,
                        target_name,
                        num_axes_to_generate,
                        prior_metrics_context or "",
                        residual_note or "",
                    )
            else:
                axes_pred = GenerateAxisOfVariation()(
                    task_description,
                    good_ex,
                    bad_ex,
                    target_name,
                    num_axes_to_generate,
                    prior_metrics_context or "",
                    residual_note or "",
                )
        return axes_pred.axes_of_variation
    
    # Fallback strategy: try fewer examples first, then truncate if needed
    fallback_configs = [
        # Try original examples first
        {"good": good_examples, "bad": bad_examples, "description": "full examples"},
        # Try fewer examples 
        {"good": good_examples[:3], "bad": bad_examples[:3], "description": "3 examples each"},
        {"good": good_examples[:2], "bad": bad_examples[:2], "description": "2 examples each"},
        {"good": good_examples[:1], "bad": bad_examples[:1], "description": "1 example each"},
        # Try truncated examples
        {"good": truncate_examples_if_needed(good_examples[:2], 1500), "bad": truncate_examples_if_needed(bad_examples[:2], 1500), "description": "2 examples truncated to 1500 chars"},
        {"good": truncate_examples_if_needed(good_examples[:1], 1000), "bad": truncate_examples_if_needed(bad_examples[:1], 1000), "description": "1 example truncated to 1000 chars"}
    ]
    
    for i, config in enumerate(fallback_configs):
        try:
            return try_generate_with_examples(
                config["good"],
                config["bad"],
                config["description"],
            )
        except Exception as e:
            error_str = str(e)
            if is_context_length_error(error_str):
                print(f"Context length error with {config['description']}, trying fallback {i+2}/{len(fallback_configs)}...")
                if i == len(fallback_configs) - 1:  # Last attempt failed
                    print(f"All fallback attempts failed. Final error: {error_str}")
                    raise Exception(f"Context length exceeded even with minimal examples. Original error: {error_str}")
                continue
            else:
                # Non-context-length error, re-raise immediately
                raise
    
    # Should never reach here
    raise Exception("Unexpected error in fallback logic")
