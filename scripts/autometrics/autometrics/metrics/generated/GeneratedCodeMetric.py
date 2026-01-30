import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import re
import time
from collections import Counter

import dspy

# Pre-import common packages for code execution
import numpy as np
import math
import statistics
from collections import defaultdict, Counter
import itertools

# Try to import common NLP/ML packages
try:
    import scipy
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from autometrics.metrics.generated.utils.metric_card import generate_further_reading
from autometrics.metrics.generated.utils.metric_card import MetricCardBuilder
from autometrics.metrics.generated.GeneratedRefFreeMetric import GeneratedRefFreeMetric
from autometrics.metrics.generated.GeneratedRefBasedMetric import GeneratedRefBasedMetric

# Import our custom interpreter first
try:
    from autometrics.util.custom_python_interpreter import CustomPythonInterpreter
    CUSTOM_INTERPRETER_AVAILABLE = True
except ImportError:
    CUSTOM_INTERPRETER_AVAILABLE = False
    CustomPythonInterpreter = None

# Import DSPy's Python interpreter as fallback
try:
    from dspy.primitives.python_interpreter import PythonInterpreter as DSPyInterpreter
    DSPY_INTERPRETER_AVAILABLE = True
except ImportError:
    DSPY_INTERPRETER_AVAILABLE = False
    DSPyInterpreter = None

__all__ = ["GeneratedRefFreeCodeMetric", "GeneratedRefBasedCodeMetric"]

# Regex pattern to detect and strip function headers (from user's old code)
_HEADER_RE = re.compile(r"""(?mx)
    ^\s*def\s+compute_score\s*\([^)]*\)
    \s*(?:->\s*[^:]+)?\s*:\s*   # up to the colon
    \n                          # newline ONLY – leave the 4 spaces
""")

def _smart_dedent(code: str) -> str:
    """
    Smart dedent that preserves relative indentation while removing base indentation.
    Fixed to handle code structure properly - considers ALL lines when determining indentation.
    """
    lines = code.split('\n')
    
    # Find non-empty lines
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return code
    
    # Calculate indentation for ALL non-empty lines (not just indented ones)
    indentations = [len(line) - len(line.lstrip()) for line in non_empty_lines]
    
    # Find the minimum indentation level
    min_indent = min(indentations)
    
    # If minimum is 0, check if we have a uniform base indentation we can remove
    if min_indent == 0:
        # Check if most lines (excluding imports) have a common indentation
        code_lines = [line for line in non_empty_lines 
                     if not line.strip().startswith(('import ', 'from '))]
        if code_lines:
            code_indentations = [len(line) - len(line.lstrip()) for line in code_lines]
            # If most code lines have the same indentation level > 0, use that as base
            indent_counts = Counter(code_indentations)
            most_common_indent = indent_counts.most_common(1)[0][0]
            if most_common_indent > 0 and indent_counts[most_common_indent] > len(code_lines) / 2:
                min_indent = most_common_indent
    
    # Remove the minimum indentation from all lines
    dedented_lines = []
    for line in lines:
        if line.strip():  # Non-empty line
            current_indent = len(line) - len(line.lstrip())
            if current_indent >= min_indent:
                dedented_lines.append(line[min_indent:])
            else:
                dedented_lines.append(line)  # Keep as-is if less indented
        else:
            dedented_lines.append('')  # Empty line
    
    return '\n'.join(dedented_lines)

def _strip_header_and_dedent(code: str) -> str:
    """Strip function header and dedent the body. If no function header, do NOT dedent (preserve indentation)."""
    code_lines = code.split("\n")
    code = ""
    for line in code_lines:
        if not line.startswith("#"):
            code += line + "\n"
    if "```python" in code:
        code = code.split("```python")[1].split("```", 1)[0]
    elif "```" in code:
        code = code.split("```", 1)[1]

    code_parts = _HEADER_RE.split(code)
    if len(code_parts) > 1:
        # Found function header, dedent the function body
        dedented_body = _smart_dedent(code_parts[1])
        result = code_parts[0] + dedented_body.rstrip()
        return result
    else:
        # No header found, DO NOT dedent—just return cleaned code as-is
        result = code_parts[0].rstrip("\n")
        return result


class SecurityError(Exception):
    """Raised when secure execution is not available"""
    pass


# Base mixin for shared Code metric functionality
class _CodeMetricMixin:
    """Shared functionality for both reference-free and reference-based code metrics."""

    DEFAULT_MAX_WORKERS = 32

    def __init__(
        self,
        name: str,
        description: str,
        generated_code: str,
        task_description: Optional[str] = None,
        measurement_axis: Optional[str] = None,
        metric_card: Optional[str] = None,
        metric_card_author_model: Optional[dspy.LM] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        is_reference_based: bool = False,
        **kwargs,
    ):
        self.generated_code = generated_code
        self.task_description = task_description or "No task description provided"
        self.measurement_axis = measurement_axis or "Quality assessment"
        self.max_workers = max_workers
        self.is_reference_based = kwargs.get("is_reference_based", is_reference_based)
        self._interpreter = None

        if metric_card_author_model is None:
            metric_card_author_model = dspy.settings.lm if hasattr(dspy.settings, 'lm') else None

        if metric_card == "provided":
            self.metric_card = self.__doc__
            metric_card = self.metric_card

        # Initialize parent with shared parameters
        super().__init__(
            name,
            description,
            metric_card=metric_card,
            metric_card_author_model=metric_card_author_model,
            generated_code=generated_code,
            task_description=self.task_description,
            measurement_axis=self.measurement_axis,
            **kwargs,
        )

        # Exclude heavy objects from cache key
        self.exclude_from_cache_key("_interpreter", "metric_card_author_model")

    def _get_interpreter(self):
        """Get or create an interpreter instance, preferring our custom one"""
        if self._interpreter is None:
            # Prefer our custom interpreter with enhanced package loading
            if CUSTOM_INTERPRETER_AVAILABLE:
                self._interpreter = CustomPythonInterpreter()
            elif DSPY_INTERPRETER_AVAILABLE:
                self._interpreter = DSPyInterpreter()
            else:
                raise SecurityError(
                    "No Python interpreter available. Please install DSPy for secure code execution: "
                    "pip install dspy-ai"
                )
        return self._interpreter

    def _parse_generated_code(self, code: str) -> tuple[str, str]:
        """
        Parse generated code to separate imports from the main logic.
        Returns (imports_section, logic_section)
        """
        lines = code.strip().split('\n')
        import_lines = []
        logic_lines = []
        
        # Track if we're still in the imports section
        in_imports = True
        
        for line in lines:
            stripped_line = line.strip()
            
            # Check if this line is an import statement
            if (stripped_line.startswith('import ') or 
                stripped_line.startswith('from ') or
                stripped_line == '' and in_imports):  # Empty lines in imports section
                import_lines.append(line)
            else:
                # Once we hit non-import code, everything else is logic
                in_imports = False
                logic_lines.append(line)
        
        imports_section = '\n'.join(import_lines)
        logic_section = '\n'.join(logic_lines)
        
        return imports_section.strip(), logic_section.strip()

    def _indent_code(self, code: str) -> str:
        """Indent code for function wrapping - adds 4 spaces to ALL lines"""
        if not code.strip():
            return "    return 0.0"
        
        lines = code.split('\n')
        indented_lines = []
        
        for line in lines:
            if line.strip():  # Non-empty line - always add 4 spaces
                indented_lines.append('    ' + line)
            else:  # Empty line
                indented_lines.append('')
        
        indented_result = '\n'.join(indented_lines)
        
        # Ensure the function ends with a return statement if it doesn't have one
        if not any('return ' in line.strip() for line in lines if line.strip()):
            indented_result += '\n    return 0.0'
        
        return indented_result

    def _is_loading_message(self, result) -> bool:
        """Check if a result looks like a package loading message"""
        if not isinstance(result, str):
            return False
        
        result_lower = result.lower()
        loading_indicators = [
            "loading ",
            "downloading ",
            "installing ",
            "cdn.jsdelivr.net",
            ".whl",
            "pyodide",
            "fetching",
            "regex",
            "sqlite3",
            "nltk",
            "package",
            "wheel",
            "download",
            "failed to load"
        ]
        return any(indicator in result_lower for indicator in loading_indicators)

    def _ensure_numeric_result(self, result) -> float:
        """Ensure the result is a numeric value with better error reporting"""
        if result is None:
            raise ValueError("Generated code returned None - this likely indicates the code did not execute properly or did not return a value")
        elif isinstance(result, (int, float)):
            return float(result)
        elif isinstance(result, bool):
            return float(result)
        else:
            try:
                numeric_result = float(result)
                return numeric_result
            except (ValueError, TypeError) as e:
                raise ValueError(f"Generated code returned non-numeric result: {result} (type: {type(result)}). Error: {e}")

    def _execute_generated_code(self, input_text: str, output_text: str, references: Optional[List[str]] = None) -> float:
        """Execute the generated code with the given inputs using direct eval (simpler approach)"""
        
        input_text = str(input_text) if input_text is not None else ""
        output_text = str(output_text) if output_text is not None else ""
        if references is not None:
            references = [str(ref) if ref is not None else "" for ref in references]
        
        # Clean the generated code using the proven approach from the old CodeGenerator
        cleaned_code = _strip_header_and_dedent(self.generated_code)
        
        # If the cleaned code is empty or too short, return 0
        if not cleaned_code or len(cleaned_code.strip()) < 3:
            return 0.0
        
        # Check if the cleaned code references the 'references' variable
        # If so, we need to include it in the function signature even for reference-free metrics
        code_needs_references = 'references' in cleaned_code
        has_references = references is not None
        
        # Create the execution namespace with pre-imported packages
        exec_namespace = {
            # Basic variables
            'input': input_text,
            'output': output_text,
            'references': references,
            
            # Pre-imported packages
            'np': np,
            'numpy': np,
            'math': math,
            'statistics': statistics,
            'defaultdict': defaultdict,
            'Counter': Counter,
            'itertools': itertools,
            're': re,
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
            'all': all,
            'any': any,
            'enumerate': enumerate,
            'zip': zip,
            'range': range,
        }
        
        # Add scipy if available
        if SCIPY_AVAILABLE:
            exec_namespace['scipy'] = scipy
        
        # Add nltk if available
        if NLTK_AVAILABLE:
            exec_namespace['nltk'] = nltk
        
        # Handle dynamic imports if needed
        self._handle_dynamic_imports(cleaned_code, exec_namespace)
        
        try:
            # If the code contains return statements, wrap it in a function and execute
            if 'return ' in cleaned_code:
                # Create a temporary function to execute the code and get the result
                func_code = f"def _temp_func():\n" + '\n'.join(f"    {line}" for line in cleaned_code.split('\n'))
                exec(func_code, exec_namespace)
                result = exec_namespace['_temp_func']()
            else:
                # If no return statement, assume the code is an expression
                result = eval(cleaned_code, exec_namespace)
            
            # Convert to numeric result with detailed error reporting
            numeric_result = self._ensure_numeric_result(result)
            return numeric_result
            
        except Exception as e:
            # Try to provide helpful debugging information
            raise ValueError(f"Error executing generated code: {e}\nGenerated code:\n{self.generated_code}\nCleaned code:\n{cleaned_code}")

    def _handle_dynamic_imports(self, code: str, namespace: dict):
        """Handle dynamic imports requested in the code"""
        lines = code.split('\n')
        
        for line in lines:
            stripped = line.strip()
            
            # Handle simple imports
            if stripped.startswith('import '):
                module_name = stripped[7:].split('.')[0].split(' as ')[0].strip()
                try:
                    if module_name not in namespace:
                        module = __import__(module_name)
                        namespace[module_name] = module
                except ImportError:
                    print(f"Warning: Could not import {module_name}")
            
            # Handle from imports
            elif stripped.startswith('from '):
                parts = stripped.split()
                if len(parts) >= 4 and parts[2] == 'import':
                    module_name = parts[1]
                    import_items = ' '.join(parts[3:]).split(',')
                    try:
                        module = __import__(module_name, fromlist=import_items)
                        for item in import_items:
                            item = item.strip().split(' as ')[0].strip()
                            if hasattr(module, item):
                                namespace[item] = getattr(module, item)
                    except ImportError:
                        print(f"Warning: Could not import {import_items} from {module_name}")

    # =======================================================================
    # OLD INTERPRETER-BASED CODE (KEPT FOR FUTURE USE)
    # =======================================================================
    
    def _execute_generated_code_with_interpreter(self, input_text: str, output_text: str, references: Optional[List[str]] = None) -> float:
        """OLD VERSION: Execute the generated code with the given inputs using interpreters (kept for potential future use)"""
        try:
            interpreter = self._get_interpreter()
        except Exception as e:
            raise RuntimeError(f"Failed to get interpreter: {e}")
        
        # Clean the generated code using the proven approach from the old CodeGenerator
        cleaned_code = _strip_header_and_dedent(self.generated_code)
        
        # Pre-load third-party packages best-effort to avoid long loading messages / failures
        try:
            self._preload_packages(interpreter, cleaned_code)
        except Exception:
            pass  # Non-fatal – continue even if preload fails
        
        # If the cleaned code is empty or too short, return 0
        if not cleaned_code or len(cleaned_code.strip()) < 3:
            return 0.0
        
        # Build the complete code with function signature
        has_references = references is not None
        
        # Check if the cleaned code references the 'references' variable
        # If so, we need to include it in the function signature even for reference-free metrics
        code_needs_references = 'references' in cleaned_code
        
        # Create appropriate function signature and body
        if has_references or code_needs_references:
            function_signature = "def compute_score(input, output, references=None):"
        else:
            function_signature = "def compute_score(input, output):"
        
        # Indent the cleaned code for the function body
        indented_code = self._indent_code(cleaned_code) if cleaned_code else "    return 0.0"
        
        # Build the complete setup code
        setup_code = f"{function_signature}\n{indented_code}\n"
        
        # Execute the setup code to define the function
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Execute setup code to define the function
                setup_result = interpreter.execute(setup_code, {})
                
                # Check if this still looks like a loading message (edge case)
                if isinstance(setup_result, str) and self._is_loading_message(setup_result):
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1}: Got loading message during setup, retrying...")
                        time.sleep(5)
                        continue
                    else:
                        raise RuntimeError(f"Setup consistently returned loading messages: {setup_result}")
                
                break  # Setup successful
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}: Setup failed with error {e}, retrying...")
                    if hasattr(e, 'stack_trace'):
                        print(e.stack_trace)
                    time.sleep(2)
                    continue
                else:
                    print(f"Failed to execute setup code after {max_retries} attempts. Setup code:\n{setup_code}\nError: {e}")
                    raise e
            
        # Prepare variables for execution
        variables = {
            'input': input_text,
            'output': output_text
        }
        
        # Always include references in variables if the function signature includes it
        if has_references or code_needs_references:
            variables['references'] = references
        
        # Create the function call code based on function signature
        if has_references or code_needs_references:
            call_code = "result = compute_score(input, output, references)\nresult"
        else:
            call_code = "result = compute_score(input, output)\nresult"
        
        # Execute the function call
        try:
            result = interpreter.execute(call_code, variables)
        except Exception as e:
            # If the function call failed, it might be a scope issue. Try to diagnose.
            diagnostic_code = "compute_score"
            try:
                diagnostic_result = interpreter.execute(diagnostic_code, {})
                if diagnostic_result is None:
                    print(f"Function 'compute_score' was not properly defined. This might be due to function parsing issues. Original error: {e}")
                    raise e
                else:
                    print(f"Function call failed with error: {e}. Function exists but call failed.")
                    raise e
            except:
                print(f"Function 'compute_score' is not defined. Setup may have failed silently. Original error: {e}")
                raise e
                
        # Convert to numeric result with detailed error reporting
        try:
            numeric_result = self._ensure_numeric_result(result)
            return numeric_result
        except ValueError as e:
            # Re-raise with more context
            raise ValueError(f"{e}\nGenerated code:\n{self.generated_code}\nCleaned code:\n{cleaned_code}\nSetup code:\n{setup_code}\nCall code:\n{call_code}")

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        del kwargs  # pragma: no cover
        results: List[float] = [0.0] * len(outputs)

        # Fail-fast if workers=1 - better for debugging
        if self.max_workers == 1:
            for i, (inp, out, ref) in enumerate(zip(inputs, outputs, references or [None] * len(outputs))):
                try:
                    results[i] = self._execute_generated_code(inp, out, ref)
                except Exception as e:
                    print(f"Error processing item {i} (input: {inp[:50]}{'...' if len(inp) > 50 else ''}): {e}")
                    raise  # Re-raise the exception instead of silently continuing
            return results

        # Use threading for multiple workers, but still propagate errors
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._execute_generated_code, i, o, r): idx 
                for idx, (i, o, r) in enumerate(zip(inputs, outputs, references or [None] * len(outputs)))
            }
            
            # Collect results
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Error processing item {index}: {e}")
                    raise  # Re-raise the exception instead of silently continuing
        
        return results

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _generate_python_code(self, include_metric_card: bool = True) -> str:
        """Export a standalone python file that re-creates this metric."""
        class_name = "GeneratedRefBasedCodeMetric" if self.is_reference_based else "GeneratedRefFreeCodeMetric"
        
        # Parse the generated code to separate imports from logic
        imports_section, logic_section = self._parse_generated_code(self.generated_code)
        
        # Clean up the logic section - remove any existing function definitions
        logic_lines = []
        for line in logic_section.split('\n'):
            stripped = line.strip()
            # Skip function definition lines and empty lines at the start
            if not stripped.startswith('def ') and not (not stripped and not logic_lines):
                logic_lines.append(line)
        
        # Join the logic lines and ensure proper indentation
        cleaned_logic = '\n'.join(logic_lines)
        if not cleaned_logic.strip():
            cleaned_logic = "return 0.0"  # Default fallback
        
        # Indent the logic for the method body (needs 8 spaces total for method body)
        indented_logic = self._indent_code(self._indent_code(cleaned_logic))
        
        # Create method signature based on reference type
        if self.is_reference_based:
            method_signature = "    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401"
            method_body = f"""        del kwargs  # pragma: no cover
        input = str(input)
        output = str(output)
        references = [str(ref) for ref in references] if references is not None else None

{indented_logic}"""
        else:
            method_signature = "    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401"
            method_body = f"""        del references, kwargs  # pragma: no cover
        input = str(input)
        output = str(output)
        references = [str(ref) for ref in references] if references is not None else None

{indented_logic}"""
        
        # Include imports at module level if any
        imports_code = f"\n{imports_section}\n" if imports_section else ""
        
        code = f"""# Auto-generated code metric file for {self.name}
import os{imports_code}
from autometrics.metrics.generated.GeneratedCodeMetric import {class_name}
from typing import ClassVar

class {self.name.replace(" ", "_").replace("-", "_")}_Code({class_name}):
    \"\"\"{self.metric_card if include_metric_card else ""}\"\"\"

    description: ClassVar[str] = {json.dumps(self.description)}

    def __init__(self):
        super().__init__(
            name={json.dumps(self.name)},
            description={json.dumps(self.description)},
            generated_code={json.dumps(self.generated_code)},
            task_description={json.dumps(self.task_description)},
            measurement_axis={json.dumps(self.measurement_axis)},
            metric_card={json.dumps("provided" if include_metric_card else "None")},
            max_workers={self.max_workers},
        )

{method_signature}
{method_body}

    def __repr__(self):
        return f"{self.name.replace(' ', '_').replace('-', '_')}_Code()"

"""
        return code

    def _serialize(self) -> dict:
        """Serialize the metric to a dictionary for in-memory operations."""
        return {
            "name": self.name,
            "description": self.description,
            "generated_code": self.generated_code,
            "task_description": self.task_description,
            "measurement_axis": self.measurement_axis,
            "metric_card": self.metric_card,
            "max_workers": self.max_workers,
            "is_reference_based": self.is_reference_based,
        }

    @classmethod
    def _deserialize(cls, data: dict):
        """Deserialize a dictionary to create a metric instance."""
        return cls(**data)
    
    # ------------------------------------------------------------------
    # Metric-card helpers
    # ------------------------------------------------------------------

    def generate_code_explanation(self):
        """Generate an LLM explanation of what the code does."""
        class CodeExplanationSignature(dspy.Signature):
            """Given a task description, measurement axis, and generated Python code, provide a clear explanation of what the code does, how it works, and what it measures."""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            measurement_axis: str = dspy.InputField(desc="The measurement axis or quality dimension being assessed.")
            generated_code: str = dspy.InputField(desc="The Python code that will be executed to compute the metric.")
            explanation: str = dspy.OutputField(desc="A clear, detailed explanation of what the code does and how it computes the metric score.")

        with dspy.settings.context(lm=self.metric_card_author_model):
            outputs = dspy.ChainOfThought(CodeExplanationSignature)(
                task_description=self.task_description,
                measurement_axis=self.measurement_axis,
                generated_code=self.generated_code[:1000] + "..." if len(self.generated_code) > 1000 else self.generated_code,
            )
        
        return outputs.explanation

    def _metric_details_template(self, *, reference_based: bool) -> str:
        """Return the *Metric Details* section for ref-free / ref-based code metrics."""
        kind = "reference-based" if reference_based else "reference-free"
        ref_flag = "Yes" if reference_based else "No"
        input_req = "Yes (plus reference)" if reference_based else "Yes"

        # --- Header & description ----------------------------------------
        lines = [
            f"**{self.name}** is a **{kind}** code-based metric that executes dynamically generated Python code to evaluate system outputs.",
            f"This metric was generated for the measurement axis: `{self.measurement_axis}`.",
            "",
            "The metric executes custom Python code that:",
            "",
            "1. **Analyzes input text** *x*",
            "2. **Evaluates output text** *y*",
        ]
        if reference_based:
            lines.append("3. **Compares against reference text** *r*")
            lines.append("4. **Returns a numerical score**")
        else:
            lines.append("3. **Returns a numerical score**")

        lines.extend([
            "",
            "The code is executed in a secure sandbox environment using Python interpreters.",
            "",
            "- **Metric Type:** Code-based (dynamically generated)",
            "- **Range:** Variable (depends on generated code)",
            "- **Higher is Better?:** Usually (depends on implementation)",
            f"- **Reference-Based?:** {ref_flag}",
            f"- **Input-Required?:** {input_req}",
            "",
            "### Generated Code",
            "",
            "The following Python code is executed for each evaluation:",
            "",
            "```python",
        ])
        
        # Add the actual generated code
        lines.append(self.generated_code)
        lines.append("```")
        
        # Add LLM-generated explanation of the code
        lines.extend([
            "",
            "### Code Explanation",
            "",
        ])
        
        try:
            code_explanation = self.generate_code_explanation()
            lines.append(code_explanation)
        except Exception as e:
            lines.append(f"*Code explanation could not be generated: {e}*")
        
        lines.extend([
            "",
            "### Inputs and Outputs",
            "- **Inputs:**",
            "  - **Input text** *x*",
            "  - **Output text** *y*",
        ])
        if reference_based:
            lines.append("  - **Reference text** *r*")
        lines.extend([
            "- **Outputs:**",
            "  - Numerical score (type and range depend on generated code)",
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
            """Given the task description, measurement axis, and generated code, consider a code-based metric that executes this code to evaluate text. Your task is to generate the domain, a list of tasks, and a set of circumstances where this code metric is best suited to be used as well as where it should not be used."""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            measurement_axis: str = dspy.InputField(desc="The measurement axis or quality dimension being assessed.")
            generated_code: str = dspy.InputField(desc="The Python code that will be executed to compute the metric.")
            domain: str = dspy.OutputField(desc="The domain of the task. Some examples are: Text Generation, Code Generation, Discourse, etc.")
            tasks: List[str] = dspy.OutputField(desc="A list of tasks that this code metric is best suited to be used for.")
            best_suited_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where this code metric is best suited to be used. (approximately one sentence each)")
            not_recommended_for_circumstances: List[str] = dspy.OutputField(desc="A list of circumstances where this code metric is not recommended to be used. (approximately one sentence each)")

        with dspy.settings.context(lm=self.metric_card_author_model):
            outputs = dspy.ChainOfThought(IntendedUseSignature)(
                task_description=self.task_description,
                measurement_axis=self.measurement_axis,
                generated_code=self.generated_code[:3000] + "..." if len(self.generated_code) > 3000 else self.generated_code,
            )
        
        tasks_list = "\n  - " + "\n  - ".join(outputs.tasks)
        suited_list = "\n  - " + "\n  - ".join(outputs.best_suited_for_circumstances)
        not_recommended_list = "\n  - " + "\n  - ".join(outputs.not_recommended_for_circumstances)
        
        return f"""- **Domain:** {outputs.domain}
- **Tasks:** {tasks_list}
- **Best Suited For:** {suited_list}
- **Not Recommended For:** {not_recommended_list}"""

    def generate_metric_implementation(self):
        ref_type = "reference-based" if self.is_reference_based else "reference-free"
        return f"""### Reference Implementations

- **Libraries/Packages:**
  - [AutoMetrics Code Metrics ({ref_type})](https://github.com/XenonMolecule/autometrics/blob/main/autometrics/metrics/generated/GeneratedCodeMetric.py)

### Computational Complexity

- **Efficiency:**
  - Depends on the complexity of the generated code.
  - Code is executed in a secure Python interpreter (Pyodide or DSPy).
  - AutoMetrics supports parallel execution on batched inputs.

- **Scalability:**
  - Performance is linear in the number of input-output pairs.
  - Performance depends on the generated code complexity and required libraries.
  - Code execution overhead varies by interpreter type."""

    def generate_known_limitations(self):
        class KnownLimitationsSignature(dspy.Signature):
            """Given the task description, measurement axis, and generated code, consider a code-based metric that executes this code. Your task is to generate a list of biases, task misalignment risks, and failure cases that could be present in this evaluation."""
            task_description: str = dspy.InputField(desc="Brief description of the underlying task which is being evaluated.")
            measurement_axis: str = dspy.InputField(desc="The measurement axis or quality dimension being assessed.")
            generated_code: str = dspy.InputField(desc="The Python code that will be executed to compute the metric.")
            biases: List[str] = dspy.OutputField(desc="A list of biases that could be present in this evaluation (approximately one sentence each).")
            task_misalignment_risks: List[str] = dspy.OutputField(desc="A list of ways in which this evaluation could be misaligned with the task (approximately one sentence each).")
            failure_cases: List[str] = dspy.OutputField(desc="A list of failure cases that could occur in this evaluation (approximately one sentence each).")

        with dspy.settings.context(lm=self.metric_card_author_model):
            outputs = dspy.ChainOfThought(KnownLimitationsSignature)(
                task_description=self.task_description,
                measurement_axis=self.measurement_axis,
                generated_code=self.generated_code[:3000] + "..." if len(self.generated_code) > 3000 else self.generated_code,
            )
        
        biases_list = "\n  - " + "\n  - ".join(outputs.biases)
        risks_list = "\n  - " + "\n  - ".join(outputs.task_misalignment_risks)
        failures_list = "\n  - " + "\n  - ".join(outputs.failure_cases)
        
        return f"""- **Biases:** {biases_list}
- **Task Misalignment Risks:** {risks_list}
- **Failure Cases:** {failures_list}"""

    def _generate_metric_card(self, author_model: Optional[dspy.LM] = None):
        """Produce a metric card via a custom builder."""
        
        class CodeMetricCardBuilder(MetricCardBuilder):
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
            
        with dspy.settings.context(lm=author_model or self.metric_card_author_model):
            builder = CodeMetricCardBuilder(self)
            return builder.build()

    # ------------------------------------------------------------------
    # 🛠️  Package pre-loading helpers  (adapted from legacy implementation)
    # ------------------------------------------------------------------

    _BUILTIN_PACKAGES = {
        'sys', 'os', 'math', 're', 'json', 'time', 'collections', 'itertools',
        'datetime', 'random', 'statistics', 'functools', 'typing', 'string',
    }

    def _extract_imports(self, code: str) -> List[str]:
        """Extract top-level import statements from code and return package names."""
        packages: List[str] = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import '):
                pkg = stripped[7:].split('.')[0].split(' as ')[0].strip()
                if pkg and pkg not in self._BUILTIN_PACKAGES:
                    packages.append(pkg)
            elif stripped.startswith('from '):
                pkg = stripped[5:].split('.')[0].split(' ')[0].strip()
                if pkg and pkg not in self._BUILTIN_PACKAGES:
                    packages.append(pkg)
        return packages

    def _preload_packages(self, interpreter, code: str) -> None:
        """Attempt to import third-party packages ahead of running user code.

        If the package is missing, we try to install it via micropip/pyodide_js
        (best-effort). We swallow errors so a missing package can still fail
        gracefully during actual metric execution and surface a clean error.
        """
        packages = self._extract_imports(code)
        if not packages:
            return

        # Best-effort import each package (one by one to isolate failures)
        for pkg in packages:
            try:
                interpreter.execute(f"import {pkg}\n{pkg}.__name__", {})
            except Exception:
                # Attempt micropip install if available
                try:
                    install_code = (
                        "import pyodide_js, micropip, asyncio; "
                        f"await pyodide_js.loadPackage('micropip'); "
                        f"await micropip.install('{pkg}');"
                    )
                    interpreter.execute(install_code, {})
                except Exception:
                    # Give up; actual metric execution will raise a clear error
                    pass


class GeneratedRefFreeCodeMetric(_CodeMetricMixin, GeneratedRefFreeMetric):
    """Reference-free metric that executes dynamically generated Python code to evaluate outputs.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    generated_code  The Python code to execute for evaluation
    task_description Optional task context
    measurement_axis The measurement axis this metric was generated for
    metric_card_author_model  LLM used to generate the metric-card
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = False
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del references, kwargs  # pragma: no cover
        return self._execute_generated_code(input, output)


class GeneratedRefBasedCodeMetric(_CodeMetricMixin, GeneratedRefBasedMetric):
    """Reference-based metric that executes dynamically generated Python code to evaluate outputs using reference text.

    Parameters
    ----------
    name            Human-readable metric identifier
    description     Short description
    generated_code  The Python code to execute for evaluation
    task_description Optional task context
    measurement_axis The measurement axis this metric was generated for
    metric_card_author_model  LLM used to generate the metric-card
    """

    def __init__(self, *args, **kwargs):
        kwargs['is_reference_based'] = True
        super().__init__(*args, **kwargs)

    def _calculate_impl(self, input, output, references=None, **kwargs):  # noqa: D401
        del kwargs  # pragma: no cover
        return self._execute_generated_code(input, output, references) 