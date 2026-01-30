import json
import subprocess
from typing import Any, Dict, List, Optional
import os
import time
import select
import sys

class InterpreterError(ValueError):
    pass

class CustomPythonInterpreter:
    """
    Custom Python interpreter that properly separates stdout/stderr from return values.
    
    Uses our custom runner.js that fixes the package loading message issue.
    """

    def __init__(self, deno_command: Optional[List[str]] = None) -> None:
        if isinstance(deno_command, dict):
            deno_command = None
        # Use our custom runner
        self.deno_command = deno_command or [
            "deno", "run", "--allow-read", "--allow-net", "--allow-write", self._get_custom_runner_path()
        ]
        self.deno_process = None
        self._loaded_packages = set()  # Track which packages we've loaded

    def _get_custom_runner_path(self) -> str:
        """Get the path to our custom runner script"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        custom_runner = os.path.join(current_dir, "custom_runner.js")
        
        if os.path.exists(custom_runner):
            return custom_runner
        else:
            # Fallback to DSPy's runner if ours doesn't exist
            try:
                import dspy.primitives.python_interpreter as dspy_interp
                current_dir = os.path.dirname(os.path.abspath(dspy_interp.__file__))
                return os.path.join(current_dir, "runner.js")
            except:
                import dspy
                dspy_dir = os.path.dirname(dspy.__file__)
                return os.path.join(dspy_dir, "primitives", "runner.js")

    def _ensure_deno_process(self) -> None:
        if self.deno_process is None or self.deno_process.poll() is not None:
            try:
                self.deno_process = subprocess.Popen(
                    self.deno_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            except FileNotFoundError as e:
                install_instructions = (
                    "Deno executable not found. Please install Deno to proceed.\n"
                    "Installation instructions:\n"
                    "> curl -fsSL https://deno.land/install.sh | sh\n"
                    "*or*, on macOS with Homebrew:\n"
                    "> brew install deno\n"
                    "For additional configurations: https://docs.deno.com/runtime/getting_started/installation/"
                )
                raise InterpreterError(install_instructions) from e

    def _inject_variables(self, code: str, variables: Dict[str, Any]) -> str:
        # Insert Python assignments for each variable at the top of the code
        injected_lines = []
        for key, value in variables.items():
            if not key.isidentifier():
                raise InterpreterError(f"Invalid variable name: '{key}'")
            python_value = self._serialize_value(value)
            injected_lines.append(f"{key} = {python_value}")
        injected_code = "\n".join(injected_lines) + "\n" + code
        return injected_code

    def _serialize_value(self, value: Any) -> str:
        # Basic safe serialization
        if isinstance(value, str):
            return repr(value)
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif value is None:
            return 'None'
        elif isinstance(value, list) or isinstance(value, dict):
            return json.dumps(value)
        else:
            raise InterpreterError(f"Unsupported value type: {type(value).__name__}")

    def _detect_packages_in_code(self, code: str) -> set:
        """Detect which packages are imported in the code"""
        packages = set()
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import '):
                # Extract package name from "import package" or "import package.module"
                package = line[7:].split('.')[0].split(' as ')[0].strip()
                packages.add(package)
            elif line.startswith('from '):
                # Extract package name from "from package import ..."
                package = line[5:].split('.')[0].split(' ')[0].strip()
                packages.add(package)
        return packages

    def _pre_load_packages(self, packages: set) -> None:
        """Pre-load packages to avoid loading messages during actual execution"""
        new_packages = packages - self._loaded_packages
        if not new_packages:
            return
            
        # Create a simple import script to load packages
        import_code = '\n'.join([f"import {pkg}" for pkg in new_packages])
        
        self._ensure_deno_process()
        input_data = json.dumps({"code": import_code})
        
        try:
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()
        except BrokenPipeError:
            self._ensure_deno_process()
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()

        # Read the response (might be loading messages)
        output_line = self.deno_process.stdout.readline().strip()
        
        # Mark these packages as loaded regardless of the response
        self._loaded_packages.update(new_packages)

    def _read_output_with_timeout(self, timeout_seconds: float = 60.0) -> str:
        """Read output from Deno process with timeout to prevent hanging"""
        if sys.platform == "win32":
            # Windows doesn't support select on pipes, use a simpler timeout approach
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                if self.deno_process.poll() is not None:
                    # Process has ended
                    break
                try:
                    line = self.deno_process.stdout.readline()
                    if line:
                        return line.strip()
                except:
                    pass
                time.sleep(0.1)
            return ""
        else:
            # Unix-like systems can use select
            ready, _, _ = select.select([self.deno_process.stdout], [], [], timeout_seconds)
            if ready:
                line = self.deno_process.stdout.readline()
                return line.strip() if line else ""
            else:
                return ""

    def execute(self, code: str, variables: Optional[Dict[str, Any]] = None) -> Any:
        variables = variables or {}
        code = self._inject_variables(code, variables)
        
        self._ensure_deno_process()

        # Send the code as JSON - our custom runner handles the rest
        input_data = json.dumps({"code": code})
        
        try:
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()
        except BrokenPipeError:
            self._ensure_deno_process()
            self.deno_process.stdin.write(input_data + "\n")
            self.deno_process.stdin.flush()

        # Read response(s) until we get valid JSON with timeout
        max_attempts = 3
        timeout_per_attempt = 60.0  # 60 seconds per attempt
        
        for attempt in range(max_attempts):
            # Keep reading lines until we get a valid JSON response or timeout
            json_found = False
            while not json_found and attempt < max_attempts:
                output_line = self._read_output_with_timeout(timeout_per_attempt)
                
                if not output_line:
                    break
                
                # Check if this looks like a loading message
                if self._is_loading_message(output_line):
                    continue  # Keep reading more lines
                
                # Try to parse as JSON
                try:
                    deno_result = json.loads(output_line)
                    
                    # Check if this is an NLTK resource error that we can handle
                    if ("error" in deno_result and 
                        deno_result.get("errorType") == "LookupError" and
                        "Resource" in str(deno_result.get("errorArgs", []))):
                        # This is an NLTK resource missing error - we should treat this as an error, not continue
                        json_found = True
                        break
                    
                    json_found = True
                    break
                except json.JSONDecodeError:
                    # If it's not a loading message and not valid JSON, this is an error
                    if attempt == max_attempts - 1:
                        raise InterpreterError(f"Invalid JSON response after {max_attempts} attempts: {output_line}")
                    continue  # Keep reading more lines
            
            if json_found:
                break
                
            if attempt == max_attempts - 1:
                # Final attempt - check if process is still alive
                if self.deno_process.poll() is None:
                    # Process is still running but not responding
                    self.deno_process.terminate()
                    self.deno_process.wait(timeout=5)
                    raise InterpreterError(f"Deno process timed out after {timeout_per_attempt}s.")
                else:
                    # Process ended, read stderr for clues
                    err_output = self.deno_process.stderr.read()
                    raise InterpreterError(f"Deno process ended unexpectedly. Stderr: {err_output}")
        else:
            raise InterpreterError(f"Failed to get valid JSON response after {max_attempts} attempts")
        
        # Check for Deno-level errors
        if "error" in deno_result:
            error_msg = deno_result["error"]
            error_type = deno_result.get("errorType", "Sandbox Error")
            if error_type == "SyntaxError":
                raise SyntaxError(f"Invalid Python syntax: {error_msg}")
            else:
                raise InterpreterError(f"{error_type}: {error_msg}")

        # Our custom runner should return the actual result, not loading messages
        result = deno_result.get("output", None)
        return result

    def _is_loading_message(self, line: str) -> bool:
        """Check if a line looks like a package loading message"""
        loading_indicators = [
            "Loading ",
            "Downloading ",
            "Installing ",
            "cdn.jsdelivr.net",
            ".whl",
            "pyodide",
            "Fetching",
            "Loaded numpy",
            "numpy already loaded",
            "nltk already loaded",
            "No new packages to load"
        ]
        return any(indicator in line for indicator in loading_indicators)

    def __call__(self, code: str, variables: Optional[Dict[str, Any]] = None) -> Any:
        return self.execute(code, variables)

    def shutdown(self) -> None:
        if self.deno_process and self.deno_process.poll() is None:
            shutdown_message = json.dumps({"shutdown": True}) + "\n"
            self.deno_process.stdin.write(shutdown_message)
            self.deno_process.stdin.flush()
            self.deno_process.stdin.close()
            self.deno_process.wait()
            self.deno_process = None 