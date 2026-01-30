#!/usr/bin/env python3
"""
Utility for profiling metrics in a separate process with clean memory state.
This provides accurate memory measurements for import, construction, and first execution.
"""

import os
import sys
import json
import importlib
import subprocess
from typing import Dict, Any, List, Optional
import traceback

def measure_metric_phases(metric_class_path: str, constructor_kwargs: Optional[Dict[str, Any]] = None, 
                          sample_data: Optional[List] = None) -> List[Dict[str, Any]]:
    """
    Spawn a fresh Python process to measure memory usage during different phases
    of metric loading and execution.
    
    Args:
        metric_class_path: Fully qualified path to the metric class (e.g. 'autometrics.metrics.reference_based.BLEU.BLEU')
        constructor_kwargs: Optional kwargs to pass to the metric constructor
        sample_data: Optional tuple of (input_text, output_text, reference_texts) for first calculation
        
    Returns:
        List of resource snapshots at different phases
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_file = os.path.abspath(__file__)
    
    if constructor_kwargs is None:
        constructor_kwargs = {}
    
    # Default sample if none provided
    if sample_data is None:
        sample_data = ("This is a test input.", "This is a test output.", ["This is a reference."])
    
    # Prepare payload for the child process
    payload = {
        "metric_class_path": metric_class_path,
        "constructor_kwargs": constructor_kwargs,
        "sample_data": sample_data
    }
    
    # Call the child process
    cmd = [
        sys.executable,
        current_file,  # This file acts as both module and script
        json.dumps(payload)
    ]
    
    # Define JSON markers that match the child process
    JSON_START_MARKER = "\n===JSON_DATA_START===\n"
    JSON_END_MARKER = "\n===JSON_DATA_END===\n"
    
    try:
        # Use subprocess.run with a timeout instead of check_output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=1800  # Add a 30-minute timeout
        )
        
        # Check for errors
        if result.returncode != 0:
            print(f"Error profiling metric {metric_class_path} (return code {result.returncode})", file=sys.stderr)
            
            # Log stderr output
            if result.stderr:
                print(f"Stderr: {result.stderr}", file=sys.stderr)
            
            # Try to find JSON data in stdout
            stdout = result.stdout
            json_data = None
            
            # Look for markers
            start_pos = stdout.find(JSON_START_MARKER)
            if start_pos >= 0:
                end_pos = stdout.find(JSON_END_MARKER, start_pos)
                if end_pos >= 0:
                    json_str = stdout[start_pos + len(JSON_START_MARKER):end_pos]
                    try:
                        json_data = json.loads(json_str)
                    except json.decoder.JSONDecodeError:
                        print(f"Failed to parse JSON data between markers", file=sys.stderr)
            
            # If we found and parsed JSON data with an error message, display it
            if json_data and isinstance(json_data, list) and json_data and "error" in json_data[0]:
                print(f"Error details: {json_data[0]['error']}", file=sys.stderr)
                if "traceback" in json_data[0]:
                    print(f"Traceback:\n{json_data[0]['traceback']}", file=sys.stderr)
            else:
                # If we couldn't extract JSON data, just print stdout
                if stdout:
                    print(f"Stdout: {stdout}", file=sys.stderr)
            
            return []
        
        output = result.stdout
        
        # First, check for JSON markers
        start_pos = output.find(JSON_START_MARKER)
        if start_pos >= 0:
            end_pos = output.find(JSON_END_MARKER, start_pos)
            if end_pos >= 0:
                json_str = output[start_pos + len(JSON_START_MARKER):end_pos]
                try:
                    return json.loads(json_str)
                except json.decoder.JSONDecodeError as e:
                    print(f"Failed to parse JSON data between markers: {str(e)}", file=sys.stderr)
                    # If we can't parse the JSON between markers, print it for debugging
                    print(f"JSON string between markers: {json_str}", file=sys.stderr)
        
        # If no markers found or parsing failed, fall back to more aggressive extraction methods
        print("No JSON markers found in output, trying alternative extraction methods", file=sys.stderr)
        
        # Improved JSON extraction - try to find the actual JSON array in the output
        # by looking for the first '[' character and matching it with the last ']'
        try:
            # Print the raw output for debugging
            if output:
                print(f"Raw subprocess output: {output}", file=sys.stderr)
                
            # Find the first '[' which should be the start of our JSON array
            json_start = output.find('[')
            if json_start >= 0:
                # Find the matching closing bracket by parsing from this position
                json_str = output[json_start:]
                # Try to parse just this substring
                output = json.loads(json_str)
            else:
                # If no '[' is found, try parsing the whole thing anyway
                # (this will likely fail but we'll catch the exception)
                output = json.loads(output)
                
        except json.decoder.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}", file=sys.stderr)
            print(f"Could not parse JSON from subprocess output", file=sys.stderr)
            
            # Try a more aggressive approach - look for sequences that could be JSON objects
            try:
                # Try to find anything that looks like a JSON array
                import re
                json_matches = re.findall(r'\[\s*{.*}\s*\]', output, re.DOTALL)
                if json_matches:
                    # Try parsing the first match that works
                    for potential_json in json_matches:
                        try:
                            output = json.loads(potential_json)
                            print(f"Successfully extracted JSON using regex", file=sys.stderr)
                            break
                        except json.decoder.JSONDecodeError:
                            continue
                    else:
                        # If none of the matches worked
                        print(f"Found potential JSON blocks but couldn't parse any of them", file=sys.stderr)
                        return []
                else:
                    print(f"No JSON-like patterns found in output", file=sys.stderr)
                    return []
            except Exception as regex_error:
                print(f"Error in regex extraction attempt: {str(regex_error)}", file=sys.stderr)
                return []
        
        # Process completed successfully, parse output
        return output
        
    except subprocess.TimeoutExpired:
        print(f"Timeout while profiling metric {metric_class_path} (exceeded 10 minutes)", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Unexpected error profiling metric {metric_class_path}: {str(e)}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return []


def _child_main():
    """Entry point for the child process when launched as a script."""
    import importlib
    
    # Redirect warnings to stderr to keep stdout clean for JSON output
    import warnings
    import sys
    
    # Save original stdout and stderr for later
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Define JSON marker strings to help the parent process locate the JSON
    JSON_START_MARKER = "\n===JSON_DATA_START===\n"
    JSON_END_MARKER = "\n===JSON_DATA_END===\n"
    
    # Import resource tracker - must be done before any other imports
    # to get accurate baseline memory
    try:
        from autometrics.experiments.utilization.resource import snap
    except Exception as e:
        error_msg = {"error": f"Error importing resource tracker: {str(e)}", "traceback": traceback.format_exc()}
        original_stdout.write(JSON_START_MARKER)
        original_stdout.write(json.dumps([error_msg]))
        original_stdout.write(JSON_END_MARKER)
        sys.exit(1)
    
    # Get payload from command line
    payload = json.loads(sys.argv[1])
    metric_class_path = payload["metric_class_path"]
    constructor_kwargs = payload["constructor_kwargs"]
    sample_data = payload["sample_data"]
    
    # Extract module and class name
    module_path, class_name = metric_class_path.rsplit(".", 1)
    
    # Record checkpoints
    checkpoints = []
    
    # Initial memory state (baseline)
    checkpoints.append(snap("start"))
    
    # Import the module
    try:
        module = importlib.import_module(module_path)
        checkpoints.append(snap("after_import"))
    except Exception as e:
        error_msg = {"error": f"Error importing module {module_path}: {str(e)}", "traceback": traceback.format_exc()}
        original_stdout.write(JSON_START_MARKER)
        original_stdout.write(json.dumps([error_msg]))
        original_stdout.write(JSON_END_MARKER)
        sys.exit(1)
    
    # Construct the metric
    try:
        MetricClass = getattr(module, class_name)
        metric = MetricClass(**constructor_kwargs)
        checkpoints.append(snap("after_construct"))
    except Exception as e:
        error_msg = {"error": f"Error constructing metric {class_name}: {str(e)}", "traceback": traceback.format_exc()}
        original_stdout.write(JSON_START_MARKER)
        original_stdout.write(json.dumps([error_msg]))
        original_stdout.write(JSON_END_MARKER)
        sys.exit(1)
    
    # First calculation (this will initialize any lazy-loaded components)
    try:
        input_text, output_text, references = sample_data
        metric.calculate(input_text, output_text, references)
        checkpoints.append(snap("after_first_call"))
    except Exception as e:
        error_msg = {"error": f"Error during first calculation: {str(e)}", "traceback": traceback.format_exc()}
        original_stdout.write(JSON_START_MARKER)
        original_stdout.write(json.dumps([error_msg]))
        original_stdout.write(JSON_END_MARKER)
        sys.exit(1)
    
    # Try unloading if the metric supports it
    try:
        if hasattr(metric, "_unload_model") and callable(getattr(metric, "_unload_model")):
            metric._unload_model()
            checkpoints.append(snap("after_unload"))
    except Exception as e:
        # Unloading is optional, don't fail if it doesn't work
        pass
    
    # Return the checkpoints as JSON with markers
    original_stdout.write(JSON_START_MARKER)
    original_stdout.write(json.dumps(checkpoints))
    original_stdout.write(JSON_END_MARKER)


# When run as a script, execute the child process logic
if __name__ == "__main__":
    try:
        _child_main()
    except Exception as e:
        import traceback
        error_msg = {"error": f"Unhandled error in child process: {str(e)}", "traceback": traceback.format_exc()}
        print(json.dumps([error_msg]))
        sys.exit(1) 