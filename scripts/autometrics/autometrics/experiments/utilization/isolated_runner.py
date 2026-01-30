#!/usr/bin/env python3
"""
Utility for running metric trials in isolated subprocesses.
This ensures true memory isolation between different trials and length categories.
"""

import os
import sys
import json
import tempfile
import subprocess
import tracemalloc
import gc
import psutil
from typing import Dict, Any, List, Optional

def get_rss_mb():
    """Return the current Resident Set Size in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)

def run_isolated_category(
    metric_class_path: str, 
    constructor_kwargs: Dict[str, Any], 
    category: str,
    num_examples: int,
    num_burn_in: int,
    use_synthetic: bool,
    seed: int = 42,
    use_deterministic_examples: bool = False
) -> List[Dict[str, Any]]:
    """
    Run trials for a specific length category in an isolated subprocess.
    
    Args:
        metric_class_path: Fully qualified path to the metric class
        constructor_kwargs: Constructor parameters for the metric
        category: Length category ('short', 'medium', 'long') or 'dataset'
        num_examples: Number of examples to test
        num_burn_in: Number of burn-in examples
        use_synthetic: Whether to use synthetic data
        seed: Random seed
        
    Returns:
        List of resource measurement dictionaries
    """
    # Get the path to this file
    current_file = os.path.abspath(__file__)
    
    # Prepare the payload
    payload = {
        "metric_class_path": metric_class_path,
        "constructor_kwargs": constructor_kwargs,
        "category": category,
        "num_examples": num_examples,
        "num_burn_in": num_burn_in,
        "use_synthetic": use_synthetic,
        "seed": seed,
        "use_deterministic_examples": use_deterministic_examples
    }
    
    # Create a temporary file to store the results
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
        result_file = tmp.name
    
    # Command to run the isolated trials with a clean Python interpreter
    cmd = [
        sys.executable,
        "-Xfrozen_modules=off",  # Disable frozen modules to reduce memory overhead
        current_file,  # This file acts as both module and script
        json.dumps(payload),
        result_file
    ]
    
    try:
        # Run the subprocess with a timeout
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},  # Add deterministic hashing
            timeout=3600  # 1-hour timeout
        )
        
        # Check for errors
        if process.returncode != 0:
            print(f"Error running isolated trials for {metric_class_path} category {category} (return code {process.returncode})", file=sys.stderr)
            if process.stderr:
                print(f"Stderr: {process.stderr}", file=sys.stderr)
            if process.stdout:
                print(f"Stdout: {process.stdout}", file=sys.stderr)
            return []
        
        # Read results from the temporary file
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            # Check for error in results
            if isinstance(results, dict) and "error" in results:
                print(f"Error in isolated trials: {results['error']}", file=sys.stderr)
                if "traceback" in results:
                    print(f"Traceback:\n{results['traceback']}", file=sys.stderr)
                return []
                
            return results
        except Exception as e:
            print(f"Error reading results: {str(e)}", file=sys.stderr)
            return []
        finally:
            # Clean up the temporary file
            try:
                os.unlink(result_file)
            except:
                pass
            
    except subprocess.TimeoutExpired:
        print(f"Timeout running isolated trials for {metric_class_path} category {category}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return []


def _subprocess_main():
    """Entry point for the child process when launched as a script."""
    import json
    import sys
    import os
    import gc
    
    # Start memory tracking at the very beginning
    tracemalloc.start()
    
    # Set environment variables for more deterministic behavior
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    
    # Get payload from command line
    payload = json.loads(sys.argv[1])
    result_file = sys.argv[2]
    
    # Explicitly collect garbage before doing anything
    gc.collect()
    
    # Log initial memory state
    initial_rss = get_rss_mb()
    print(f"Initial RSS: {initial_rss:.2f} MB")
    
    # Run the actual trials
    try:
        # Import utilization modules only after measuring initial memory
        from autometrics.experiments.utilization.utilization import run_isolated_trials
        
        # Force garbage collection again after imports
        gc.collect()
        
        # Log post-import memory state
        post_import_rss = get_rss_mb()
        print(f"After imports RSS: {post_import_rss:.2f} MB (delta: {post_import_rss - initial_rss:.2f} MB)")
        
        # Run the isolated trials
        results = run_isolated_trials(
            metric_class_path=payload["metric_class_path"],
            constructor_kwargs=payload["constructor_kwargs"],
            category=payload["category"],
            num_examples=payload["num_examples"],
            num_burn_in=payload["num_burn_in"],
            use_synthetic=payload["use_synthetic"],
            seed=payload["seed"],
            use_deterministic_examples=payload["use_deterministic_examples"]
        )
        
        # Final garbage collection
        gc.collect()
        
        # Log final memory state
        final_rss = get_rss_mb()
        print(f"Final RSS: {final_rss:.2f} MB (total delta: {final_rss - initial_rss:.2f} MB)")
        
        # Take tracemalloc snapshot for top memory users
        snapshot = tracemalloc.take_snapshot()
        print(f"Top 10 memory allocations:")
        for stat in snapshot.statistics('lineno')[:10]:
            print(f"  {stat}")
        
        # Write results to the specified file
        with open(result_file, 'w') as f:
            json.dump(results, f)
            
    except Exception as e:
        import traceback
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "memory_info": {
                "initial_rss_mb": initial_rss,
                "final_rss_mb": get_rss_mb()
            }
        }
        with open(result_file, 'w') as f:
            json.dump(error_info, f)
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


# When run as a script, execute the subprocess logic
if __name__ == "__main__":
    _subprocess_main() 