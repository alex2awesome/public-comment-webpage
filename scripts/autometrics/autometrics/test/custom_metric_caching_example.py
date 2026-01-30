#!/usr/bin/env python
"""
Example of how to properly implement caching in a custom metric class.

This demonstrates how initialization parameters automatically affect caching,
and how to explicitly pass parameters to the parent constructor.
"""

import sys
import os
import time

# Add parent directory to path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric


class CustomMetric(ReferenceBasedMetric):
    """
    Example custom metric with proper caching implementation.
    """
    def __init__(self, 
                 name="CustomMetric", 
                 description="An example custom metric with caching",
                 model_type="large",  # Parameter that affects results
                 threshold=0.5,       # Parameter that affects results
                 debug=False,         # Parameter that doesn't affect results 
                 **kwargs):
        
        # Call the parent constructor with ALL parameters explicitly
        # This ensures they are properly included in the cache key
        super().__init__(
            name=name, 
            description=description,
            model_type=model_type,  # Pass all parameters that affect the metric
            threshold=threshold,    # Pass all parameters that affect the metric
            debug=debug,            # Pass parameters even if they don't affect the output
            **kwargs
        )
        
        # Store the parameters as instance variables
        self.model_type = model_type
        self.threshold = threshold
        self.debug = debug
        
        # By default, all initialization parameters are included in the cache key
        # except 'name', 'description', 'use_cache', and 'cache_dir'
        # We only need to exclude additional parameters that don't affect the metric output
        self.exclude_from_cache_key('debug')
        
    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Actual implementation of the metric calculation.
        
        This is where you'd put your actual metric computation code.
        The result of this will be cached based on inputs and init params.
        """
        if self.debug:
            print(f"Computing metric with model_type={self.model_type}, threshold={self.threshold}")
            print(f"Input: {input}")
            print(f"Output: {output}")
            print(f"References: {references}")
        
        # Simulate expensive computation
        time.sleep(1.0)
        
        # Different model_type parameters produce different results
        if self.model_type == "large":
            score = len(set(output.split()) & set(references[0].split())) / max(len(output.split()), 1)
        else:  # small model
            score = len(set(output.split()) & set(input.split())) / max(len(output.split()), 1)
            
        # The threshold also affects results
        if score < self.threshold:
            score = 0.0
            
        return score


def demonstrate_caching():
    """Run a simple demonstration of caching with the custom metric."""
    
    # Create metrics with different parameters
    metric1 = CustomMetric(model_type="large", threshold=0.5)
    metric2 = CustomMetric(model_type="small", threshold=0.5)
    metric3 = CustomMetric(model_type="large", threshold=0.2)
    
    # Create metrics with same affecting parameters but different non-affecting parameters
    # These should use the same cache entry
    metric4 = CustomMetric(model_type="large", threshold=0.5, debug=True)  # Same as metric1 but with debug=True
    metric5 = CustomMetric(model_type="large", threshold=0.5, debug=False) # Same as metric1 but with debug=False
    
    # Some test data
    input_text = "The quick brown fox jumps over the lazy dog."
    output_text = "A fast fox jumps over a dog."
    references = ["The speedy fox leaped over the sleepy hound."]
    
    # First runs - should be cache misses
    print("\n1. First runs (cache misses):")
    
    start = time.time()
    result1 = metric1.calculate(input_text, output_text, references)
    time1 = time.time() - start
    print(f"  Metric1 (large, 0.5): {result1:.4f} in {time1:.4f}s")
    
    start = time.time()
    result2 = metric2.calculate(input_text, output_text, references)
    time2 = time.time() - start
    print(f"  Metric2 (small, 0.5): {result2:.4f} in {time2:.4f}s")
    
    start = time.time()
    result3 = metric3.calculate(input_text, output_text, references)
    time3 = time.time() - start
    print(f"  Metric3 (large, 0.2): {result3:.4f} in {time3:.4f}s")
    
    # Second runs - should be cache hits and much faster
    print("\n2. Second runs (cache hits):")
    
    start = time.time()
    result1_again = metric1.calculate(input_text, output_text, references)
    time1_again = time.time() - start
    print(f"  Metric1 (large, 0.5): {result1_again:.4f} in {time1_again:.4f}s - Speedup: {time1/time1_again:.1f}x")
    
    start = time.time()
    result2_again = metric2.calculate(input_text, output_text, references)
    time2_again = time.time() - start
    print(f"  Metric2 (small, 0.5): {result2_again:.4f} in {time2_again:.4f}s - Speedup: {time2/time2_again:.1f}x")
    
    start = time.time()
    result3_again = metric3.calculate(input_text, output_text, references)
    time3_again = time.time() - start
    print(f"  Metric3 (large, 0.2): {result3_again:.4f} in {time3_again:.4f}s - Speedup: {time3/time3_again:.1f}x")
    
    # Test metrics with different non-affecting parameters (should use same cache)
    print("\n3. Testing metrics with different non-affecting parameters (should use same cache):")
    
    start = time.time()
    result4 = metric4.calculate(input_text, output_text, references)
    time4 = time.time() - start
    print(f"  Metric4 (large, 0.5, debug=True): {result4:.4f} in {time4:.4f}s")
    
    start = time.time()
    result5 = metric5.calculate(input_text, output_text, references)
    time5 = time.time() - start
    print(f"  Metric5 (large, 0.5, debug=False): {result5:.4f} in {time5:.4f}s")
    
    # Verify different parameters produce different results
    print("\n4. Verifying different parameters produce different results:")
    print(f"  Metric1 (large, 0.5): {result1:.4f}")
    print(f"  Metric2 (small, 0.5): {result2:.4f}")
    print(f"  Metric3 (large, 0.2): {result3:.4f}")
    
    assert result1 != result2, "Different model_types should produce different results"
    assert result1 != result3, "Different thresholds should produce different results"
    assert result1 == result4 == result5, "Metrics with same affecting parameters should produce same results regardless of debug flag"
    
    print("\nSuccess! Caching works correctly with initialization parameters!")
    print("All parameters affect caching by default, and non-affecting parameters are explicitly excluded.")


if __name__ == "__main__":
    demonstrate_caching() 