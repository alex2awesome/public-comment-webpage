#!/usr/bin/env python
"""
Test script for verifying caching with different initialization parameters.

This script ensures metrics with different initialization parameters use separate caches,
and that parameters are automatically included in the cache key without explicit registration.
"""

import pytest
import time
import os
import shutil
from typing import Callable

from autometrics.metrics.reference_based.BERTScore import BERTScore

@pytest.fixture(scope="module")
def setup_and_teardown(test_cache_dir):
    """Setup and teardown for cache testing"""
    # Clean any existing cache before tests
    if os.path.exists(test_cache_dir):
        shutil.rmtree(test_cache_dir)
    
    yield  # Run the tests
    
    # Clean up after tests
    if os.path.exists(test_cache_dir):
        shutil.rmtree(test_cache_dir)

@pytest.fixture
def test_data():
    """Provide test data for the metrics"""
    input_text = "The cat sat on the mat."
    output_text = "A cat was sitting on a mat."
    references = ["The feline was on the carpet.", "A cat was resting on the mat."]
    return input_text, output_text, references

def measure_execution_time(func: Callable, *args, **kwargs):
    """Measure the execution time of a function"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

@pytest.mark.slow
def test_different_param_caching(setup_and_teardown, test_data, test_cache_dir):
    """Test caching for metrics with different initialization parameters"""
    input_text, output_text, references = test_data
    
    # Create two BERTScore metrics with different models
    metric1 = BERTScore(model="distilbert-base-uncased", use_cache=True, cache_dir=test_cache_dir)
    metric2 = BERTScore(model="roberta-base", use_cache=True, cache_dir=test_cache_dir)
    
    # First run with metric1 (cache miss)
    result1, time1 = measure_execution_time(
        metric1.calculate, input_text, output_text, references
    )
    
    # Second run with metric1 should use cache (cache hit)
    result2, time2 = measure_execution_time(
        metric1.calculate, input_text, output_text, references
    )
    
    # Cache hit should be faster
    assert time2 < time1, "Cache hit should be faster than cache miss"
    
    # First run with metric2 should be a cache miss despite same input
    result3, time3 = measure_execution_time(
        metric2.calculate, input_text, output_text, references
    )
    
    # Second run with metric2 should be a cache hit
    result4, time4 = measure_execution_time(
        metric2.calculate, input_text, output_text, references
    )
    
    # Cache hit should be faster for second metric too
    assert time4 < time3, "Cache hit should be faster than cache miss for the second metric"
    
    # Verify results are different between the two metrics (different models)
    assert result1 != result3, "Results should be different with different models"
    assert result1 == result2, "Results should be the same for same metric"
    assert result3 == result4, "Results should be the same for same metric"

    print("\nCACHING TEST PASSED: Different initialization parameters automatically use separate caches!")
    print("No explicit parameter registration needed - model parameter was automatically included in cache key.")


if __name__ == "__main__":
    pytest.main() 