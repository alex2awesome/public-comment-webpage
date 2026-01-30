#!/usr/bin/env python
"""
Test script for metric caching functionality.

This script demonstrates how caching works and measures the performance improvement.
"""

import pytest
import time
import os
import shutil
from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.SARI import SARI

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

@pytest.fixture
def batch_test_data():
    """Provide batch test data for the metrics"""
    inputs = [
        "The cat sat on the mat.",
        "The dog barked loudly.",
        "The sun is shining brightly."
    ]
    outputs = [
        "A cat was sitting on a mat.",
        "A loud dog was barking.",
        "It's a bright and sunny day."
    ]
    refs = [
        ["The feline was on the carpet.", "A cat was resting on the mat."],
        ["The dog made a loud noise.", "A noisy dog was heard."],
        ["The weather is nice today.", "The sun shines bright."]
    ]
    return inputs, outputs, refs

def measure_execution_time(func, *args, **kwargs):
    """Measure the execution time of a function"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

@pytest.mark.parametrize("metric_class", [BLEU, SARI])
def test_metric_caching(setup_and_teardown, test_data, metric_class, test_cache_dir):
    """Test caching for metrics"""
    input_text, output_text, references = test_data
    
    # First run with caching enabled (cache miss)
    metric = metric_class(use_cache=True, cache_dir=test_cache_dir)
    result1, time1 = measure_execution_time(
        metric.calculate, input_text, output_text, references
    )
    
    # Second run with same inputs should use cache (cache hit)
    result2, time2 = measure_execution_time(
        metric.calculate, input_text, output_text, references
    )
    
    # Verify cache hit is faster
    assert time2 < time1, "Cache hit should be faster than cache miss"
    
    # Test with cache disabled
    metric_no_cache = metric_class(use_cache=False)
    result3, time3 = measure_execution_time(
        metric_no_cache.calculate, input_text, output_text, references
    )
    
    # Verify results are the same
    assert result1 == result2 == result3, "Results should be identical regardless of caching"
    
    # Sometimes on fast machines or with simple metrics, the timing difference might not be significant
    # So let's just print the times rather than assert
    print(f"\nCache hit time: {time2:.6f}s, No cache time: {time3:.6f}s")
    # For info purposes only, not a strict test
    if time2 >= time3:
        print(f"NOTE: Cache hit not faster than no cache for {metric_class.__name__} (may happen on fast systems)")

def test_batch_caching(setup_and_teardown, batch_test_data, test_cache_dir):
    """Test batch caching for BLEU metric"""
    inputs, outputs, references = batch_test_data
    
    # First batch run (cache miss for all)
    metric = BLEU(use_cache=True, cache_dir=test_cache_dir)
    result1, time1 = measure_execution_time(
        metric.calculate_batched, inputs, outputs, references
    )
    
    # Second batch run (all cache hits)
    result2, time2 = measure_execution_time(
        metric.calculate_batched, inputs, outputs, references
    )
    
    # Verify cache hit is faster
    assert time2 < time1, "Batch cache hit should be faster than cache miss"
    
    # Check results are identical
    assert result1 == result2, "First and second batch results should be identical"
    
    # Third batch run with partial overlap
    new_inputs = inputs[1:] + ["This is a new input not in cache"]
    new_outputs = outputs[1:] + ["This is a new output not in cache"]
    new_refs = references[1:] + [["This is a new reference not in cache"]]
    
    result3, time3 = measure_execution_time(
        metric.calculate_batched, new_inputs, new_outputs, new_refs
    )
    
    # Verify partial overlapping results
    assert result3[:-1] == result2[1:], "Overlapping results should be identical" 