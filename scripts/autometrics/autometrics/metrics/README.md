# Autometrics Metrics Implementation Guide

This guide outlines the key considerations and best practices for implementing new metrics in the Autometrics library. The metrics system includes a robust caching mechanism and standardized interfaces for both reference-based and reference-free metrics.

## Core Concepts

### Metric Base Classes

The library provides several base classes for implementing metrics:

- `Metric`: The abstract base class for all metrics
- `ReferenceBasedMetric`: For metrics that require reference texts
- `ReferenceFreeMetric`: For metrics that only use input and output
- `ReferenceBasedMultiMetric`: For reference-based metrics that return multiple scores
- `ReferenceFreeMultiMetric`: For reference-free metrics that return multiple scores

### Caching System

The metrics system includes an automatic caching mechanism that:

1. Caches computation results based on:
   - All initialization parameters (except those explicitly excluded)
   - Input text
   - Output text
   - Reference texts (if applicable)
   - Any additional kwargs passed to calculate methods

2. Cache Configuration:
   - By default, caching is enabled (`DEFAULT_USE_CACHE = True`)
   - Cache location is configurable via `cache_dir`
   - Cache size limits and TTL can be set via `cache_size_limit` and `cache_ttl`
   - Uses LRU eviction when size limits are reached
   - Thread-safe by design using diskcache's built-in mechanisms

3. Cache Key Generation:
   - Automatically includes all initialization parameters by default
   - Excludes non-affecting parameters (name, description, cache settings)
   - Uses deterministic hashing for consistent keys

## Implementing a New Metric

### Basic Implementation

```python
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

class MyNewMetric(ReferenceBasedMetric):
    def __init__(self, name="MyNewMetric", description="Description of my metric", 
                 param1=None, param2=None, **kwargs):
        # Pass ALL parameters to parent constructor for proper caching
        super().__init__(
            name=name,
            description=description,
            param1=param1,  # Parameters that affect results
            param2=param2,  # Parameters that affect results
            **kwargs
        )
        
        # Store parameters as instance variables
        self.param1 = param1
        self.param2 = param2
        
        # Exclude any parameters that don't affect results
        self.exclude_from_cache_key('debug_flag', 'verbose')
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Implement the actual metric calculation.
        This method is called by the caching wrapper.
        """
        # Your implementation here
        return score
```

### Key Implementation Points

1. **Constructor Parameters**:
   - Pass ALL parameters to parent constructor
   - Even parameters that don't affect results should be passed
   - Use `exclude_from_cache_key()` for non-affecting parameters

2. **Cache-Affecting Parameters**:
   - All initialization parameters affect caching by default
   - Exclude parameters that don't affect results:
     - Debug flags
     - Verbosity settings
     - Cache configuration (size_limit, ttl)
     - Dataset objects
     - API endpoints/keys

3. **Calculation Methods**:
   - Implement `_calculate_impl` for single-item calculation
   - Optionally override `_calculate_batched_impl` for efficient batch processing
   - The caching wrapper handles cache management automatically

4. **Exception Handling**:
   - Exceptions are automatically handled and not cached
   - Failed calculations will be retried on next call
   - Original exceptions are preserved and re-raised

### Example: Reference-Based Metric

```python
class MyReferenceMetric(ReferenceBasedMetric):
    def __init__(self, model_name="default", threshold=0.5, debug=False, **kwargs):
        super().__init__(
            name="MyReferenceMetric",
            description="A reference-based metric example",
            model_name=model_name,
            threshold=threshold,
            debug=debug,
            **kwargs
        )
        
        self.model_name = model_name
        self.threshold = threshold
        self.debug = debug
        
        # Exclude debug flag from cache key
        self.exclude_from_cache_key('debug')
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        if self.debug:
            print(f"Computing with model: {self.model_name}")
        
        # Your implementation here
        score = compute_score(input, output, references[0])
        
        # Apply threshold
        if score < self.threshold:
            score = 0.0
            
        return score
```

### Example: Multi-Score Metric

```python
class MyMultiMetric(ReferenceBasedMultiMetric):
    def __init__(self, model_name="default", **kwargs):
        super().__init__(
            name="MyMultiMetric",
            description="A multi-score metric example",
            model_name=model_name,
            submetric_names=["score1", "score2", "score3"],
            **kwargs
        )
        
        self.model_name = model_name
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        # Return multiple scores as a tuple
        return score1, score2, score3
```

## Best Practices

1. **Parameter Management**:
   - Always pass all parameters to parent constructor
   - Document which parameters affect results
   - Explicitly exclude non-affecting parameters

2. **Caching Considerations**:
   - Don't cache exceptions
   - Consider memory usage for large inputs
   - Use batch processing for efficiency
   - Set appropriate cache size limits

3. **Thread Safety**:
   - The caching system is thread-safe by default
   - Don't modify shared state in calculation methods
   - Use instance variables for configuration

4. **Performance**:
   - Override `_calculate_batched_impl` for efficient batch processing
   - Consider using batch processing for LLM-based metrics
   - Set appropriate cache size limits for memory management

5. **Error Handling**:
   - Let exceptions propagate naturally
   - Don't catch and cache exceptions
   - Document expected exceptions

## Common Pitfalls

1. **Forgetting to Pass Parameters**:
   ```python
   # Wrong
   super().__init__(name, description)
   
   # Correct
   super().__init__(name=name, description=description, param1=param1, **kwargs)
   ```

2. **Not Excluding Non-Affecting Parameters**:
   ```python
   # Wrong
   # debug flag affects cache key unnecessarily
   
   # Correct
   self.exclude_from_cache_key('debug')
   ```

3. **Modifying Shared State**:
   ```python
   # Wrong
   def _calculate_impl(self, input, output, references=None, **kwargs):
       self.counter += 1  # Modifying instance state during calculation
       self.last_result = compute()  # Storing results in instance variables
       return self.last_result
   
   # Correct
   def _calculate_impl(self, input, output, references=None, **kwargs):
       # Use local variables only
       result = compute()
       return result
   ```

   Important notes about state management:
   - Instance variables should only be used for configuration (set in __init__)
   - Never modify instance variables during calculation methods
   - Don't store calculation results in instance variables
   - Use local variables for all computation state
   - The caching system is thread-safe, but modifying instance state during calculation is not

4. **Caching Exceptions**:
   ```python
   # Wrong
   try:
       result = compute()
       self._cache[key] = result
   except Exception as e:
       self._cache[key] = e  # Don't cache exceptions!
   
   # Correct
   # Let the base class handle exceptions
   ```

## Testing

When implementing a new metric, ensure you test:

1. Caching behavior with different parameters
2. Exception handling
3. Batch processing
4. Thread safety
5. Memory usage with large inputs
6. Cache size limits and TTL

## Contributing

When contributing new metrics:

1. Follow the implementation patterns above
2. Document all parameters and their effects
3. Include example usage
4. Add appropriate tests
5. Consider performance implications
6. Document any special considerations

Remember:
- The caching system will automatically handle caching based on parameters
- You don't need to implement your own caching
- Keep calculation methods stateless
- Let the base class handle exceptions
- Use the built-in batch processing if you don't need custom optimization

## References

- [diskcache Documentation](https://grantjenks.com/docs/diskcache/)
- [Metric Base Classes](../metrics/Metric.py)
- [Example Implementations](../metrics/reference_based/) 