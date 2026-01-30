from abc import ABC, abstractmethod
import os
import hashlib
import re
from diskcache import Cache
from functools import wraps
from typing import Any, List, Optional, Union, Tuple, Dict, ClassVar
from dataclasses import dataclass

def _get_cache_dir() -> str:
    """
    Get the cache directory from environment variable AUTOMETRICS_CACHE_DIR,
    with fallback to "./autometrics_cache" if not set.
    
    Returns:
        Cache directory path as string
    """
    return os.environ.get("AUTOMETRICS_CACHE_DIR", "./autometrics_cache")


def make_safe_path_component(name: str, max_length: int = 96) -> str:
    """Return a filesystem-friendly slug capped at *max_length* characters."""
    if not isinstance(name, str):
        name = str(name)
    cleaned = re.sub(r"[^\w\-.]+", "_", name).strip("._") or "metric"
    if len(cleaned) <= max_length:
        return cleaned
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    prefix_len = max(8, max_length - len(digest) - 1)
    return f"{cleaned[:prefix_len]}_{digest}"

@dataclass
class MetricResult:
    score: float
    feedback: str = ""


class Metric(ABC):
    """
    Abstract class for metrics
    """
    # Whether this metric produces non-empty feedback/rationale alongside scores
    has_feedback: ClassVar[bool] = False
    # Class-level default that subclasses can override
    DEFAULT_USE_CACHE = True
    
    def __init__(self, name, description, use_cache=None, seed=None, cache_dir=None, 
                 cache_size_limit=None, cache_ttl=None, force_cache=False, **kwargs):
        self.name = name
        self.description = description
        
        # Use the class-specific default if use_cache is not explicitly provided
        if use_cache is None:
            use_cache = self.DEFAULT_USE_CACHE
        
        # Use environment variable or default for cache_dir if not provided
        if cache_dir is None:
            cache_dir = _get_cache_dir()
        
        self.use_cache = use_cache
        self.seed = seed
        
        # Store all parameters directly from explicit kwargs
        self._init_params = {
            **kwargs,
            'name': name,
            'description': description,
            'use_cache': use_cache,
            'seed': seed,
            'cache_dir': cache_dir,
            'cache_size_limit': cache_size_limit,
            'cache_ttl': cache_ttl
        }
        
        # Parameters explicitly excluded from cache key
        # 'name' and 'description' don't affect the metric's output, just its labeling
        # 'use_cache', 'cache_dir', 'cache_size_limit', and 'cache_ttl' are cache configuration, not metric behavior
        self._excluded_params = set(['name', 'description', 'use_cache', 'cache_dir', 
                                   'cache_size_limit', 'cache_ttl'])
        
        # Set up cache if enabled
        self._cache = None
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            seed_suffix = f"_{seed}" if seed is not None else ""
            safe_component = make_safe_path_component(name)
            cache_path = os.path.join(cache_dir, f"{safe_component}{seed_suffix}")
            # Ensure cache_size_limit and cache_ttl have appropriate default values
            size_limit = cache_size_limit if cache_size_limit is not None else 10e9  # Default to 10GB
            ttl = cache_ttl if cache_ttl is not None else 0  # Default to no expiration (0)
            try:
                self._cache = Cache(
                    cache_path,
                    size_limit=size_limit,  # Uses LRU eviction when limit is reached
                    timeout=ttl             # Time-based expiration
                )
            except Exception as e:
                print(f"Error initializing cache: {e}")
                self._cache = None

                if force_cache:
                    raise e

    def exclude_from_cache_key(self, *param_names):
        """
        Mark specific initialization parameters to be excluded from the cache key.
        Call this method in subclasses' __init__ for parameters that DON'T affect the metric's result.
        For example, debug flags, verbosity settings, etc.
        """
        self._excluded_params.update(param_names)

    def _make_hashable(self, obj):
        """Convert an object to a hashable representation"""
        if isinstance(obj, list):
            if len(obj) == 0:
                return "[]"
            # If it's a list of strings, sort them
            if all(isinstance(x, str) for x in obj):
                return tuple(sorted(obj))
            # Otherwise convert each item and sort
            return tuple(sorted(self._make_hashable(x) for x in obj))
        elif isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        else:
            return str(obj)
    
    def _make_cache_key(self, method_name, *args, **kwargs):
        """Create a deterministic cache key from method arguments"""
        components = [method_name]
        
        # Add instance-specific initialization parameters to the key (excluding those marked)
        for k, v in sorted(self._init_params.items()):
            if k not in self._excluded_params:
                components.append(f"init_{k}={self._make_hashable(v)}")
        
        # Add args
        for arg in args:
            components.append(self._make_hashable(arg))
        
        # Add kwargs
        for k, v in sorted(kwargs.items()):
            components.append(f"{k}={self._make_hashable(v)}")
        
        # Create hash
        key_str = "_".join(str(c) for c in components)
        return hashlib.md5(key_str.encode()).hexdigest()

    @abstractmethod
    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Actual implementation of calculate - to be implemented by subclasses
        """
        pass
        
    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """
        Default implementation of batch calculation - calls _calculate_impl for each item.
        Subclasses should override this method if they can implement batch calculation more efficiently.
        """
        if references is None:
            references = [None] * len(inputs)
        
        results = []
        for i, o, r in zip(inputs, outputs, references):
            results.append(self._calculate_impl(i, o, r, **kwargs))
        
        return results

    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate the metric with automatic caching
        """
        # Skip caching if disabled
        if not self.use_cache or self._cache is None:
            return self._calculate_impl(input, output, references, **kwargs)
        
        # Generate cache key
        key = self._make_cache_key('calculate', input, output, references, **kwargs)
        
        # Try to get from cache
        result = self._cache.get(key)
        
        # If not in cache, compute and store
        if result is None:
            # Wrap the calculation in a try/except to avoid caching exceptions
            try:
                result = self._calculate_impl(input, output, references, **kwargs)
                # Only cache the result if no exception occurred
                self._cache[key] = result
            except Exception as e:
                # Re-raise the exception without caching it
                raise e
        
        return result

    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate the metric for a batch of inputs and outputs with caching.
        """
        # Skip caching if disabled
        if not self.use_cache or self._cache is None:
            return self._calculate_batched_impl(inputs, outputs, references, **kwargs)
        
        # Prepare references
        if references is None:
            refs = [None] * len(inputs)
        else:
            refs = references
        
        # Check cache for each individual item
        results = []
        missing_indices = []
        missing_inputs = []
        missing_outputs = []
        missing_refs = []
        
        for i, (inp, out, ref) in enumerate(zip(inputs, outputs, refs)):
            # Try to get from cache
            key = self._make_cache_key('calculate', inp, out, ref, **kwargs)
            cached_result = self._cache.get(key)
            
            if cached_result is not None:
                results.append(cached_result)
            else:
                # Mark for computation
                results.append(None)  # Placeholder
                missing_indices.append(i)
                missing_inputs.append(inp)
                missing_outputs.append(out)
                missing_refs.append(ref)
        
        # Calculate missing results if any
        if missing_indices:
            try:
                # Use batched implementation for missing items
                batch_refs = missing_refs if any(r is not None for r in missing_refs) else None
                missing_results = self._calculate_batched_impl(missing_inputs, missing_outputs, batch_refs, **kwargs)
                
                # Update cache and results
                for i, idx in enumerate(missing_indices):
                    inp = missing_inputs[i]
                    out = missing_outputs[i]
                    ref = missing_refs[i]
                    result = missing_results[i]
                    
                    # Only cache successful results
                    key = self._make_cache_key('calculate', inp, out, ref, **kwargs)
                    self._cache[key] = result
                    
                    # Update the results list
                    results[idx] = result
            except Exception as e:
                # If the batch calculation fails, don't cache any results
                # and re-raise the exception
                raise e
        
        return results

    def calculate_with_feedback(self, input, output, references=None, **kwargs):
        """
        Calculate the metric and return MetricResult(score, feedback).
        Uses separate cache namespace from calculate.
        """
        # Skip caching if disabled
        if not self.use_cache or self._cache is None:
            try:
                if hasattr(self, '_calculate_with_feedback_impl'):
                    res = self._calculate_with_feedback_impl(input, output, references, **kwargs)
                    if isinstance(res, MetricResult):
                        return res
                    # Fallback if subclass returned tuple
                    try:
                        score, feedback = res
                        return MetricResult(float(score), str(feedback) if feedback is not None else "")
                    except Exception:
                        return MetricResult(float(res), "")
                else:
                    score = self._calculate_impl(input, output, references, **kwargs)
                    return MetricResult(float(score), "")
            except Exception as e:
                raise e

        key = self._make_cache_key('calculate_with_feedback', input, output, references, **kwargs)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Compute and cache
        if hasattr(self, '_calculate_with_feedback_impl'):
            res = self._calculate_with_feedback_impl(input, output, references, **kwargs)
            if not isinstance(res, MetricResult):
                try:
                    score, feedback = res
                    res = MetricResult(float(score), str(feedback) if feedback is not None else "")
                except Exception:
                    res = MetricResult(float(res), "")
        else:
            score = self._calculate_impl(input, output, references, **kwargs)
            res = MetricResult(float(score), "")
        self._cache[key] = res
        return res

    def calculate_batched_with_feedback(self, inputs, outputs, references=None, **kwargs):
        """
        Batched calculation returning a list of MetricResult, with caching per item similar to calculate_batched.
        """
        if not self.use_cache or self._cache is None:
            # No cache; try subclass batch impl if provided
            if hasattr(self, '_calculate_batched_with_feedback_impl'):
                return self._calculate_batched_with_feedback_impl(inputs, outputs, references, **kwargs)
            # Fallback: compute scores and wrap
            scores = self._calculate_batched_impl(inputs, outputs, references, **kwargs)
            return [MetricResult(float(s), "") for s in scores]

        # Prepare references
        if references is None:
            refs = [None] * len(inputs)
        else:
            refs = references

        results: List[Optional[MetricResult]] = []
        missing_indices: List[int] = []
        missing_inputs: List[Any] = []
        missing_outputs: List[Any] = []
        missing_refs: List[Any] = []

        for i, (inp, out, ref) in enumerate(zip(inputs, outputs, refs)):
            key = self._make_cache_key('calculate_with_feedback', inp, out, ref, **kwargs)
            cached = self._cache.get(key)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                missing_indices.append(i)
                missing_inputs.append(inp)
                missing_outputs.append(out)
                missing_refs.append(ref)

        if missing_indices:
            if hasattr(self, '_calculate_batched_with_feedback_impl'):
                batch_refs = missing_refs if any(r is not None for r in missing_refs) else None
                missing_results = self._calculate_batched_with_feedback_impl(missing_inputs, missing_outputs, batch_refs, **kwargs)
            else:
                # Fallback via score-only batched + wrap
                batch_refs = missing_refs if any(r is not None for r in missing_refs) else None
                scores = self._calculate_batched_impl(missing_inputs, missing_outputs, batch_refs, **kwargs)
                missing_results = [MetricResult(float(s), "") for s in scores]

            for local_idx, global_idx in enumerate(missing_indices):
                res = missing_results[local_idx]
                key = self._make_cache_key('calculate_with_feedback', missing_inputs[local_idx], missing_outputs[local_idx], missing_refs[local_idx], **kwargs)
                self._cache[key] = res
                results[global_idx] = res

        return results

    @abstractmethod
    def predict(self, dataset, update_dataset=True, **kwargs):
        """
        Calculate the metric for the dataset
        """
        pass

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description
    
    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return self.__str__()
