from autometrics.metrics.Metric import Metric
from autometrics.metrics.Metric import MetricResult

class MultiMetric(Metric):
    """
    Abstract class for metrics that return multiple values
    """
    def __init__(self, name, description, submetric_names=[], **kwargs) -> None:
        super().__init__(name, description, **kwargs)
        self.submetric_names = submetric_names

    def _calculate_impl(self, input, output, references=None, **kwargs):
        """
        Actual implementation of the metric calculation
        """
        pass

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate the metric for a batch of inputs and outputs. The default implementation simply calls _calculate_impl for each input/output pair.
        Override this method if you can calculate the metric more efficiently for a batch of inputs/outputs.
        """
        if references is None:
            references = [None] * len(inputs)

        results = []
        for i, o, r in zip(inputs, outputs, references):
            results.append(self._calculate_impl(i, o, r, **kwargs))

        # Swap the indices so that each submetric has its own list
        results = list(zip(*results))

        return results

    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        """
        Efficient batched version that is aware of *multi*-metric output shapes **and** integrates
        seamlessly with the caching logic from ``Metric``.

        Returns a list whose outer dimension enumerates sub-metrics and whose inner dimension
        enumerates the examples (``[submetric][example]``), mirroring what ``_calculate_batched_impl``
        produces for a cache-miss batch.
        """
        # Prepare references
        if references is None:
            references = [None] * len(inputs)
        if not (len(inputs) == len(outputs) == len(references)):
            raise ValueError("Length of inputs, outputs, and references must match")

        n_examples = len(inputs)
        expected_submetrics = len(self.submetric_names) if self.submetric_names else None
        n_submetrics = expected_submetrics  # will stay None if unknown

        # Allocate result container: results_by_submetric[sub_idx][example_idx]
        # We'll fill with None placeholders and overwrite when values become available.
        results_by_submetric = None  # allocate lazily when we know n_submetrics

        # Track indices that still need to be computed
        missing_indices = []
        missing_inputs, missing_outputs, missing_refs = [], [], []

        # First pass: try cache for every example
        for idx, (inp, out, ref) in enumerate(zip(inputs, outputs, references)):
            key = self._make_cache_key('calculate', inp, out, ref, **kwargs)
            cached = self._cache.get(key) if (self.use_cache and self._cache is not None) else None

            cache_valid = False
            if cached is not None:
                if isinstance(cached, (list, tuple)):
                    # Check that cached length matches expected submetric count (if known)
                    if expected_submetrics is None or len(cached) == expected_submetrics:
                        cache_valid = True

            if cache_valid:
                # Lazy allocation if first valid cache hit
                if results_by_submetric is None:
                    n_submetrics = len(cached)
                    results_by_submetric = [[None] * n_examples for _ in range(n_submetrics)]

                for s_idx, val in enumerate(cached):
                    results_by_submetric[s_idx][idx] = val
            else:
                # Remove malformed cache to avoid future issues
                if cached is not None and self.use_cache and self._cache is not None:
                    self._cache.pop(key, None)
                missing_indices.append(idx)
                missing_inputs.append(inp)
                missing_outputs.append(out)
                missing_refs.append(ref)

        # If nothing allocated yet (all cache misses), allocate now using known submetric count later
        if results_by_submetric is None:
            # We still don't know n_submetrics – we'll discover after computing
            pass  # postpone allocation

        # Compute missing results, if any
        if missing_indices:
            batch_refs = missing_refs if any(r is not None for r in missing_refs) else None
            # Call subclass implementation once for all missing examples
            missing_results_by_submetric = self._calculate_batched_impl(
                missing_inputs, missing_outputs, batch_refs, **kwargs)

            # Determine n_submetrics if still unknown
            if results_by_submetric is None:
                n_submetrics = len(missing_results_by_submetric)
                results_by_submetric = [[None] * n_examples for _ in range(n_submetrics)]

            # Sanity check dimensions
            if n_submetrics is not None and len(missing_results_by_submetric) != n_submetrics:
                raise ValueError("Unexpected number of sub-metrics returned by _calculate_batched_impl")
            for sub_list in missing_results_by_submetric:
                if len(sub_list) != len(missing_indices):
                    raise ValueError("_calculate_batched_impl must return len(missing_indices) results per sub-metric")

            # Write computed values into result container and cache them example-wise
            for local_idx, global_idx in enumerate(missing_indices):
                # Collect values for this example across submetrics
                example_values = [missing_results_by_submetric[s][local_idx] for s in range(n_submetrics)]

                # Cache
                if self.use_cache and self._cache is not None:
                    key = self._make_cache_key('calculate', missing_inputs[local_idx], missing_outputs[local_idx], missing_refs[local_idx], **kwargs)
                    self._cache[key] = example_values

                # Fill results_by_submetric
                for s_idx, val in enumerate(example_values):
                    results_by_submetric[s_idx][global_idx] = val

        # Final safety: no None should remain
        for sub_idx, sub_list in enumerate(results_by_submetric):
            if any(v is None for v in sub_list):
                raise RuntimeError(f"Internal error: sub-metric {sub_idx} has unset results after batching")

        return results_by_submetric

    def calculate_batched_with_feedback(self, inputs, outputs, references=None, **kwargs):
        """
        Batched variant returning feedback for each submetric per example.
        Shape: List[List[MetricResult]] => [submetric][example].
        """
        # Compute using subclass batch with-feedback if available
        if hasattr(self, '_calculate_batched_with_feedback_impl'):
            return self._calculate_batched_with_feedback_impl(inputs, outputs, references, **kwargs)

        # Default: reuse score-only path and wrap
        results_by_submetric = self.calculate_batched(inputs, outputs, references, **kwargs)
        wrapped = []
        for sub_list in results_by_submetric:
            wrapped.append([MetricResult(float(v), "") for v in sub_list])
        return wrapped

    def predict(self, dataset, update_dataset=True, with_feedback: bool = True, **kwargs):
        """
        Calculate the metric for the dataset
        """
        df = dataset.get_dataframe()
        
        # Use the appropriate method based on metric type
        if hasattr(self, 'calculate_row'):
            # For specific implementations (ReferenceBasedMultiMetric, etc.)
            results_list = []
            for _, row in df.iterrows():
                results = self.calculate_row(row, dataset, False, **kwargs)
                results_list.append(results)
            
            # Transpose results to get one list per submetric
            results_by_submetric = list(zip(*results_list))
            
            if update_dataset:
                for i, submetric_name in enumerate(self.submetric_names):
                    df[submetric_name] = results_by_submetric[i]
                    if submetric_name not in dataset.get_metric_columns():
                        dataset.get_metric_columns().append(submetric_name)
                
                dataset.set_dataframe(df)
            
            return results_list
        else:
            # Generic implementation
            input_column = dataset.get_input_column()
            output_column = dataset.get_output_column()
            
            inputs = df[input_column].values.tolist()
            outputs = df[output_column].values.tolist()
            
            # Determine if we need references
            references = None
            reference_columns = dataset.get_reference_columns()
            if reference_columns:
                references = df[reference_columns].values.tolist()
            
            if with_feedback and getattr(self, 'has_feedback', False):
                results_wrapped = self.calculate_batched_with_feedback(inputs, outputs, references, **kwargs)
                # Unwrap scores and gather feedback
                results = []
                feedback_cols = {}
                for i, submetric_name in enumerate(self.submetric_names):
                    scores_i = [mr.score for mr in results_wrapped[i]]
                    if update_dataset and getattr(self, 'has_feedback', False):
                        feedback_cols[f"{submetric_name}__feedback"] = [mr.feedback for mr in results_wrapped[i]]
                    results.append(scores_i)
            else:
                results = self.calculate_batched(inputs, outputs, references, **kwargs)
            
            if update_dataset:
                for i, submetric_name in enumerate(self.submetric_names):
                    df[submetric_name] = results[i]
                    if submetric_name not in dataset.get_metric_columns():
                        dataset.get_metric_columns().append(submetric_name)
                if with_feedback and getattr(self, 'has_feedback', False):
                    for col, vals in (feedback_cols if 'feedback_cols' in locals() else {}).items():
                        df[col] = vals
                
                dataset.set_dataframe(df)
            
            return results

    def get_submetric_names(self):
        return self.submetric_names