from evaluate import load
from autometrics.metrics.reference_free.ReferenceFreeMultiMetric import ReferenceFreeMultiMetric
from typing import List

class HuggingFaceReferenceFreeMultiMetric(ReferenceFreeMultiMetric):
    """
    Generic wrapper for HuggingFace Evaluate reference-free multi-metrics.
    Allows loading any evaluate metric that returns multiple keys.
    """
    def __init__(
        self,
        name: str,
        description: str,
        metric_id: str,
        submetric_keys: List[str],
        load_kwargs: dict = None,
        compute_kwargs: dict = None,
        **kwargs
    ):
        # Pass ALL parameters to parent constructor
        super().__init__(
            name=name,
            description=description,
            submetric_names=submetric_keys,
            metric_id=metric_id,
            submetric_keys=submetric_keys,
            load_kwargs=load_kwargs,
            compute_kwargs=compute_kwargs,
            **kwargs
        )
        self.metric_id = metric_id
        self.submetric_keys = submetric_keys
        self.load_kwargs = load_kwargs or {}
        self.compute_kwargs = compute_kwargs or {}
        self.metric = None

    def _load_metric(self):
        if self.metric is None:
            self.metric = load(self.metric_id, **self.load_kwargs)

    def _calculate_impl(self, input_text: str, output: str, references=None, **kwargs):
        """
        Compute submetrics for a single output by calling the underlying HF evaluate metric.
        """
        self._load_metric()
        # Prepare single-item lists for compute
        preds = [output]
        args = {'predictions': preds}
        if references is not None:
            # Expect references as a list of references for this item
            refs = references if isinstance(references, list) and not isinstance(references[0], str) else [references]
            args['references'] = [refs]
        # Include any fixed compute kwargs
        args.update(self.compute_kwargs)
        result = self.metric.compute(**args)
        # Extract each submetric and return tuple
        values = []
        for key in self.submetric_keys:
            val = result.get(key)
            # Unpack single-element lists or return scalar
            if isinstance(val, (list, tuple)):
                values.append(float(val[0]))
            else:
                values.append(float(val))
        return tuple(values)

    def _calculate_batched_impl(self, inputs: List[str], outputs: List[str], references=None, **kwargs):
        """
        Batch evaluation with fallback: test on first 2 outputs; if vectorized runs (returns lists of length 2), apply to full batch,
        else fallback to per-sample calculate loop.
        """
        self._load_metric()
        
        # For empty batch, return empty list of tuples
        if not outputs:
            return []
            
        # Try vectorized compute for first two items to test if metric supports batching
        if len(outputs) >= 2:
            try:
                # Test with first two items
                test_preds = outputs[:2]
                args = {'predictions': test_preds}
                if references is not None:
                    if isinstance(references[0], list) and not isinstance(references[0][0], str):
                        # list of lists of references
                        test_refs = references[:2]
                        args['references'] = test_refs
                    else:
                        # single list of references
                        args['references'] = [references[:2]]
                args.update(self.compute_kwargs)
                args.update(kwargs)
                result = self.metric.compute(**args)
                
                # Check if result contains lists of length 2 for all keys
                all_batch_results = True
                for key in self.submetric_keys:
                    val = result.get(key)
                    if not isinstance(val, (list, tuple)) or len(val) != 2:
                        all_batch_results = False
                        break
                
                if all_batch_results:
                    # Metric supports batching, apply to full batch
                    full_args = {'predictions': outputs}
                    if references is not None:
                        if isinstance(references[0], list) and not isinstance(references[0][0], str):
                            full_args['references'] = references
                        else:
                            full_args['references'] = [references]
                    full_args.update(self.compute_kwargs)
                    full_args.update(kwargs)
                    full_result = self.metric.compute(**full_args)
                    
                    # Prepare list of tuples in same order as submetric_keys
                    batch_results = []
                    for i in range(len(outputs)):
                        item_values = []
                        for key in self.submetric_keys:
                            val = full_result.get(key)
                            item_values.append(float(val[i]))
                        batch_results.append(tuple(item_values))
                    return batch_results
            except Exception:
                # Vectorized approach failed, fallback to per-item
                pass
                
        # Fallback to per-sample calculation
        results = []
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            ref = None
            if references is not None:
                ref = references[i] if i < len(references) else None
            # Use the parent's calculate method to leverage caching
            results.append(super().calculate(inp, out, ref, **kwargs))
        return results 