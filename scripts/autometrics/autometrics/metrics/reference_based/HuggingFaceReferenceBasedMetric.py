from evaluate import load
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

class HuggingFaceReferenceBasedMetric(ReferenceBasedMetric):
    """
    Generic wrapper for HuggingFace Evaluate reference-based metrics.
    Loads the metric via evaluate.load and extracts a specified score key.
    """
    def __init__(
        self,
        name: str,
        description: str,
        metric_id: str,
        score_key: str = "score",
        load_kwargs: dict = None,
        **kwargs
    ):
        # Pass ALL parameters to parent constructor
        super().__init__(
            name=name,
            description=description,
            metric_id=metric_id,
            score_key=score_key,
            load_kwargs=load_kwargs,
            **kwargs
        )
        self.metric_id = metric_id
        self.score_key = score_key
        self.load_kwargs = load_kwargs or {}
        self.metric = None

    def _load_metric(self):
        if self.metric is None:
            self.metric = load(self.metric_id, **self.load_kwargs)

    def _calculate_impl(self, input: str, output: str, references=None, **kwargs):
        self._load_metric()
        # Expect references as list of strings
        refs = references if isinstance(references[0], list) or isinstance(references[0], tuple) else references
        result = self.metric.compute(predictions=[output], references=[refs], **kwargs)
        val = result.get(self.score_key)
        # broadcast scalar to single entry
        return float(val) if not isinstance(val, (list, tuple)) else float(val[0])

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        """
        Batch evaluation with probe: test first two samples for vectorization, then only compute remainder.
        """
        self._load_metric()
        refs = references if references is not None else [None] * len(outputs)
        scores = []
        # Try vectorized compute for first two items
        if len(outputs) >= 2:
            try:
                test_preds = outputs[:2]
                test_refs = refs[:2]
                result_small = self.metric.compute(predictions=test_preds, references=test_refs, **kwargs)
                val_small = result_small.get(self.score_key)
                if isinstance(val_small, (list, tuple)) and len(val_small) == 2:
                    # add first two
                    scores.extend([float(v) for v in val_small])
                    # compute for remainder
                    if len(outputs) > 2:
                        rest_preds = outputs[2:]
                        rest_refs = refs[2:]
                        result_rest = self.metric.compute(predictions=rest_preds, references=rest_refs, **kwargs)
                        val_rest = result_rest.get(self.score_key)
                        if isinstance(val_rest, (list, tuple)) and len(val_rest) == len(rest_preds):
                            scores.extend([float(v) for v in val_rest])
                        else:
                            # fallback for each remaining
                            for i, (out, ref) in enumerate(zip(rest_preds, rest_refs)):
                                # Use the parent's calculate method to leverage caching
                                scores.append(super().calculate(inputs[i+2] if i+2 < len(inputs) else None, out, ref, **kwargs))
                    return scores
            except Exception:
                pass
        # Fallback to per-sample for entire batch
        scores = []
        for i, (inp, out, ref) in enumerate(zip(inputs, outputs, refs)):
            # Use the parent's calculate method to leverage caching
            scores.append(super().calculate(inp, out, ref, **kwargs))
        return scores 