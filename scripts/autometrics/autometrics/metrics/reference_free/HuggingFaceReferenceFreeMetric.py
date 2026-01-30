from evaluate import load
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric
import torch

class HuggingFaceReferenceFreeMetric(ReferenceFreeMetric):
    """
    Generic wrapper for HuggingFace Evaluate reference-free metrics that return a single score per input.
    """
    def __init__(
        self,
        name: str,
        description: str,
        metric_id: str,
        score_key: str = None,
        load_kwargs: dict = None,
        compute_kwargs: dict = None,
        **kwargs
    ):
        # Pass ALL parameters to parent constructor
        super().__init__(
            name=name,
            description=description,
            metric_id=metric_id,
            score_key=score_key,
            load_kwargs=load_kwargs,
            compute_kwargs=compute_kwargs,
            **kwargs
        )
        self.metric_id = metric_id
        self.score_key = score_key
        self.load_kwargs = load_kwargs or {}
        self.compute_kwargs = compute_kwargs or {}
        self.metric = None

    def _coerce_metric_model_float32(self):
        """
        Best-effort: if the underlying HF evaluate metric exposes a model/pipeline,
        move it to CPU and cast weights and buffers to float32 to avoid BF16/FP32
        matmul dtype mismatches on CPU.
        """
        try:
            m = None
            cand_names = [
                'model', '_model', 'classifier', '_classifier', 'pipeline', '_pipeline'
            ]
            for name in cand_names:
                if hasattr(self.metric, name):
                    obj = getattr(self.metric, name)
                    # If it's a pipeline, try to get its model
                    try:
                        from transformers import Pipeline
                        if isinstance(obj, Pipeline) and hasattr(obj, 'model'):
                            m = obj.model
                            break
                    except Exception:
                        pass
                    # Otherwise if it's a module
                    try:
                        import torch.nn as nn
                        if isinstance(obj, nn.Module):
                            m = obj
                            break
                    except Exception:
                        pass
            if m is None:
                return
            # Move to CPU and cast to float32
            try:
                m.to('cpu')
            except Exception:
                pass
            for p in m.parameters(recurse=True):
                if p.dtype.is_floating_point:
                    p.data = p.data.float()
            for b_name, b in list(m.named_buffers(recurse=True)):
                try:
                    if hasattr(b, 'dtype') and b.dtype.is_floating_point:
                        m._buffers[b_name] = b.float()  # type: ignore[index]
                except Exception:
                    pass
            try:
                m.eval()
            except Exception:
                pass
        except Exception:
            # Best-effort only
            pass

    def _load_metric(self):
        if self.metric is None:
            try:
                # First attempt: try loading with provided kwargs
                self.metric = load(self.metric_id, **self.load_kwargs)
                # Normalize dtype to float32 on CPU if possible
                self._coerce_metric_model_float32()
            except NotImplementedError as e:
                # Handle meta tensor issue
                if "Cannot copy out of meta tensor" in str(e):
                    print(f"    🔧 Meta tensor issue detected for {self.metric_id}, trying CPU fallback...")
                    # Force CPU usage and try again
                    cpu_kwargs = self.load_kwargs.copy()
                    cpu_kwargs["device"] = "cpu"
                    try:
                        self.metric = load(self.metric_id, **cpu_kwargs)
                        print(f"    ✅ Successfully loaded {self.metric_id} on CPU")
                        self._coerce_metric_model_float32()
                    except Exception as e2:
                        print(f"    ❌ Failed to load {self.metric_id} even on CPU: {e2}")
                        # Try one more time with no device specification at all
                        try:
                            no_device_kwargs = self.load_kwargs.copy()
                            if "device" in no_device_kwargs:
                                del no_device_kwargs["device"]
                            self.metric = load(self.metric_id, **no_device_kwargs)
                            print(f"    ✅ Successfully loaded {self.metric_id} without device specification")
                            self._coerce_metric_model_float32()
                        except Exception as e3:
                            print(f"    ❌ Failed to load {self.metric_id} without device specification: {e3}")
                            raise e3
                else:
                    raise e
            except Exception as e:
                print(f"    ❌ Failed to load {self.metric_id}: {e}")
                raise e

    def _calculate_impl(self, input_text: str, output: str, references=None, **kwargs) -> float:
        self._load_metric()
        # single prediction
        compute_args = {**self.compute_kwargs, **kwargs, 'predictions': [output]}
        # Ensure no autocast mixes dtypes on CPU
        try:
            ctx = torch.autocast('cpu', enabled=False)
        except Exception:
            class _Noop:
                def __enter__(self):
                    return self
                def __exit__(self, *exc):
                    return False
            ctx = _Noop()
        with ctx:
            result = self.metric.compute(**compute_args)
        val = result.get(self.score_key)
        # If list, take first element
        if isinstance(val, (list, tuple)):
            return float(val[0])
        return float(val)

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs) -> list:
        self._load_metric()
        scores = []
        # Try vectorized compute for first two items
        if len(outputs) >= 2:
            try:
                small_preds = outputs[:2]
                small_args = {**self.compute_kwargs, **kwargs, 'predictions': small_preds}
                try:
                    ctx = torch.autocast('cpu', enabled=False)
                except Exception:
                    class _Noop:
                        def __enter__(self):
                            return self
                        def __exit__(self, *exc):
                            return False
                    ctx = _Noop()
                with ctx:
                    small_res = self.metric.compute(**small_args)
                val_small = small_res.get(self.score_key)
                if isinstance(val_small, (list, tuple)) and len(val_small) == 2:
                    # add first two
                    scores.extend([float(v) for v in val_small])
                    # compute for remainder
                    if len(outputs) > 2:
                        rest_preds = outputs[2:]
                        rest_args = {**self.compute_kwargs, **kwargs, 'predictions': rest_preds}
                        with ctx:
                            rest_res = self.metric.compute(**rest_args)
                        val_rest = rest_res.get(self.score_key)
                        if isinstance(val_rest, (list, tuple)) and len(val_rest) == len(rest_preds):
                            scores.extend([float(v) for v in val_rest])
                        else:
                            # fallback for each remaining
                            for i, out in enumerate(rest_preds):
                                # Use the parent's calculate method to leverage caching
                                scores.append(super().calculate(inputs[i+2] if i+2 < len(inputs) else None, out, None, **kwargs))
                    return scores
            except Exception:
                pass
        # Fallback to per-sample for entire batch
        scores = []
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            # Use the parent's calculate method to leverage caching
            scores.append(super().calculate(inp, out, None, **kwargs))
        return scores 