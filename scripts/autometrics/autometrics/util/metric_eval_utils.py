from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import gc

from autometrics.metrics.MultiMetric import MultiMetric
import dspy


def _unload_metric(metric) -> None:
    try:
        if hasattr(metric, '_unload_model') and callable(getattr(metric, '_unload_model')):
            metric._unload_model()
        elif hasattr(metric, '_unload_models') and callable(getattr(metric, '_unload_models')):
            metric._unload_models()
        else:
            for attr in ['model', 'tokenizer', 'qg', 'qa']:
                if not hasattr(metric, attr):
                    continue
                # Preserve dspy.LM instances to avoid losing LLM configuration
                if attr == 'model' and hasattr(metric, 'model'):
                    if isinstance(getattr(metric, 'model', None), dspy.LM):
                        print(f"[DEBUG][_unload_metric] Skipping clearing dspy.LM for {getattr(metric,'name', type(metric).__name__)}")
                        continue
                setattr(metric, attr, None)
    except Exception:
        pass


def _unload_all(metrics: List) -> None:
    for m in metrics:
        _unload_metric(m)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def evaluate_metric_instances(
    dataset,
    metric_instances: List,
    enable_parallel: bool = True,
    max_parallel_workers: int = 20,
    allowed_failed_metrics: int = 0,
):
    """
    Memory-safe evaluation of metric instances on a dataset.

    Two-wave strategy:
      1) Regular metrics (no device_map="auto") evaluated in parallel (if enabled), with OOM fallback queue.
      2) device_map="auto" metrics evaluated sequentially, unloading between runs.

    Returns list of successfully evaluated metric instances (same objects passed in).
    """
    if not metric_instances:
        return []

    # Partition metrics into regular and auto-device ones
    regular: List[Tuple[int, object]] = []
    auto_device: List[Tuple[int, object]] = []

    for idx, metric in enumerate(metric_instances):
        try:
            if hasattr(metric, 'device_map') and metric.device_map == "auto":
                auto_device.append((idx, metric))
            elif hasattr(metric, 'load_kwargs') and isinstance(metric.load_kwargs, dict) and metric.load_kwargs.get('device_map') == "auto":
                auto_device.append((idx, metric))
            else:
                regular.append((idx, metric))
        except Exception:
            regular.append((idx, metric))

    successful: List[object] = []
    failed: List[object] = []
    oom_queue: List[Tuple[int, object]] = []

    # Wave 1: regular metrics, try parallel then fallback
    if regular:
        if enable_parallel and len(regular) > 1:
            successful += _evaluate_parallel_with_fallback(dataset, regular, oom_queue, failed, max_parallel_workers, allowed_failed_metrics)
        else:
            successful += _evaluate_sequential(dataset, regular, failed, allowed_failed_metrics)

    # Process any OOM from wave 1 sequentially with unloads
    if oom_queue:
        _unload_all(successful)
        for _, metric in oom_queue:
            try:
                _unload_all(successful)
                lm = getattr(metric, 'model', None)
                if lm is not None:
                    with dspy.settings.context(lm=lm):
                        metric.predict(dataset, update_dataset=True, with_feedback=True)
                else:
                    metric.predict(dataset, update_dataset=True, with_feedback=True)
                successful.append(metric)
            except Exception as e:
                failed.append(metric)
                if len(failed) > allowed_failed_metrics:
                    raise e
            finally:
                _unload_metric(metric)

    # Wave 2: device_map="auto" metrics, always sequential with unloads
    if auto_device:
        for _, metric in auto_device:
            try:
                _unload_all(successful)
                lm = getattr(metric, 'model', None)
                if lm is not None:
                    with dspy.settings.context(lm=lm):
                        metric.predict(dataset, update_dataset=True, with_feedback=True)
                else:
                    metric.predict(dataset, update_dataset=True, with_feedback=True)
                successful.append(metric)
            except Exception as e:
                failed.append(metric)
                if len(failed) > allowed_failed_metrics:
                    raise e
            finally:
                _unload_metric(metric)

    return successful


def _evaluate_sequential(dataset, indexed_metrics: List[Tuple[int, object]], failed: List[object], allowed_failed: int) -> List[object]:
    successful: List[object] = []
    for _, metric in indexed_metrics:
        try:
            lm = getattr(metric, 'model', None)
            if lm is not None:
                with dspy.settings.context(lm=lm):
                    metric.predict(dataset, update_dataset=True)
            else:
                metric.predict(dataset, update_dataset=True)
            successful.append(metric)
            _unload_metric(metric)
        except Exception as e:
            failed.append(metric)
            _unload_metric(metric)
            if len(failed) > allowed_failed:
                raise e
    return successful


def _evaluate_parallel_with_fallback(
    dataset,
    indexed_metrics: List[Tuple[int, object]],
    oom_queue: List[Tuple[int, object]],
    failed: List[object],
    max_parallel_workers: int,
    allowed_failed: int,
) -> List[object]:
    import concurrent.futures

    def run_metric(metric):
        try:
            ds_copy = dataset.copy()
            lm = getattr(metric, 'model', None)
            if lm is not None:
                with dspy.settings.context(lm=lm):
                    metric.predict(ds_copy, update_dataset=True, with_feedback=True)
            else:
                metric.predict(ds_copy, update_dataset=True, with_feedback=True)

            # Extract results from copy to avoid recompute on main dataset
            results_dict: Dict[str, List] = {}
            if isinstance(metric, MultiMetric):
                for submetric_name in metric.get_submetric_names():
                    if submetric_name in ds_copy.get_dataframe().columns:
                        results_dict[submetric_name] = ds_copy.get_dataframe()[submetric_name].tolist()
            else:
                metric_name = metric.get_name()
                if metric_name in ds_copy.get_dataframe().columns:
                    results_dict[metric_name] = ds_copy.get_dataframe()[metric_name].tolist()

            return (metric, results_dict, None)
        except Exception as e:
            return (metric, None, e)

    successful: List[object] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(indexed_metrics), max_parallel_workers)) as executor:
        futures = {executor.submit(run_metric, metric): metric for _, metric in indexed_metrics}
        for future in concurrent.futures.as_completed(futures):
            metric, results_dict, err = future.result()
            if err is None:
                try:
                    # Merge back results without re-running heavy compute
                    dataset.add_metric(metric, update_dataset=False)
                    if results_dict:
                        df = dataset.get_dataframe()
                        for column_name, values in results_dict.items():
                            if len(values) == len(df):
                                df[column_name] = values
                        dataset.set_dataframe(df)
                    successful.append(metric)
                    _unload_metric(metric)
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                except Exception as e:
                    failed.append(metric)
                    if len(failed) > allowed_failed:
                        raise e
            else:
                msg = str(err).lower()
                if "cuda out of memory" in msg or "cublas_status_alloc_failed" in msg or "meta tensor" in msg or "device" in msg:
                    oom_queue.append((0, metric))
                else:
                    failed.append(metric)
                    if len(failed) > allowed_failed:
                        raise err
    return successful


