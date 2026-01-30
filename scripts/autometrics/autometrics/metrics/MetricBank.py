# NOTE: This file was refactored to delay heavy metric instantiation until
# they are actually needed.  We provide factory helpers that build metrics on
# demand, with optional common parameters such as cache_dir and seed.

from __future__ import annotations

from typing import List, Dict, Type, Any
import inspect
import os

# ---------------------------------------------------------------------------
# Import metric *classes* only (light-weight) – do NOT instantiate here.
# ---------------------------------------------------------------------------

from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.CHRF import CHRF
from autometrics.metrics.reference_based.TER import TER
from autometrics.metrics.reference_based.GLEU import GLEU
from autometrics.metrics.reference_based.SARI import SARI
from autometrics.metrics.reference_based.BERTScore import BERTScore
from autometrics.metrics.reference_based.ROUGE import ROUGE
from autometrics.metrics.reference_based.MOVERScore import MOVERScore
from autometrics.metrics.reference_based.BARTScore import BARTScore
from autometrics.metrics.reference_based.UniEvalDialogue import UniEvalDialogue
from autometrics.metrics.reference_based.UniEvalSum import UniEvalSum
from autometrics.metrics.reference_based.CIDEr import CIDEr
from autometrics.metrics.reference_based.METEOR import METEOR
from autometrics.metrics.reference_based.StringSimilarity import (
    LevenshteinDistance,
    LevenshteinRatio,
    HammingDistance,
    JaroSimilarity,
    JaroWinklerSimilarity,
    JaccardDistance,
)
from autometrics.metrics.reference_based.ParaScore import ParaScore
from autometrics.metrics.reference_based.YiSi import YiSi
from autometrics.metrics.reference_based.MAUVE import MAUVE
from autometrics.metrics.reference_based.PseudoPARENT import PseudoPARENT
from autometrics.metrics.reference_based.NIST import NIST
from autometrics.metrics.reference_based.IBLEU import IBLEU
from autometrics.metrics.reference_based.UpdateROUGE import UpdateROUGE
from autometrics.metrics.reference_based.BLEURT import BLEURT
from autometrics.metrics.reference_based.LENS import LENS
from autometrics.metrics.reference_based.CharCut import CharCut
from autometrics.metrics.reference_based.InfoLM import InfoLM

from autometrics.metrics.reference_free.FKGL import FKGL
from autometrics.metrics.reference_free.UniEvalFact import UniEvalFact
from autometrics.metrics.reference_free.Perplexity import Perplexity
from autometrics.metrics.reference_free.ParaScoreFree import ParaScoreFree
from autometrics.metrics.reference_free.INFORMRewardModel import INFORMRewardModel
from autometrics.metrics.reference_free.PRMRewardModel import MathProcessRewardModel
from autometrics.metrics.reference_free.SummaQA import SummaQA
from autometrics.metrics.reference_free.DistinctNGram import DistinctNGram
from autometrics.metrics.reference_free.FastTextToxicity import FastTextToxicity
from autometrics.metrics.reference_free.FastTextNSFW import FastTextNSFW
from autometrics.metrics.reference_free.FastTextEducationalValue import FastTextEducationalValue
from autometrics.metrics.reference_free.SelfBLEU import SelfBLEU
from autometrics.metrics.reference_free.FactCC import FactCC
from autometrics.metrics.reference_free.Toxicity import Toxicity
from autometrics.metrics.reference_free.GRMRewardModel import GRMRewardModel
from autometrics.metrics.reference_free.LDLRewardModel import LDLRewardModel
from autometrics.metrics.reference_free.Sentiment import Sentiment
from autometrics.metrics.reference_free.LENS_SALSA import LENS_SALSA

# ---------------------------------------------------------------------------
# Metric class registries
# ---------------------------------------------------------------------------

reference_based_metric_classes: List[Type] = [
    BLEU, CHRF, TER, GLEU, SARI, BERTScore, ROUGE, MOVERScore, BARTScore,
    UniEvalDialogue, UniEvalSum, CIDEr, METEOR, BLEURT, LevenshteinDistance,
    LevenshteinRatio, HammingDistance, JaroSimilarity, JaroWinklerSimilarity,
    JaccardDistance, ParaScore, YiSi, MAUVE, PseudoPARENT, NIST, IBLEU,
    UpdateROUGE, LENS, CharCut, InfoLM,
]

reference_free_metric_classes: List[Type] = [
    FKGL, UniEvalFact, Perplexity, ParaScoreFree, INFORMRewardModel,
    MathProcessRewardModel, SummaQA, DistinctNGram, FastTextToxicity,
    FastTextNSFW, FastTextEducationalValue, SelfBLEU, FactCC, Toxicity,
    Sentiment, GRMRewardModel, LENS_SALSA, LDLRewardModel,
]

all_metric_classes: List[Type] = reference_based_metric_classes + reference_free_metric_classes

# ---------------------------------------------------------------------------
# Default per-metric kwargs (to replicate previous behaviour)
# ---------------------------------------------------------------------------

_DEFAULT_EXTRA_KWARGS: Dict[str, Dict[str, Any]] = {
    "Perplexity": {"batch_size": 2},
}

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

# GPU allocation helper - moved to lazy import to avoid 4.8s startup delay

def _instantiate_metric(cls: Type, kwargs: Dict[str, Any]):
    """Instantiate a metric class with the provided kwargs (already filtered)."""
    try:
        return cls(**kwargs)
    except Exception as e:
        print(f"[MetricBank] Failed to instantiate {cls.__name__} with kwargs {kwargs}: {e}. Trying default constructor …")
        try:
            return cls()
        except Exception as e2:
            print(f"[MetricBank] Giving up on {cls.__name__}: {e2}")
            return None


def _get_cache_dir() -> str:
    """
    Get the cache directory from environment variable AUTOMETRICS_CACHE_DIR,
    with fallback to "./autometrics_cache" if not set.
    
    Returns:
        Cache directory path as string
    """
    return os.environ.get("AUTOMETRICS_CACHE_DIR", "./autometrics_cache")


def build_metrics(
    classes: List[Type],
    cache_dir: str | None = None,
    seed: int | None = None,
    use_cache: bool = True,
    overrides: Dict[str, Dict[str, Any]] | None = None,
    gpu_buffer_ratio: float = 0.10,
) -> List[Any]:
    """Instantiate a list of metric classes with common kwargs and cache override."""
    # Ensure global meta-tensor safe patch is applied exactly once for all models
    try:
        from autometrics.metrics.utils.device_utils import (
            apply_meta_tensor_safe_module_to_patch,
            apply_roberta_token_type_guard,
        )
        apply_meta_tensor_safe_module_to_patch()
        apply_roberta_token_type_guard()
    except Exception as _patch_err:
        print(f"[MetricBank] Warning: failed to apply meta-tensor safe Module.to patch: {_patch_err}")
    common_kwargs = {
        "cache_dir": cache_dir or _get_cache_dir(),
        "seed": seed,
        "use_cache": use_cache,
    }
    overrides = overrides or {}

    # --------------------------------------------------------
    # GPU allocation planning (performed once per batch)
    # Check if any metrics actually need GPUs before attempting allocation
    # --------------------------------------------------------
    allocation_map = {}
    try:
        # Check if any metrics actually need GPUs
        needs_gpu = any(getattr(cls, "gpu_mem", 0) > 0 for cls in classes)
        
        if needs_gpu:
            # Lazy import GPU allocation utilities only when needed (avoids 4.8s startup delay)
            from autometrics.metrics.utils import allocate_gpus
            allocation_map = allocate_gpus(classes, buffer_ratio=gpu_buffer_ratio)
        else:
            # No metrics need GPUs, skip allocation entirely
            pass
    except Exception as e:
        # If GPU allocation fails (e.g., NVML not available), warn and continue with CPU
        print(f"[MetricBank] GPU allocation failed: {e}. Falling back to CPU-only execution.")
        allocation_map = {}

    metrics = []
    for cls in classes:
        # Build merged kwargs
        sig = inspect.signature(cls.__init__)
        merged: Dict[str, Any] = {}
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        # common first
        for k, v in common_kwargs.items():
            if k in sig.parameters or has_var_kw:
                merged[k] = v
        # per-metric overrides highest priority (call-supplied)
        for k, v in overrides.get(cls.__name__, {}).items():
            if k in sig.parameters or has_var_kw:
                merged[k] = v
        # GPU allocation overrides (device/device_map) take precedence over
        # defaults but *not* over explicit user-supplied overrides above.
        for k, v in allocation_map.get(cls.__name__, {}).items():
            if k in merged:
                continue  # user already set explicitly via overrides
            if k in sig.parameters or has_var_kw:
                merged[k] = v
        
        # Debug: Show what kwargs are being passed to each metric
        if allocation_map.get(cls.__name__):
            print(f"[MetricBank] {cls.__name__} kwargs: {merged}")
            print(f"[MetricBank] {cls.__name__} GPU allocation: {allocation_map.get(cls.__name__)}")
        # fill with metric defaults if still missing
        for k, v in _DEFAULT_EXTRA_KWARGS.get(cls.__name__, {}).items():
            if k in sig.parameters or has_var_kw:
                merged.setdefault(k, v)

        metric = _instantiate_metric(cls, merged)
        if metric is None:
            continue
        
        # Debug: Show what device the metric is actually using
        if hasattr(metric, 'model') and metric.model is not None:
            try:
                if hasattr(metric.model, 'device'):
                    print(f"[MetricBank] {cls.__name__} model device: {metric.model.device}")
                elif hasattr(metric.model, 'hf_device_map'):
                    print(f"[MetricBank] {cls.__name__} model hf_device_map: {metric.model.hf_device_map}")
                else:
                    print(f"[MetricBank] {cls.__name__} model has no device info")
            except Exception as e:
                print(f"[MetricBank] {cls.__name__} could not determine model device: {e}")
        
        metrics.append(metric)
    return metrics


def build_reference_based_metrics(**kwargs) -> List[Any]:
    return build_metrics(reference_based_metric_classes, **kwargs)


def build_reference_free_metrics(**kwargs) -> List[Any]:
    return build_metrics(reference_free_metric_classes, **kwargs)


def build_all_metrics(**kwargs) -> List[Any]:
    return build_metrics(all_metric_classes, **kwargs)