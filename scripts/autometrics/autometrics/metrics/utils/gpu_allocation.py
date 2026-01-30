from __future__ import annotations

"""Utility helpers for GPU-aware metric allocation.

This module lets `MetricBank` (or any caller) assign metrics to the
available GPUs in a resource-conscious way.  The design keeps phase 1
(light-weight planning) completely free of heavy model instantiation.
Instead we rely on the static `gpu_mem` class attributes that now exist
on every metric class.

Main entry-points
-----------------
1. `get_gpu_info()` – returns total / free memory for each visible GPU.
2. `collect_metric_requirements(metric_classes)` – inspects metrics and
   records GPU memory + supported device kwargs.
3. `allocate_gpus(metric_classes, buffer_ratio=0.10)` – best-effort
   heuristic packing of metrics onto GPUs, returning the kwargs you
   should pass when instantiating each metric.
4. `safe_torch_load(path, **kwargs)` – torch.load with automatic CPU fallback

The packing is a greedy *best-fit decreasing* heuristic (simple and fast
in practice) with a fallback to multi-GPU (`device_map="auto"`) when the
required memory exceeds that of any single card *and* the metric class
supports `device_map`.

If a metric is GPU-compatible but neither a `device_map` nor `device`
parameter is present in its constructor we fall back to **GPU 0**.  CPU-
only metrics (those with `gpu_mem == 0`) are ignored by the allocator.

When CUDA is unavailable, all GPU allocations automatically fall back to CPU.
"""

from typing import List, Dict, Any, NamedTuple, Optional
import inspect
import warnings
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# CUDA availability detection
# ---------------------------------------------------------------------------

def is_cuda_available() -> bool:
    """Check if CUDA is available, with proper error handling."""
    if not TORCH_AVAILABLE:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        # Handle any CUDA initialization errors
        return False

def safe_torch_load(file_path: str, **kwargs):
    """
    Load a torch model with automatic CPU fallback when CUDA is unavailable.
    
    This function automatically adds map_location=torch.device('cpu') when
    CUDA is not available, preventing the common error:
    "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False"
    
    Args:
        file_path: Path to the torch model file
        **kwargs: Additional arguments to pass to torch.load
        
    Returns:
        Loaded model/state_dict
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for safe_torch_load")
    
    # If CUDA is not available, force CPU mapping
    if not is_cuda_available():
        if 'map_location' not in kwargs:
            kwargs['map_location'] = torch.device('cpu')
    
    return torch.load(file_path, **kwargs)

# ---------------------------------------------------------------------------
# GPU discovery helpers
# ---------------------------------------------------------------------------

class GPUInfo(NamedTuple):
    index: int
    name: str
    total_mb: float
    free_mb: float


def _query_gpu_info_pynvml() -> List[GPUInfo]:
    """Return GPU stats using NVML (preferred – accurate free memory)."""
    try:
        import pynvml  # type: ignore
    except ImportError:
        return []

    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        infos: List[GPUInfo] = []
        for idx in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(raw_name, bytes):
                name = raw_name.decode("utf-8")
            else:
                name = str(raw_name)
            infos.append(
                GPUInfo(
                    index=idx,
                    name=name,
                    total_mb=mem_info.total / 1024 ** 2,
                    free_mb=mem_info.free / 1024 ** 2,
                )
            )
        return infos
    finally:
        pynvml.nvmlShutdown()


def _query_gpu_info_torch() -> List[GPUInfo]:
    """Fallback GPU stats via torch (gives *total* memory only)."""
    if not TORCH_AVAILABLE:
        return []

    infos: List[GPUInfo] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_mb = props.total_memory / 1024 ** 2
        # torch has no free-mem query that is CUDA-context-free; conservatively
        # assume *all* memory is available.
        infos.append(GPUInfo(index=idx, name=props.name, total_mb=total_mb, free_mb=total_mb))
    return infos


def _parse_cuda_visible_devices_env() -> Optional[List[int]]:
    """Parse CUDA_VISIBLE_DEVICES into a list of physical GPU indices if possible.

    Returns None when env var is unset/empty, or when it contains non-numeric
    identifiers (e.g., UUIDs), in which case we prefer torch-based discovery.
    """
    env_val = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not env_val:
        return None
    parts = [p.strip() for p in env_val.split(",") if p.strip() != ""]
    try:
        indices = [int(p) for p in parts]
        return indices
    except ValueError:
        # Non-numeric (UUIDs). We'll rely on torch which already respects visibility.
        return None


def _remap_infos_to_visible_ordinals(all_infos: List[GPUInfo], visible_phys: List[int]) -> List[GPUInfo]:
    """Filter and reorder `all_infos` to the physical indices specified in
    `visible_phys`, and reindex them to contiguous visible ordinals [0..n-1]."""
    phys_to_info: Dict[int, GPUInfo] = {info.index: info for info in all_infos}
    remapped: List[GPUInfo] = []
    for vis_idx, phys_idx in enumerate(visible_phys):
        info = phys_to_info.get(phys_idx)
        if info is None:
            # If a specified physical index doesn't exist, skip it silently
            # (better than crashing) – torch will error later if nothing is visible.
            continue
        remapped.append(GPUInfo(index=vis_idx, name=info.name, total_mb=info.total_mb, free_mb=info.free_mb))
    return remapped


def get_gpu_info() -> List[GPUInfo]:
    """Return a list of `GPUInfo` for all visible GPUs.

    NVML is preferred for accurate free memory; fall back to torch when NVML
    is unavailable.  If neither mechanism works we return an empty list.
    """
    # If CUDA visibility is numerically specified, filter NVML results to those
    # physical indices and remap to visible ordinals. Otherwise, prefer torch.
    visible_phys = _parse_cuda_visible_devices_env()
    env_val = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()

    infos = _query_gpu_info_pynvml()
    if infos:
        if visible_phys is not None:
            # Respect numeric visibility; remap to visible ordinals
            remapped = _remap_infos_to_visible_ordinals(infos, visible_phys)
            if remapped:
                return remapped
            # If remap failed (e.g., bad indices), fall back to torch
        else:
            # If env isn't set at all, NVML full list is fine.
            # If env is set but non-numeric (e.g., UUIDs), prefer torch which respects visibility.
            if not env_val:
                return infos

    # Torch already respects CUDA visibility (including UUID-based). Indices are visible ordinals.
    return _query_gpu_info_torch()

# ---------------------------------------------------------------------------
# Metric requirement introspection
# ---------------------------------------------------------------------------

class MetricRequirement(NamedTuple):
    cls: type
    name: str
    gpu_mem_mb: float
    has_device_arg: bool
    has_device_map_arg: bool


def _inspect_metric_class(cls: type) -> MetricRequirement:
    """Inspect a metric class to determine its GPU requirements and device handling capabilities."""
    # Check if this is a HuggingFace evaluate metric (like Toxicity, MAUVE)
    # These metrics have their own device management and shouldn't get device_map
    is_huggingface_evaluate = (
        hasattr(cls, '__bases__') and 
        any('HuggingFaceReferenceFreeMetric' in str(base) for base in cls.__bases__)
    )
    
    # Check if this is a HuggingFace reference-based metric (like MAUVE)
    is_huggingface_reference_based = (
        hasattr(cls, '__bases__') and 
        any('HuggingFaceReferenceBasedMetric' in str(base) for base in cls.__bases__)
    )
    
    # Get GPU memory requirement
    gpu_mem = getattr(cls, "gpu_mem", 0.0)
    
    # Check constructor signature for device-related parameters
    sig = inspect.signature(cls.__init__)
    has_device_arg = "device" in sig.parameters
    has_device_map_arg = "device_map" in sig.parameters
    has_load_kwargs_arg = "load_kwargs" in sig.parameters
    
    # For HuggingFace evaluate metrics, treat them as CPU-only to avoid device conflicts
    # These metrics have internal device management that conflicts with our GPU allocation
    if is_huggingface_evaluate or is_huggingface_reference_based:
        has_device_map_arg = False
        has_device_arg = False  # Don't allocate GPU to these metrics
        gpu_mem = 0.0  # Force CPU-only
    
    return MetricRequirement(
        cls=cls,
        name=cls.__name__,
        gpu_mem_mb=gpu_mem,
        has_device_arg=has_device_arg,
        has_device_map_arg=has_device_map_arg,
    )


def collect_metric_requirements(metric_classes: List[type]) -> List[MetricRequirement]:
    """Return a list of `MetricRequirement` for the given metric classes."""
    return [_inspect_metric_class(cls) for cls in metric_classes]

# ---------------------------------------------------------------------------
# Greedy best-fit decreasing allocator
# ---------------------------------------------------------------------------

AllocationResult = Dict[str, Dict[str, Any]]  # metric name -> kwarg overrides


def allocate_gpus(
    metric_classes: List[type],
    buffer_ratio: float = 0.10,
    gpu_infos: Optional[List[GPUInfo]] = None,
) -> AllocationResult:
    """Heuristically allocate metrics to GPUs.

    Parameters
    ----------
    metric_classes: list of metric *classes* (not instances)
        The metrics the user intends to instantiate.
    buffer_ratio: float, default 0.10 (10 %)
        Fractional safety buffer subtracted from each GPU's free memory *and*
        added to each metric's requirement.
    gpu_infos: optional pre-queried GPU list (for tests/mocks)
    Returns
    -------
    dict
        Mapping from metric class name to a dict of kwarg overrides (e.g.
        {"device": torch.device("cuda:2")} or {"device_map": "auto"}).  CPU-
        only metrics are omitted. When CUDA is unavailable, GPU metrics
        get CPU fallback allocations.
    """

    # Check if CUDA is available first
    if not is_cuda_available():
        warnings.warn("CUDA is not available – forcing all metrics to run on CPU.")
        # Return CPU allocations for all GPU-requiring metrics
        requirements = collect_metric_requirements(metric_classes)
        cpu_allocations: AllocationResult = {}
        for req in requirements:
            if req.gpu_mem_mb > 0:  # This was a GPU metric
                # Check if this is a HuggingFace evaluate metric
                is_huggingface_evaluate = (
                    hasattr(req.cls, '__bases__') and 
                    any('HuggingFaceReferenceFreeMetric' in str(base) for base in req.cls.__bases__)
                )
                
                if is_huggingface_evaluate:
                    cpu_allocations[req.name] = {"load_kwargs": {"device": "cpu"}}
                elif req.has_device_arg:
                    cpu_allocations[req.name] = {"device": torch.device("cpu") if TORCH_AVAILABLE else "cpu"}
                elif req.has_device_map_arg:
                    cpu_allocations[req.name] = {"device_map": {"": "cpu"}}
                # For metrics without device args, they'll default to CPU anyway
        return cpu_allocations

    gpu_infos = list(gpu_infos or get_gpu_info())
    if not gpu_infos:
        warnings.warn("No GPU detected – all metrics will run on CPU.")
        # Return CPU allocations for GPU-requiring metrics
        requirements = collect_metric_requirements(metric_classes)
        cpu_allocations: AllocationResult = {}
        for req in requirements:
            if req.gpu_mem_mb > 0:  # This was a GPU metric
                # Check if this is a HuggingFace evaluate metric
                is_huggingface_evaluate = (
                    hasattr(req.cls, '__bases__') and 
                    any('HuggingFaceReferenceFreeMetric' in str(base) for base in req.cls.__bases__)
                )
                
                if is_huggingface_evaluate:
                    cpu_allocations[req.name] = {"load_kwargs": {"device": "cpu"}}
                elif req.has_device_arg:
                    cpu_allocations[req.name] = {"device": torch.device("cpu") if TORCH_AVAILABLE else "cpu"}
                elif req.has_device_map_arg:
                    cpu_allocations[req.name] = {"device_map": {"": "cpu"}}
        return cpu_allocations
    
    # Debug: Show what GPUs are available
    print(f"[GPU Allocation] Found {len(gpu_infos)} GPU(s):")
    for info in gpu_infos:
        print(f"  GPU {info.index}: {info.name} - {info.total_mb:.1f}MB total, {info.free_mb:.1f}MB free")

    # Show CUDA visibility mapping, if applicable
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    parsed_visible = _parse_cuda_visible_devices_env()
    if cuda_visible:
        if parsed_visible is not None:
            mapping_pairs = ", ".join([f"{phys}->#{vis}" for vis, phys in enumerate(parsed_visible)])
            print(f"[GPU Allocation] CUDA_VISIBLE_DEVICES={cuda_visible} (physical->visible mapping: {mapping_pairs})")
        else:
            print(f"[GPU Allocation] CUDA_VISIBLE_DEVICES set (non-numeric). Using torch-visible ordinals 0..{max(0, len(gpu_infos)-1)}")
    
    # Debug: Show PyTorch CUDA device info
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"[GPU Allocation] PyTorch CUDA info:")
        print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"  torch.cuda.current_device(): {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            print(f"  Device {i}: {device_name} ({device_memory:.1f}GB)")

    # Working copy of available memory (apply buffer)
    avail_mb = {
        info.index: info.free_mb * (1.0 - buffer_ratio) for info in gpu_infos
    }

    requirements = collect_metric_requirements(metric_classes)
    # Consider only GPU-using metrics (gpu_mem > 0)
    gpu_metrics = [r for r in requirements if r.gpu_mem_mb > 0]
    
    print(f"[GPU Allocation] Found {len(gpu_metrics)} GPU-requiring metrics:")
    for req in gpu_metrics:
        print(f"  {req.name}: {req.gpu_mem_mb:.1f}MB, device_arg={req.has_device_arg}, device_map_arg={req.has_device_map_arg}")

    # Sort by descending memory requirement (best-fit decreasing)
    gpu_metrics.sort(key=lambda r: r.gpu_mem_mb, reverse=True)

    allocations: AllocationResult = {}

    for req in gpu_metrics:
        needed = req.gpu_mem_mb * (1.0 + buffer_ratio)

        # Attempt to place on a single GPU
        candidate_gpu = None
        best_free_after = -1.0
        for idx, free in avail_mb.items():
            if free >= needed:
                # prefer GPU that will remain most free *after* placement to keep load balanced
                free_after = free - needed
                if free_after > best_free_after:
                    best_free_after = free_after
                    candidate_gpu = idx
        if candidate_gpu is not None:
            # Single-GPU fit – update available memory and record allocation
            avail_mb[candidate_gpu] -= needed
            
            # Check if this is a HuggingFace evaluate metric
            is_huggingface_evaluate = (
                hasattr(req.cls, '__bases__') and 
                any('HuggingFaceReferenceFreeMetric' in str(base) for base in req.cls.__bases__)
            )
            
            if is_huggingface_evaluate:
                # For HuggingFace evaluate metrics, use load_kwargs with device
                allocation = {"load_kwargs": {"device": f"cuda:{candidate_gpu}"}}
                print(f"[GPU Allocation] {req.name} -> GPU {candidate_gpu} (HuggingFace evaluate, load_kwargs)")
            elif req.has_device_arg:
                allocation = {"device": f"cuda:{candidate_gpu}"}
                print(f"[GPU Allocation] {req.name} -> GPU {candidate_gpu} (device arg)")
            elif req.has_device_map_arg:
                # Fix: Use proper device string format for device_map
                # The device_map should use device strings, not integer indices
                allocation = {"device_map": {"": f"cuda:{candidate_gpu}"}}
                print(f"[GPU Allocation] {req.name} -> GPU {candidate_gpu} (device_map)")
            else:
                # No explicit arg – assume global torch.device context; advise user
                allocation = {"_hint_gpu_index": candidate_gpu}
                print(f"[GPU Allocation] {req.name} -> GPU {candidate_gpu} (hint only)")
            
            allocations[req.name] = allocation
            continue

        # Cannot fit on a single GPU
        if req.has_device_map_arg and len(avail_mb) > 1:
            # Let transformers/accelerate handle splitting across GPUs
            allocations[req.name] = {"device_map": "auto"}
            print(f"[GPU Allocation] {req.name} -> device_map='auto' (multi-GPU split required)")

            # Rough bookkeeping: assume balanced split across all GPUs so we
            # reduce each GPU's available memory proportionally.  This is a
            # heuristic – the actual allocation performed by `accelerate`
            # may differ.
            per_gpu_share = needed / max(1, len(avail_mb))
            for idx in avail_mb:
                avail_mb[idx] = max(0.0, avail_mb[idx] - per_gpu_share)
        elif req.has_device_map_arg:
            # Only one GPU present – splitting not possible, warn and assign GPU0
            warnings.warn(
                f"[gpu_allocation] Metric {req.name} too large for single GPU; assigning to cuda:0 with potential OOM."
            )
            allocations[req.name] = {"device_map": {"": "cuda:0"}}
        else:
            # No way to split ‑ assign to GPU 0 anyway and warn
            warnings.warn(
                f"[gpu_allocation] Metric {req.name} requires {req.gpu_mem_mb:.1f} MB but no single GPU has enough free memory; assigning to cuda:0 which may OOM."
            )
            
            # Check if this is a HuggingFace evaluate metric
            is_huggingface_evaluate = (
                hasattr(req.cls, '__bases__') and 
                any('HuggingFaceReferenceFreeMetric' in str(base) for base in req.cls.__bases__)
            )
            
            if is_huggingface_evaluate:
                # For HuggingFace evaluate metrics, use load_kwargs with device
                allocations[req.name] = {"load_kwargs": {"device": "cuda:0"}}
            elif req.has_device_arg:
                allocations[req.name] = {"device": "cuda:0"}
            else:
                allocations[req.name] = {"_hint_gpu_index": 0}

    return allocations 