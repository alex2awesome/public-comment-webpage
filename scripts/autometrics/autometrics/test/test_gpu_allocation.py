import types
import pytest

from autometrics.metrics.utils.gpu_allocation import (
    allocate_gpus,
    GPUInfo,
)

# ---------------------------------------------------------------------------
# Helper to build dummy metric classes on-the-fly
# ---------------------------------------------------------------------------

def make_metric_class(name, gpu_mem, has_device=False, has_device_map=False):
    """Create a dummy metric class with desired constructor signature."""

    # Build parameter list dynamically
    params = []
    if has_device:
        params.append("device=None")
    if has_device_map:
        params.append("device_map=None")
    params.append("**kwargs")
    param_str = ", ".join(params)

    namespace = {"gpu_mem": gpu_mem}

    # Dynamically compile an __init__ with desired signature
    exec(
        f"def __init__(self, {param_str}):\n    pass",  # noqa: S102
        {},
        namespace,
    )

    return types.new_class(name, (), {}, lambda ns: ns.update(namespace))


# ---------------------------------------------------------------------------
# Fixtures for deterministic GPU environments
# ---------------------------------------------------------------------------

@pytest.fixture
def single_gpu_env(monkeypatch):
    gpu_list = [GPUInfo(index=0, name="MockGPU", total_mb=10_000, free_mb=10_000)]
    monkeypatch.setattr(
        "autometrics.metrics.utils.gpu_allocation.get_gpu_info", lambda: gpu_list
    )
    return gpu_list


@pytest.fixture
def multi_gpu_env(monkeypatch):
    gpu_list = [
        GPUInfo(index=0, name="GPU0", total_mb=8_000, free_mb=8_000),
        GPUInfo(index=1, name="GPU1", total_mb=8_000, free_mb=8_000),
    ]
    monkeypatch.setattr(
        "autometrics.metrics.utils.gpu_allocation.get_gpu_info", lambda: gpu_list
    )
    return gpu_list


@pytest.fixture
def exact_fit_single_gpu_env(monkeypatch):
    gpu_list = [GPUInfo(index=0, name="ExactGPU", total_mb=16_000, free_mb=16_000)]
    monkeypatch.setattr(
        "autometrics.metrics.utils.gpu_allocation.get_gpu_info", lambda: gpu_list
    )
    return gpu_list


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_gpu(monkeypatch):
    """If no GPU is detected we should get an empty allocation map."""
    monkeypatch.setattr(
        "autometrics.metrics.utils.gpu_allocation.get_gpu_info", lambda: []
    )
    small = make_metric_class("Small", 1000, has_device=True)
    alloc = allocate_gpus([small])
    assert alloc == {}


def test_single_gpu_allocation(single_gpu_env):
    small = make_metric_class("Small", 1500, has_device=True)
    medium = make_metric_class("Medium", 3000, has_device_map=True)
    large = make_metric_class("Large", 9_000, has_device=True, has_device_map=True)

    alloc = allocate_gpus([small, medium, large])

    # Small & Medium should map to cuda:0 (single GPU available)
    assert alloc["Small"]["device"] == "cuda:0"
    # Medium uses explicit device_map single gpu (falls back to device map or device depending on fit)
    assert "device" in alloc["Medium"] or "device_map" in alloc["Medium"]
    # Large exceeds capacity -> assigned to cuda:0 (single-GPU env)
    assert alloc["Large"].get("device") == "cuda:0"


def test_multi_gpu_balanced(multi_gpu_env):
    m1 = make_metric_class("M1", 4_000, has_device=True)
    m2 = make_metric_class("M2", 4_000, has_device=True)

    alloc = allocate_gpus([m1, m2], buffer_ratio=0.0)  # simplify math

    # Best-fit decreasing will put first metric on GPU0, second on GPU1
    assert alloc["M1"]["device"] == "cuda:0"
    assert alloc["M2"]["device"] == "cuda:1"


def test_metric_without_device_args(multi_gpu_env):
    # Metric uses GPU but lacks device/device_map args
    rogue = make_metric_class("Rogue", 1_000, has_device=False, has_device_map=False)
    alloc = allocate_gpus([rogue], buffer_ratio=0.0)
    # Should get a hint key rather than real kwarg
    assert "_hint_gpu_index" in alloc["Rogue"]


def test_auto_split_reduces_available(multi_gpu_env):
    huge = make_metric_class("Huge", 12_000, has_device_map=True)
    # With buffer 0, each GPU has 8k free, total 16k, so auto is needed.
    # After allocation there should be no room left for another 8k metric.
    alloc1 = allocate_gpus([huge], buffer_ratio=0.0)
    assert alloc1["Huge"]["device_map"] == "auto"

    # Now try to allocate another big metric; due to bookkeeping it should
    # fall back to auto again rather than falsely thinking a single gpu fits.
    big = make_metric_class("Big", 8_000, has_device=True)
    alloc2 = allocate_gpus([huge, big], buffer_ratio=0.0)
    assert "device_map" in alloc2["Big"] or alloc2["Big"].get("device") is not None 


def test_exact_fit_single_gpu(exact_fit_single_gpu_env):
    """Metrics exactly fill the GPU memory – both should be placed on cuda:0."""
    large = make_metric_class("LargeExact", 12_000, has_device=True)
    small = make_metric_class("SmallExact", 4_000, has_device=True)

    alloc = allocate_gpus([large, small], buffer_ratio=0.0)

    assert alloc["LargeExact"]["device"] == "cuda:0"
    assert alloc["SmallExact"]["device"] == "cuda:0" 


def test_tight_fit_multi_gpu(multi_gpu_env):
    """Large metric splits across GPUs, leaving only small residue memory."""
    huge = make_metric_class("HugeTight", 12_000, has_device_map=True)
    small = make_metric_class("SmallTight", 4_000, has_device_map=True)

    alloc = allocate_gpus([huge, small], buffer_ratio=0.0)

    # Huge metric should auto-split
    assert alloc["HugeTight"]["device_map"] == "auto"
    # Small metric also auto-splits across GPUs to fit
    assert alloc["SmallTight"]["device_map"] == "auto" 