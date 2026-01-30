import os
import time
import psutil
import json
from typing import Dict, Any, Optional

# Try to import optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pynvml
    HAS_PYNVML = True
    pynvml.nvmlInit()
except (ImportError, pynvml.NVMLError):
    HAS_PYNVML = False

# Global timestamp reference for consistent timing
t0 = time.time()

def get_gpu_memory_mb() -> float:
    """Get GPU memory usage across all available devices in MB."""
    total_memory = 0.0
    
    # Try PyTorch first
    if HAS_TORCH and torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                total_memory = sum(torch.cuda.memory_allocated(i) for i in range(device_count)) / (1024 * 1024)
                return total_memory
        except Exception:
            pass
    
    # Try NVML as fallback
    if HAS_PYNVML:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory += info.used / (1024 * 1024)
            return total_memory
        except Exception:
            pass
    
    return 0.0

def snap(label: str) -> Dict[str, Any]:
    """Take a snapshot of current resource usage."""
    # Get the process inside the function to avoid circular imports
    process = psutil.Process(os.getpid())
    
    return {
        "label": label,
        "timestamp_s": time.time() - t0,
        "cpu_ram_mb": process.memory_info().rss / (1024 * 1024),
        "gpu_ram_mb": get_gpu_memory_mb(),
        "disk_used_mb": psutil.disk_usage('/').used / (1024 * 1024)
    }

def calc_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate resource changes between two snapshots."""
    return {
        "phase": f"{before['label']}→{after['label']}",
        "duration_milliseconds": (after["timestamp_s"] - before["timestamp_s"]) * 1000,
        "cpu_ram_mb": max(0, after["cpu_ram_mb"] - before["cpu_ram_mb"]),
        "gpu_ram_mb": max(0, after["gpu_ram_mb"] - before["gpu_ram_mb"]),
        "disk_usage_change_mb": max(0, after["disk_used_mb"] - before["disk_used_mb"]),
        "baseline_cpu_ram_mb": before["cpu_ram_mb"],
        "baseline_gpu_ram_mb": before["gpu_ram_mb"],
        "total_cpu_ram_mb": after["cpu_ram_mb"],
        "total_gpu_ram_mb": after["gpu_ram_mb"]
    } 