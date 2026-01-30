# Device utilities for handling PyTorch tensors and models
from autometrics.metrics.utils.device_utils import get_model_device, ensure_tensor_on_device
from autometrics.metrics.utils.gpu_allocation import (
    get_gpu_info,
    collect_metric_requirements,
    allocate_gpus,
)

__all__ = [
    'get_model_device',
    'ensure_tensor_on_device',
    'get_gpu_info',
    'collect_metric_requirements',
    'allocate_gpus',
] 