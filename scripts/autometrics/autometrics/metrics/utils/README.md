# Autometrics Utility Functions

This directory contains utility functions that are used across multiple metrics in the Autometrics package.

## Device Utilities

The `device_utils.py` module contains functions for managing PyTorch devices and ensuring tensors are on the correct device.

### `get_model_device(model, fallback_device=None)`

Detects the device a PyTorch model is on by checking various model attributes.

**Parameters:**
- `model`: A PyTorch model or module
- `fallback_device`: Optional device to use if no device can be determined from the model

**Returns:**
- `torch.device`: The device the model is on

**Usage:**
```python
from autometrics.metrics.utils.device_utils import get_model_device

# Get the device a model is on
model_device = get_model_device(model, fallback_device=torch.device("cpu"))
```

### `ensure_tensor_on_device(tensor, device)`

Ensures a tensor is on the specified device. If the input is not a tensor, it is returned unchanged.

**Parameters:**
- `tensor`: A PyTorch tensor or any other object
- `device`: The target device

**Returns:**
- The tensor moved to the specified device, or the unchanged input if not a tensor

**Usage:**
```python
from autometrics.metrics.utils.device_utils import ensure_tensor_on_device

# Ensure input_ids tensor is on the model's device
input_ids = ensure_tensor_on_device(input_ids, model_device)
```

## Common Device Issues and Solutions

When working with PyTorch models, especially those utilizing HuggingFace's device_map for efficient GPU utilization, device mismatch errors are common. These typically manifest as errors like:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```

The utilities in this directory help prevent these errors by:

1. Properly detecting the actual device a model is on, even when using complex device maps
2. Ensuring input tensors are moved to the same device as the model before operations

For best practices:
- Always use `get_model_device()` to determine a model's device instead of assuming it
- Always move input tensors to the model's device using `ensure_tensor_on_device()` before passing them to the model 