import pytest
import torch
from unittest.mock import MagicMock
from autometrics.metrics.utils.device_utils import get_model_device, ensure_tensor_on_device

class TestDeviceUtils:
    """Test suite for device utility functions."""

    def test_get_model_device_with_device_property(self):
        """Test get_model_device when model has a device property."""
        # Create mock model with device property
        model = MagicMock()
        model.device = torch.device("cpu")
        
        # Get device
        device = get_model_device(model)
        
        # Check result
        assert device == torch.device("cpu")

    def test_get_model_device_with_get_device_method(self):
        """Test get_model_device when model has a get_device method."""
        # Create mock model with get_device method
        model = MagicMock()
        model.device = None  # Ensure property lookup fails
        model.get_device.return_value = torch.device("cpu")
        
        # Get device
        device = get_model_device(model)
        
        # Check result
        assert device == torch.device("cpu")

    def test_get_model_device_from_parameters(self):
        """Test get_model_device when device is determined from parameters."""
        # Create mock model with parameters
        model = MagicMock()
        model.device = None  # Ensure property lookup fails
        model.get_device = None  # Ensure method lookup fails
        
        # Mock parameters
        param = MagicMock()
        param.device = torch.device("cpu")
        model.parameters.return_value = iter([param])
        
        # Get device
        device = get_model_device(model)
        
        # Check result
        assert device == torch.device("cpu")

    def test_get_model_device_with_fallback(self):
        """Test get_model_device with fallback device when no device can be determined."""
        # Create mock model with no device info
        model = MagicMock()
        model.device = None
        model.get_device = None
        model.parameters.return_value = iter([])  # Empty iterator
        
        # Create fallback device
        fallback_device = torch.device("cuda:0")
        
        # Get device
        device = get_model_device(model, fallback_device=fallback_device)
        
        # Check result
        assert device == fallback_device

    def test_get_model_device_default_fallback(self):
        """Test get_model_device with default fallback when no device provided."""
        # Create mock model with no device info
        model = MagicMock()
        model.device = None
        model.get_device = None
        model.parameters.return_value = iter([])  # Empty iterator
        
        # Get device (should default to CPU or CUDA if available)
        device = get_model_device(model)
        
        # Check result is either cuda or cpu
        assert device in [torch.device("cuda"), torch.device("cpu")]

    def test_ensure_tensor_on_device_with_tensor(self):
        """Test ensure_tensor_on_device with a tensor input."""
        # Create tensor and target device
        tensor = torch.tensor([1, 2, 3])
        device = torch.device("cpu")
        
        # Mock the to() method to check if it's called
        original_to = torch.Tensor.to
        called_with_device = None
        
        def mock_to(self, device, *args, **kwargs):
            nonlocal called_with_device
            called_with_device = device
            return original_to(self, device, *args, **kwargs)
        
        # Replace to() method
        torch.Tensor.to = mock_to
        
        try:
            # Call function
            result = ensure_tensor_on_device(tensor, device)
            
            # Check result
            assert called_with_device == device
            assert result.device == device
        finally:
            # Restore original to() method
            torch.Tensor.to = original_to

    def test_ensure_tensor_on_device_with_non_tensor(self):
        """Test ensure_tensor_on_device with a non-tensor input."""
        # Test with different non-tensor inputs
        test_cases = [
            "string input",
            123,
            [1, 2, 3],
            {"key": "value"},
            None
        ]
        
        device = torch.device("cpu")
        
        for input_val in test_cases:
            # Call function
            result = ensure_tensor_on_device(input_val, device)
            
            # Should return input unchanged
            assert result == input_val 