import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from autometrics.metrics.reference_free.PRMRewardModel import MathProcessRewardModel

class TestMathProcessRewardModel:
    """Test suite for the MathProcessRewardModel metric."""

    @pytest.fixture
    def mock_prm_instance(self):
        """Create a MathProcessRewardModel instance with mocked components for testing."""
        with patch('autometrics.metrics.reference_free.PRMRewardModel.AutoTokenizer') as mock_tokenizer, \
             patch('autometrics.metrics.reference_free.PRMRewardModel.AutoModel') as mock_model, \
             patch('autometrics.metrics.reference_free.PRMRewardModel.nltk') as mock_nltk:
            
            # Configure mock tokenizer
            tokenizer_instance = MagicMock()
            tokenizer_instance.apply_chat_template.return_value = "mocked chat template"
            tokenizer_instance.encode.return_value = torch.tensor([[1, 2, 3, 4]])
            mock_tokenizer.from_pretrained.return_value = tokenizer_instance
            
            # Configure mock model
            model_instance = MagicMock()
            model_output = MagicMock()
            # Create dummy logits with shape [batch_size, seq_len, num_classes]
            model_output.logits = torch.tensor([[[0.1, 0.9], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]]])
            model_instance.return_value = model_output
            model_instance.to.return_value = model_instance
            model_instance.eval.return_value = model_instance
            # Mock the device property
            model_instance.device = torch.device("cpu")
            mock_model.from_pretrained.return_value = model_instance
            
            # Configure mock nltk
            mock_nltk.sent_tokenize.return_value = ["This is a sentence.", "This is another sentence."]
            
            # Create instance with CPU to avoid CUDA issues in tests
            prm = MathProcessRewardModel(device_map="cpu")
            # Force pre-loading of mocked components
            prm._load_model()
            
            yield prm

    def test_initialization(self):
        """Test that MathProcessRewardModel initializes with correct parameters."""
        metric = MathProcessRewardModel(
            name="CustomPRM", 
            description="Custom PRM description",
            model_name="Qwen/Qwen2.5-Math-PRM-7B",
            device_map="cpu",
            persistent=False
        )
        
        assert metric.name == "CustomPRM"
        assert metric.description == "Custom PRM description"
        assert metric.model_name == "Qwen/Qwen2.5-Math-PRM-7B"
        assert metric.device_map == "cpu"
        assert metric.persistent is False
        assert metric.submetric_names == ["min", "max", "mean"]

    def test_calculate_impl_basic(self, mock_prm_instance):
        """Test the basic calculation functionality with mocked components."""
        # Configure the mocked tokenizer.encode to return a test separator token id
        mock_prm_instance.tokenizer.encode.side_effect = [
            torch.tensor([[1, 2, 3, 4]]),  # For the main input
            torch.tensor([3])              # For the separator token
        ]
        
        # Calculate with mock data
        result = mock_prm_instance._calculate_impl(
            input_text="What is 2+2?", 
            output="I will add 2 and 2. The result is 4."
        )
        
        # Check result format (min, max, mean)
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        # Verify results are in the expected range
        assert all(0 <= score <= 1 for score in result)
        
        # Check model and tokenizer were called
        mock_prm_instance.tokenizer.apply_chat_template.assert_called_once()
        mock_prm_instance.tokenizer.encode.assert_called()
        mock_prm_instance.model.assert_called_once()

    def test_device_handling(self, mock_prm_instance):
        """Test that the model handles devices correctly."""
        # Mock the tensor.to() method to test device logic
        original_to = torch.Tensor.to
        
        try:
            # Create a test class for checking if to() was called with expected device
            class DeviceCheck:
                def __init__(self):
                    self.called_with_device = None
                
                def mock_to(self, obj, device, *args, **kwargs):
                    self.called_with_device = device
                    return original_to(obj, device, *args, **kwargs)
            
            checker = DeviceCheck()
            
            # Patch tensor.to method
            torch.Tensor.to = lambda obj, device, *args, **kwargs: checker.mock_to(obj, device, *args, **kwargs)
            
            # Set a test device on the mock model
            test_device = torch.device("cpu")
            mock_prm_instance.model.device = test_device
            
            # Calculate with mock data
            mock_prm_instance._calculate_impl("test input", "test output")
            
            # Check that tensor was moved to the model's device
            assert checker.called_with_device == test_device
        
        finally:
            # Restore original tensor.to
            torch.Tensor.to = original_to

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_integration_cuda_device_management(self):
        """
        Integration test for proper CUDA device management.
        
        This test is skipped if CUDA is not available.
        """
        # Skip this test explicitly if we don't have GPU
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Create a real instance with default auto device mapping
        prm = MathProcessRewardModel(persistent=False)
        
        try:
            # In a real setting, this would run the model, but it might take
            # too much memory, so we'll just check that the model loads
            prm._load_model()
            
            # Test if model loaded successfully
            assert prm.model is not None
            assert prm.tokenizer is not None
            
            # Cleanup
            prm._unload_model()
        except (ImportError, RuntimeError) as e:
            # Skip if dependencies are missing or out of CUDA memory
            pytest.skip(f"Could not run CUDA test: {str(e)}") 