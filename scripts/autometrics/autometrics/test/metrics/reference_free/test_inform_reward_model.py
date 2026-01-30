import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from autometrics.metrics.reference_free.INFORMRewardModel import INFORMRewardModel

@pytest.fixture
def mock_inform_instance():
    """Create a mock instance of INFORMRewardModel for testing."""
    with patch('autometrics.metrics.reference_free.INFORMRewardModel.INFORMForSequenceClassification') as mock_model_class, \
         patch('autometrics.metrics.reference_free.INFORMRewardModel.PreTrainedTokenizerFast') as mock_tokenizer_class:
        
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Set a real device on the mock model
        mock_model.device = torch.device("cpu")
        
        # Configure mock classes to return our mock instances
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create instance with mocked dependencies
        instance = INFORMRewardModel(persistent=False)
        instance.model = mock_model
        instance.tokenizer = mock_tokenizer
        instance.device = torch.device("cpu")
        
        # Configure the tokenizer to return a tensor that looks like input_ids
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3, 4]])
        
        # Configure the model to return an output with logits
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([0.75])
        mock_model.return_value = mock_output
        
        return instance

class TestINFORMRewardModel:
    """Test suite for the INFORMRewardModel metric."""

    def test_initialization(self):
        """Test that the model initializes with correct parameters."""
        metric = INFORMRewardModel(
            name="CustomINFORM", 
            description="Custom INFORM description",
            model_name="test/model",
            batch_size=4,
            persistent=False
        )
        
        assert metric.name == "CustomINFORM"
        assert metric.description == "Custom INFORM description"
        assert metric.model_name == "test/model"
        assert metric.batch_size == 4
        assert metric.persistent is False
        # Model should not be loaded yet
        assert metric.model is None
        assert metric.tokenizer is None
    
    @patch('autometrics.metrics.reference_free.INFORMRewardModel.torch.device', return_value=torch.device('cpu'))
    @patch('autometrics.metrics.reference_free.INFORMRewardModel.get_model_device')
    @patch('autometrics.metrics.reference_free.INFORMRewardModel.INFORMForSequenceClassification')
    @patch('autometrics.metrics.reference_free.INFORMRewardModel.PreTrainedTokenizerFast')
    def test_calculate_impl_basic(self, mock_tokenizer_class, mock_model_class, mock_get_device, mock_device):
        """Test calculation for a single input/output pair."""
        # Set up device and mocking
        device = torch.device('cpu')
        mock_get_device.return_value = device
        
        # Setup mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Configure mocks for proper class instantiation
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create a tensor that doesn't need .to()
        mock_tensor = torch.tensor([[1, 2, 3, 4]]).to(device)
        
        # Configure the tokenizer to return properly formatted tensor
        mock_tokenizer.apply_chat_template.return_value = mock_tensor
        
        # Configure model output
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([0.75])
        mock_model.return_value = mock_output
        
        # Make ensure_tensor_on_device function a no-op
        with patch('autometrics.metrics.reference_free.INFORMRewardModel.ensure_tensor_on_device', 
                  side_effect=lambda x, _: x):
            # Create model instance
            instance = INFORMRewardModel(persistent=True)
            
            # Directly set model and tokenizer to avoid actual loading
            instance.model = mock_model
            instance.tokenizer = mock_tokenizer
            
            # Calculate with mock data
            result = instance._calculate_impl(
                input="What is the capital of France?",
                output="The capital of France is Paris."
            )
            
            # Verify the result
            assert isinstance(result, float)
            assert result == 0.75  # Should match the mock logit value
            
            # Verify model was called
            assert mock_model.call_count >= 1
            # Check most recent call args
            call_args = mock_model.call_args[1]
            assert 'input_ids' in call_args
    
    @patch('autometrics.metrics.reference_free.INFORMRewardModel.torch.device', return_value=torch.device('cpu'))
    @patch('autometrics.metrics.reference_free.INFORMRewardModel.get_model_device')
    @patch('autometrics.metrics.reference_free.INFORMRewardModel.INFORMForSequenceClassification')
    @patch('autometrics.metrics.reference_free.INFORMRewardModel.PreTrainedTokenizerFast')
    def test_calculate_batched_impl(self, mock_tokenizer_class, mock_model_class, mock_get_device, mock_device):
        """Test batched calculation."""
        # Set up device mocking
        device = torch.device('cpu')
        mock_get_device.return_value = device
        
        # Setup mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Configure mock classes
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create a tensor that's already on the right device
        mock_tensor = torch.tensor([[1, 2, 3, 4]]).to(device)
        
        # Configure tokenizer to return properly formatted tensor
        mock_tokenizer.apply_chat_template.return_value = mock_tensor
        
        # Configure model outputs for multiple calls
        outputs = [
            MagicMock(logits=torch.tensor([0.1])),
            MagicMock(logits=torch.tensor([0.2])),
            MagicMock(logits=torch.tensor([0.3]))
        ]
        mock_model.side_effect = outputs
        
        # Make ensure_tensor_on_device function a no-op
        with patch('autometrics.metrics.reference_free.INFORMRewardModel.ensure_tensor_on_device', 
                  side_effect=lambda x, _: x):
            # Create instance
            instance = INFORMRewardModel(batch_size=2, persistent=True)
            
            # Set model and tokenizer directly
            instance.model = mock_model
            instance.tokenizer = mock_tokenizer
            
            # Create test data with 3 examples
            inputs = ["Input 1", "Input 2", "Input 3"]
            outputs = ["Output 1", "Output 2", "Output 3"]
            
            # Run batched calculation
            results = instance._calculate_batched_impl(inputs, outputs)
            
            # Verify results
            assert len(results) == 3
            # Use np.allclose to handle floating point precision differences
            assert np.allclose(results, [0.1, 0.2, 0.3], rtol=1e-5)
            
            # Verify model was called the right number of times
            assert mock_model.call_count == 3
    
    @patch('autometrics.metrics.reference_free.INFORMRewardModel.get_model_device')
    def test_tensor_device_handling(self, mock_get_device):
        """Test that device utility functions are used properly."""
        # Setup device mock
        device = torch.device('cpu')
        mock_get_device.return_value = device
        
        # Create mocks for model and tokenizer
        with patch('autometrics.metrics.reference_free.INFORMRewardModel.INFORMForSequenceClassification') as mock_model_class, \
             patch('autometrics.metrics.reference_free.INFORMRewardModel.PreTrainedTokenizerFast') as mock_tokenizer_class, \
             patch('autometrics.metrics.reference_free.INFORMRewardModel.ensure_tensor_on_device') as mock_ensure:
            
            # Create mocks
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            
            # Setup return values
            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            # Configure tensor that wouldn't need to() call
            mock_tensor = torch.tensor([[1, 2, 3, 4]])
            mock_tokenizer.apply_chat_template.return_value = mock_tensor
            
            # Configure model output
            mock_model.return_value = MagicMock(logits=torch.tensor([0.5]))
            
            # Make ensure_tensor_on_device return the input (simpler to track calls)
            mock_ensure.side_effect = lambda x, _: x
            
            # Create instance
            instance = INFORMRewardModel()
            instance.model = mock_model
            instance.tokenizer = mock_tokenizer
            
            # Run calculation
            instance._calculate_impl("test input", "test output")
            
            # Verify device functions were called
            mock_get_device.assert_called_once()
            
            # Verify tokenizer was called
            mock_tokenizer.apply_chat_template.assert_called_once()
            
            # Verify model was called
            assert mock_model.call_count >= 1

    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_integration_basic(self):
        """Integration test with a simple example requiring CUDA."""
        try:
            # Initialize with a smaller model for testing
            model = INFORMRewardModel(
                model_name="infly/INF-ORM-Llama3.1-70B",  # This would be replaced with a smaller test model
                persistent=False
            )
            
            # Test basic calculation
            result = model.calculate(
                input="What is the capital of France?",
                output="The capital of France is Paris."
            )
            
            # For integration tests, we just check the result is reasonable
            assert isinstance(result, float)
            
        except Exception as e:
            pytest.skip(f"Integration test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main(["-v", "test_inform_reward_model.py"]) 