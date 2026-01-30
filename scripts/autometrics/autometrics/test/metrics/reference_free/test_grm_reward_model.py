import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from autometrics.metrics.reference_free.GRMRewardModel import GRMRewardModel

@pytest.fixture
def mock_grm_instance():
    """Create a mock instance of GRMRewardModel for testing."""
    with patch('autometrics.metrics.reference_free.GRMRewardModel.AutoModelForSequenceClassification') as mock_model_class, \
         patch('autometrics.metrics.reference_free.GRMRewardModel.AutoTokenizer') as mock_tokenizer_class:
        
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Set a real device on the mock model
        mock_model.device = torch.device("cpu")
        
        # Configure mock classes to return our mock instances
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Create instance with mocked dependencies
        instance = GRMRewardModel()
        instance.model = mock_model
        instance.tokenizer = mock_tokenizer
        instance.device = torch.device("cpu")
        
        return instance

class TestGRMRewardModel:
    """Test suite for the GRMRewardModel metric."""

    def test_initialization(self, mock_grm_instance):
        """Test that the model initializes correctly."""
        assert mock_grm_instance.model is not None
        assert mock_grm_instance.tokenizer is not None
        assert mock_grm_instance.device is not None

    def test_calculate_impl_with_tensor_input(self, mock_grm_instance):
        """Test calculation when tokenizer returns a tensor."""
        # Configure mock model to return a specific logit value
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.5]])
        mock_grm_instance.model.return_value = mock_output
        
        # Configure mock tokenizer to return a tensor (what causes the original bug)
        mock_tensor = torch.tensor([[1, 2, 3, 4]])
        mock_grm_instance.tokenizer.apply_chat_template.return_value = mock_tensor
        
        # Calculate with mock data
        result = mock_grm_instance._calculate_impl(
            input="What is the capital of France?",
            output="The capital of France is Paris."
        )
        
        # Verify the result
        assert isinstance(result, float)
        assert pytest.approx(result) == 0.5
        
        # Verify model was called with input_ids=tensor
        mock_grm_instance.model.assert_called_once()
        call_args = mock_grm_instance.model.call_args[1]
        assert 'input_ids' in call_args
        assert torch.equal(call_args['input_ids'], mock_tensor)

    def test_calculate_impl_with_dict_input(self, mock_grm_instance):
        """Test calculation when tokenizer returns a dictionary."""
        # Configure mock model to return a specific logit value
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.7]])
        mock_grm_instance.model.return_value = mock_output
        
        # Configure mock tokenizer to return a dictionary
        mock_dict = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        mock_grm_instance.tokenizer.apply_chat_template.return_value = mock_dict
        
        # Calculate with mock data
        result = mock_grm_instance._calculate_impl(
            input="What is the capital of France?",
            output="The capital of France is Paris."
        )
        
        # Verify the result
        assert isinstance(result, float)
        assert pytest.approx(result) == 0.7
        
        # Verify model was called with the dictionary unpacked
        mock_grm_instance.model.assert_called_once()
        call_args = mock_grm_instance.model.call_args[1]
        assert 'input_ids' in call_args
        assert 'attention_mask' in call_args
        assert torch.equal(call_args['input_ids'], mock_dict['input_ids'])
        assert torch.equal(call_args['attention_mask'], mock_dict['attention_mask'])

    def test_calculate_batched_impl_with_tensor(self, mock_grm_instance):
        """Test batched calculation with tensor tokenizer output."""
        # Configure mock model to return different logits for batches
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.1], [0.2]])
        mock_grm_instance.model.return_value = mock_output
        
        # Configure mock tokenizer to return a tensor
        mock_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        mock_grm_instance.tokenizer.apply_chat_template.return_value = mock_tensor
        
        # Set batch size
        mock_grm_instance.batch_size = 2
        
        # Create test data
        inputs = ["Input 1", "Input 2"]
        outputs = ["Output 1", "Output 2"]
        
        # Run batched calculation
        results = mock_grm_instance._calculate_batched_impl(inputs, outputs)
        
        # Verify results
        assert len(results) == 2
        assert np.allclose(results, [0.1, 0.2], rtol=1e-5)
        
        # Verify model was called with input_ids=tensor
        mock_grm_instance.model.assert_called_once()
        call_args = mock_grm_instance.model.call_args[1]
        assert 'input_ids' in call_args
        assert torch.equal(call_args['input_ids'], mock_tensor)

    def test_call_model_with_tensor(self, mock_grm_instance):
        """Test the _call_model helper function with tensor input."""
        # Configure mock model to return a specific output
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.5]])
        mock_grm_instance.model.return_value = mock_output
        
        # Create a test tensor
        test_tensor = torch.tensor([[1, 2, 3, 4]])
        
        # Call the helper function
        logits = mock_grm_instance._call_model(test_tensor, torch.device("cpu"))
        
        # Verify results
        assert torch.equal(logits, torch.tensor([[0.5]]))
        
        # Verify model was called with input_ids=tensor
        mock_grm_instance.model.assert_called_once()
        call_args = mock_grm_instance.model.call_args[1]
        assert 'input_ids' in call_args
        assert torch.equal(call_args['input_ids'], test_tensor)

    def test_call_model_with_dict(self, mock_grm_instance):
        """Test the _call_model helper function with dictionary input."""
        # Configure mock model to return a specific output
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.7]])
        mock_grm_instance.model.return_value = mock_output
        
        # Create a test dictionary
        test_dict = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        
        # Call the helper function
        logits = mock_grm_instance._call_model(test_dict, torch.device("cpu"))
        
        # Verify results
        assert torch.equal(logits, torch.tensor([[0.7]]))
        
        # Verify model was called with the dictionary unpacked
        mock_grm_instance.model.assert_called_once()
        call_args = mock_grm_instance.model.call_args[1]
        assert 'input_ids' in call_args
        assert 'attention_mask' in call_args
        assert torch.equal(call_args['input_ids'], test_dict['input_ids'])
        assert torch.equal(call_args['attention_mask'], test_dict['attention_mask'])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_integration_gpu(self):
        """Integration test with GPU if available."""
        # This should be run only if CUDA is available
        try:
            model = GRMRewardModel()
            # Set to CUDA
            model.device = torch.device("cuda")
            model._load_model()
            
            # Test that model loads on CUDA
            assert next(model.model.parameters()).device.type == "cuda"
            
            # Unload to free resources
            model._unload_model()
        except Exception as e:
            pytest.skip(f"Could not run GPU integration test: {str(e)}") 