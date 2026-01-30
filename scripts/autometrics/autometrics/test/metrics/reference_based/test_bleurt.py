import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from autometrics.metrics.reference_based.BLEURT import BLEURT

class TestBLEURT:
    """Test suite for the BLEURT metric."""

    @pytest.fixture
    def mock_bleurt_instance(self):
        """Create a BLEURT instance with mocked components for testing."""
        with patch('autometrics.metrics.reference_based.BLEURT.BleurtTokenizer') as mock_tokenizer, \
             patch('autometrics.metrics.reference_based.BLEURT.BleurtConfig') as mock_config, \
             patch('autometrics.metrics.reference_based.BLEURT.BleurtForSequenceClassification') as mock_model:
            
            # Create mock tokenizer
            tokenizer_mock = MagicMock()
            def tokenizer_side_effect(*args, **kwargs):
                return {
                    'input_ids': torch.tensor([[1, 2, 3]]),
                    'attention_mask': torch.tensor([[1, 1, 1]])
                }
            tokenizer_mock.side_effect = tokenizer_side_effect
            mock_tokenizer.from_pretrained.return_value = tokenizer_mock
            
            # Create mock config
            mock_config.from_pretrained.return_value = {}
            
            # Create mock model
            model_mock = MagicMock()
            model_output = MagicMock()
            model_output.logits = torch.tensor([[0.75]])
            model_mock.return_value = model_output
            model_mock.to.return_value = model_mock
            mock_model.from_pretrained.return_value = model_mock
            
            # Create instance
            bleurt = BLEURT(device='cpu')
            bleurt._load_model = MagicMock()
            bleurt.tokenizer = tokenizer_mock
            bleurt.model = model_mock
            
            yield bleurt

    @pytest.fixture
    def real_bleurt_instance(self):
        """Create a real BLEURT instance for integration testing."""
        try:
            # Create with small model and CPU for testing
            return BLEURT(device='cpu', model_name='lucadiliello/bleurt-tiny-512')
        except Exception as e:
            pytest.skip(f"Could not initialize real BLEURT instance: {str(e)}")

    def test_initialization(self):
        """Test that BLEURT initializes with correct parameters."""
        metric = BLEURT(
            name="CustomBLEURT", 
            description="Custom BLEURT description",
            model_name="lucadiliello/bleurt-tiny-512",
            device="cpu"
        )
        
        assert metric.name == "CustomBLEURT"
        assert metric.description == "Custom BLEURT description"
        assert metric.model_name == "lucadiliello/bleurt-tiny-512"
        assert metric.device == torch.device("cpu")

    def test_calculate_impl_basic(self, mock_bleurt_instance):
        """Test calculation for a single input/output pair."""
        # Model is already mocked in the fixture
        result = mock_bleurt_instance._calculate_impl("input text", "output text", ["reference text"])
        
        # Check result
        assert isinstance(result, float)
        # Check model was called
        mock_bleurt_instance.model.assert_called_once()

    def test_tokenization_length_limit(self, mock_bleurt_instance):
        """Test that tokenization handles long sequences properly."""
        # Create long input
        long_input = "word " * 600  # Generates a string with 600 words
        mock_output = "output"
        mock_reference = "reference"
        
        # Should not raise an error with truncation
        result = mock_bleurt_instance._calculate_impl(long_input, mock_output, [mock_reference])
        assert isinstance(result, float)
        mock_bleurt_instance.model.assert_called_once()

    @pytest.mark.integration
    def test_integration_basic_example(self, real_bleurt_instance):
        """Integration test with a real BLEURT instance using a basic example."""
        # Skip if BLEURT instance couldn't be initialized
        if real_bleurt_instance is None:
            pytest.skip("Real BLEURT instance not available")

        input_text = "The cat is on the mat."
        output_text = "A cat sits on a mat."
        references = ["The feline is resting on the rug."]
        
        try:
            # This should return a single score
            result = real_bleurt_instance._calculate_impl(input_text, output_text, references)
            
            # Basic checks on the result format
            assert isinstance(result, float)
            assert -1 <= result <= 1, "Score should be in range [-1,1]"
            
            # Check that the score is within 5% of the expected value
            expected_score = -0.32  # Actual value from test run
            assert abs(result - expected_score) <= 0.05, f"Score {result} differs by more than 5% from expected {expected_score}"
        except Exception as e:
            pytest.fail(f"Integration test failed: {str(e)}")

    @pytest.mark.integration
    def test_integration_complex_example(self, real_bleurt_instance):
        """Integration test with a real BLEURT instance using more complex text."""
        # Skip if BLEURT instance couldn't be initialized
        if real_bleurt_instance is None:
            pytest.skip("Real BLEURT instance not available")

        input_text = "The researchers published their findings in the latest issue of the journal."
        output_text = "The scientists released their discoveries in the most recent publication of the periodical."
        references = ["The research team published their results in the current issue of the journal."]
        
        try:
            # This should return a single score
            result = real_bleurt_instance._calculate_impl(input_text, output_text, references)
            
            # Basic checks on the result format
            assert isinstance(result, float)
            assert -1 <= result <= 1, "Score should be in range [-1,1]"
            
            # Check that the score is within 5% of the expected value
            expected_score = 0.438  # Actual value from test run
            assert abs(result - expected_score) <= 0.05, f"Score {result} differs by more than 5% from expected {expected_score}"
        except Exception as e:
            pytest.fail(f"Integration test failed: {str(e)}")

    @pytest.mark.integration
    def test_integration_long_input(self, real_bleurt_instance):
        """Integration test with a real BLEURT instance using a long input."""
        # Skip if BLEURT instance couldn't be initialized
        if real_bleurt_instance is None:
            pytest.skip("Real BLEURT instance not available")

        # Create a long input that would exceed the 512 token limit
        input_text = "word " * 300  # Should create more than 512 tokens when tokenized with reference
        output_text = "A summary of many words."
        references = ["This text talks about many words repeated."]
        
        try:
            # This should return a single score without raising an error
            result = real_bleurt_instance._calculate_impl(input_text, output_text, references)
            
            # Basic checks on the result format
            assert isinstance(result, float)
            assert -1 <= result <= 1, "Score should be in range [-1,1]"
        except Exception as e:
            pytest.fail(f"Integration test failed: {str(e)}") 