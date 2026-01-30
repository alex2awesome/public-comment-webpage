import pytest
import torch
from unittest.mock import patch, MagicMock
from autometrics.metrics.reference_free.LENS_SALSA import LENS_SALSA

class TestLENS_SALSA:
    """Test suite for the LENS_SALSA metric."""

    @pytest.fixture
    def mock_lens_salsa_instance(self):
        """Create a LENS_SALSA instance with mocked model for testing."""
        with patch('autometrics.metrics.reference_free.LENS_SALSA.download_model') as mock_download, \
             patch('autometrics.metrics.reference_free.LENS_SALSA._LENS_SALSA_Model') as mock_model_class:
            
            # Create a mock model instance
            mock_model = MagicMock()
            mock_model.source_column = "src"
            mock_model.target_column = "edit_id_simplified"
            
            # Set up the mock prediction result
            mock_prediction = MagicMock()
            mock_prediction.scores = [0.85]  # Default mock score
            
            # Configure the mock model's predict method
            mock_model.model = MagicMock()
            mock_model.model.predict.return_value = mock_prediction
            
            # Configure the tokenizer
            mock_encoder = MagicMock()
            mock_encoder.tokenizer.tokenize.return_value = ["mocked", "tokens"]
            mock_model.model.encoder = mock_encoder
            
            # Make the model class return our mock model
            mock_model_class.return_value = mock_model
            
            # Create and return an instance of LENS_SALSA
            lens_salsa = LENS_SALSA(persistent=True)
            lens_salsa._load_model()  # Force loading the mocked model
            
            return lens_salsa
    
    def test_initialization(self):
        """Test that LENS_SALSA initializes properly with default and custom parameters."""
        # Test with default parameters
        default_metric = LENS_SALSA()
        assert default_metric.name == "LENS_SALSA"
        assert default_metric.model_id == "davidheineman/lens-salsa"
        assert default_metric.batch_size == 16
        assert default_metric.persistent is True
        assert default_metric.max_length == 512
        assert default_metric.max_retries == 3
        assert default_metric.model_token_limit == 512
        assert default_metric.model is None
        
        # Test with custom parameters
        custom_metric = LENS_SALSA(
            name="test_lens_salsa", 
            description="Test description",
            model_id="test/model",
            batch_size=32,
            devices=[0],
            persistent=False,
            max_length=256,
            max_retries=5
        )
        
        assert custom_metric.name == "test_lens_salsa"
        assert custom_metric.description == "Test description"
        assert custom_metric.model_id == "test/model"
        assert custom_metric.batch_size == 32
        assert custom_metric.devices == [0]
        assert custom_metric.persistent is False
        assert custom_metric.max_length == 256
        assert custom_metric.max_retries == 5
        assert custom_metric.model is None

    def test_get_tokenizer_length(self, mock_lens_salsa_instance):
        """Test the tokenizer length estimation functionality."""
        # Test with model tokenizer (mocked)
        mock_lens_salsa_instance.model.model.encoder.tokenizer.tokenize.return_value = ["test", "tok", "##en", "##iz", "##er"]
        length = mock_lens_salsa_instance._get_tokenizer_length("test tokenizer")
        assert length == 5
        
        # Test the fallback estimation without model
        metric = LENS_SALSA()
        assert metric.model is None
        length = metric._get_tokenizer_length("testing fallback estimation")
        # Should use word count * 2 as estimate
        assert length == 6

    def test_truncate_text(self, mock_lens_salsa_instance):
        """Test the text truncation functionality with tokenizer awareness."""
        # Mock tokenizer to return specific lengths
        def tokenize_mock(text):
            # Return 1 token per word for simplicity in testing
            return text.split()
        
        mock_lens_salsa_instance.model.model.encoder.tokenizer.tokenize = tokenize_mock
        
        # Test with empty text
        assert mock_lens_salsa_instance._truncate_text("") == ""
        
        # Test with short text (under token limit)
        short_text = "This is short."
        mock_lens_salsa_instance._get_tokenizer_length = lambda text: len(text.split())
        assert mock_lens_salsa_instance._truncate_text(short_text, 10) == short_text
        
        # Test with text just at token limit
        text_at_limit = "One two three four five"
        assert mock_lens_salsa_instance._truncate_text(text_at_limit, 5) == text_at_limit
        
        # Test with text exceeding token limit
        long_text = "One two three four five six seven"
        truncated = mock_lens_salsa_instance._truncate_text(long_text, 5)
        assert truncated == "One two three four five"
        
        # Test with custom max_tokens
        assert mock_lens_salsa_instance._truncate_text(long_text, 3) == "One two three"

    def test_calculate_with_fallback_success(self, mock_lens_salsa_instance):
        """Test successful calculation with the first attempt."""
        result = mock_lens_salsa_instance._calculate_with_fallback("input text", "output text")
        
        # Should succeed on first try
        mock_lens_salsa_instance.model.model.predict.assert_called_once()
        assert result == 85.0  # From the mocked score

    def test_calculate_with_fallback_retry(self, mock_lens_salsa_instance):
        """Test calculation with retry on failure."""
        # Make first attempt fail, second succeed
        mock_lens_salsa_instance.model.model.predict.side_effect = [
            Exception("Padding error"),  # First attempt fails
            MagicMock(scores=[0.75])     # Second attempt succeeds
        ]
        
        result = mock_lens_salsa_instance._calculate_with_fallback("input text", "output text")
        
        # Should have called predict twice
        assert mock_lens_salsa_instance.model.model.predict.call_count == 2
        assert result == 75.0

    def test_calculate_with_fallback_all_fails(self, mock_lens_salsa_instance):
        """Test when all fallback attempts fail."""
        # Make all attempts fail
        mock_lens_salsa_instance.model.model.predict.side_effect = Exception("Persistent error")
        mock_lens_salsa_instance.max_retries = 2
        
        # Should raise an exception after exhausting all retries
        with pytest.raises(Exception):
            mock_lens_salsa_instance._calculate_with_fallback("input text", "output text")
        
        # Should have called predict max_retries times
        assert mock_lens_salsa_instance.model.model.predict.call_count == 2

    def test_calculate_impl_normal_case(self, mock_lens_salsa_instance):
        """Test the basic functionality of calculate_impl with normal inputs."""
        # Mock _calculate_with_fallback to isolate the test
        mock_lens_salsa_instance._calculate_with_fallback = MagicMock(return_value=85.0)
        
        score = mock_lens_salsa_instance._calculate_impl("input text", "output text")
        
        # Should call the fallback method
        mock_lens_salsa_instance._calculate_with_fallback.assert_called_once_with("input text", "output text")
        assert score == 85.0

    def test_calculate_impl_with_exception(self, mock_lens_salsa_instance):
        """Test that calculate_impl properly propagates exceptions."""
        # Mock _calculate_with_fallback to raise an exception
        mock_lens_salsa_instance._calculate_with_fallback = MagicMock(
            side_effect=Exception("Test failure")
        )
        
        # Should raise a RuntimeError with appropriate message
        with pytest.raises(RuntimeError) as excinfo:
            mock_lens_salsa_instance._calculate_impl("input text", "output text")
        
        assert "LENS_SALSA failed after" in str(excinfo.value)
        assert "Test failure" in str(excinfo.value)

    def test_calculate_batched_impl_normal_case(self, mock_lens_salsa_instance):
        """Test the batched functionality with normal inputs."""
        # Mock _calculate_with_fallback to return different values for different inputs
        mock_lens_salsa_instance._calculate_with_fallback = MagicMock(side_effect=[85.0, 72.0])
        
        inputs = ["input1", "input2"]
        outputs = ["output1", "output2"]
        
        scores = mock_lens_salsa_instance._calculate_batched_impl(inputs, outputs)
        
        # Should call _calculate_with_fallback for each input
        assert mock_lens_salsa_instance._calculate_with_fallback.call_count == 2
        assert scores == [85.0, 72.0]

    def test_calculate_batched_impl_with_exceptions(self, mock_lens_salsa_instance):
        """Test batched calculation with some failures."""
        # Mock _calculate_with_fallback to succeed for first input, fail for second
        mock_lens_salsa_instance._calculate_with_fallback = MagicMock(side_effect=[
            85.0,
            Exception("Test failure")
        ])
        
        inputs = ["input1", "input2"]
        outputs = ["output1", "output2"]
        
        # Should raise a RuntimeError with appropriate message
        with pytest.raises(RuntimeError) as excinfo:
            mock_lens_salsa_instance._calculate_batched_impl(inputs, outputs)
        
        assert "LENS_SALSA failed on 1 examples" in str(excinfo.value)
        assert "Example 1: Test failure" in str(excinfo.value)

    def test_model_lifecycle(self, mock_lens_salsa_instance):
        """Test that model loading and unloading work correctly."""
        # Reset the mock functions for this test
        mock_lens_salsa_instance._load_model = MagicMock()
        mock_lens_salsa_instance._unload_model = MagicMock()
        mock_lens_salsa_instance._calculate_with_fallback = MagicMock(return_value=85.0)
        
        # Test with persistent=True
        mock_lens_salsa_instance.persistent = True
        mock_lens_salsa_instance.model = None
        
        # Should load model if it's None
        result = mock_lens_salsa_instance._calculate_impl("input", "output")
        mock_lens_salsa_instance._load_model.assert_called_once()
        mock_lens_salsa_instance._unload_model.assert_not_called()
        assert result == 85.0
        
        # Reset mocks
        mock_lens_salsa_instance._load_model.reset_mock()
        mock_lens_salsa_instance._unload_model.reset_mock()
        
        # Test with persistent=False
        mock_lens_salsa_instance.persistent = False
        
        result = mock_lens_salsa_instance._calculate_impl("input", "output")
        mock_lens_salsa_instance._unload_model.assert_called_once()
        assert result == 85.0
        
        # Test error case with persistent=False
        mock_lens_salsa_instance._unload_model.reset_mock()
        mock_lens_salsa_instance._calculate_with_fallback = MagicMock(side_effect=Exception("Test error"))
        
        with pytest.raises(RuntimeError):
            mock_lens_salsa_instance._calculate_impl("input", "output")
        
        # Should still unload the model even if an error occurs
        mock_lens_salsa_instance._unload_model.assert_called_once()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for integration test")
    def test_integration_with_real_model(self):
        """Integration test with real model - requires GPU."""
        # This test will be skipped if no GPU is available
        try:
            metric = LENS_SALSA(persistent=False)
            
            input_text = "The phenomenon was inexplicable to the researchers."
            output_text = "The researchers couldn't explain it."
            
            score = metric.calculate(input_text, output_text)
            
            # Just check that we get a reasonable score
            assert isinstance(score, float)
            assert 0 <= score <= 100
        except ImportError:
            pytest.skip("Lens library not available")
            
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for integration test")
    def test_integration_with_long_text(self):
        """Integration test with very long text that requires fallback."""
        # This test will be skipped if no GPU is available
        try:
            metric = LENS_SALSA(persistent=False)
            
            # Generate a very long input text that would exceed model limits
            input_text = "This is a very long text. " * 100
            output_text = "This is a simplified version. " * 50
            
            score = metric.calculate(input_text, output_text)
            
            # Just check that we get a reasonable score
            assert isinstance(score, float)
            assert 0 <= score <= 100
        except ImportError:
            pytest.skip("Lens library not available") 