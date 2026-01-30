import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from autometrics.metrics.reference_based.ParaScore import ParaScore

class TestParaScore:
    """Test suite for the ParaScore metric."""

    @pytest.fixture
    def mock_parascore_instance(self):
        """Create a ParaScore instance with mocked ParaScorer for testing."""
        with patch('autometrics.metrics.reference_based.ParaScore.ParaScorer') as mock_scorer:
            mock_instance = ParaScore()
            # Configure mock to avoid actual model loading
            mock_instance.scorer = mock_scorer.return_value
            yield mock_instance

    @pytest.fixture
    def parascore_instance(self):
        """Create a real ParaScore instance for integration testing."""
        # Note: This will try to load the actual model, which might be slow
        # and require dependencies. Mock version above is preferred for unit tests.
        try:
            return ParaScore()
        except Exception as e:
            pytest.skip(f"Could not initialize real ParaScore instance: {str(e)}")

    def test_initialization(self):
        """Test that ParaScore initializes with correct parameters."""
        metric = ParaScore(name="CustomName", 
                           description="Custom description",
                           lang="en",
                           model_type="bert-base-uncased")
        
        assert metric.name == "CustomName"
        assert metric.description == "Custom description"
        assert metric.scorer is not None

    def test_calculate_impl_normal(self, mock_parascore_instance):
        """Test normal calculation works correctly."""
        # Setup mock to return expected base_score
        mock_parascore_instance.scorer.base_score.return_value = [0.75]
        
        # Test calculation
        input_text = "This is a source sentence."
        output_text = "This is a paraphrase of the source."
        refs = ["This is a reference."]
        
        result = mock_parascore_instance._calculate_impl(input_text, output_text, refs)
        
        # Check the result
        assert result == 0.75
        
        # Check that ParaScorer was called correctly
        mock_parascore_instance.scorer.base_score.assert_called_once()
        call_args = mock_parascore_instance.scorer.base_score.call_args[0]
        assert call_args[0] == [output_text]  # candidates
        assert call_args[1] == [input_text]   # sources
        assert call_args[2] == [refs]         # references
    
    def test_calculate_impl_multi_reference(self, mock_parascore_instance):
        """Test calculation with multiple references."""
        # Setup mock return
        mock_parascore_instance.scorer.base_score.return_value = [0.85]
        
        # Test with multiple references
        input_text = "This is a source sentence."
        output_text = "This is a paraphrase of the source."
        refs = [["Ref 1 for 1st sample.", "Ref 2 for 1st sample."]]
        
        result = mock_parascore_instance._calculate_impl(input_text, output_text, refs)
        
        # Check result
        assert result == 0.85
    
    def test_calculate_impl_no_references(self, mock_parascore_instance):
        """Test that an error is raised when no references are provided."""
        with pytest.raises(ValueError, match="ParaScore .* requires `references`"):
            mock_parascore_instance._calculate_impl("input", "output", None)
        
        with pytest.raises(ValueError, match="ParaScore .* requires `references`"):
            mock_parascore_instance._calculate_impl("input", "output", [])

    def test_calculate_impl_empty_result(self, mock_parascore_instance):
        """Test handling when underlying score returns empty list."""
        # Setup mock to return an empty list
        mock_parascore_instance.scorer.base_score.return_value = []
        
        input_text = "This is a source sentence."
        output_text = "This is a paraphrase of the source."
        refs = ["This is a reference."]
        
        # After our fix, this should handle empty lists gracefully
        result = mock_parascore_instance._calculate_impl(input_text, output_text, refs)
        
        # Check that we get the expected fallback value
        assert result == 0.0

    def test_calculate_batched(self, mock_parascore_instance):
        """Test batched calculation."""
        # Setup mock to return batch results
        mock_parascore_instance.scorer.base_score.return_value = [0.7, 0.8]
        
        inputs = ["Source 1", "Source 2"]
        outputs = ["Paraphrase 1", "Paraphrase 2"]
        references = [["Ref for 1"], ["Ref for 2"]]
        
        results = mock_parascore_instance.calculate_batched(inputs, outputs, references)
        
        # Check the results - should be a list of scores
        assert len(results) == 2
        assert results == [0.7, 0.8]
        
        # Check that ParaScorer was called correctly
        mock_parascore_instance.scorer.base_score.assert_called_once()
        assert mock_parascore_instance.scorer.base_score.call_args[0][0] == outputs
        assert mock_parascore_instance.scorer.base_score.call_args[0][1] == inputs
        assert mock_parascore_instance.scorer.base_score.call_args[0][2] == references

    def test_calculate_batched_empty_references(self, mock_parascore_instance):
        """Test batched calculation with empty references."""
        # Setup mock
        mock_parascore_instance.scorer.base_score.return_value = [0.7, 0.8]
        
        inputs = ["Source 1", "Source 2"]
        outputs = ["Paraphrase 1", "Paraphrase 2"]
        references = None
        
        results = mock_parascore_instance.calculate_batched(inputs, outputs, references)
        
        # Check the call generated empty refs for each input
        expected_refs = [[] for _ in inputs]
        mock_parascore_instance.scorer.base_score.assert_called_once()
        assert mock_parascore_instance.scorer.base_score.call_args[0][2] == expected_refs

    @pytest.mark.integration
    def test_integration_real_score(self, parascore_instance):
        """Integration test with a real ParaScore instance."""
        # Skip if parascore_instance couldn't be initialized
        if parascore_instance is None:
            pytest.skip("Real ParaScore instance not available")

        input_text = "The cat is sitting on the mat."
        output_text = "A feline is resting on the floor covering."
        references = ["A cat is on the mat.", "The feline sits on the rug."]
        
        try:
            # This should return a single score
            result = parascore_instance._calculate_impl(input_text, output_text, references)
            
            # Basic checks on the result format
            assert isinstance(result, float)
            assert 0 <= result <= 1, "Score should be in range [0,1]"
        except Exception as e:
            pytest.fail(f"Integration test failed: {str(e)}")

    @pytest.mark.integration
    def test_integration_cat_example(self, parascore_instance):
        """Integration test with a real ParaScore instance using a cat example."""
        # Skip if parascore_instance couldn't be initialized
        if parascore_instance is None:
            pytest.skip("Real ParaScore instance not available")

        input_text = "The cat is sitting on the mat."
        output_text = "A feline is resting on the floor covering."
        references = ["A cat is on the mat.", "The feline sits on the rug."]
        
        try:
            # This should return a single score
            result = parascore_instance._calculate_impl(input_text, output_text, references)
            
            # Basic checks on the result format
            assert isinstance(result, float)
            assert 0 <= result <= 1, "Score should be in range [0,1]"
            
            # Check that the score is within 1% of the expected value
            expected_score = 0.7805029320716858  # This value was obtained by running the test
            assert abs(result - expected_score) <= 0.01, f"Score {result} differs by more than 1% from expected {expected_score}"
        except Exception as e:
            pytest.fail(f"Integration test failed: {str(e)}")

    @pytest.mark.integration
    def test_integration_weather_example(self, parascore_instance):
        """Integration test with a real ParaScore instance using a weather example."""
        # Skip if parascore_instance couldn't be initialized
        if parascore_instance is None:
            pytest.skip("Real ParaScore instance not available")

        input_text = "The weather forecast predicts heavy rain tomorrow."
        output_text = "Tomorrow's forecast indicates significant precipitation."
        references = ["Heavy rainfall is expected tomorrow.", "The forecast shows rain tomorrow."]
        
        try:
            # This should return a single score
            result = parascore_instance._calculate_impl(input_text, output_text, references)
            
            # Basic checks on the result format
            assert isinstance(result, float)
            assert 0 <= result <= 1, "Score should be in range [0,1]"
            
            # Check that the score is within 1% of the expected value
            expected_score = 0.790425853729248  # This value was obtained by running the test
            assert abs(result - expected_score) <= 0.01, f"Score {result} differs by more than 1% from expected {expected_score}"
        except Exception as e:
            pytest.fail(f"Integration test failed: {str(e)}") 