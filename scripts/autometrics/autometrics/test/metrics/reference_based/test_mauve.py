import pytest
import torch
from unittest.mock import patch, MagicMock
from autometrics.metrics.reference_based.MAUVE import MAUVE, _aggregate

class TestMAUVE:
    """Test suite for the MAUVE metric."""

    @pytest.fixture
    def mock_mauve_instance(self):
        """Create a MAUVE instance with mocked components for testing."""
        with patch('autometrics.metrics.reference_based.HuggingFaceReferenceBasedMetric.load') as mock_load:
            # Mock the evaluate.load function
            mock_metric = MagicMock()
            mock_metric.compute.return_value = {'mauve': 0.75}
            mock_load.return_value = mock_metric
            
            # Create instance
            mauve = MAUVE()
            mauve.metric = mock_metric
            
            yield mauve

    def test_initialization(self):
        """Test that MAUVE initializes with correct parameters."""
        metric = MAUVE(
            name="CustomMAUVE", 
            description="Custom MAUVE description",
            aggregation="max"
        )
        
        assert metric.name == "CustomMAUVE"
        assert metric.description == "Custom MAUVE description"
        assert metric.metric_id == "mauve"
        assert metric.score_key == "mauve"
        assert metric.aggregation == "max"
        
    def test_initialization_with_aggregation(self):
        """Test that MAUVE initializes with different aggregation methods."""
        metric_min = MAUVE(aggregation="min")
        metric_max = MAUVE(aggregation="max")
        metric_avg = MAUVE(aggregation="avg")
        
        assert metric_min.name == "MAUVE_min"
        assert metric_max.name == "MAUVE_max"
        assert metric_avg.name == "MAUVE_avg"
        
        assert metric_min.aggregation == "min"
        assert metric_max.aggregation == "max"
        assert metric_avg.aggregation == "avg"
        
    def test_invalid_aggregation(self):
        """Test that MAUVE raises error on invalid aggregation."""
        with pytest.raises(ValueError):
            MAUVE(aggregation="invalid")

    def test_calculate_impl_basic(self, mock_mauve_instance):
        """Test calculation for a single input/output pair."""
        # Test with single reference
        result = mock_mauve_instance._calculate_impl(
            "input text", 
            "output text", 
            ["reference text"]
        )
        
        # Check result
        assert isinstance(result, float)
        assert result == 0.75
        
        # Check that compute was called with properly formatted references
        args, kwargs = mock_mauve_instance.metric.compute.call_args
        assert 'predictions' in kwargs
        assert kwargs['predictions'] == ['output text']
        assert kwargs['references'] == ['reference text']

    def test_calculate_impl_multiple_references(self, mock_mauve_instance):
        """Test calculation with multiple references - should compute for each reference."""
        # Configure mock to return different scores for each reference
        mock_mauve_instance.metric.compute.side_effect = [
            {'mauve': 0.6},  # First reference score
            {'mauve': 0.8}   # Second reference score
        ]
        
        # Test with multiple references
        result = mock_mauve_instance._calculate_impl(
            "input text", 
            "output text", 
            [["reference 1", "reference 2"]]
        )
        
        # Check result - should use max by default
        assert isinstance(result, float)
        assert result == 0.8  # Should be max of 0.6 and 0.8
        
        # Check that compute was called twice, once for each reference
        assert mock_mauve_instance.metric.compute.call_count == 2
        
        # First call should be with first reference
        args1, kwargs1 = mock_mauve_instance.metric.compute.call_args_list[0]
        assert kwargs1['references'] == ['reference 1']
        
        # Second call should be with second reference
        args2, kwargs2 = mock_mauve_instance.metric.compute.call_args_list[1]
        assert kwargs2['references'] == ['reference 2']
        
    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        # Test min aggregation
        assert _aggregate([0.1, 0.2, 0.3], "min") == 0.1
        
        # Test max aggregation
        assert _aggregate([0.1, 0.2, 0.3], "max") == 0.3
        
        # Test avg aggregation - use pytest.approx to handle floating point imprecision
        assert _aggregate([0.1, 0.2, 0.3], "avg") == pytest.approx(0.2)
        
        # Test empty list
        assert _aggregate([], "max") is None
        
        # Test invalid method
        with pytest.raises(ValueError):
            _aggregate([0.1, 0.2], "invalid")
    
    def test_aggregation_behavior(self, mock_mauve_instance):
        """Test that aggregation method affects the final score."""
        # Create MAUVE instances with different aggregation methods
        with patch('autometrics.metrics.reference_based.HuggingFaceReferenceBasedMetric.load') as mock_load:
            mock_metric = MagicMock()
            mock_load.return_value = mock_metric
            
            mauve_min = MAUVE(aggregation="min")
            mauve_max = MAUVE(aggregation="max")
            mauve_avg = MAUVE(aggregation="avg")
            
            mauve_min.metric = mock_metric
            mauve_max.metric = mock_metric
            mauve_avg.metric = mock_metric
            
            # Configure mock to return different scores for references
            scores = [{'mauve': 0.3}, {'mauve': 0.7}, {'mauve': 0.5}]
            mock_metric.compute.side_effect = scores * 3  # Repeat for each aggregation method
            
            # Test with multiple references
            refs = ["ref1", "ref2", "ref3"]
            
            # Test min aggregation
            result_min = mauve_min._calculate_impl("input", "output", refs)
            assert result_min == 0.3
            
            # Test max aggregation
            result_max = mauve_max._calculate_impl("input", "output", refs)
            assert result_max == 0.7
            
            # Test avg aggregation
            result_avg = mauve_avg._calculate_impl("input", "output", refs)
            assert result_avg == pytest.approx(0.5)

    def test_batched_calculation(self, mock_mauve_instance):
        """Test batch calculation functionality."""
        # Test with a batch of inputs/outputs
        inputs = ["input1", "input2", "input3"]
        outputs = ["output1", "output2", "output3"]
        references = [["ref1"], ["ref2a", "ref2b"], ["ref3"]]
        
        # Set up mock to return different scores for each reference
        mock_mauve_instance.metric.compute.side_effect = [
            {'mauve': 0.6},  # First input
            {'mauve': 0.7},  # Second input, first reference 
            {'mauve': 0.9},  # Second input, second reference
            {'mauve': 0.8}   # Third input
        ]
        
        # Call with our implementation
        results = mock_mauve_instance._calculate_batched_impl(inputs, outputs, references)
        
        # Check results
        assert len(results) == 3
        assert results[0] == 0.6  # Single reference
        assert results[1] == 0.9  # Max of two references (0.7 and 0.9)
        assert results[2] == 0.8  # Single reference
        
        # Verify correct calls were made to compute
        calls = mock_mauve_instance.metric.compute.call_args_list
        assert len(calls) == 4
        
        # Check first call (first example)
        _, kwargs1 = calls[0]
        assert kwargs1['predictions'] == ['output1']
        assert kwargs1['references'] == ['ref1']
        
        # Check second call (second example, first reference)
        _, kwargs2 = calls[1]
        assert kwargs2['predictions'] == ['output2']
        assert kwargs2['references'] == ['ref2a']
        
        # Check third call (second example, second reference)
        _, kwargs3 = calls[2]
        assert kwargs3['predictions'] == ['output2']
        assert kwargs3['references'] == ['ref2b']
        
        # Check fourth call (third example)
        _, kwargs4 = calls[3]
        assert kwargs4['predictions'] == ['output3']
        assert kwargs4['references'] == ['ref3']

    @pytest.mark.integration
    def test_integration_with_flattened_references(self):
        """Test handling of flattened references in MAUVE."""
        try:
            # Skip if evaluate package is not installed
            import evaluate
        except ImportError:
            pytest.skip("evaluate package not installed")
            
        # Create a real MAUVE instance with mocked compute method
        with patch('evaluate.EvaluationModule.compute') as mock_compute:
            mock_compute.return_value = {'mauve': 0.85}
            
            mauve = MAUVE()
            # Mock the metric directly to avoid loading
            mauve.metric = MagicMock()
            mauve.metric.compute = mock_compute
            
            # Test with references in list format
            result = mauve._calculate_impl(
                "input text",
                "output text",
                [["reference text"]]
            )
            
            # Check that compute was called with flattened references
            mock_compute.assert_called_once()
            args, kwargs = mock_compute.call_args
            
            # Check prediction format
            assert 'predictions' in kwargs
            assert kwargs['predictions'] == ["output text"]
            
            # Check reference format - this is the key part that should be fixed
            assert 'references' in kwargs
            assert isinstance(kwargs['references'][0], str)
            assert kwargs['references'][0] == "reference text"
            
            # Check result format
            assert isinstance(result, float)
            assert result == 0.85

    @pytest.mark.integration
    def test_integration_reference_formatting(self):
        """Test that references are correctly formatted for MAUVE metric."""
        try:
            # Skip if evaluate package is not installed
            import evaluate
        except ImportError:
            pytest.skip("evaluate package not installed")
            
        # Create a real MAUVE instance with mocked compute method
        with patch('evaluate.EvaluationModule.compute') as mock_compute:
            mock_compute.side_effect = [
                {'mauve': 0.8},  # First reference 
                {'mauve': 0.9},  # Second reference
                {'mauve': 0.7}   # Third reference
            ]
            
            mauve = MAUVE()
            # Mock the metric directly to avoid loading
            mauve.metric = MagicMock()
            mauve.metric.compute = mock_compute
            
            # Test with complex nested references to ensure they're handled correctly
            refs = [["ref1", "ref2"], "ref3"]
            result = mauve._calculate_impl(
                "input text",
                "output text",
                refs
            )
            
            # Verify compute was called with each reference
            assert mock_compute.call_count == 3
            call_args = mock_compute.call_args_list
            
            # First call - ref1
            _, kwargs1 = call_args[0]
            assert kwargs1['references'] == ['ref1']
            
            # Second call - ref2
            _, kwargs2 = call_args[1]
            assert kwargs2['references'] == ['ref2']
            
            # Third call - ref3
            _, kwargs3 = call_args[2]
            assert kwargs3['references'] == ['ref3']
            
            # Result should be max of all scores
            assert result == 0.9
            
    @pytest.mark.integration
    def test_integration_different_aggregation_methods(self):
        """Test integration of different aggregation methods with real references."""
        try:
            import evaluate
        except ImportError:
            pytest.skip("evaluate package not installed")
            
        # Create instances with different aggregation methods
        with patch('evaluate.load') as mock_load:
            mock_metric = MagicMock()
            mock_metric.compute.side_effect = [
                {'mauve': 0.4},  # Will be used for ref1 for min
                {'mauve': 0.8},  # Will be used for ref2 for min
                {'mauve': 0.4},  # Will be used for ref1 for max
                {'mauve': 0.8},  # Will be used for ref2 for max
                {'mauve': 0.4},  # Will be used for ref1 for avg
                {'mauve': 0.8},  # Will be used for ref2 for avg
            ]
            mock_load.return_value = mock_metric
            
            # Create MAUVE instances with different aggregation
            mauve_min = MAUVE(aggregation="min")
            mauve_max = MAUVE(aggregation="max")
            mauve_avg = MAUVE(aggregation="avg")
            
            # Set the mocked metric for each instance
            mauve_min.metric = mock_metric
            mauve_max.metric = mock_metric
            mauve_avg.metric = mock_metric
            
            # Test with multiple references
            references = ["ref1", "ref2"]
            
            # Test each aggregation method
            result_min = mauve_min._calculate_impl("input", "output", references)
            result_max = mauve_max._calculate_impl("input", "output", references)
            result_avg = mauve_avg._calculate_impl("input", "output", references)
            
            # Verify results
            assert result_min == 0.4  # Min of 0.4 and 0.8
            assert result_max == 0.8  # Max of 0.4 and 0.8
            assert result_avg == pytest.approx(0.6)  # Avg of 0.4 and 0.8
            
    @pytest.mark.integration
    def test_real_use_case_format(self):
        """Test with realistic but simple inputs without loading the full model."""
        with patch('evaluate.load') as mock_load:
            # Create mock metric
            mock_metric = MagicMock()
            # Different scores for the different references
            mock_metric.compute.side_effect = [
                {'mauve': 0.5},  # Will be used for 1st reference in 1st call
                {'mauve': 0.7},  # Will be used for 2nd reference in 1st call
                {'mauve': 0.8},  # Will be used for single reference in 2nd call
            ]
            mock_load.return_value = mock_metric
            
            # Create MAUVE instance with max aggregation (default)
            mauve = MAUVE() 
            mauve.metric = mock_metric
            
            # Test with sample text and complex reference format
            input_text = "This is a sample input."
            output_text = "This is a generated response."
            reference_texts = [
                ["This is the primary reference.", "This is an alternative reference."],
                "This is a single reference string."
            ]
            
            # Calculate scores
            result1 = mauve._calculate_impl(input_text, output_text, reference_texts[0])
            result2 = mauve._calculate_impl(input_text, output_text, reference_texts[1])
            
            # Verify both calls worked
            assert isinstance(result1, float)
            assert isinstance(result2, float)
            
            # Result1 should be max of the two reference scores
            assert result1 == 0.7
            
            # Result2 should be the single reference score
            assert result2 == 0.8
            
            # Verify references were properly formatted
            calls = mock_metric.compute.call_args_list
            assert len(calls) == 3
            
            # First call should use first reference
            _, kwargs1 = calls[0]
            assert kwargs1['references'] == ['This is the primary reference.']
            
            # Second call should use second reference 
            _, kwargs2 = calls[1]
            assert kwargs2['references'] == ['This is an alternative reference.']
            
            # Third call should use the string directly
            _, kwargs3 = calls[2]
            assert kwargs3['references'] == ['This is a single reference string.'] 

    @pytest.mark.integration
    def test_integration_weather_example(self):
        """Integration test with a real MAUVE instance using a weather example."""
        try:
            # Skip if evaluate package is not installed
            import evaluate
        except ImportError:
            pytest.skip("evaluate package not installed")
            
        # Create a real MAUVE instance
        mauve = MAUVE()
        
        input_text = "The weather forecast predicts heavy rain tomorrow."
        output_text = "Tomorrow's forecast indicates significant precipitation."
        references = ["Heavy rainfall is expected tomorrow.", "The forecast shows rain tomorrow."]
        
        try:
            # This should return a single score
            result = mauve._calculate_impl(input_text, output_text, references)
            
            # Basic checks on the result format
            assert isinstance(result, float)
            assert 0 <= result <= 1, "Score should be in range [0,1]"
            
            # Check that the score is within 1% of the expected value
            expected_score = 0.0040720962619612555
            assert abs(result - expected_score) <= 0.01, f"Score {result} differs by more than 1% from expected {expected_score}"
        except Exception as e:
            pytest.fail(f"Integration test failed: {str(e)}") 