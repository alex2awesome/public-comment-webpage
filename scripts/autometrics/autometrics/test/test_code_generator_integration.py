import pytest
import pandas as pd
import tempfile
import os
from autometrics.dataset.Dataset import Dataset
from autometrics.metrics.generated.GeneratedCodeMetric import GeneratedCodeReferenceBasedMetric, GeneratedCodeReferenceFreeMetric


@pytest.fixture
def summarization_dataset():
    """Create a realistic test dataset for text summarization task."""
    data = {
        'input': [
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "Climate change refers to long-term shifts in global or regional climate patterns.",
            "The process of photosynthesis allows plants to convert light energy into chemical energy."
        ],
        'output': [
            "Quick fox jumps over dog with all letters.",
            "ML lets computers learn automatically.",
            "Climate change means long-term weather shifts.",
            "Photosynthesis converts light to chemical energy."
        ],
        'reference': [
            "A fox jumps over a dog in a sentence with all alphabet letters.",
            "Machine learning enables automatic computer learning.",
            "Climate change involves long-term climate pattern shifts.",
            "Plants use photosynthesis to convert light into energy."
        ],
        'quality_score': [4.2, 4.8, 3.9, 4.5]
    }
    
    df = pd.DataFrame(data)
    return Dataset(
        dataframe=df,
        target_columns=['quality_score'],
        ignore_columns=[],
        metric_columns=[],
        name="summarization_dataset",
        input_column='input', 
        output_column='output',
        reference_columns=['reference']
    )


class TestCodeGeneratorIntegration:
    """Integration tests for the CodeGenerator functionality."""

    def test_reference_free_compression_metric(self, summarization_dataset):
        """Test a reference-free metric that measures compression ratio."""
        code = """
input_words = len(input.split())
output_words = len(output.split())
if input_words == 0:
    return 0.0
# Return compression ratio (good summaries should be shorter)
compression_ratio = output_words / input_words
# Convert to a quality score (higher is better, so invert)
return max(0.0, 1.0 - compression_ratio) if compression_ratio <= 1.0 else 0.0
"""
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="compression_ratio_metric",
            description="Measures how well the output compresses the input",
            generated_code=code,
            task_description="Text summarization task"
        )
        
        # Test on sample data
        input_text = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."
        output_text = "Quick fox jumps over dog with all letters."
        
        result = metric.calculate(input_text, output_text)
        
        # Should return a reasonable compression score
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        
        # Test on dataset
        results = metric.predict(summarization_dataset, update_dataset=False)
        assert len(results) == 4
        assert all(isinstance(r, float) for r in results)

    def test_reference_based_jaccard_similarity(self, summarization_dataset):
        """Test a reference-based metric that measures Jaccard similarity."""
        code = """
if not references or len(references) == 0:
    return 0.0

output_words = set(output.lower().split())
reference_words = set(references[0].lower().split())

if len(reference_words) == 0:
    return 0.0

# Calculate Jaccard similarity
intersection = output_words & reference_words
union = output_words | reference_words

return len(intersection) / len(union) if len(union) > 0 else 0.0
"""
        
        metric = GeneratedCodeReferenceBasedMetric(
            name="jaccard_similarity_metric",
            description="Measures word overlap similarity with reference",
            generated_code=code,
            task_description="Text summarization task"
        )
        
        # Test on sample data
        input_text = "The quick brown fox jumps over the lazy dog."
        output_text = "Quick fox jumps over dog."
        reference_text = "A fox jumps over a dog."
        
        result = metric.calculate(input_text, output_text, [reference_text])
        
        # Should return a reasonable similarity score
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        
        # Test on dataset
        results = metric.predict(summarization_dataset, update_dataset=False)
        assert len(results) == 4
        assert all(isinstance(r, float) for r in results)

    def test_complex_multi_factor_metric(self):
        """Test a complex metric using multiple quality indicators."""
        code = """
import math

# Feature 1: Length appropriateness
input_len = len(input.split())
output_len = len(output.split())
ideal_ratio = 0.3  # Good summaries are ~30% of original length
length_score = 1.0 - abs(output_len / max(input_len, 1) - ideal_ratio)
length_score = max(0.0, min(1.0, length_score))

# Feature 2: Vocabulary diversity
output_words = output.split()
unique_words = len(set(output_words))
diversity_score = unique_words / max(len(output_words), 1)

# Feature 3: Sentence structure
sentences = output.split('.')
if len(sentences) > 1:
    valid_sentences = [s for s in sentences if s.strip()]
    if valid_sentences:
        avg_sentence_len = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
        structure_score = min(1.0, avg_sentence_len / 8.0)
    else:
        structure_score = 0.0
else:
    structure_score = 0.5

# Combine scores
final_score = (length_score * 0.4 + diversity_score * 0.3 + structure_score * 0.3)
return final_score
"""
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="complex_quality_metric",
            description="Multi-factor text quality assessment",
            generated_code=code,
            task_description="Text summarization task"
        )
        
        # Test on sample data
        input_text = "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."
        output_text = "ML lets computers learn automatically."
        
        result = metric.calculate(input_text, output_text)
        
        # Should return a reasonable quality score
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_error_recovery_graceful_handling(self):
        """Test that metrics handle errors gracefully."""
        code = """
words = output.split()
return 1.0 / len(words)  # Will fail if output is empty
"""
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="division_error_metric",
            description="Metric that might cause division by zero",
            generated_code=code,
            task_description="Test error handling"
        )
        
        # Test with empty output (should handle error)
        result = metric.calculate("Some input", "")
        assert result == 0.0  # Should return 0.0 on error
        
        # Test with valid output
        result = metric.calculate("Some input", "valid output")
        assert isinstance(result, float)

    def test_metric_caching_functionality(self, summarization_dataset):
        """Test that metric caching works correctly."""
        code = "return len(output)"
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="length_metric_cached",
            description="Test caching functionality",
            generated_code=code,
            use_cache=True
        )
        
        # First calculation
        result1 = metric.calculate("test input", "test output")
        
        # Second calculation with same inputs (should use cache)
        result2 = metric.calculate("test input", "test output")
        
        assert result1 == result2
        assert result1 == 11.0  # "test output" has 11 characters

    def test_batch_processing_efficiency(self, summarization_dataset):
        """Test that batch processing works correctly."""
        code = "return len(output.split())"  # Count words
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="word_count_metric",
            description="Count words in output",
            generated_code=code,
            task_description="Word counting"
        )
        
        results = metric.predict(summarization_dataset, update_dataset=False)
        
        # Should get word counts for each output
        expected_counts = [
            len("Quick fox jumps over dog with all letters.".split()),
            len("ML lets computers learn automatically.".split()),
            len("Climate change means long-term weather shifts.".split()),
            len("Photosynthesis converts light to chemical energy.".split())
        ]
        
        assert len(results) == 4
        assert results == [float(count) for count in expected_counts]

    def test_dataset_integration_with_update(self, summarization_dataset):
        """Test integration with dataset updates."""
        code = "return len(output)"
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="char_count_metric",
            description="Count characters in output",
            generated_code=code,
            task_description="Character counting"
        )
        
        # Test with dataset update
        results = metric.predict(summarization_dataset, update_dataset=True)
        
        # Check that the dataset was updated
        df = summarization_dataset.get_dataframe()
        assert "char_count_metric" in df.columns
        assert list(df["char_count_metric"]) == results
        
        # Check that metric column was added to dataset metadata
        assert "char_count_metric" in summarization_dataset.get_metric_columns()

    def test_edge_case_empty_inputs(self):
        """Test handling of edge cases like empty inputs."""
        code = """
if not input.strip() or not output.strip():
    return 0.0
return len(output) / len(input)
"""
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="ratio_metric",
            description="Test empty input handling",
            generated_code=code,
            task_description="Edge case testing"
        )
        
        # Test with empty input
        result = metric.calculate("", "some output")
        assert result == 0.0
        
        # Test with empty output
        result = metric.calculate("some input", "")
        assert result == 0.0
        
        # Test with both empty
        result = metric.calculate("", "")
        assert result == 0.0
        
        # Test with valid inputs
        result = metric.calculate("ab", "abcd")
        assert result == 2.0  # 4/2 = 2.0 