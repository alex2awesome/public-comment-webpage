import pytest
import numpy as np
from autometrics.metrics.reference_based.CharCut import CharCut

class TestCharCut:
    def test_exact_match(self):
        """Test that exact matches receive a perfect score of 0.0"""
        charcut = CharCut()
        input_text = "This is a test sentence."
        output_text = "This is a test sentence."
        references = ["This is a test sentence."]
        
        score = charcut.calculate(input_text, output_text, references)
        assert score == 0.0
        
    def test_complete_mismatch(self):
        """Test that completely different strings get a high score"""
        charcut = CharCut()
        input_text = "This is a test sentence."
        output_text = "Completely different text here."
        references = ["This is a test sentence."]
        
        score = charcut.calculate(input_text, output_text, references)
        assert score > 0.4  # Significant difference should give a high score
        
    def test_small_edit(self):
        """Test that a small edit results in a low score"""
        charcut = CharCut()
        input_text = "This is a test sentence."
        output_text = "This is a test sentance."  # 'e' changed to 'a'
        references = ["This is a test sentence."]
        
        score = charcut.calculate(input_text, output_text, references)
        assert 0.0 < score < 0.2  # Small difference should give a low score
        
    def test_multiple_references(self):
        """Test that with multiple references, the best (minimum) score is returned"""
        charcut = CharCut()
        input_text = "The cat sat on the mat."
        output_text = "The cat sat on the mat."
        references = [
            "A feline was sitting on a rug.",  # Very different
            "The cat sat on the mat.",  # Exact match
            "The cat was sitting on the mat."  # Similar but not exact
        ]
        
        score = charcut.calculate(input_text, output_text, references)
        assert score == 0.0  # Should match the second reference exactly
        
    def test_batch_calculation(self):
        """Test batch calculation of CharCut scores"""
        charcut = CharCut()
        inputs = [
            "This is sentence one.",
            "This is sentence two.",
            "This is sentence three."
        ]
        outputs = [
            "This is sentence 1.",  # Minor difference
            "This is sentence two.",  # Exact match
            "This is completely different."  # Major difference
        ]
        references = [
            ["This is sentence one."],
            ["This is sentence two."],
            ["This is sentence three."]
        ]
        
        scores = charcut.calculate_batched(inputs, outputs, references)
        assert len(scores) == 3
        assert 0.0 < scores[0] < 0.2  # Minor difference
        assert scores[1] == 0.0  # Exact match
        assert scores[2] > 0.3  # Major difference
        
    def test_match_size_parameter(self):
        """Test the effect of the match_size parameter"""
        input_text = "This is a test sentence."
        output_text = "This is a sample sentence."  # 'test' changed to 'sample'
        references = ["This is a test sentence."]
        
        # Default match_size = 3
        charcut_default = CharCut()
        score_default = charcut_default.calculate(input_text, output_text, references)
        
        # Larger match_size = 5
        charcut_larger = CharCut(match_size=5)
        score_larger = charcut_larger.calculate(input_text, output_text, references)
        
        # Larger match_size should result in fewer matches, potentially higher score
        assert score_default <= score_larger
        
    def test_alt_norm_parameter(self):
        """Test the effect of the alt_norm parameter"""
        input_text = "This is a test sentence."
        output_text = "This is a modified test sentence."  # Added 'modified'
        references = ["This is a test sentence."]
        
        # Default alt_norm = False (normalize by sum of lengths)
        charcut_default = CharCut()
        score_default = charcut_default.calculate(input_text, output_text, references)
        
        # With alt_norm = True (normalize by 2*candidate length)
        charcut_alt = CharCut(alt_norm=True)
        score_alt = charcut_alt.calculate(input_text, output_text, references)
        
        # Scores should be different due to normalization change
        assert score_default != score_alt
        
    def test_word_movements(self):
        """Test how CharCut handles word movements/shifts"""
        charcut = CharCut()
        input_text = "The quick brown fox jumps over the lazy dog."
        output_text = "The brown quick fox jumps over the lazy dog."  # 'quick' and 'brown' swapped
        references = ["The quick brown fox jumps over the lazy dog."]
        
        score = charcut.calculate(input_text, output_text, references)
        # Shifts should be penalized but less severely than insertions/deletions
        assert 0.0 < score < 0.3
        
    def test_edge_cases(self):
        """Test edge cases like empty strings"""
        charcut = CharCut()
        
        # Both empty
        score1 = charcut.calculate("", "", [""])
        assert score1 == 0.0
        
        # Reference empty but output not
        score2 = charcut.calculate("", "Some text", [""])
        assert score2 > 0.0
        
        # Empty output compared to non-empty reference should count as a deletion
        score3 = charcut.calculate("dummy", "", ["Some text"])
        assert score3 > 0.0
        
    def test_caching(self):
        """Test that caching works correctly"""
        charcut = CharCut(use_cache=True)
        input_text = "This is a test sentence."
        output_text = "This is a modified test sentence."
        references = ["This is a test sentence."]
        
        # First calculation should compute and cache
        score1 = charcut.calculate(input_text, output_text, references)
        
        # Second calculation with same inputs should use cache
        score2 = charcut.calculate(input_text, output_text, references)
        
        assert score1 == score2 