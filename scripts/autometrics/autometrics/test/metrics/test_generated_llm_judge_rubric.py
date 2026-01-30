import pytest
from typing import List

from autometrics.metrics.generated.GeneratedLLMJudgeMetric import (
    GeneratedRefFreeLLMJudgeMetric,
    GeneratedRefBasedLLMJudgeMetric
)


class DummyDSPyModel:
    """Minimal dummy model to satisfy DSPy requirements in tests."""

    def __init__(self):
        self.model = "dummy/dspy"


@pytest.fixture()
def sample_rubric():
    """Sample rubric for testing."""
    return {
        "criteria": "Clarity and coherence of the response",
        "score1_description": "Response is completely unclear and incoherent",
        "score2_description": "Response has some clarity but is mostly confusing",
        "score3_description": "Response has moderate clarity with some unclear parts",
        "score4_description": "Response is mostly clear with minor unclear elements",
        "score5_description": "Response is completely clear and coherent"
    }


@pytest.fixture()
def sample_axis():
    """Sample axis for testing (non-rubric)."""
    return "*Clarity*: How clear and understandable is the response?"


def test_has_structured_rubric_with_rubric(sample_rubric):
    """Test that _has_structured_rubric returns True when metric has a structured rubric."""
    metric = GeneratedRefFreeLLMJudgeMetric(
        name="test_clarity_rubric",
        description="Test clarity metric with rubric",
        axis="Test axis",
        model=DummyDSPyModel(),
        rubric=sample_rubric,  # This makes it a rubric-based metric
        metric_card="Mock metric card"  # Disable metric card generation
    )
    
    assert metric._has_structured_rubric() is True


def test_has_structured_rubric_without_rubric(sample_axis):
    """Test that _has_structured_rubric returns False when metric only has an axis."""
    metric = GeneratedRefFreeLLMJudgeMetric(
        name="test_clarity_axis",
        description="Test clarity metric with axis only",
        axis=sample_axis,
        model=DummyDSPyModel(),
        metric_card="Mock metric card"  # Disable metric card generation
        # No rubric attribute
    )
    
    assert metric._has_structured_rubric() is False


def test_format_rubric_as_markdown_with_rubric(sample_rubric):
    """Test that _format_rubric_as_markdown returns proper markdown when rubric exists."""
    metric = GeneratedRefFreeLLMJudgeMetric(
        name="test_clarity_rubric",
        description="Test clarity metric with rubric",
        axis="Test axis",
        model=DummyDSPyModel(),
        rubric=sample_rubric,
        metric_card="Mock metric card"  # Disable metric card generation
    )
    
    markdown_lines = metric._format_rubric_as_markdown()
    
    # Should return markdown table
    assert len(markdown_lines) == 7  # header + separator + 5 score rows
    assert markdown_lines[0] == "| Score | Description |"
    assert markdown_lines[1] == "|-------|-------------|"
    assert "| 1 | Response is completely unclear and incoherent |" in markdown_lines
    assert "| 5 | Response is completely clear and coherent |" in markdown_lines


def test_format_rubric_as_markdown_without_rubric(sample_axis):
    """Test that _format_rubric_as_markdown returns empty list when no rubric."""
    metric = GeneratedRefFreeLLMJudgeMetric(
        name="test_clarity_axis",
        description="Test clarity metric with axis only",
        axis=sample_axis,
        model=DummyDSPyModel(),
        metric_card="Mock metric card"  # Disable metric card generation
    )
    
    markdown_lines = metric._format_rubric_as_markdown()
    
    # Should return empty list since no rubric
    assert markdown_lines == []


def test_metric_details_contains_rubric_table(sample_rubric):
    """Test that the metric details contain the rubric table when rubric is present."""
    metric = GeneratedRefFreeLLMJudgeMetric(
        name="test_clarity_rubric", 
        description="Test clarity metric with rubric",
        axis="Test axis",
        model=DummyDSPyModel(),
        rubric=sample_rubric,
        metric_card="Mock metric card"  # Disable metric card generation
    )
    
    details = metric.generate_metric_details_ref_free()
    
    # Should contain rubric details section
    assert "### Rubric Details" in details
    assert "**Criteria:** Clarity and coherence of the response" in details
    assert "#### Scoring Rubric" in details
    assert "| Score | Description |" in details
    assert "| 1 | Response is completely unclear and incoherent |" in details


def test_metric_details_without_rubric_table(sample_axis):
    """Test that the metric details don't contain rubric table when only axis is present."""
    metric = GeneratedRefFreeLLMJudgeMetric(
        name="test_clarity_axis",
        description="Test clarity metric with axis only", 
        axis=sample_axis,
        model=DummyDSPyModel(),
        metric_card="Mock metric card"  # Disable metric card generation
    )
    
    details = metric.generate_metric_details_ref_free()
    
    # Should not contain rubric details section
    assert "### Rubric Details" not in details
    assert "#### Scoring Rubric" not in details
    assert "| Score | Description |" not in details
    # But should still have the axis
    assert "Axis rubric" in details
    assert sample_axis in details


def test_input_field_labeling_with_rubric(sample_rubric):
    """Test that input fields are labeled correctly when rubric is present."""
    metric = GeneratedRefFreeLLMJudgeMetric(
        name="test_clarity_rubric",
        description="Test clarity metric with rubric",
        axis="Test axis",
        model=DummyDSPyModel(),
        rubric=sample_rubric,
        metric_card="Mock metric card"  # Disable metric card generation
    )
    
    details = metric.generate_metric_details_ref_free()
    
    # Should say "Rubric" instead of "Axis rubric"
    assert "**Rubric** `Test axis`" in details
    assert "**Axis rubric** `Test axis`" not in details


def test_input_field_labeling_without_rubric(sample_axis):
    """Test that input fields are labeled correctly when only axis is present."""
    metric = GeneratedRefFreeLLMJudgeMetric(
        name="test_clarity_axis",
        description="Test clarity metric with axis only",
        axis=sample_axis,
        model=DummyDSPyModel(),
        metric_card="Mock metric card"  # Disable metric card generation
    )
    
    details = metric.generate_metric_details_ref_free()
    
    # Should say "Axis rubric"
    assert f"**Axis rubric** `{sample_axis}`" in details
    assert "**Rubric**" not in details


def test_reference_based_rubric_functionality(sample_rubric):
    """Test that reference-based metrics also support rubric formatting."""
    metric = GeneratedRefBasedLLMJudgeMetric(
        name="test_clarity_rubric_ref",
        description="Test clarity metric with rubric (reference-based)",
        axis="Test axis",
        model=DummyDSPyModel(),
        rubric=sample_rubric,
        metric_card="Mock metric card"  # Disable metric card generation
    )
    
    assert metric._has_structured_rubric() is True
    
    details = metric.generate_metric_details_ref_based()
    assert "### Rubric Details" in details
    assert "**Criteria:** Clarity and coherence of the response" in details
    assert "| Score | Description |" in details 