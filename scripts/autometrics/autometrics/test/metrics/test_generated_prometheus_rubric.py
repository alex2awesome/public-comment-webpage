import pytest

from autometrics.metrics.generated.GeneratedPrometheus import (
    GeneratedRefFreePrometheusMetric,
    GeneratedRefBasedPrometheusMetric
)


class DummyPrometheusModel:
    """Minimal dummy model to satisfy Prometheus requirements in tests."""

    def __init__(self):
        # Mimic the attribute that the real model exposes
        self.model = "dummy/prometheus"
        self.api_base = "http://test:8000/v1"
        self.api_key = "None"

    def validate_mockllm(self):
        """Validation method expected by PrometheusEval"""
        return True


@pytest.fixture()
def sample_rubric():
    """Sample rubric for testing."""
    return {
        "criteria": "Clarity and coherence of the response",
        "score1_description": "Very unclear, incoherent, and difficult to understand",
        "score2_description": "Somewhat unclear with multiple confusing elements",
        "score3_description": "Moderately clear with some minor issues",
        "score4_description": "Clear and well-structured with minimal issues",
        "score5_description": "Exceptionally clear, coherent, and easy to understand"
    }


@pytest.fixture()
def sample_rubric_with_special_chars():
    """Sample rubric with special characters that need escaping."""
    return {
        "criteria": "Quality | Assessment",
        "score1_description": "Poor quality\nMultiple issues | problems",
        "score2_description": "Below average | Some issues",
        "score3_description": "Average quality\nAcceptable performance",
        "score4_description": "Good quality | Well done",
        "score5_description": "Excellent | Outstanding work"
    }


@pytest.fixture()
def mock_llm_calls(monkeypatch):
    """Mock all LLM calls to avoid making actual API requests during testing."""
    
    # Mock DSPy ChainOfThought calls
    def mock_chain_of_thought_call(self, *args, **kwargs):
        class MockOutput:
            domain = "Text Generation"
            tasks = ["Task 1", "Task 2"]
            best_suited_for_circumstances = ["Circumstance 1", "Circumstance 2"]
            not_recommended_for_circumstances = ["Bad circumstance 1", "Bad circumstance 2"]
            biases = ["Bias 1", "Bias 2"]
            task_misalignment_risks = ["Risk 1", "Risk 2"]
            failure_cases = ["Failure 1", "Failure 2"]
        return MockOutput()
    
    # Patch the DSPy ChainOfThought constructor to return a mock
    def mock_chain_of_thought(signature):
        class MockChainOfThought:
            def __call__(self, *args, **kwargs):
                return mock_chain_of_thought_call(self, *args, **kwargs)
        return MockChainOfThought()
    
    import dspy
    monkeypatch.setattr(dspy, "ChainOfThought", mock_chain_of_thought)
    
    # Mock dspy.settings.context to avoid actual model calls
    class MockContext:
        def __init__(self, lm=None):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    if hasattr(dspy, 'settings'):
        monkeypatch.setattr(dspy.settings, "context", MockContext)
    
    # Mock the MetricCardBuilder.build method to avoid LLM calls
    from autometrics.metrics.generated.utils.metric_card import MetricCardBuilder
    def mock_build(self):
        return f"# Mock Metric Card for {self.metric.name}\n\nThis is a mock metric card."
    monkeypatch.setattr(MetricCardBuilder, "build", mock_build)
    
    # Mock prometheus_eval.mock components to prevent import errors
    class MockPrometheusLLM:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return 4.0

    class AsyncMockPrometheusLLM:
        def __init__(self, *args, **kwargs):
            pass
        async def __call__(self, *args, **kwargs):
            return 4.0
    
    try:
        import sys
        import types
        prometheus_mock = types.ModuleType('prometheus_eval.mock')
        prometheus_mock.MockLLM = MockPrometheusLLM
        prometheus_mock.AsyncMockLLM = AsyncMockPrometheusLLM
        sys.modules['prometheus_eval.mock'] = prometheus_mock
    except Exception:
        pass  # If mocking fails, tests might still work
    
    # Mock PrometheusEval to avoid actual initialization
    class MockPrometheusEval:
        def __init__(self, *args, **kwargs):
            pass
        def single_absolute_grade(self, *args, **kwargs):
            return "Test feedback", 4.0
    
    from prometheus_eval import PrometheusEval
    monkeypatch.setattr('prometheus_eval.PrometheusEval', MockPrometheusEval)


def test_format_rubric_as_markdown_basic(mock_llm_calls, sample_rubric):
    """Test basic rubric to markdown table formatting."""
    metric = GeneratedRefFreePrometheusMetric(
        name="test_clarity",
        description="Test clarity metric",
        rubric=sample_rubric,
        model=DummyPrometheusModel(),
        task_description="Test task",
        max_workers=1,
    )
    
    markdown_lines = metric._format_rubric_as_markdown()
    
    # Check table structure
    assert len(markdown_lines) == 7  # Header + separator + 5 score rows
    assert markdown_lines[0] == "| Score | Description |"
    assert markdown_lines[1] == "|-------|-------------|"
    
    # Check score rows
    assert markdown_lines[2] == "| 1 | Very unclear, incoherent, and difficult to understand |"
    assert markdown_lines[3] == "| 2 | Somewhat unclear with multiple confusing elements |"
    assert markdown_lines[4] == "| 3 | Moderately clear with some minor issues |"
    assert markdown_lines[5] == "| 4 | Clear and well-structured with minimal issues |"
    assert markdown_lines[6] == "| 5 | Exceptionally clear, coherent, and easy to understand |"


def test_format_rubric_as_markdown_with_special_chars(mock_llm_calls, sample_rubric_with_special_chars):
    """Test rubric formatting with special characters that need escaping."""
    metric = GeneratedRefFreePrometheusMetric(
        name="test_quality",
        description="Test quality metric",
        rubric=sample_rubric_with_special_chars,
        model=DummyPrometheusModel(),
        task_description="Test task",
        max_workers=1,
    )
    
    markdown_lines = metric._format_rubric_as_markdown()
    
    # Check that pipe characters are escaped
    assert "| 1 | Poor quality Multiple issues \\| problems |" in markdown_lines
    assert "| 2 | Below average \\| Some issues |" in markdown_lines
    assert "| 4 | Good quality \\| Well done |" in markdown_lines
    assert "| 5 | Excellent \\| Outstanding work |" in markdown_lines
    
    # Check that newlines are replaced with spaces
    assert "Poor quality Multiple issues" in markdown_lines[2]
    assert "Average quality Acceptable performance" in markdown_lines[4]


def test_rubric_in_metric_card_contains_markdown_table(mock_llm_calls, sample_rubric):
    """Test that the metric card includes the rubric as a markdown table."""
    metric = GeneratedRefFreePrometheusMetric(
        name="test_clarity",
        description="Test clarity metric",
        rubric=sample_rubric,
        model=DummyPrometheusModel(),
        task_description="Test task",
        max_workers=1,
    )
    
    # Generate metric details (which should include the rubric table)
    metric_details = metric.generate_metric_details_ref_free()
    
    # Check that markdown table structure is present
    assert "| Score | Description |" in metric_details
    assert "|-------|-------------|" in metric_details
    assert "| 1 | Very unclear, incoherent, and difficult to understand |" in metric_details
    assert "| 5 | Exceptionally clear, coherent, and easy to understand |" in metric_details


def test_rubric_ref_based_vs_ref_free_markdown(mock_llm_calls, sample_rubric):
    """Test that both reference-based and reference-free metrics generate markdown tables."""
    ref_free_metric = GeneratedRefFreePrometheusMetric(
        name="test_clarity_ref_free",
        description="Test clarity metric (ref-free)",
        rubric=sample_rubric,
        model=DummyPrometheusModel(),
        task_description="Test task",
        max_workers=1,
    )
    
    ref_based_metric = GeneratedRefBasedPrometheusMetric(
        name="test_clarity_ref_based",
        description="Test clarity metric (ref-based)",
        rubric=sample_rubric,
        model=DummyPrometheusModel(),
        task_description="Test task",
        max_workers=1,
    )
    
    # Both should generate identical markdown tables
    ref_free_markdown = ref_free_metric._format_rubric_as_markdown()
    ref_based_markdown = ref_based_metric._format_rubric_as_markdown()
    
    assert ref_free_markdown == ref_based_markdown
    
    # Both metric details should contain the table
    ref_free_details = ref_free_metric.generate_metric_details_ref_free()
    ref_based_details = ref_based_metric.generate_metric_details_ref_based()
    
    assert "| Score | Description |" in ref_free_details
    assert "| Score | Description |" in ref_based_details


def test_empty_rubric_handling(mock_llm_calls):
    """Test handling of empty or malformed rubric."""
    empty_rubric = {}
    
    metric = GeneratedRefFreePrometheusMetric(
        name="test_empty",
        description="Test empty rubric",
        rubric=empty_rubric,
        model=DummyPrometheusModel(),
        task_description="Test task",
        max_workers=1,
    )
    
    markdown_lines = metric._format_rubric_as_markdown()
    
    # Should still generate table structure with N/A values
    assert len(markdown_lines) == 7
    assert markdown_lines[0] == "| Score | Description |"
    assert markdown_lines[1] == "|-------|-------------|"
    assert "| 1 | N/A |" in markdown_lines
    assert "| 5 | N/A |" in markdown_lines


def test_display_rubric_backward_compatibility(mock_llm_calls, sample_rubric):
    """Test that the existing display_rubric method still works."""
    metric = GeneratedRefFreePrometheusMetric(
        name="test_display",
        description="Test display method",
        rubric=sample_rubric,
        model=DummyPrometheusModel(),
        task_description="Test task",
        max_workers=1,
    )
    
    # The display_rubric method should work without errors
    # (It's designed for Jupyter notebooks so we can't test the actual display,
    # but we can ensure it doesn't crash)
    try:
        metric.display_rubric(sample_rubric, "Test Metric")
        metric.display()
        # If we get here without exception, the methods work
        assert True
    except Exception as e:
        # Some exceptions might be expected in test environment (like IPython not available)
        # but we want to catch actual implementation errors
        if "IPython" not in str(e) and "display" not in str(e):
            raise e 