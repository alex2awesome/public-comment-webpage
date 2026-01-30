import pandas as pd
import pytest

from autometrics.dataset.Dataset import Dataset
from autometrics.generator.CodeGenerator import CodeGenerator
from autometrics.metrics.generated.GeneratedCodeMetric import (
    GeneratedRefFreeCodeMetric,
    GeneratedRefBasedCodeMetric
)
from autometrics.generator import utils as gen_utils


@pytest.fixture()
def small_dataset():
    df = pd.DataFrame(
        {
            "input": ["a", "b", "c", "d", "e"],
            "output": ["A", "B", "C", "D", "E"],
            "score": [0.9, 0.8, 0.2, 0.6, 0.7],
        }
    )

    return Dataset(
        dataframe=df,
        target_columns=["score"],
        ignore_columns=[],
        metric_columns=[],
        name="dummy",
        input_column="input",
        output_column="output",
    )


@pytest.fixture()
def small_dataset_with_references():
    df = pd.DataFrame(
        {
            "input": ["a", "b", "c", "d", "e"],
            "output": ["A", "B", "C", "D", "E"],
            "reference": ["A+", "B+", "C+", "D+", "E+"],
            "score": [0.9, 0.8, 0.2, 0.6, 0.7],
        }
    )

    return Dataset(
        dataframe=df,
        target_columns=["score"],
        ignore_columns=[],
        metric_columns=[],
        name="dummy_with_ref",
        input_column="input",
        output_column="output",
        reference_columns=["reference"],
    )


@pytest.fixture()
def mock_llm_calls(monkeypatch):
    """Mock all LLM calls to avoid making actual API requests during testing."""
    
    # Mock the axis generation
    monkeypatch.setattr(
        gen_utils,
        "generate_axes_of_variation",
        lambda *args, **kwargs: [
            "*Length*: Character count quality",
            "*Word Count*: Word count assessment",
        ],
    )
    
    # Mock DSPy ChainOfThought calls for code generation and metric card generation
    def mock_chain_of_thought_call(self, *args, **kwargs):
        class MockOutput:
            metric_name = "test_metric"
            code = "return len(output)"
            domain = "Text Generation"
            tasks = ["Task 1", "Task 2"]
            best_suited_for_circumstances = ["Circumstance 1", "Circumstance 2"]
            not_recommended_for_circumstances = ["Bad circumstance 1", "Bad circumstance 2"]
            biases = ["Bias 1", "Bias 2"]
            task_misalignment_risks = ["Risk 1", "Risk 2"]
            failure_cases = ["Failure 1", "Failure 2"]
            axes_of_variation = [
                "*Length*: Character count quality",
                "*Word Count*: Word count assessment",
            ]
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
    
    # Mock the metric card builder
    def mock_generate_further_reading(metric):
        return "- Mock further reading"
    
    from autometrics.metrics.generated.utils import metric_card
    monkeypatch.setattr(metric_card, "generate_further_reading", mock_generate_further_reading)
    
    # Mock the MetricCardBuilder.build method to avoid LLM calls
    from autometrics.metrics.generated.utils.metric_card import MetricCardBuilder
    def mock_build(self):
        return f"# Mock Metric Card for {self.metric.name}\n\nThis is a mock metric card."
    monkeypatch.setattr(MetricCardBuilder, "build", mock_build)


def test_code_generator_returns_expected_metrics_ref_free(mock_llm_calls, small_dataset):
    """Test that CodeGenerator returns reference-free metrics for datasets without references."""
    
    generator = CodeGenerator()

    metrics = generator.generate(small_dataset, target_measure="score", n_metrics=2)

    # Assertions -----------------------------------------------------------------
    assert isinstance(metrics, list)
    assert len(metrics) == 2

    for m in metrics:
        assert isinstance(m, GeneratedRefFreeCodeMetric)
        assert not m.is_reference_based
        # Metric name should include generated suffix
        assert "generated" in m.name


def test_code_generator_returns_expected_metrics_ref_based(mock_llm_calls, small_dataset_with_references):
    """Test that CodeGenerator returns reference-based metrics for datasets with references."""
    
    generator = CodeGenerator()

    metrics = generator.generate(small_dataset_with_references, target_measure="score", n_metrics=2)

    # Assertions -----------------------------------------------------------------
    assert isinstance(metrics, list)
    assert len(metrics) == 2

    for m in metrics:
        assert isinstance(m, GeneratedRefBasedCodeMetric)
        assert m.is_reference_based
        # Metric name should include generated suffix
        assert "generated" in m.name


def test_code_generator_automatic_detection(mock_llm_calls, small_dataset, small_dataset_with_references):
    """Test that the generator automatically detects reference vs reference-free datasets."""
    
    generator = CodeGenerator()
    
    # Test reference-free detection
    ref_free_class = generator._determine_executor_class(small_dataset)
    assert ref_free_class == GeneratedRefFreeCodeMetric
    
    # Test reference-based detection
    ref_based_class = generator._determine_executor_class(small_dataset_with_references)
    assert ref_based_class == GeneratedRefBasedCodeMetric


def test_metric_properties(mock_llm_calls, small_dataset):
    """Test that generated metrics have the correct properties."""
    
    generator = CodeGenerator()
    metrics = generator.generate(small_dataset, target_measure="score", n_metrics=1)
    
    # Test that we can access the metric properties
    assert len(metrics) == 1
    metric = metrics[0]
    assert hasattr(metric, 'generated_code')
    assert hasattr(metric, 'task_description')
    assert hasattr(metric, 'measurement_axis')
    
    # Test that the code is actually there
    assert metric.generated_code is not None
    assert len(metric.generated_code) > 0


def test_custom_executor_class(mock_llm_calls, small_dataset):
    """Test that custom executor class can be specified."""
    
    generator = CodeGenerator(executor_class=GeneratedRefBasedCodeMetric)

    metrics = generator.generate(small_dataset, target_measure="score", n_metrics=1)

    # Should use the specified executor class even though dataset is reference-free
    assert len(metrics) == 1
    assert isinstance(metrics[0], GeneratedRefBasedCodeMetric)


def test_generator_name_and_description():
    """Test that the generator has correct name and description."""
    generator = CodeGenerator()
    
    assert generator.get_name() == "CodeGenerator"
    assert "code-based" in generator.get_description().lower()
    assert str(generator) == f"{generator.name}: {generator.description}"
    assert repr(generator) == str(generator)


def test_code_cleaning(mock_llm_calls):
    """Test that code cleaning works correctly."""
    generator = CodeGenerator()
    
    # Test markdown removal
    code_with_markdown = "```python\nreturn len(output)\n```"
    cleaned = generator._clean_generated_code(code_with_markdown)
    assert cleaned == "return len(output)"
    
    # Test generic code block removal
    code_with_generic = "```\nreturn len(output)\n```"
    cleaned = generator._clean_generated_code(code_with_generic)
    assert cleaned == "return len(output)"
    
    # Test no markdown
    plain_code = "return len(output)"
    cleaned = generator._clean_generated_code(plain_code)
    assert cleaned == "return len(output)" 