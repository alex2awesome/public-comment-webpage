import pandas as pd
import pytest

from autometrics.dataset.Dataset import Dataset
from autometrics.generator.LLMJudgeProposer import BasicLLMJudgeProposer
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import (
    GeneratedRefFreeLLMJudgeMetric,
    GeneratedRefBasedLLMJudgeMetric
)
from autometrics.generator import utils as gen_utils


class DummyModel:
    """Minimal dummy model to satisfy LLMJudge requirements in tests."""

    def __init__(self):
        # Mimic the attribute that the real model exposes
        self.model = "dummy/model"


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
            "*Clarity*: Quality of clarity",
            "*Relevance*: Quality of relevance",
        ],
    )
    
    # Mock DSPy ChainOfThought calls for metric card generation
    def mock_chain_of_thought_call(self, *args, **kwargs):
        class MockOutput:
            score = "4"
            domain = "Text Generation"
            tasks = ["Task 1", "Task 2"]
            best_suited_for_circumstances = ["Circumstance 1", "Circumstance 2"]
            not_recommended_for_circumstances = ["Bad circumstance 1", "Bad circumstance 2"]
            biases = ["Bias 1", "Bias 2"]
            task_misalignment_risks = ["Risk 1", "Risk 2"]
            failure_cases = ["Failure 1", "Failure 2"]
            axes_of_variation = [
                "*Clarity*: Quality of clarity",
                "*Relevance*: Quality of relevance",
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


def test_llm_judge_proposer_returns_expected_metrics_ref_free(mock_llm_calls, small_dataset):
    """Test that LLMJudgeProposer returns reference-free metrics for datasets without references."""
    
    proposer = BasicLLMJudgeProposer(
        executor_kwargs={"model": DummyModel()},
    )

    metrics = proposer.generate(small_dataset, target_measure="score", n_metrics=2)

    # Assertions -----------------------------------------------------------------
    assert isinstance(metrics, list)
    assert len(metrics) == 2

    for m in metrics:
        assert isinstance(m, GeneratedRefFreeLLMJudgeMetric)
        assert not m.is_reference_based
        # Metric name should include axis name root + model name suffix
        assert m.name.endswith("dummy/model".split("/")[-1])


def test_llm_judge_proposer_returns_expected_metrics_ref_based(mock_llm_calls, small_dataset_with_references):
    """Test that LLMJudgeProposer returns reference-based metrics for datasets with references."""
    
    proposer = BasicLLMJudgeProposer(
        executor_kwargs={"model": DummyModel()},
    )

    metrics = proposer.generate(small_dataset_with_references, target_measure="score", n_metrics=2)

    # Assertions -----------------------------------------------------------------
    assert isinstance(metrics, list)
    assert len(metrics) == 2

    for m in metrics:
        assert isinstance(m, GeneratedRefBasedLLMJudgeMetric)
        assert m.is_reference_based
        # Metric name should include axis name root + model name suffix
        assert m.name.endswith("dummy/model".split("/")[-1])


def test_llm_judge_proposer_automatic_detection(mock_llm_calls, small_dataset, small_dataset_with_references):
    """Test that the proposer automatically detects reference vs reference-free datasets."""
    
    proposer = BasicLLMJudgeProposer(executor_kwargs={"model": DummyModel()})
    
    # Test reference-free detection
    ref_free_class = proposer._determine_executor_class(small_dataset)
    assert ref_free_class == GeneratedRefFreeLLMJudgeMetric
    
    # Test reference-based detection
    ref_based_class = proposer._determine_executor_class(small_dataset_with_references)
    assert ref_based_class == GeneratedRefBasedLLMJudgeMetric


def test_metric_card_generation_no_llm_calls(mock_llm_calls, small_dataset):
    """Test that metric card generation works without making actual LLM calls."""
    
    proposer = BasicLLMJudgeProposer(executor_kwargs={"model": DummyModel()})
    metrics = proposer.generate(small_dataset, target_measure="score", n_metrics=1)
    
    # Test that we can access the metric card without LLM calls
    assert len(metrics) == 1
    metric = metrics[0]
    assert hasattr(metric, 'metric_card')
    assert metric.metric_card is not None
    
    # Test that we can generate metric details without LLM calls
    details = metric.generate_metric_details_ref_free()
    assert "reference-free" in details 