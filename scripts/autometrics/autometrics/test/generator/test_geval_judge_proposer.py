import pandas as pd
import pytest

from autometrics.dataset.Dataset import Dataset
from autometrics.generator.GEvalJudgeProposer import GEvalJudgeProposer
from autometrics.metrics.generated.GeneratedGEvalMetric import (
    GeneratedRefFreeGEvalMetric,
    GeneratedRefBasedGEvalMetric
)
from autometrics.generator import utils as gen_utils


class DummyModel:
    """Minimal dummy model to satisfy GEval requirements in tests."""

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
    
    # Mock DSPy ChainOfThought calls for metric card generation and evaluation steps
    def mock_chain_of_thought_call(self, *args, **kwargs):
        class MockOutput:
            score = "4"
            evaluation_steps = "1. Read the input and output\n2. Evaluate clarity\n3. Assign score 1-5"
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


def test_geval_judge_proposer_returns_expected_metrics_ref_free(mock_llm_calls, small_dataset):
    """Test that GEvalJudgeProposer returns reference-free metrics for datasets without references."""
    
    proposer = GEvalJudgeProposer(
        executor_kwargs={"model": DummyModel()},
    )

    metrics = proposer.generate(small_dataset, target_measure="score", n_metrics=2)

    # Assertions -----------------------------------------------------------------
    assert isinstance(metrics, list)
    assert len(metrics) == 2

    for m in metrics:
        assert isinstance(m, GeneratedRefFreeGEvalMetric)
        assert not m.is_reference_based
        # Metric name should include axis name root + model name suffix + geval
        assert m.name.endswith("dummy/model".split("/")[-1] + "_geval")


def test_geval_judge_proposer_returns_expected_metrics_ref_based(mock_llm_calls, small_dataset_with_references):
    """Test that GEvalJudgeProposer returns reference-based metrics for datasets with references."""
    
    proposer = GEvalJudgeProposer(
        executor_kwargs={"model": DummyModel()},
    )

    metrics = proposer.generate(small_dataset_with_references, target_measure="score", n_metrics=2)

    # Assertions -----------------------------------------------------------------
    assert isinstance(metrics, list)
    assert len(metrics) == 2

    for m in metrics:
        assert isinstance(m, GeneratedRefBasedGEvalMetric)
        assert m.is_reference_based
        # Metric name should include axis name root + model name suffix + geval
        assert m.name.endswith("dummy/model".split("/")[-1] + "_geval")


def test_geval_judge_proposer_automatic_detection(mock_llm_calls, small_dataset, small_dataset_with_references):
    """Test that the proposer automatically detects reference vs reference-free datasets."""
    
    proposer = GEvalJudgeProposer(executor_kwargs={"model": DummyModel()})
    
    # Test reference-free detection
    ref_free_class = proposer._determine_executor_class(small_dataset)
    assert ref_free_class == GeneratedRefFreeGEvalMetric
    
    # Test reference-based detection
    ref_based_class = proposer._determine_executor_class(small_dataset_with_references)
    assert ref_based_class == GeneratedRefBasedGEvalMetric


def test_metric_properties(mock_llm_calls, small_dataset):
    """Test that generated metrics have the correct properties."""
    
    proposer = GEvalJudgeProposer(executor_kwargs={"model": DummyModel()})
    metrics = proposer.generate(small_dataset, target_measure="score", n_metrics=1)
    
    # Test that we can access the metric properties
    assert len(metrics) == 1
    metric = metrics[0]
    assert hasattr(metric, 'evaluation_criteria')
    assert hasattr(metric, 'task_description')
    assert hasattr(metric, 'evaluation_steps')
    assert hasattr(metric, 'possible_scores')
    
    # Test default values
    assert metric.possible_scores == [1, 2, 3, 4, 5]
    assert metric.auto_generate_steps == True


def test_custom_executor_class(mock_llm_calls, small_dataset):
    """Test that custom executor class can be specified."""
    
    proposer = GEvalJudgeProposer(
        executor_class=GeneratedRefBasedGEvalMetric,
        executor_kwargs={"model": DummyModel()},
    )

    metrics = proposer.generate(small_dataset, target_measure="score", n_metrics=1)

    # Should use the specified executor class even though dataset is reference-free
    assert len(metrics) == 1
    assert isinstance(metrics[0], GeneratedRefBasedGEvalMetric)


def test_proposer_name_and_description():
    """Test that the proposer has correct name and description."""
    proposer = GEvalJudgeProposer()
    
    assert proposer.get_name() == "GEvalJudgeProposer"
    assert "G-Eval" in proposer.get_description()
    assert str(proposer) == f"{proposer.name}: {proposer.description}"
    assert repr(proposer) == str(proposer) 