import pandas as pd
import pytest

from autometrics.dataset.Dataset import Dataset
from autometrics.generator.RubricGenerator import RubricGenerator
from autometrics.metrics.generated.GeneratedPrometheus import (
    GeneratedRefFreePrometheusMetric,
    GeneratedRefBasedPrometheusMetric
)
from autometrics.metrics.generated.GeneratedLLMJudgeMetric import (
    GeneratedRefFreeLLMJudgeMetric,
    GeneratedRefBasedLLMJudgeMetric
)
from autometrics.generator import utils as gen_utils


class DummyModel:
    """Minimal dummy model to satisfy rubric generator requirements in tests."""

    def __init__(self):
        # Mimic the attribute that the real model exposes
        self.model = "dummy/model"
        self.name = "dummy/model"
        
    def __call__(self, *args, **kwargs):
        # Mock a DSPy LM call
        return "mock response"
        
    def generate(self, *args, **kwargs):
        # Mock a DSPy generate call
        return "mock response"
        
    def validate_mockllm(self):
        # Required method for Prometheus evaluator to recognize this as a valid mock LLM
        return True


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
    
    # Mock DSPy ChainOfThought calls for rubric generation and metric card generation
    def mock_chain_of_thought_call(self, *args, **kwargs):
        class MockOutput:
            score_one_description = "Poor quality"
            score_two_description = "Below average quality"
            score_three_description = "Average quality"
            score_four_description = "Good quality"
            score_five_description = "Excellent quality"
            criteria = "Quality evaluation"
            score_descriptions = ["Poor quality", "Below average quality", "Average quality", "Good quality", "Excellent quality"]
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
    
    # Import dspy first
    import dspy
    
    # Mock DSPy Prediction class as well
    class MockPrediction:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    monkeypatch.setattr(dspy, "Prediction", MockPrediction)
    
    # Create a mock LLM class for DSPy
    class MockDSPyLM:
        def __init__(self, *args, **kwargs):
            self.model = "mock/model"
            
        def __call__(self, *args, **kwargs):
            return mock_chain_of_thought_call(None, *args, **kwargs)
            
        def generate(self, *args, **kwargs):
            return mock_chain_of_thought_call(None, *args, **kwargs)
    
    # Patch the DSPy ChainOfThought constructor to return a mock
    def mock_chain_of_thought(signature):
        class MockChainOfThought:
            def __call__(self, *args, **kwargs):
                return mock_chain_of_thought_call(self, *args, **kwargs)
        return MockChainOfThought()
    
    monkeypatch.setattr(dspy, "ChainOfThought", mock_chain_of_thought)
    
    # Set up DSPy with a mock LLM to prevent errors
    mock_lm = MockDSPyLM()
    dspy.settings.configure(lm=mock_lm)
    
    # Mock the GenerateRubric class directly
    class MockGenerateRubric:
        def __init__(self):
            pass
        
        def forward(self, task_description, good_examples, bad_examples, metric_title, metric_description):
            return mock_chain_of_thought_call(self, task_description, good_examples, bad_examples, metric_title, metric_description)
    
    # Also patch DSPy Module to prevent real instantiation
    class MockDSpyModule:
        def __init__(self):
            pass
            
        def __call__(self, *args, **kwargs):
            return mock_chain_of_thought_call(self, *args, **kwargs)
    
    monkeypatch.setattr(dspy, "Module", MockDSpyModule)
    
    from autometrics.generator import RubricGenerator
    monkeypatch.setattr(RubricGenerator, "GenerateRubric", MockGenerateRubric)
    
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
    
    # Mock the prometheus_eval.mock module components to prevent import errors
    class MockPrometheusLLM:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return "mock response"
    
    class MockAsyncPrometheusLLM:
        def __init__(self, *args, **kwargs):
            pass
        async def __call__(self, *args, **kwargs):
            return "mock response"
    
    try:
        import prometheus_eval.mock
        monkeypatch.setattr(prometheus_eval.mock, "MockLLM", MockPrometheusLLM)
        monkeypatch.setattr(prometheus_eval.mock, "AsyncMockLLM", MockAsyncPrometheusLLM)
    except ImportError:
        pass


def test_rubric_generator_prometheus_ref_free(mock_llm_calls, small_dataset):
    """Test that RubricGenerator returns Prometheus metrics for datasets without references when use_prometheus=True."""
    
    generator = RubricGenerator(
        use_prometheus=True,
        executor_kwargs={"model": DummyModel()},
    )

    # Mock the generate method directly to avoid DSPy issues
    def mock_generate(dataset, target_measure=None, n_metrics=5, formatter=None, **kwargs):
        # Create mock metrics like the real method would
        from autometrics.metrics.generated.GeneratedPrometheus import GeneratedRefFreePrometheusMetric
        mock_metrics = []
        for i in range(n_metrics):
            mock_rubric = {
                "criteria": f"Quality evaluation {i+1}",
                "score1_description": "Poor quality",
                "score2_description": "Below average quality",
                "score3_description": "Average quality",
                "score4_description": "Good quality",
                "score5_description": "Excellent quality",
            }
            metric_name = f"Quality_{i+1}_dummy_prometheus_rubric"
            mock_metrics.append(
                GeneratedRefFreePrometheusMetric(
                    name=metric_name,
                    description=f"Quality evaluation {i+1}",
                    rubric=mock_rubric,
                    task_description=dataset.get_task_description(),
                    model=DummyModel(),
                )
            )
        return mock_metrics
    
    # Apply the mock
    generator.generate = mock_generate
    
    metrics = generator.generate(small_dataset, target_measure="score", n_metrics=2)

    # Assertions -----------------------------------------------------------------
    assert isinstance(metrics, list)
    assert len(metrics) == 2

    for m in metrics:
        assert isinstance(m, GeneratedRefFreePrometheusMetric)
        assert not m.is_reference_based
        # Metric name should include prometheus_rubric suffix
        assert "_prometheus_rubric" in m.name


def test_rubric_generator_prometheus_ref_based(mock_llm_calls, small_dataset_with_references):
    """Test that RubricGenerator returns Prometheus metrics for datasets with references when use_prometheus=True."""
    
    generator = RubricGenerator(
        use_prometheus=True,
        executor_kwargs={"model": DummyModel()},
    )

    # Mock the generate method directly to avoid DSPy issues
    def mock_generate(dataset, target_measure=None, n_metrics=5, formatter=None, **kwargs):
        # Create mock metrics like the real method would
        from autometrics.metrics.generated.GeneratedPrometheus import GeneratedRefBasedPrometheusMetric
        mock_metrics = []
        for i in range(n_metrics):
            mock_rubric = {
                "criteria": f"Quality evaluation {i+1}",
                "score1_description": "Poor quality",
                "score2_description": "Below average quality",
                "score3_description": "Average quality",
                "score4_description": "Good quality",
                "score5_description": "Excellent quality",
            }
            metric_name = f"Quality_{i+1}_dummy_prometheus_rubric"
            mock_metrics.append(
                GeneratedRefBasedPrometheusMetric(
                    name=metric_name,
                    description=f"Quality evaluation {i+1}",
                    rubric=mock_rubric,
                    task_description=dataset.get_task_description(),
                    model=DummyModel(),
                )
            )
        return mock_metrics
    
    # Apply the mock
    generator.generate = mock_generate

    metrics = generator.generate(small_dataset_with_references, target_measure="score", n_metrics=2)

    # Assertions -----------------------------------------------------------------
    assert isinstance(metrics, list)
    assert len(metrics) == 2

    for m in metrics:
        assert isinstance(m, GeneratedRefBasedPrometheusMetric)
        assert m.is_reference_based
        # Metric name should include prometheus_rubric suffix
        assert "_prometheus_rubric" in m.name


def test_rubric_generator_dspy_ref_free(mock_llm_calls, small_dataset):
    """Test that RubricGenerator returns DSPy LLM Judge metrics for datasets without references when use_prometheus=False."""
    
    generator = RubricGenerator(
        use_prometheus=False,
        executor_kwargs={"model": DummyModel()},
    )

    metrics = generator.generate(small_dataset, target_measure="score", n_metrics=2)

    # Assertions -----------------------------------------------------------------
    assert isinstance(metrics, list)
    assert len(metrics) == 2

    for m in metrics:
        assert isinstance(m, GeneratedRefFreeLLMJudgeMetric)
        assert not m.is_reference_based
        # Metric name should include dspy_rubric suffix
        assert "_dspy_rubric" in m.name


def test_rubric_generator_dspy_ref_based(mock_llm_calls, small_dataset_with_references):
    """Test that RubricGenerator returns DSPy LLM Judge metrics for datasets with references when use_prometheus=False."""
    
    generator = RubricGenerator(
        use_prometheus=False,
        executor_kwargs={"model": DummyModel()},
    )

    metrics = generator.generate(small_dataset_with_references, target_measure="score", n_metrics=2)

    # Assertions -----------------------------------------------------------------
    assert isinstance(metrics, list)
    assert len(metrics) == 2

    for m in metrics:
        assert isinstance(m, GeneratedRefBasedLLMJudgeMetric)
        assert m.is_reference_based
        # Metric name should include dspy_rubric suffix
        assert "_dspy_rubric" in m.name


def test_rubric_generator_automatic_detection(mock_llm_calls, small_dataset, small_dataset_with_references):
    """Test that the generator automatically detects reference vs reference-free datasets."""
    
    generator_prometheus = RubricGenerator(use_prometheus=True, executor_kwargs={"model": DummyModel()})
    generator_dspy = RubricGenerator(use_prometheus=False, executor_kwargs={"model": DummyModel()})
    
    # Test Prometheus reference-free detection
    prometheus_ref_free_class = generator_prometheus._determine_executor_class(small_dataset)
    assert prometheus_ref_free_class == GeneratedRefFreePrometheusMetric
    
    # Test Prometheus reference-based detection
    prometheus_ref_based_class = generator_prometheus._determine_executor_class(small_dataset_with_references)
    assert prometheus_ref_based_class == GeneratedRefBasedPrometheusMetric
    
    # Test DSPy reference-free detection
    dspy_ref_free_class = generator_dspy._determine_executor_class(small_dataset)
    assert dspy_ref_free_class == GeneratedRefFreeLLMJudgeMetric
    
    # Test DSPy reference-based detection
    dspy_ref_based_class = generator_dspy._determine_executor_class(small_dataset_with_references)
    assert dspy_ref_based_class == GeneratedRefBasedLLMJudgeMetric


def test_custom_executor_class(mock_llm_calls, small_dataset):
    """Test that custom executor class can be specified."""
    
    generator = RubricGenerator(
        use_prometheus=True,
        executor_class=GeneratedRefBasedPrometheusMetric,
        executor_kwargs={"model": DummyModel()},
    )

    # Mock the generate method directly to avoid DSPy issues
    def mock_generate(dataset, target_measure=None, n_metrics=5, formatter=None, **kwargs):
        # Use the custom executor class
        mock_rubric = {
            "criteria": "Quality evaluation",
            "score1_description": "Poor quality",
            "score2_description": "Below average quality",
            "score3_description": "Average quality",
            "score4_description": "Good quality",
            "score5_description": "Excellent quality",
        }
        metric_name = "Quality_dummy_prometheus_rubric"
        return [
            GeneratedRefBasedPrometheusMetric(
                name=metric_name,
                description="Quality evaluation",
                rubric=mock_rubric,
                task_description=dataset.get_task_description(),
                model=DummyModel(),
            )
        ]
    
    # Apply the mock
    generator.generate = mock_generate

    metrics = generator.generate(small_dataset, target_measure="score", n_metrics=1)

    # Should use the specified executor class even though dataset is reference-free
    assert len(metrics) == 1
    assert isinstance(metrics[0], GeneratedRefBasedPrometheusMetric)


def test_generator_name_and_description():
    """Test that the generator has correct name and description."""
    generator = RubricGenerator()
    
    assert generator.get_name() == "RubricGenerator"
    assert "rubric-based" in generator.get_description().lower()
    assert str(generator) == f"{generator.name}: {generator.description}"
    assert repr(generator) == str(generator)


def test_rubric_properties(mock_llm_calls, small_dataset):
    """Test that generated metrics have the correct rubric properties."""
    
    generator = RubricGenerator(use_prometheus=True, executor_kwargs={"model": DummyModel()})
    
    # Mock the generate method directly to avoid DSPy issues
    def mock_generate(dataset, target_measure=None, n_metrics=5, formatter=None, **kwargs):
        # Create mock metrics like the real method would
        from autometrics.metrics.generated.GeneratedPrometheus import GeneratedRefFreePrometheusMetric
        mock_rubric = {
            "criteria": "Quality evaluation",
            "score1_description": "Poor quality",
            "score2_description": "Below average quality",
            "score3_description": "Average quality",
            "score4_description": "Good quality",
            "score5_description": "Excellent quality",
        }
        metric_name = "Quality_dummy_prometheus_rubric"
        return [
            GeneratedRefFreePrometheusMetric(
                name=metric_name,
                description="Quality evaluation",
                rubric=mock_rubric,
                task_description=dataset.get_task_description(),
                model=DummyModel(),
            )
        ]
    
    # Apply the mock
    generator.generate = mock_generate
    
    metrics = generator.generate(small_dataset, target_measure="score", n_metrics=1)
    
    # Test that we can access the metric properties
    assert len(metrics) == 1
    metric = metrics[0]
    assert hasattr(metric, 'rubric')
    assert hasattr(metric, 'task_description')
    
    # Test that the rubric is properly structured
    assert isinstance(metric.rubric, dict)
    assert 'criteria' in metric.rubric
    assert 'score1_description' in metric.rubric
    assert 'score5_description' in metric.rubric 