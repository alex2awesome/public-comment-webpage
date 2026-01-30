import pytest
from unittest.mock import Mock, patch, MagicMock
import dspy

from autometrics.generator.LLMJudgeExampleProposer import LLMJudgeExampleProposer
from autometrics.generator.utils import build_metric_name_from_axis
from autometrics.metrics.generated.GeneratedExampleRubric import (
    GeneratedRefFreeExampleRubricMetric,
    GeneratedRefBasedExampleRubricMetric
)


class DummyModel(dspy.LM):
    """Mock LLM model for testing that inherits from dspy.LM."""
    def __init__(self, name="test-model"):
        self.name = name
        self.model = name
        super().__init__(model=name)
    
    def __call__(self, *args, **kwargs):
        # Mock response for DSPy calls with proper format
        return MockResponse()
    
    def generate(self, *args, **kwargs):
        return ["Mock response"]
    
    def basic_request(self, prompt, **kwargs):
        """Mock the basic_request method that DSPy uses internally"""
        return MockCompletion()


class MockResponse:
    """Mock response that DSPy expects"""
    def __init__(self):
        self.choices = [MockChoice()]


class MockChoice:
    """Mock choice object"""
    def __init__(self):
        self.message = MockMessage()


class MockMessage:
    """Mock message with proper content"""
    def __init__(self):
        self.content = """
Reasoning: This is mock reasoning for the axis generation.

Axes Of Variation:
1. *Clarity*: How clear and understandable is the response
2. *Relevance*: How relevant is the response to the given input
"""


class MockCompletion:
    """Mock completion for basic_request"""
    def __init__(self):
        self.choices = [MockChoice()]


@pytest.fixture()
def small_dataset():
    """Mock dataset for testing."""
    mock_dataset = Mock()
    mock_dataset.get_task_description.return_value = "Test task description"
    mock_dataset.get_target_columns.return_value = ["score"]
    mock_dataset.get_reference_columns.return_value = []  # Reference-free by default
    
    # Mock dataframe with simple data for quintile buckets
    import pandas as pd
    df = pd.DataFrame({
        'input': ['input1', 'input2', 'input3', 'input4', 'input5'] * 4,
        'output': ['output1', 'output2', 'output3', 'output4', 'output5'] * 4,
        'score': [1, 2, 3, 4, 5] * 4
    })
    mock_dataset.get_dataframe.return_value = df
    mock_dataset.get_input_column.return_value = 'input'
    mock_dataset.get_output_column.return_value = 'output'
    
    return mock_dataset


@pytest.fixture()
def reference_based_dataset():
    """Mock reference-based dataset for testing."""
    mock_dataset = Mock()
    mock_dataset.get_task_description.return_value = "Test task description with references"
    mock_dataset.get_target_columns.return_value = ["score"]
    mock_dataset.get_reference_columns.return_value = ["reference"]  # Reference-based
    
    # Mock dataframe with reference data
    import pandas as pd
    df = pd.DataFrame({
        'input': ['input1', 'input2', 'input3', 'input4', 'input5'] * 4,
        'output': ['output1', 'output2', 'output3', 'output4', 'output5'] * 4,
        'reference': ['ref1', 'ref2', 'ref3', 'ref4', 'ref5'] * 4,
        'score': [1, 2, 3, 4, 5] * 4
    })
    mock_dataset.get_dataframe.return_value = df
    mock_dataset.get_input_column.return_value = 'input'
    mock_dataset.get_output_column.return_value = 'output'
    
    return mock_dataset


def test_init_basic():
    """Test basic initialization of LLMJudgeExampleProposer."""
    generator = LLMJudgeExampleProposer()
    
    assert generator.name == "LLMJudgeExampleProposer"
    assert "optimized examples" in generator.description
    assert generator.attempts == 5
    assert generator.examples_per_range == 2
    assert generator.seed == 42
    assert generator.eval_function_name == 'inverse_distance'


def test_init_with_custom_params():
    """Test initialization with custom parameters."""
    model = DummyModel("test-judge")
    generator = LLMJudgeExampleProposer(
        name="CustomExampleProposer",
        description="Custom description",
        generator_llm=model,
        executor_kwargs={"model": model},
        attempts=10,
        examples_per_range=5,
        seed=123,
        eval_function_name='exact_match_rounded'
    )
    
    assert generator.name == "CustomExampleProposer"
    assert generator.description == "Custom description"
    assert generator.attempts == 10
    assert generator.examples_per_range == 5
    assert generator.seed == 123
    assert generator.eval_function_name == 'exact_match_rounded'
    assert generator.judge_model_name == "test-judge"


def test_determine_executor_class_reference_free(small_dataset):
    """Test executor class determination for reference-free datasets."""
    generator = LLMJudgeExampleProposer()
    executor_class = generator._determine_executor_class(small_dataset)
    
    assert executor_class == GeneratedRefFreeExampleRubricMetric


def test_determine_executor_class_reference_based(reference_based_dataset):
    """Test executor class determination for reference-based datasets."""
    generator = LLMJudgeExampleProposer()
    executor_class = generator._determine_executor_class(reference_based_dataset)
    
    assert executor_class == GeneratedRefBasedExampleRubricMetric


def test_prepare_dataset_bucketted(small_dataset):
    """Test dataset bucketing functionality."""
    generator = LLMJudgeExampleProposer()
    
    # Mock formatter - row is passed as pandas Series, not tuple
    def mock_formatter(row):
        if isinstance(row, tuple):  # iterrows() format
            return f"Input: {row[1]['input']}, Output: {row[1]['output']}"
        else:  # Series format
            return f"Input: {row['input']}, Output: {row['output']}"
    
    buckets, trainset = generator._prepare_dataset_bucketted(
        small_dataset,
        target_column="score",
        task_description="Test task",
        metric_name="Test metric",
        formatter=mock_formatter,
        suggested_range=(1, 5)
    )
    
    # Check that we have 5 buckets (quintiles)
    assert len(buckets) == 5
    
    # Check that examples are distributed across buckets
    total_examples = sum(len(bucket) for bucket in buckets)
    assert total_examples == 20  # 20 rows in our mock dataset
    
    # Check trainset format
    assert len(trainset) == 20
    for example in trainset[:3]:  # Check first few examples
        assert hasattr(example, 'text')
        assert hasattr(example, 'task_description')
        assert hasattr(example, 'metric')
        assert hasattr(example, 'suggested_range')
        assert hasattr(example, 'score')


def test_generate_creates_correct_metrics(small_dataset):
    """Test that generate creates the correct number and type of metrics."""
    model = DummyModel("test-model")
    generator_lm = DummyModel("generator-model")
    generator = LLMJudgeExampleProposer(
        generator_llm=generator_lm,
        executor_kwargs={"model": model}
    )
    
    # Mock the complex parts - no more axis generation needed
    with patch.object(generator, '_determine_executor_class') as mock_executor_class, \
         patch.object(generator, '_prepare_dataset_bucketted') as mock_prepare_buckets:
        
        # Mock the executor class
        mock_metric_class = Mock()
        mock_executor_class.return_value = mock_metric_class
        
        # Mock the dataset bucketing
        mock_prepare_buckets.return_value = ([], [])  # buckets, trainset
        
        # Mock metric instance
        mock_metric = Mock()
        mock_metric.name = "score_test-model_examples"
        mock_metric_class.return_value = mock_metric
        
        metrics = generator.generate(small_dataset, target_measure="score", n_metrics=2)
        
        # Should only return 1 metric since example-based works on target measure directly
        assert len(metrics) == 1
        
        # Check that the executor class was called with correct parameters
        assert mock_metric_class.call_count == 1


def test_generate_respects_n_metrics_limit(small_dataset):
    """Test that generate respects the n_metrics parameter (but returns max 1 for example-based)."""
    model = DummyModel("test-model")
    generator_lm = DummyModel("generator-model")
    generator = LLMJudgeExampleProposer(
        generator_llm=generator_lm,
        executor_kwargs={"model": model}
    )
    
    # Mock the complex parts
    with patch.object(generator, '_determine_executor_class') as mock_executor_class, \
         patch.object(generator, '_prepare_dataset_bucketted') as mock_prepare_buckets:
        
        # Mock the executor class
        mock_metric_class = Mock()
        mock_executor_class.return_value = mock_metric_class
        mock_metric_class.return_value = Mock()
        
        # Mock the dataset bucketing
        mock_prepare_buckets.return_value = ([], [])  # buckets, trainset
        
        # Request multiple metrics
        metrics = generator.generate(small_dataset, target_measure="score", n_metrics=5)
        
        # Should only create 1 metric since example-based works on target measure directly
        assert len(metrics) == 1
        assert mock_metric_class.call_count == 1


def test_metric_name_parsing():
    """Test that metric names are correctly parsed from axes."""
    generator = LLMJudgeExampleProposer(executor_kwargs={"model": DummyModel("gpt4")})
    
    # Test different name formats
    test_cases = [
        ("*Clarity*: How clear is the response", "Clarity_gpt4_examples"),
        ("**Relevance**: How relevant is the response", "Relevance_gpt4_examples"),
        ("Simple Name: Description", "Simple_Name_gpt4_examples"),
        ("No Asterisks", "No_Asterisks_gpt4_examples"),
    ]
    
    for axis, expected_name in test_cases:
        metric_name = build_metric_name_from_axis(axis, suffix="_gpt4_examples")
        assert metric_name == expected_name


def test_get_formatter_fallback(small_dataset):
    """Test that _get_formatter provides a fallback when no dataset is provided."""
    generator = LLMJudgeExampleProposer()
    
    # Test with dataset
    formatter_with_dataset = generator._get_formatter(small_dataset)
    assert callable(formatter_with_dataset)
    
    # Test with None dataset
    formatter_without_dataset = generator._get_formatter(None)
    assert callable(formatter_without_dataset)
    
    # Test fallback behavior
    test_input = "test"
    result = formatter_without_dataset(test_input)
    assert result == str(test_input)


def test_string_representations():
    """Test string representation methods."""
    generator = LLMJudgeExampleProposer(
        name="Test Generator",
        description="Test Description"
    )
    
    assert generator.get_name() == "Test Generator"
    assert generator.get_description() == "Test Description"
    assert str(generator) == "Test Generator: Test Description"
    assert repr(generator) == "Test Generator: Test Description"


def test_custom_eval_function():
    """Test initialization with custom evaluation function."""
    def custom_eval_func(x, y):
        return abs(x - y)
    
    generator = LLMJudgeExampleProposer(
        custom_eval_function=custom_eval_func
    )
    
    assert generator.custom_eval_function == custom_eval_func


def test_executor_kwargs_handling():
    """Test that executor_kwargs are properly handled and passed through."""
    model = DummyModel("test-model")
    custom_kwargs = {
        "model": model,
        "max_workers": 16,
        "custom_param": "test_value"
    }
    
    generator = LLMJudgeExampleProposer(
        executor_kwargs=custom_kwargs
    )
    
    assert generator.executor_kwargs == custom_kwargs
    assert generator.judge_model == model
    assert generator.judge_model_name == "test-model"


def test_random_seed_setting():
    """Test that random seed is properly set for reproducibility."""
    import random
    
    # Test with custom seed
    generator = LLMJudgeExampleProposer(seed=999)
    
    # The generator should have set the random seed
    # This is hard to test directly, but we can check the seed was stored
    assert generator.seed == 999 