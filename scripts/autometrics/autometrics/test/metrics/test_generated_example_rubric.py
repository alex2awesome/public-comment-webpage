import pytest
from unittest.mock import Mock, patch, MagicMock
import dspy
import json

from autometrics.metrics.generated.GeneratedExampleRubric import (
    GeneratedRefFreeExampleRubricMetric,
    GeneratedRefBasedExampleRubricMetric,
    exact_match_rounded,
    inverse_distance,
    get_wrapped_metric,
    LLMAsAJudge,
    LLMAsAJudgeSignature,
    _ExampleRubricMetricMixin,
)


class DummyDSPyModel:
    """Mock DSPy model for testing."""
    def __init__(self, name="test-model"):
        self.name = name
        self.model = name
        
    def __call__(self, *args, **kwargs):
        return "4.0"


@pytest.fixture()
def sample_train_buckets():
    """Sample training buckets for testing."""
    return [
        [  # Bucket 0 (score 1)
            {"text": "bad example 1", "task_description": "test task", "metric": "test metric", "suggested_range": (1, 5), "score": 1},
            {"text": "bad example 2", "task_description": "test task", "metric": "test metric", "suggested_range": (1, 5), "score": 1}
        ],
        [  # Bucket 1 (score 2)
            {"text": "poor example 1", "task_description": "test task", "metric": "test metric", "suggested_range": (1, 5), "score": 2}
        ],
        [  # Bucket 2 (score 3)
            {"text": "average example 1", "task_description": "test task", "metric": "test metric", "suggested_range": (1, 5), "score": 3}
        ],
        [  # Bucket 3 (score 4)
            {"text": "good example 1", "task_description": "test task", "metric": "test metric", "suggested_range": (1, 5), "score": 4}
        ],
        [  # Bucket 4 (score 5)
            {"text": "excellent example 1", "task_description": "test task", "metric": "test metric", "suggested_range": (1, 5), "score": 5},
            {"text": "excellent example 2", "task_description": "test task", "metric": "test metric", "suggested_range": (1, 5), "score": 5}
        ]
    ]


@pytest.fixture()
def sample_trainset():
    """Sample DSPy trainset for testing."""
    return [
        dspy.Example(
            text="bad example 1",
            task_description="test task",
            metric="test metric",
            suggested_range=(1, 5),
            score=1
        ).with_inputs('text', 'task_description', 'metric', 'suggested_range'),
        dspy.Example(
            text="excellent example 1",
            task_description="test task",
            metric="test metric",
            suggested_range=(1, 5),
            score=5
        ).with_inputs('text', 'task_description', 'metric', 'suggested_range')
    ]


@pytest.fixture()
def mock_dataset():
    """Mock dataset for testing."""
    dataset = Mock()
    dataset.name = "TestDataset"
    dataset.get_input_column.return_value = "input"
    dataset.get_output_column.return_value = "output"
    return dataset


def test_evaluation_functions():
    """Test the evaluation functions used in optimization."""
    # Test exact_match_rounded
    assert exact_match_rounded(3.1, 3.2) == 1  # Both round to 3
    assert exact_match_rounded(3.1, 4.2) == 0  # 3 vs 4
    assert exact_match_rounded(2.9, 3.1) == 1  # Both round to 3
    
    # Test inverse_distance
    assert inverse_distance(3, 3) == 1  # Exact match
    assert inverse_distance(3, 4) == 0.5  # Distance 1 -> 1/(1+1) = 0.5
    assert inverse_distance(3, 5) == 1/3  # Distance 2 -> 1/(2+1) = 1/3


def test_get_wrapped_metric():
    """Test the metric wrapper function."""
    def test_metric(x, y):
        return abs(x - y)
    
    wrapped = get_wrapped_metric(test_metric)
    
    # Create mock example and prediction
    example = Mock()
    example.score = 3
    pred = Mock()
    pred.score = 4
    
    result = wrapped(example, pred)
    assert result == 1  # abs(3 - 4) = 1


def test_llm_as_a_judge_signature():
    """Test that the DSPy signature is properly defined."""
    sig = LLMAsAJudgeSignature
    
    # Check that the signature exists and has the expected fields
    # Note: DSPy signatures use 'measure' not 'metric'
    # We'll check if the signature can be instantiated properly
    assert sig is not None
    
    # Test that we can create an instance (this validates the signature structure)
    try:
        # This should work if the signature is properly defined
        signature_instance = sig
        assert hasattr(signature_instance, '__annotations__')
        # Check that required fields are present in the annotations
        annotations = signature_instance.__annotations__
        expected_fields = ['text', 'task_description', 'measure', 'suggested_range', 'score']
        for field in expected_fields:
            assert field in annotations
    except Exception as e:
        pytest.fail(f"Failed to validate signature: {e}")


def test_generated_ref_free_metric_init():
    """Test initialization of reference-free example rubric metric."""
    model = DummyDSPyModel()
    
    metric = GeneratedRefFreeExampleRubricMetric(
        name="test_clarity",
        description="Test clarity metric",
        axis="*Clarity*: How clear is the response",
        model=model,
        task_description="Test task",
        
        metric_card="Mock metric card"  # Skip metric card generation
    )
    
    assert metric.name == "test_clarity"
    assert metric.description == "Test clarity metric"
    assert metric.axis == "*Clarity*: How clear is the response"
    assert metric.task_description == "Test task"
    assert metric.model == model
    assert metric.is_reference_based is False
    assert metric.attempts == 5
    assert metric.examples_per_range == 2
    assert metric.seed == 42
    assert metric.eval_function_name == 'inverse_distance'


def test_generated_ref_based_metric_init():
    """Test initialization of reference-based example rubric metric."""
    model = DummyDSPyModel()
    
    metric = GeneratedRefBasedExampleRubricMetric(
        name="test_relevance",
        description="Test relevance metric",
        axis="*Relevance*: How relevant is the response",
        model=model,
        task_description="Test task",
        
        metric_card="Mock metric card"  # Skip metric card generation
    )
    
    assert metric.name == "test_relevance"
    assert metric.description == "Test relevance metric"
    assert metric.axis == "*Relevance*: How relevant is the response"
    assert metric.is_reference_based is True


def test_custom_eval_function_setting():
    """Test that custom evaluation functions are properly set."""
    def custom_eval(x, y):
        return x * y
    
    model = DummyDSPyModel()
    metric = GeneratedRefFreeExampleRubricMetric(
        name="test_metric",
        description="Test metric",
        axis="Test axis",
        model=model,
        custom_eval_function=custom_eval,
        
        metric_card="Mock metric card"
    )
    
    assert metric.eval_function == custom_eval


def test_eval_function_name_setting():
    """Test that evaluation function names are properly resolved."""
    model = DummyDSPyModel()
    
    # Test exact_match_rounded
    metric1 = GeneratedRefFreeExampleRubricMetric(
        name="test_metric1",
        description="Test metric",
        axis="Test axis",
        model=model,
        eval_function_name='exact_match_rounded',
        
        metric_card="Mock metric card"
    )
    assert metric1.eval_function == exact_match_rounded
    
    # Test inverse_distance (default)
    metric2 = GeneratedRefFreeExampleRubricMetric(
        name="test_metric2",
        description="Test metric",
        axis="Test axis",
        model=model,
        eval_function_name='inverse_distance',
        
        metric_card="Mock metric card"
    )
    assert metric2.eval_function == inverse_distance


# NOTE: Optimization is now handled in the Generator (LLMJudgeExampleProposer), not the Executor
# The test_load_optimized_examples test above verifies that the Executor properly loads pre-optimized examples


def test_call_llm_judge_ref_free(mock_dataset):
    """Test LLM judge calling for reference-free metrics."""
    model = DummyDSPyModel()
    
    # Mock the dataset properly to avoid formatter issues
    mock_dataset.get_input_column.return_value = "input"
    mock_dataset.get_output_column.return_value = "output"
    mock_dataset.get_reference_columns.return_value = None  # Reference-free
    
    metric = GeneratedRefFreeExampleRubricMetric(
        name="test_metric",
        description="Test metric",
        axis="Test axis",
        model=model,
        train_dataset=mock_dataset,
        
        metric_card="Mock metric card"
    )
    
    # Mock the program
    mock_program = Mock()
    mock_prediction = Mock()
    mock_prediction.score = 4.0
    mock_program.return_value = mock_prediction
    metric.program = mock_program
    
    # Mock the format to avoid formatter complexities
    with patch('autometrics.util.format.get_default_formatter') as mock_get_formatter:
        mock_formatter = Mock()
        mock_formatter.return_value = "formatted text"
        mock_get_formatter.return_value = mock_formatter
        
        # Test the LLM judge call
        result = metric._call_llm_judge("input text", "output text")
        
        # Should return the mocked score
        assert result == 4.0
        
        # Verify the program was called
        assert mock_program.called


def test_call_llm_judge_ref_based(mock_dataset):
    """Test LLM judge calling for reference-based metrics."""
    model = DummyDSPyModel()
    
    # Mock the dataset properly for reference-based
    mock_dataset.get_input_column.return_value = "input"
    mock_dataset.get_output_column.return_value = "output"
    mock_dataset.get_reference_columns.return_value = ["reference"]
    
    metric = GeneratedRefBasedExampleRubricMetric(
        name="test_metric",
        description="Test metric",
        axis="Test axis",
        model=model,
        train_dataset=mock_dataset,
        
        metric_card="Mock metric card"
    )
    
    # Mock the program
    mock_program = Mock()
    mock_prediction = Mock()
    mock_prediction.score = 3.5
    mock_program.return_value = mock_prediction
    metric.program = mock_program
    
    # Mock the format to avoid formatter complexities
    with patch('autometrics.util.format.get_default_formatter') as mock_get_formatter:
        mock_formatter = Mock()
        mock_formatter.return_value = "formatted text with reference"
        mock_get_formatter.return_value = mock_formatter
        
        # Test the LLM judge call with references
        result = metric._call_llm_judge("input text", "output text", ["reference text"])
        
        # Should return the mocked score  
        assert result == 3.5
        
        # Verify the program was called
        assert mock_program.called


def test_calculate_impl_ref_free():
    """Test the _calculate_impl method for reference-free metrics."""
    model = DummyDSPyModel()
    metric = GeneratedRefFreeExampleRubricMetric(
        name="test_metric",
        description="Test metric",
        axis="Test axis",
        model=model,
        
        metric_card="Mock metric card"
    )
    
    with patch.object(metric, '_call_llm_judge') as mock_call_llm:
        mock_call_llm.return_value = 4.2
        
        result = metric._calculate_impl("input", "output", references="ignored")
        
        assert result == 4.2
        mock_call_llm.assert_called_once_with("input", "output")


def test_calculate_impl_ref_based():
    """Test the _calculate_impl method for reference-based metrics."""
    model = DummyDSPyModel()
    metric = GeneratedRefBasedExampleRubricMetric(
        name="test_metric",
        description="Test metric",
        axis="Test axis",
        model=model,
        
        metric_card="Mock metric card"
    )
    
    with patch.object(metric, '_call_llm_judge') as mock_call_llm:
        mock_call_llm.return_value = 3.8
        
        result = metric._calculate_impl("input", "output", references="reference")
        
        assert result == 3.8
        mock_call_llm.assert_called_once_with("input", "output", "reference")


def test_serialize_deserialize():
    """Test serialization and deserialization of metrics."""
    model = DummyDSPyModel("test-model")
    
    original_metric = GeneratedRefFreeExampleRubricMetric(
        name="test_metric",
        description="Test metric",
        axis="Test axis",
        model=model,
        task_description="Test task",
        suggested_range=(1, 5),
        attempts=3,
        examples_per_range=4,
        seed=123,
        eval_function_name='exact_match_rounded',
        metric_card="Mock metric card"
    )
    
    # Mock examples data
    examples_data = [{"text": "example", "score": 3}]
    # Use the correct DSPy path we discovered through testing
    original_metric.program.generate_score.predict.demos = examples_data
    
    # Serialize
    serialized = original_metric._serialize()
    
    # Check serialized data
    assert serialized["name"] == "test_metric"
    assert serialized["description"] == "Test metric"
    assert serialized["axis"] == "Test axis"
    assert serialized["task_description"] == "Test task"
    assert serialized["suggested_range"] == (1, 5)
    assert serialized["attempts"] == 3
    assert serialized["examples_per_range"] == 4
    assert serialized["seed"] == 123
    assert serialized["eval_function_name"] == 'exact_match_rounded'
    assert serialized["examples_data"] == examples_data


def test_generate_python_code():
    """Test generation of standalone Python code."""
    model = DummyDSPyModel("test-model")
    
    metric = GeneratedRefFreeExampleRubricMetric(
        name="test clarity metric",
        description="Test clarity metric",
        axis="*Clarity*: How clear is the response",
        model=model,
        task_description="Test task description",
        suggested_range=(1, 5),
        attempts=5,
        examples_per_range=2,
        seed=42,
        eval_function_name='inverse_distance',
        metric_card="Mock metric card"
    )
    
    # Mock examples data
    examples_data = [{"text": "example", "score": 3}]
    # Use the correct DSPy path we discovered through testing
    metric.program.generate_score.predict.demos = examples_data
    
    code = metric._generate_python_code(include_metric_card=True)
    
    # Check that the code contains expected elements
    assert "class test_clarity_metric_ExampleRubric" in code
    assert "GeneratedRefFreeExampleRubricMetric" in code
    assert '"test clarity metric"' in code
    assert '"Test clarity metric"' in code
    assert '"*Clarity*: How clear is the response"' in code
    assert '"Test task description"' in code
    assert "(1, 5)" in code
    assert "seed=42" in code  # Seed is still included for cache busting
    # Optimization parameters are not included in exported code since optimization happens in Generator
    assert "optimized_examples" in code  # But optimized examples should be included


def test_metric_details_template():
    """Test metric details template generation."""
    model = DummyDSPyModel("test-model")
    
    metric = GeneratedRefFreeExampleRubricMetric(
        name="test_metric",
        description="Test metric",
        axis="*Clarity*: How clear is the response",
        model=model,
        task_description="Test task",
        attempts=3,
        examples_per_range=2,
        seed=42,
        eval_function_name='inverse_distance',
        suggested_range=(1, 5),
        
        metric_card="Mock metric card"
    )
    
    # Test reference-free template
    details_ref_free = metric.generate_metric_details_ref_free()
    assert "reference-free" in details_ref_free
    assert "Example-based LLM judging" in details_ref_free
    assert "quintile-based bucketing" in details_ref_free
    assert "Optimization Attempts**: 3" in details_ref_free
    assert "Examples per Score Range**: 2" in details_ref_free
    assert "Random Seed**: 42" in details_ref_free
    assert "inverse_distance" in details_ref_free
    assert "Reference-Based?:** No" in details_ref_free
    
    # Test reference-based template
    ref_based_metric = GeneratedRefBasedExampleRubricMetric(
        name="test_metric_ref",
        description="Test metric",
        axis="*Clarity*: How clear is the response",
        model=model,
        
        metric_card="Mock metric card"
    )
    
    details_ref_based = ref_based_metric.generate_metric_details_ref_based()
    assert "reference-based" in details_ref_based
    assert "Reference-Based?:** Yes" in details_ref_based


@patch('dspy.ChainOfThought')
def test_metric_card_generation_components(mock_chain_of_thought):
    """Test that metric card generation components work properly."""
    # Mock the DSPy chain of thought responses
    mock_response = Mock()
    mock_response.domain = "Text Generation"
    mock_response.tasks = ["Text summarization", "Text classification"]
    mock_response.best_suited_for_circumstances = ["When examples are available", "For consistent scoring"]
    mock_response.not_recommended_for_circumstances = ["When training data is limited"]
    mock_response.biases = ["Model-specific biases", "Training data biases"]
    mock_response.task_misalignment_risks = ["Overfitting to examples"]
    mock_response.failure_cases = ["Contradictory examples", "Noisy training data"]
    
    mock_chain_instance = Mock()
    mock_chain_instance.return_value = mock_response
    mock_chain_of_thought.return_value = mock_chain_instance
    
    model = DummyDSPyModel("test-model")
    
    metric = GeneratedRefFreeExampleRubricMetric(
        name="test_metric",
        description="Test metric",
        axis="*Clarity*: How clear is the response",
        model=model,
        task_description="Test task",
        attempts=5,
        examples_per_range=2,
        eval_function_name='inverse_distance',
        
        metric_card="Mock metric card",
        metric_card_author_model=model
    )
    
    # Test intended use generation
    intended_use = metric.generate_intended_use()
    assert "Text Generation" in intended_use
    assert "Text summarization" in intended_use
    assert "When examples are available" in intended_use
    assert "When training data is limited" in intended_use
    
    # Test known limitations generation
    limitations = metric.generate_known_limitations()
    assert "Model-specific biases" in limitations
    assert "Overfitting to examples" in limitations
    assert "Contradictory examples" in limitations


def test_exclude_from_cache_key():
    """Test that heavy objects are excluded from cache key generation."""
    model = DummyDSPyModel("test-model")
    dataset = Mock()
    
    metric = GeneratedRefFreeExampleRubricMetric(
        name="test_metric",
        description="Test metric",
        axis="Test axis",
        model=model,
        train_dataset=dataset,
        
        metric_card="Mock metric card"
    )
    
    # The cache key exclusion should be set up during initialization
    # This is tested implicitly by ensuring the metric can be created
    # without issues from heavy objects in the cache key
    assert metric.name == "test_metric"
    assert metric.model == model
    assert metric.train_dataset == dataset


def test_optimization_warning_handling():
    """Test handling of optimization warnings when required data is missing."""
    model = DummyDSPyModel("test-model")
    
    # Test with missing buckets
    metric = GeneratedRefFreeExampleRubricMetric(
        name="test_metric",
        description="Test metric",
        axis="Test axis",
        model=model,
        train_buckets=[],  # Empty buckets
        trainset=[],       # Empty trainset
        
        metric_card="Mock metric card"
    )
    
    # Should handle missing data gracefully
    # The warning should be printed but metric should still be created
    assert metric.name == "test_metric"


def test_llm_as_a_judge_module():
    """Test the LLMAsAJudge DSPy module directly."""
    judge = LLMAsAJudge()
    
    # Mock the generate_score method
    mock_response = Mock()
    mock_response.score = "4.2\nExtra text"  # Test score extraction
    
    judge.generate_score = Mock()
    judge.generate_score.return_value = mock_response
    
    result = judge.forward(
        text="Test text",
        measure="Test measure",
        suggested_range=(1, 5),
        task_description="Test task"
    )
    
    # Should extract the score correctly and strip extra text
    assert result.score == 4.2
    assert result.text == "Test text"
    assert result.measure == "Test measure"


def test_llm_as_a_judge_score_extraction():
    """Test score extraction and error handling in LLMAsAJudge."""
    judge = LLMAsAJudge()
    
    # Mock the generate_score method
    mock_response = Mock()
    judge.generate_score = Mock()
    judge.generate_score.return_value = mock_response
    
    # Test normal score extraction
    mock_response.score = "3.5"
    result = judge.forward("text", "measure")
    assert result.score == 3.5
    
    # Test score with extra text
    mock_response.score = "4.2\nSome explanation"
    result = judge.forward("text", "measure")
    assert result.score == 4.2
    
    # Test invalid score (should default to 0.0)
    mock_response.score = "invalid"
    result = judge.forward("text", "measure")
    assert result.score == 0.0


def test_load_optimized_examples(sample_train_buckets, sample_trainset):
    """Test that the executor properly loads pre-optimized examples from the Generator."""
    model = DummyDSPyModel()
    
    # Sample optimized examples (what the Generator would pass)
    optimized_examples = [
        {"text": "example 1", "score": 4, "task_description": "test task", "metric": "test metric", "suggested_range": (1, 5)},
        {"text": "example 2", "score": 2, "task_description": "test task", "metric": "test metric", "suggested_range": (1, 5)}
    ]
    
    # Mock the _load_optimized_examples method to verify it's called
    with patch.object(_ExampleRubricMetricMixin, '_load_optimized_examples') as mock_load:
        metric = GeneratedRefFreeExampleRubricMetric(
            name="test_metric",
            description="Test metric",
            axis="Test axis",
            model=model,
            optimized_examples=optimized_examples,  # Pre-optimized examples from Generator
            attempts=2,
            examples_per_range=1,
            
            metric_card="Mock metric card"
        )
        
        # Verify that _load_optimized_examples was called with the correct examples
        mock_load.assert_called_once_with(optimized_examples)
    
    # Also verify that the optimize flag is correctly set to False (no optimization in Executor)
    assert metric.optimize == False
    assert metric.attempts == 2
    assert metric.examples_per_range == 1 