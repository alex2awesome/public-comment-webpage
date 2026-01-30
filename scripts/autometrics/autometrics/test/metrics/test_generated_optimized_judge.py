import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch
import dspy
from autometrics.metrics.generated.GeneratedOptimizedJudge import (
    GeneratedRefFreeOptimizedJudge,
    GeneratedRefBasedOptimizedJudge,
)


class DummyLM:
    model = "test-model"

    def __init__(self, constant_score: float = 4.0):
        self.constant_score = constant_score


@pytest.fixture()
def mock_llm_calls(monkeypatch):
    # Mock DSPy settings
    mock_settings = MagicMock()
    monkeypatch.setattr("dspy.settings", mock_settings)

    # Mock chain of thought responses
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

        return MockOutput()

    # Mock DSPy ChainOfThought
    def mock_chain_of_thought(signature):
        class MockChainOfThought:
            def __call__(self, *args, **kwargs):
                return mock_chain_of_thought_call(self, *args, **kwargs)
            
            def load(self, path):
                pass  # Mock loading optimized prompt

        return MockChainOfThought()

    monkeypatch.setattr("dspy.ChainOfThought", mock_chain_of_thought)

    # Mock DSPy context manager
    class MockContext:
        def __init__(self, lm=None):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass

    monkeypatch.setattr("dspy.settings.context", MockContext)

    # Mock further reading generation
    def mock_generate_further_reading(metric):
        return "- [Example Paper](https://example.com/paper)"

    monkeypatch.setattr("autometrics.metrics.generated.GeneratedOptimizedJudge.generate_further_reading", mock_generate_further_reading)

    # Mock metric card builder
    def mock_build(self):
        return f"# Metric Card for {self.metric.name}\n\nThis is a mock metric card."

    monkeypatch.setattr("autometrics.metrics.generated.utils.metric_card.MetricCardBuilder.build", mock_build)


@pytest.fixture()
def dummy_ref_free_metric(mock_llm_calls):
    """Create a dummy reference-free optimized judge metric for testing."""
    return GeneratedRefFreeOptimizedJudge(
        name="TestOptimizedJudge",
        description="Test optimized judge metric",
        axis="**Clarity**: How clear and understandable is the response?",
        model=DummyLM(),
        task_description="Answer questions helpfully",
        suggested_range=(1, 5),
        metric_card="provided"
    )


@pytest.fixture()
def dummy_ref_based_metric(mock_llm_calls):
    """Create a dummy reference-based optimized judge metric for testing."""
    return GeneratedRefBasedOptimizedJudge(
        name="TestOptimizedJudgeRefBased",
        description="Test optimized judge metric with references",
        axis="**Accuracy**: How factually correct is the information?",
        model=DummyLM(),
        task_description="Answer questions accurately with reference support",
        suggested_range=(1, 5),
        metric_card="provided"
    )


def test_ref_free_metric_creation(dummy_ref_free_metric):
    """Test basic creation of reference-free optimized judge metric."""
    metric = dummy_ref_free_metric
    
    assert metric.name == "TestOptimizedJudge"
    assert metric.description == "Test optimized judge metric"
    assert not metric.is_reference_based
    assert hasattr(metric, '_optimized_module')


def test_ref_based_metric_creation(dummy_ref_based_metric):
    """Test basic creation of reference-based optimized judge metric."""
    metric = dummy_ref_based_metric
    
    assert metric.name == "TestOptimizedJudgeRefBased"
    assert metric.description == "Test optimized judge metric with references"
    assert metric.is_reference_based
    assert hasattr(metric, '_optimized_module')


def test_optimized_prompt_loading_success(mock_llm_calls):
    """Test successful loading of optimized prompt."""
    # Create a temporary file with mock prompt data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"mock": "prompt_data"}')
        temp_path = f.name
    
    try:
        metric = GeneratedRefFreeOptimizedJudge(
            name="TestWithPrompt",
            description="Test with optimized prompt",
            axis="**Clarity**: Test axis",
            model=DummyLM(),
            optimized_prompt_path=temp_path,
        )
        
        assert metric.optimized_prompt_path == temp_path
        assert hasattr(metric, '_optimized_module')
    finally:
        os.unlink(temp_path)


def test_optimized_prompt_loading_failure(mock_llm_calls):
    """Test fallback when optimized prompt loading fails."""
    metric = GeneratedRefFreeOptimizedJudge(
        name="TestWithBadPrompt",
        description="Test with bad prompt path",
        axis="**Clarity**: Test axis",
        model=DummyLM(),
        optimized_prompt_path="/nonexistent/path.json",
    )
    
    # Should still work with fallback
    assert hasattr(metric, '_optimized_module')


def test_ref_free_calculate_impl(dummy_ref_free_metric):
    """Test _calculate_impl for reference-free metric."""
    result = dummy_ref_free_metric._calculate_impl(
        "What is 2+2?", 
        "2+2 equals 4."
    )
    
    assert isinstance(result, (int, float))
    assert 0 <= result <= 5  # Should be in expected range


def test_ref_based_calculate_impl(dummy_ref_based_metric):
    """Test _calculate_impl for reference-based metric."""
    result = dummy_ref_based_metric._calculate_impl(
        "What is 2+2?", 
        "2+2 equals 4.",
        references=["The sum of 2 and 2 is 4."]
    )
    
    assert isinstance(result, (int, float))
    assert 0 <= result <= 5  # Should be in expected range


def test_ref_free_calculate_batched_impl(dummy_ref_free_metric):
    """Test _calculate_batched_impl for reference-free metric."""
    inputs = ["What is 2+2?", "What is the capital of France?"]
    outputs = ["2+2 equals 4.", "The capital of France is Paris."]
    
    results = dummy_ref_free_metric._calculate_batched_impl(inputs, outputs)
    
    assert len(results) == 2
    assert all(isinstance(result, (int, float)) for result in results)


def test_ref_based_calculate_batched_impl(dummy_ref_based_metric):
    """Test _calculate_batched_impl for reference-based metric."""
    inputs = ["What is 2+2?", "What is the capital of France?"]
    outputs = ["2+2 equals 4.", "The capital of France is Paris."]
    references = [["The sum is 4."], ["Paris is the capital."]]
    
    results = dummy_ref_based_metric._calculate_batched_impl(inputs, outputs, references)
    
    assert len(results) == 2
    assert all(isinstance(result, (int, float)) for result in results)


def test_metric_card_generation(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test that metric cards are generated correctly."""
    ref_free_card = dummy_ref_free_metric.metric_card
    ref_based_card = dummy_ref_based_metric.metric_card
    
    assert isinstance(ref_free_card, str)
    assert isinstance(ref_based_card, str)
    assert len(ref_free_card) > 0
    assert len(ref_based_card) > 0


def test_metric_details_generation(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test that metric details are generated correctly."""
    ref_free_details = dummy_ref_free_metric.generate_metric_details_ref_free()
    ref_based_details = dummy_ref_based_metric.generate_metric_details_ref_based()
    
    assert isinstance(ref_free_details, str)
    assert isinstance(ref_based_details, str)
    assert "MIPROv2" in ref_free_details
    assert "MIPROv2" in ref_based_details
    assert "reference-free" in ref_free_details.lower()
    assert "reference-based" in ref_based_details.lower()


def test_python_code_generation(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test that Python code can be generated correctly."""
    ref_free_code = dummy_ref_free_metric._generate_python_code(include_metric_card=False)
    ref_based_code = dummy_ref_based_metric._generate_python_code(include_metric_card=False)
    
    assert isinstance(ref_free_code, str)
    assert isinstance(ref_based_code, str)
    assert "class" in ref_free_code
    assert "class" in ref_based_code
    assert "GeneratedRefFreeOptimizedJudge" in ref_free_code
    assert "GeneratedRefBasedOptimizedJudge" in ref_based_code


def test_save_and_load(dummy_ref_free_metric, tmp_path):
    """Test saving and loading of metrics (serialization/deserialization)."""
    # Test serialization
    serialized = dummy_ref_free_metric._serialize()
    
    assert isinstance(serialized, dict)
    assert "name" in serialized
    assert "description" in serialized
    assert "axis" in serialized
    assert "model" in serialized
    
    # Test deserialization
    deserialized = GeneratedRefFreeOptimizedJudge._deserialize(serialized)
    
    assert deserialized.name == dummy_ref_free_metric.name
    assert deserialized.description == dummy_ref_free_metric.description
    assert deserialized.axis == dummy_ref_free_metric.axis


def test_serialize_and_deserialize(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test comprehensive serialization and deserialization."""
    # Test reference-free metric
    ref_free_serialized = dummy_ref_free_metric._serialize()
    ref_free_deserialized = GeneratedRefFreeOptimizedJudge._deserialize(ref_free_serialized)
    
    assert ref_free_deserialized.name == dummy_ref_free_metric.name
    assert ref_free_deserialized.axis == dummy_ref_free_metric.axis
    assert ref_free_deserialized.task_description == dummy_ref_free_metric.task_description
    assert ref_free_deserialized.suggested_range == dummy_ref_free_metric.suggested_range
    assert not ref_free_deserialized.is_reference_based
    
    # Test reference-based metric
    ref_based_serialized = dummy_ref_based_metric._serialize()
    ref_based_deserialized = GeneratedRefBasedOptimizedJudge._deserialize(ref_based_serialized)
    
    assert ref_based_deserialized.name == dummy_ref_based_metric.name
    assert ref_based_deserialized.axis == dummy_ref_based_metric.axis
    assert ref_based_deserialized.task_description == dummy_ref_based_metric.task_description
    assert ref_based_deserialized.suggested_range == dummy_ref_based_metric.suggested_range
    assert ref_based_deserialized.is_reference_based


def test_score_parsing():
    """Test score parsing from various string formats."""
    from autometrics.metrics.generated.GeneratedOptimizedJudge import _OptimizedJudgeMetricMixin
    
    # Create a simple test instance
    class TestMixin(_OptimizedJudgeMetricMixin):
        def __init__(self):
            self.model = DummyLM()
            self.is_reference_based = False
            self.task_description = "Test"
            self.axis = "Test axis"
            self._optimized_module = MagicMock()
    
    mixin = TestMixin()
    
    # Mock the module response with various score formats
    test_cases = [
        ("4", 4.0),
        ("4.5", 4.5),
        ("4\nExtra text", 4.0),
        ("The score is 3", 3.0),
        ("Score: 5.0", 5.0),
        ("invalid", 0.0),
        (42, 42.0),
    ]
    
    for input_score, expected in test_cases:
        mixin._optimized_module.return_value.score = input_score
        result = mixin._call_optimized_llm("test input", "test output")
        assert result == expected, f"Failed for input {input_score}, expected {expected}, got {result}"


def test_optimization_specific_limitations(dummy_ref_free_metric):
    """Test that optimization-specific limitations are included."""
    limitations = dummy_ref_free_metric.generate_known_limitations()
    
    assert "optimization" in limitations.lower() or "miprov2" in limitations.lower()
    assert "training data" in limitations.lower()


def test_intended_use_generation(dummy_ref_free_metric):
    """Test that intended use information is generated."""
    intended_use = dummy_ref_free_metric.generate_intended_use()
    
    assert isinstance(intended_use, str)
    assert "Domain:" in intended_use
    assert "Tasks:" in intended_use
    assert "Best Suited For:" in intended_use
    assert "Not Recommended For:" in intended_use


def test_metric_implementation_details(dummy_ref_free_metric):
    """Test that metric implementation details are provided."""
    implementation = dummy_ref_free_metric.generate_metric_implementation()
    
    assert isinstance(implementation, str)
    assert "MIPROv2" in implementation
    assert "optimization" in implementation.lower()
    assert "efficiency" in implementation.lower() or "scalability" in implementation.lower() 