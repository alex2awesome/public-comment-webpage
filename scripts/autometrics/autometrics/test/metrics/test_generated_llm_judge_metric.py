import pytest

from autometrics.metrics.generated.GeneratedLLMJudgeMetric import (
    GeneratedRefFreeLLMJudgeMetric,
    GeneratedRefBasedLLMJudgeMetric
)


class DummyLM:
    """A minimal stand-in for a dspy.LM that always returns a constant score."""

    def __init__(self, constant_score: float = 4.0):
        self.model = "dummy/lm"
        self._score = constant_score

    # dspy expects objects to be callable returning str – we monkeypatch via ChainOfThought later.


@pytest.fixture()
def mock_llm_calls(monkeypatch):
    """Mock all LLM calls to avoid making actual API requests during testing."""
    
    # Mock DSPy ChainOfThought calls
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
    
    # Patch the DSPy ChainOfThought constructor to return a mock
    def mock_chain_of_thought(signature):
        class MockChainOfThought:
            def __call__(self, *args, **kwargs):
                return mock_chain_of_thought_call(self, *args, **kwargs)
        return MockChainOfThought()
    
    import dspy
    monkeypatch.setattr(dspy, "ChainOfThought", mock_chain_of_thought)
    
    # Mock the metric card builder
    def mock_generate_further_reading(metric):
        return "- Mock further reading"
    
    from autometrics.metrics.generated.utils import metric_card
    monkeypatch.setattr(metric_card, "generate_further_reading", mock_generate_further_reading)


@pytest.fixture()
def dummy_ref_free_metric(mock_llm_calls):
    dummy_lm = DummyLM(3.0)

    metric = GeneratedRefFreeLLMJudgeMetric(
        name="clarity_dummy",
        description="Clarity axis",
        axis="*Clarity*: How clear is the response?",
        model=dummy_lm,
        task_description="dummy task",
        max_workers=1,
    )
    return metric


@pytest.fixture()
def dummy_ref_based_metric(mock_llm_calls):
    dummy_lm = DummyLM(3.0)

    metric = GeneratedRefBasedLLMJudgeMetric(
        name="clarity_dummy_ref",
        description="Clarity axis with reference",
        axis="*Clarity*: How clear is the response compared to the reference?",
        model=dummy_lm,
        task_description="dummy task",
        max_workers=1,
    )
    return metric


def test_ref_free_metric_creation(dummy_ref_free_metric):
    """Test that the reference-free metric is created correctly."""
    assert dummy_ref_free_metric.name == "clarity_dummy"
    assert dummy_ref_free_metric.description == "Clarity axis"
    assert dummy_ref_free_metric.axis == "*Clarity*: How clear is the response?"
    assert dummy_ref_free_metric.task_description == "dummy task"
    assert not dummy_ref_free_metric.is_reference_based


def test_ref_based_metric_creation(dummy_ref_based_metric):
    """Test that the reference-based metric is created correctly."""
    assert dummy_ref_based_metric.name == "clarity_dummy_ref"
    assert dummy_ref_based_metric.description == "Clarity axis with reference"
    assert dummy_ref_based_metric.axis == "*Clarity*: How clear is the response compared to the reference?"
    assert dummy_ref_based_metric.task_description == "dummy task"
    assert dummy_ref_based_metric.is_reference_based


def test_ref_free_calculate_impl(dummy_ref_free_metric):
    """Test that _calculate_impl works correctly for reference-free."""
    result = dummy_ref_free_metric._calculate_impl("input text", "output text")
    assert isinstance(result, float)
    assert result == 4.0  # from the mocked LLM


def test_ref_based_calculate_impl(dummy_ref_based_metric):
    """Test that _calculate_impl works correctly for reference-based."""
    result = dummy_ref_based_metric._calculate_impl("input text", "output text", references=["reference text"])
    assert isinstance(result, float)
    assert result == 4.0  # from the mocked LLM


def test_ref_free_calculate_batched_impl(dummy_ref_free_metric):
    """Test that _calculate_batched_impl works correctly for reference-free."""
    inputs = ["input1", "input2"]
    outputs = ["output1", "output2"]
    
    results = dummy_ref_free_metric._calculate_batched_impl(inputs, outputs)
    assert len(results) == 2
    assert all(isinstance(r, float) for r in results)
    assert all(r == 4.0 for r in results)  # from the mocked LLM


def test_ref_based_calculate_batched_impl(dummy_ref_based_metric):
    """Test that _calculate_batched_impl works correctly for reference-based."""
    inputs = ["input1", "input2"]
    outputs = ["output1", "output2"]
    references = [["ref1"], ["ref2"]]
    
    results = dummy_ref_based_metric._calculate_batched_impl(inputs, outputs, references)
    assert len(results) == 2
    assert all(isinstance(r, float) for r in results)
    assert all(r == 4.0 for r in results)  # from the mocked LLM


def test_metric_card_generation(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test that metric card generation works without LLM calls."""
    assert hasattr(dummy_ref_free_metric, 'metric_card')
    assert dummy_ref_free_metric.metric_card is not None
    
    assert hasattr(dummy_ref_based_metric, 'metric_card')
    assert dummy_ref_based_metric.metric_card is not None


def test_metric_details_generation(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test that metric details can be generated without LLM calls."""
    details_free = dummy_ref_free_metric.generate_metric_details_ref_free()
    assert "reference-free" in details_free
    assert dummy_ref_free_metric.name in details_free
    assert dummy_ref_free_metric.axis in details_free
    
    details_based = dummy_ref_based_metric.generate_metric_details_ref_based()
    assert "reference-based" in details_based
    assert dummy_ref_based_metric.name in details_based
    assert dummy_ref_based_metric.axis in details_based


def test_python_code_generation(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test that Python code can be generated."""
    code_free = dummy_ref_free_metric._generate_python_code()
    assert "GeneratedRefFreeLLMJudgeMetric" in code_free
    assert dummy_ref_free_metric.name.replace(" ", "_").replace("-", "_") in code_free
    
    code_based = dummy_ref_based_metric._generate_python_code()
    assert "GeneratedRefBasedLLMJudgeMetric" in code_based
    assert dummy_ref_based_metric.name.replace(" ", "_").replace("-", "_") in code_based


def test_save_and_load(dummy_ref_free_metric, tmp_path):
    """Test that save and load work correctly."""
    save_path = tmp_path / "test_metric.json"
    dummy_ref_free_metric.save(str(save_path))
    
    # Check that file was created
    assert save_path.exists()
    
    # Load the metric (note: we can't test full loading without mocking the model creation)
    # But we can verify the file contains the expected data
    import json
    with open(save_path) as f:
        data = json.load(f)
    
    assert data["name"] == dummy_ref_free_metric.name
    assert data["description"] == dummy_ref_free_metric.description
    assert data["axis"] == dummy_ref_free_metric.axis
    assert data["task_description"] == dummy_ref_free_metric.task_description
    assert data["is_reference_based"] == dummy_ref_free_metric.is_reference_based


def test_serialize_and_deserialize(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test that GeneratedLLMJudgeMetric can be serialized and deserialized in memory."""
    
    # Test reference-free metric serialization
    serialized_data = dummy_ref_free_metric._serialize()
    
    # Check serialized data structure
    assert isinstance(serialized_data, dict)
    assert serialized_data["name"] == dummy_ref_free_metric.name
    assert serialized_data["description"] == dummy_ref_free_metric.description
    assert serialized_data["axis"] == dummy_ref_free_metric.axis
    assert serialized_data["task_description"] == dummy_ref_free_metric.task_description
    assert serialized_data["max_workers"] == dummy_ref_free_metric.max_workers
    assert serialized_data["is_reference_based"] == False
    
    # Deserialize from dictionary
    deserialized_metric = GeneratedRefFreeLLMJudgeMetric._deserialize(serialized_data)
    
    # Check that deserialized metric has same properties
    assert deserialized_metric.name == dummy_ref_free_metric.name
    assert deserialized_metric.description == dummy_ref_free_metric.description
    assert deserialized_metric.axis == dummy_ref_free_metric.axis
    assert deserialized_metric.task_description == dummy_ref_free_metric.task_description
    assert deserialized_metric.max_workers == dummy_ref_free_metric.max_workers
    assert deserialized_metric.is_reference_based == dummy_ref_free_metric.is_reference_based
    
    # Test reference-based metric serialization
    ref_based_data = dummy_ref_based_metric._serialize()
    deserialized_ref_based = GeneratedRefBasedLLMJudgeMetric._deserialize(ref_based_data)
    
    assert deserialized_ref_based.name == dummy_ref_based_metric.name
    assert deserialized_ref_based.is_reference_based == True 