import pytest

from autometrics.metrics.generated.GeneratedGEvalMetric import (
    GeneratedRefFreeGEvalMetric,
    GeneratedRefBasedGEvalMetric
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
            evaluation_steps = "1. Read the input and output\n2. Evaluate clarity\n3. Assign score 1-5"
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
    
    # Mock the G-Eval function to return predictable scores
    def mock_geval(formatted_prompt, source, references, model_output, dspy_lm):
        return 4.0  # Fixed score for testing
    
    from autometrics.metrics.generated import GeneratedGEvalMetric
    monkeypatch.setattr(GeneratedGEvalMetric, "GEval", mock_geval)
    
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


@pytest.fixture()
def dummy_ref_free_metric(mock_llm_calls):
    dummy_lm = DummyLM(3.0)

    metric = GeneratedRefFreeGEvalMetric(
        name="clarity_geval",
        description="Clarity G-Eval metric",
        evaluation_criteria="*Clarity*: How clear is the response?",
        model=dummy_lm,
        task_description="dummy task",
        auto_generate_steps=False,
        evaluation_steps="1. Read input and output\n2. Evaluate clarity\n3. Assign score 1-5",
        max_workers=1,
    )
    return metric


@pytest.fixture()
def dummy_ref_based_metric(mock_llm_calls):
    dummy_lm = DummyLM(3.0)

    metric = GeneratedRefBasedGEvalMetric(
        name="clarity_geval_ref",
        description="Clarity G-Eval metric with reference",
        evaluation_criteria="*Clarity*: How clear is the response compared to the reference?",
        model=dummy_lm,
        task_description="dummy task",
        auto_generate_steps=False,
        evaluation_steps="1. Read input, output, and reference\n2. Evaluate clarity\n3. Assign score 1-5",
        max_workers=1,
    )
    return metric


def test_ref_free_metric_creation(dummy_ref_free_metric):
    """Test that the reference-free metric is created correctly."""
    assert dummy_ref_free_metric.name == "clarity_geval"
    assert dummy_ref_free_metric.description == "Clarity G-Eval metric"
    assert dummy_ref_free_metric.evaluation_criteria == "*Clarity*: How clear is the response?"
    assert dummy_ref_free_metric.task_description == "dummy task"
    assert not dummy_ref_free_metric.is_reference_based
    assert dummy_ref_free_metric.possible_scores == [1, 2, 3, 4, 5]
    assert dummy_ref_free_metric.auto_generate_steps == False


def test_ref_based_metric_creation(dummy_ref_based_metric):
    """Test that the reference-based metric is created correctly."""
    assert dummy_ref_based_metric.name == "clarity_geval_ref"
    assert dummy_ref_based_metric.description == "Clarity G-Eval metric with reference"
    assert dummy_ref_based_metric.evaluation_criteria == "*Clarity*: How clear is the response compared to the reference?"
    assert dummy_ref_based_metric.task_description == "dummy task"
    assert dummy_ref_based_metric.is_reference_based
    assert dummy_ref_based_metric.possible_scores == [1, 2, 3, 4, 5]
    assert dummy_ref_based_metric.auto_generate_steps == False


def test_evaluation_steps_generation(mock_llm_calls):
    """Test automatic evaluation steps generation."""
    dummy_lm = DummyLM(3.0)
    
    metric = GeneratedRefFreeGEvalMetric(
        name="auto_steps_geval",
        description="G-Eval metric with auto-generated steps",
        evaluation_criteria="clarity",
        model=dummy_lm,
        task_description="test task",
        auto_generate_steps=True,  # Should auto-generate
        max_workers=1,
    )
    
    # Should have generated evaluation steps
    assert metric.evaluation_steps is not None
    assert "Read the input and output" in metric.evaluation_steps


def test_ref_free_calculate_impl(dummy_ref_free_metric):
    """Test that _calculate_impl works correctly for reference-free."""
    result = dummy_ref_free_metric._calculate_impl("input text", "output text")
    assert isinstance(result, float)
    assert result == 4.0  # from the mocked G-Eval function


def test_ref_based_calculate_impl(dummy_ref_based_metric):
    """Test that _calculate_impl works correctly for reference-based."""
    result = dummy_ref_based_metric._calculate_impl("input text", "output text", references=["reference text"])
    assert isinstance(result, float)
    assert result == 4.0  # from the mocked G-Eval function


def test_ref_free_calculate_batched_impl(dummy_ref_free_metric):
    """Test that _calculate_batched_impl works correctly for reference-free."""
    inputs = ["input1", "input2"]
    outputs = ["output1", "output2"]
    
    results = dummy_ref_free_metric._calculate_batched_impl(inputs, outputs)
    assert len(results) == 2
    assert all(isinstance(r, float) for r in results)
    assert all(r == 4.0 for r in results)  # from the mocked G-Eval function


def test_ref_based_calculate_batched_impl(dummy_ref_based_metric):
    """Test that _calculate_batched_impl works correctly for reference-based."""
    inputs = ["input1", "input2"]
    outputs = ["output1", "output2"]
    references = [["ref1"], ["ref2"]]
    
    results = dummy_ref_based_metric._calculate_batched_impl(inputs, outputs, references)
    assert len(results) == 2
    assert all(isinstance(r, float) for r in results)
    assert all(r == 4.0 for r in results)  # from the mocked G-Eval function


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
    assert "G-Eval" in details_free
    assert dummy_ref_free_metric.name in details_free
    assert dummy_ref_free_metric.evaluation_criteria in details_free
    
    details_based = dummy_ref_based_metric.generate_metric_details_ref_based()
    assert "reference-based" in details_based
    assert "G-Eval" in details_based
    assert dummy_ref_based_metric.name in details_based
    assert dummy_ref_based_metric.evaluation_criteria in details_based


def test_python_code_generation(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test that Python code can be generated."""
    code_free = dummy_ref_free_metric._generate_python_code()
    assert "GeneratedRefFreeGEvalMetric" in code_free
    assert dummy_ref_free_metric.name.replace(" ", "_").replace("-", "_") in code_free
    assert "G-Eval" in code_free
    
    code_based = dummy_ref_based_metric._generate_python_code()
    assert "GeneratedRefBasedGEvalMetric" in code_based
    assert dummy_ref_based_metric.name.replace(" ", "_").replace("-", "_") in code_based
    assert "G-Eval" in code_based


def test_save_and_load(dummy_ref_free_metric, tmp_path):
    """Test that save and load work correctly."""
    save_path = tmp_path / "test_geval_metric.json"
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
    assert data["evaluation_criteria"] == dummy_ref_free_metric.evaluation_criteria
    assert data["task_description"] == dummy_ref_free_metric.task_description
    assert data["is_reference_based"] == dummy_ref_free_metric.is_reference_based


def test_serialize_and_deserialize(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test that GeneratedGEvalMetric can be serialized and deserialized in memory."""
    
    # Test reference-free metric serialization
    serialized_data = dummy_ref_free_metric._serialize()
    
    # Check serialized data structure
    assert isinstance(serialized_data, dict)
    assert serialized_data["name"] == dummy_ref_free_metric.name
    assert serialized_data["description"] == dummy_ref_free_metric.description
    assert serialized_data["evaluation_criteria"] == dummy_ref_free_metric.evaluation_criteria
    assert serialized_data["task_description"] == dummy_ref_free_metric.task_description
    assert serialized_data["evaluation_steps"] == dummy_ref_free_metric.evaluation_steps
    assert serialized_data["auto_generate_steps"] == dummy_ref_free_metric.auto_generate_steps
    assert serialized_data["possible_scores"] == dummy_ref_free_metric.possible_scores
    assert serialized_data["max_workers"] == dummy_ref_free_metric.max_workers
    assert serialized_data["is_reference_based"] == False
    
    # Deserialize from dictionary
    deserialized_metric = GeneratedRefFreeGEvalMetric._deserialize(serialized_data)
    
    # Check that deserialized metric has same properties
    assert deserialized_metric.name == dummy_ref_free_metric.name
    assert deserialized_metric.description == dummy_ref_free_metric.description
    assert deserialized_metric.evaluation_criteria == dummy_ref_free_metric.evaluation_criteria
    assert deserialized_metric.task_description == dummy_ref_free_metric.task_description
    assert deserialized_metric.evaluation_steps == dummy_ref_free_metric.evaluation_steps
    assert deserialized_metric.auto_generate_steps == dummy_ref_free_metric.auto_generate_steps
    assert deserialized_metric.possible_scores == dummy_ref_free_metric.possible_scores
    assert deserialized_metric.max_workers == dummy_ref_free_metric.max_workers
    assert deserialized_metric.is_reference_based == dummy_ref_free_metric.is_reference_based
    
    # Test reference-based metric serialization
    ref_based_data = dummy_ref_based_metric._serialize()
    deserialized_ref_based = GeneratedRefBasedGEvalMetric._deserialize(ref_based_data)
    
    assert deserialized_ref_based.name == dummy_ref_based_metric.name
    assert deserialized_ref_based.is_reference_based == True


def test_formatted_prompt_creation(dummy_ref_free_metric):
    """Test that the formatted prompt is created correctly."""
    assert hasattr(dummy_ref_free_metric, 'formatted_prompt')
    assert dummy_ref_free_metric.task_description in dummy_ref_free_metric.formatted_prompt
    assert dummy_ref_free_metric.evaluation_criteria in dummy_ref_free_metric.formatted_prompt
    assert dummy_ref_free_metric.evaluation_steps in dummy_ref_free_metric.formatted_prompt
    assert "{source}" in dummy_ref_free_metric.formatted_prompt
    assert "{references}" in dummy_ref_free_metric.formatted_prompt
    assert "{model_output}" in dummy_ref_free_metric.formatted_prompt


def test_criteria_generation_model_separation(mock_llm_calls):
    """Test that criteria generation model can be different from evaluation model."""
    eval_model = DummyLM()
    eval_model.model = "eval/model"
    
    criteria_model = DummyLM()
    criteria_model.model = "criteria/model"
    
    metric = GeneratedRefFreeGEvalMetric(
        name="dual_model_geval",
        description="G-Eval with separate models",
        evaluation_criteria="clarity",
        model=eval_model,
        criteria_generation_model=criteria_model,
        task_description="test task",
        auto_generate_steps=True,
        max_workers=1,
    )
    
    assert metric.model == eval_model
    assert metric.criteria_generation_model == criteria_model
    assert metric.criteria_generation_model != metric.model 