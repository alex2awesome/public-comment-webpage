import pytest

from autometrics.metrics.generated.GeneratedCodeMetric import (
    GeneratedRefFreeCodeMetric,
    GeneratedRefBasedCodeMetric
)


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
def mock_interpreter(monkeypatch):
    """Mock the interpreter to avoid actual code execution during testing."""
    
    class MockInterpreter:
        def execute(self, code, variables):
            # Mock execution - return length of output for most tests
            if 'result = compute_score' in code:
                output = variables.get('output', '')
                return len(output)
            return None
    
    def mock_get_interpreter(self):
        return MockInterpreter()
    
    # Mock both classes
    from autometrics.metrics.generated.GeneratedCodeMetric import _CodeMetricMixin
    monkeypatch.setattr(_CodeMetricMixin, "_get_interpreter", mock_get_interpreter)


@pytest.fixture()
def dummy_ref_free_metric(mock_llm_calls, mock_interpreter):
    metric = GeneratedRefFreeCodeMetric(
        name="length_code",
        description="Length-based code metric",
        generated_code="return len(output)",
        task_description="dummy task",
        measurement_axis="*Length*: Character count quality",
        max_workers=1,
    )
    return metric


@pytest.fixture()
def dummy_ref_based_metric(mock_llm_calls, mock_interpreter):
    metric = GeneratedRefBasedCodeMetric(
        name="length_code_ref",
        description="Length-based code metric with reference",
        generated_code="return len(output) / len(references[0]) if references else len(output)",
        task_description="dummy task",
        measurement_axis="*Length*: Character count quality relative to reference",
        max_workers=1,
    )
    return metric


def test_ref_free_metric_creation(dummy_ref_free_metric):
    """Test that the reference-free metric is created correctly."""
    assert dummy_ref_free_metric.name == "length_code"
    assert dummy_ref_free_metric.description == "Length-based code metric"
    assert dummy_ref_free_metric.generated_code == "return len(output)"
    assert dummy_ref_free_metric.task_description == "dummy task"
    assert not dummy_ref_free_metric.is_reference_based


def test_ref_based_metric_creation(dummy_ref_based_metric):
    """Test that the reference-based metric is created correctly."""
    assert dummy_ref_based_metric.name == "length_code_ref"
    assert dummy_ref_based_metric.description == "Length-based code metric with reference"
    assert dummy_ref_based_metric.task_description == "dummy task"
    assert dummy_ref_based_metric.is_reference_based


def test_ref_free_calculate_impl(dummy_ref_free_metric):
    """Test that _calculate_impl works correctly for reference-free."""
    result = dummy_ref_free_metric._calculate_impl("input text", "output text")
    assert isinstance(result, float)
    assert result == 11.0  # len("output text")


def test_ref_based_calculate_impl(dummy_ref_based_metric):
    """Test that _calculate_impl works correctly for reference-based."""
    result = dummy_ref_based_metric._calculate_impl("input text", "output text", references=["reference text"])
    assert isinstance(result, float)
    # The generated code is: return len(output) / len(references[0]) if references else len(output)
    # which should be len("output text") / len("reference text") = 11 / 14 ≈ 0.786
    assert abs(result - (11.0 / 14.0)) < 0.001


def test_ref_free_calculate_batched_impl(dummy_ref_free_metric):
    """Test that _calculate_batched_impl works correctly for reference-free."""
    inputs = ["input1", "input2"]
    outputs = ["output1", "output2"]
    
    results = dummy_ref_free_metric._calculate_batched_impl(inputs, outputs)
    assert len(results) == 2
    assert all(isinstance(r, float) for r in results)


def test_ref_based_calculate_batched_impl(dummy_ref_based_metric):
    """Test that _calculate_batched_impl works correctly for reference-based."""
    inputs = ["input1", "input2"]
    outputs = ["output1", "output2"]
    references = [["ref1"], ["ref2"]]
    
    results = dummy_ref_based_metric._calculate_batched_impl(inputs, outputs, references)
    assert len(results) == 2
    assert all(isinstance(r, float) for r in results)


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
    assert "code-based" in details_free
    assert dummy_ref_free_metric.name in details_free
    assert dummy_ref_free_metric.generated_code in details_free
    
    details_based = dummy_ref_based_metric.generate_metric_details_ref_based()
    assert "reference-based" in details_based
    assert "code-based" in details_based
    assert dummy_ref_based_metric.name in details_based
    assert dummy_ref_based_metric.generated_code in details_based


def test_python_code_generation(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test that Python code can be generated."""
    code_free = dummy_ref_free_metric._generate_python_code()
    assert "GeneratedRefFreeCodeMetric" in code_free
    assert dummy_ref_free_metric.name.replace(" ", "_").replace("-", "_") in code_free
    assert "code" in code_free.lower()
    
    code_based = dummy_ref_based_metric._generate_python_code()
    assert "GeneratedRefBasedCodeMetric" in code_based
    assert dummy_ref_based_metric.name.replace(" ", "_").replace("-", "_") in code_based
    assert "code" in code_based.lower()


def test_save_and_load(dummy_ref_free_metric, tmp_path):
    """Test that save and load work correctly."""
    save_path = tmp_path / "test_code_metric.json"
    dummy_ref_free_metric.save(str(save_path))
    
    # Check that file was created
    assert save_path.exists()
    
    # Load the metric (note: we can't test full loading without mocking the interpreter)
    # But we can verify the file contains the expected data
    import json
    with open(save_path) as f:
        data = json.load(f)
    
    assert data["name"] == dummy_ref_free_metric.name
    assert data["description"] == dummy_ref_free_metric.description
    assert data["generated_code"] == dummy_ref_free_metric.generated_code
    assert data["task_description"] == dummy_ref_free_metric.task_description
    assert data["is_reference_based"] == dummy_ref_free_metric.is_reference_based


def test_serialize_and_deserialize(dummy_ref_free_metric, dummy_ref_based_metric):
    """Test that GeneratedCodeMetric can be serialized and deserialized in memory."""
    
    # Test reference-free metric serialization
    serialized_data = dummy_ref_free_metric._serialize()
    
    # Check serialized data structure
    assert isinstance(serialized_data, dict)
    assert serialized_data["name"] == dummy_ref_free_metric.name
    assert serialized_data["description"] == dummy_ref_free_metric.description
    assert serialized_data["generated_code"] == dummy_ref_free_metric.generated_code
    assert serialized_data["task_description"] == dummy_ref_free_metric.task_description
    assert serialized_data["measurement_axis"] == dummy_ref_free_metric.measurement_axis
    assert serialized_data["max_workers"] == dummy_ref_free_metric.max_workers
    assert serialized_data["is_reference_based"] == False
    
    # Deserialize from dictionary
    deserialized_metric = GeneratedRefFreeCodeMetric._deserialize(serialized_data)
    
    # Check that deserialized metric has same properties
    assert deserialized_metric.name == dummy_ref_free_metric.name
    assert deserialized_metric.description == dummy_ref_free_metric.description
    assert deserialized_metric.generated_code == dummy_ref_free_metric.generated_code
    assert deserialized_metric.task_description == dummy_ref_free_metric.task_description
    assert deserialized_metric.measurement_axis == dummy_ref_free_metric.measurement_axis
    assert deserialized_metric.max_workers == dummy_ref_free_metric.max_workers
    assert deserialized_metric.is_reference_based == dummy_ref_free_metric.is_reference_based
    
    # Test reference-based metric serialization
    ref_based_data = dummy_ref_based_metric._serialize()
    deserialized_ref_based = GeneratedRefBasedCodeMetric._deserialize(ref_based_data)
    
    assert deserialized_ref_based.name == dummy_ref_based_metric.name
    assert deserialized_ref_based.is_reference_based == True


def test_code_parsing(dummy_ref_free_metric):
    """Test that code parsing works correctly."""
    # Test simple code without imports
    imports, logic = dummy_ref_free_metric._parse_generated_code("return len(output)")
    assert imports == ""
    assert logic == "return len(output)"
    
    # Test code with imports
    code_with_imports = "import re\nfrom collections import Counter\nreturn len(output)"
    imports, logic = dummy_ref_free_metric._parse_generated_code(code_with_imports)
    assert "import re" in imports
    assert "from collections import Counter" in imports
    assert logic == "return len(output)"


def test_code_indentation(dummy_ref_free_metric):
    """Test that code indentation works correctly."""
    code = "x = 1\nreturn x"
    indented = dummy_ref_free_metric._indent_code(code)
    expected = "    x = 1\n    return x"
    assert indented == expected
    
    # Test empty code
    empty_indented = dummy_ref_free_metric._indent_code("")
    assert empty_indented == "    return 0.0" 