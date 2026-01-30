import pytest
import pandas as pd
import dspy
from unittest.mock import Mock, patch
from autometrics.dataset.Dataset import Dataset
from autometrics.generator.CodeGenerator import CodeGenerator
from autometrics.metrics.generated.GeneratedCodeMetric import GeneratedCodeReferenceBasedMetric, GeneratedCodeReferenceFreeMetric


@pytest.fixture
def test_dataset():
    """Create a test dataset for use in tests."""
    data = {
        'input': ['This is a test input', 'Another test input', 'Short input', 'Very long input text with many words'],
        'output': ['This is a test output', 'Another test output', 'Short out', 'Very long output text with many words and more'],
        'reference': ['This is a reference', 'Another reference', 'Short ref', 'Very long reference text'],
        'quality_score': [4.5, 3.2, 2.1, 4.8]
    }
    df = pd.DataFrame(data)
    return Dataset(
        dataframe=df,
        target_columns=['quality_score'],
        ignore_columns=[],
        metric_columns=[],
        name="test_dataset",
        input_column='input', 
        output_column='output',
        reference_columns=['reference']
    )


@pytest.fixture
def mock_model():
    """Create a mock DSPy model for testing."""
    model = Mock()
    model.forward = Mock()
    return model


class TestCodeGenerator:
    """Test the CodeGenerator class."""

    def test_initialization(self, test_dataset, mock_model):
        """Test that CodeGenerator initializes correctly."""
        generator = CodeGenerator(
            name="test_code_gen",
            description="Test code generator",
            train_dataset=test_dataset,
            task_description="Test task",
            proposer_model=mock_model,
            generate_axes=False
        )
        
        assert generator.name == "test_code_gen"
        assert generator.description == "Test code generator"
        assert generator.task_description == "Test task"
        assert generator.dataset == test_dataset
        assert generator.proposer_model == mock_model
        assert generator.generate_axes is False

    def test_preprocess_dataset(self, test_dataset, mock_model):
        """Test dataset preprocessing for good/bad examples."""
        generator = CodeGenerator(
            task_description="Test task",
            proposer_model=mock_model
        )
        
        good_examples, bad_examples = generator._preprocess_dataset(test_dataset, 'quality_score')
        
        assert isinstance(good_examples, list)
        assert isinstance(bad_examples, list)
        assert len(good_examples) > 0
        assert len(bad_examples) > 0

    def test_clean_generated_code(self, mock_model):
        """Test code cleaning functionality."""
        generator = CodeGenerator(proposer_model=mock_model)
        
        # Test markdown removal
        code_with_markdown = "```python\nreturn len(output)\n```"
        cleaned = generator._clean_generated_code(code_with_markdown)
        assert cleaned == "return len(output)"
        
        # Test plain code blocks
        code_with_blocks = "```\nreturn len(output)\n```"
        cleaned = generator._clean_generated_code(code_with_blocks)
        assert cleaned == "return len(output)"
        
        # Test code without blocks
        plain_code = "return len(output)"
        cleaned = generator._clean_generated_code(plain_code)
        assert cleaned == "return len(output)"

    @patch('autometrics.generator.CodeGenerator.CodeGenReferenceFree')
    def test_generate_reference_free_metrics(self, mock_code_gen_class, test_dataset, mock_model):
        """Test generating reference-free metrics."""
        # Mock the code generation
        mock_instance = Mock()
        mock_instance.forward.return_value = ("length_metric", "return len(output)")
        mock_code_gen_class.return_value = mock_instance
        
        generator = CodeGenerator(
            train_dataset=test_dataset,
            task_description="Test task",
            proposer_model=mock_model,
            generate_axes=False  # Avoid axis generation complexity
        )
        
        with patch.object(generator, '_get_axes_of_variation', return_value=["Quality"]) as mock_axes:
            metrics = generator.generate(
                train_dataset=test_dataset,
                target_column='quality_score',
                metric_type="reference_free",
                max_workers=1
            )
            
            assert isinstance(metrics, list)

    def test_generate_without_dataset_raises_error(self, mock_model):
        """Test that generate raises error when no dataset provided."""
        generator = CodeGenerator(
            task_description="Test task",
            proposer_model=mock_model,
            generate_axes=False
        )
        
        with pytest.raises(ValueError, match="No dataset provided"):
            generator.generate(metric_type="reference_free")


class TestGeneratedCodeMetrics:
    """Test the GeneratedCodeMetric classes."""

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple test dataset."""
        data = {
            'input': ['Hello world', 'Test input'],
            'output': ['Hello world!', 'Test output'],
            'reference': ['Hello world.', 'Test reference']
        }
        df = pd.DataFrame(data)
        return Dataset(
            dataframe=df,
            target_columns=[],
            ignore_columns=[],
            metric_columns=[],
            name="test_dataset",
            input_column='input', 
            output_column='output',
            reference_columns=['reference']
        )

    def test_reference_free_metric_basic_functionality(self):
        """Test basic functionality of reference-free generated code metric."""
        code = "return len(output)"
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="test_length_metric",
            description="Test length metric",
            generated_code=code,
            task_description="Test task"
        )
        
        # Test basic calculation
        result = metric.calculate("Hello", "Hello world!")
        assert result == 12.0
        
        # Test with empty output
        result = metric.calculate("Hello", "")
        assert result == 0.0

    def test_reference_based_metric_basic_functionality(self):
        """Test basic functionality of reference-based generated code metric."""
        code = """
if references and len(references) > 0:
    return len(set(output.split()) & set(references[0].split())) / max(len(set(references[0].split())), 1)
else:
    return 0.0
"""
        
        metric = GeneratedCodeReferenceBasedMetric(
            name="test_overlap_metric",
            description="Test word overlap metric",
            generated_code=code,
            task_description="Test task"
        )
        
        # Test with reference
        result = metric.calculate("Hello", "Hello world", ["Hello universe"])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_error_handling_in_generated_code(self):
        """Test error handling when generated code fails."""
        code = "return 1 / 0"  # Division by zero
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="test_error_metric",
            description="Test error handling",
            generated_code=code,
            task_description="Test task"
        )
        
        # Should return 0.0 when code fails
        result = metric.calculate("Hello", "Hello world!")
        assert result == 0.0

    def test_non_numeric_result_handling(self):
        """Test handling of non-numeric results from generated code."""
        code = "return 'not a number'"
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="test_string_metric",
            description="Test string result handling",
            generated_code=code,
            task_description="Test task"
        )
        
        # Should return 0.0 when result is not numeric
        result = metric.calculate("Hello", "Hello world!")
        assert result == 0.0

    def test_boolean_result_conversion(self):
        """Test conversion of boolean results to float."""
        code = "return len(output) > 5"
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="test_boolean_metric",
            description="Test boolean result conversion",
            generated_code=code,
            task_description="Test task"
        )
        
        # Should convert True to 1.0
        result = metric.calculate("Hello", "Hello world!")
        assert result == 1.0
        
        # Should convert False to 0.0
        result = metric.calculate("Hello", "Hi")
        assert result == 0.0

    def test_predict_method(self, simple_dataset):
        """Test the predict method on a dataset."""
        code = "return len(output)"
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="test_predict_metric",
            description="Test predict method",
            generated_code=code,
            task_description="Test task"
        )
        
        # Test predict method
        results = metric.predict(simple_dataset, update_dataset=False)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0] == 12.0  # "Hello world!" has 12 characters
        assert results[1] == 11.0  # "Test output" has 11 characters

    def test_get_generated_code(self):
        """Test getting the generated code."""
        code = "return len(output)"
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="test_code_access",
            description="Test code access",
            generated_code=code
        )
        
        assert metric.get_generated_code() == code

    def test_library_availability_constraint(self):
        """Test that code with unavailable libraries fails gracefully."""
        # Code that tries to use unavailable library
        code = """
import some_nonexistent_library_that_definitely_does_not_exist
return len(output)
"""
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="test_library_metric",
            description="Test library constraint",
            generated_code=code,
            task_description="Test task"
        )
        
        # Should return 0.0 when library import fails
        result = metric.calculate("Hello", "Hello world!")
        assert result == 0.0

    def test_complex_code_execution(self):
        """Test execution of more complex generated code."""
        code = """
import math
words = output.split()
if len(words) == 0:
    return 0.0
avg_word_length = sum(len(word) for word in words) / len(words)
return math.log(avg_word_length + 1)  # Log to normalize
"""
        
        metric = GeneratedCodeReferenceFreeMetric(
            name="test_complex_metric",
            description="Test complex code execution",
            generated_code=code,
            task_description="Test task"
        )
        
        result = metric.calculate("Hello", "Hello world!")
        assert isinstance(result, float)
        assert result > 0.0 