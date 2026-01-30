import unittest
import dspy
import pandas as pd
from autometrics.metrics.llm_judge.LLMJudgeGEval import LLMJudgeGEval
from autometrics.dataset.Dataset import Dataset

class TestLLMJudgeGEval(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test dataset
        data = {
            'input': ['This is a test input', 'Another test input'],
            'output': ['This is a test output', 'Another test output'],
            'reference': ['This is a reference', 'Another reference']
        }
        df = pd.DataFrame(data)
        self.dataset = Dataset(
            dataframe=df,
            target_columns=[],
            ignore_columns=[],
            metric_columns=[],
            name="test_dataset",
            input_column='input', 
            output_column='output',
            reference_columns=['reference']
        )
        
        # Mock DSPy model for testing
        self.mock_model = dspy.LM("litellm_proxy/meta-llama/Meta-Llama-3.1-8B-Instruct", 
                                 api_base="http://future-hgx-1:7400/v1", api_key="None")

    def test_initialization(self):
        """Test that LLMJudgeGEval initializes correctly."""
        metric = LLMJudgeGEval(
            name="test_geval",
            description="Test G-Eval metric",
            dataset=self.dataset,
            evaluation_criteria="clarity",
            model=self.mock_model,
            task_description="Test task",
            auto_generate_steps=False
        )
        
        self.assertEqual(metric.name, "test_geval")
        self.assertEqual(metric.description, "Test G-Eval metric")
        self.assertEqual(metric.evaluation_criteria, "clarity")
        self.assertEqual(metric.task_description, "Test task")
        self.assertIsNotNone(metric.formatted_prompt)
        
        # Test that criteria generation model defaults to main model
        self.assertEqual(metric.model, metric.criteria_generation_model)

    def test_dual_model_initialization(self):
        """Test initialization with separate criteria generation model."""
        criteria_model = dspy.LM("litellm_proxy/gpt-4", api_base="http://example:8000/v1", api_key="None")
        
        metric = LLMJudgeGEval(
            name="test_geval_dual",
            description="Test G-Eval metric with dual models",
            dataset=self.dataset,
            evaluation_criteria="clarity",
            model=self.mock_model,
            criteria_generation_model=criteria_model,
            task_description="Test task",
            auto_generate_steps=False
        )
        
        # Test that models are different
        self.assertNotEqual(metric.model, metric.criteria_generation_model)
        self.assertEqual(metric.model, self.mock_model)
        self.assertEqual(metric.criteria_generation_model, criteria_model)

    def test_calculate_impl(self):
        """Test the _calculate_impl method."""
        metric = LLMJudgeGEval(
            name="test_geval",
            description="Test G-Eval metric",
            dataset=self.dataset,
            evaluation_criteria="clarity",
            model=self.mock_model,
            auto_generate_steps=False
        )
        
        # Mock the GEval function to return a fixed score
        original_geval = metric._calculate_impl
        
        def mock_calculate_impl(input, output, references=None, **kwargs):
            return 3.5  # Fixed score for testing
        
        metric._calculate_impl = mock_calculate_impl
        
        score = metric._calculate_impl("test input", "test output", ["reference"])
        self.assertEqual(score, 3.5)

    def test_predict_method(self):
        """Test the predict method with a small dataset."""
        metric = LLMJudgeGEval(
            name="test_geval",
            description="Test G-Eval metric",
            dataset=self.dataset,
            evaluation_criteria="clarity",
            model=self.mock_model,
            auto_generate_steps=False
        )
        
        # Mock the _calculate_impl to avoid actual API calls
        def mock_calculate_impl(input, output, references=None, **kwargs):
            return 3.0  # Fixed score for testing
        
        metric._calculate_impl = mock_calculate_impl
        
        # Test predict method
        results = metric.predict(self.dataset, update_dataset=False, num_workers=1)
        
        self.assertEqual(len(results), 2)  # Should have 2 scores
        self.assertTrue(all(isinstance(score, (int, float)) for score in results))

    def test_criteria_generation_model_usage(self):
        """Test that criteria generation uses the correct model."""
        criteria_model = dspy.LM("litellm_proxy/gpt-4", api_base="http://example:8000/v1", api_key="None")
        
        metric = LLMJudgeGEval(
            name="test_geval_criteria",
            description="Test criteria generation model usage",
            dataset=self.dataset,
            evaluation_criteria="clarity",
            model=self.mock_model,
            criteria_generation_model=criteria_model,
            task_description="Test task",
            auto_generate_steps=False  # We'll manually test generation
        )
        
        # Mock the criteria generation
        original_generate = metric._generate_evaluation_steps
        generation_model_used = None
        
        def mock_generate():
            nonlocal generation_model_used
            generation_model_used = metric.criteria_generation_model
            metric.evaluation_steps = "Mocked evaluation steps"
        
        metric._generate_evaluation_steps = mock_generate
        metric._generate_evaluation_steps()
        
        # Verify the correct model was referenced for generation
        self.assertEqual(generation_model_used, criteria_model)
        self.assertEqual(metric.evaluation_steps, "Mocked evaluation steps")

if __name__ == '__main__':
    unittest.main() 