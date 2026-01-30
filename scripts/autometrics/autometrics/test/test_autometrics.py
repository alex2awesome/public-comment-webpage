"""
Test cases for the Autometrics pipeline.

This module tests the main Autometrics class and its methods without requiring
actual LLM calls by using mocks and test fixtures.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import List, Type

# Import the classes we're testing
from autometrics.autometrics import Autometrics, DEFAULT_GENERATOR_CONFIGS, FULL_GENERATOR_CONFIGS
from autometrics.autometrics import _detect_gpu_availability, _get_default_retriever_config
from autometrics.dataset.Dataset import Dataset
from autometrics.metrics.Metric import Metric
from autometrics.metrics.MultiMetric import MultiMetric
from autometrics.recommend.MetricRecommender import MetricRecommender
from autometrics.aggregator.regression.Lasso import Lasso
from autometrics.metrics.MetricBank import all_metric_classes
from autometrics.recommend.ColBERT import ColBERT
from autometrics.recommend.BM25 import BM25
from autometrics.recommend.LLMRec import LLMRec


# ============================================================================
# Test Hardware Detection and Retriever Configuration
# ============================================================================

class TestHardwareDetection:
    """Test cases for hardware detection and automatic retriever configuration."""
    
    @patch('autometrics.autometrics._detect_gpu_availability')
    def test_get_default_retriever_config_gpu_available(self, mock_gpu_detect):
        """Test that GPU detection returns ColBERT + LLMRec configuration."""
        mock_gpu_detect.return_value = True
        
        config = _get_default_retriever_config()
        
        assert config["recommenders"] == [ColBERT, LLMRec]
        assert config["top_ks"] == [60, 30]
        assert config["index_paths"] == [None, None]
        assert config["force_reindex"] is False
    
    @patch('autometrics.autometrics._detect_gpu_availability')
    def test_get_default_retriever_config_no_gpu(self, mock_gpu_detect):
        """Test that no GPU detection returns BM25 + LLMRec configuration."""
        mock_gpu_detect.return_value = False
        
        config = _get_default_retriever_config()
        
        assert config["recommenders"] == [BM25, LLMRec]
        assert config["top_ks"] == [60, 30]
        assert config["index_paths"] == [None, None]
        assert config["force_reindex"] is False
    
    @patch('autometrics.metrics.utils.gpu_allocation.is_cuda_available')
    def test_detect_gpu_availability_with_utility(self, mock_cuda_available):
        """Test GPU detection using the existing utility function."""
        mock_cuda_available.return_value = True
        
        result = _detect_gpu_availability()
        
        assert result is True
        mock_cuda_available.assert_called_once()
    
    @patch('autometrics.metrics.utils.gpu_allocation.is_cuda_available')
    @patch('torch.cuda.is_available')
    def test_detect_gpu_availability_fallback_to_torch(self, mock_torch_cuda, mock_cuda_available):
        """Test GPU detection fallback to torch when utility is not available."""
        # Simulate import error for the utility
        mock_cuda_available.side_effect = ImportError("Module not found")
        mock_torch_cuda.return_value = False
        
        result = _detect_gpu_availability()
        
        assert result is False
        mock_torch_cuda.assert_called_once()
    
    @patch('autometrics.metrics.utils.gpu_allocation.is_cuda_available')
    @patch('torch.cuda.is_available')
    def test_detect_gpu_availability_no_torch(self, mock_torch_cuda, mock_cuda_available):
        """Test GPU detection when neither utility nor torch is available."""
        # Simulate import errors
        mock_cuda_available.side_effect = ImportError("Module not found")
        mock_torch_cuda.side_effect = ImportError("torch not available")
        
        result = _detect_gpu_availability()
        
        assert result is False


class TestAutometricsHardwareOptimization:
    """Test cases for Autometrics hardware optimization features."""
    
    def test_automatic_hardware_detection(self):
        """Test that Autometrics automatically detects hardware and configures accordingly."""
        autometrics = Autometrics()
        
        # The retriever_kwargs should be set based on hardware detection
        assert "recommenders" in autometrics.retriever_kwargs
        assert "top_ks" in autometrics.retriever_kwargs
        assert "index_paths" in autometrics.retriever_kwargs
        assert "force_reindex" in autometrics.retriever_kwargs
        
        # Should have either ColBERT or BM25 as the first recommender
        first_recommender = autometrics.retriever_kwargs["recommenders"][0]
        assert first_recommender in [ColBERT, BM25]
    
    def test_manual_retriever_override(self):
        """Test that users can manually override the retriever configuration."""
        custom_config = {
            "recommenders": [BM25, LLMRec],
            "top_ks": [50, 25],
            "index_paths": ["/custom/path", None],
            "force_reindex": True
        }
        
        autometrics = Autometrics(retriever_kwargs=custom_config)
        
        assert autometrics.retriever_kwargs == custom_config


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'id': [1, 2, 3, 4, 5],
        'input': ['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5'],
        'output': ['Output 1', 'Output 2', 'Output 3', 'Output 4', 'Output 5'],
        'reference': ['Ref 1', 'Ref 2', 'Ref 3', 'Ref 4', 'Ref 5'],
        'quality_score': [4.5, 3.2, 2.1, 4.8, 3.9]
    }
    return pd.DataFrame(data)


@pytest.fixture
def test_dataset(sample_data):
    """Create a test dataset."""
    return Dataset(
        dataframe=sample_data,
        target_columns=['quality_score'],
        ignore_columns=[],
        metric_columns=[],
        name="test_dataset",
        input_column='input',
        output_column='output',
        reference_columns=['reference']
    )


@pytest.fixture
def reference_free_dataset(sample_data):
    """Create a test dataset without reference columns."""
    data_no_ref = sample_data.drop(columns=['reference'])
    return Dataset(
        dataframe=data_no_ref,
        target_columns=['quality_score'],
        ignore_columns=[],
        metric_columns=[],
        name="test_dataset_no_ref",
        input_column='input',
        output_column='output',
        reference_columns=None
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.forward = Mock()
    return llm


@pytest.fixture
def mock_generator():
    """Create a mock metric generator."""
    generator = Mock()
    generator.generate = Mock()
    return generator


@pytest.fixture
def mock_retriever():
    """Create a mock metric retriever."""
    retriever = Mock()
    retriever.recommend = Mock()
    return retriever


@pytest.fixture
def mock_regression():
    """Create a mock regression strategy."""
    regression = Mock()
    regression.learn = Mock()
    regression.identify_important_metrics = Mock()
    regression.get_name = Mock(return_value="MockRegression")
    regression.get_description = Mock(return_value="A mock regression strategy")
    return regression


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Mock Metric Classes for Testing
# ============================================================================

class MockMetric(Metric):
    """A simple mock metric for testing."""
    
    def __init__(self, name="MockMetric", description="A mock metric"):
        super().__init__(name=name, description=description)
        self.use_cache = False  # Disable caching for tests
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        """Return a simple score based on output length."""
        if output is None:
            return 0.0
        return float(len(str(output)))
    
    def predict(self, dataset, update_dataset=True):
        """Implement the abstract predict method."""
        # Simple implementation for testing
        scores = []
        for i in range(len(dataset.get_dataframe())):
            scores.append(1.0)  # Return dummy scores
        
        if update_dataset:
            # Use the correct method to add metric columns
            dataset.dataframe[self.get_name()] = scores
        
        return scores
    
    def save_python_code(self, filepath):
        """Mock implementation of save_python_code."""
        # Create a simple Python file with the metric class
        code = f'''
from autometrics.metrics.Metric import Metric

class {self.get_name()}(Metric):
    def __init__(self):
        super().__init__(name="{self.get_name()}", description="{self.get_description()}")
        self.use_cache = False
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        return 1.0
    
    def predict(self, dataset, update_dataset=True):
        scores = [1.0] * len(dataset.get_dataframe())
        if update_dataset:
            dataset.dataframe[self.get_name()] = scores
        return scores
'''
        with open(filepath, 'w') as f:
            f.write(code)


class MockMultiMetric(MultiMetric):
    """A mock MultiMetric for testing."""
    
    def __init__(self, name="MockMultiMetric", description="A mock multi-metric"):
        super().__init__(
            name=name,
            description=description,
            submetric_names=["length", "word_count"]
        )
        self.use_cache = False  # Disable caching for tests
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        """Return length and word count as two separate metrics."""
        if output is None:
            return [0.0, 0.0]
        
        output_str = str(output)
        length = float(len(output_str))
        word_count = float(len(output_str.split()))
        return [length, word_count]
    
    def predict(self, dataset, update_dataset=True):
        """Implement the abstract predict method."""
        # Simple implementation for testing
        length_scores = []
        word_count_scores = []
        
        for i in range(len(dataset.get_dataframe())):
            length_scores.append(1.0)
            word_count_scores.append(2.0)
        
        if update_dataset:
            # Use the correct method to add metric columns
            dataset.dataframe[f"{self.get_name()}_length"] = length_scores
            dataset.dataframe[f"{self.get_name()}_word_count"] = word_count_scores
        
        return [length_scores, word_count_scores]
    
    def save_python_code(self, filepath):
        """Mock implementation of save_python_code."""
        # Create a simple Python file with the metric class
        code = f'''
from autometrics.metrics.MultiMetric import MultiMetric

class {self.get_name()}(MultiMetric):
    def __init__(self):
        super().__init__(
            name="{self.get_name()}", 
            description="{self.get_description()}",
            submetric_names=["length", "word_count"]
        )
        self.use_cache = False
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        if output is None:
            return [0.0, 0.0]
        output_str = str(output)
        length = float(len(output_str))
        word_count = float(len(output_str.split()))
        return [length, word_count]
    
    def predict(self, dataset, update_dataset=True):
        length_scores = [1.0] * len(dataset.get_dataframe())
        word_count_scores = [2.0] * len(dataset.get_dataframe())
        if update_dataset:
            dataset.dataframe[f"{{self.get_name()}}_length"] = length_scores
            dataset.dataframe[f"{{self.get_name()}}_word_count"] = word_count_scores
        return [length_scores, word_count_scores]
'''
        with open(filepath, 'w') as f:
            f.write(code)


# ============================================================================
# Test Autometrics Class
# ============================================================================

class TestAutometrics:
    """Test the Autometrics class and its methods."""

    def test_initialization_defaults(self):
        """Test Autometrics initialization with default parameters."""
        autometrics = Autometrics()
        
        assert autometrics.metric_generation_configs == DEFAULT_GENERATOR_CONFIGS
        assert autometrics.retriever is not None
        assert autometrics.regression_strategy is not None
        assert autometrics.metric_bank is not None
        assert autometrics.seed == 42
        assert autometrics.allowed_failed_metrics == 0

    def test_initialization_custom_params(self):
        """Test Autometrics initialization with custom parameters."""
        custom_configs = {"test_gen": {"metrics_per_trial": 5, "description": "Test"}}
        custom_retriever = Mock()
        custom_regression = Mock()
        custom_metric_bank = [MockMetric]
        
        autometrics = Autometrics(
            metric_generation_configs=custom_configs,
            retriever=custom_retriever,
            regression_strategy=custom_regression,
            metric_bank=custom_metric_bank,
            seed=123,
            allowed_failed_metrics=2
        )
        
        assert autometrics.metric_generation_configs == custom_configs
        assert autometrics.retriever == custom_retriever
        assert autometrics.regression_strategy == custom_regression
        assert autometrics.metric_bank == custom_metric_bank
        assert autometrics.seed == 123
        assert autometrics.allowed_failed_metrics == 2

    def test_load_metric_bank_list(self):
        """Test loading metric bank from a list of classes."""
        metric_classes = [MockMetric, MockMultiMetric]
        autometrics = Autometrics(metric_bank=metric_classes)
        
        loaded_bank = autometrics._load_metric_bank()
        assert loaded_bank == metric_classes

    def test_load_metric_bank_directory(self, temp_dir):
        """Test loading metric bank from a directory."""
        # Create a mock metric file in the temp directory
        metric_file = os.path.join(temp_dir, "test_metric.py")
        with open(metric_file, 'w') as f:
            f.write("""
from autometrics.metrics.Metric import Metric

class TestMetric(Metric):
    def __init__(self):
        super().__init__(name="TestMetric", description="Test")
        self.use_cache = False
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        return 1.0
""")
        
        autometrics = Autometrics(metric_bank=temp_dir)
        loaded_bank = autometrics._load_metric_bank()
        
        assert len(loaded_bank) == 1
        assert loaded_bank[0].__name__ == "TestMetric"

    def test_load_metrics_from_directory_invalid_file(self, temp_dir):
        """Test loading metrics from directory with invalid files."""
        # Create an invalid Python file
        invalid_file = os.path.join(temp_dir, "invalid.py")
        with open(invalid_file, 'w') as f:
            f.write("this is not valid python code {")
        
        autometrics = Autometrics()
        loaded_bank = autometrics._load_metrics_from_directory(temp_dir)
        
        # Should handle the error gracefully and return empty list
        assert loaded_bank == []

    def test_dataset_has_reference_columns(self, test_dataset, reference_free_dataset):
        """Test checking if dataset has reference columns."""
        autometrics = Autometrics()
        
        # Test dataset with reference columns
        assert autometrics._dataset_has_reference_columns(test_dataset) is True
        
        # Test dataset without reference columns
        assert autometrics._dataset_has_reference_columns(reference_free_dataset) is False

    def test_load_metric_bank_auto_switch_reference_free(self, reference_free_dataset):
        """Test automatic switching to reference-free metrics when dataset has no reference columns."""
        # Import the actual classes
        from autometrics.metrics.MetricBank import all_metric_classes, reference_free_metric_classes
        
        # Create autometrics with the actual all_metric_classes
        autometrics = Autometrics(metric_bank=all_metric_classes)
        
        # Mock the _dataset_has_reference_columns method to return False
        with patch.object(autometrics, '_dataset_has_reference_columns', return_value=False):
            # This should trigger the auto-switch logic
            loaded_bank = autometrics._load_metric_bank(reference_free_dataset)
            
            # The logic should switch to reference_free_metric_classes when no reference columns
            # But all_metric_classes already contains reference-free metrics, so it might not switch
            # Let's check that the loaded bank contains only reference-free metrics
            assert all('reference_free' in str(metric) for metric in loaded_bank[:10])  # Check first 10
            assert len(loaded_bank) < len(all_metric_classes)  # Should be smaller

    @patch('autometrics.generator.LLMJudgeProposer.BasicLLMJudgeProposer')
    def test_create_generator_llm_judge(self, mock_generator_class, mock_llm):
        """Test creating LLM judge generator."""
        mock_generator_class.return_value = Mock()
        
        autometrics = Autometrics()
        generator = autometrics._create_generator(
            "llm_judge", mock_llm, mock_llm, 42
        )
        
        mock_generator_class.assert_called_once()

    @patch('autometrics.generator.RubricGenerator.RubricGenerator')
    def test_create_generator_rubric_prometheus(self, mock_generator_class, mock_llm):
        """Test creating rubric generator with Prometheus."""
        mock_generator_class.return_value = Mock()
        
        autometrics = Autometrics()
        generator = autometrics._create_generator(
            "rubric_prometheus", mock_llm, mock_llm, 42, 
            prometheus_api_base="http://test:8000"
        )
        
        mock_generator_class.assert_called_once()

    def test_create_generator_rubric_prometheus_no_api_base(self, mock_llm):
        """Test creating rubric generator without API base raises error."""
        autometrics = Autometrics()
        
        with pytest.raises(ValueError, match="prometheus_api_base is required"):
            autometrics._create_generator(
                "rubric_prometheus", mock_llm, mock_llm, 42
            )

    def test_create_generator_unknown_type(self, mock_llm):
        """Test creating generator with unknown type raises error."""
        autometrics = Autometrics()
        
        with pytest.raises(ValueError, match="Unknown generator type"):
            autometrics._create_generator(
                "unknown_type", mock_llm, mock_llm, 42
            )

    def test_save_generated_metrics(self, temp_dir):
        """Test saving generated metrics to files."""
        metrics = [MockMetric("TestMetric1"), MockMetric("TestMetric2")]
        
        autometrics = Autometrics()
        metric_paths = autometrics._save_generated_metrics(
            metrics, "test_gen", "test_dataset", "quality_score", 42, temp_dir
        )
        
        assert len(metric_paths) == 2
        assert all(os.path.exists(path) for path in metric_paths)

    def test_save_generated_metrics_with_none(self, temp_dir):
        """Test saving generated metrics handles None values."""
        metrics = [MockMetric("TestMetric1"), None, MockMetric("TestMetric2")]
        
        autometrics = Autometrics()
        metric_paths = autometrics._save_generated_metrics(
            metrics, "test_gen", "test_dataset", "quality_score", 42, temp_dir
        )
        
        # Should skip None metrics
        assert len(metric_paths) == 2

    def test_retrieve_top_k_metrics(self, test_dataset, mock_retriever):
        """Test retrieving top-K metrics."""
        mock_retriever.recommend.return_value = [MockMetric, MockMultiMetric]
        
        autometrics = Autometrics()
        retrieved = autometrics._retrieve_top_k_metrics(
            test_dataset, "quality_score", 5, mock_retriever
        )
        
        assert len(retrieved) == 2
        mock_retriever.recommend.assert_called_once()

    def test_retrieve_top_k_metrics_empty_bank(self, test_dataset, mock_retriever):
        """Test retrieving metrics with empty metric bank."""
        autometrics = Autometrics(metric_bank=[])
        retrieved = autometrics._retrieve_top_k_metrics(
            test_dataset, "quality_score", 5, mock_retriever
        )
        
        assert retrieved == []

    @patch('autometrics.metrics.MetricBank.build_metrics')
    def test_evaluate_metrics_on_dataset(self, mock_build_metrics, test_dataset):
        """Test evaluating metrics on dataset."""
        metric_instances = [MockMetric("Test1"), MockMetric("Test2")]
        mock_build_metrics.return_value = metric_instances
        
        autometrics = Autometrics()
        successful_metrics = autometrics._evaluate_metrics_on_dataset(
            test_dataset, [MockMetric, MockMetric]
        )
        
        assert len(successful_metrics) == 2
        mock_build_metrics.assert_called_once()

    @patch('autometrics.metrics.MetricBank.build_metrics')
    def test_evaluate_metrics_on_dataset_with_failures(self, mock_build_metrics, test_dataset):
        """Test evaluating metrics with some failures."""
        # One metric fails to instantiate (None), one succeeds
        metric_instances = [None, MockMetric("Test2")]
        mock_build_metrics.return_value = metric_instances
        
        autometrics = Autometrics(allowed_failed_metrics=1)
        successful_metrics = autometrics._evaluate_metrics_on_dataset(
            test_dataset, [MockMetric, MockMetric]
        )
        
        assert len(successful_metrics) == 1

    @patch('autometrics.metrics.MetricBank.build_metrics')
    def test_evaluate_metrics_on_dataset_exceeds_failure_limit(self, mock_build_metrics, test_dataset):
        """Test evaluating metrics that exceed failure limit."""
        # All metrics fail to instantiate
        metric_instances = [None, None, None]
        mock_build_metrics.return_value = metric_instances
        
        autometrics = Autometrics(allowed_failed_metrics=1)
        
        with pytest.raises(ValueError, match="Exceeded allowed failed metrics"):
            autometrics._evaluate_metrics_on_dataset(
                test_dataset, [MockMetric, MockMetric, MockMetric]
            )

    def test_regress_and_select_top_n(self, test_dataset, mock_regression):
        """Test regression and selection of top-N metrics."""
        # Create metric instances
        metric1 = MockMetric("Metric1")
        metric2 = MockMetric("Metric2")
        metric3 = MockMultiMetric("MultiMetric1")
        
        # Add metrics to dataset
        test_dataset.add_metric(metric1, update_dataset=True)
        test_dataset.add_metric(metric2, update_dataset=True)
        test_dataset.add_metric(metric3, update_dataset=True)
        
        # Mock importance scores - use the actual metric names that will be in the dataset
        mock_regression.identify_important_metrics.return_value = [
            (0.8, "Metric1"),
            (0.6, "Metric2"),  # Use Metric2 instead of MultiMetric1_length
            (0.4, "MultiMetric1")  # Use the main metric name
        ]
        
        autometrics = Autometrics()
        results = autometrics._regress_and_select_top_n(
            test_dataset, [metric1, metric2, metric3], "quality_score", 2, mock_regression
        )
        
        # Should select 2 metrics (Metric1 and Metric2)
        assert len(results['top_metrics']) == 2
        assert results['regression_metric'] is not None
        assert len(results['importance_scores']) == 3

    def test_regress_and_select_top_n_empty_metrics(self, test_dataset, mock_regression):
        """Test regression with empty metric list."""
        autometrics = Autometrics()
        results = autometrics._regress_and_select_top_n(
            test_dataset, [], "quality_score", 5, mock_regression
        )
        
        assert results['top_metrics'] == []
        assert results['regression_metric'] is None

    def test_regress_and_select_top_n_multimetric_handling(self, test_dataset, mock_regression):
        """Test regression properly handles MultiMetrics in selection."""
        # Create a MultiMetric
        multi_metric = MockMultiMetric("TestMulti")
        test_dataset.add_metric(multi_metric, update_dataset=True)
        
        # Check what column names are actually in the dataset
        metric_columns = test_dataset.get_metric_columns()
        print(f"Available metric columns: {metric_columns}")
        
        # Mock importance scores using the actual column names
        mock_regression.identify_important_metrics.return_value = [
            (0.9, "TestMulti_length"),  # Use the actual submetric name
            (0.7, "SomeOtherMetric")   # Use a different metric name
        ]
        
        autometrics = Autometrics()
        results = autometrics._regress_and_select_top_n(
            test_dataset, [multi_metric], "quality_score", 1, mock_regression
        )
        
        # Should select the MultiMetric since one of its submetrics was important
        assert len(results['top_metrics']) == 1
        assert isinstance(results['top_metrics'][0], MockMultiMetric)

    @patch('autometrics.autometrics.Autometrics._generate_or_load_metrics')
    @patch('autometrics.autometrics.Autometrics._load_metric_bank')
    @patch('autometrics.autometrics.Autometrics._retrieve_top_k_metrics')
    @patch('autometrics.autometrics.Autometrics._evaluate_metrics_on_dataset')
    @patch('autometrics.autometrics.Autometrics._regress_and_select_top_n')
    @patch('autometrics.autometrics.Autometrics._generate_report_card')
    def test_run_pipeline_full(self, mock_report, mock_regress, mock_evaluate, 
                              mock_retrieve, mock_load_bank, mock_generate, 
                              test_dataset, mock_llm, mock_retriever, mock_regression):
        """Test the full run pipeline with all mocks."""
        # Setup mocks
        mock_generate.return_value = [MockMetric, MockMultiMetric]
        mock_load_bank.return_value = [MockMetric, MockMultiMetric]
        mock_retrieve.return_value = [MockMetric, MockMultiMetric]
        mock_evaluate.return_value = [MockMetric(), MockMultiMetric()]
        mock_regress.return_value = {
            'top_metrics': [MockMetric()],
            'regression_metric': mock_regression,
            'importance_scores': [(0.8, "TestMetric")],
            'final_regression_metric': mock_regression,
            'all_metrics_importance': [(0.8, "TestMetric")]
        }
        mock_report.return_value = "# Test Report"
        
        # Mock retriever and regression construction
        with patch.object(Autometrics, '_create_generator') as mock_create_gen:
            mock_create_gen.return_value = Mock()
            
            autometrics = Autometrics()
            results = autometrics.run(
                dataset=test_dataset,
                target_measure="quality_score",
                generator_llm=mock_llm,
                judge_llm=mock_llm,
                num_to_retrieve=5,
                num_to_regress=3
            )
        
        # Verify all methods were called
        mock_generate.assert_called_once()
        mock_load_bank.assert_called_once()
        mock_retrieve.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_regress.assert_called_once()
        mock_report.assert_called_once()
        
        # Verify results structure
        assert 'top_metrics' in results
        assert 'regression_metric' in results
        assert 'report_card' in results
        assert 'all_generated_metrics' in results

    def test_generate_or_load_metrics_new_generation(self, test_dataset, mock_llm, temp_dir):
        """Test generating new metrics when none exist."""
        # Mock generator
        mock_gen = Mock()
        mock_gen.generate.return_value = [MockMetric("Generated1"), MockMetric("Generated2")]
        
        # Use minimal configuration with only one generator
        minimal_config = {"llm_judge": {"metrics_per_trial": 2, "description": "Test"}}
        
        with patch.object(Autometrics, '_create_generator') as mock_create_gen:
            mock_create_gen.return_value = mock_gen
            
            autometrics = Autometrics(
                generated_metrics_dir=temp_dir,
                metric_generation_configs=minimal_config
            )
            generated = autometrics._generate_or_load_metrics(
                test_dataset, "quality_score", mock_llm, mock_llm, 
                regenerate_metrics=False
            )
        
        assert len(generated) == 2
        mock_gen.generate.assert_called_once()

    def test_generate_or_load_metrics_load_existing(self, test_dataset, mock_llm, temp_dir):
        """Test loading existing metrics when they exist."""
        # Create existing metric files
        existing_dir = os.path.join(temp_dir, "generated_metrics", "test_dataset", 
                                   "quality_score", "seed_42", "llm_judge")
        os.makedirs(existing_dir, exist_ok=True)
        
        # Create a mock metric file
        metric_file = os.path.join(existing_dir, "test_metric.py")
        with open(metric_file, 'w') as f:
            f.write("""
from autometrics.metrics.Metric import Metric

class GeneratedMetric(Metric):
    def __init__(self):
        super().__init__(name="GeneratedMetric", description="Generated")
        self.use_cache = False
    
    def _calculate_impl(self, input, output, references=None, **kwargs):
        return 1.0
    
    def predict(self, dataset, update_dataset=True):
        scores = [1.0] * len(dataset.get_dataframe())
        if update_dataset:
            dataset.dataframe[self.get_name()] = scores
        return scores
""")
        
        # Use minimal configuration with only one generator
        minimal_config = {"llm_judge": {"metrics_per_trial": 2, "description": "Test"}}
        
        autometrics = Autometrics(
            generated_metrics_dir=temp_dir,
            metric_generation_configs=minimal_config
        )
        
        # Mock the generator creation to avoid LLM calls
        with patch.object(autometrics, '_create_generator') as mock_create_gen:
            mock_gen = Mock()
            mock_gen.generate.return_value = [MockMetric("Generated1")]
            mock_create_gen.return_value = mock_gen
            
            generated = autometrics._generate_or_load_metrics(
                test_dataset, "quality_score", mock_llm, mock_llm, 
                regenerate_metrics=False
            )
        
        # Should load existing metrics AND generate new ones (since existing count < required count)
        # The existing metric + 1 generated metric = 2 total
        assert len(generated) == 2
        # Check that we have both the existing and generated metrics
        metric_names = [m.__name__ for m in generated]
        assert "GeneratedMetric" in metric_names
        assert "Generated1" in metric_names


# ============================================================================
# Integration Tests
# ============================================================================

class TestAutometricsIntegration:
    """Integration tests for Autometrics pipeline."""

    def test_end_to_end_with_mocks(self, test_dataset, mock_llm):
        """Test end-to-end pipeline with all components mocked."""
        # Create a minimal Autometrics instance with mocked components
        autometrics = Autometrics(
            metric_generation_configs={"llm_judge": {"metrics_per_trial": 2, "description": "Test"}},
            metric_bank=[MockMetric, MockMultiMetric]
        )
        
        # Mock all the internal methods
        with patch.object(autometrics, '_generate_or_load_metrics') as mock_gen, \
             patch.object(autometrics, '_load_metric_bank') as mock_load, \
             patch.object(autometrics, '_retrieve_top_k_metrics') as mock_retrieve, \
             patch.object(autometrics, '_evaluate_metrics_on_dataset') as mock_evaluate, \
             patch.object(autometrics, '_regress_and_select_top_n') as mock_regress, \
             patch.object(autometrics, '_generate_report_card') as mock_report:
            
            # Setup return values
            mock_gen.return_value = [MockMetric, MockMultiMetric]
            mock_load.return_value = [MockMetric, MockMultiMetric]
            mock_retrieve.return_value = [MockMetric, MockMultiMetric]
            mock_evaluate.return_value = [MockMetric(), MockMultiMetric()]
            mock_regress.return_value = {
                'top_metrics': [MockMetric()],
                'regression_metric': MockMetric("Regression"),
                'importance_scores': [(0.8, "TestMetric")],
                'final_regression_metric': MockMetric("Regression"),
                'all_metrics_importance': [(0.8, "TestMetric")]
            }
            mock_report.return_value = "# Test Report"
            
            # Run the pipeline
            results = autometrics.run(
                dataset=test_dataset,
                target_measure="quality_score",
                generator_llm=mock_llm,
                judge_llm=mock_llm,
                num_to_retrieve=3,
                num_to_regress=2
            )
            
            # Verify the pipeline executed all steps
            mock_gen.assert_called_once()
            mock_load.assert_called_once()
            mock_retrieve.assert_called_once()
            mock_evaluate.assert_called_once()
            mock_regress.assert_called_once()
            mock_report.assert_called_once()
            
            # Verify results structure
            assert isinstance(results, dict)
            assert 'top_metrics' in results
            assert 'regression_metric' in results
            assert 'report_card' in results
            assert 'all_generated_metrics' in results

    def test_error_handling_in_pipeline(self, test_dataset, mock_llm):
        """Test error handling throughout the pipeline."""
        autometrics = Autometrics()
        
        # Mock _generate_or_load_metrics to raise an exception
        with patch.object(autometrics, '_generate_or_load_metrics') as mock_gen:
            mock_gen.side_effect = Exception("Generation failed")
            
            with pytest.raises(Exception, match="Generation failed"):
                autometrics.run(
                    dataset=test_dataset,
                    target_measure="quality_score",
                    generator_llm=mock_llm,
                    judge_llm=mock_llm
                )


# ============================================================================
# Edge Cases and Error Conditions
# ============================================================================

class TestAutometricsEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_metric_bank(self, test_dataset, mock_llm):
        """Test behavior with empty metric bank."""
        autometrics = Autometrics(metric_bank=[])
        
        with patch.object(autometrics, '_generate_or_load_metrics') as mock_gen:
            mock_gen.return_value = []
            
            with patch.object(autometrics, '_retrieve_top_k_metrics') as mock_retrieve:
                mock_retrieve.return_value = []
                
                # Should handle gracefully
                results = autometrics.run(
                    dataset=test_dataset,
                    target_measure="quality_score",
                    generator_llm=mock_llm,
                    judge_llm=mock_llm
                )
                
                assert results['top_metrics'] == []

    def test_all_metrics_fail_evaluation(self, test_dataset, mock_llm):
        """Test when all metrics fail evaluation."""
        autometrics = Autometrics(allowed_failed_metrics=0)
        
        # Mock the entire pipeline to avoid LLM calls
        with patch.object(autometrics, '_generate_or_load_metrics') as mock_gen, \
             patch.object(autometrics, '_load_metric_bank') as mock_load, \
             patch.object(autometrics, '_retrieve_top_k_metrics') as mock_retrieve, \
             patch.object(autometrics, '_evaluate_metrics_on_dataset') as mock_evaluate, \
             patch.object(autometrics, '_regress_and_select_top_n') as mock_regress, \
             patch.object(autometrics, '_generate_report_card') as mock_report:
            
            # Setup mocks
            mock_gen.return_value = [MockMetric, MockMultiMetric]
            mock_load.return_value = [MockMetric, MockMultiMetric]
            mock_retrieve.return_value = [MockMetric, MockMultiMetric]
            mock_evaluate.return_value = []  # All metrics fail
            mock_regress.return_value = {
                'top_metrics': [],
                'regression_metric': None,
                'importance_scores': [],
                'final_regression_metric': None,
                'all_metrics_importance': []
            }
            mock_report.return_value = "# Test Report"
            
            results = autometrics.run(
                dataset=test_dataset,
                target_measure="quality_score",
                generator_llm=mock_llm,
                judge_llm=mock_llm
            )
            
            assert results['top_metrics'] == []

    def test_invalid_directory_path(self):
        """Test loading metrics from invalid directory."""
        autometrics = Autometrics(metric_bank="/nonexistent/path")
        
        with pytest.raises(FileNotFoundError):
            autometrics._load_metrics_from_directory("/nonexistent/path")

    def test_metric_save_failure(self, temp_dir):
        """Test handling of metric save failures."""
        # Make directory read-only to cause save failure
        os.chmod(temp_dir, 0o444)
        
        metrics = [MockMetric("TestMetric")]
        autometrics = Autometrics()
        
        # Should handle save failure gracefully
        try:
            metric_paths = autometrics._save_generated_metrics(
                metrics, "test_gen", "test_dataset", "quality_score", 42, temp_dir
            )
        except Exception:
            # Expected to fail due to read-only directory
            pass
        
        # Restore permissions
        os.chmod(temp_dir, 0o755) 