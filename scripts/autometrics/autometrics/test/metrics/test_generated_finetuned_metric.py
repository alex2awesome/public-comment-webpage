import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import os
import torch

from autometrics.metrics.generated.GeneratedFinetunedMetric import (
    GeneratedRefFreeFinetunedMetric,
    GeneratedRefBasedFinetunedMetric
)


@pytest.fixture()
def mock_model_components():
    """Mock the model loading components to avoid loading actual models during tests."""
    mock_model = Mock()
    mock_tokenizer = Mock()
    
    # Mock model outputs
    mock_outputs = Mock()
    mock_logits = torch.tensor([[2.5]])  # Mock prediction
    mock_outputs.logits = mock_logits
    mock_model.return_value = mock_outputs
    mock_model.eval.return_value = None
    mock_model.cuda.return_value = mock_model
    
    # Mock tokenizer outputs
    mock_tokenizer_output = {
        'input_ids': torch.tensor([[1, 2, 3, 4]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1]])
    }
    mock_tokenizer.return_value = mock_tokenizer_output
    
    return mock_model, mock_tokenizer


@pytest.fixture()
def mock_model_loading(monkeypatch, mock_model_components):
    """Mock all model loading functionality."""
    mock_model, mock_tokenizer = mock_model_components
    
    def mock_from_pretrained(*args, **kwargs):
        return mock_model, mock_tokenizer
    
    # Mock torch.cuda.is_available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    
    # Mock the model loading
    with patch('autometrics.metrics.generated.GeneratedFinetunedMetric.FastModel') as mock_fast_model:
        mock_fast_model.from_pretrained = mock_from_pretrained
        
        with patch('autometrics.metrics.generated.GeneratedFinetunedMetric.AutoModelForSequenceClassification'):
            with patch('autometrics.metrics.generated.GeneratedFinetunedMetric.AutoTokenizer'):
                yield mock_model, mock_tokenizer


@pytest.fixture()
def sample_training_stats():
    return {
        "train_size": 100,
        "val_size": 25,
        "target_mean": 0.7,
        "target_std": 0.2,
        "epochs": 3,
        "learning_rate": 5e-5,
    }


def test_ref_free_metric_creation(sample_training_stats):
    """Test that the reference-free metric is created correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_finetuned_metric",
            description="Test finetuned metric",
            model_path=temp_dir,
            task_description="Test task",
            target_measure="quality",
            dataset_name="test_dataset",
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        assert metric.name == "test_finetuned_metric"
        assert metric.description == "Test finetuned metric"
        assert metric.model_path == temp_dir
        assert metric.task_description == "Test task"
        assert metric.target_measure == "quality"
        assert metric.dataset_name == "test_dataset"
        assert metric.training_stats == sample_training_stats
        assert not metric.is_reference_based
        assert metric.max_workers == 1


def test_ref_based_metric_creation(sample_training_stats):
    """Test that the reference-based metric is created correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefBasedFinetunedMetric(
            name="test_finetuned_metric_ref",
            description="Test finetuned metric with reference",
            model_path=temp_dir,
            task_description="Test task",
            target_measure="quality",
            dataset_name="test_dataset",
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        assert metric.name == "test_finetuned_metric_ref"
        assert metric.description == "Test finetuned metric with reference"
        assert metric.model_path == temp_dir
        assert metric.task_description == "Test task"
        assert metric.target_measure == "quality"
        assert metric.dataset_name == "test_dataset"
        assert metric.training_stats == sample_training_stats
        assert metric.is_reference_based
        assert metric.max_workers == 1


def test_lazy_model_loading(mock_model_loading, sample_training_stats):
    """Test that models are loaded lazily."""
    mock_model, mock_tokenizer = mock_model_loading
    
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        # Model should not be loaded yet
        assert metric._model is None
        assert metric._tokenizer is None
        
        # Load the model
        model, tokenizer = metric._load_model_and_tokenizer()
        
        # Now model should be loaded
        assert model is mock_model
        assert tokenizer is mock_tokenizer
        assert metric._model is mock_model
        assert metric._tokenizer is mock_tokenizer


def test_predict_single_ref_free(mock_model_loading, sample_training_stats):
    """Test single prediction for reference-free metric."""
    mock_model, mock_tokenizer = mock_model_loading
    
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        result = metric._predict_single("input text", "output text")
        
        # Should return the mocked prediction (2.5)
        assert isinstance(result, float)
        assert result == 2.5
        
        # Check that tokenizer was called with correct format
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args[0]
        assert call_args[0] == "Input: input text Output: output text"


def test_predict_single_ref_based(mock_model_loading, sample_training_stats):
    """Test single prediction for reference-based metric."""
    mock_model, mock_tokenizer = mock_model_loading
    
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefBasedFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        result = metric._predict_single("input text", "output text", "reference text")
        
        # Should return the mocked prediction (2.5)
        assert isinstance(result, float)
        assert result == 2.5
        
        # Check that tokenizer was called with correct format (including reference)
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args[0]
        assert call_args[0] == "Input: input text Output: output text Reference: reference text"


def test_predict_single_ref_based_list_references(mock_model_loading, sample_training_stats):
    """Test single prediction for reference-based metric with list of references."""
    mock_model, mock_tokenizer = mock_model_loading
    
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefBasedFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        result = metric._predict_single("input text", "output text", ["ref1", "ref2"])
        
        # Should return the mocked prediction (2.5)
        assert isinstance(result, float)
        assert result == 2.5
        
        # Check that tokenizer was called with correct format (joined references)
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args[0]
        assert call_args[0] == "Input: input text Output: output text Reference: ref1 ref2"


def test_calculate_impl_ref_free(mock_model_loading, sample_training_stats):
    """Test _calculate_impl for reference-free metric."""
    mock_model, mock_tokenizer = mock_model_loading
    
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        result = metric._calculate_impl("input text", "output text")
        assert result == 2.5


def test_calculate_impl_ref_based(mock_model_loading, sample_training_stats):
    """Test _calculate_impl for reference-based metric."""
    mock_model, mock_tokenizer = mock_model_loading
    
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefBasedFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        result = metric._calculate_impl("input text", "output text", ["reference text"])
        assert result == 2.5


def test_calculate_batched_impl(mock_model_loading, sample_training_stats):
    """Test batched calculation with max_workers=1."""
    mock_model, mock_tokenizer = mock_model_loading
    
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        inputs = ["input1", "input2"]
        outputs = ["output1", "output2"]
        
        results = metric._calculate_batched_impl(inputs, outputs)
        
        assert len(results) == 2
        assert all(r == 2.5 for r in results)


def test_python_code_generation_ref_free(sample_training_stats):
    """Test Python code generation for reference-free metric."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        code = metric._generate_python_code()
        
        assert "GeneratedRefFreeFinetunedMetric" in code
        assert "test_metric" in code
        assert temp_dir in code
        assert "class test_metric_Finetuned" in code


def test_python_code_generation_ref_based(sample_training_stats):
    """Test Python code generation for reference-based metric."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefBasedFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        code = metric._generate_python_code()
        
        assert "GeneratedRefBasedFinetunedMetric" in code
        assert "test_metric" in code
        assert temp_dir in code
        assert "class test_metric_Finetuned" in code


def test_serialize_and_deserialize_ref_free(sample_training_stats):
    """Test serialization and deserialization for reference-free metric."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            task_description="Test task",
            target_measure="quality",
            dataset_name="test_dataset",
            training_stats=sample_training_stats,
            max_workers=4,
            max_seq_length=1024,
        )
        
        # Serialize
        serialized_data = original_metric._serialize()
        
        # Check serialized data structure
        assert isinstance(serialized_data, dict)
        assert serialized_data["name"] == "test_metric"
        assert serialized_data["description"] == "Test metric"
        assert serialized_data["model_path"] == temp_dir
        assert serialized_data["task_description"] == "Test task"
        assert serialized_data["target_measure"] == "quality"
        assert serialized_data["dataset_name"] == "test_dataset"
        assert serialized_data["training_stats"] == sample_training_stats
        assert serialized_data["max_workers"] == 4
        assert serialized_data["max_seq_length"] == 1024
        assert serialized_data["is_reference_based"] == False
        
        # Deserialize
        deserialized_metric = GeneratedRefFreeFinetunedMetric._deserialize(serialized_data)
        
        # Check that deserialized metric has same properties
        assert deserialized_metric.name == original_metric.name
        assert deserialized_metric.description == original_metric.description
        assert deserialized_metric.model_path == original_metric.model_path
        assert deserialized_metric.task_description == original_metric.task_description
        assert deserialized_metric.target_measure == original_metric.target_measure
        assert deserialized_metric.dataset_name == original_metric.dataset_name
        assert deserialized_metric.training_stats == original_metric.training_stats
        assert deserialized_metric.max_workers == original_metric.max_workers
        assert deserialized_metric.max_seq_length == original_metric.max_seq_length
        assert deserialized_metric.is_reference_based == original_metric.is_reference_based


def test_serialize_and_deserialize_ref_based(sample_training_stats):
    """Test serialization and deserialization for reference-based metric."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_metric = GeneratedRefBasedFinetunedMetric(
            name="test_metric_ref",
            description="Test metric with reference",
            model_path=temp_dir,
            task_description="Test task",
            target_measure="quality",
            dataset_name="test_dataset",
            training_stats=sample_training_stats,
            max_workers=4,
            max_seq_length=1024,
        )
        
        # Serialize and deserialize
        serialized_data = original_metric._serialize()
        deserialized_metric = GeneratedRefBasedFinetunedMetric._deserialize(serialized_data)
        
        # Check reference-based specific properties
        assert deserialized_metric.is_reference_based == True
        assert original_metric.is_reference_based == True


def test_metric_details_generation_ref_free(sample_training_stats):
    """Test metric details generation for reference-free metric."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            task_description="Test task",
            target_measure="quality",
            dataset_name="test_dataset",
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        details = metric.generate_metric_details_ref_free()
        
        assert "reference-free" in details
        assert "test_metric" in details
        assert "ModernBERT" in details
        assert "test_dataset" in details
        assert "quality" in details
        assert str(sample_training_stats["train_size"]) in details
        assert str(sample_training_stats["val_size"]) in details


def test_metric_details_generation_ref_based(sample_training_stats):
    """Test metric details generation for reference-based metric."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefBasedFinetunedMetric(
            name="test_metric_ref",
            description="Test metric with reference",
            model_path=temp_dir,
            task_description="Test task",
            target_measure="quality",
            dataset_name="test_dataset",
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        details = metric.generate_metric_details_ref_based()
        
        assert "reference-based" in details
        assert "test_metric_ref" in details
        assert "ModernBERT" in details
        assert "test_dataset" in details
        assert "quality" in details
        assert "Reference text" in details


def test_intended_use_generation(sample_training_stats):
    """Test intended use section generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            dataset_name="test_dataset",
            target_measure="quality",
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        intended_use = metric.generate_intended_use()
        
        assert "test_dataset" in intended_use
        assert "quality" in intended_use
        assert "Best Suited For" in intended_use
        assert "Not Recommended For" in intended_use


def test_metric_implementation_generation(sample_training_stats):
    """Test metric implementation section generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        implementation = metric.generate_metric_implementation()
        
        assert "AutoMetrics Fine-tuned Metric" in implementation
        assert "Unsloth" in implementation
        assert "ModernBERT" in implementation
        assert temp_dir in implementation


def test_known_limitations_generation(sample_training_stats):
    """Test known limitations section generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            dataset_name="test_dataset",
            target_measure="quality",
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        limitations = metric.generate_known_limitations()
        
        assert "Domain Specificity" in limitations
        assert "test_dataset" in limitations
        assert "quality" in limitations
        assert "Interpretability" in limitations


def test_further_reading_generation(sample_training_stats):
    """Test further reading section generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        further_reading = metric.generate_further_reading()
        
        assert "ModernBERT" in further_reading
        assert "Unsloth" in further_reading
        assert "arxiv.org" in further_reading or "github.com" in further_reading


def test_metric_card_generation(sample_training_stats):
    """Test metric card generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        metric = GeneratedRefFreeFinetunedMetric(
            name="test_metric",
            description="Test metric",
            model_path=temp_dir,
            task_description="Test task",
            dataset_name="test_dataset",
            target_measure="quality",
            training_stats=sample_training_stats,
            max_workers=1,
        )
        
        # Metric card should be generated during initialization
        assert hasattr(metric, 'metric_card')
        assert metric.metric_card is not None
        assert isinstance(metric.metric_card, str)
        assert len(metric.metric_card) > 0 