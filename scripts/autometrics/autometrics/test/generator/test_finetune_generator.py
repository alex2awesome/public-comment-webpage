import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import os

from autometrics.dataset.Dataset import Dataset
from autometrics.generator.FinetuneGenerator import FinetuneGenerator
from autometrics.metrics.generated.GeneratedFinetunedMetric import (
    GeneratedRefFreeFinetunedMetric,
    GeneratedRefBasedFinetunedMetric
)


@pytest.fixture()
def small_dataset():
    df = pd.DataFrame(
        {
            "input": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "output": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "score": [0.9, 0.8, 0.2, 0.6, 0.7, 0.5, 0.3, 0.85, 0.4, 0.65],
        }
    )

    return Dataset(
        dataframe=df,
        target_columns=["score"],
        ignore_columns=[],
        metric_columns=[],
        name="dummy",
        input_column="input",
        output_column="output",
    )


@pytest.fixture()
def small_dataset_with_references():
    df = pd.DataFrame(
        {
            "input": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "output": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "reference": ["A+", "B+", "C+", "D+", "E+", "F+", "G+", "H+", "I+", "J+"],
            "score": [0.9, 0.8, 0.2, 0.6, 0.7, 0.5, 0.3, 0.85, 0.4, 0.65],
        }
    )

    return Dataset(
        dataframe=df,
        target_columns=["score"],
        ignore_columns=[],
        metric_columns=[],
        name="dummy_with_ref",
        input_column="input",
        output_column="output",
        reference_columns=["reference"],
    )


@pytest.fixture()
def mock_training_components(monkeypatch):
    """Mock all the heavy training components to avoid actual model training during tests."""
    
    # Mock unsloth FastModel
    mock_model = Mock()
    mock_tokenizer = Mock()
    
    def mock_from_pretrained(*args, **kwargs):
        return mock_model, mock_tokenizer
    
    with patch('autometrics.generator.FinetuneGenerator.FastModel') as mock_fast_model:
        mock_fast_model.from_pretrained = mock_from_pretrained
        
        # Mock transformers components
        with patch('autometrics.generator.FinetuneGenerator.AutoModelForSequenceClassification'):
            with patch('autometrics.generator.FinetuneGenerator.TrainingArguments') as mock_training_args:
                with patch('autometrics.generator.FinetuneGenerator.Trainer') as mock_trainer_class:
                    with patch('autometrics.generator.FinetuneGenerator.Dataset') as mock_dataset:
                        with patch('autometrics.generator.FinetuneGenerator.mean_squared_error'):
                            
                            # Set up mock trainer
                            mock_trainer = Mock()
                            mock_trainer.train.return_value = Mock(training_history=[{"train_loss": 0.5}])
                            mock_trainer_class.return_value = mock_trainer
                            
                            # Set up mock dataset
                            mock_dataset_instance = Mock()
                            mock_dataset_instance.map.return_value = mock_dataset_instance
                            mock_dataset.from_dict.return_value = mock_dataset_instance
                            
                            yield {
                                'model': mock_model,
                                'tokenizer': mock_tokenizer,
                                'trainer': mock_trainer,
                                'dataset': mock_dataset_instance
                            }


def test_finetune_generator_initialization():
    """Test that FinetuneGenerator initializes correctly with default parameters."""
    generator = FinetuneGenerator()
    
    assert generator.name == "FinetuneGenerator"
    assert generator.description == "Generate fine-tuned ModernBERT metrics based on dataset regression training"
    assert generator.model_name == "answerdotai/ModernBERT-large"
    assert generator.max_seq_length == 2048
    assert generator.num_train_epochs == 3
    assert generator.batch_size == 16
    assert generator.learning_rate == 5e-5
    assert generator.upload_to_hf == False
    assert generator.hf_repo_name is None


def test_finetune_generator_initialization_custom():
    """Test that FinetuneGenerator initializes correctly with custom parameters."""
    generator = FinetuneGenerator(
        name="CustomFinetuner",
        description="Custom description",
        model_name="custom/model",
        max_seq_length=1024,
        num_train_epochs=5,
        batch_size=8,
        learning_rate=1e-5,
        upload_to_hf=True,
        hf_repo_name="my-repo"
    )
    
    assert generator.name == "CustomFinetuner"
    assert generator.description == "Custom description"
    assert generator.model_name == "custom/model"
    assert generator.max_seq_length == 1024
    assert generator.num_train_epochs == 5
    assert generator.batch_size == 8
    assert generator.learning_rate == 1e-5
    assert generator.upload_to_hf == True
    assert generator.hf_repo_name == "my-repo"


def test_determine_executor_class_ref_free(small_dataset):
    """Test that the generator correctly identifies reference-free datasets."""
    generator = FinetuneGenerator()
    executor_class = generator._determine_executor_class(small_dataset)
    assert executor_class == GeneratedRefFreeFinetunedMetric


def test_determine_executor_class_ref_based(small_dataset_with_references):
    """Test that the generator correctly identifies reference-based datasets."""
    generator = FinetuneGenerator()
    executor_class = generator._determine_executor_class(small_dataset_with_references)
    assert executor_class == GeneratedRefBasedFinetunedMetric


def test_prepare_training_data_ref_free(small_dataset):
    """Test training data preparation for reference-free datasets."""
    generator = FinetuneGenerator()
    train_texts, val_texts, train_targets, val_targets = generator._prepare_training_data(
        small_dataset, "score"
    )
    
    # Check that we have the right split (80/20)
    assert len(train_texts) == 8
    assert len(val_texts) == 2
    assert len(train_targets) == 8
    assert len(val_targets) == 2
    
    # Check text format (should be "Input: ... Output: ...")
    for text in train_texts:
        assert text.startswith("Input: ")
        assert "Output: " in text
        assert "Reference: " not in text
    
    # Check that targets are numeric
    assert all(isinstance(t, (int, float, np.number)) for t in train_targets)
    assert all(isinstance(t, (int, float, np.number)) for t in val_targets)


def test_prepare_training_data_ref_based(small_dataset_with_references):
    """Test training data preparation for reference-based datasets."""
    generator = FinetuneGenerator()
    train_texts, val_texts, train_targets, val_targets = generator._prepare_training_data(
        small_dataset_with_references, "score"
    )
    
    # Check that we have the right split (80/20)
    assert len(train_texts) == 8
    assert len(val_texts) == 2
    assert len(train_targets) == 8
    assert len(val_targets) == 2
    
    # Check text format (should include references)
    for text in train_texts:
        assert text.startswith("Input: ")
        assert "Output: " in text
        assert "Reference: " in text
    
    # Check that targets are numeric
    assert all(isinstance(t, (int, float, np.number)) for t in train_targets)
    assert all(isinstance(t, (int, float, np.number)) for t in val_targets)


def test_generate_default_n_metrics(mock_training_components, small_dataset):
    """Test that generate() defaults to n_metrics=1 for fine-tuning."""
    generator = FinetuneGenerator()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generator.model_save_dir = temp_dir
        
        # Mock the _finetune_model method to avoid actual training
        def mock_finetune_model(*args, **kwargs):
            return os.path.join(temp_dir, "mock_model")
        
        generator._finetune_model = mock_finetune_model
        
        metrics = generator.generate(small_dataset, target_measure="score")
        
        # Should default to 1 metric
        assert len(metrics) == 1
        assert isinstance(metrics[0], GeneratedRefFreeFinetunedMetric)


def test_generate_custom_n_metrics(mock_training_components, small_dataset):
    """Test that generate() respects custom n_metrics parameter."""
    generator = FinetuneGenerator()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generator.model_save_dir = temp_dir
        
        # Mock the _finetune_model method to avoid actual training
        def mock_finetune_model(*args, **kwargs):
            return os.path.join(temp_dir, "mock_model")
        
        generator._finetune_model = mock_finetune_model
        
        metrics = generator.generate(small_dataset, target_measure="score", n_metrics=2)
        
        # Should generate 2 metrics
        assert len(metrics) == 2
        assert all(isinstance(m, GeneratedRefFreeFinetunedMetric) for m in metrics)


def test_generate_ref_based_metrics(mock_training_components, small_dataset_with_references):
    """Test that generate() creates reference-based metrics for datasets with references."""
    generator = FinetuneGenerator()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generator.model_save_dir = temp_dir
        
        # Mock the _finetune_model method to avoid actual training
        def mock_finetune_model(*args, **kwargs):
            return os.path.join(temp_dir, "mock_model")
        
        generator._finetune_model = mock_finetune_model
        
        metrics = generator.generate(small_dataset_with_references, target_measure="score")
        
        assert len(metrics) == 1
        assert isinstance(metrics[0], GeneratedRefBasedFinetunedMetric)
        assert metrics[0].is_reference_based == True


def test_metric_properties(mock_training_components, small_dataset):
    """Test that generated metrics have the correct properties."""
    generator = FinetuneGenerator()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generator.model_save_dir = temp_dir
        
        # Mock the _finetune_model method to avoid actual training
        def mock_finetune_model(*args, **kwargs):
            return os.path.join(temp_dir, "mock_model")
        
        generator._finetune_model = mock_finetune_model
        
        metrics = generator.generate(small_dataset, target_measure="score")
        metric = metrics[0]
        
        # Check basic properties
        assert metric.name.endswith("_ModernBERT")
        assert "Fine-tuned ModernBERT metric" in metric.description
        assert metric.model_path == os.path.join(temp_dir, "mock_model")
        assert metric.task_description == small_dataset.get_task_description()
        assert metric.target_measure == "score"
        assert metric.dataset_name == "dummy"
        
        # Check training stats
        assert "train_size" in metric.training_stats
        assert "val_size" in metric.training_stats
        assert "target_mean" in metric.training_stats
        assert "target_std" in metric.training_stats
        assert "epochs" in metric.training_stats
        assert "learning_rate" in metric.training_stats


def test_finetune_model_mocked(mock_training_components, small_dataset):
    """Test the _finetune_model method with mocked components."""
    generator = FinetuneGenerator()
    
    # Prepare mock data
    train_texts = ["Input: a Output: A", "Input: b Output: B"]
    train_targets = np.array([0.9, 0.8])
    val_texts = ["Input: c Output: C"]
    val_targets = np.array([0.2])
    
    with tempfile.TemporaryDirectory() as temp_dir:
        model_save_path = os.path.join(temp_dir, "test_model")
        os.makedirs(model_save_path, exist_ok=True)
        
        result_path = generator._finetune_model(
            train_texts, train_targets, val_texts, val_targets, model_save_path
        )
        
        assert result_path == model_save_path


def test_model_save_directory_creation():
    """Test that the model save directory is created correctly."""
    generator = FinetuneGenerator()
    
    # Check that the directory path is set correctly
    assert "autometrics" in str(generator.model_save_dir)
    assert "models" in str(generator.model_save_dir)


def test_interface_methods():
    """Test that the generator implements the required interface methods."""
    generator = FinetuneGenerator()
    
    assert generator.get_name() == "FinetuneGenerator"
    assert generator.get_description() == "Generate fine-tuned ModernBERT metrics based on dataset regression training"
    assert str(generator) == "FinetuneGenerator: Generate fine-tuned ModernBERT metrics based on dataset regression training"
    assert repr(generator) == str(generator) 