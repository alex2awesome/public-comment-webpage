#!/usr/bin/env python3
"""Unit tests for the UtilizationExperiment class."""

import os
import pytest
import tempfile
import shutil
import pandas as pd
from unittest.mock import MagicMock, patch

from autometrics.experiments.utilization.utilization import (
    ResourceTracker, 
    generate_synthetic_text, 
    UtilizationExperiment
)

from autometrics.experiments.results import TabularResult

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_metrics():
    """Return mock metrics for testing."""
    
    class MockMetric:
        def __init__(self, name, use_cache=False):
            self.name = name
            self.use_cache = use_cache
            
        def get_name(self):
            return self.name
        
        def calculate(self, input_text, output_text, references):
            # Simulate work
            if len(input_text) > 1000:
                # More work for longer texts
                _ = ["x" * 100000 for _ in range(10)]
            return 0.5
    
    return [MockMetric("TestMetric1"), MockMetric("TestMetric2")]

@pytest.mark.parametrize("length_category", ["short", "medium", "long"])
def test_synthetic_text_generation(length_category):
    """Test generating synthetic text samples."""
    input_text, output_text, references = generate_synthetic_text(length_category)
    
    if length_category == "short":
        assert len(input_text.split()) <= 10
        assert len(output_text.split()) <= 10
        for ref in references:
            assert len(ref.split()) <= 10
    elif length_category == "medium":
        assert 80 <= len(input_text.split()) <= 120
        assert 80 <= len(output_text.split()) <= 120
        for ref in references:
            assert 80 <= len(ref.split()) <= 120
    else:  # long
        assert 800 <= len(input_text.split()) <= 1200
        assert 800 <= len(output_text.split()) <= 1200
        for ref in references:
            assert 800 <= len(ref.split()) <= 1200
    
    # Check reference generation
    assert len(references) > 0
    for ref in references:
        assert isinstance(ref, str)

def test_resource_tracker():
    """Test basic usage of the ResourceTracker."""
    tracker = ResourceTracker()
    tracker.start()
    
    # Do some work to generate memory usage
    _ = ["x" * 1000000 for _ in range(10)]
    
    tracker.stop()
    results = tracker.get_results()
    
    assert 'duration_milliseconds' in results
    assert 'cpu_ram_mb' in results
    assert 'gpu_ram_mb' in results
    assert 'disk_usage_change_mb' in results
    
    assert results['duration_milliseconds'] > 0
    assert results['cpu_ram_mb'] > 0

def test_experiment_creation(temp_dir, mock_metrics):
    """Test creating an experiment."""
    experiment = UtilizationExperiment(
        name="Test Experiment",
        description="Test Description",
        metrics=mock_metrics,
        output_dir=temp_dir,
        num_examples=2,
        num_burn_in=1,
        lengths=["short"]
    )
    
    assert experiment.name == "Test Experiment"
    assert experiment.description == "Test Description"
    assert experiment.metrics == mock_metrics
    assert experiment.output_dir == os.path.join(temp_dir, "synthetic")
    assert experiment.num_examples == 2
    assert experiment.num_burn_in == 1
    assert experiment.lengths == ["short"]

@patch('autometrics.experiments.utilization.utilization.track_resources')
def test_experiment_run_synthetic(mock_track_resources, temp_dir, mock_metrics):
    """Test running an experiment with synthetic data."""
    # Set up mock for resource tracker
    mock_tracker = MagicMock()
    mock_tracker.get_results.return_value = {
        'duration_milliseconds': 100.0,
        'cpu_ram_mb': 100.0,
        'gpu_ram_mb': 0.0,
        'disk_usage_change_mb': 0.0,
        'baseline_cpu_ram_mb': 500.0,
        'baseline_gpu_ram_mb': 0.0,
        'total_cpu_ram_mb': 600.0,
        'total_gpu_ram_mb': 0.0
    }
    mock_track_resources.return_value.__enter__.return_value = mock_tracker
    
    experiment = UtilizationExperiment(
        name="Test Experiment",
        description="Test Description",
        metrics=mock_metrics,
        output_dir=temp_dir,
        num_examples=2,
        num_burn_in=1,
        lengths=["short"],
        use_synthetic=True
    )
    
    experiment.run(print_results=False)
    
    # Verify resources are measured
    assert mock_tracker.get_results.called
    
    # Check that results are created for the right paths
    metric_name = mock_metrics[0].get_name()
    assert f"{metric_name}/short/raw_data" in experiment.results
    assert f"{metric_name}/short/summary" in experiment.results

@patch('autometrics.experiments.utilization.utilization.track_resources')
def test_experiment_with_real_data(mock_track_resources, temp_dir, mock_metrics):
    """Test running an experiment with real dataset."""
    # Mock dataset
    mock_dataset = MagicMock()
    mock_dataset.get_dataframe.return_value = pd.DataFrame({
        'input': ['short text', 'medium ' * 15, 'long ' * 150],
        'output': ['short output', 'medium output ' * 10, 'long output ' * 120],
        'reference': ['short ref', 'medium ref ' * 12, 'long ref ' * 130]
    })
    mock_dataset.get_input_column.return_value = 'input'
    mock_dataset.get_output_column.return_value = 'output'
    mock_dataset.get_reference_columns.return_value = ['reference']
    
    # Set up mock for resource tracker
    mock_tracker = MagicMock()
    mock_tracker.get_results.return_value = {
        'duration_milliseconds': 100.0,
        'cpu_ram_mb': 100.0,
        'gpu_ram_mb': 0.0,
        'disk_usage_change_mb': 0.0,
        'baseline_cpu_ram_mb': 500.0,
        'baseline_gpu_ram_mb': 0.0,
        'total_cpu_ram_mb': 600.0,
        'total_gpu_ram_mb': 0.0
    }
    mock_track_resources.return_value.__enter__.return_value = mock_tracker
    
    experiment = UtilizationExperiment(
        name="Test Real Data Experiment",
        description="Test with real data",
        metrics=mock_metrics,
        output_dir=temp_dir,
        dataset=mock_dataset,
        num_examples=2,
        num_burn_in=1,
        lengths=["short", "medium", "long"],
        use_synthetic=False
    )
    
    experiment.run(print_results=False)
    
    # Check that results are created without length categorization
    metric_name = mock_metrics[0].get_name()
    assert f"{metric_name}/raw_data" in experiment.results
    assert f"{metric_name}/summary" in experiment.results
    
    # Verify model wasn't filtering the dataset by length categories
    test_examples = experiment._get_test_examples("dataset")
    assert len(test_examples) == 3  # Should have all examples, not filtered

@patch('autometrics.experiments.results.TabularResult.save')
def test_save_results(mock_save, temp_dir, mock_metrics):
    """Test saving experiment results."""
    experiment = UtilizationExperiment(
        name="Test Experiment",
        description="Test Description",
        metrics=mock_metrics,
        output_dir=temp_dir,
        num_examples=2,
        num_burn_in=1,
        lengths=["short"]
    )
    
    # Create a simple result
    experiment.results["test/path"] = TabularResult(pd.DataFrame({"test": [1, 2, 3]}))
    
    experiment.save_results()
    mock_save.assert_called() 