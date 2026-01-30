#!/usr/bin/env python3
"""Test the import cost tracking functionality in UtilizationExperiment."""

import os
import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, patch

from autometrics.experiments.utilization.utilization import UtilizationExperiment
from autometrics.experiments.utilization.metric_profiler import measure_metric_phases


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class MockMetric:
    """Mock metric class for testing."""
    def __init__(self, name="TestMetric"):
        self.name = name
        self._init_params = {
            'name': name,
            'description': "Test description",
            'use_cache': False
        }
        self._excluded_params = set(['name', 'description'])
        self.model = None
        
    def get_name(self):
        return self.name
        
    def calculate(self, input_text, output_text, references):
        return 0.5


@patch('autometrics.experiments.utilization.metric_profiler.measure_metric_phases')
def test_import_cost_tracking(mock_measure_phases, temp_dir):
    """Test that import costs are properly tracked."""
    # Setup mock data
    mock_checkpoints = [
        {"label": "start", "timestamp_s": 0.0, "cpu_ram_mb": 20.0, "gpu_ram_mb": 0.0, "disk_used_mb": 1000.0},
        {"label": "after_import", "timestamp_s": 0.1, "cpu_ram_mb": 80.0, "gpu_ram_mb": 0.0, "disk_used_mb": 1000.0},
        {"label": "after_construct", "timestamp_s": 0.2, "cpu_ram_mb": 90.0, "gpu_ram_mb": 0.0, "disk_used_mb": 1000.0},
        {"label": "after_first_call", "timestamp_s": 0.3, "cpu_ram_mb": 120.0, "gpu_ram_mb": 50.0, "disk_used_mb": 1000.0}
    ]
    
    # Configure the mock to return our checkpoints
    mock_measure_phases.return_value = mock_checkpoints
    
    # Create experiment with mock metric
    metric = MockMetric()
    experiment = UtilizationExperiment(
        name="Test Experiment",
        description="Test Description",
        metrics=[metric],
        output_dir=temp_dir,
        num_examples=2,
        num_burn_in=1,
        lengths=["short"],
        use_synthetic=True,
        measure_import_costs=True
    )
    
    # Run experiment
    experiment.run(print_results=False)
    
    # Verify measure_metric_phases was called
    mock_measure_phases.assert_called_once()
    
    # Check results
    assert "TestMetric/import_costs" in experiment.results, "Import costs results should be saved"
    
    # Basic checks to make sure the experiment ran successfully
    assert "TestMetric/short/summary" in experiment.results


@patch('subprocess.check_output')
def test_measure_metric_phases(mock_subprocess, temp_dir):
    """Test the measure_metric_phases function."""
    # Setup mock subprocess output
    mock_subprocess.return_value = """[
        {"label": "start", "timestamp_s": 0.0, "cpu_ram_mb": 20.0, "gpu_ram_mb": 0.0, "disk_used_mb": 1000.0},
        {"label": "after_import", "timestamp_s": 0.1, "cpu_ram_mb": 80.0, "gpu_ram_mb": 0.0, "disk_used_mb": 1000.0}
    ]"""
    
    # Call the function
    result = measure_metric_phases(
        "autometrics.metrics.reference_based.BLEU.BLEU",
        {"use_cache": False},
        ("input", "output", ["reference"])
    )
    
    # Verify subprocess was called
    assert mock_subprocess.called
    
    # Check results
    assert len(result) == 2
    assert result[0]["label"] == "start"
    assert result[1]["label"] == "after_import"
    assert result[1]["cpu_ram_mb"] == 80.0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 