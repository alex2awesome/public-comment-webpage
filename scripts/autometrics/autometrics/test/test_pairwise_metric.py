import pytest
import pandas as pd
from autometrics.metrics.PairwiseMetric import PairwiseMetric
from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.dataset.PairwiseDataset import PairwiseDataset

# Simple mock metric for testing
class MockMetric(Metric):
    def __init__(self):
        super().__init__("mock_metric", "A mock metric that returns the length of the output")

    def _calculate_impl(self, input, output, references=None, **kwargs):
        return len(output)

    def predict(self, dataset, update_dataset=True, **kwargs):
        # Simplified implementation for testing
        df = dataset.get_dataframe()
        output_column = dataset.get_output_column()
        results = [len(out) for out in df[output_column]]
        
        if update_dataset:
            df[self.name] = results
            dataset.set_dataframe(df)
            
            if self.name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(self.name)
                
        return results

# Direct pairwise metric without wrapping another metric
class DirectPairwiseMetric(PairwiseMetric):
    def __init__(self):
        super().__init__(scalar_metric=None, 
                         name="direct_pairwise", 
                         description="A metric that directly compares two outputs")
    
    def _calculate_pairwise_impl(self, input, output_1, output_2, references=None, **kwargs):
        # Simple implementation: return difference in lengths
        return len(output_1) - len(output_2)

@pytest.fixture
def mock_metric():
    return MockMetric()

@pytest.fixture
def pairwise_metric(mock_metric):
    return PairwiseMetric(mock_metric)

@pytest.fixture
def direct_metric():
    return DirectPairwiseMetric()

@pytest.fixture
def test_data():
    data = {
        'id': [1, 2, 3],
        'model_id_1': ['model_A', 'model_A', 'model_A'],
        'model_id_2': ['model_B', 'model_B', 'model_B'],
        'input': ['Question 1', 'Question 2', 'Question 3'],
        'output1': ['Short', 'Medium length', 'Very long response indeed'],
        'output2': ['Longer response', 'Tiny', 'Medium']
    }
    return pd.DataFrame(data)

@pytest.fixture
def pairwise_dataset(test_data):
    return PairwiseDataset(
        dataframe=test_data,
        target_columns=[],
        ignore_columns=[],
        metric_columns=[],
        name="test_dataset",
        data_id_column='id',
        model_id_column_1='model_id_1',
        model_id_column_2='model_id_2',
        input_column='input',
        output_column_1='output1',
        output_column_2='output2',
        reference_columns=None
    )

def test_initialization(mock_metric, pairwise_metric, direct_metric):
    """Test that the PairwiseMetric initializes correctly"""
    assert pairwise_metric.metric == mock_metric
    assert pairwise_metric.get_name() == "pairwise_mock_metric"
    assert pairwise_metric.get_description() == "Pairwise comparison using A mock metric that returns the length of the output"
    
    # Test direct pairwise metric
    assert direct_metric.metric is None
    assert direct_metric.get_name() == "direct_pairwise"
    assert direct_metric.get_description() == "A metric that directly compares two outputs"

def test_direct_pairwise_requires_name():
    """Test that direct pairwise metrics require name and description"""
    with pytest.raises(ValueError):
        PairwiseMetric(scalar_metric=None)

def test_custom_naming(mock_metric):
    """Test custom naming"""
    custom_metric = PairwiseMetric(mock_metric, name="custom_name", description="custom description")
    assert custom_metric.get_name() == "custom_name"
    assert custom_metric.get_description() == "custom description"

def test_calculate(pairwise_metric, direct_metric):
    """Test the calculate method"""
    # Create inputs and outputs for testing
    input_text = "Test input"
    output_1 = "Short"  # len = 5
    output_2 = "Longer output"  # len = 13
    
    # Test wrapped metric
    result = pairwise_metric.calculate(input_text, output_1, output_2)
    assert result == 5 - 13
    
    # Test direct pairwise metric
    result = direct_metric.calculate(input_text, output_1, output_2)
    assert result == 5 - 13  # Same result, different implementation

def test_calculate_batched(pairwise_metric, direct_metric):
    """Test the calculate_batched method"""
    inputs = ["Input 1", "Input 2", "Input 3"]
    outputs_1 = ["Short", "Medium text", "Equal"]
    outputs_2 = ["Longer output", "Tiny", "Equal"]
    
    # Expected results
    expected = [
        len("Short") - len("Longer output"),       # 5 - 13 = -8
        len("Medium text") - len("Tiny"),          # 11 - 4 = 7
        len("Equal") - len("Equal")                # 5 - 5 = 0
    ]
    
    # Test wrapped metric
    results = pairwise_metric.calculate_batched(inputs, outputs_1, outputs_2)
    assert results == expected
    
    # Test direct pairwise metric
    results = direct_metric.calculate_batched(inputs, outputs_1, outputs_2)
    assert results == expected  # Same result, different implementation

def test_with_pairwise_dataset(pairwise_metric, direct_metric, pairwise_dataset):
    """Test integration with a PairwiseDataset"""
    # Add metrics to the dataset
    pairwise_dataset.add_metric(pairwise_metric)
    
    # Check that the metric was added and calculated
    df = pairwise_dataset.get_dataframe()
    assert "pairwise_mock_metric" in df.columns
    
    # Verify results
    expected = [
        len('Short') - len('Longer response'),               # 5 - 15 = -10
        len('Medium length') - len('Tiny'),                  # 13 - 4 = 9
        len('Very long response indeed') - len('Medium')     # 25 - 6 = 19
    ]
    
    results = df["pairwise_mock_metric"].tolist()
    assert results == expected
    
    # Test with direct pairwise metric
    pairwise_dataset.add_metric(direct_metric)
    df = pairwise_dataset.get_dataframe()
    assert "direct_pairwise" in df.columns
    
    # Should have the same results as they implement the same logic
    results = df["direct_pairwise"].tolist()
    assert results == expected

def test_auto_wrapping_metric(mock_metric):
    """Test that regular metrics get automatically wrapped as PairwiseMetric"""
    # Create test data
    data = {
        'id': [1, 2, 3],
        'model_id_1': ['model_A', 'model_A', 'model_A'],
        'model_id_2': ['model_B', 'model_B', 'model_B'],
        'input': ['Question 1', 'Question 2', 'Question 3'],
        'output1': ['Short', 'Medium length', 'Very long response indeed'],
        'output2': ['Longer response', 'Tiny', 'Medium']
    }
    df = pd.DataFrame(data)
    
    # Create a new dataset
    pairwise_dataset = PairwiseDataset(
        dataframe=df.copy(),
        target_columns=[],
        ignore_columns=[],
        metric_columns=[],
        name="test_dataset",
        data_id_column='id',
        model_id_column_1='model_id_1',
        model_id_column_2='model_id_2',
        input_column='input',
        output_column_1='output1',
        output_column_2='output2'
    )
    
    # Add a regular Metric (not PairwiseMetric)
    pairwise_dataset.add_metric(mock_metric)
    
    # It should be wrapped as PairwiseMetric
    df = pairwise_dataset.get_dataframe()
    assert "pairwise_mock_metric" in df.columns
    
    # Results should match our expected differences
    expected = [
        len('Short') - len('Longer response'),
        len('Medium length') - len('Tiny'),
        len('Very long response indeed') - len('Medium')
    ]
    
    results = df["pairwise_mock_metric"].tolist()
    assert results == expected

def test_dataset_getters(pairwise_dataset):
    """Test the PairwiseDataset getters"""
    assert pairwise_dataset.get_input_column() == 'input'
    assert pairwise_dataset.get_output_column_1() == 'output1'
    assert pairwise_dataset.get_output_column_2() == 'output2'
    assert pairwise_dataset.get_model_id_column_1() == 'model_id_1'
    assert pairwise_dataset.get_model_id_column_2() == 'model_id_2'

def test_dataset_splitting(pairwise_dataset):
    """Test that the dataset can be properly split"""
    train, val, test = pairwise_dataset.get_splits(train_ratio=0.6, val_ratio=0.2, seed=42)
    
    # Verify that all splits are of correct type
    assert isinstance(train, PairwiseDataset)
    assert isinstance(val, PairwiseDataset)
    assert isinstance(test, PairwiseDataset)
    
    # Verify all splits have correct structure
    assert train.get_input_column() == 'input'
    assert train.get_output_column_1() == 'output1'
    assert train.get_output_column_2() == 'output2'
    
    # Verify all original rows are preserved across splits
    total_rows = len(train.get_dataframe()) + len(val.get_dataframe()) + len(test.get_dataframe())
    assert total_rows == len(pairwise_dataset.get_dataframe()) 