import pytest
import pandas as pd
import numpy as np
import warnings
import os
import tempfile
import shutil
from autometrics.dataset.Dataset import Dataset
from autometrics.dataset.PairwiseDataset import PairwiseDataset

@pytest.fixture
def sample_data():
    data = {
        'id': [1, 2, 3, 4, 5],
        'input': ['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5'],
        'output': ['Output 1', 'Output 2', 'Output 3', 'Output 4', 'Output 5']
    }
    return pd.DataFrame(data)

@pytest.fixture
def basic_dataset(sample_data):
    return Dataset(
        dataframe=sample_data,
        target_columns=[],
        ignore_columns=[],
        metric_columns=[],
        name="test_dataset",
        data_id_column='id',
        input_column='input',
        output_column='output',
        reference_columns=None
    )

@pytest.fixture
def pairwise_data():
    data = {
        'id': [1, 2, 3, 4, 5],
        'model_id_1': ['A'] * 5,
        'model_id_2': ['B'] * 5,
        'input': ['Input 1', 'Input 2', 'Input 3', 'Input 4', 'Input 5'],
        'output1': ['Output A1', 'Output A2', 'Output A3', 'Output A4', 'Output A5'],
        'output2': ['Output B1', 'Output B2', 'Output B3', 'Output B4', 'Output B5']
    }
    return pd.DataFrame(data)

@pytest.fixture
def pairwise_dataset(pairwise_data):
    return PairwiseDataset(
        dataframe=pairwise_data,
        target_columns=[],
        ignore_columns=[],
        metric_columns=[],
        name="test_pairwise_dataset",
        data_id_column='id',
        model_id_column_1='model_id_1',
        model_id_column_2='model_id_2',
        input_column='input',
        output_column_1='output1',
        output_column_2='output2',
        reference_columns=None
    )

@pytest.fixture
def larger_sample_data():
    """Create a larger dataset for more comprehensive split testing"""
    data = {
        'id': list(range(1, 21)),  # 20 items
        'input': [f'Input {i}' for i in range(1, 21)],
        'output': [f'Output {i}' for i in range(1, 21)],
        'category': ['A'] * 10 + ['B'] * 10
    }
    return pd.DataFrame(data)

@pytest.fixture
def larger_dataset(larger_sample_data):
    return Dataset(
        dataframe=larger_sample_data,
        target_columns=[],
        ignore_columns=[],
        metric_columns=[],
        name="test_larger_dataset",
        data_id_column='id',
        input_column='input',
        output_column='output',
        reference_columns=None
    )

def test_get_subset_normal(basic_dataset):
    """Test get_subset with a size smaller than the available data"""
    subset = basic_dataset.get_subset(3, seed=42)
    
    # Verify subset size
    assert len(subset.get_dataframe()) == 3
    
    # Verify subset maintains properties of original dataset
    assert subset.get_input_column() == 'input'
    assert subset.get_output_column() == 'output'

def test_get_subset_exact_size(basic_dataset):
    """Test get_subset with a size equal to the available data"""
    subset = basic_dataset.get_subset(5, seed=42)
    
    # Should get all rows
    assert len(subset.get_dataframe()) == 5
    
    # Should still have all the original data (but possibly in different order)
    assert set(subset.get_dataframe()['id']) == set(range(1, 6))

def test_get_subset_oversized(basic_dataset):
    """Test get_subset with a size larger than the available data"""
    # This should issue a warning but not fail
    with pytest.warns() as recorded_warnings:
        subset = basic_dataset.get_subset(10, seed=42)
        
        # Verify a warning was issued
        assert any("Requested subset size 10 is larger than available data" in str(warning.message) for warning in recorded_warnings)
    
    # Should get all available rows
    assert len(subset.get_dataframe()) == 5
    
    # Should have all the original data
    assert set(subset.get_dataframe()['id']) == set(range(1, 6))

def test_splits_with_max_size(basic_dataset):
    """Test that get_splits works with max_size larger than available data"""
    # This should not fail even though max_size is larger than available data
    train, val, test = basic_dataset.get_splits(train_ratio=0.6, val_ratio=0.2, seed=42, max_size=100)
    
    # Verify splits were created
    assert isinstance(train, Dataset)
    assert isinstance(val, Dataset)
    assert isinstance(test, Dataset)
    
    # Check that the splits together contain all the original data
    total_rows = len(train.get_dataframe()) + len(val.get_dataframe()) + len(test.get_dataframe())
    assert total_rows == 5

def test_pairwise_splits_with_max_size(pairwise_dataset):
    """Test that PairwiseDataset.get_splits works with max_size larger than available data"""
    # This should not fail even though max_size is larger than available data
    train, val, test = pairwise_dataset.get_splits(train_ratio=0.6, val_ratio=0.2, seed=42, max_size=100)
    
    # Verify splits are PairwiseDataset instances
    assert isinstance(train, PairwiseDataset)
    assert isinstance(val, PairwiseDataset)
    assert isinstance(test, PairwiseDataset)
    
    # Check that the splits together contain all the original data
    total_rows = len(train.get_dataframe()) + len(val.get_dataframe()) + len(test.get_dataframe())
    assert total_rows == 5
    
    # Verify the pairwise-specific properties are preserved
    assert train.get_output_column_1() == 'output1'
    assert train.get_output_column_2() == 'output2'

def test_get_splits_consistent_with_kfold(larger_dataset):
    """Test that get_splits and get_kfold_splits produce identical test sets"""
    # Create splits using both methods with same parameters
    train, val, test = larger_dataset.get_splits(
        split_column='id',
        train_ratio=0.5,
        val_ratio=0.2,
        seed=42
    )
    
    # test_ratio should be 1.0 - 0.5 - 0.2 = 0.3
    kfold_splits, kfold_train, kfold_test = larger_dataset.get_kfold_splits(
        k=5,
        split_column='id',
        seed=42,
        test_ratio=0.3
    )
    
    # Check that test sets are identical
    assert test is not None
    assert kfold_test is not None
    
    # Sort both test sets and compare
    test_ids = sorted(test.get_dataframe()['id'].tolist())
    kfold_test_ids = sorted(kfold_test.get_dataframe()['id'].tolist())
    
    assert test_ids == kfold_test_ids, "Test sets from get_splits and get_kfold_splits should be identical"

def test_no_overlaps_in_splits(larger_dataset):
    """Test that there are no overlaps between train/val/test splits"""
    train, val, test = larger_dataset.get_splits(
        split_column='id',
        train_ratio=0.5,
        val_ratio=0.2,
        seed=42
    )
    
    # Get the IDs from each split
    train_ids = set(train.get_dataframe()['id'])
    val_ids = set(val.get_dataframe()['id'])
    test_ids = set(test.get_dataframe()['id'])
    
    # Check no overlaps
    assert len(train_ids & val_ids) == 0, "Train and validation sets should not overlap"
    assert len(train_ids & test_ids) == 0, "Train and test sets should not overlap"
    assert len(val_ids & test_ids) == 0, "Validation and test sets should not overlap"
    
    # Check that all original data is present
    all_split_ids = train_ids | val_ids | test_ids
    original_ids = set(larger_dataset.get_dataframe()['id'])
    assert all_split_ids == original_ids, "All original data should be present in splits"

def test_no_overlaps_in_kfold_splits(larger_dataset):
    """Test that there are no overlaps in k-fold splits"""
    kfold_splits, kfold_train, kfold_test = larger_dataset.get_kfold_splits(
        k=5,
        split_column='id',
        seed=42,
        test_ratio=0.3
    )
    
    # Test that test set doesn't overlap with any fold
    test_ids = set(kfold_test.get_dataframe()['id']) if kfold_test else set()
    
    for i, (fold_train, fold_val) in enumerate(kfold_splits):
        fold_train_ids = set(fold_train.get_dataframe()['id'])
        fold_val_ids = set(fold_val.get_dataframe()['id'])
        
        # Check no overlap with test set
        assert len(fold_train_ids & test_ids) == 0, f"Fold {i} train should not overlap with test"
        assert len(fold_val_ids & test_ids) == 0, f"Fold {i} val should not overlap with test"
        
        # Check no overlap within fold
        assert len(fold_train_ids & fold_val_ids) == 0, f"Fold {i} train and val should not overlap"
    
    # Check that validation sets don't overlap across folds
    val_sets = [set(fold_val.get_dataframe()['id']) for _, fold_val in kfold_splits]
    for i in range(len(val_sets)):
        for j in range(i + 1, len(val_sets)):
            assert len(val_sets[i] & val_sets[j]) == 0, f"Validation sets of fold {i} and {j} should not overlap"

def test_permanent_splits_save_load(larger_dataset):
    """Test saving and loading permanent splits"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override the dataset name to use our temp directory
        test_dataset = Dataset(
            dataframe=larger_dataset.get_dataframe(),
            target_columns=larger_dataset.get_target_columns(),
            ignore_columns=larger_dataset.get_ignore_columns(),
            metric_columns=larger_dataset.get_metric_columns(),
            name="temp_test_dataset",
            data_id_column=larger_dataset.get_data_id_column(),
            input_column=larger_dataset.get_input_column(),
            output_column=larger_dataset.get_output_column(),
            reference_columns=larger_dataset.get_reference_columns()
        )
        
        # Mock the user_data_dir to use our temp directory
        import autometrics.dataset.Dataset as dataset_module
        original_user_data_dir = dataset_module.user_data_dir
        dataset_module.user_data_dir = lambda: temp_dir
        
        try:
            # Save permanent splits
            train, val, test = test_dataset.save_permanent_splits(
                split_column='id',
                train_ratio=0.5,
                val_ratio=0.2,
                seed=42
            )
            
            # Load permanent splits
            loaded_train, loaded_val, loaded_test = test_dataset.load_permanent_splits()
            
            # Verify loaded splits match original splits
            assert len(loaded_train.get_dataframe()) == len(train.get_dataframe())
            assert len(loaded_val.get_dataframe()) == len(val.get_dataframe())
            assert len(loaded_test.get_dataframe()) == len(test.get_dataframe())
            
            # Verify data integrity
            assert set(loaded_train.get_dataframe()['id']) == set(train.get_dataframe()['id'])
            assert set(loaded_val.get_dataframe()['id']) == set(val.get_dataframe()['id'])
            assert set(loaded_test.get_dataframe()['id']) == set(test.get_dataframe()['id'])
            
        finally:
            # Restore original user_data_dir
            dataset_module.user_data_dir = original_user_data_dir

def test_permanent_kfold_splits_save_load(larger_dataset):
    """Test saving and loading permanent k-fold splits"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override the dataset name to use our temp directory
        test_dataset = Dataset(
            dataframe=larger_dataset.get_dataframe(),
            target_columns=larger_dataset.get_target_columns(),
            ignore_columns=larger_dataset.get_ignore_columns(),
            metric_columns=larger_dataset.get_metric_columns(),
            name="temp_kfold_dataset",
            data_id_column=larger_dataset.get_data_id_column(),
            input_column=larger_dataset.get_input_column(),
            output_column=larger_dataset.get_output_column(),
            reference_columns=larger_dataset.get_reference_columns()
        )
        
        # Mock the user_data_dir to use our temp directory
        import autometrics.dataset.Dataset as dataset_module
        original_user_data_dir = dataset_module.user_data_dir
        dataset_module.user_data_dir = lambda: temp_dir
        
        try:
            # Save permanent k-fold splits
            splits, train, test = test_dataset.save_permanent_kfold_splits(
                k=3,
                split_column='id',
                seed=42,
                test_ratio=0.3
            )
            
            # Load permanent k-fold splits
            loaded_splits, loaded_train, loaded_test = test_dataset.load_permanent_kfold_splits()
            
            # Verify loaded splits match original splits
            assert len(loaded_splits) == len(splits)
            assert len(loaded_train.get_dataframe()) == len(train.get_dataframe())
            if test is not None:
                assert len(loaded_test.get_dataframe()) == len(test.get_dataframe())
            
            # Verify k-fold data integrity
            for i, ((orig_train, orig_val), (loaded_train_fold, loaded_val_fold)) in enumerate(zip(splits, loaded_splits)):
                assert len(orig_train.get_dataframe()) == len(loaded_train_fold.get_dataframe())
                assert len(orig_val.get_dataframe()) == len(loaded_val_fold.get_dataframe())
                assert set(orig_train.get_dataframe()['id']) == set(loaded_train_fold.get_dataframe()['id'])
                assert set(orig_val.get_dataframe()['id']) == set(loaded_val_fold.get_dataframe()['id'])
            
        finally:
            # Restore original user_data_dir
            dataset_module.user_data_dir = original_user_data_dir

def test_validate_permanent_splits_consistency(larger_dataset):
    """Test that permanent splits validation works correctly"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override the dataset name to use our temp directory
        test_dataset = Dataset(
            dataframe=larger_dataset.get_dataframe(),
            target_columns=larger_dataset.get_target_columns(),
            ignore_columns=larger_dataset.get_ignore_columns(),
            metric_columns=larger_dataset.get_metric_columns(),
            name="temp_validation_dataset",
            data_id_column=larger_dataset.get_data_id_column(),
            input_column=larger_dataset.get_input_column(),
            output_column=larger_dataset.get_output_column(),
            reference_columns=larger_dataset.get_reference_columns()
        )
        
        # Mock the user_data_dir to use our temp directory
        import autometrics.dataset.Dataset as dataset_module
        original_user_data_dir = dataset_module.user_data_dir
        dataset_module.user_data_dir = lambda: temp_dir
        
        try:
            # Create and validate permanent splits
            success, results = test_dataset.create_and_validate_permanent_splits(
                split_column='id',
                train_ratio=0.5,
                val_ratio=0.2,
                k=3,
                seed=42
            )
            
            # Verify the operation succeeded
            assert success == True
            assert results['splits_created'] == True
            assert results['kfold_created'] == True
            assert results['test_sets_match'] == True
            assert len(results['errors']) == 0
            
            # Verify the validation method works
            assert test_dataset.validate_permanent_splits_consistency() == True
            
        finally:
            # Restore original user_data_dir
            dataset_module.user_data_dir = original_user_data_dir

def test_pairwise_dataset_save_load_file(pairwise_dataset):
    """Test PairwiseDataset save_to_file and load_from_file methods"""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_pairwise.csv")
        
        # Save the dataset
        pairwise_dataset.save_to_file(file_path)
        
        # Verify files were created
        assert os.path.exists(file_path)
        assert os.path.exists(file_path.replace('.csv', '_metadata.json'))
        
        # Load the dataset
        loaded_dataset = pairwise_dataset.load_from_file(file_path)
        
        # Verify it's a PairwiseDataset
        assert isinstance(loaded_dataset, PairwiseDataset)
        
        # Verify data integrity
        assert len(loaded_dataset.get_dataframe()) == len(pairwise_dataset.get_dataframe())
        assert loaded_dataset.get_output_column_1() == pairwise_dataset.get_output_column_1()
        assert loaded_dataset.get_output_column_2() == pairwise_dataset.get_output_column_2()
        assert loaded_dataset.get_model_id_column_1() == pairwise_dataset.get_model_id_column_1()
        assert loaded_dataset.get_model_id_column_2() == pairwise_dataset.get_model_id_column_2()
        
        # Verify data content
        original_data = pairwise_dataset.get_dataframe().sort_values('id').reset_index(drop=True)
        loaded_data = loaded_dataset.get_dataframe().sort_values('id').reset_index(drop=True)
        pd.testing.assert_frame_equal(original_data, loaded_data)

def test_pairwise_dataset_splits_consistency(pairwise_dataset):
    """Test that PairwiseDataset splits maintain pairwise properties"""
    # Test get_splits
    train, val, test = pairwise_dataset.get_splits(
        split_column='id',
        train_ratio=0.6,
        val_ratio=0.2,
        seed=42
    )
    
    # Verify all splits are PairwiseDataset instances
    assert isinstance(train, PairwiseDataset)
    assert isinstance(val, PairwiseDataset)
    assert isinstance(test, PairwiseDataset)
    
    # Verify pairwise properties are preserved
    for split in [train, val, test]:
        assert split.get_output_column_1() == 'output1'
        assert split.get_output_column_2() == 'output2'
        assert split.get_model_id_column_1() == 'model_id_1'
        assert split.get_model_id_column_2() == 'model_id_2'
    
    # Test get_kfold_splits
    kfold_splits, kfold_train, kfold_test = pairwise_dataset.get_kfold_splits(
        k=3,
        split_column='id',
        seed=42,
        test_ratio=0.2
    )
    
    # Verify k-fold splits are PairwiseDataset instances
    assert isinstance(kfold_train, PairwiseDataset)
    assert isinstance(kfold_test, PairwiseDataset)
    
    for fold_train, fold_val in kfold_splits:
        assert isinstance(fold_train, PairwiseDataset)
        assert isinstance(fold_val, PairwiseDataset)
        assert fold_train.get_output_column_1() == 'output1'
        assert fold_val.get_output_column_2() == 'output2'

def test_task_description_preservation():
    """Test that task_description is preserved across operations"""
    data = pd.DataFrame({
        'id': [1, 2, 3],
        'input': ['a', 'b', 'c'],
        'output': ['x', 'y', 'z']
    })
    
    dataset = Dataset(
        dataframe=data,
        target_columns=[],
        ignore_columns=[],
        metric_columns=[],
        name="test_task_desc",
        data_id_column='id',
        input_column='input',
        output_column='output',
        task_description="Test task description"
    )
    
    # Test that task description is preserved in splits
    train, val, test = dataset.get_splits(train_ratio=0.5, val_ratio=0.2, seed=42)
    
    assert train.get_task_description() == "Test task description"
    assert val.get_task_description() == "Test task description"
    assert test.get_task_description() == "Test task description"
    
    # Test that task description is preserved in k-fold splits
    kfold_splits, kfold_train, kfold_test = dataset.get_kfold_splits(k=2, seed=42, test_ratio=0.3)
    
    assert kfold_train.get_task_description() == "Test task description"
    if kfold_test:
        assert kfold_test.get_task_description() == "Test task description"
    
    for fold_train, fold_val in kfold_splits:
        assert fold_train.get_task_description() == "Test task description"
        assert fold_val.get_task_description() == "Test task description" 