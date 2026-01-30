#!/usr/bin/env python3
"""
Script to create permanent splits for all datasets and validate no overlaps.

This script:
1. Loads all available datasets from autometrics
2. Creates permanent splits (train/val/test) and permanent k-fold splits
3. Validates that there are no overlaps between train/val and test sets
4. Reports results and any issues found
"""

import sys
import os
import traceback
from typing import List, Tuple, Any, Dict, Set, Optional
import pandas as pd

# Add the autometrics package to the path
sys.path.append('/nlp/scr2/nlp/personal-rm/autometrics')

from autometrics.dataset.Dataset import Dataset
from autometrics.dataset.PairwiseDataset import PairwiseDataset

# Import all available datasets
from autometrics.dataset.datasets.primock57.primock57 import Primock57
from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer, HelpSteer2
from autometrics.dataset.datasets.evalgen.evalgen import EvalGen, EvalGenProduct, EvalGenMedical
from autometrics.dataset.datasets.design2code.design2code import Design2Code
from autometrics.dataset.datasets.realhumaneval.realhumaneval import RealHumanEval
from autometrics.dataset.datasets.simplification.simplification import SimpDA, SimpEval
from autometrics.dataset.datasets.summeval.summeval import SummEval
from autometrics.dataset.datasets.cogym.cogym import (
    CoGym, CoGymTabularOutcome, CoGymTabularProcess, 
    CoGymTravelOutcome, CoGymTravelProcess, CoGymLessonOutcome, CoGymLessonProcess
)
from autometrics.dataset.datasets.airesearcher.ai_researcher import AI_Researcher
from autometrics.dataset.datasets.iclr.iclr import ICLR
from autometrics.dataset.datasets.taubench.taubench import TauBench, TauBenchBigger, TauBenchHighTemperature

def check_test_sets_match(dataset_name: str) -> Tuple[bool, str]:
    """
    Check if test.csv and test_kfold.csv contain the same data.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (match: bool, error_message: str)
    """
    from platformdirs import user_data_dir
    
    try:
        test_path = f"{user_data_dir()}/autometrics/datasets/{dataset_name}/test.csv"
        test_kfold_path = f"{user_data_dir()}/autometrics/datasets/{dataset_name}/test_kfold.csv"
        
        # Check if both files exist
        if not os.path.exists(test_path):
            return False, f"test.csv does not exist at {test_path}"
        if not os.path.exists(test_kfold_path):
            return False, f"test_kfold.csv does not exist at {test_kfold_path}"
        
        # Load both test datasets
        test_df = pd.read_csv(test_path)
        test_kfold_df = pd.read_csv(test_kfold_path)
        
        # Check if they have the same shape
        if test_df.shape != test_kfold_df.shape:
            return False, f"Shape mismatch: test.csv {test_df.shape} vs test_kfold.csv {test_kfold_df.shape}"
        
        # Check if they have the same columns
        if list(test_df.columns) != list(test_kfold_df.columns):
            return False, f"Column mismatch: test.csv has {list(test_df.columns)} vs test_kfold.csv has {list(test_kfold_df.columns)}"
        
        # Sort both dataframes by all columns to ensure consistent ordering
        test_df_sorted = test_df.sort_values(by=list(test_df.columns)).reset_index(drop=True)
        test_kfold_df_sorted = test_kfold_df.sort_values(by=list(test_kfold_df.columns)).reset_index(drop=True)
        
        # Check if the sorted dataframes are identical
        if not test_df_sorted.equals(test_kfold_df_sorted):
            return False, "DataFrames contain different data after sorting"
        
        return True, "Test sets match perfectly"
        
    except Exception as e:
        return False, f"Error comparing test sets: {str(e)}"


def check_splits_overlap(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, 
                        split_column: str) -> Tuple[bool, List[str]]:
    """
    Check if there are any overlaps between train/val/test splits based on the split column.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset
        split_column: Column to check for overlaps
        
    Returns:
        Tuple of (has_overlap: bool, overlap_messages: List[str])
    """
    errors = []
    has_overlap = False
    
    if test_dataset is None:
        # No test dataset, only check train vs val
        if split_column and split_column in train_dataset.get_dataframe().columns:
            train_ids = set(train_dataset.get_dataframe()[split_column].unique())
            val_ids = set(val_dataset.get_dataframe()[split_column].unique())
            
            overlap = train_ids & val_ids
            if overlap:
                has_overlap = True
                errors.append(f"Train/Val overlap in {split_column}: {len(overlap)} items")
        else:
            # Check by row indices if no split column
            train_indices = set(train_dataset.get_dataframe().index)
            val_indices = set(val_dataset.get_dataframe().index)
            
            overlap = train_indices & val_indices
            if overlap:
                has_overlap = True
                errors.append(f"Train/Val overlap by index: {len(overlap)} rows")
    else:
        # Check all pairwise overlaps
        if split_column and split_column in train_dataset.get_dataframe().columns:
            train_ids = set(train_dataset.get_dataframe()[split_column].unique())
            val_ids = set(val_dataset.get_dataframe()[split_column].unique())
            test_ids = set(test_dataset.get_dataframe()[split_column].unique())
            
            train_val = train_ids & val_ids
            train_test = train_ids & test_ids  
            val_test = val_ids & test_ids
            
            if train_val:
                has_overlap = True
                errors.append(f"Train/Val overlap in {split_column}: {len(train_val)} items")
            if train_test:
                has_overlap = True
                errors.append(f"Train/Test overlap in {split_column}: {len(train_test)} items")
            if val_test:
                has_overlap = True
                errors.append(f"Val/Test overlap in {split_column}: {len(val_test)} items")
        else:
            # Check by row indices if no split column
            train_indices = set(train_dataset.get_dataframe().index)
            val_indices = set(val_dataset.get_dataframe().index)
            test_indices = set(test_dataset.get_dataframe().index)
            
            train_val = train_indices & val_indices
            train_test = train_indices & test_indices
            val_test = val_indices & test_indices
            
            if train_val:
                has_overlap = True
                errors.append(f"Train/Val overlap by index: {len(train_val)} rows")
            if train_test:
                has_overlap = True
                errors.append(f"Train/Test overlap by index: {len(train_test)} rows")
            if val_test:
                has_overlap = True
                errors.append(f"Val/Test overlap by index: {len(val_test)} rows")
    
    return has_overlap, errors


def check_kfold_splits_overlap(split_datasets: List[Tuple[Dataset, Dataset]], 
                              train_dataset: Dataset, test_dataset: Dataset,
                              split_column: str) -> Tuple[bool, List[str]]:
    """
    Check if there are any overlaps in k-fold splits.
    
    Args:
        split_datasets: List of (train_fold, val_fold) pairs
        train_dataset: Full training dataset
        test_dataset: Test dataset (can be None)
        split_column: Column to check for overlaps
        
    Returns:
        Tuple of (has_overlap: bool, overlap_messages: List[str])
    """
    errors = []
    has_overlap = False
    
    # Check each fold doesn't overlap with test set
    if test_dataset is not None:
        if split_column and split_column in test_dataset.get_dataframe().columns:
            test_ids = set(test_dataset.get_dataframe()[split_column].unique())
            
            for i, (fold_train, fold_val) in enumerate(split_datasets):
                fold_train_ids = set(fold_train.get_dataframe()[split_column].unique())
                fold_val_ids = set(fold_val.get_dataframe()[split_column].unique())
                
                train_test_overlap = fold_train_ids & test_ids
                val_test_overlap = fold_val_ids & test_ids
                
                if train_test_overlap:
                    has_overlap = True
                    errors.append(f"Fold {i} train/test overlap in {split_column}: {len(train_test_overlap)} items")
                if val_test_overlap:
                    has_overlap = True
                    errors.append(f"Fold {i} val/test overlap in {split_column}: {len(val_test_overlap)} items")
        else:
            # Check by indices
            test_indices = set(test_dataset.get_dataframe().index)
            
            for i, (fold_train, fold_val) in enumerate(split_datasets):
                fold_train_indices = set(fold_train.get_dataframe().index)
                fold_val_indices = set(fold_val.get_dataframe().index)
                
                train_test_overlap = fold_train_indices & test_indices
                val_test_overlap = fold_val_indices & test_indices
                
                if train_test_overlap:
                    has_overlap = True
                    errors.append(f"Fold {i} train/test overlap by index: {len(train_test_overlap)} rows")
                if val_test_overlap:
                    has_overlap = True
                    errors.append(f"Fold {i} val/test overlap by index: {len(val_test_overlap)} rows")
    
    # Check that folds don't overlap with each other
    if split_column and len(split_datasets) > 0 and split_column in split_datasets[0][0].get_dataframe().columns:
        all_fold_train_ids = []
        all_fold_val_ids = []
        
        for fold_train, fold_val in split_datasets:
            all_fold_train_ids.append(set(fold_train.get_dataframe()[split_column].unique()))
            all_fold_val_ids.append(set(fold_val.get_dataframe()[split_column].unique()))
        
        # Check train/val within each fold
        for i, (train_ids, val_ids) in enumerate(zip(all_fold_train_ids, all_fold_val_ids)):
            overlap = train_ids & val_ids
            if overlap:
                has_overlap = True
                errors.append(f"Fold {i} train/val overlap in {split_column}: {len(overlap)} items")
        
        # Check that validation sets don't overlap across folds
        for i in range(len(all_fold_val_ids)):
            for j in range(i + 1, len(all_fold_val_ids)):
                overlap = all_fold_val_ids[i] & all_fold_val_ids[j]
                if overlap:
                    has_overlap = True
                    errors.append(f"Fold {i}/Fold {j} val overlap in {split_column}: {len(overlap)} items")
    
    return has_overlap, errors


def process_dataset(dataset_class, dataset_name: str, max_size: Optional[int] = None, *args, **kwargs) -> Dict[str, Any]:
    """
    Process a single dataset: instantiate, create splits, validate, and save.
    
    Args:
        dataset_class: Dataset class to instantiate
        dataset_name: Name for logging
        max_size: Maximum size for dataset splits (for experimental purposes)
        *args, **kwargs: Arguments to pass to dataset constructor
        
    Returns:
        Dictionary with results
    """
    result = {
        'name': dataset_name,
        'success': False,
        'error': None,
        'splits_created': False,
        'kfold_created': False,
        'splits_valid': False,
        'kfold_valid': False,
        'test_sets_match': False,
        'split_errors': [],
        'kfold_errors': [],
        'test_match_error': None
    }
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Instantiate dataset
        print(f"Instantiating {dataset_name}...")
        dataset = dataset_class(*args, **kwargs)
        
        # Get split column
        split_column = dataset.get_data_id_column()
        print(f"Using split column: {split_column}")
        print(f"Dataset shape: {dataset.get_dataframe().shape}")
        
        # Apply max_size limit if specified (for experimental purposes)
        if max_size and len(dataset.get_dataframe()) > max_size:
            print(f"⚠️  Applying experimental max_size limit: {len(dataset.get_dataframe())} → {max_size} rows")
        
        # Create both regular and k-fold splits with validation using the clean Dataset method
        print(f"Creating permanent splits and k-fold splits for {dataset_name}...")
        success, split_results = dataset.create_and_validate_permanent_splits(
            split_column=split_column,
            train_ratio=0.5,  # 50% train
            val_ratio=0.2,   # 20% val, 30% test
            k=5,
            seed=42,
            max_size=max_size
        )
        
        # Update results
        result['splits_created'] = split_results['splits_created']
        result['kfold_created'] = split_results['kfold_created']
        result['test_sets_match'] = split_results['test_sets_match']
        
        if split_results['errors']:
            result['split_errors'].extend(split_results['errors'])
            result['kfold_errors'].extend(split_results['errors'])
        
        if success:
            print(f"✅ Permanent splits and k-fold splits created successfully")
            
            # Load the created datasets for validation
            try:
                train_ds, val_ds, test_ds = dataset.load_permanent_splits()
                kfold_splits, kfold_train, kfold_test = dataset.load_permanent_kfold_splits()
                
                print(f"   Train: {len(train_ds.get_dataframe())} rows")
                print(f"   Val: {len(val_ds.get_dataframe())} rows") 
                print(f"   Test: {len(test_ds.get_dataframe()) if test_ds else 0} rows")
                print(f"   K-fold train: {len(kfold_train.get_dataframe())} rows")
                print(f"   K-fold test: {len(kfold_test.get_dataframe()) if kfold_test else 0} rows")
                print(f"   Number of folds: {len(kfold_splits)}")
                
                # Validate splits for overlaps
                print(f"Validating splits for {dataset_name}...")
                has_overlap, errors = check_splits_overlap(train_ds, val_ds, test_ds, split_column)
                result['splits_valid'] = not has_overlap
                result['split_errors'].extend(errors)
                
                if has_overlap:
                    print(f"❌ Split validation FAILED:")
                    for error in errors:
                        print(f"   - {error}")
                else:
                    print(f"✅ Split validation PASSED")
                
                # Validate k-fold splits for overlaps
                print(f"Validating k-fold splits for {dataset_name}...")
                has_overlap, errors = check_kfold_splits_overlap(kfold_splits, kfold_train, kfold_test, split_column)
                result['kfold_valid'] = not has_overlap
                result['kfold_errors'].extend(errors)
                
                if has_overlap:
                    print(f"❌ K-fold validation FAILED:")
                    for error in errors:
                        print(f"   - {error}")
                else:
                    print(f"✅ K-fold validation PASSED")
                
                # Check test set matching
                if result['test_sets_match']:
                    print(f"✅ Test sets match perfectly")
                else:
                    print(f"❌ Test sets DO NOT match")
                    result['test_match_error'] = "Test sets do not match"
                    
            except Exception as e:
                print(f"❌ Failed to load and validate splits: {str(e)}")
                result['split_errors'].append(f"Validation error: {str(e)}")
        else:
            print(f"❌ Failed to create splits")
            for error in split_results['errors']:
                print(f"   - {error}")
        
        result['success'] = True
        print(f"✅ {dataset_name} processing completed successfully")
        
    except Exception as e:
        error_msg = f"Failed to process {dataset_name}: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        result['error'] = error_msg
    
    return result


def main():
    """Main function to process all datasets."""
    print("Starting dataset splits creation and validation...")
    
    # Configuration for experimental max_size limit
    # This helps with memory constraints and focuses on limited data scenarios
    MAX_SIZE_LIMIT = 1000  # Set to None to disable, or adjust as needed
    
    if MAX_SIZE_LIMIT:
        print(f"📊 Using experimental max_size limit: {MAX_SIZE_LIMIT} rows per dataset")
        print("   This helps test metrics in limited data settings and reduces run time.")
    else:
        print("📊 No max_size limit applied - using full datasets")
    
    # Define all datasets to process
    datasets_to_process = [
        # # EvalGen datasets (use explicit subclasses to ensure unique names and correct task descriptions)
        # (EvalGenProduct, "EvalGenProduct"),
        # (EvalGenMedical, "EvalGenMedical"),
        
        # # Regular datasets (starting with smaller/simpler ones first)
        # (SimpEval, "SimpEval"),
        # (SimpDA, "SimpDA"),
        # (Primock57, "Primock57"),
        # (SummEval, "SummEval"),
        
        # # CoGym variants (smaller datasets)
        # (CoGymTabularOutcome, "CoGymTabularOutcome"),
        # (CoGymTabularProcess, "CoGymTabularProcess"),
        # (CoGymTravelOutcome, "CoGymTravelOutcome"),
        # (CoGymTravelProcess, "CoGymTravelProcess"),
        # (CoGymLessonOutcome, "CoGymLessonOutcome"),
        # (CoGymLessonProcess, "CoGymLessonProcess"),
        
        # # Larger datasets (might be memory intensive)
        # (RealHumanEval, "RealHumanEval"),
        
        # # HelpSteer datasets
        # (HelpSteer, "HelpSteer"),
        # (HelpSteer2, "HelpSteer2"),
        
        # # Design2Code
        # (Design2Code, "Design2Code"),

        # # AI_Researcher dataset
        # (AI_Researcher, "AI_Researcher"),

        # # ICLR dataset
        # (ICLR, "ICLR"),

        # # TauBench dataset
        # (TauBench, "TauBench"),
        # (TauBenchBigger, "TauBenchBigger"),
        (TauBenchHighTemperature, "TauBenchHighTemperature"),
    ]
    
    # Process all datasets
    results = []
    total_datasets = len(datasets_to_process)
    
    for i, (dataset_class, dataset_name) in enumerate(datasets_to_process, 1):
        print(f"\n[{i}/{total_datasets}] Processing {dataset_name}...")
        result = process_dataset(dataset_class, dataset_name, max_size=MAX_SIZE_LIMIT)
        results.append(result)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r['success'])
    splits_created = sum(1 for r in results if r['splits_created'])
    kfold_created = sum(1 for r in results if r['kfold_created'])
    splits_valid = sum(1 for r in results if r['splits_valid'])
    kfold_valid = sum(1 for r in results if r['kfold_valid'])
    test_sets_match = sum(1 for r in results if r['test_sets_match'])
    
    print(f"Datasets processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Permanent splits created: {splits_created}")
    print(f"K-fold splits created: {kfold_created}")
    print(f"Valid permanent splits: {splits_valid}")
    print(f"Valid k-fold splits: {kfold_valid}")
    print(f"Test sets matching: {test_sets_match}")
    
    # Report failures
    failures = [r for r in results if not r['success']]
    if failures:
        print(f"\n❌ FAILED DATASETS ({len(failures)}):")
        for failure in failures:
            print(f"   - {failure['name']}: {failure['error']}")
    
    # Report validation failures
    split_failures = [r for r in results if r['splits_created'] and not r['splits_valid']]
    if split_failures:
        print(f"\n❌ SPLIT VALIDATION FAILURES ({len(split_failures)}):")
        for failure in split_failures:
            print(f"   - {failure['name']}:")
            for error in failure['split_errors']:
                print(f"     * {error}")
    
    kfold_failures = [r for r in results if r['kfold_created'] and not r['kfold_valid']]
    if kfold_failures:
        print(f"\n❌ K-FOLD VALIDATION FAILURES ({len(kfold_failures)}):")
        for failure in kfold_failures:
            print(f"   - {failure['name']}:")
            for error in failure['kfold_errors']:
                print(f"     * {error}")
    
    # Report test set matching failures
    test_match_failures = [r for r in results if r['splits_created'] and r['kfold_created'] and not r['test_sets_match']]
    if test_match_failures:
        print(f"\n❌ TEST SET MATCHING FAILURES ({len(test_match_failures)}):")
        for failure in test_match_failures:
            print(f"   - {failure['name']}: {failure['test_match_error']}")
    
    # Success summary
    all_validations_passed = (splits_valid == splits_created and 
                             kfold_valid == kfold_created and 
                             test_sets_match == sum(1 for r in results if r['splits_created'] and r['kfold_created']))
    
    if all_validations_passed:
        print(f"\n🎉 ALL VALIDATIONS PASSED! No overlaps detected and all test sets match perfectly.")
    else:
        print(f"\n⚠️  Some validation failures detected. Please review above.")
    
    print(f"\nScript completed!")


if __name__ == "__main__":
    main()

# Example Usage
# python analysis/main_experiments/create_dataset_splits.py 