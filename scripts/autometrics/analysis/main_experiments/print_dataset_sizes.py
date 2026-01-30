#!/usr/bin/env python3
"""
Script to print the sizes of all dataset splits.

This script:
1. Loops over all available datasets from autometrics
2. Loads their permanent splits (train/val/test)
3. Prints the size of each split and the overall dataset
"""

import sys
import os
from typing import List, Tuple, Optional
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


def print_dataset_sizes(dataset_class, dataset_name: str, *args, **kwargs) -> bool:
    """
    Print the sizes of a single dataset's splits.
    
    Args:
        dataset_class: Dataset class to instantiate
        dataset_name: Name for logging
        *args, **kwargs: Arguments to pass to dataset constructor
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Instantiate dataset
        dataset = dataset_class(*args, **kwargs)
        
        # Get overall dataset size
        overall_size = len(dataset.get_dataframe())
        print(f"Overall dataset size: {overall_size:,} rows")
        
        # Try to load permanent splits
        try:
            train_ds, val_ds, test_ds = dataset.load_permanent_splits()
            
            train_size = len(train_ds.get_dataframe()) if train_ds else 0
            val_size = len(val_ds.get_dataframe()) if val_ds else 0
            test_size = len(test_ds.get_dataframe()) if test_ds else 0
            
            print(f"  Train split: {train_size:,} rows")
            print(f"  Val split:   {val_size:,} rows")
            print(f"  Test split:  {test_size:,} rows")
            
            # Verify total adds up
            total_splits = train_size + val_size + test_size
            if total_splits == overall_size:
                print(f"  ✅ Total splits ({total_splits:,}) matches overall size")
            else:
                print(f"  ⚠️  Total splits ({total_splits:,}) does NOT match overall size ({overall_size:,})")
                
        except Exception as e:
            print(f"  ❌ Could not load permanent splits: {str(e)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to process {dataset_name}: {str(e)}")
        return False


def main():
    """Main function to print sizes of all dataset splits."""
    print("Printing dataset split sizes...")
    
    # Define all datasets to process
    datasets_to_process = [
        # EvalGen datasets
        (EvalGenProduct, "EvalGenProduct"),
        (EvalGenMedical, "EvalGenMedical"),
        
        # Regular datasets
        (SimpEval, "SimpEval"),
        (SimpDA, "SimpDA"),
        (Primock57, "Primock57"),
        (SummEval, "SummEval"),
        
        # CoGym variants
        (CoGymTabularOutcome, "CoGymTabularOutcome"),
        (CoGymTabularProcess, "CoGymTabularProcess"),
        (CoGymTravelOutcome, "CoGymTravelOutcome"),
        (CoGymTravelProcess, "CoGymTravelProcess"),
        (CoGymLessonOutcome, "CoGymLessonOutcome"),
        (CoGymLessonProcess, "CoGymLessonProcess"),
        
        # Larger datasets
        (RealHumanEval, "RealHumanEval"),
        
        # HelpSteer datasets
        (HelpSteer, "HelpSteer"),
        (HelpSteer2, "HelpSteer2"),
        
        # Design2Code
        (Design2Code, "Design2Code"),

        # AI_Researcher dataset
        (AI_Researcher, "AI_Researcher"),

        # ICLR dataset
        (ICLR, "ICLR"),
    ]
    
    # Process all datasets
    successful = 0
    total_datasets = len(datasets_to_process)
    
    for i, (dataset_class, dataset_name) in enumerate(datasets_to_process, 1):
        print(f"\n[{i}/{total_datasets}] Processing {dataset_name}...")
        if print_dataset_sizes(dataset_class, dataset_name):
            successful += 1
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Datasets processed: {total_datasets}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_datasets - successful}")
    print(f"\nScript completed!")


if __name__ == "__main__":
    main()

# Example Usage
# python analysis/main_experiments/print_dataset_sizes.py
