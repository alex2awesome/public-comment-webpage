import warnings
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import Field
from autometrics.dataset.Dataset import Dataset
from autometrics.metrics.MultiMetric import MultiMetric
from autometrics.metrics.PairwiseMetric import PairwiseMetric
from autometrics.metrics.PairwiseMultiMetric import PairwiseMultiMetric
from autometrics.metrics.Metric import Metric

class PairwiseDataset(Dataset):
    """
    Dataset class for handling paired model outputs for comparison.
    """

    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 target_columns: List[str], 
                 ignore_columns: List[str], 
                 metric_columns: List[str], 
                 name: str, 
                 data_id_column: Optional[str] = None, 
                 model_id_column_1: Optional[str] = None,
                 model_id_column_2: Optional[str] = None,
                 input_column: Optional[str] = None, 
                 output_column_1: Optional[str] = None, 
                 output_column_2: Optional[str] = None, 
                 reference_columns: Optional[List[str]] = None, 
                 metrics: List[Metric] = None, 
                 task_description: Optional[str] = None):
        """
        Initialize a PairwiseDataset with paired outputs.
        
        Args:
            dataframe: DataFrame containing the data
            target_columns: Names of target columns
            ignore_columns: Names of columns to ignore
            metric_columns: Names of metric columns
            name: Name of the dataset
            data_id_column: Name of column with data IDs
            model_id_column_1: Name of column with model IDs for first model outputs
            model_id_column_2: Name of column with model IDs for second model outputs
            input_column: Name of column with inputs
            output_column_1: Name of column with outputs from first model
            output_column_2: Name of column with outputs from second model
            reference_columns: Names of columns with reference outputs
            metrics: List of metrics to use
            task_description: Description of the task
        """
        # Call the parent constructor
        super().__init__(
            dataframe=dataframe,
            target_columns=target_columns,
            ignore_columns=ignore_columns,
            metric_columns=metric_columns,
            name=name,
            data_id_column=data_id_column,
            model_id_column=model_id_column_1,  # Use model_id_column_1 as default model_id_column
            input_column=input_column,
            output_column=output_column_1,  # Use output_column_1 as default output_column
            reference_columns=reference_columns,
            metrics=metrics if metrics is not None else [],
            task_description=task_description
        )
        
        # Store additional attributes specific to pairwise datasets
        self.output_column_1 = output_column_1
        self.output_column_2 = output_column_2
        self.model_id_column_1 = model_id_column_1
        self.model_id_column_2 = model_id_column_2
        
        # Validate required columns
        if not output_column_1 or not output_column_2:
            raise ValueError("Both output_column_1 and output_column_2 must be provided for PairwiseDataset")
            
        if output_column_1 not in dataframe.columns or output_column_2 not in dataframe.columns:
            raise ValueError(f"Output columns must exist in dataframe: {output_column_1}, {output_column_2}")
        
        # Check model ID columns if provided
        if model_id_column_1 and model_id_column_1 not in dataframe.columns:
            raise ValueError(f"Model ID column 1 '{model_id_column_1}' not found in dataframe")
        if model_id_column_2 and model_id_column_2 not in dataframe.columns:
            raise ValueError(f"Model ID column 2 '{model_id_column_2}' not found in dataframe")

    def get_output_column_1(self) -> str:
        """
        Get the name of the column containing outputs from the first model.
        
        Returns:
            The name of the first output column
        """
        return self.output_column_1
        
    def get_output_column_2(self) -> str:
        """
        Get the name of the column containing outputs from the second model.
        
        Returns:
            The name of the second output column
        """
        return self.output_column_2
    
    def get_model_id_column_1(self) -> Optional[str]:
        """
        Get the name of the column containing model IDs for the first model.
        
        Returns:
            The name of the first model ID column, or None if not defined
        """
        return self.model_id_column_1
        
    def get_model_id_column_2(self) -> Optional[str]:
        """
        Get the name of the column containing model IDs for the second model.
        
        Returns:
            The name of the second model ID column, or None if not defined
        """
        return self.model_id_column_2

    def add_metric(self, metric: Metric, update_dataset: bool = True):
        """
        Add a metric to the dataset. If it's not a PairwiseMetric, wrap it.
        
        Args:
            metric: The metric to add
            update_dataset: Whether to update the dataset with metric values
            
        Returns:
            None
        """
        # If the metric is not already a pairwise metric, wrap it
        if not isinstance(metric, (PairwiseMetric, PairwiseMultiMetric)):
            pairwise_metric = None
            if isinstance(metric, MultiMetric):
                pairwise_metric = PairwiseMultiMetric(multi_metric=metric)
            else:
                pairwise_metric = PairwiseMetric(scalar_metric=metric)
            self.metrics.append(pairwise_metric)
            
            # Add metric names to metric_columns
            if isinstance(pairwise_metric, PairwiseMultiMetric):
                # For MultiMetrics, add all submetric names
                for submetric_name in pairwise_metric.get_submetric_names():
                    if submetric_name not in self.metric_columns:
                        self.metric_columns.append(submetric_name)
            else:
                # For regular metrics, add the single metric name
                if pairwise_metric.get_name() not in self.metric_columns:
                    self.metric_columns.append(pairwise_metric.get_name())
                
            # Calculate and update the dataset if requested
            if update_dataset and self.dataframe is not None:
                pairwise_metric.predict(self, update_dataset=True)
        else:
            # The metric is already a pairwise metric
            self.metrics.append(metric)
            
            # Add metric names to metric_columns
            if isinstance(metric, PairwiseMultiMetric):
                # For MultiMetrics, add all submetric names
                for submetric_name in metric.get_submetric_names():
                    if submetric_name not in self.metric_columns:
                        self.metric_columns.append(submetric_name)
            else:
                # For regular metrics, add the single metric name
                if metric.get_name() not in self.metric_columns:
                    self.metric_columns.append(metric.get_name())
                
            # Calculate and update the dataset if requested
            if update_dataset and self.dataframe is not None:
                metric.predict(self, update_dataset=True)

    def calculate_metrics(self, update_dataset: bool = True, **kwargs):
        """
        Calculate all metrics for the dataset.
        
        Args:
            update_dataset: Whether to update the dataset with metric values
            **kwargs: Additional arguments for metric calculation
            
        Returns:
            None
        """
        for metric in self.metrics:
            if isinstance(metric, (PairwiseMetric, PairwiseMultiMetric)):
                metric.predict(self, update_dataset=update_dataset, **kwargs)
            else:
                # Not a PairwiseMetric - create a wrapper and use it
                pairwise_metric = None
                if isinstance(metric, MultiMetric):
                    pairwise_metric = PairwiseMultiMetric(multi_metric=metric)
                else:
                    pairwise_metric = PairwiseMetric(scalar_metric=metric)
                pairwise_metric.predict(self, update_dataset=update_dataset, **kwargs)
    
    def get_metric_values(self, metric: Metric, update_dataset: bool = True, **kwargs):
        """
        Get the values for a specific metric.
        
        Args:
            metric: The metric to use
            update_dataset: Whether to update the dataset with metric values
            **kwargs: Additional arguments for metric calculation
            
        Returns:
            The metric values for all rows
        """
        # Make sure we're using a pairwise metric
        if not isinstance(metric, (PairwiseMetric, PairwiseMultiMetric)):
            if isinstance(metric, MultiMetric):
                pairwise_metric = PairwiseMultiMetric(multi_metric=metric)
            else:
                pairwise_metric = PairwiseMetric(scalar_metric=metric)
        else:
            pairwise_metric = metric
            
        # Calculate metric values if they don't exist in the dataset
        if update_dataset:
            # For MultiMetrics, check if all submetric columns exist
            if isinstance(pairwise_metric, PairwiseMultiMetric):
                submetric_names = pairwise_metric.get_submetric_names()
                if any(name not in self.get_dataframe().columns for name in submetric_names):
                    pairwise_metric.predict(self, update_dataset=True, **kwargs)
            # For regular metrics, check if the metric column exists
            elif pairwise_metric.get_name() not in self.get_metric_columns():
                pairwise_metric.predict(self, update_dataset=True, **kwargs)
            
        df = self.get_dataframe()
        
        # Return appropriate columns based on metric type
        if isinstance(pairwise_metric, PairwiseMultiMetric):
            return df[pairwise_metric.get_submetric_names()]
        else:
            return df[pairwise_metric.get_name()]
    
    def get_splits(self, split_column: Optional[str] = None, train_ratio: float = 0.5, val_ratio: float = 0.2, seed: Optional[int] = None, max_size: Optional[int] = None):
        """
        Split the dataset into training, validation, and test sets using consistent logic.
        
        Args:
            split_column: Column to use for splitting
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            seed: Random seed for reproducibility
            max_size: Maximum size of each split
            
        Returns:
            train_dataset, val_dataset, test_dataset (all as PairwiseDatasets)
        """
        # Use the parent class get_splits method for consistent logic
        train_dataset, val_dataset, test_dataset = super().get_splits(
            split_column=split_column,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            max_size=max_size
        )
        
        # Convert each split to a PairwiseDataset
        train_pairwise = self._convert_to_pairwise(train_dataset)
        val_pairwise = self._convert_to_pairwise(val_dataset)
        test_pairwise = self._convert_to_pairwise(test_dataset)
        
        return train_pairwise, val_pairwise, test_pairwise
    
    def get_kfold_splits(self, k: int = 5, split_column: Optional[str] = None, seed: Optional[int] = None, test_ratio: float = 0.3, max_size: Optional[int] = None):
        """
        Split the dataset into k folds.
        
        Args:
            k: Number of folds
            split_column: Column to use for splitting
            seed: Random seed for reproducibility
            test_ratio: Ratio of data to use for testing
            max_size: Maximum size of each split
            
        Returns:
            List of (train, val) splits, train_dataset, test_dataset
        """
        split_datasets, train_dataset, test_dataset = super().get_kfold_splits(
            k=k,
            split_column=split_column,
            seed=seed,
            test_ratio=test_ratio,
            max_size=max_size
        )
        
        # Convert each split to a PairwiseDataset
        pairwise_splits = []
        for train_split, val_split in split_datasets:
            train_pairwise = self._convert_to_pairwise(train_split)
            val_pairwise = self._convert_to_pairwise(val_split)
            pairwise_splits.append((train_pairwise, val_pairwise))
            
        train_pairwise = self._convert_to_pairwise(train_dataset)
        test_pairwise = None if test_dataset is None else self._convert_to_pairwise(test_dataset)
        
        return pairwise_splits, train_pairwise, test_pairwise
    
    def get_subset(self, size: int, seed: Optional[int] = None) -> 'PairwiseDataset':
        """
        Get a random subset of the dataset.
        
        Args:
            size: Size of the subset
            seed: Random seed for reproducibility
            
        Returns:
            A new PairwiseDataset with a subset of the data
        """
        subset = super().get_subset(size, seed)
        return self._convert_to_pairwise(subset)
    
    def copy(self) -> 'PairwiseDataset':
        """
        Create a deep copy of the dataset.
        
        Returns:
            A new PairwiseDataset with copied data
        """
        copied_dataset = super().copy()
        return self._convert_to_pairwise(copied_dataset)
        
    def _convert_to_pairwise(self, dataset: Dataset) -> 'PairwiseDataset':
        """
        Convert a Dataset to a PairwiseDataset.
        
        Args:
            dataset: The Dataset to convert
            
        Returns:
            A new PairwiseDataset with the same data
        """
        return PairwiseDataset(
            dataframe=dataset.get_dataframe().copy(),
            target_columns=dataset.get_target_columns(),
            ignore_columns=dataset.get_ignore_columns(),
            metric_columns=dataset.get_metric_columns(),
            name=dataset.get_name(),
            data_id_column=dataset.get_data_id_column(),
            model_id_column_1=self.model_id_column_1,
            model_id_column_2=self.model_id_column_2,
            input_column=dataset.get_input_column(),
            output_column_1=self.output_column_1,
            output_column_2=self.output_column_2,
            reference_columns=dataset.get_reference_columns(),
            metrics=[metric for metric in self.metrics],
            task_description=dataset.get_task_description()
        )
    
    def save_to_file(self, path: str):
        """
        Save the dataset to a CSV file.
        
        Args:
            path: Path to save the CSV file
            
        Returns:
            None
        """
        # Create a metadata dictionary with pairwise-specific attributes
        import json
        
        metadata = {
            'target_columns': self.target_columns,
            'ignore_columns': self.ignore_columns,
            'metric_columns': self.metric_columns,
            'name': self.name,
            'data_id_column': self.data_id_column,
            'model_id_column_1': self.model_id_column_1,
            'model_id_column_2': self.model_id_column_2,
            'input_column': self.input_column,
            'output_column_1': self.output_column_1,
            'output_column_2': self.output_column_2,
            'reference_columns': self.reference_columns,
            'task_description': self.task_description,
            'dataset_type': 'PairwiseDataset'
        }
        
        # Save the dataframe
        self.dataframe.to_csv(path, index=False)
        
        # Save metadata to a companion JSON file
        metadata_path = path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_from_file(self, path: str) -> 'PairwiseDataset':
        """
        Load a PairwiseDataset from a CSV file and its metadata.
        
        Args:
            path: Path to the CSV file
            
        Returns:
            A new PairwiseDataset with the loaded data
        """
        import json
        
        # Load the dataframe
        df = pd.read_csv(path)
        
        # Load metadata from companion JSON file
        metadata_path = path.replace('.csv', '_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify this is a PairwiseDataset
            if metadata.get('dataset_type') != 'PairwiseDataset':
                # Fall back to treating as regular Dataset if metadata indicates otherwise
                return super().load_from_file(path)
                
            return PairwiseDataset(
                dataframe=df,
                target_columns=metadata['target_columns'],
                ignore_columns=metadata['ignore_columns'],
                metric_columns=metadata['metric_columns'],
                name=metadata['name'],
                data_id_column=metadata['data_id_column'],
                model_id_column_1=metadata['model_id_column_1'],
                model_id_column_2=metadata['model_id_column_2'],
                input_column=metadata['input_column'],
                output_column_1=metadata['output_column_1'],
                output_column_2=metadata['output_column_2'],
                reference_columns=metadata['reference_columns'],
                metrics=[],  # Don't load metrics, they should be added separately
                task_description=metadata['task_description']
            )
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            # Fallback: try to create PairwiseDataset using current instance's attributes
            # This handles cases where metadata file doesn't exist or is corrupted
            return PairwiseDataset(
                dataframe=df,
                target_columns=self.target_columns,
                ignore_columns=self.ignore_columns,
                metric_columns=self.metric_columns,
                name=self.name,
                data_id_column=self.data_id_column,
                model_id_column_1=self.model_id_column_1,
                model_id_column_2=self.model_id_column_2,
                input_column=self.input_column,
                output_column_1=self.output_column_1,
                output_column_2=self.output_column_2,
                reference_columns=self.reference_columns,
                metrics=[],  # Don't load metrics, they should be added separately
                task_description=self.task_description
            )