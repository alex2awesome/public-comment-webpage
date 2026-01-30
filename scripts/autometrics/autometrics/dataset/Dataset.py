import warnings
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any
from autometrics.metrics.MultiMetric import MultiMetric
from autometrics.metrics.Metric import Metric
from platformdirs import user_data_dir
import os

class Dataset():
    """
    Dataset class for handling and manipulating datasets.
    """
    # dataframe: pd.DataFrame
    # target_columns: List[str]
    # ignore_columns: List[str]
    # metric_columns: List[str]
    # name: str
    # data_id_column: Optional[str] = None
    # model_id_column: Optional[str] = None
    # input_column: Optional[str] = None
    # output_column: Optional[str] = None
    # reference_columns: Optional[List[str]] = None
    # metrics: List[Metric] = Field(default_factory=list)

    def __init__(self, dataframe: pd.DataFrame, target_columns: List[str], ignore_columns: List[str], metric_columns: List[str], name: str, data_id_column: Optional[str] = None, model_id_column: Optional[str] = None, input_column: Optional[str] = None, output_column: Optional[str] = None, reference_columns: Optional[List[str]] = None, metrics: List[Metric] = None, task_description: Optional[str] = None):
        self.dataframe = dataframe
        self.target_columns = target_columns
        self.ignore_columns = ignore_columns
        self.metric_columns = metric_columns
        self.name = name
        self.data_id_column = data_id_column
        self.model_id_column = model_id_column
        self.input_column = input_column
        self.output_column = output_column
        self.reference_columns = reference_columns
        self.metrics = metrics if metrics else []
        self.task_description = task_description
    class Config:
        arbitrary_types_allowed = True
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.dataframe
    
    def get_target_columns(self) -> List[str]:
        return self.target_columns
    
    def get_ignore_columns(self) -> List[str]:
        return self.ignore_columns
    
    def get_metric_columns(self) -> List[str]:
        return self.metric_columns
    
    def get_name(self) -> str:
        return self.name
    
    def get_data_id_column(self) -> Optional[str]:
        return self.data_id_column
    
    def get_model_id_column(self) -> Optional[str]:
        return self.model_id_column
    
    def get_input_column(self) -> Optional[str]:
        return self.input_column
    
    def get_output_column(self) -> Optional[str]:
        return self.output_column
    
    def get_reference_columns(self) -> Optional[List[str]]:
        return self.reference_columns
    
    def get_metrics(self) -> List[Metric]:
        return self.metrics
    
    def set_dataframe(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def add_metric(self, metric: Metric, update_dataset: bool = True):
        
        if isinstance(metric, MultiMetric):
            self.metrics.append(metric)
            for submetric_name in metric.get_submetric_names():
                if submetric_name not in self.metric_columns:
                    self.metric_columns.append(submetric_name)

            if self.dataframe is not None and update_dataset and any(submetric_name not in self.dataframe.columns for submetric_name in metric.get_submetric_names()):
                metric.predict(self, update_dataset=update_dataset)
        else:
            self.metrics.append(metric)
            if metric.get_name() not in self.metric_columns:
                self.metric_columns.append(metric.get_name())
            if self.dataframe is not None and update_dataset and metric.get_name() not in self.dataframe.columns:
                metric.predict(self, update_dataset=update_dataset)

    def add_metrics(self, metrics: List[Metric], update_dataset: bool = True):
        for metric in metrics:
            self.add_metric(metric, update_dataset=update_dataset)
    
    def __str__(self) -> str:
        return (f"Dataset: {self.name}, Target Columns: {self.target_columns}, "
                f"Ignore Columns: {self.ignore_columns}, Metric Columns: {self.metric_columns}\n"
                f"{self.dataframe.head()}")
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_splits(self, split_column: Optional[str] = None, train_ratio: float = 0.5, val_ratio: float = 0.2, seed: Optional[int] = None, max_size: Optional[int] = None):
        # To ensure consistent test sets, we'll use the same test-splitting logic as get_kfold_splits
        test_ratio = 1.0 - train_ratio - val_ratio
        df = self.get_dataframe()

        # First, split off the test set using the same logic as get_kfold_splits
        if test_ratio > 0:
            if seed:
                np.random.seed(seed)

            if not split_column:
                split_column = self.data_id_column

            if not split_column:
                items = np.arange(len(df))
                test_size = int(test_ratio * len(items))
                test_items = np.random.choice(np.arange(len(df)), test_size, replace=False)
                test_df = df[df.index.isin(test_items)].copy()
                df = df[~df.index.isin(test_items)].copy()
                train_val_df = df.copy()
            else:
                items = df[split_column].unique()
                test_size = int(test_ratio * len(items))
                test_items = np.random.choice(df[split_column].unique(), test_size, replace=False)
                test_df = df[df[split_column].isin(test_items)].copy()
                df = df[~df[split_column].isin(test_items)].copy()
                train_val_df = df.copy()
        else:
            train_val_df = df.copy()
            test_df = None

        # Now split the remaining data into train and val
        if not split_column:
            split_column = self.data_id_column
        if not split_column:
            warnings.warn("No split column specified. Splitting based on index which is not recommended. "
                          "This means that we could be testing on data that is partially represented in the training set due to rows with similar data but different indices.")
            items = np.arange(len(train_val_df))
        else:
            items = train_val_df[split_column].unique()

        # Calculate train/val split from remaining data
        total_train_val_ratio = train_ratio + val_ratio
        adjusted_train_ratio = train_ratio / total_train_val_ratio
        adjusted_val_ratio = val_ratio / total_train_val_ratio

        train_size = int(adjusted_train_ratio * len(items))
        val_size = int(adjusted_val_ratio * len(items))

        if train_size + val_size < len(items):
            train_size += len(items) - (train_size + val_size)

        if seed:
            np.random.seed(seed)

        np.random.shuffle(items)

        train_items = items[:train_size]
        val_items = items[train_size:train_size + val_size]

        if split_column:
            train_df = train_val_df[train_val_df[split_column].isin(train_items)].copy()
            val_df = train_val_df[train_val_df[split_column].isin(val_items)].copy()
        else:
            train_df = train_val_df.iloc[train_items].copy()
            val_df = train_val_df.iloc[val_items].copy()

        train_dataset = Dataset(
            dataframe=train_df,
            target_columns=self.target_columns,
            ignore_columns=self.ignore_columns,
            metric_columns=self.metric_columns,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns,
            metrics=[],
            task_description=self.task_description
        )
        val_dataset = Dataset(
            dataframe=val_df,
            target_columns=self.target_columns,
            ignore_columns=self.ignore_columns,
            metric_columns=self.metric_columns,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns,
            metrics=[],
            task_description=self.task_description
        )
        
        # Handle test dataset
        if test_df is not None:
            test_dataset = Dataset(
                dataframe=test_df,
                target_columns=self.target_columns,
                ignore_columns=self.ignore_columns,
                metric_columns=self.metric_columns,
                name=self.name,
                data_id_column=self.data_id_column,
                model_id_column=self.model_id_column,
                input_column=self.input_column,
                output_column=self.output_column,
                reference_columns=self.reference_columns,
                metrics=[],
                task_description=self.task_description
            )
        else:
            test_dataset = None

        # Apply max_size to each split if specified
        if max_size:
            if seed is not None:
                train_dataset = train_dataset.get_subset(max_size, seed=seed)
                val_dataset = val_dataset.get_subset(max_size, seed=seed)
                if test_dataset is not None:
                    test_dataset = test_dataset.get_subset(max_size, seed=seed)
            else:
                train_dataset = train_dataset.get_subset(max_size, seed=42)
                val_dataset = val_dataset.get_subset(max_size, seed=42)
                if test_dataset is not None:
                    test_dataset = test_dataset.get_subset(max_size, seed=42)

        return train_dataset, val_dataset, test_dataset

    def get_kfold_splits(self, k: int = 5, split_column: Optional[str] = None, seed: Optional[int] = None, test_ratio: float = 0.3, max_size: Optional[int] = None):
        df = self.get_dataframe()

        if test_ratio and test_ratio > 0:
            if seed:
                np.random.seed(seed)

            if not split_column:
                split_column = self.data_id_column

            if not split_column:
                items = np.arange(len(df))
                test_size = int(test_ratio * len(items))
                test_items = np.random.choice(np.arange(len(df)), test_size, replace=False)
                test_df = df[df.index.isin(test_items)].copy()
                df = df[~df.index.isin(test_items)].copy()
                train_df = df.copy()
            else:
                items = df[split_column].unique()
                test_size = int(test_ratio * len(items))
                test_items = np.random.choice(df[split_column].unique(), test_size, replace=False)
                test_df = df[df[split_column].isin(test_items)].copy()
                df = df[~df[split_column].isin(test_items)].copy()
                train_df = df.copy()
        else:
            train_df = df.copy()
            test_df = None

        # Create overall train and test datasets
        train_dataset = Dataset(
            dataframe=train_df,
            target_columns=self.target_columns,
            ignore_columns=self.ignore_columns,
            metric_columns=self.metric_columns,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns,
            metrics=[],
            task_description=self.task_description
        )
        
        # Handle case where test_df is None
        if test_df is not None:
            test_dataset = Dataset(
                dataframe=test_df,
                target_columns=self.target_columns,
                ignore_columns=self.ignore_columns,
                metric_columns=self.metric_columns,
                name=self.name,
                data_id_column=self.data_id_column,
                model_id_column=self.model_id_column,
                input_column=self.input_column,
                output_column=self.output_column,
                reference_columns=self.reference_columns,
                metrics=[],
                task_description=self.task_description
            )
        else:
            test_dataset = None

        # Apply max_size to overall train and test datasets FIRST
        if max_size:
            if seed is not None:
                train_dataset = train_dataset.get_subset(max_size, seed=seed)
                if test_dataset is not None:
                    test_dataset = test_dataset.get_subset(max_size, seed=seed)
            else:
                train_dataset = train_dataset.get_subset(max_size, seed=42)
                if test_dataset is not None:
                    test_dataset = test_dataset.get_subset(max_size, seed=42)

        # Now create k-folds from the (possibly limited) train dataset
        working_train_df = train_dataset.get_dataframe()
        
        if not split_column:
            split_column = self.data_id_column
        if not split_column:
            warnings.warn("No split column specified. Splitting based on index which is not recommended. "
                          "This means that we could be testing on data that is partially represented in the training set due to rows with similar data but different indices.")
            items = np.arange(len(working_train_df))
        else:
            items = working_train_df[split_column].unique()

        if seed:
            np.random.seed(seed)

        np.random.shuffle(items)

        splits = np.array_split(items, k)

        split_datasets = []
        for i in range(k):
            split_items = splits[i]
            if k == 1:
                # Special case: when k=1, the validation fold is the single split,
                # and the training fold is empty (we'll return empty training data)
                non_split_items = np.array([])  # Empty array for training
            else:
                non_split_items = np.concatenate([splits[j] for j in range(k) if j != i])
                
            if split_column:
                split_df = working_train_df[working_train_df[split_column].isin(split_items)].copy()
                if k == 1:
                    # For k=1, create an empty dataframe for training
                    non_split_df = working_train_df.iloc[0:0].copy()  # Empty dataframe with same structure
                else:
                    non_split_df = working_train_df[working_train_df[split_column].isin(non_split_items)].copy()
            else:
                split_df = working_train_df.iloc[split_items].copy()
                if k == 1:
                    # For k=1, create an empty dataframe for training
                    non_split_df = working_train_df.iloc[0:0].copy()  # Empty dataframe with same structure
                else:
                    non_split_df = working_train_df.iloc[non_split_items].copy()
                    
            split_val_dataset = Dataset(
                dataframe=split_df,
                target_columns=self.target_columns,
                ignore_columns=self.ignore_columns,
                metric_columns=self.metric_columns,
                name=self.name,
                data_id_column=self.data_id_column,
                model_id_column=self.model_id_column,
                input_column=self.input_column,
                output_column=self.output_column,
                reference_columns=self.reference_columns,
                metrics=[],
                task_description=self.task_description
            )
            split_train_dataset = Dataset(
                dataframe=non_split_df,
                target_columns=self.target_columns,
                ignore_columns=self.ignore_columns,
                metric_columns=self.metric_columns,
                name=self.name,
                data_id_column=self.data_id_column,
                model_id_column=self.model_id_column,
                input_column=self.input_column,
                output_column=self.output_column,
                reference_columns=self.reference_columns,
                metrics=[],
                task_description=self.task_description
            )
            
            split_datasets.append((split_train_dataset, split_val_dataset))

        return split_datasets, train_dataset, test_dataset
    
    def calculate_metrics(self, update_dataset: bool = True, **kwargs):
        for metric in self.metrics:
            self.get_metric_values(metric, update_dataset=update_dataset, **kwargs)

    def get_metric_values(self, metric: Metric, update_dataset: bool = True, **kwargs):
        if metric.get_name() not in self.get_metric_columns() and update_dataset:
            metric.predict(self, update_dataset=update_dataset, **kwargs)

        df = self.get_dataframe()

        if update_dataset:
            for _, row in df.iterrows():
                if isinstance(metric, MultiMetric):
                    for submetric_name in metric.get_submetric_names():
                        if submetric_name not in row:
                            metric.calculate_row(row, self, update_dataset=update_dataset)
                else:
                    if metric.get_name() not in row:
                        metric.calculate_row(row, self, update_dataset=update_dataset)

        if isinstance(metric, MultiMetric):
            return df[metric.get_submetric_names()]
        else:
            return df[metric.get_name()]

    def get_subset(self, size: int, seed: Optional[int] = None) -> 'Dataset':
        df = self.get_dataframe()
        if seed:
            np.random.seed(seed)
            
        # Cap the size to the number of rows in the dataframe to prevent sampling errors
        actual_size = min(size, len(df))
        if actual_size < size:
            warnings.warn(f"Requested subset size {size} is larger than available data ({len(df)} rows). Using all available data.")
            
        # If we're using all data, no need to sample
        if actual_size == len(df):
            subset_df = df.copy()
        else:
            subset_df = df.sample(n=actual_size)
            
        return Dataset(
            dataframe=subset_df,
            target_columns=self.target_columns,
            ignore_columns=self.ignore_columns,
            metric_columns=self.metric_columns,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns,
            metrics=[],
            task_description=self.task_description
        )
    
    def copy(self) -> 'Dataset':
        return Dataset(
            dataframe=self.dataframe.copy(),
            target_columns=self.target_columns.copy(),
            ignore_columns=self.ignore_columns.copy(),
            metric_columns=self.metric_columns.copy(),
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns.copy() if self.reference_columns is not None else None,
            metrics=[metric for metric in self.metrics],
            task_description=self.task_description
        )
    
    def get_task_description(self) -> Optional[str]:
        return self.task_description
    
    def save_permanent_splits(self, split_column: str, train_ratio: float = 0.5, val_ratio: float = 0.2, seed: Optional[int] = None, max_size: Optional[int] = None):
        # Now that get_splits uses get_kfold_splits internally, we can use it directly
        train_dataset, val_dataset, test_dataset = self.get_splits(split_column, train_ratio, val_ratio, seed, max_size)
        
        # Create directory if it doesn't exist
        directory = f"{user_data_dir()}/autometrics/datasets/{self.name}"
        os.makedirs(directory, exist_ok=True)
        
        train_dataset.save_to_file(f"{directory}/train.csv")
        val_dataset.save_to_file(f"{directory}/val.csv")
        test_dataset.save_to_file(f"{directory}/test.csv")

        return train_dataset, val_dataset, test_dataset
    
    def load_permanent_splits(self, directory: Optional[str] = None, resized: bool = False):
        if directory is None:
            directory = f"{user_data_dir()}/autometrics/datasets/{self.name}"

        train_dataset = self.load_from_file(f"{directory}/train{'_resized' if resized else ''}.csv")
        val_dataset = self.load_from_file(f"{directory}/val{'_resized' if resized else ''}.csv")
        test_dataset = self.load_from_file(f"{directory}/test.csv")
        return train_dataset, val_dataset, test_dataset
    
    def save_permanent_kfold_splits(self, k: int = 5, split_column: Optional[str] = None, seed: Optional[int] = None, test_ratio: float = 0.3, max_size: Optional[int] = None):
        split_datasets, train_dataset, test_dataset = self.get_kfold_splits(k, split_column, seed, test_ratio, max_size)

        # Create directory if it doesn't exist
        directory = f"{user_data_dir()}/autometrics/datasets/{self.name}"
        os.makedirs(directory, exist_ok=True)
        
        train_dataset.save_to_file(f"{directory}/train_kfold.csv")
        if test_dataset is not None:
            test_dataset.save_to_file(f"{directory}/test_kfold.csv")

        for i, (split_train_dataset, split_val_dataset) in enumerate(split_datasets):
            split_train_dataset.save_to_file(f"{directory}/split_{i}_train.csv")
            split_val_dataset.save_to_file(f"{directory}/split_{i}_val.csv")

        return split_datasets, train_dataset, test_dataset
    
    def load_permanent_kfold_splits(self, directory: Optional[str] = None):
        if directory is None:
            directory = f"{user_data_dir()}/autometrics/datasets/{self.name}"

        train_dataset = self.load_from_file(f"{directory}/train_kfold.csv")
        
        # Check if test dataset file exists
        test_dataset = None
        test_file_path = f"{directory}/test_kfold.csv"
        if os.path.exists(test_file_path):
            test_dataset = self.load_from_file(test_file_path)

        # Count only CSV files that match the split pattern, ignore metadata files
        split_files = [f for f in os.listdir(directory) if f.startswith("split_") and f.endswith(".csv")]
        # Each fold has 2 files (train and val), so k = number of unique split indices
        split_indices = set()
        for f in split_files:
            # Extract the split index (e.g., "split_0_train.csv" -> 0)
            parts = f.replace(".csv", "").split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                split_indices.add(int(parts[1]))
        
        k = len(split_indices)

        split_datasets = []
        for i in range(k):
            split_train_dataset = self.load_from_file(f"{directory}/split_{i}_train.csv")
            split_val_dataset = self.load_from_file(f"{directory}/split_{i}_val.csv")
            split_datasets.append((split_train_dataset, split_val_dataset))

        return split_datasets, train_dataset, test_dataset
    
    def save_to_file(self, path: str):
        self.dataframe.to_csv(path, index=False)
    
    def load_from_file(self, path: str):
        df = pd.read_csv(path)
        return Dataset(
            dataframe=df,
            target_columns=self.target_columns,
            ignore_columns=self.ignore_columns,
            metric_columns=self.metric_columns,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns,
            metrics=self.metrics,
            task_description=self.task_description
        )
    
    def validate_permanent_splits_consistency(self, directory: Optional[str] = None) -> bool:
        """
        Validate that test.csv and test_kfold.csv contain identical data.
        
        Args:
            directory: Directory containing the split files
            
        Returns:
            True if test sets match, False otherwise
        """
        if directory is None:
            directory = f"{user_data_dir()}/autometrics/datasets/{self.name}"
        
        test_path = f"{directory}/test.csv"
        test_kfold_path = f"{directory}/test_kfold.csv"
        
        # Check if both files exist
        if not os.path.exists(test_path) or not os.path.exists(test_kfold_path):
            return False
        
        # Load both test datasets
        test_df = pd.read_csv(test_path)
        test_kfold_df = pd.read_csv(test_kfold_path)
        
        # Check if they have the same shape
        if test_df.shape != test_kfold_df.shape:
            return False
        
        # Sort both dataframes to ensure consistent ordering
        test_df_sorted = test_df.sort_values(by=list(test_df.columns)).reset_index(drop=True)
        test_kfold_df_sorted = test_kfold_df.sort_values(by=list(test_kfold_df.columns)).reset_index(drop=True)
        
        # Check if the sorted dataframes are identical
        return test_df_sorted.equals(test_kfold_df_sorted)
    
    def create_and_validate_permanent_splits(self, split_column: str, train_ratio: float = 0.5, val_ratio: float = 0.2, k: int = 5, seed: Optional[int] = None, max_size: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Create both regular and k-fold permanent splits and validate consistency.
        
        Args:
            split_column: Column to use for splitting
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            k: Number of folds for k-fold splits
            seed: Random seed
            max_size: Maximum size for subsets
            
        Returns:
            Tuple of (success: bool, results: Dict)
        """
        results = {
            'splits_created': False,
            'kfold_created': False,
            'test_sets_match': False,
            'errors': []
        }
        
        # Create regular splits
        train_ds, val_ds, test_ds = self.save_permanent_splits(
            split_column=split_column,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            max_size=max_size
        )
        results['splits_created'] = True
        
        # Create k-fold splits
        test_ratio = 1.0 - train_ratio - val_ratio
        split_datasets, train_dataset, test_dataset = self.save_permanent_kfold_splits(
            k=k,
            split_column=split_column,
            seed=seed,
            test_ratio=test_ratio,
            max_size=max_size
        )
        results['kfold_created'] = True
        
        # Validate test sets match
        results['test_sets_match'] = self.validate_permanent_splits_consistency()
        
        return True, results

class DummyDataset(Dataset):
    """
    A dummy dataset for simple purposes.  Not to be used for actual metric generation/recommendation.
    """
    def __init__(self, task_description: str):
        super().__init__(dataframe=pd.DataFrame(), target_columns=[], ignore_columns=[], metric_columns=[], name="DummyDataset", data_id_column=None, model_id_column=None, input_column=None, output_column=None, reference_columns=None, metrics=[], task_description=task_description)