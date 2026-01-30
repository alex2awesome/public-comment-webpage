from autometrics.dataset import Dataset
import numpy as np

def calculate_accuracy(
    dataset: Dataset,
    group_by: str = None,
    compute_all: bool = False,
    post_processing=lambda x: np.round(x)
) -> dict:
    """
    Calculate the Exact Match Accuracy for the given dataset after optional post-processing.

    Parameters
    ----------
    dataset : Dataset
        The dataset object containing the data to be analyzed.
        It should have methods to get the dataframe, target columns,
        ignore columns, and metric columns.
    group_by : str, optional
        The column name by which to group the data before calculating accuracy.
        Default is None.
    compute_all : bool, optional
        If True, compute accuracy for all columns not in ignore/target. If False,
        use the metric columns defined by the dataset. Default is False.
    post_processing : callable, optional
        A function to apply to both metric and target values before accuracy computation.
        Defaults to rounding to the nearest integer.

    Returns
    -------
    dict
        A dictionary containing the accuracy for each target column and metric column.
        
        If `group_by` is specified, the dictionary will contain the average accuracy
        for each target column and metric column, grouped by the specified column.
        
        If `group_by` is not specified, the dictionary will contain the accuracy
        for each target column and metric column for the entire dataset.
        
        Structure:
            {
                target_column_1: {
                    metric_column_1: accuracy_value,
                    metric_column_2: accuracy_value,
                    ...
                },
                target_column_2: {
                    metric_column_1: accuracy_value,
                    metric_column_2: accuracy_value,
                    ...
                },
                ...
            }
    """

    df = dataset.get_dataframe()
    target_columns = dataset.get_target_columns()
    ignore_columns = dataset.get_ignore_columns()

    if compute_all:
        metric_columns = [col for col in df.columns if col not in ignore_columns and col not in target_columns]
    else:
        metric_columns = dataset.get_metric_columns()

    if metric_columns is None:
        metric_columns = [col for col in df.columns if col not in ignore_columns and col not in target_columns]

    # Helper function to compute accuracy after post-processing
    def compute_accuracy_for_columns(df_slice, target_col, metric_col):
        target_vals = post_processing(df_slice[target_col].values)
        metric_vals = post_processing(df_slice[metric_col].values)
        return np.mean(target_vals == metric_vals)

    if group_by:
        # Group by the specified column and calculate average accuracy
        ignore_columns_minus_group_id = [col for col in ignore_columns if col != group_by]
        grouped_df = df.drop(columns=ignore_columns_minus_group_id).groupby([group_by])

        accuracy_grouped = {}
        for target_column in target_columns:
            accuracy_grouped[target_column] = {}
            for metric_column in metric_columns:
                # Compute accuracy per group and average
                accuracy_per_group = grouped_df.apply(lambda x: compute_accuracy_for_columns(x, target_column, metric_column))
                accuracy_grouped[target_column][metric_column] = accuracy_per_group.mean()

        return accuracy_grouped

    else:
        # Compute accuracy for the entire dataset
        accuracy_all = {}
        for target_column in target_columns:
            accuracy_all[target_column] = {}
            for metric_column in metric_columns:
                accuracy_all[target_column][metric_column] = compute_accuracy_for_columns(df, target_column, metric_column)

        return accuracy_all