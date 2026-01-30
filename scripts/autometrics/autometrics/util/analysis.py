from autometrics.evaluate.correlation import calculate_correlation
import pandas as pd

def abbreviate_metric_name(metric_name):
    # Define abbreviations for common long parts of the names
    return metric_name.replace("inc_plus_omi_", "ipo_").replace("predictions_", "pred_").replace("ElasticNet", "ENet").replace("GradientBoosting", "GB").replace("Ridge", "Rg").replace("Lasso", "L").replace("PLS", "PLS").replace("Meta-Llama-3-70b-Instruct", "llama70b")

def display_top_5_metrics_by_validation_precomputed(validation_data, test_data):
    top_correlations = {}

    # Iterate over each target category (time_sec, inc_plus_omi, etc.)
    for target_column, val_data in validation_data.items():
        # Sort validation correlations and get top 5
        sorted_val_correlations = sorted(val_data.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Prepare row for this target
        top_correlations[target_column] = []
        
        for metric, val_corr in sorted_val_correlations:
            test_corr = test_data.get(target_column, {}).get(metric, "N/A")
            metric_abbr = abbreviate_metric_name(metric)
            # Format as (metric_abbr, val_corr, test_corr)
            top_correlations[target_column].append(f"{metric_abbr} ({test_corr})")

    num_cols = min(5, max(len(v) for v in top_correlations.values()))

    # Create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(top_correlations, orient='index', columns=[f'Top {i+1} Metric & Value' for i in range(num_cols)])
    
    return df

def display_top_5_metrics_by_validation(validation_dataset, test_dataset, compute_all=False, func=calculate_correlation):
    """
    Returns dataframe of the top 5 metrics by validation score

    Parameters:
    -----------
    validation_dataset : Dataset
        The dataset object containing the validation data.
    test_dataset : Dataset
        The dataset object containing the test data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the top 5 metrics by validation score for each target category, along with the test score.
    """

    validation_data = func(validation_dataset, compute_all=compute_all)
    test_data = func(test_dataset, compute_all=compute_all)

    return display_top_5_metrics_by_validation_precomputed(validation_data, test_data)
    
def get_top_metric_by_validation_precomputed(validation_data, target_column=None, k=1):
    """
    Returns the top metric(s) by validation score for a specified target column or all target columns if None.

    Parameters:
    -----------
    validation_data : dict
        A dictionary containing the validation data for each target column.
    target_column : str or None
        The target column to get the top metric for. If None, the top metric for all target columns will be returned.
    k : int
        The number of top metrics to return for each target column. Default is 1.
    
    Returns:
    --------
    metric or dict
        The top metric(s) by validation score for the specified target column, or a dictionary of top metrics for all target columns.
    If k > 1, a list of top metrics will be returned for each target column.
    If k = 1, a single metric will be returned for each target column.
    If target_column is None, a dictionary of top metrics for all target columns will be returned.
    If target_column is specified, a single metric will be returned for that target column.
    If no metrics are found, None will be returned.
    """

    top_correlations = {}

    # Iterate over each target category (time_sec, inc_plus_omi, etc.)
    targets = validation_data.keys() if target_column is None else [target_column]
    for target_column_ in targets:
        val_data = validation_data[target_column_]
        # Sort validation correlations and get top 5
        best_val_correlation = sorted(val_data.items(), key=lambda x: abs(x[1]), reverse=True)[0:k]

        metric_list = [metric for metric, _ in best_val_correlation]

        if k == 1:
            # If k=1, just take the best metric
            metric = metric_list[0]
        else:
            # If k>1, take the top k metrics
            metric = metric_list
        
        # Format as (metric_abbr, val_corr, test_corr)
        top_correlations[target_column_] = metric

    return top_correlations if target_column is None else top_correlations[target_column]

def get_top_metric_by_validation(validation_dataset, target_column=None, compute_all=False, k=1):
    """
    Returns the top metric itself by validation score, not a dataframe

    Parameters:
    -----------
    validation_dataset : Dataset
        The dataset object containing the validation data.
    test_dataset : Dataset
        The dataset object containing the test data.
    compute_all : bool
        Whether to compute correlations with all columns or just the dedicated metric columns from the dataset. Default is False.
    target_column : str
        The target column to get the top metric for. If None, the top metric for all target columns will be returned.
    k : int
        The number of top metrics to return for each target column. Default is 1.

    Returns:
    --------
    metric or dict
        The top metric by validation score for the specified target column, or a dictionary of top metrics for all target columns.
    """
    
    validation_data = calculate_correlation(validation_dataset, compute_all=compute_all)

    return get_top_metric_by_validation_precomputed(validation_data, target_column, k=k)


def plot_metric_target_scatterplot(dataset, metric_column, target_column):
    """
    Plot a scatterplot of a metric against a target column

    Parameters:
    -----------
    dataset : Dataset
        The dataset object containing the data.
    metric_column : str
        The column name of the metric to plot.
    target_column : str
        The column name of the target to plot.
    """
    df = dataset.get_dataframe()
    df.plot.scatter(x=metric_column, y=target_column, title=f"{metric_column} vs {target_column}")