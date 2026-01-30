from autometrics.dataset import Dataset
from scipy.stats import spearmanr

def calculate_correlation(dataset: Dataset, group_by: str = None, correlation=spearmanr, compute_all: bool = False) -> dict:
    '''
    Calculate the correlation for the given dataset.

    Parameters:
    -----------
    dataset : Dataset
        The dataset object containing the data to be analyzed. It should have methods to get the dataframe, target columns, ignore columns, and metric columns.
    group_by : str, optional
        The column name by which to group the data before calculating the Spearman correlation. Default is 'None'.
    correlation : function, optional
        The correlation function to use. Default is `spearmanr` from `scipy.stats`.

    Returns:
    --------
    dict
        A dictionary containing the Spearman correlation coefficients. If `group_by` is specified, the dictionary will contain the average Spearman correlation for each target column and metric column, grouped by the specified column. If `group_by` is not specified, the dictionary will contain the Spearman correlation for each target column and metric column for the entire dataset.
        Dictionary structure:
            {
                target_column_1: {
                    metric_column_1: correlation_1,
                    metric_column_2: correlation_2,
                    ...
                },
                target_column_2: {
                    metric_column_1: correlation_1,
                    metric_column_2: correlation_2,
                    ...
                },
                ...
            }
    '''
    df = dataset.get_dataframe().copy()
    # Determine target columns from dataset
    target_columns = dataset.get_target_columns()
    ignore_columns = dataset.get_ignore_columns()
    if compute_all:
        metric_columns = [col for col in df.columns if col not in ignore_columns and col not in target_columns]
    else:
        metric_columns = dataset.get_metric_columns()
        if metric_columns is None:
            metric_columns = [col for col in df.columns if col not in ignore_columns and col not in target_columns]

    # Filter to only columns that actually exist in the DataFrame to avoid KeyError
    metric_columns = [col for col in metric_columns if col in df.columns]
    # Do not drop across all metrics. We'll drop pairwise per (target, metric).

    if group_by:
        # group by the column 'group_by' and calculate the average spearman correlation for each target_column and metric_column.
        ignore_columns_minus_group_id = [col for col in ignore_columns if col != group_by]
        grouped_df = df.drop(columns=ignore_columns_minus_group_id).groupby([group_by])
        correlations_grouped = {}
        for target_column in target_columns:
            correlations_grouped[target_column] = {}
            for metric_column in metric_columns:
                r_values = []
                for _, x in grouped_df:
                    pair = x[[target_column, metric_column]].dropna()
                    if len(pair) < 2:
                        continue
                    try:
                        r = correlation(pair[target_column], pair[metric_column])[0]
                        r_values.append(r)
                    except Exception:
                        continue
                correlations_grouped[target_column][metric_column] = (sum(r_values) / len(r_values)) if r_values else None

        return correlations_grouped

    else:
        # for each of the target_columns, calculate correlation with each metric using pairwise valid rows
        correlations_all = {}
        for target_column in target_columns:
            correlations_all[target_column] = {}
            for metric_column in metric_columns:
                pair = df[[target_column, metric_column]].dropna()
                if len(pair) < 2:
                    correlations_all[target_column][metric_column] = None
                    continue
                try:
                    correlations_all[target_column][metric_column] = correlation(pair[target_column], pair[metric_column])[0]
                except Exception:
                    correlations_all[target_column][metric_column] = None

        return correlations_all


def calculate_correlation_with_p_val(dataset: Dataset, group_by: str = None, correlation=spearmanr, compute_all: bool = False) -> dict:
    """
    Calculate correlation and p-value for the given dataset using the provided
    correlation function (e.g., scipy.stats.spearmanr/pearsonr/kendalltau).

    Returns a nested dictionary with the same keys as calculate_correlation, but
    mapping each metric to a dict containing both correlation and p-value:

        {
            target_column: {
                metric_column: {"correlation": r_value, "p_value": p_value},
                ...
            },
            ...
        }

    For grouped calculations, the function averages the per-group correlation and
    p-value across groups (only over groups where the correlation function returns
    valid results).
    """
    df = dataset.get_dataframe().copy()
    # Determine target columns from dataset
    target_columns = dataset.get_target_columns()
    ignore_columns = dataset.get_ignore_columns()
    if compute_all:
        metric_columns = [col for col in df.columns if col not in ignore_columns and col not in target_columns]
    else:
        metric_columns = dataset.get_metric_columns()
        if metric_columns is None:
            metric_columns = [col for col in df.columns if col not in ignore_columns and col not in target_columns]

    metric_columns = [col for col in metric_columns if col in df.columns]

    # Only drop NaNs from targets here; per-metric NaNs will be handled pairwise below
    if target_columns:
        df = df.dropna(subset=[c for c in target_columns if c in df.columns])

    if group_by:
        ignore_columns_minus_group_id = [col for col in ignore_columns if col != group_by]
        grouped_df = df.drop(columns=ignore_columns_minus_group_id).groupby([group_by])
        correlations_grouped = {}
        for target_column in target_columns:
            correlations_grouped[target_column] = {}
            for metric_column in metric_columns:
                r_values = []
                p_values = []
                for _, x in grouped_df:
                    pair = x[[target_column, metric_column]].dropna()
                    if len(pair) < 2:
                        continue
                    try:
                        r, p = correlation(pair[target_column], pair[metric_column])
                        if r is not None and p is not None:
                            r_values.append(r)
                            p_values.append(p)
                    except Exception:
                        continue
                avg_r = (sum(r_values) / len(r_values)) if r_values else None
                avg_p = (sum(p_values) / len(p_values)) if p_values else None
                correlations_grouped[target_column][metric_column] = {
                    "correlation": avg_r,
                    "p_value": avg_p,
                }

        return correlations_grouped

    else:
        correlations_all = {}
        for target_column in target_columns:
            correlations_all[target_column] = {}
            for metric_column in metric_columns:
                # Guard: ensure at least 2 valid pairs before calling scipy
                try:
                    pair_df = df[[target_column, metric_column]].dropna()
                    if len(pair_df) < 2:
                        print(f"⚠️ correlation_with_p_val: insufficient pairs (<2) for target='{target_column}', metric='{metric_column}'. Returning (0.0, None)")
                        r, p = 0.0, None
                    else:
                        r, p = correlation(pair_df[target_column], pair_df[metric_column])
                except Exception as e:
                    print(f"⚠️ correlation_with_p_val: error computing correlation for target='{target_column}', metric='{metric_column}': {e}")
                    r, p = 0.0, None
                correlations_all[target_column][metric_column] = {
                    "correlation": r,
                    "p_value": p,
                }

        return correlations_all
