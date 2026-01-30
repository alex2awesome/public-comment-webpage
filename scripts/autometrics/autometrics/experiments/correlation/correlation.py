from autometrics.experiments.experiment import Experiment
from autometrics.experiments.results import TabularResult, FigureResult, PlotlyResult
from typing import Callable, Optional, List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from math import ceil

# Import common correlation functions for easy referencing in main()
try:
    from scipy.stats import pearsonr, spearmanr, kendalltau
except ImportError:
    # Fallback definitions to avoid runtime errors if scipy is missing.
    # These are *very* naive and should be replaced by proper SciPy versions.
    def _naive_corr(x, y):
        return np.corrcoef(x, y)[0, 1], np.nan
    pearsonr = _naive_corr
    spearmanr = _naive_corr
    kendalltau = _naive_corr

# Helper to map string name to scipy correlation function
def correlation_func_from_name(name: str):
    name = name.lower()
    if name.startswith("pearson"):
        return pearsonr
    if name.startswith("spearman"):
        return spearmanr
    if name.startswith("kendall") or name.startswith("tau"):
        return kendalltau
    raise ValueError(f"Unknown correlation function '{name}'. Supported: pearson, spearman, kendall.")

class CorrelationExperiment(Experiment):
    """
    Experiment to measure correlation between a suite of metrics and one or more
    target columns provided by a dataset.
    
    The experiment ranks metrics by correlation on the validation split, then
    optionally evaluates (and plots) on the test split if it exists.  If no test
    split is available, it falls back to the train split for final evaluation.
    
    The primary results include:
      • A CSV per target column with correlation coefficient and p-value for every
        metric (sorted by |correlation| desc).
      • Matplotlib & Plotly scatterplots for the top-k metrics per target column.
    """

    def __init__(
        self,
        name: str,
        description: str,
        metrics: List['Metric'],
        output_dir: str,
        dataset: 'Dataset',
        correlation_funcs = None,
        top_k: Optional[int] = None,
        seed: int = 42,
        should_split: bool = True,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            metrics=metrics,
            output_dir=output_dir,
            dataset=dataset,
            seed=seed,
            should_split=should_split,
            **kwargs,
        )
        # Normalize correlation_funcs parameter
        self.correlation_funcs = self._normalize_corr_funcs(correlation_funcs)
        self.top_k = top_k

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    def _normalize_corr_funcs(self, corr_spec):
        """Return an ordered dict mapping name->function from input spec."""
        from collections import OrderedDict
        # Default to Pearson
        if corr_spec is None:
            corr_spec = {'pearson': pearsonr}
        # If single callable provided in legacy param position
        if callable(corr_spec):
            return OrderedDict({getattr(corr_spec, '__name__', 'corr'): corr_spec})
        # If list or tuple
        if isinstance(corr_spec, (list, tuple)):
            mapping = OrderedDict()
            for item in corr_spec:
                if callable(item):
                    mapping[getattr(item, '__name__', 'corr')] = item
                elif isinstance(item, str):
                    mapping[item.lower()] = correlation_func_from_name(item)
            return mapping
        # If dict
        if isinstance(corr_spec, dict):
            return OrderedDict(corr_spec)
        raise ValueError("Unsupported correlation_funcs specification")

    def _ensure_metric_columns(self, dataset, verbose=False):
        """Run each metric's prediction method so that its columns exist."""
        for metric in self.metrics:
            if verbose:
                print(f"Predicting {metric.get_name()} on {dataset.get_name()}")
            metric.predict(dataset, update_dataset=True)

    def _gather_metric_columns(self, dataset) -> List[str]:
        """Return all metric column names corresponding to `self.metrics`."""
        metric_cols: List[str] = []
        metric_classes: List[str] = []
        for metric in self.metrics:
            if hasattr(metric, 'get_submetric_names'):
                submetric_names = metric.get_submetric_names()
                num_submetrics = len(submetric_names)
                metric_cols += list(submetric_names)
                metric_classes.extend([type(metric).__name__] * num_submetrics)
            else:
                metric_classes.append(type(metric).__name__)
                metric_cols.append(metric.get_name())
        # Keep only columns that actually exist in the DF (some metrics may have errored)
        existing_cols = [c for c in metric_cols if c in dataset.get_dataframe().columns]
        return existing_cols, metric_classes

    def _compute_correlations(self, dataset, corr_func) -> Dict[str, pd.DataFrame]:
        """Compute correlation & p-value for every metric vs each target column."""
        df = dataset.get_dataframe()
        metric_cols, metric_classes = self._gather_metric_columns(dataset)

        # Attempt to fetch target columns attribute / fallback
        if hasattr(dataset, 'target_columns'):
            target_cols = list(dataset.target_columns)
        elif hasattr(dataset, 'get_target_columns'):
            target_cols = list(dataset.get_target_columns())
        else:
            raise AttributeError("Dataset object must expose `target_columns` or `get_target_columns()`.")

        correlations: Dict[str, pd.DataFrame] = {}
        for target_col in target_cols:
            rows = []
            y = df[target_col]
            for metric_col, metric_class in zip(metric_cols, metric_classes):
                x = df[metric_col]
                mask = x.notna() & y.notna()
                if mask.sum() < 2:
                    # Not enough data points to compute correlation
                    corr_val, p_val = np.nan, np.nan
                else:
                    try:
                        res = corr_func(x[mask], y[mask])
                        # Handle different SciPy versions (tuple vs. object)
                        if isinstance(res, tuple):
                            corr_val, p_val = res[0], res[1]
                        elif hasattr(res, 'statistic') and hasattr(res, 'pvalue'):
                            corr_val, p_val = res.statistic, res.pvalue
                        elif hasattr(res, 'correlation') and hasattr(res, 'pvalue'):
                            corr_val, p_val = res.correlation, res.pvalue
                        else:
                            corr_val, p_val = np.nan, np.nan
                    except Exception:
                        corr_val, p_val = np.nan, np.nan
                rows.append({
                    'Metric': metric_col,
                    'Metric_Class': metric_class,
                    'Correlation': corr_val,
                    'P-value': p_val,
                })
            df_corr = pd.DataFrame(rows)
            # Sort by absolute correlation value (descending)
            df_corr = df_corr.sort_values(by='Correlation', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
            correlations[target_col] = df_corr
        return correlations

    def _scale_to_target_range(self, metric_series: pd.Series, target_min: float, target_max: float) -> pd.Series:
        """Min-max scale a metric series to lie within the [target_min, target_max] range."""
        # Coerce boolean targets to numeric bounds [0.0, 1.0] for arithmetic
        if isinstance(target_min, (bool, np.bool_)) or isinstance(target_max, (bool, np.bool_)):
            target_min = float(target_min)
            target_max = float(target_max)
        # If metric series is boolean, cast to float for scaling
        if pd.api.types.is_bool_dtype(metric_series):
            metric_series = metric_series.astype(float)
        metric_min, metric_max = metric_series.min(), metric_series.max()
        # Handle constant or NaN series gracefully
        if pd.isna(metric_min) or pd.isna(metric_max):
            return pd.Series([np.nan] * len(metric_series), index=metric_series.index)
        if metric_max == metric_min:
            mid = (target_min + target_max) / 2.0
            return pd.Series([mid] * len(metric_series), index=metric_series.index)
        scaled = (metric_series - metric_min) / (metric_max - metric_min)
        return scaled * (target_max - target_min) + target_min

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _create_scatterplot_mpl(self, df: pd.DataFrame, metric_cols: List[str], target_col: str):
        fig, ax = plt.subplots(figsize=(8, 6))
        tgt_min, tgt_max = df[target_col].min(), df[target_col].max()
        for metric_col in metric_cols:
            scaled_x = self._scale_to_target_range(df[metric_col], tgt_min, tgt_max)
            ax.scatter(scaled_x, df[target_col], label=metric_col, alpha=0.6)
        ax.set_xlabel('Metric value (min-max scaled)', fontsize=14)
        ax.set_ylabel(target_col, fontsize=14)
        ax.set_title(f'{target_col} vs. Top Metrics', fontsize=16)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig

    def _create_scatterplot_plotly(self, df: pd.DataFrame, metric_cols: List[str], target_col: str):
        fig = go.Figure()
        tgt_min, tgt_max = df[target_col].min(), df[target_col].max()
        for metric_col in metric_cols:
            scaled_x = self._scale_to_target_range(df[metric_col], tgt_min, tgt_max)
            fig.add_trace(
                go.Scatter(
                    x=scaled_x,
                    y=df[target_col],
                    mode='markers',
                    name=metric_col,
                    hovertemplate=f'{metric_col}: %{{x}}<br>{target_col}: %{{y}}<extra></extra>',
                )
            )
        fig.update_layout(
            title=f'{target_col} vs. Top Metrics',
            xaxis_title='Metric value (min-max scaled)',
            yaxis_title=target_col,
            height=600,
        )
        return fig

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def run(self, print_results: bool = False):
        # ------------------------------------------------------------------
        # 1) Ensure metric predictions exist on splits we care about
        # ------------------------------------------------------------------
        self._ensure_metric_columns(self.val_dataset, verbose=True)

        # Decide which split is our *evaluation* set (test preferred, else train)
        evaluation_dataset = self.test_dataset if self.test_dataset is not None and self.test_dataset is not self.train_dataset else self.train_dataset
        self._ensure_metric_columns(evaluation_dataset, verbose=True)

        # ------------------------------------------------------------------
        # 2) Compute correlations on validation & evaluation sets
        # ------------------------------------------------------------------
        all_val_corrs = {}
        all_eval_corrs = {}

        for name, func in self.correlation_funcs.items():
            all_val_corrs[name] = self._compute_correlations(self.val_dataset, func)
            all_eval_corrs[name] = self._compute_correlations(evaluation_dataset, func)

        # Use the first correlation metric as reference for scatterplot ranking
        primary_corr_name = next(iter(self.correlation_funcs.keys()))
        val_corrs_primary = all_val_corrs[primary_corr_name]
        eval_corrs_primary = all_eval_corrs[primary_corr_name]

        # ------------------------------------------------------------------
        # 3) Persist results & optionally print
        # ------------------------------------------------------------------
        for corr_name, eval_corrs in all_eval_corrs.items():
            val_corrs_cur = all_val_corrs[corr_name]

            for target_col, df_eval in eval_corrs.items():
                # Save correlation table (CSV) into subfolder
                key_csv = f"{corr_name}/correlation_{target_col}"
                self.results[key_csv] = TabularResult(df_eval)

                if print_results:
                    print(f"\n=== {corr_name.title()} Correlation results for target: {target_col} ===")
                    print(df_eval.head(10))

                # Determine ranking for scatterplot based on this correlation type
                df_val_ranked = val_corrs_cur[target_col]
                k = self.top_k if (self.top_k is not None and self.top_k > 0) else len(df_val_ranked)
                top_metric_cols = df_val_ranked['Metric'].iloc[:k].tolist()

                # Create scatterplots on evaluation set
                df_full = evaluation_dataset.get_dataframe()
                mpl_fig = self._create_scatterplot_mpl(df_full, top_metric_cols, target_col)
                plotly_fig = self._create_scatterplot_plotly(df_full, top_metric_cols, target_col)

                self.results[f"{corr_name}/scatter_{target_col}_mpl"] = FigureResult(mpl_fig)
                self.results[f"{corr_name}/scatter_{target_col}_plotly"] = PlotlyResult(plotly_fig)

                # Raw data for scatter
                tgt_min, tgt_max = df_full[target_col].min(), df_full[target_col].max()
                records = []
                for metric_col in top_metric_cols:
                    scaled = self._scale_to_target_range(df_full[metric_col], tgt_min, tgt_max)
                    for idx, (orig_val, scaled_val, tgt) in enumerate(zip(df_full[metric_col], scaled, df_full[target_col])):
                        records.append({
                            'Index': idx,
                            'Metric': metric_col,
                            'Metric_Original': orig_val,
                            'Metric_Scaled': scaled_val,
                            target_col: tgt,
                        })
                scatter_df = pd.DataFrame(records)
                self.results[f"{corr_name}/scatter_data_{target_col}"] = TabularResult(scatter_df)

        return all_eval_corrs


# ----------------------------------------------------------------------
# Example usage (mirrors timing experiment style)
# ----------------------------------------------------------------------

def main():
    """Example invocation of CorrelationExperiment over SummEval."""
    try:
        from autometrics.dataset.datasets.summeval.summeval import SummEval
        dataset = SummEval()
    except ImportError:
        raise RuntimeError("SummEval dataset could not be imported. Ensure appropriate dataset package is installed.")

    # Retrieve a small subset of metrics for a quick demo
    from autometrics.metrics.reference_based.ROUGE import ROUGE
    from autometrics.metrics.reference_based.BLEU import BLEU
    metrics = [ROUGE(), BLEU()]

    experiment = CorrelationExperiment(
        name="Metric-Target Correlation Experiment",
        description="Evaluates correlation between metrics and SummEval human scores.",
        metrics=metrics,
        output_dir="outputs/correlation",
        dataset=dataset,
        correlation_funcs=None,
        top_k=5,
    )

    experiment.run(print_results=True)
    experiment.save_results()
    print("Correlation experiment completed!")

if __name__ == "__main__":
    main()
