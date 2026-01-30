# File: timing/timing.py

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.ticker as mticker
from scipy import stats
import os

from autometrics.experiments.experiment import Experiment
from autometrics.metrics.Metric import Metric
from autometrics.experiments.results import TabularResult, FigureResult, PlotlyResult

class TimingExperiment(Experiment):
    """
    Experiment to measure the execution time of metrics.
    """
    
    def __init__(self, name, description, metrics, output_dir, dataset, trials=1, burnin=5, seed=42, **kwargs):
        """
        Initialize the timing experiment.
        
        Args:
            name: Name of the experiment
            description: Description of the experiment
            metrics: List of metrics to evaluate
            output_dir: Output directory for results
            dataset: Dataset to use for evaluation
            trials: Number of repetitions per metric (default: 1)
            burnin: Number of burn-in samples to run before timing (default: 5)
            seed: Random seed for reproducibility
            **kwargs: Additional arguments
        """
        super().__init__(
            name=name,
            description=description,
            metrics=metrics,
            output_dir=output_dir,
            dataset=dataset,
            seed=seed,
            should_split=False,  # We don't need to split for timing
            **kwargs
        )
        self.trials = trials
        self.burnin = burnin
        
    def _time_metric(self, metric, inputs, outputs, references=None, max_samples=None):
        """
        Time the execution of a metric on the given inputs and outputs.
        
        Args:
            metric: The metric to time
            inputs: List of inputs
            outputs: List of outputs
            references: List of references (optional)
            max_samples: Maximum number of samples to use (optional)
            
        Returns:
            List of execution times in milliseconds per sample
        """
        if max_samples and max_samples < len(inputs):
            inputs = inputs[:max_samples]
            outputs = outputs[:max_samples]
            if references:
                references = references[:max_samples]
                
        # First run burn-in samples to warm up any JIT compilation, etc.
        if self.burnin > 0:
            print(f"Running {self.burnin} burn-in samples for {metric.get_name()}...")
            for i in range(min(self.burnin, len(inputs))):
                if references:
                    # Pass references correctly - they're already structured properly
                    metric.calculate(inputs[i], outputs[i], references[i])
                else:
                    metric.calculate(inputs[i], outputs[i])
        
        # Time each sample individually
        times_ms = []
        
        print(f"Timing {len(inputs)} samples for {metric.get_name()}...")
        for i in range(len(inputs)):
            start_time = time.perf_counter()
            
            if references:
                # Pass references correctly
                metric.calculate(inputs[i], outputs[i], references[i])
            else:
                metric.calculate(inputs[i], outputs[i])
                
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            times_ms.append(execution_time_ms)
            
        return times_ms
    
    def _calculate_statistics(self, times_ms):
        """
        Calculate statistics for the timing results.
        
        Args:
            times_ms: List of execution times in milliseconds
            
        Returns:
            Dictionary with statistics (mean, std, ci_lower, ci_upper)
        """
        times_array = np.array(times_ms)
        mean_time = np.mean(times_array)
        std_time = np.std(times_array)
        
        # Calculate 95% confidence interval
        n = len(times_array)
        sem = std_time / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n-1, loc=mean_time, scale=sem)
        
        return {
            'mean': mean_time,
            'std': std_time,
            'ci_lower': ci_95[0],
            'ci_upper': ci_95[1],
            'min': np.min(times_array),
            'max': np.max(times_array),
            'median': np.median(times_array),
            'q1': np.percentile(times_array, 25),
            'q3': np.percentile(times_array, 75)
        }
    
    def _filter_outliers(self, data, extreme_only=True):
        """
        Filter extreme outliers from timing data.
        
        Args:
            data: Dictionary mapping metric names to timing arrays
            extreme_only: If True, only filter extreme outliers (>3*IQR from median)
                         If False, filter outliers (>1.5*IQR from median)
                         
        Returns:
            Dictionary with filtered timing data and count of outliers removed
        """
        filtered_data = {}
        outlier_counts = {}
        
        # Multiplier for IQR - 3.0 for extreme outliers, 1.5 for regular outliers
        k = 3.0 if extreme_only else 1.5
        
        for metric_name, times in data.items():
            q1 = np.percentile(times, 25)
            q3 = np.percentile(times, 75)
            iqr = q3 - q1
            
            # Define bounds for outliers
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            
            # Filter outliers
            filtered_times = [t for t in times if lower_bound <= t <= upper_bound]
            
            # Count removed outliers
            removed = len(times) - len(filtered_times)
            outlier_counts[metric_name] = removed
            
            # Store filtered data
            filtered_data[metric_name] = filtered_times
            
            if removed > 0:
                print(f"Removed {removed} {'extreme ' if extreme_only else ''}outliers from {metric_name} timing data.")
        
        return filtered_data, outlier_counts
    
    def _create_boxplot(self, timing_data):
        """
        Create a box plot of timing results using matplotlib.
        
        Args:
            timing_data: Dictionary mapping metric names to timing arrays
            
        Returns:
            Matplotlib figure
        """
        # Filter extreme outliers for visualization
        filtered_data, _ = self._filter_outliers(timing_data, extreme_only=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for boxplot
        data = []
        labels = []
        
        for metric_name, times in filtered_data.items():
            data.append(times)
            labels.append(metric_name)
        
        # Create boxplot
        bp = ax.boxplot(data, patch_artist=True, vert=True)
        
        # Customize boxplot appearance
        for box in bp['boxes']:
            box.set(facecolor='lightblue', alpha=0.8)
        
        # Set labels and title with larger font sizes
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
        ax.set_ylabel('Time per sample (ms)', fontsize=14)
        ax.set_title('Metric Execution Time', fontsize=18, pad=20)
        
        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Format y-axis to show milliseconds with 2 decimal places
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f ms'))
        
        # Add grid lines for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def _create_plotly_boxplot(self, timing_data):
        """
        Create an interactive box plot using plotly.
        
        Args:
            timing_data: Dictionary mapping metric names to timing arrays
            
        Returns:
            Plotly figure
        """
        # Filter extreme outliers for visualization
        filtered_data, outlier_counts = self._filter_outliers(timing_data, extreme_only=True)
        
        # Create a list for each metric's data, preserving order
        data = []
        for metric_name, times in filtered_data.items():
            tooltip = f"{metric_name}"
            if outlier_counts.get(metric_name, 0) > 0:
                tooltip += f" (removed {outlier_counts[metric_name]} extreme outliers)"
                
            data.append(
                go.Box(
                    y=times,
                    name=metric_name,  # This becomes the x-axis label
                    boxmean=True,      # Show mean as dashed line
                    marker_color='lightblue',
                    line_color='blue',
                    hovertext=tooltip
                )
            )
        
        # Create figure with simple box traces
        fig = go.Figure(data=data)
        
        # Update layout with minimal configuration
        fig.update_layout(
            title={
                'text': 'Metric Execution Time',
                'font': {'size': 24},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis_title={
                'text': 'Time per sample (ms)',
                'font': {'size': 18}
            },
            yaxis=dict(
                ticksuffix=' ms',
                tickfont={'size': 14},
            ),
            xaxis=dict(
                title='',  # No x-axis title needed as boxplot names serve as labels
                tickfont={'size': 16},
            ),
            height=700,
            margin=dict(t=100, b=100, l=100, r=50),
            font=dict(size=16),
        )
        
        # Add annotation about outlier filtering
        total_outliers = sum(outlier_counts.values())
        if total_outliers > 0:
            fig.add_annotation(
                text=f"Note: {total_outliers} extreme outliers filtered out (>3×IQR from median)",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=14),
                bordercolor="gray",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
                opacity=0.8
            )
        
        return fig
    
    def run(self, print_results=False, max_samples=30):
        """
        Run the timing experiment.
        
        Args:
            print_results: Whether to print results
            max_samples: Maximum number of samples to use from the dataset (default: 30)
                         (independent of trials/burnin which control repetitions)
        """
        # Store original cache states to restore them later
        original_cache_states = {}
        for metric in self.metrics:
            original_cache_states[metric] = metric.use_cache
            # Temporarily disable caching for timing measurements
            metric.use_cache = False
            
        try:
            test_dataset = self.test_dataset
            
            # If max_samples is specified, limit the dataset size
            if max_samples and max_samples < len(test_dataset.get_dataframe()):
                test_dataset = test_dataset.get_subset(max_samples, seed=self.seed)
            
            # Prepare inputs, outputs, and references
            inputs = test_dataset.get_dataframe()[test_dataset.get_input_column()].tolist()
            outputs = test_dataset.get_dataframe()[test_dataset.get_output_column()].tolist()
            
            # Check if there are reference columns
            reference_columns = test_dataset.get_reference_columns()
            references = None
            if reference_columns:
                # For each input/output pair, create a list of reference strings
                references = []
                for i in range(len(inputs)):
                    ref_list = []
                    for ref_col in reference_columns:
                        ref_value = test_dataset.get_dataframe()[ref_col].iloc[i]
                        if pd.notna(ref_value):  # Only add non-NA references
                            ref_list.append(ref_value)
                    references.append(ref_list)
            
            # Run timing for each metric
            timing_data = {}
            timing_stats = {}
            
            for metric in self.metrics:
                metric_name = metric.get_name()
                print(f"Evaluating timing for {metric_name}...")
                
                # Run the trials
                all_times = []
                for trial in range(self.trials):
                    print(f"  Trial {trial+1}/{self.trials}")
                    times = self._time_metric(
                        metric, inputs, outputs, references, max_samples)
                    all_times.extend(times)
                
                # Store raw timing data
                timing_data[metric_name] = all_times
                
            # Filter outliers for statistics calculation (keeping raw data for reference)
            filtered_data, outlier_counts = self._filter_outliers(timing_data, extreme_only=True)
                
            # Calculate statistics using filtered data
            for metric_name, times in filtered_data.items():
                stats = self._calculate_statistics(times)
                timing_stats[metric_name] = stats
                
                if print_results:
                    print(f"\nTiming results for {metric_name}:")
                    if outlier_counts.get(metric_name, 0) > 0:
                        print(f"  (Removed {outlier_counts[metric_name]} extreme outliers)")
                    print(f"  Mean time per sample: {stats['mean']:.2f} ms")
                    print(f"  95% CI: [{stats['ci_lower']:.2f}, {stats['ci_upper']:.2f}] ms")
                    print(f"  Min: {stats['min']:.2f} ms, Max: {stats['max']:.2f} ms")
                    print(f"  Median: {stats['median']:.2f} ms")
                    print()
            
            # Create summary dataframe
            summary_df = pd.DataFrame({
                'Metric': list(timing_stats.keys()),
                'Mean (ms)': [stats['mean'] for stats in timing_stats.values()],
                'Std Dev (ms)': [stats['std'] for stats in timing_stats.values()],
                'CI Lower (ms)': [stats['ci_lower'] for stats in timing_stats.values()],
                'CI Upper (ms)': [stats['ci_upper'] for stats in timing_stats.values()],
                'Min (ms)': [stats['min'] for stats in timing_stats.values()],
                'Max (ms)': [stats['max'] for stats in timing_stats.values()],
                'Median (ms)': [stats['median'] for stats in timing_stats.values()],
                'Q1 (ms)': [stats['q1'] for stats in timing_stats.values()],
                'Q3 (ms)': [stats['q3'] for stats in timing_stats.values()],
                'Outliers Removed': [outlier_counts.get(metric, 0) for metric in timing_stats.keys()]
            })
            
            # Save data and plots
            self.results['timing_summary'] = TabularResult(summary_df)
            
            # Create raw data dataframe (with all data, including outliers)
            raw_data = []
            for metric_name, times in timing_data.items():
                for t in times:
                    raw_data.append({
                        'Metric': metric_name,
                        'Time (ms)': t,
                        'Is Outlier': t not in filtered_data[metric_name]
                    })
            raw_df = pd.DataFrame(raw_data)
            self.results['timing_raw_data'] = TabularResult(raw_df)
            
            # Create and save plots (will use filtered data)
            mpl_fig = self._create_boxplot(timing_data)
            self.results['timing_boxplot'] = FigureResult(mpl_fig)
            
            plotly_fig = self._create_plotly_boxplot(timing_data)
            self.results['timing_boxplot_interactive'] = PlotlyResult(plotly_fig)
            
            return summary_df
        
        finally:
            # Restore original cache states
            for metric, original_state in original_cache_states.items():
                metric.use_cache = original_state

def main():
    """
    Example usage of the timing experiment.
    """
    import os
    try:
        # Try to import a simple dataset first
        from autometrics.dataset.datasets.simplification.simplification import SimpDA
        dataset = SimpDA()
    except ImportError:
        # Fall back to another dataset if SimpDA is not available
        from autometrics.dataset.datasets.summeval.summeval import SummEval
        dataset = SummEval()
    
    from autometrics.metrics.reference_based.ROUGE import ROUGE
    from autometrics.metrics.reference_based.BLEU import BLEU
    from autometrics.metrics.reference_based.BERTScore import BERTScore
    from autometrics.metrics.reference_based.SARI import SARI
    
    # Create metrics
    metrics = [
        ROUGE(), 
        BLEU(),
        SARI(),
        BERTScore(model="roberta-large"),
    ]
    
    # Create experiment
    experiment = TimingExperiment(
        name="Metric Timing Experiment",
        description="Measuring execution time of different metrics",
        metrics=metrics,
        output_dir="outputs/timing",
        dataset=dataset,
    )
    
    # Run experiment with default max_samples=30
    experiment.run(print_results=True)
    
    # Save results
    experiment.save_results()
    
    print("Timing experiment completed!")

if __name__ == "__main__":
    main() 