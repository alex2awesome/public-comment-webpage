# Metric Utilization Experiment

This module provides tools to benchmark and analyze the resource utilization of evaluation metrics in the `autometrics` library.

## Features

- Measures multiple resource aspects:
  - Runtime performance (milliseconds)
  - CPU memory usage (MB)
  - GPU memory usage (summed across all GPUs, in MB)
  - Disk usage changes (MB)
  
  - Tracks memory across different phases:
  - Import costs (libraries and dependencies)
  - Construction costs (instantiating the metric)
  - First-time use costs (lazy-loaded models and resources)
  - Runtime costs (during actual metric calculation)
  
- Supports memory-isolated trials:
  - Can run each length category in a separate process
  - Ensures clean memory measurements between trials
  - Prevents memory accumulation between length categories
  
- Supports two data sources:
  - Synthetic text generation with configurable lengths
  - Real data from existing datasets (uses all available examples without filtering)
  
- Tests with configurable settings:
  - For synthetic data: short, medium, and long text lengths
  - For real data: uses the actual dataset examples as-is
  - Configurable number of examples and burn-in runs
  
- Produces comprehensive analysis:
  - Raw data for every test run
  - Statistical summary with means and confidence intervals
  - Visualization plots for easy comparison
  - Breakdown of import vs. runtime costs
  - JSON export for programmatic analysis

## Requirements

- Python 3.6+
- `psutil` for process monitoring
- `matplotlib` and `numpy` for plotting and analysis
- `pandas` for data manipulation
- Optional: `nltk` for better vocabulary in synthetic data generation
- Optional: `torch` for GPU memory monitoring with PyTorch
- Optional: `pynvml` for GPU memory monitoring with NVIDIA Management Library

## Usage

### Using Synthetic Data

```python
from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.ROUGE import ROUGE
from autometrics.experiments.utilization import UtilizationExperiment

# Create and configure the experiment with synthetic data
experiment = UtilizationExperiment(
    name="Metric Utilization Benchmark",
    description="Measuring resource usage for NLG metrics",
    metrics=[BLEU(), ROUGE()],
    output_dir="outputs/utilization_synthetic",
    num_examples=30,  # Number of test examples per length category
    num_burn_in=5,    # Number of warm-up runs to avoid cold start effects
    lengths=["short", "medium", "long"],  # Text length categories to test
    use_synthetic=True,  # Use synthetic data (default)
    measure_import_costs=True  # Track import and construction costs (default: True)
)

# Run the experiment
experiment.run(print_results=True)

# Save results to the output directory
experiment.save_results()
```

### Using Real Dataset

```python
from autometrics.dataset.datasets.simplification.simplification import SimpDA
from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.ROUGE import ROUGE
from autometrics.experiments.utilization import UtilizationExperiment

# Load an actual dataset
dataset = SimpDA()

# Create and configure the experiment with real data
experiment = UtilizationExperiment(
    name="Real Data Utilization Benchmark",
    description="Measuring resource usage on real data",
    metrics=[BLEU(), ROUGE()],
    output_dir="outputs/utilization_real_data",
    dataset=dataset,
    num_examples=30,
    num_burn_in=5,
    use_synthetic=False  # Use the provided dataset
)

# Run the experiment
experiment.run(print_results=True)

# Save results to the output directory
experiment.save_results()
```

### Command-line Interface

The module provides a command-line interface through `run_utilization.py`:

```bash
# Run with synthetic data
python autometrics/experiments/utilization/run_utilization.py --output-dir=outputs/utilization --num-examples=50 --burn-in=10 --metrics=BLEU,ROUGE,BERTScore --lengths=short,medium,long --synthetic

# Run with real dataset
python autometrics/experiments/utilization/run_utilization.py --output-dir=outputs/utilization --num-examples=50 --burn-in=10 --metrics=BLEU,ROUGE,BERTScore --dataset=SimpDA

# Run without measuring import costs (faster but less comprehensive)
python run_utilization.py --metrics=BLEU,ROUGE --skip-import-costs

# Explicitly enable import cost measurement (enabled by default)
python run_utilization.py --metrics=BLEU,ROUGE --measure-import-costs
```

## Output Format

The experiment produces a well-organized directory structure of output files:

### For Synthetic Data

```
outputs/utilization_synthetic/
├── full_results.json                  # Complete results in JSON format
├── BLEU/
│   ├── import_costs.csv               # Import and construction costs
│   ├── short/                         # Results for short text inputs
│   │   ├── raw_data.csv               # Raw measurements for each run
│   │   ├── summary.csv                # Statistical summary
│   │   ├── duration_timeseries.pdf    # Time series plot of durations
│   │   ├── memory_timeseries.pdf      # Time series plot of memory usage
│   │   └── duration_histogram.pdf     # Histogram of durations
│   ├── medium/...                     # Similar files for medium text inputs
│   └── long/...                       # Similar files for long text inputs
├── ROUGE/...                          # Similar structure for ROUGE metric
├── metric_comparison/                 # Cross-metric comparisons
│   ├── import_costs.csv               # Import cost comparison across metrics
│   ├── memory_by_phase.pdf            # Memory usage broken down by phase
│   ├── duration_by_phase.pdf          # Time usage broken down by phase
│   ├── import_vs_runtime.pdf          # Import vs runtime memory comparison
│   ├── short/                         # Comparisons for short texts
│   │   ├── summary.csv                # Summary statistics
│   │   ├── duration.pdf               # Duration comparison
│   │   └── memory.pdf                 # Memory usage comparison
│   ├── medium/...                     # Similar files for medium texts
│   └── long/...                       # Similar files for long texts
└── length_comparison/                 # Length impact analysis
    ├── BLEU/                          # Analysis for BLEU
    │   ├── summary.csv                # Summary across lengths
    │   ├── duration.pdf               # Duration vs length
    │   └── memory.pdf                 # Memory usage vs length
    └── ROUGE/...                      # Similar files for ROUGE
```

### For Real Dataset

```
outputs/utilization_real_data/
├── full_results.json                  # Complete results in JSON format
├── BLEU/                              # Results for BLEU metric
│   ├── raw_data.csv                   # Raw measurements
│   ├── summary.csv                    # Statistical summary
│   ├── duration_timeseries.pdf        # Time series plot
│   ├── memory_timeseries.pdf          # Memory usage plot
│   └── duration_histogram.pdf         # Distribution of durations
├── ROUGE/...                          # Similar structure for ROUGE
└── metric_comparison/                 # Cross-metric comparisons
    ├── summary.csv                    # Summary statistics
    ├── duration.pdf                   # Duration comparison
    └── memory.pdf                     # Memory usage comparison
```

## Synthetic Text Length Categories

When using synthetic data, these categories are used:
- **Short**: 3-10 words
- **Medium**: 80-120 words
- **Long**: 800-1200 words

## Resource Tracking API

The module provides a standalone resource tracking API that can be used outside the experiment:

```python
from autometrics.experiments.utilization import track_resources

# Track resources for any operation
with track_resources() as tracker:
    # Do some work here
    result = my_function()

# Get resource usage statistics
stats = tracker.get_results()
print(f"CPU RAM: {stats['cpu_ram_mb']} MB")
print(f"Duration: {stats['duration_milliseconds']} ms")
print(f"Total GPU Memory: {stats['gpu_ram_mb']} MB")
``` 

## Sequential Benchmarking of All Metrics

A script `benchmark_utilization.py` is provided to automatically benchmark all metrics in the `MetricBank` sequentially:

### Features

- Runs benchmarks for all reference-based and reference-free metrics
- Checks for existing results before running to avoid duplicate work
- Saves progress as it goes, can be stopped and resumed at any time
- Aggregates all results into a single CSV file for easy analysis
- Provides detailed logging and progress tracking

### Usage

```bash
# Run with default settings
python benchmark_utilization.py

# Customize the benchmark parameters
python benchmark_utilization.py --output-dir=my_benchmark_results --num-examples=100 --burn-in=10

# Run only reference-based metrics
python benchmark_utilization.py --skip-reference-free

# Run only reference-free metrics
python benchmark_utilization.py --skip-reference-based

# Force re-run of all metrics, even if results exist
python benchmark_utilization.py --force-rerun

# Enable verbose output
python benchmark_utilization.py --verbose
```

### Output Files

- `aggregated_results.csv`: Combined results from all metrics
- `benchmark_summary.csv`: Summary statistics of the benchmark run
- `benchmark_detailed_summary.csv`: Detailed list of all metrics and their status
- `benchmark_utilizer.log`: Complete log file with detailed progress information

### Resuming an Interrupted Run

If the script is interrupted, simply run it again with the same parameters. It will automatically:
1. Detect which metrics already have complete results
2. Skip those metrics and continue with the remaining ones
3. Update the aggregate results with all completed metrics

### Tips for Long Runs

- Running all metrics can take several hours to days depending on your hardware
- Use a persistent terminal like `screen` or `tmux` to prevent script termination if your connection drops
- Check the log file periodically to monitor progress
- The script saves individual metric results immediately, so no work is lost if interrupted