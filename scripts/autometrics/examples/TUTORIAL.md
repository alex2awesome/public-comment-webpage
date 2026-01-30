# Autometrics Tutorial

## Introduction

Autometrics automatically finds the best evaluation metrics for your NLP task by:
1. **Generating** task-specific metrics using LLMs
2. **Retrieving** relevant metrics from a bank of 40+ built-in metrics  
3. **Evaluating** all metrics on your dataset
4. **Selecting** the top metrics using regression
5. **Aggregating** into a single optimized metric

**Intended Use**: Evaluate text generation quality (summarization, translation, dialogue, etc.) with human-aligned metrics.

## Prerequisites: System Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Java Requirements
```bash
# Ubuntu/Debian
sudo apt install openjdk-21-jdk

# macOS
brew install openjdk@21

# Verify
java -version  # Should show Java 21
```

### GPU Requirements
Some metrics require GPUs. Check requirements:

```python
from autometrics.metrics.MetricBank import all_metric_classes

# Check GPU requirements
for metric_class in all_metric_classes:
    gpu_mem = getattr(metric_class, 'gpu_mem', 0)
    if gpu_mem > 0:
        print(f"{metric_class.__name__}: {gpu_mem:.0f}MB GPU")
```

**Examples:**
- `PRMRewardModel`: 130,000MB GPU (requires high-end GPU)
- `BERTScore`: 8MB GPU (works on most GPUs)
- `BLEU`: 0MB GPU (CPU-only)

## Step 1: Adding your Dataset

### Using Built-in Datasets
```python
from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer
from autometrics.dataset.datasets.simplification.simplification import SimpDA

# Load built-in dataset
dataset = HelpSteer()  # or SimpDA(), etc.
target_measure = "helpfulness"  # column name with human scores
```

### Creating Custom Datasets
```python
import pandas as pd
from autometrics.dataset.Dataset import Dataset

# Your data
df = pd.DataFrame({
    'id': ['1', '2', '3'],
    'input': ['prompt 1', 'prompt 2', 'prompt 3'],
    'output': ['response 1', 'response 2', 'response 3'],
    'reference': ['ref 1', 'ref 2', 'ref 3'],  # optional
    'human_score': [4.5, 3.2, 4.8]  # target measure
})

# Create dataset
dataset = Dataset(
    dataframe=df,
    target_columns=['human_score'],
    ignore_columns=['id'],
    metric_columns=[],  # will be populated automatically
    name="MyCustomDataset",
    data_id_column="id",
    input_column="input", 
    output_column="output",
    reference_columns=['reference'],  # optional
    task_description="Evaluate response quality"
)
```

## Step 2: Running the Autometrics Pipeline

### Basic Usage (All Defaults)
```python
import os
import dspy
from autometrics.autometrics import Autometrics

# Set API key
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Configure LLMs
generator_llm = dspy.LM("openai/gpt-4o-mini")
judge_llm = dspy.LM("openai/gpt-4o-mini")

# Create pipeline with defaults
autometrics = Autometrics()

# Run pipeline
results = autometrics.run(
    dataset=dataset,
    target_measure="human_score",
    generator_llm=generator_llm,
    judge_llm=judge_llm
)

print(f"Top metrics: {[m.get_name() for m in results['top_metrics']]}")
print(f"Regression metric: {results['regression_metric'].get_name()}")
```

### Advanced Configuration
```python
from autometrics.recommend.ColBERT import ColBERT
from autometrics.recommend.LLMRec import LLMRec

# Custom configuration
autometrics = Autometrics(
    # Generate fewer metrics per type
    metric_generation_configs={
        "llm_judge": {"metrics_per_trial": 3},
        "codegen": {"metrics_per_trial": 2}
    },
    
    # Use specific retriever pipeline
    retriever_kwargs={
        "recommenders": [ColBERT, LLMRec],
        "top_ks": [20, 10]
    },
    
    # Include specific metrics upfront
    metric_priors=[LDLRewardModel, BLEU],
    
    # Allow some metric failures
    allowed_failed_metrics=2,
    
    # Custom output directory
    generated_metrics_dir="my_metrics"
)

results = autometrics.run(
    dataset=dataset,
    target_measure="human_score", 
    generator_llm=generator_llm,
    judge_llm=judge_llm,
    num_to_retrieve=20,  # fewer metrics
    num_to_regress=3     # fewer final metrics
)
```

<details>
<summary><strong>Hyperparameters</strong></summary>

### Metric Generation Configs
```python
# Available generators with descriptions
FULL_GENERATOR_CONFIGS = {
    "llm_judge": {"metrics_per_trial": 10},           # Basic LLM judge
    "rubric_prometheus": {"metrics_per_trial": 10},   # Prometheus rubric
    "rubric_dspy": {"metrics_per_trial": 5},          # DSPy rubric  
    "geval": {"metrics_per_trial": 10},               # G-Eval
    "codegen": {"metrics_per_trial": 10},             # Code generation
    "llm_judge_optimized": {"metrics_per_trial": 1},  # Optimized judge
    "finetune": {"metrics_per_trial": 1},             # Fine-tuned model
    "llm_judge_examples": {"metrics_per_trial": 1}    # Example-based
}

# Custom configuration example
custom_configs = {
    "llm_judge": {"metrics_per_trial": 3},      # Generate 3 LLM judge metrics
    "codegen": {"metrics_per_trial": 2},        # Generate 2 code-based metrics
    "rubric_dspy": {"metrics_per_trial": 1}     # Generate 1 rubric metric
}
```

### Retriever Options

#### Single Retrievers
```python
from autometrics.recommend.BM25 import BM25
from autometrics.recommend.ColBERT import ColBERT
from autometrics.recommend.LLMRec import LLMRec

# BM25 (fast, CPU-only)
bm25_retriever = BM25(
    metric_classes=all_metric_classes,
    index_path=None,  # Auto-generate index
    force_reindex=False
)

# ColBERT (GPU-accelerated semantic search)
colbert_retriever = ColBERT(
    metric_classes=all_metric_classes,
    index_path=None,  # Auto-generate index
    force_reindex=False,
    model_name="colbert-ir/colbertv2.0"  # Default model
)

# LLMRec (LLM-based retrieval)
llmrec_retriever = LLMRec(
    metric_classes=all_metric_classes,
    model=generator_llm,  # LLM for retrieval
    index_path=None,
    force_reindex=False
)
```

#### Pipeline Retrievers (Recommended)
```python
from autometrics.recommend.PipelinedRec import PipelinedRec

# GPU pipeline: ColBERT → LLMRec
gpu_pipeline = PipelinedRec(
    recommenders=[ColBERT, LLMRec],
    top_ks=[60, 30],  # ColBERT gets 60, LLMRec narrows to 30
    index_paths=[None, None],  # Auto-generate indices
    force_reindex=False,
    metric_classes=all_metric_classes,
    model=generator_llm  # For LLMRec
)

# CPU pipeline: BM25 → LLMRec  
cpu_pipeline = PipelinedRec(
    recommenders=[BM25, LLMRec],
    top_ks=[60, 30],
    index_paths=[None, None],
    force_reindex=False,
    metric_classes=all_metric_classes,
    model=generator_llm
)

# Custom pipeline: BM25 → ColBERT → LLMRec
custom_pipeline = PipelinedRec(
    recommenders=[BM25, ColBERT, LLMRec],
    top_ks=[100, 50, 25],  # Progressive narrowing
    index_paths=[None, None, None],
    force_reindex=False,
    metric_classes=all_metric_classes,
    model=generator_llm
)
```

### Retrieval and Regression Numbers

#### Setting Retrieval Count
```python
# In autometrics.run()
results = autometrics.run(
    dataset=dataset,
    target_measure="human_score",
    generator_llm=generator_llm,
    judge_llm=judge_llm,
    num_to_retrieve=30,  # How many metrics to retrieve from bank
    num_to_regress=5     # How many top metrics to select via regression
)

# For different scenarios:
# Quick test: num_to_retrieve=10, num_to_regress=3
# Standard: num_to_retrieve=30, num_to_regress=5  
# Comprehensive: num_to_retrieve=50, num_to_regress=10
```

#### Pipeline Top-K Configuration
```python
# The top_ks in PipelinedRec must be >= final num_to_retrieve
# Example: If num_to_retrieve=25, then top_ks[-1] must be >= 25

# Correct configuration
pipeline = PipelinedRec(
    recommenders=[BM25, LLMRec],
    top_ks=[60, 25],  # Final k (25) matches num_to_retrieve
    # ...
)

# Wrong configuration (will be auto-adjusted)
pipeline = PipelinedRec(
    recommenders=[BM25, LLMRec], 
    top_ks=[60, 10],  # Final k (10) < num_to_retrieve (25)
    # Autometrics will auto-adjust to [60, 25]
)
```

### Regression Strategies
```python
from autometrics.aggregator.regression.Lasso import Lasso
from autometrics.aggregator.regression.Ridge import Ridge
from autometrics.aggregator.regression.ElasticNet import ElasticNet

# Lasso (default) - sparse selection, good for interpretability
lasso = Lasso(
    name="LassoRegression",
    description="L1-regularized regression for sparse metric selection",
    dataset=dataset  # Added automatically
)

# Ridge - dense selection, good for stability
ridge = Ridge(
    name="RidgeRegression", 
    description="L2-regularized regression for stable predictions",
    dataset=dataset
)

# ElasticNet - balanced L1/L2 regularization
elasticnet = ElasticNet(
    name="ElasticNetRegression",
    description="Combined L1/L2 regularization",
    dataset=dataset
)

# Use in Autometrics
autometrics = Autometrics(
    regression_strategy=Ridge,  # Change from default Lasso
    regression_kwargs={}  # Additional kwargs for regression
)
```

### Parallelization Settings
```python
autometrics = Autometrics(
    enable_parallel_evaluation=True,  # Speed up evaluation (default: True)
    max_parallel_workers=20           # Max concurrent workers (default: 20)
)

# For different environments:
# CPU-only: max_parallel_workers=4-8
# GPU with network metrics: max_parallel_workers=20-50
# High-end server: max_parallel_workers=50-100
```

### Metric Priors (Include Specific Metrics)
```python
from autometrics.metrics.reference_free.LDLRewardModel import LDLRewardModel
from autometrics.metrics.reference_based.BLEU import BLEU

# Include specific metrics upfront (before retrieval)
autometrics = Autometrics(
    metric_priors=[LDLRewardModel, BLEU],  # Always include these
    generated_metric_priors={
        "llm_judge_optimized": 1,  # Generate 1 optimized judge as prior
        "llm_judge_examples": 1    # Generate 1 example-based judge as prior
    }
)
```

### Caching and Storage
```python
autometrics = Autometrics(
    # Save generated metrics
    generated_metrics_dir="my_generated_metrics",
    merge_generated_with_bank=False,  # Keep separate from built-in metrics
    
    # Allow metric failures
    allowed_failed_metrics=2,  # Don't crash if 2 metrics fail
    
    # Reproducibility
    seed=42  # Random seed for consistent results
)
```

### Complete Advanced Example
```python
# Full custom configuration
autometrics = Autometrics(
    # Generation
    metric_generation_configs={
        "llm_judge": {"metrics_per_trial": 5},
        "codegen": {"metrics_per_trial": 3},
        "rubric_dspy": {"metrics_per_trial": 2}
    },
    
    # Retrieval
    retriever_kwargs={
        "recommenders": [ColBERT, LLMRec],
        "top_ks": [50, 25],
        "index_paths": [None, None],
        "force_reindex": False
    },
    
    # Regression
    regression_strategy=Ridge,
    regression_kwargs={},
    
    # Priors
    metric_priors=[LDLRewardModel],
    generated_metric_priors={"llm_judge_optimized": 1},
    
    # Performance
    enable_parallel_evaluation=True,
    max_parallel_workers=30,
    allowed_failed_metrics=3,
    
    # Storage
    generated_metrics_dir="experiment_1",
    seed=123
)

# Run with custom numbers
results = autometrics.run(
    dataset=dataset,
    target_measure="human_score",
    generator_llm=generator_llm,
    judge_llm=judge_llm,
    num_to_retrieve=25,  # Must match top_ks[-1]
    num_to_regress=8
)
```

</details>

## Step 3: Using your Metrics

### Understanding Results
```python
# Access results
top_metrics = results['top_metrics']           # List of selected metrics
regression_metric = results['regression_metric']  # Final aggregated metric
importance_scores = results['importance_scores']  # Metric importance
report_card = results['report_card']           # Summary report

# Use metrics on new data
for metric in top_metrics:
    scores = metric.predict(new_dataset)
    print(f"{metric.get_name()}: {scores.mean():.3f}")

# Use final regression metric
final_scores = regression_metric.predict(new_dataset)
```

### Metric Report Card
```python
print(results['report_card'])
```

**Example Output:**
```
# Autometrics Report Card

## Dataset Information
- Dataset: HelpSteer
- Target Measure: helpfulness
- Dataset Size: 1000 examples

## Top Metrics Selected
- 1. LDLRewardModel (MultiMetric: helpfulness, safety)
- 2. BERTScore
- 3. CustomLLMJudge_helpfulness

## Regression Aggregator
- Type: Lasso
- Name: Autometrics_Regression_helpfulness
```

### Adding to Metric Bank
```python
# Save generated metrics for reuse
autometrics = Autometrics(
    merge_generated_with_bank=True,  # Save to metric bank
    generated_metrics_dir="my_metric_bank"
)

# Load custom metric bank
from autometrics.metrics.MetricBank import all_metric_classes
custom_bank = all_metric_classes + [MyCustomMetric]
autometrics = Autometrics(metric_bank=custom_bank)
```

### Best Practices

1. **Start Simple**: Use defaults first, then customize
2. **Check GPU Requirements**: Some metrics need significant GPU memory
3. **Use Appropriate LLMs**: GPT-4o-mini works well, larger models for complex tasks
4. **Validate Results**: Check correlation with human scores
5. **Cache Results**: Metrics cache automatically in `./autometrics_cache/` (configurable via `AUTOMETRICS_CACHE_DIR` environment variable)

```python
# Quick validation
import numpy as np
from scipy.stats import pearsonr

human_scores = dataset.get_dataframe()['human_score']
predicted_scores = regression_metric.predict(dataset)

correlation, p_value = pearsonr(human_scores, predicted_scores)
print(f"Correlation: {correlation:.3f} (p={p_value:.3f})")
``` 