# autometrics
AutoMetrics: Automatically discover, generate, and aggregate evaluation metrics for NLP tasks.

Autometrics helps you evaluate text generation systems by:

1. Generating task-specific candidate metrics with LLMs (LLM-as-a-judge, rubric/code generated metrics)
2. Retrieving the most relevant metrics from a bank of 40+ built-in metrics
3. Evaluating all metrics on your dataset (reference-free and reference-based)
4. Selecting the top metrics using regression
5. Aggregating them into a single, optimized metric and producing a report card

The repository includes simple scripts, examples, notebooks, and a full library to run the end-to-end pipeline.

## Installation (pip + optional extras)

Install the published package (recommended):

```bash
pip install autometrics-research
```

Install with extras (examples):

```bash
pip install "autometrics-research[mauve]"
pip install "autometrics-research[bleurt,bert-score,rouge]"
pip install "autometrics-research[reward-models,gpu]"  # reward models + GPU accel
```

Developer install (from source):

```bash
pip install -e .
```

<details>
  <summary>Optional extras (summary)</summary>

  - fasttext: FastText classifiers — metrics: FastTextEducationalValue, FastTextToxicity, FastTextNSFW
  - lens: LENS metrics — metrics: LENS, LENS_SALSA
  - parascore: Paraphrase metrics — metrics: ParaScore, ParaScoreFree
  - bert-score: metrics: BERTScore
  - bleurt: metrics: BLEURT
  - moverscore: metrics: MOVERScore (adds pyemd)
  - rouge: metrics: ROUGE, UpdateROUGE
  - meteor: metrics: METEOR (adds beautifulsoup4)
  - infolm: metrics: InfoLM (adds torchmetrics)
  - mauve: metrics: MAUVE (evaluate + mauve-text)
  - spacy: metrics: SummaQA (requires `spacy` model; install with `python -m spacy download en_core_web_sm`)
  - hf-evaluate: HF evaluate wrappers — metrics: Toxicity; also used by some wrappers
  - reward-models: Large HF reward models — metrics: PRMRewardModel, INFORMRewardModel, LDLRewardModel, GRMRewardModel
  - readability: metrics: FKGL (textstat)
  - gpu: FlashAttention + NV libs (optional acceleration; benefits large reward models)

</details>

## Quickstart

1) Install dependencies

```bash
pip install autometrics-research
```

2) Ensure Java 21 is installed (required by some retrieval components). See Java section below.

3) Set an API key for an OpenAI-compatible endpoint (for LLM-based generation/judging):

```bash
export OPENAI_API_KEY="your-api-key-here"
```

4) Run the simplest end-to-end example with sensible defaults:

```bash
python autometrics_simple_example.py
```

This will:

- load the `HelpSteer` dataset
- generate and retrieve metrics
- select top-k via regression
- print a summary and report card

For a power-user example with customization, run:

```bash
python autometrics_example.py
```

## Examples and Tutorials

- Simple script with all defaults: `examples/autometrics_simple_example.py`
- Power-user/custom configuration: `examples/autometrics_example.py`
- Notebook tutorials: `examples/tutorial.ipynb`, `demo.ipynb`
- Text walkthrough tutorial: `examples/TUTORIAL.md` and runnable `examples/tutorial.py`

If you prefer an experiments-style entry point with CLI arguments, see:

```bash
python analysis/main_experiments/run_main_autometrics.py <dataset_name> <target_name> <seed> <output_dir>
```

There are also convenience scripts in `analysis/` for ablations and scaling.

## Repository Structure

- `autometrics/dataset/datasets`: Built-in datasets (e.g., `helpsteer`, `simplification`, `evalgen`, `iclr`, ...). The main dataset interface lives in `autometrics/dataset/Dataset.py`.
- `autometrics/metrics`: Metric implementations and utilities. See `autometrics/metrics/README.md` for how to write new metrics.
- `autometrics/metrics/llm_judge`: LLM-as-a-judge rubric generators (e.g., G-Eval, Prometheus-style, example-based).
- `autometrics/aggregator/regression`: Regression-based selection/aggregation (Lasso, Ridge, ElasticNet, PLS, etc.).
- `autometrics/recommend`: Metric retrieval modules (BM25/ColBERT/LLMRec and `PipelinedRec`).
- `autometrics/test`: Unit and integration tests, including caching behavior and generator tests.
- `analysis/`: Experiment drivers (CLI), ablations, robustness/scaling studies, and utilities.

## Basic Usage (Library)

```python
import os
import dspy
from autometrics.autometrics import Autometrics
from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer

os.environ["OPENAI_API_KEY"] = "your-key-here"

dataset = HelpSteer()
generator_llm = dspy.LM("openai/gpt-4o-mini")
judge_llm = dspy.LM("openai/gpt-4o-mini")

autometrics = Autometrics()
results = autometrics.run(
    dataset=dataset,
    target_measure="helpfulness",
    generator_llm=generator_llm,
    judge_llm=judge_llm,
)

print([m.get_name() for m in results['top_metrics']])
print(results['regression_metric'].get_name())
```

For more advanced configuration (custom generators, retrieval pipelines, priors, parallelism), see `TUTORIAL.md`.

## System Requirements

### Python

Install dependencies:

```bash
pip install -r requirements.txt
```

Some metrics require GPUs. You can inspect GPU memory needs by checking `gpu_mem` on metric classes. Many metrics run on CPU.

### Java (required for certain retrieval options)

This package requires Java Development Kit (JDK) 21 for some of its search functionality.

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install openjdk-21-jdk
```

#### macOS (using Homebrew)
```bash
brew install openjdk@21
```

#### Windows
Download and install from `https://www.oracle.com/java/technologies/downloads/#java21` or use Chocolatey:
```bash
choco install openjdk21
```

Verify:
```bash
java -version
```

You should see something like:
```
openjdk version "21.0.x"
OpenJDK Runtime Environment ...
OpenJDK 64-Bit Server VM ...
```

Note: Java 17 or lower versions will not work as Pyserini requires Java 21.

## Datasets

Built-in datasets are in `autometrics/dataset/datasets` (e.g., `HelpSteer`, `SimpDA`, `ICLR`, `RealHumanEval`, etc.). You can also construct your own via the `Dataset` class.

Minimal custom dataset example:

```python
import pandas as pd
from autometrics.dataset.Dataset import Dataset

df = pd.DataFrame({
    'id': ['1', '2'],
    'input': ['prompt 1', 'prompt 2'],
    'output': ['response 1', 'response 2'],
    'reference': ['ref 1', 'ref 2'],
    'human_score': [4.5, 3.2]
})

dataset = Dataset(
    dataframe=df,
    target_columns=['human_score'],
    ignore_columns=['id'],
    metric_columns=[],
    name="MyCustomDataset",
    data_id_column="id",
    input_column="input",
    output_column="output",
    reference_columns=['reference'],
    task_description="Evaluate response quality",
)
```

## Disk Caching

The library implements disk caching for all metrics to improve performance when running scripts multiple times. Key features:

- All metrics cache results by default in the `./autometrics_cache` directory (configurable via `AUTOMETRICS_CACHE_DIR`)
- Cache keys include input/output/references and all initialization parameters
- Non-behavioral parameters are excluded automatically (name, description, cache config)
- You can exclude additional parameters via `self.exclude_from_cache_key()`
- Disable per-metric with `use_cache=False`
- Very fast metrics like BLEU/SARI may disable cache by default

See examples in `autometrics/test/custom_metric_caching_example.py`. For guidance on writing new metrics, see `autometrics/metrics/README.md`.

## Where to Go Next

- Read the tutorial: `examples/TUTORIAL.md` (and `examples/tutorial.ipynb`)
- Browse built-in metrics under `autometrics/metrics/`
- Explore experiment drivers in `analysis/`

## Citation

If you use this software, please cite it as below.

```
@software{Ryan_Autometrics_2025,
author = {Ryan, Michael J. and Zhang, Yanzhe and Salunkhe, Amol and Chu, Yi and Xu, Di and Yang, Diyi},
license = {MIT},
title = {{Autometrics}},
url = {https://github.com/XenonMolecule/autometrics},
version = {1.0.0},
year = {2025}
}
```