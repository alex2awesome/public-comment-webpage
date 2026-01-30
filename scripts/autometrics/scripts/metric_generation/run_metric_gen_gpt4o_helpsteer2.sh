#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --open-mode=append
#SBATCH --partition=john-lo
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --job-name=metric_gen_gpt4o_helpsteer2
#SBATCH --output=scripts/metric_generation/logs/metric_gen_gpt4o_helpsteer2.out
#SBATCH --error=scripts/metric_generation/logs/metric_gen_gpt4o_helpsteer2.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Script for HelpSteer2 dataset with GPT-4o-mini

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate autometrics

cd /nlp/scr2/nlp/personal-rm/autometrics

# Set environment variables for GPT-4o-mini
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_metric_gen_gpt4o_helpsteer2"
export AUTOMETRICS_MODEL_DIR="/sphinx/u/salt-checkpoints/autometrics/models"

# Verify OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

echo "Starting Metric Generation Benchmark with GPT-4o-mini for HelpSteer2 dataset..."
echo "Using DSPY cache: $DSPY_CACHEDIR"
echo "Model save directory: $AUTOMETRICS_MODEL_DIR"
echo "Processing dataset: HelpSteer2"
echo "Seeds: 42 43 44 45 46"
echo "Correlation: all"

# Run the benchmark for HelpSteer2 dataset only
python analysis/ablations/run_metric_generation_benchmark.py \
    --generator-model gpt4o_mini \
    --judge-model gpt4o_mini \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --dataset HelpSteer2 \
    --output-dir results/ablations/metric_generation \
    --model-save-dir $AUTOMETRICS_MODEL_DIR

echo "HelpSteer2 dataset benchmark completed with GPT-4o-mini!"

echo ""
echo "Summary of processed dataset-measure combinations:"
echo "  - HelpSteer2: helpfulness, correctness, coherence, complexity, verbosity"
echo ""
echo "Total: 5 dataset-measure combinations"
echo "Results saved to: results/ablations/metric_generation/" 