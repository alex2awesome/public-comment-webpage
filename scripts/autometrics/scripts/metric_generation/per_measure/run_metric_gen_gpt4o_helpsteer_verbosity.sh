#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --open-mode=append
#SBATCH --partition=john-lo
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=metric_gen_gpt4o_helpsteer_verbosity
#SBATCH --output=scripts/metric_generation/logs/metric_gen_gpt4o_helpsteer_verbosity.out
#SBATCH --error=scripts/metric_generation/logs/metric_gen_gpt4o_helpsteer_verbosity.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate autometrics

cd /nlp/scr2/nlp/personal-rm/autometrics

export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_metric_gen_gpt4o_helpsteer_verbosity"
export AUTOMETRICS_MODEL_DIR="/sphinx/u/salt-checkpoints/autometrics/models"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

echo "Starting Metric Generation Benchmark with GPT-4o-mini for HelpSteer (verbosity)..."
python analysis/ablations/run_metric_generation_benchmark.py \
    --generator-model gpt4o_mini \
    --judge-model gpt4o_mini \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --dataset HelpSteer \
    --measure verbosity \
    --output-dir results/ablations/metric_generation \
    --model-save-dir $AUTOMETRICS_MODEL_DIR \
    --per-measure-files

echo "HelpSteer (verbosity) benchmark completed with GPT-4o-mini!" 