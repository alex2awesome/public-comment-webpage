#!/bin/bash

#SBATCH --job-name=dna_eval_gpt4o_helpsteer2
#SBATCH --output=scripts/dna_eval/logs/dna_eval_gpt4o_helpsteer2.out
#SBATCH --error=scripts/dna_eval/logs/dna_eval_gpt4o_helpsteer2.err
#SBATCH --time=12:00:00
#SBATCH --partition=john-lo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu
#SBATCH --account=nlp
#SBATCH --requeue

# GPT-4o-mini DNAEval Correlation Analysis - HelpSteer2 Dataset

# Load environment if running under SLURM
if [[ -n "$SLURM_JOB_ID" ]]; then
    . /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
    cd /nlp/scr2/nlp/personal-rm/autometrics
    conda activate autometrics
fi

set -e

echo "Starting GPT-4o-mini DNAEval correlation analysis for HelpSteer2 dataset..."

# Set DSPy cache directory
export DSPY_CACHEDIR="./.cache/dspy_dna_eval_gpt4o_helpsteer2"

# Ensure output directories exist
mkdir -p results/main_runs/baselines/dna_eval_sub_results
mkdir -p scripts/dna_eval/logs

# Run all three correlation types for HelpSteer2
python analysis/main_experiments/run_dna_eval.py \
    --models gpt4o_mini \
    --dataset HelpSteer2 \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --output-dir results/main_runs/baselines

echo "GPT-4o-mini DNAEval correlation analysis for HelpSteer2 completed!"


