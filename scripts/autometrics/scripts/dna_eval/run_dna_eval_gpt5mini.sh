#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --open-mode=append
#SBATCH --partition=john-lo
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=dna_eval_gpt5mini
#SBATCH --output=scripts/dna_eval/logs/dna_eval_gpt5mini.out
#SBATCH --error=scripts/dna_eval/logs/dna_eval_gpt5mini.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Production script for DNAEval correlation stability analysis
# Using GPT-5-mini across all datasets

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

# Set environment variables
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_dna_eval_gpt5mini"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

echo "Starting DNAEval Production Run with GPT-5-mini..."
echo "Processing all datasets with 5 seeds: 42 43 44 45 46"
echo "Running all correlation types: kendall, pearson, spearman"
echo "Model: gpt5_mini"
echo "DSPy Cache: $DSPY_CACHEDIR"

# Create output directory
mkdir -p results/main_runs/baselines
mkdir -p scripts/dna_eval/logs

# Run the full analysis
python analysis/main_experiments/run_dna_eval.py \
    --models gpt5_mini \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --output-dir results/main_runs/baselines

echo "Production run completed!"
echo "Results saved to: results/main_runs/baselines/dna_eval_gpt5_mini_*.csv"

# Generate summary statistics
echo ""
echo "Generating summary statistics..."
python -c "
import pandas as pd
import numpy as np
import os

print('=== DNAEval Correlation Results Summary ===')

correlation_types = ['kendall', 'pearson', 'spearman']

for corr_type in correlation_types:
    filepath = f'results/main_runs/baselines/dna_eval_gpt5_mini_{corr_type}.csv'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"\n--- {corr_type.upper()} Correlation ---")
        print(f"Total metrics analyzed: {len(df)}")
        print(f"Mean correlation (absolute): {df['mean_correlation'].abs().mean():.4f}")
        print(f"Median correlation (absolute): {df['mean_correlation'].abs().median():.4f}")
        print(f"Successful runs: {df['num_successful_runs'].sum()}/{len(df) * 5}")
        print(f"\nTop 5 Strongest {corr_type.upper()} Correlations:")
        df_sorted = df.reindex(df['mean_correlation'].abs().sort_values(ascending=False).index)
        for i, (_, row) in enumerate(df_sorted.head(5).iterrows()):
            print(f"{i+1:2d}. {row['dataset']}.{row['measure']}: {row['mean_correlation']:.4f} ± {row['std_correlation']:.4f}")
"

echo ""
echo "DNAEval GPT-5-mini production run completed!"


