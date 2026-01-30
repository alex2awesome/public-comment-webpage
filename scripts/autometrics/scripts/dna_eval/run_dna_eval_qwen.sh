#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=dna_eval_qwen
#SBATCH --output=scripts/dna_eval/logs/dna_eval_qwen.out
#SBATCH --error=scripts/dna_eval/logs/dna_eval_qwen.err
#SBATCH --constraint=141G
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Production script for DNAEval correlation stability analysis
# Using Qwen3-32B with local server startup

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate sglang

cd /nlp/scr2/nlp/personal-rm/autometrics

# Server configuration
model="Qwen/Qwen3-32B"
port=7010  # Avoid 7400 range and ensure spacing
model_nickname="qwen3_32b"

echo "Starting Qwen3-32B server..."
python -m sglang.launch_server --model-path ${model} --port ${port} --host 0.0.0.0 --tp 1 --dtype bfloat16 --mem-fraction-static 0.8 --trust-remote-code > /dev/null 2>&1 &

# Wait for server to be ready
TIMEOUT=90
START_TIME=$(date +%s)

while ! curl -s http://localhost:${port}/v1/get_model_info > /dev/null; do
    echo "Waiting for server to start..."
    sleep 20
    CURRENT_TIME=$(date +%s)
    ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
    if [ $ELAPSED_TIME -gt $((TIMEOUT * 60)) ]; then
        echo "Timeout reached after 90 minutes. Killing job."
        exit 1
    fi
done

echo "Server is up and running!"

# Switch to autometrics environment
conda activate autometrics

# Set environment variables
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_qwen3_32b_dna"

# Set API base URL
API_BASE=http://localhost:${port}/v1

echo "Starting DNAEval Production Run with Qwen3-32B..."
echo "Processing all datasets with 5 seeds: 42 43 44 45 46"
echo "Running all correlation types: kendall, pearson, spearman"
echo "Model: qwen3_32b"
echo "API Base: $API_BASE"
echo "DSPy Cache: $DSPY_CACHEDIR"

# Create output directory
mkdir -p results/main_runs/baselines
mkdir -p scripts/dna_eval/logs

# Run the full analysis
python analysis/main_experiments/run_dna_eval.py \
    --models qwen3_32b \
    --api-base $API_BASE \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --output-dir results/main_runs/baselines

echo "Production run completed!"
echo "Results saved to: results/main_runs/baselines/dna_eval_qwen3_32b_*.csv"

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
    filepath = f'results/main_runs/baselines/dna_eval_qwen3_32b_{corr_type}.csv'
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
echo "DNAEval Qwen production run completed!"


