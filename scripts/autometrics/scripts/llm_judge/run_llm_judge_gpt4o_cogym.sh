#!/bin/bash

#SBATCH --job-name=llm_judge_gpt4o_cogym
#SBATCH --output=scripts/llm_judge/logs/llm_judge_gpt4o_cogym.out
#SBATCH --error=scripts/llm_judge/logs/llm_judge_gpt4o_cogym.err
#SBATCH --time=8:00:00
#SBATCH --partition=john-lo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu
#SBATCH --account=nlp
#SBATCH --requeue

# GPT-4o-mini LLM Judge Correlation Analysis - CoGym Datasets
# This script runs LLM judge correlation analysis specifically for CoGym datasets

# Load environment if running under SLURM
if [[ -n "$SLURM_JOB_ID" ]]; then
    . /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
    cd /nlp/scr2/nlp/personal-rm/autometrics
    conda activate autometrics
fi

set -e

echo "Starting GPT-4o-mini LLM Judge correlation analysis for CoGym datasets..."

# Set DSPy cache directory (separate from others to avoid conflicts)
export DSPY_CACHEDIR="./.cache/dspy_llm_judge_gpt4o_cogym"

# Ensure output directories exist
mkdir -p results/main_runs/baselines/llm_judge_sub_results

# Run all three correlation types for CoGym datasets
echo "Running correlation analysis for CoGym datasets..."

python analysis/main_experiments/run_llm_judge_correlation.py \
    --model gpt4o_mini \
    --dataset CoGymTravelOutcome CoGymTravelProcess CoGymTabularOutcome CoGymTabularProcess CoGymLessonOutcome CoGymLessonProcess \
    --seeds 42 43 44 45 46 \
    --correlation all

echo "GPT-4o-mini LLM Judge correlation analysis for CoGym datasets completed!"

# Display summary statistics
echo ""
echo "=== RESULTS SUMMARY ==="
for corr in kendall pearson spearman; do
    echo ""
    echo "Top correlations for $corr:"
    if [ -f "results/main_runs/baselines/llm_judge_gpt4o_mini_${corr}.csv" ]; then
        # Show top 3 results with mean_correlation and confidence intervals
        python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('results/main_runs/baselines/llm_judge_gpt4o_mini_${corr}.csv')
    if not df.empty:
        df = df.dropna(subset=['mean_correlation'])
        # Filter for CoGym datasets only
        cogym_datasets = ['CoGymTabularOutcome', 'CoGymTabularProcess', 'CoGymTravelOutcome', 'CoGymTravelProcess', 'CoGymLessonOutcome', 'CoGymLessonProcess']
        df = df[df['dataset'].isin(cogym_datasets)]
        if not df.empty:
            df['abs_mean'] = df['mean_correlation'].abs()
            top3 = df.nlargest(3, 'abs_mean')
            for _, row in top3.iterrows():
                ci_width = (row['ci_upper_correlation'] - row['ci_lower_correlation']) / 2
                print(f'  {row[\"dataset\"]}.{row[\"measure\"]}: {row[\"mean_correlation\"]:.4f} ± {ci_width:.4f}')
        else:
            print('  No CoGym results found')
    else:
        print('  No results found')
except Exception as e:
    print(f'  Error reading results: {e}')
"
    else
        echo "  Results file not found"
    fi
done

echo ""
echo "Individual dataset files saved in: results/main_runs/baselines/llm_judge_sub_results/"
echo "Merged results saved in: results/main_runs/baselines/" 