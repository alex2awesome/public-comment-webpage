#!/bin/bash

#SBATCH --job-name=dna_eval_gpt4o_simplification
#SBATCH --output=scripts/dna_eval/logs/dna_eval_gpt4o_simplification.out
#SBATCH --error=scripts/dna_eval/logs/dna_eval_gpt4o_simplification.err
#SBATCH --time=6:00:00
#SBATCH --partition=john-lo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu
#SBATCH --account=nlp
#SBATCH --requeue

# DNAEval Correlation Analysis - Simplification Datasets with GPT-4o-mini

# Load environment if running under SLURM
if [[ -n "$SLURM_JOB_ID" ]]; then
    . /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
    cd /nlp/scr2/nlp/personal-rm/autometrics
    conda activate autometrics
fi

set -e

# Ensure output and logs directories exist
mkdir -p results/main_runs/baselines/dna_eval_sub_results
mkdir -p scripts/dna_eval/logs

echo "Starting DNAEval correlation analysis for simplification datasets with GPT-4o-mini..."

# Separate DSPy cache to avoid collisions
export DSPY_CACHEDIR="./.cache/dspy_dna_eval_gpt4o_simplification"

# Run all three correlation types for simplification datasets
# SimpDA SimpEval
python analysis/main_experiments/run_dna_eval.py \
    --models gpt4o_mini \
    --dataset SimpEval \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --output-dir results/main_runs/baselines

echo "DNAEval GPT-4o-mini correlation analysis for simplification datasets completed!"

echo ""
echo "=== RESULTS SUMMARY ==="
for corr in kendall pearson spearman; do
    echo ""
    echo "Top correlations for $corr:"
    if [ -f "results/main_runs/baselines/dna_eval_gpt4o_mini_${corr}.csv" ]; then
        python3 -c "
import pandas as pd
try:
    df = pd.read_csv('results/main_runs/baselines/dna_eval_gpt4o_mini_${corr}.csv')
    if not df.empty:
        df = df.dropna(subset=['mean_correlation'])
        df = df[df['dataset'].isin(['SimpEval'])]
        if not df.empty:
            df['abs_mean'] = df['mean_correlation'].abs()
            top3 = df.nlargest(3, 'abs_mean')
            for _, row in top3.iterrows():
                ci_width = (row['ci_upper_correlation'] - row['ci_lower_correlation']) / 2
                print(f"  {row['dataset']}.{row['measure']}: {row['mean_correlation']:.4f} ± {ci_width:.4f}")
        else:
            print('  No simplification results found')
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
echo "Individual dataset files saved in: results/main_runs/baselines/dna_eval_sub_results/"
echo "Merged results saved in: results/main_runs/baselines/"


