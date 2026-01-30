#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --open-mode=append
#SBATCH --partition=john
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --job-name=dna_eval_gpt4o_remaining
#SBATCH --output=scripts/dna_eval/logs/dna_eval_gpt4o_remaining.out
#SBATCH --error=scripts/dna_eval/logs/dna_eval_gpt4o_remaining.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# DNAEval for remaining datasets: EvalGenMedical, EvalGenProduct, Primock57, RealHumanEval, SummEval

# Load environment if running under SLURM
if [[ -n "$SLURM_JOB_ID" ]]; then
    . /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
    cd /nlp/scr2/nlp/personal-rm/autometrics
    conda activate autometrics
fi

# Ensure output and logs directories exist
mkdir -p results/main_runs/baselines/dna_eval_sub_results
mkdir -p scripts/dna_eval/logs

# Set environment variables
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_dna_eval_gpt4o_mini"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

echo "Starting DNAEval GPT-4o-mini for remaining datasets..."
echo "Processing datasets: EvalGenMedical, EvalGenProduct, Primock57, RealHumanEval, SummEval"
echo "Seeds: 42 43 44 45 46"
echo "Correlations: kendall, pearson, spearman"
echo "Model: gpt4o_mini"

# EvalGenMedical EvalGenProduct Primock57 RealHumanEval SummEval

python analysis/main_experiments/run_dna_eval.py \
    --models gpt4o_mini \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --dataset RealHumanEval \
    --output-dir results/main_runs/baselines

echo "Remaining datasets DNAEval analysis completed!"
echo "Individual dataset files saved in: results/main_runs/baselines/dna_eval_sub_results/"

