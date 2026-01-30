#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --open-mode=append
#SBATCH --partition=john
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --job-name=llm_judge_gpt4o_remaining
#SBATCH --output=scripts/llm_judge/logs/llm_judge_gpt4o_remaining.out
#SBATCH --error=scripts/llm_judge/logs/llm_judge_gpt4o_remaining.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Script for remaining datasets: EvalGenMedical, EvalGenProduct, Primock57, RealHumanEval, SummEval

# Load environment if running under SLURM
if [[ -n "$SLURM_JOB_ID" ]]; then
    . /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
    cd /nlp/scr2/nlp/personal-rm/autometrics
    conda activate autometrics
fi

# Create output directory
mkdir -p results/main_runs/baselines/llm_judge_sub_results

# Set environment variables
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_gpt4o_mini"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

echo "Starting LLM Judge GPT-4o for remaining datasets..."
echo "Processing datasets: EvalGenMedical, EvalGenProduct, Primock57, RealHumanEval, SummEval"
echo "Seeds: 42 43 44 45 46"
echo "Correlations: kendall, pearson, spearman"
echo "Model: gpt4o_mini"

# Run the analysis for remaining datasets
python analysis/main_experiments/run_llm_judge_correlation.py \
    --model gpt4o_mini \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --dataset EvalGenMedical EvalGenProduct Primock57 RealHumanEval SummEval

echo "Remaining datasets analysis completed!"
echo "Individual dataset files saved in: results/main_runs/baselines/llm_judge_sub_results/"

# Generate summary
echo ""
echo "Summary of processed datasets:"
echo "  - EvalGenMedical: Binary classification (good/bad)"
echo "  - EvalGenProduct: Binary classification (good/bad)"  
echo "  - Primock57: Medical note evaluation (multiple metrics)"
echo "  - RealHumanEval: Code acceptance (binary)"
echo "  - SummEval: Summary evaluation (multiple dimensions)"
echo ""
echo "Results will be merged with existing data using the merge script." 