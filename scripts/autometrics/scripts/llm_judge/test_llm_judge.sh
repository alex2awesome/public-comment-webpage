#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --job-name=test_llm_judge
#SBATCH --output=scripts/llm_judge/logs/test_llm_judge.out
#SBATCH --error=scripts/llm_judge/logs/test_llm_judge.err
#SBATCH --requeue

# Test script for LLM as a Judge correlation stability analysis
# Uses limited datasets and seeds for validation

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

# Set environment variables
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_gpt4o_mini_test"

echo "Starting LLM Judge Test Run..."
echo "Using limited datasets: HelpSteer + SimpEval"
echo "Using reduced seeds: 42 43 44"
echo "Running all correlation types: kendall, pearson, spearman"
echo "Model: gpt4o_mini"
echo "DSPy Cache: $DSPY_CACHEDIR"

# Create output directory
mkdir -p results/main_runs/baselines

# Run the test with limited datasets and seeds
python analysis/main_experiments/run_llm_judge_correlation.py \
    --model gpt4o_mini \
    --dataset HelpSteer SimpEval \
    --seeds 42 43 44 \
    --verbose

echo "Test run completed!"
echo "Results saved to: results/main_runs/baselines/llm_judge_gpt4o_mini_*.csv"

# Quick validation - check that all files were created
echo ""
echo "Validating outputs:"
for corr_type in kendall pearson spearman; do
    filepath="results/main_runs/baselines/llm_judge_gpt4o_mini_${corr_type}.csv"
    if [ -f "$filepath" ]; then
        echo "✓ $filepath created"
        # Count the number of rows (excluding header)
        row_count=$(tail -n +2 "$filepath" | wc -l)
        echo "  Contains $row_count results"
    else
        echo "✗ $filepath missing"
    fi
done

# Test argument validation
echo ""
echo "Testing argument validation..."
python analysis/main_experiments/run_llm_judge_correlation.py --help

echo ""
echo "Testing with invalid model..."
python analysis/main_experiments/run_llm_judge_correlation.py --model invalid_model 2>&1 | head -5

echo ""
echo "LLM Judge test script completed!" 