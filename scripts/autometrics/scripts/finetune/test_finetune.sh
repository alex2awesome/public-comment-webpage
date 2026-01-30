#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --job-name=test_finetune
#SBATCH --output=scripts/finetune/logs/test_finetune.out
#SBATCH --error=scripts/finetune/logs/test_finetune.err
#SBATCH --constraint=[141G|80G]
#SBATCH --requeue

# Test script for Fine-tuned Metric correlation stability analysis
# Uses limited datasets and seeds for validation

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh

cd /nlp/scr2/nlp/personal-rm/autometrics

conda activate autometrics

echo "Starting Fine-tuned Metric Test Run..."
echo "Using limited datasets: HelpSteer + SimpEval"
echo "Using reduced seeds: 42 43"
echo "Running all correlation types: kendall, pearson, spearman"
echo "Model: ModernBERT-Large with PEFT/LoRA"

# Create output directory
mkdir -p results/main_runs/baselines
mkdir -p scripts/finetune/logs

# Set environment variables for fine-tuning
export AUTOMETRICS_MODEL_DIR="/sphinx/u/salt-checkpoints/autometrics/models"
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_finetune_test"

echo "Environment setup:"
echo "  AUTOMETRICS_MODEL_DIR: $AUTOMETRICS_MODEL_DIR"
echo "  DSPY_CACHEDIR: $DSPY_CACHEDIR"

# Run the test with limited datasets and seeds
python analysis/main_experiments/run_finetune_correlation.py \
    --dataset HelpSteer SimpEval \
    --seeds 42 43 \
    --correlation all \
    --verbose

echo "Test run completed!"
echo "Results saved to: results/main_runs/baselines/finetune_*.csv"

# Quick validation - check that all files were created
echo ""
echo "Validating outputs:"
for corr_type in kendall pearson spearman; do
    filepath="results/main_runs/baselines/finetune_${corr_type}.csv"
    if [ -f "$filepath" ]; then
        echo "✓ $filepath created"
        # Count the number of rows (excluding header)
        row_count=$(tail -n +2 "$filepath" | wc -l)
        echo "  Contains $row_count results"
    else
        echo "✗ $filepath missing"
    fi
done

# Check sub-results too
echo ""
echo "Checking sub-results:"
for dataset in HelpSteer SimpEval; do
    for corr_type in kendall pearson spearman; do
        filepath="results/main_runs/baselines/finetune_sub_results/finetune_${corr_type}_${dataset}.csv"
        if [ -f "$filepath" ]; then
            echo "✓ $filepath created"
        else
            echo "✗ $filepath missing"
        fi
    done
done

# Test argument validation
echo ""
echo "Testing argument validation..."
python analysis/main_experiments/run_finetune_correlation.py --help

echo ""
echo "Testing with invalid correlation..."
python analysis/main_experiments/run_finetune_correlation.py --correlation invalid_corr 2>&1 | head -5

echo ""
echo "Fine-tuned metric test script completed!"

# Check GPU memory usage
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
fi 