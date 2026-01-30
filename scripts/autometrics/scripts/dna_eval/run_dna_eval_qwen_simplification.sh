#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --job-name=dna_eval_qwen_simplification
#SBATCH --output=scripts/dna_eval/logs/dna_eval_qwen_simplification.out
#SBATCH --error=scripts/dna_eval/logs/dna_eval_qwen_simplification.err
#SBATCH --constraint=141G
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Script for Simplification datasets with Qwen3-32B (DNAEval)

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate sglang

cd /nlp/scr2/nlp/personal-rm/autometrics

# Server configuration
model="Qwen/Qwen3-32B"
port=7370  # Avoid 7400 range and maintain spacing
model_nickname="qwen3_32b"

echo "Starting Qwen3-32B server for Simplification datasets (DNAEval)..."
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

# Create output directory
mkdir -p results/main_runs/baselines/dna_eval_sub_results
mkdir -p scripts/dna_eval/logs

# Set environment variables
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_qwen3_32b_dna_simplification"

# Set API base URL
API_BASE=http://localhost:${port}/v1

echo "Starting DNAEval analysis with Qwen3-32B for Simplification datasets..."
echo "Processing datasets: SimpDA, SimpEval"
echo "Seeds: 42 43 44 45 46"
echo "Correlations: kendall, pearson, spearman"

# Run the analysis for Simplification datasets
python analysis/main_experiments/run_dna_eval.py \
    --models qwen3_32b \
    --api-base $API_BASE \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --dataset SimpDA SimpEval \
    --output-dir results/main_runs/baselines

echo "Simplification datasets DNAEval analysis completed!"
echo "Individual dataset files saved in: results/main_runs/baselines/dna_eval_sub_results/"

# Cleanup: Kill the server
pkill -f "sglang.launch_server"

echo ""
echo "Summary of processed datasets:"
echo "  - SimpDA: 3 measures (fluency, meaning, simplicity)"
echo "  - SimpEval: 1 measure (score)"
echo ""
echo "Results will be merged with existing data using the merge script."


