#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=340GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --job-name=metric_gen_qwen_remaining
#SBATCH --output=scripts/metric_generation/logs/metric_gen_qwen_remaining.out
#SBATCH --error=scripts/metric_generation/logs/metric_gen_qwen_remaining.err
#SBATCH --constraint=141G
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Script for remaining datasets with Qwen3-32B

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate sglang

cd /nlp/scr2/nlp/personal-rm/autometrics

# Server configuration
model="Qwen/Qwen3-32B"
port=7450  # Different port, 10 apart spacing
model_nickname="qwen3_32b"

echo "Starting Qwen3-32B server for remaining datasets metric generation..."
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

# Set environment variables for Qwen
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_metric_gen_qwen_remaining"
export AUTOMETRICS_MODEL_DIR="/sphinx/u/salt-checkpoints/autometrics/models"

# Set API base URL
API_BASE=http://localhost:${port}/v1

echo "Starting Metric Generation Benchmark with Qwen3-32B for remaining datasets..."
echo "Using DSPY cache: $DSPY_CACHEDIR"
echo "Model save directory: $AUTOMETRICS_MODEL_DIR"
echo "API Base: $API_BASE"
echo "Processing datasets: EvalGen, Primock57, RealHumanEval, SummEval"
echo "Seeds: 42 43 44 45 46"
echo "Correlation: all"

# Run the benchmark for remaining datasets
python analysis/ablations/run_metric_generation_benchmark.py \
    --generator-model qwen3_32b \
    --judge-model qwen3_32b \
    --api-base $API_BASE \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --dataset EvalGenMedical EvalGenProduct Primock57 RealHumanEval SummEval \
    --output-dir results/ablations/metric_generation \
    --model-save-dir $AUTOMETRICS_MODEL_DIR

echo "Remaining datasets benchmark completed with Qwen3-32B!"

# Cleanup: Kill the server
pkill -f "sglang.launch_server"

echo ""
echo "Summary of processed dataset-measure combinations:"
echo "  - EvalGenMedical: quality"
echo "  - EvalGenProduct: quality"
echo "  - Primock57: helpfulness"
echo "  - RealHumanEval: quality"
echo "  - SummEval: consistency, relevance, fluency, coherence"
echo ""
echo "Total: 8 dataset-measure combinations"
echo "Results saved to: results/ablations/metric_generation/" 