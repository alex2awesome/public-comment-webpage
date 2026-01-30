#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=180GB
#SBATCH --open-mode=append
#SBATCH --partition=jag-lo 
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --job-name=metric_gen_qwen_helpsteer_correctness
#SBATCH --output=scripts/metric_generation/logs/metric_gen_qwen_helpsteer_correctness.out
#SBATCH --error=scripts/metric_generation/logs/metric_gen_qwen_helpsteer_correctness.err
#SBATCH --exclude=jagupard[19-20,26-31]
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate sglang

cd /nlp/scr2/nlp/personal-rm/autometrics

model="Qwen/Qwen3-32B"
port=7610
API_BASE=http://localhost:${port}/v1

echo "Starting Qwen3-32B server for HelpSteer (correctness)..."
python -m sglang.launch_server --model-path ${model} --port ${port} --host 0.0.0.0 --tp 2 --dtype bfloat16 --mem-fraction-static 0.8 --trust-remote-code > /dev/null 2>&1 &

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

conda activate autometrics

export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_metric_gen_qwen_helpsteer_correctness"
export AUTOMETRICS_MODEL_DIR="/sphinx/u/salt-checkpoints/autometrics/models"

echo "Starting Metric Generation Benchmark with Qwen3-32B for HelpSteer (correctness)..."
python analysis/ablations/run_metric_generation_benchmark.py \
    --generator-model qwen3_32b \
    --judge-model qwen3_32b \
    --api-base $API_BASE \
    --seeds 42 43 44 45 46 \
    --correlation all \
    --dataset HelpSteer \
    --measure correctness \
    --output-dir results/ablations/metric_generation \
    --model-save-dir $AUTOMETRICS_MODEL_DIR \
    --per-measure-files \
    --skip-generators rubric_prometheus

echo "HelpSteer (correctness) benchmark completed with Qwen3-32B!"

pkill -f "sglang.launch_server" 