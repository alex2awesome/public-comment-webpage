#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --open-mode=append
#SBATCH --partition=sc-loprio
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --job-name=all_recommendation
#SBATCH --output=logs/all_recommendation.out
#SBATCH --error=logs/all_recommendation.err
#SBATCH --constraint=141G
#SBATCH --requeue

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh ; conda activate sglang

cd /nlp/scr2/nlp/personal-rm/autometrics

model="Qwen/Qwen3-32B"
port=7410
model_nickname="qwen3_32b"

python -m sglang.launch_server --model-path ${model} --port ${port} --host 0.0.0.0 --tp 1 --dtype bfloat16 --mem-fraction-static 0.8 --trust-remote-code > /dev/null 2>&1 &

# Try every 20 seconds to curl the server until it's up (get_model_info is a good status check)
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
export DSPY_CACHEDIR=/nlp/scr3/nlp/20questions/dspy_cache/autometrics_qwen3_32b

API_BASE=http://localhost:${port}/v1

python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset SimpDA --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymLessonOutcome --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymLessonProcess --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymTabularOutcome --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymTabularProcess --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymTravelOutcome --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset CoGymTravelProcess --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset Design2Code --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset EvalGenMedical --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset EvalGenProduct --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset HelpSteer --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset HelpSteer2 --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset Primock57 --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset RealHumanEval --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset SimpEval --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat
python autometrics/experiments/recommendation/benchmark_recommendation.py --dataset SummEval --top-k 20 --llm litellm_proxy/$model --llm-api-base $API_BASE --llm-model-type chat