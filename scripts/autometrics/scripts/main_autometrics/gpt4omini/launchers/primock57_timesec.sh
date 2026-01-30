#!/bin/bash

# Primock57 time_sec - Autometrics Main Run (GPT-4o-mini)

export DATASET_NAME="Primock57"
export TARGET_MEASURE="time_sec"

export SEEDS="42 43 44 45 46"

echo "=============================================================================="
echo "Primock57 time_sec - Autometrics Main Run (GPT-4o-mini)"
echo "=============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Target: $TARGET_MEASURE"
echo "Seeds: $SEEDS"
echo "=============================================================================="

mkdir -p ../logs logs 2>/dev/null || true

MODEL_NAME=${MODEL_NAME:-"openai/gpt-4o-mini"}
ABLA_TAG=${MAIN_ABLATION_TAG:-"full_k30_n5"}
API=${API_BASE:-"https://api.openai.com/v1"}

for SEED in $SEEDS; do
  job_name="gpt4omini_${DATASET_NAME}_${TARGET_MEASURE}_seed${SEED}"
  sbatch \
    --job-name="${job_name}" \
    --output="logs/${job_name}_%j.out" \
    --error="logs/${job_name}_%j.err" \
    --export=ALL,DATASET_NAME=${DATASET_NAME},TARGET_MEASURE=${TARGET_MEASURE},SEED=${SEED},MODEL_NAME=${MODEL_NAME},API_BASE=${API},MAIN_ABLATION_TAG=${ABLA_TAG} \
    ../run_autometrics_gpt4omini.sh
done


