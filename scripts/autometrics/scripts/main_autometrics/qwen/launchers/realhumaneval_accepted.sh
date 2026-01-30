#!/bin/bash

# RealHumanEval accepted - Autometrics Main Run

export DATASET_NAME="RealHumanEval"
export TARGET_MEASURE="accepted"

export SEEDS="42 43 44 45 46"

echo "=============================================================================="
echo "RealHumanEval accepted - Autometrics Main Run"
echo "=============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Target: $TARGET_MEASURE"
echo "Seeds: $SEEDS"
echo "=============================================================================="

sbatch --job-name="qwen_${DATASET_NAME}_${TARGET_MEASURE}" \
       --output="logs/qwen_${DATASET_NAME}_${TARGET_MEASURE}_%j.out" \
       --error="logs/qwen_${DATASET_NAME}_${TARGET_MEASURE}_%j.err" \
       ../run_autometrics_qwen.sh


