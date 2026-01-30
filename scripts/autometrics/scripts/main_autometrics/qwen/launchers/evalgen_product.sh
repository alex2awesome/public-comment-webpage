#!/bin/bash

# EvalGen Product - Autometrics Main Run (non-resized)

export DATASET_NAME="EvalGenProduct"
export TARGET_MEASURE="grade"

# Run five seeds by default
export SEEDS="42 43 44 45 46"

echo "=============================================================================="
echo "EvalGen Product - Autometrics Main Run"
echo "=============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Target: $TARGET_MEASURE"
echo "Seeds: $SEEDS"
echo "=============================================================================="

sbatch --job-name="qwen_${DATASET_NAME}_${TARGET_MEASURE}" \
       --output="logs/qwen_${DATASET_NAME}_${TARGET_MEASURE}_%j.out" \
       --error="logs/qwen_${DATASET_NAME}_${TARGET_MEASURE}_%j.err" \
       ../run_autometrics_qwen.sh


