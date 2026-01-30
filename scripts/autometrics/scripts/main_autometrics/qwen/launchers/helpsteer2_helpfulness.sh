#!/bin/bash

# HelpSteer2 Helpfulness - Autometrics Main Run

export DATASET_NAME="HelpSteer2"
export TARGET_MEASURE="helpfulness"

export SEEDS="42 43 44 45 46"

echo "=============================================================================="
echo "HelpSteer2 Helpfulness - Autometrics Main Run"
echo "=============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Target: $TARGET_MEASURE"
echo "Seeds: $SEEDS"
echo "=============================================================================="

sbatch --job-name="qwen_${DATASET_NAME}_${TARGET_MEASURE}" \
       --output="logs/qwen_${DATASET_NAME}_${TARGET_MEASURE}_%j.out" \
       --error="logs/qwen_${DATASET_NAME}_${TARGET_MEASURE}_%j.err" \
       ../run_autometrics_qwen.sh


