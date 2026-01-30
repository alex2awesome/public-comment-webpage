#!/bin/bash

# EvalGen Product - Autometrics Experiment
# This script runs the unified autometrics experiment for EvalGen Product

# Set experiment parameters
export DATASET_NAME="EvalGenProduct"
export TARGET_MEASURE="grade"

# Run all seeds by default
export SEEDS="42"

# Uncomment to run specific seeds only:
# export SEEDS="42 43"  # Run only seeds 42 and 43
# export SEEDS="42"     # Run only seed 42

echo "=============================================================================="
echo "EvalGen Product - Autometrics Experiment"
echo "=============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Target: $TARGET_MEASURE"
echo "Seeds: $SEEDS"
echo "=============================================================================="

# Run the unified autometrics script with custom job name
sbatch --job-name="gpt5mini_${DATASET_NAME}_${TARGET_MEASURE}" \
       --output="logs/gpt5mini_${DATASET_NAME}_${TARGET_MEASURE}_%j.out" \
       --error="logs/gpt5mini_${DATASET_NAME}_${TARGET_MEASURE}_%j.err" \
       run_autometrics_gpt5mini.sh
