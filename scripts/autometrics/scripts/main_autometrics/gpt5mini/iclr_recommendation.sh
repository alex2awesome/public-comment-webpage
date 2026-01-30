#!/bin/bash

# ICLR Recommendation - Autometrics Experiment (GPT-4o-mini)
# This script sets env vars and submits the unified GPT-4o-mini launcher via SLURM.

# Set experiment parameters
export DATASET_NAME="ICLR"
export TARGET_MEASURE="recommendation"

# Run a single seed by default (adjust as needed)
export SEEDS="42 43 44 45 46"

echo "=============================================================================="
echo "ICLR Recommendation - Autometrics Experiment (GPT-5-mini)"
echo "=============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Target:  $TARGET_MEASURE"
echo "Seeds:   $SEEDS"
echo "=============================================================================="

# Submit job using the GPT-4o-mini unified runner
sbatch \
  --job-name="gpt5mini_${DATASET_NAME}_${TARGET_MEASURE}" \
  --output="logs/gpt5mini_${DATASET_NAME}_${TARGET_MEASURE}_%j.out" \
  --error="logs/gpt5mini_${DATASET_NAME}_${TARGET_MEASURE}_%j.err" \
  run_autometrics_gpt5mini.sh



