#!/bin/bash

# ICLR Recommendation - Regression Sweep (GPT-4o-mini)
# This script sets env vars and submits the sweep launcher via SLURM.

# Set experiment parameters
export DATASET_NAME="ICLR"
export TARGET_MEASURE="recommendation"

# Run a single seed by default (adjust as needed)
export SEEDS="42"

# Sweep n values (comma-separated)
export N_VALUES="1,3,5,10,15,20"

# Regression selection
export REGRESSION="lasso"   # or "lasso"
export LASSO_ALPHAS="0.01,0.1,0.5,1.0" # sweep over these alphas when REGRESSION=lasso

echo "=============================================================================="
echo "ICLR Recommendation - Regression Sweep (GPT-4o-mini)"
echo "=============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Target:  $TARGET_MEASURE"
echo "Seeds:   $SEEDS"
echo "n vals:  $N_VALUES"
echo "Regression: $REGRESSION"
if [ "$REGRESSION" = "lasso" ]; then
  echo "Lasso alpha: $LASSO_ALPHA"
fi
echo "=============================================================================="

# Submit job using the GPT-4o-mini sweep runner
sbatch \
  --job-name="gpt4omini_${DATASET_NAME}_${TARGET_MEASURE}_sweep" \
  --output="logs/gpt4omini_${DATASET_NAME}_${TARGET_MEASURE}_sweep_%j.out" \
  --error="logs/gpt4omini_${DATASET_NAME}_${TARGET_MEASURE}_sweep_%j.err" \
  run_regression_sweep_gpt4omini.sh



