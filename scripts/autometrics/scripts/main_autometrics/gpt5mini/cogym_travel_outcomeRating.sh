#!/bin/bash

# CoGym Travel Outcome Rating - Autometrics Experiment
# This script runs the unified autometrics experiment for CoGym Travel Outcome

# Set experiment parameters
export DATASET_NAME="CoGymTravelOutcome"
export TARGET_MEASURE="outcomeRating"

# Run all seeds by default
export SEEDS="43"

# Uncomment to run specific seeds only:
# export SEEDS="42 43"  # Run only seeds 42 and 43
# export SEEDS="42"     # Run only seed 42

echo "=============================================================================="
echo "CoGym Travel Outcome Rating - Autometrics Experiment"
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
