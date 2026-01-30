#!/bin/bash

# Example: How to create new autometrics experiments
# This shows the pattern for creating new experiment scripts

# =============================================================================
# Example 1: HelpSteer Helpfulness Experiment
# =============================================================================

echo "Example 1: HelpSteer Helpfulness Experiment"
echo "=========================================="

# Set experiment parameters
export DATASET_NAME="HelpSteer"
export TARGET_MEASURE="helpfulness"
export SEEDS="42 43 44 45 46"

echo "Dataset: $DATASET_NAME"
echo "Target: $TARGET_MEASURE"
echo "Seeds: $SEEDS"

# Run the experiment
# sbatch scripts/main_autometrics/qwen/run_autometrics_qwen.sh

echo ""
echo ""

# =============================================================================
# Example 2: SimpDA Simplicity Experiment (Single Seed)
# =============================================================================

echo "Example 2: SimpDA Simplicity Experiment (Single Seed)"
echo "===================================================="

# Set experiment parameters
export DATASET_NAME="SimpDA"
export TARGET_MEASURE="simplicity"
export SEEDS="42"  # Only run seed 42

echo "Dataset: $DATASET_NAME"
echo "Target: $TARGET_MEASURE"
echo "Seeds: $SEEDS"

# Run the experiment
# sbatch scripts/main_autometrics/qwen/run_autometrics_qwen.sh

echo ""
echo ""

# =============================================================================
# Example 3: Custom Seeds for Debugging
# =============================================================================

echo "Example 3: Custom Seeds for Debugging"
echo "===================================="

# Set experiment parameters
export DATASET_NAME="SummEval"
export TARGET_MEASURE="coherence"
export SEEDS="42 43"  # Only run first two seeds for quick testing

echo "Dataset: $DATASET_NAME"
echo "Target: $TARGET_MEASURE"
echo "Seeds: $SEEDS"

# Run the experiment
# sbatch scripts/main_autometrics/qwen/run_autometrics_qwen.sh

echo ""
echo "=============================================================================="
echo "PATTERN FOR CREATING NEW EXPERIMENTS:"
echo "=============================================================================="
echo "1. Create a new script file (e.g., helpsteer_helpfulness.sh)"
echo "2. Set the environment variables:"
echo "   export DATASET_NAME=\"YourDataset\""
echo "   export TARGET_MEASURE=\"your_target_measure\""
echo "   export SEEDS=\"42 43 44 45 46\"  # or custom seeds"
echo "3. Call the unified script:"
echo "   sbatch scripts/main_autometrics/qwen/run_autometrics_qwen.sh"
echo ""
echo "That's it! The unified script handles everything else."
echo "=============================================================================="
