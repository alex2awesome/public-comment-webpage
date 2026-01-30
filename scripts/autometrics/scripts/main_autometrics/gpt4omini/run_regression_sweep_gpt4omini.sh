#!/bin/bash

#SBATCH --job-name=gpt4omini_reg_sweep
#SBATCH --account=nlp
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:6
#SBATCH --mem=200G
#SBATCH --partition=jag-standard
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=logs/gpt4omini_reg_sweep_%j.out
#SBATCH --error=logs/gpt4omini_reg_sweep_%j.err
#SBATCH -x jagupard[19-20,26-31]
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Usage:
#   export DATASET_NAME="ICLR"
#   export TARGET_MEASURE="recommendation"
#   export SEEDS="42"               # space-separated
#   export N_VALUES="1,3,5,10,15,20" # comma-separated
#   export API_BASE="https://api.openai.com/v1"   # Or your proxy base
#   export MODEL_NAME="openai/gpt-4o-mini"
#   export REGRESSION="hotelling_pls"             # or "lasso"
#   export LASSO_ALPHA="0.01"                     # used if REGRESSION=lasso (single alpha)
#   export LASSO_ALPHAS="0.01,0.1,0.5,1.0"        # used if sweeping alphas
#   sbatch scripts/main_autometrics/gpt4omini/run_regression_sweep_gpt4omini.sh

. /nlp/scr/mryan0/miniconda3/etc/profile.d/conda.sh
conda activate autometrics

cd /nlp/scr2/nlp/personal-rm/autometrics

set -euo pipefail

# Validate required env vars
if [ -z "${DATASET_NAME:-}" ]; then
  echo "ERROR: DATASET_NAME is required"; exit 1; fi
if [ -z "${TARGET_MEASURE:-}" ]; then
  echo "ERROR: TARGET_MEASURE is required"; exit 1; fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is required in environment"; exit 1; fi

# Defaults
SEEDS=${SEEDS:-"42"}
API_BASE=${API_BASE:-"https://api.openai.com/v1"}
MODEL_NAME=${MODEL_NAME:-"openai/gpt-4o-mini"}
N_VALUES=${N_VALUES:-"1,3,5,10,15,20"}
REGRESSION=${REGRESSION:-"hotelling_pls"}
LASSO_ALPHA=${LASSO_ALPHA:-"0.01"}
LASSO_ALPHAS=${LASSO_ALPHAS:-"0.01,0.1,0.5,1.0"}

echo "Launching Autometrics Regression Sweep (GPT-4o-mini)"
echo "Dataset: $DATASET_NAME"
echo "Measure: $TARGET_MEASURE"
echo "Seeds:   $SEEDS"
echo "Model:   $MODEL_NAME"
echo "API:     $API_BASE"
echo "n values: $N_VALUES"
echo "Regression: $REGRESSION"
if [ "$REGRESSION" = "lasso" ]; then
  echo "Lasso alpha: $LASSO_ALPHA"
  echo "Lasso alphas: $LASSO_ALPHAS"
fi

# Ensure logs dir exists and distinct output dirs
mkdir -p logs
BASE_RESULTS_DIR="results/main_runs/autometrics/gpt4omini/${DATASET_NAME}_${TARGET_MEASURE}_sweep"
mkdir -p "$BASE_RESULTS_DIR"

# Iterate seeds
SUCCESSFUL_SEEDS=()
FAILED_SEEDS=()

for seed in $SEEDS; do
  echo "\n==================== Seed $seed ===================="

  # Seed-specific caches (distinct namespace)
  export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_gpt4omini_${DATASET_NAME}_${TARGET_MEASURE}_sweep_seed${seed}"
  export AUTOMETRICS_CACHE_DIR="/nlp/scr3/nlp/20questions/autometrics_cache/gpt4omini_${DATASET_NAME}_${TARGET_MEASURE}_sweep_seed${seed}"

  COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True python analysis/main_experiments/run_regression_sweep.py \
    --dataset "$DATASET_NAME" \
    --target "$TARGET_MEASURE" \
    --seed "$seed" \
    --output-dir "$BASE_RESULTS_DIR" \
    --n-values "$N_VALUES" \
    --model-name "$MODEL_NAME" \
    --api-base "$API_BASE" \
    --regression "$REGRESSION" \
    --lasso-alpha "$LASSO_ALPHA" \
    --lasso-alphas "$LASSO_ALPHAS"

  STATUS=$?
  if [ $STATUS -eq 0 ]; then
    echo "✅ Seed $seed completed"
    SUCCESSFUL_SEEDS+=("$seed")
  else
    echo "❌ Seed $seed failed with status $STATUS"
    FAILED_SEEDS+=("$seed")
  fi
done

echo "\n======= SWEEP SUMMARY (GPT-4o-mini) ======="
echo "Dataset: $DATASET_NAME"
echo "Measure: $TARGET_MEASURE"
echo "Results: $BASE_RESULTS_DIR"
echo "Successful seeds: ${#SUCCESSFUL_SEEDS[@]} -> ${SUCCESSFUL_SEEDS[*]:-none}"
echo "Failed seeds:     ${#FAILED_SEEDS[@]} -> ${FAILED_SEEDS[*]:-none}"

# Print a quick table of Pearson for convenience if files exist
for seed in ${SUCCESSFUL_SEEDS[@]:-}; do
  f="$BASE_RESULTS_DIR/log_${seed}_sweep.json"
  if [ -f "$f" ]; then
    echo "\nSeed $seed Pearson summary:"
    # Extract per-n Pearson using jq if available
    if command -v jq >/dev/null 2>&1; then
      jq -r '.per_n_results | to_entries[] | "  n=" + .key + ": " + (.value.test_scores.pearson|tostring)' "$f" || true
    else
      echo "  (install jq for a pretty summary)"
    fi
  fi
done



