#!/bin/bash

#SBATCH --job-name=gpt4omini_autometrics
#SBATCH --account=nlp
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --partition=john
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=logs/gpt4omini_autometrics_%j.out
#SBATCH --error=logs/gpt4omini_autometrics_%j.err
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mryan0@stanford.edu

# Usage:
#   export DATASET_NAME="EvalGenProduct"
#   export TARGET_MEASURE="quality"
#   export SEED="42"                # Single seed per job
#   export API_BASE="https://api.openai.com/v1"   # Or your proxy base
#   sbatch scripts/main_autometrics/gpt4omini/run_autometrics_gpt4omini.sh


# ADD LATER: #SBATCH -x jagupard[19-20,26-31]
# #SBATCH --gres=gpu:2

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
API_BASE=${API_BASE:-"https://api.openai.com/v1"}
MODEL_NAME=${MODEL_NAME:-"openai/gpt-4o-mini"}
ABLA_TAG=${MAIN_ABLATION_TAG:-"full_k30_n5"}

echo "Launching Autometrics (GPT-4o-mini)"
echo "Dataset: $DATASET_NAME"
echo "Measure: $TARGET_MEASURE"
if [ -z "${SEED:-}" ]; then
  echo "ERROR: SEED is required (one seed per job)"; exit 1; fi
echo "Seed:    $SEED"
echo "Model:   $MODEL_NAME"
echo "API:     $API_BASE"
echo "AblTag:  $ABLA_TAG"

# Ensure logs dir exists and distinct output dirs from other models
mkdir -p logs
BASE_RESULTS_DIR="results/main_runs/autometrics/gpt4omini/${DATASET_NAME}_${TARGET_MEASURE}"
mkdir -p "$BASE_RESULTS_DIR"

echo "\n==================== Seed $SEED ===================="

# Reuse Qwen-style ablation cache naming for compatibility
export DSPY_CACHEDIR="/nlp/scr3/nlp/20questions/dspy_cache/autometrics_ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"
export AUTOMETRICS_CACHE_DIR="/nlp/scr3/nlp/20questions/autometrics_cache/ablation_qwen_${DATASET_NAME}_${TARGET_MEASURE}_${ABLA_TAG}_seed${SEED}"

# Run a single seed per job
COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True python analysis/main_experiments/run_main_autometrics.py \
  "$DATASET_NAME" \
  "$TARGET_MEASURE" \
  "$SEED" \
  "$BASE_RESULTS_DIR" \
  --model-name "$MODEL_NAME" \
  --api-base "$API_BASE"


