#!/usr/bin/env bash

# Check for missing result artifacts across datasets, subdirectories, and seeds.
#
# Usage:
#   scripts/check_missing_qwen_runs.sh [BASE_DIR]
#
# Defaults:
#   BASE_DIR (arg or env) -> results/ablations/qwen_remote
#
# Behavior:
#   - For each dataset in the fixed list below, iterates over its immediate
#     subdirectories (e.g., ablation configs like full_k5) and checks for the
#     presence of both files per seed:
#       * log_{seed}.json
#       * score_kendall_{seed}.txt
#   - Prints a line for each missing run or missing directory.
#   - If nothing is missing, prints a final success line.

set -u -o pipefail
shopt -s nullglob

BASE_DIR="${1:-${BASE_DIR:-results/ablations/qwen_remote}}"

datasets=(
  CoGymTravelOutcome_outcomeRating_resized
  EvalGenProduct_grade_resized
  RealHumanEval_accepted
  Primock57_time_sec
  HelpSteer2_helpfulness
  SimpEval_score
)

seeds=(42 43 44 45 46)

if [ ! -d "${BASE_DIR}" ]; then
  echo "Base directory not found: ${BASE_DIR}"
  exit 1
fi

missing_count=0

for dataset in "${datasets[@]}"; do
  ds_dir="${BASE_DIR}/${dataset}"
  if [ ! -d "${ds_dir}" ]; then
    echo "MISSING: dataset directory '${ds_dir}'"
    ((missing_count++)) || true
    continue
  fi

  # Collect immediate subdirectories (e.g., full_k5, full_k30, etc.)
  subdirs=("${ds_dir}"/*/)
  if [ ${#subdirs[@]} -eq 0 ]; then
    echo "MISSING: no subdirectories in '${ds_dir}'"
    ((missing_count++)) || true
    continue
  fi

  for subdir in "${subdirs[@]}"; do
    # Trim trailing slash for consistency
    subdir_no_slash="${subdir%/}"
    subname=$(basename "${subdir_no_slash}")

    for seed in "${seeds[@]}"; do
      log_file="${subdir_no_slash}/log_${seed}.json"
      score_file="${subdir_no_slash}/score_kendall_${seed}.txt"

      missing_parts=()
      [ -f "${log_file}" ] || missing_parts+=("log_${seed}.json")
      [ -f "${score_file}" ] || missing_parts+=("score_kendall_${seed}.txt")

      if [ ${#missing_parts[@]} -ne 0 ]; then
        echo "MISSING: ${dataset}/${subname} -> ${missing_parts[*]}"
        ((missing_count++)) || true
      fi
    done
  done
done

if [ ${missing_count} -eq 0 ]; then
  echo "All runs present for the specified datasets and seeds."
fi


