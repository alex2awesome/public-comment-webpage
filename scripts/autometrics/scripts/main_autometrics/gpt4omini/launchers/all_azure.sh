#!/bin/bash

# Submit GPT-4o-mini main runs for all datasets

set -euo pipefail

REPO_ROOT="/nlp/scr2/nlp/personal-rm/autometrics"
cd "${REPO_ROOT}/scripts/main_autometrics/gpt4omini/launchers"

SEEDS=${SEEDS:-"42 43 44 45 46"}
export SEEDS
ABLA_TAG=${MAIN_ABLATION_TAG:-"full_k30_n5"}
API=$AZURE_API_BASE
export MAIN_ABLATION_TAG="${ABLA_TAG}"
export API_BASE="${API}"

mkdir -p ../logs logs 2>/dev/null || true

declare -a launchers=(
  # "realhumaneval_accepted.sh"
  "simpeval_score_azure.sh"
  "helpsteer2_helpfulness_azure.sh"
  "evalgen_product_azure.sh"
  "primock57_timesec_azure.sh"
  # "cogym_travel_outcomeRating_azure.sh"
)

for l in "${launchers[@]}"; do
  echo "[ALL] Submitting $l"
  bash "$l"
done

echo "[ALL] Submitted all GPT-4o-mini main runs."


