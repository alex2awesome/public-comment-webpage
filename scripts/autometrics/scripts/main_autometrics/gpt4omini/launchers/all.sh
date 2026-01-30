#!/bin/bash

# Submit GPT-4o-mini main runs for all datasets

set -euo pipefail

REPO_ROOT="/nlp/scr2/nlp/personal-rm/autometrics"
cd "${REPO_ROOT}/scripts/main_autometrics/gpt4omini/launchers"

SEEDS=${SEEDS:-"42 43 44 45 46"}
export SEEDS
ABLA_TAG=${MAIN_ABLATION_TAG:-"full_k30_n5"}
API=${API_BASE:-"https://api.openai.com/v1"}
export MAIN_ABLATION_TAG="${ABLA_TAG}"
export API_BASE="${API}"

mkdir -p ../logs logs 2>/dev/null || true

declare -a launchers=(
  "realhumaneval_accepted.sh"
  "simpeval_score.sh"
  "helpsteer2_helpfulness.sh"
  "evalgen_product.sh"
  "primock57_timesec.sh"
  "cogym_travel_outcomeRating.sh"
)

for l in "${launchers[@]}"; do
  echo "[ALL] Submitting $l"
  bash "$l"
done

echo "[ALL] Submitted all GPT-4o-mini main runs."


