#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_ai.sh [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--debug-connector NAME] [--debug-gov-uk]
# Discovers AI-related dockets across configured connectors, then runs the
# crawl -> download -> extract -> export pipeline for each collection.

if ! command -v jq >/dev/null 2>&1; then
  echo "[run_ai] jq is required (e.g., brew install jq)" >&2
  exit 1
fi

START_DATE=""
END_DATE=""
COMMENTS_ROOT="data/comments"
APP_DATA_ROOT="data/app_data"
DATABASE_PATH="${COMMENTS_ROOT}/ai_pipeline.sqlite"
MAX_WORKERS=1
PIPE_VERBOSE=0
DEBUG_CONNECTORS=()
DEBUG_GOV_UK=0
MANIFEST_JSON="${COMMENTS_ROOT}/collections_manifest.json"
MANIFEST_TSV="${COMMENTS_ROOT}/collections_manifest.tsv"
BLOB_DIR="${COMMENTS_ROOT}/blobs"

# Canonical AI dockets if discovery returns nothing
DEFAULT_COLLECTIONS=(
  "regulations_gov:NTIA-2023-0005"      # NTIA AI Accountability (2023)
  "regulations_gov:NTIA-2023-0009"      # NTIA Open-Weight Models (2024)
  "regulations_gov:OMB-2023-0020"       # OMB Draft AI Memo (2023)
  "nist_airmf:AI-RMF-2ND-DRAFT-2022"    # NIST AI RMF 2nd draft comments
  "nitrd_ai_rfi:90-FR-9088"             # OSTP/NITRD AI Action Plan RFI
  "cppa_admt:PR-02-2023"                # California CPPA preliminary comments
  "eu_have_your_say:12270"              # EU 2020 White Paper consultation
  "eu_have_your_say:12527"              # EU 2021 AI Act proposal feedback
  "gov_uk:ai-white-paper-2023"          # UK White Paper consultation (discover via GOV.UK API)
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-date)
      START_DATE="$2"
      shift 2
      ;;
    --end-date)
      END_DATE="$2"
      shift 2
      ;;
    --database)
      DATABASE_PATH="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --verbose)
      PIPE_VERBOSE=1
      shift
      ;;
    --debug-connector)
      DEBUG_CONNECTORS+=("$2")
      shift 2
      ;;
    --debug-gov-uk)
      DEBUG_GOV_UK=1
      DEBUG_CONNECTORS+=("gov_uk")
      shift
      ;;
    *)
      echo "[run_ai] Unknown argument: $1" >&2
      shift
      ;;
  esac
done

CMD=(python -u -m ai_corpus.cli.main)
GLOBAL_FLAGS=()
if [[ "$PIPE_VERBOSE" -eq 1 ]]; then
  GLOBAL_FLAGS+=(--verbose)
fi
if (( ${#DEBUG_CONNECTORS[@]} )); then
  for connector in "${DEBUG_CONNECTORS[@]}"; do
    GLOBAL_FLAGS+=(--debug-connector "$connector")
  done
fi

run_cli() {
  if (( ${#GLOBAL_FLAGS[@]} )); then
    "${CMD[@]}" "${GLOBAL_FLAGS[@]}" "$@"
  else
    "${CMD[@]}" "$@"
  fi
}

# Ensure directories exist
mkdir -p "$COMMENTS_ROOT" "$APP_DATA_ROOT" "$BLOB_DIR"
mkdir -p "$(dirname "$DATABASE_PATH")"

log() {
  printf '\033[1;34m[ai-run]\033[0m %s\n' "$1" >&2
}

discover_collections() {
  local tmp
  tmp=$(mktemp)
  trap 'rm -f "$tmp"' RETURN

  local args=(discover)
  [[ -n "$START_DATE" ]] && args+=(--start-date "$START_DATE")
  [[ -n "$END_DATE" ]] && args+=(--end-date "$END_DATE")
  if (( ${#DEBUG_CONNECTORS[@]} )); then
    for connector in "${DEBUG_CONNECTORS[@]}"; do
      args+=(--connector "$connector")
    done
  fi

  log "Discovering collections (${START_DATE:-all} .. ${END_DATE:-all})"
  if ! run_cli "${args[@]}" > "$tmp"; then
    log "Discovery failed; using defaults"
    echo 0
    return
  fi

  mkdir -p "$COMMENTS_ROOT"
  jq '[ to_entries[] | .key as $connector | (.value[] | {connector: $connector, collection_id: .collection_id, title: (.title // "")}) ]' \
    "$tmp" > "$MANIFEST_JSON"

  local count
  count=$(jq 'length' "$MANIFEST_JSON")
  log "Discovered $count collections"
  echo "$count"
}

run_collection() {
  local connector="$1"
  local collection_id="$2"

  local connector_dir="${COMMENTS_ROOT}/${connector}"
  local collection_dir="${connector_dir}/${collection_id}"
  local raw_dir="${collection_dir}/raw"
  local meta_path="${collection_dir}/${collection_id}.meta.jsonl"

  mkdir -p "$raw_dir"

  log "[${connector}] Crawling $collection_id"
  run_cli crawl \
    --connector "$connector" \
    --collection-id "$collection_id" \
    --output "$meta_path" \
    --target all

  log "[${connector}] Downloading call for $collection_id"
  run_cli download-call \
    --connector "$connector" \
    --collection-id "$collection_id" \
    --meta-file "$meta_path" \
    --out-dir "$raw_dir" \
    --database "$DATABASE_PATH"

  log "[${connector}] Downloading responses for $collection_id"
  run_cli download-responses \
    --connector "$connector" \
    --collection-id "$collection_id" \
    --meta-file "$meta_path" \
    --out-dir "$raw_dir" \
    --database "$DATABASE_PATH" \
    --max-workers "$MAX_WORKERS"

  log "[${connector}] Extracting $collection_id"
  run_cli extract \
    --database "$DATABASE_PATH" \
    --blob-dir "$BLOB_DIR" \
    --max-workers "$MAX_WORKERS"

  log "[${connector}] Exporting $collection_id"
  run_cli export \
    --meta-file "$meta_path" \
    --database "$DATABASE_PATH" \
    --database-url "sqlite:///${APP_DATA_ROOT}/ai_corpus.db"
}

count=$(discover_collections || echo 0)

if [[ "${count}" -eq 0 ]]; then
  mkdir -p "$(dirname "$MANIFEST_TSV")"
  if (( ${#DEBUG_CONNECTORS[@]} )); then
    log "No collections discovered for debug connector(s); attempting fallback defaults"
    : > "$MANIFEST_TSV"
    fallback_found=0
    for entry in "${DEFAULT_COLLECTIONS[@]}"; do
      connector_name="${entry%%:*}"
      collection_id="${entry#*:}"
      for dbg in "${DEBUG_CONNECTORS[@]}"; do
        if [[ "$connector_name" == "$dbg" ]]; then
          printf '%s\t%s\n' "$connector_name" "$collection_id" >> "$MANIFEST_TSV"
          fallback_found=1
          break
        fi
      done
    done
    if [[ "$fallback_found" -eq 0 ]]; then
      log "No fallback defaults available for debug connector(s); nothing to process."
    fi
  else
    log "No collections discovered; falling back to defaults"
    : > "$MANIFEST_TSV"
    for entry in "${DEFAULT_COLLECTIONS[@]}"; do
      printf '%s\t%s\n' "${entry%%:*}" "${entry#*:}" >> "$MANIFEST_TSV"
    done
  fi
else
  jq -r '.[] | [.connector, .collection_id] | @tsv' "$MANIFEST_JSON" > "$MANIFEST_TSV"
fi

processed=0
while IFS=$'\t' read -r connector collection_id; do
  [[ -z "$connector" || -z "$collection_id" ]] && continue
  run_collection "$connector" "$collection_id"
  processed=$((processed + 1))
done < "$MANIFEST_TSV"

# Hand-curated UK AI consultation responses
HAND_CURATED_CONNECTOR_DIR="${COMMENTS_ROOT}/gov_uk/hand_curated_responses"
HAND_CURATED_COLLECTION="uk_ai_regulation_responses"
HAND_CURATED_COLLECTION_DIR="${HAND_CURATED_CONNECTOR_DIR}/${HAND_CURATED_COLLECTION}"
HAND_CURATED_RAW="${HAND_CURATED_COLLECTION_DIR}/raw"
HAND_CURATED_META="${HAND_CURATED_COLLECTION_DIR}/${HAND_CURATED_COLLECTION}.meta.jsonl"

mkdir -p "$HAND_CURATED_RAW"

log "[hand-curated] Downloading published UK AI consultation responses"
python scratch/download_uk_ai_responses.py \
  --output "$HAND_CURATED_RAW" \
  --meta-file "$HAND_CURATED_META"

if [[ "$processed" -eq 0 ]]; then
  log "No collections were processed."
else
  log "All collections processed ($processed total)."
fi
