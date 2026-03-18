#!/usr/bin/env bash
# run_ai_detection_pipeline.sh
#
# V3 pipeline for AI usage detection.
#
# Changes from v2:
#   - generate: proportional agency sampling (floor=20) instead of uniform cap.
#               Keeps Q corpus large while ensuring small agencies are represented.
#   - estimate: NO dedup on P distribution (keep it representative of actual text).
#   - infer:    --dedup to filter form letter duplicates during inference.
#   - Reuses existing v1/v2 AI corpus generations to avoid redundant GPU work.
#
# Usage (on sk3 with GPU):
#   bash run_ai_detection_pipeline.sh
#
# Or wait for free GPUs first:
#   bash wait_for_gpus_and_launch_v3.sh

set -euo pipefail

PYTHON="${PYTHON:-/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python3}"
WORKDIR="${WORKDIR:-/lfs/skampere3/0/alexspan/regulations-demo}"
SCRIPT=data/bulk_downloads/scripts/corpus_ai_usage.py
OUTDIR=data/bulk_downloads/scripts/data/v3
LOG="$WORKDIR/generate_v3.log"

# Previous generation directories
V1DIR=data/bulk_downloads/scripts/data
V2DIR=data/bulk_downloads/scripts/data/v2

DOC_TYPES="public_submission notice rule proposed_rule"

# Target specific GPU (set to available GPU index)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

cd "$WORKDIR"
mkdir -p "$OUTDIR/ai_usage_distributions"

echo "$(date): === Starting v3 pipeline ===" | tee "$LOG"

# ---------------------------------------------------------------
# Step 0: Seed v3 output with v1 + v2 AI corpus generations
#   - Merges existing rewrites so generate's resume logic skips them.
#   - Deduplicates by (document_id, model) in case of overlap.
#   - Only runs if the v3 parquets don't already exist.
# ---------------------------------------------------------------
echo "$(date): Step 0 — seed v3 with v1/v2 generations" | tee -a "$LOG"
$PYTHON -c "
import pandas as pd
from pathlib import Path

v1dir = Path('$V1DIR')
v2dir = Path('$V2DIR')
outdir = Path('$OUTDIR')

for doc_type in '$DOC_TYPES'.split():
    out_path = outdir / f'ai_corpus_{doc_type}.parquet'
    if out_path.exists():
        print(f'{doc_type}: v3 parquet already exists ({out_path}), skipping seed')
        continue

    parts = []
    for label, d in [('v1', v1dir), ('v2', v2dir)]:
        p = d / f'ai_corpus_{doc_type}.parquet'
        if p.exists():
            df = pd.read_parquet(p)
            parts.append(df)
            print(f'{doc_type}: loaded {len(df)} rows from {label} ({p})')
        else:
            print(f'{doc_type}: no {label} corpus at {p}')

    if parts:
        merged = pd.concat(parts, ignore_index=True)
        # Deduplicate by (document_id, model), keeping first occurrence
        before = len(merged)
        merged = merged.drop_duplicates(subset=['document_id', 'model'], keep='first')
        print(f'{doc_type}: merged {before} rows -> {len(merged)} after dedup')
        merged.to_parquet(out_path, index=False)
        print(f'{doc_type}: seeded {out_path}')
    else:
        print(f'{doc_type}: no prior generations found, starting fresh')
" >> "$LOG" 2>&1

echo "$(date): Seed complete." | tee -a "$LOG"

# ---------------------------------------------------------------
# Step 1: Generate AI rewrites (GPU)
#   - Proportional sampling: large agencies contribute proportionally,
#     small agencies get at least 20 docs (floor).
#   - --sample-per-agency 500 sets the total budget per agency
#   - No --dedup here: sample from full corpus for maximum diversity
#   - Resume logic will skip docs already seeded from v1/v2.
# ---------------------------------------------------------------
echo "$(date): Step 1/3 — generate" | tee -a "$LOG"
$PYTHON "$SCRIPT" generate \
    --models meta-llama/Llama-3.3-70B-Instruct \
    --offline \
    --sample-per-agency 500 --sample-proportional --sample-agency-floor 20 \
    --doc-types $DOC_TYPES \
    --batch-size 128 --max-tokens 2048 --max-model-len 8192 \
    --output-dir "$OUTDIR" \
    2>&1 | tee -a "$LOG"

echo "$(date): Generate complete." | tee -a "$LOG"

# ---------------------------------------------------------------
# Step 2: Estimate P/Q distributions (CPU)
#   - NO --dedup: keep P representative of actual document mix
#     (including form letters). This prevents function word LOR
#     bias that made v2 baselines explode.
# ---------------------------------------------------------------
echo "$(date): Step 2/3 — estimate" | tee -a "$LOG"
$PYTHON "$SCRIPT" estimate \
    --doc-types $DOC_TYPES \
    --ai-corpus-dir "$OUTDIR" \
    --output-dir "$OUTDIR/ai_usage_distributions" \
    >> "$LOG" 2>&1

echo "$(date): Estimate complete." | tee -a "$LOG"

# ---------------------------------------------------------------
# Step 3: Infer alpha per stratum (CPU)
#   - --dedup: filter to cluster representatives during inference
#     so form letter campaigns don't inflate alpha estimates.
# ---------------------------------------------------------------
echo "$(date): Step 3/3 — infer" | tee -a "$LOG"
$PYTHON "$SCRIPT" infer \
    --dedup \
    --doc-types $DOC_TYPES \
    --distribution-dir "$OUTDIR/ai_usage_distributions" \
    --stratify-by agency quarter \
    --output "$OUTDIR/ai_usage_results.csv" \
    >> "$LOG" 2>&1

echo "$(date): === V3 pipeline complete ===" | tee -a "$LOG"
echo "$(date): Results: $OUTDIR/ai_usage_results.csv" | tee -a "$LOG"
