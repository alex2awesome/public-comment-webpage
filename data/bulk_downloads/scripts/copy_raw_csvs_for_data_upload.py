#!/usr/bin/env python3
"""Copy raw bulk-download CSVs into data/to_upload, preserving the agency/year tree.

Usage:
    python scripts/copy_raw_csvs_for_data_upload.py
"""
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = Path("/Users/spangher/Projects/stanford-research/rfi-research/regulations-demo/data/to_upload")
RAW_FILES = {
    "notice.csv", "proposed_rule.csv", "public_submission.csv", "rule.csv",
    "notice_all_text.csv", "proposed_rule_all_text.csv", "proposed_rules_all_text.csv",
    "public_submission_all_text.csv", "public_submissions_all_text.csv",
    "rule_all_text.csv", "rules_all_text.csv",
    "supporting_material_all_text.csv",
}

# Step 1: discover files
log.info("Scanning %s for raw CSVs...", BASE_DIR)
to_copy = sorted(p for p in BASE_DIR.rglob("*.csv") if p.name in RAW_FILES)
log.info("Found %d files to copy.", len(to_copy))

# Step 2: copy
log.info("Copying to %s", TARGET_DIR)
copied = 0
errors = 0
for csv_path in tqdm(to_copy, desc="Copying", unit="file"):
    rel = csv_path.relative_to(BASE_DIR)
    dest = TARGET_DIR / rel
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(csv_path, dest)
        copied += 1
    except Exception as exc:
        log.error("Failed to copy %s: %s", rel, exc)
        errors += 1

# Step 3: summary
log.info("Done. Copied %d files, %d errors.", copied, errors)
