#!/usr/bin/env python3
"""
Run all FOIA tracker enrichment steps in sequence.

Pipeline:
  Step 1: Add 3 new columns (retention, precedent, template notes) + 9 new rows
          (reads foia_tracker_detailed_v2.xlsx → writes foia_tracker_detailed_v3.xlsx)
  Step 2: Enrich retention policies, precedents, template notes, AI deployments
          (reads/writes foia_tracker_detailed_v3.xlsx)
  Step 3: Update Notes column with comprehensive filing guidance for all jurisdictions
          (reads/writes foia_tracker_detailed_v3.xlsx)
  Step 4: Add 12 new federal agency rows + append federal AI FOIA guidance
          (reads/writes foia_tracker_detailed_v3.xlsx)
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

STEPS = [
    ("01_add_columns_and_rows.py", "Adding columns and rows"),
    ("02_enrich_data.py", "Enriching retention, precedent, template, AI deployment data"),
    ("03_update_notes.py", "Updating Notes with filing guidance"),
    ("04_add_agencies.py", "Adding new federal agencies + federal guidance"),
]


def main():
    for script, description in STEPS:
        print(f"\n{'='*60}")
        print(f"Step: {description}")
        print(f"Script: {script}")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script)],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"\nERROR: {script} failed with return code {result.returncode}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("All steps completed successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
