#!/usr/bin/env python3
"""
Download all known completed FOIA datasets from MuckRock CDN.
Organizes files by agency/jurisdiction and year.

Usage: python3 download_datasets.py [--txdot-only] [--skip-txdot]
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ============================================================
# Known datasets with direct CDN URLs
# ============================================================
DATASETS = {
    "seattle-spd/2023_rose-terse_chatgpt-history": {
        "muckrock_id": 145174,
        "description": "Seattle PD ChatGPT chat histories (Rose Terse, 2023)",
        "files": [
            "https://cdn.muckrock.com/foia_files/2023/06/16/Installment_1.zip",
            "https://cdn.muckrock.com/foia_files/2023/08/04/Installment_2.zip",
        ],
    },
    "seattle-education/2024_todd-feathers_genai-prompts": {
        "muckrock_id": 171513,
        "description": "Seattle Education & Early Learning GenAI prompts (Todd Feathers, 2024)",
        "files": [
            "https://cdn.muckrock.com/foia_files/2024/10/04/Armstrong_GenAI_2.docx",
            "https://cdn.muckrock.com/foia_files/2024/10/04/Armstrong_GenAI_1.docx",
            "https://cdn.muckrock.com/foia_files/2024/10/04/Armstrong_GenAI_3.docx",
            "https://cdn.muckrock.com/foia_files/2024/10/04/Armstrong_GenAI_4.docx",
            "https://cdn.muckrock.com/foia_files/2024/10/04/Bakary_GenAI_1.docx",
            "https://cdn.muckrock.com/foia_files/2024/10/04/Choi_GenAI_1.docx",
            "https://cdn.muckrock.com/foia_files/2024/10/04/Dagostino_GenAI_1.docx",
            "https://cdn.muckrock.com/foia_files/2024/10/04/Swift_GenAI_1.docx",
        ],
    },
    "spokane-pd/2023-2024_rose-terse_chatgpt-history": {
        "muckrock_id": 149818,
        "description": "Spokane PD ChatGPT sessions (Rose Terse, 2023-2024)",
        "files": [
            ("https://cdn.muckrock.com/foia_files/2024/02/20/ChatGPT.pdf", "ChatGPT_installment1.pdf"),
            ("https://cdn.muckrock.com/foia_files/2024/03/26/ChatGPT.pdf", "ChatGPT_installment2.pdf"),
        ],
    },
    "kent-pd/2023_rose-terse_chatgpt-history": {
        "muckrock_id": 149817,
        "description": "Kent PD ChatGPT histories from 5 officers (Rose Terse, 2023)",
        "files": [
            "https://cdn.muckrock.com/foia_files/2023/08/10/MR__149817.docx",
            "https://cdn.muckrock.com/foia_files/2023/08/10/Wesson_ChatGPT.pdf",
            "https://cdn.muckrock.com/foia_files/2023/08/10/Stansfield_ChatGPT.pdf",
            "https://cdn.muckrock.com/foia_files/2023/08/10/OReilly_ChatGPT.pdf",
            "https://cdn.muckrock.com/foia_files/2023/08/10/Doherty_Record.pdf",
            "https://cdn.muckrock.com/foia_files/2023/08/10/Hemmen_s_ChatGPT_Redacted_R.pdf",
        ],
    },
    "minneapolis-pd/2025_joey-scott_copilot-records": {
        "muckrock_id": 194109,
        "description": "Minneapolis PD Copilot records, 37K items (Joey Scott, 2025)",
        "files": [
            "https://cdn.muckrock.com/foia_files/2025/12/15/DR2512556_RP1_Public.PDF",
            "https://cdn.muckrock.com/foia_files/2026/01/05/DR2512556_RP2_Public.PDF",
        ],
    },
    "sec/2024-2025_sungho-park_chatgpt-data": {
        "muckrock_id": 175077,
        "description": "SEC ChatGPT-related records (Sungho Park, 2024-2025)",
        "files": [
            "https://cdn.muckrock.com/foia_files/2025/01/28/25-00248-FOIA_Releasable_Record.pdf",
            "https://cdn.muckrock.com/foia_files/2025/01/28/SEC_Response_-25-00248-FOIA.pdf",
        ],
    },
    "cfpb/2024_robert-delaware_chatgpt-histories": {
        "muckrock_id": 158792,
        "description": "CFPB ChatGPT chat histories (Robert Delaware, 2024)",
        "files": [
            "https://cdn.muckrock.com/foia_files/2024/05/09/CFPB-2024-0361-F_-_Redacted.pdf",
            "https://cdn.muckrock.com/foia_files/2024/05/09/CFPB-2024-0361-F_-_Determination_Letter.pdf",
        ],
    },
    "cftc/2024_robert-delaware_chatgpt-histories": {
        "muckrock_id": 163834,
        "description": "CFTC ChatGPT chat histories and guidance (Robert Delaware, 2024)",
        "files": [
            "https://cdn.muckrock.com/foia_files/2024/05/30/24-00236-FOIA_Records_for_Release.pdf",
            "https://cdn.muckrock.com/foia_files/2024/05/22/24-00236-FOIA_Response_Letter.pdf",
        ],
    },
    "fort-worth-city-manager/2025-2026_bradford-davis_chatgpt": {
        "muckrock_id": 196212,
        "description": "Fort Worth City Manager ChatGPT histories (Bradford Davis, 2025-2026)",
        "files": [
            "https://cdn.muckrock.com/foia_files/2025/11/18/R000736-102925_Davis_B._Corr_w_brief.pdf",
            "https://cdn.muckrock.com/foia_files/2025/11/18/R000736-102925_Davis_B._Exhibit_A.pdf",
            "https://cdn.muckrock.com/foia_files/2025/11/18/R000736-102925_Davis_B._AG_Brief.pdf",
            "https://cdn.muckrock.com/foia_files/2025/11/21/ChatGPT_History_Cauner_McDonald.pdf",
            "https://cdn.muckrock.com/foia_files/2026/03/12/ChatGPT_Data_Export_-_Bethany_Warner_Redacted.pdf",
            "https://cdn.muckrock.com/foia_files/2026/03/12/ChatGPT_Data_Export_SWilliams_Redacted.pdf",
            "https://cdn.muckrock.com/foia_files/2026/03/12/ChatGPT_History_Jes_McEachern.pdf",
        ],
    },
    "fort-worth-city-attorney/2025-2026_bradford-davis_chatgpt": {
        "muckrock_id": 196226,
        "description": "Fort Worth City Attorney ChatGPT data (Bradford Davis, 2025-2026)",
        "files": [
            "https://cdn.muckrock.com/foia_files/2025/11/12/E000517-102825_Davis_B._Corr_w_brief.pdf",
            "https://cdn.muckrock.com/foia_files/2025/11/12/E000517-102825_Davis_B._Exhibit_A.pdf",
            "https://cdn.muckrock.com/foia_files/2025/11/12/E000517-102825_Davis_B._AG_Brief.pdf",
            "https://cdn.muckrock.com/foia_files/2025/11/14/E000517-102825_Davis_B._Supplemental_Brief.pdf",
            "https://cdn.muckrock.com/foia_files/2026/03/03/2-19-26_MR196226.pdf",
        ],
    },
}

# TxDOT is handled separately via API due to 1000+ files
TXDOT_DATASET = {
    "dir": "txdot/2026_bradford-davis_ai-usage",
    "muckrock_id": 196227,
    "description": "TxDOT AI usage records — 1000+ Copilot/ChatGPT HTML exports (Bradford Davis, 2026)",
}


def download_file(url, dest_path):
    """Download a file from URL to dest_path. Returns True on success."""
    if dest_path.exists():
        print(f"  SKIP (exists): {dest_path.name}")
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (FOIA research)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest_path.write_bytes(resp.read())
        print(f"  OK: {dest_path.name} ({dest_path.stat().st_size:,} bytes)")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  FAIL: {dest_path.name} — {e}")
        return False


def download_known_datasets():
    """Download all datasets with known direct URLs."""
    total, ok, fail = 0, 0, 0
    for subdir, info in DATASETS.items():
        dest_dir = BASE_DIR / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"{info['description']}")
        print(f"MuckRock #{info['muckrock_id']} → {subdir}/")
        print(f"{'='*60}")

        for entry in info["files"]:
            if isinstance(entry, tuple):
                url, filename = entry
            else:
                url = entry
                filename = url.split("/")[-1]
            total += 1
            if download_file(url, dest_dir / filename):
                ok += 1
            else:
                fail += 1
            time.sleep(0.3)  # polite rate limiting

    return total, ok, fail


def download_txdot():
    """Download TxDOT files via MuckRock API (1000+ files)."""
    dest_dir = BASE_DIR / TXDOT_DATASET["dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    rid = TXDOT_DATASET["muckrock_id"]

    print(f"\n{'='*60}")
    print(f"{TXDOT_DATASET['description']}")
    print(f"MuckRock #{rid} — fetching file list via API...")
    print(f"{'='*60}")

    # Get request data from API
    api_url = f"https://www.muckrock.com/api_v1/foia/{rid}/?format=json"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0 (FOIA research)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  FAIL: Could not fetch API data — {e}")
        return 0, 0, 0

    # Extract all file URLs from communications
    file_urls = []
    for comm in data.get("communications", []):
        for f in comm.get("files", []):
            furl = f.get("ffile")
            if furl:
                file_urls.append(furl)

    # Filter to actual data files (skip small images used in email formatting)
    data_files = [u for u in file_urls if u.endswith((".pdf", ".PDF", ".html", ".docx", ".xlsx", ".csv", ".zip", ".txt"))]
    print(f"  Found {len(data_files)} data files (of {len(file_urls)} total attachments)")

    total, ok, fail = 0, 0, 0
    for url in data_files:
        filename = url.split("/")[-1]
        total += 1
        if download_file(url, dest_dir / filename):
            ok += 1
        else:
            fail += 1
        time.sleep(0.2)

    return total, ok, fail


def main():
    skip_txdot = "--skip-txdot" in sys.argv
    txdot_only = "--txdot-only" in sys.argv

    grand_total, grand_ok, grand_fail = 0, 0, 0

    if not txdot_only:
        t, o, f = download_known_datasets()
        grand_total += t
        grand_ok += o
        grand_fail += f

    if not skip_txdot:
        t, o, f = download_txdot()
        grand_total += t
        grand_ok += o
        grand_fail += f

    print(f"\n{'='*60}")
    print(f"DONE: {grand_ok}/{grand_total} files downloaded, {grand_fail} failures")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
