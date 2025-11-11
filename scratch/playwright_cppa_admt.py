#!/usr/bin/env python3

"""
Temporary Playwright helper for exploring the CPPA ADMT comment index.

Opens the CPPA rulemaking updates page, collects all PDF comment links, and
optionally downloads them to a local directory. This mirrors the exploratory
workflow used for the EU "Have Your Say" script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright


INDEX_URL = "https://cppa.ca.gov/regulations/ccpa_updates.html"


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ai_corpus.utils.extraction import extract_pages  # noqa: E402
from ai_corpus.utils.llm_utils import init_backend  # noqa: E402
from scratch.ollama_page_pairs import (  # noqa: E402
    PROMPT_TEMPLATE,
    PagePairJudgment,
    judge_pages_openai_structured,
)


def ensure_browser_cache() -> None:
    """Store Playwright browser binaries inside the repository tree."""

    if "PLAYWRIGHT_BROWSERS_PATH" in os.environ:
        return

    module_path = Path(__file__).resolve()
    cache_dir: Path | None = None
    for parent in module_path.parents:
        candidate = parent / ".playwright"
        if candidate.exists() or (parent / ".git").exists():
            cache_dir = candidate
            break
    if cache_dir is None:
        cache_dir = module_path.parent / ".playwright"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(cache_dir)


PDF_CATEGORY_PATTERNS = [
    ("comment_summary_and_response", re.compile(r"comment summaries?\s+and\s+responses?", re.IGNORECASE)),
    ("aggregated_comments", re.compile(r"(all[_\s-]*comments|written comments\b|comments? combined|pre[_\s-]*comments|late[_\s-]*comments|comment summaries)", re.IGNORECASE)),
    ("hearing_transcript", re.compile(r"transcript", re.IGNORECASE)),
    ("notice", re.compile(r"notice", re.IGNORECASE)),
    ("final_statement_of_reasons", re.compile(r"(final statement of reasons|\bfsor\b)", re.IGNORECASE)),
    ("initial_statement_of_reasons", re.compile(r"(initial statement of reasons|\bisor\b)", re.IGNORECASE)),
    ("economic_impact", re.compile(r"(economic|fiscal impact|std[_\s-]*399|assessment)", re.IGNORECASE)),
    ("reg_text_approved", re.compile(r"approved regulations? text", re.IGNORECASE)),
    ("reg_text_modified", re.compile(r"(modified text|mod[_\s-]*txt)", re.IGNORECASE)),
    ("reg_text_proposed", re.compile(r"text of proposed", re.IGNORECASE)),
    ("invitation_preliminary", re.compile(r"invitation.*comment", re.IGNORECASE)),
    ("supporting_appendix", re.compile(r"appendix", re.IGNORECASE)),
]


def sanitize_filename(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name.strip())
    return sanitized or "document.pdf"


def classify_pdf(anchor_text: str, href: str) -> str:
    haystack_parts = [anchor_text or "", href or ""]
    haystack = " ".join(part.lower() for part in haystack_parts if part)
    for label, pattern in PDF_CATEGORY_PATTERNS:
        if pattern.search(haystack):
            return label
    return "other"


DEFAULT_LLM_MODEL = "gpt-4o"


@lru_cache(maxsize=1)
def get_openai_client() -> Any:
    """Initialise and cache a singleton OpenAI client."""

    clients = init_backend("openai")
    return clients.sync


def split_bundle_pdf(pdf_path: Path, *, model: str = DEFAULT_LLM_MODEL) -> List[Dict[str, Optional[str]]]:
    """
    Split a bundled PDF into individual letters using the LLM-based page pairing logic.
    Each detected letter is written to disk as a text file and captured as metadata.
    """

    try:
        pages, _stats = extract_pages(str(pdf_path), "auto")
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Failed to extract pages from {pdf_path}: {exc}")
        return []

    if not pages:
        print(f"⚠️  No text extracted from {pdf_path}; skipping split.")
        return []

    client = get_openai_client()
    judgments: List[PagePairJudgment] = []
    if len(pages) == 1:
        combined = pages[0].strip()
        if not combined:
            return []
        return _write_segments(pdf_path, [combined])

    for index in range(len(pages) - 1):
        page_a = pages[index]
        page_b = pages[index + 1]
        prompt_text = PROMPT_TEMPLATE.format(page_a=page_a, page_b=page_b)
        try:
            judgment = judge_pages_openai_structured(client, model, prompt_text, max_output_tokens=512)
        except Exception as exc:  # noqa: BLE001
            print(
                f"⚠️  Falling back to same-letter assumption for pages {index + 1}-{index + 2}: {exc}"
            )
            judgment = PagePairJudgment(label="1", a_is_error=False, b_is_error=False)
        judgments.append(judgment)

    if not judgments:
        non_empty_segments = [pages[0].strip()] if pages[0].strip() else []
        if not non_empty_segments:
            return []
        return _write_segments(pdf_path, non_empty_segments)

    appearances = [0] * len(pages)
    error_votes = [0] * len(pages)
    for index, judgment in enumerate(judgments):
        appearances[index] += 1
        appearances[index + 1] += 1
        if judgment.a_is_error:
            error_votes[index] += 1
        if judgment.b_is_error:
            error_votes[index + 1] += 1

    drop_pages = {
        idx for idx in range(len(pages)) if appearances[idx] >= 2 and error_votes[idx] >= 2
    }
    if drop_pages:
        friendly_pages = ", ".join(str(idx + 1) for idx in sorted(drop_pages))
        print(f"    → Filtering error pages: {friendly_pages}")

    label_values = [1 if judgment.label == "1" else 0 for judgment in judgments]
    segment_indices: List[List[int]] = []
    current_indices: List[int] = [0]
    for index, label in enumerate(label_values):
        if label == 1:
            current_indices.append(index + 1)
        else:
            segment_indices.append(current_indices)
            current_indices = [index + 1]
    segment_indices.append(current_indices)

    filtered_segments: List[List[int]] = []
    for indices in segment_indices:
        filtered = [idx for idx in indices if idx not in drop_pages]
        if filtered:
            filtered_segments.append(filtered)

    non_empty_segments = []
    for indices in filtered_segments:
        combined = "\n\n".join(pages[idx].strip() for idx in indices if pages[idx].strip())
        if combined:
            non_empty_segments.append(combined)

    if not non_empty_segments:
        print(f"⚠️  No non-empty segments produced for {pdf_path}")
        return []

    return _write_segments(pdf_path, non_empty_segments)


def _write_segments(pdf_path: Path, segments: List[str]) -> List[Dict[str, Optional[str]]]:
    letters: List[Dict[str, Optional[str]]] = []
    bundle_dir = pdf_path.parent / f"{pdf_path.stem}_letters"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for idx, cleaned in enumerate(segments, start=1):
        cleaned = cleaned.strip()
        if len(cleaned) < 400:
            continue
        submitter = _infer_submitter(cleaned)
        submitted_at = _infer_date(cleaned)
        letter_path = bundle_dir / f"{pdf_path.stem}_letter_{idx}.txt"
        letter_path.write_text(cleaned, encoding="utf-8")
        letters.append(
            {
                "submitter_name": submitter,
                "submitted_at": submitted_at,
                "text_path": str(letter_path),
            }
        )
    return letters


def _infer_submitter(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    first_line = lines[0]
    if first_line.lower().startswith("re:") and len(lines) > 1:
        return lines[1]
    return first_line


def _infer_date(text: str) -> Optional[str]:
    match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+20\d{2}",
        text,
    )
    return match.group(0) if match else None


def collect_pdf_links(page, base_url: str) -> List[Dict[str, str]]:
    """
    Return a deduplicated list of PDF anchors present on the current page along with
    surrounding context text from the nearest wrapping <div>.
    """

    entries: List[Dict[str, str]] = []
    seen: set[str] = set()
    anchors = page.query_selector_all("a[href]")
    for anchor in anchors:
        href = anchor.get_attribute("href")
        if not href:
            continue
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if not parsed.path.lower().endswith(".pdf"):
            continue
        if full_url in seen:
            continue
        seen.add(full_url)
        text = (anchor.text_content() or "").strip()
        fallback_name = Path(parsed.path).name or "comment.pdf"
        filename = sanitize_filename(fallback_name)
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"
        kind = classify_pdf(text, full_url)
        entries.append(
            {
                "text": text or fallback_name,
                "href": full_url,
                "raw_href": href,
                "filename": filename,
                "kind": kind,
            }
        )
    return entries


def download_pdfs(playwright, entries: List[Dict[str, str]], downloads_dir: Path, *, timeout: int, max_downloads: int | None) -> None:
    """Download PDFs using Playwright's request context."""

    if not entries:
        return

    target_entries = [entry for entry in entries if entry.get("kind") == "aggregated_comments"]
    if not target_entries:
        print("No aggregated comment PDFs to download; skipping download step.")
        return

    print(f"Preparing to download {len(target_entries)} aggregated comment PDF(s)...")
    downloads_dir.mkdir(parents=True, exist_ok=True)
    request_context = playwright.request.new_context(timeout=timeout)
    try:
        for idx, entry in enumerate(target_entries, start=1):
            if max_downloads is not None and idx > max_downloads:
                break
            target_path = downloads_dir / entry["filename"]
            if target_path.exists() and target_path.stat().st_size > 0:
                print(f"[{idx}] ✓ Existing file detected at {target_path}")
                entry["download_path"] = str(target_path)
                letters = split_bundle_pdf(target_path)
                if letters:
                    entry["letters"] = letters
                    print(f"[{idx}]    → Extracted {len(letters)} letter(s) from bundle")
                continue
            print(f"[{idx}] Downloading {entry['href']} -> {target_path}")
            try:
                response = request_context.get(entry["href"])
                if response.ok:
                    target_path.write_bytes(response.body())
                    entry["download_path"] = str(target_path)
                    print(f"[{idx}] ✓ Saved to {target_path}")
                    letters = split_bundle_pdf(target_path)
                    if letters:
                        entry["letters"] = letters
                        print(f"[{idx}]    → Extracted {len(letters)} letter(s) from bundle")
                else:
                    status = response.status
                    entry["download_error"] = f"HTTP {status}"
                    print(f"[{idx}] ✗ HTTP {status} when retrieving {entry['href']}")
            except Exception as exc:  # noqa: BLE001
                entry["download_error"] = str(exc)
                print(f"[{idx}] ✗ Error downloading {entry['href']}: {exc}")
            time.sleep(0.1)
    finally:
        request_context.dispose()


def launch_and_collect(
    index_url: str,
    headless: bool,
    slow_mo: int,
    wait_timeout: int,
    *,
    downloads_dir: Path,
    perform_downloads: bool,
    max_downloads: int | None,
    pause_on_complete: bool,
    dump_json: Path | None,
) -> None:
    ensure_browser_cache()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless, slow_mo=slow_mo)
        context = browser.new_context()
        page = context.new_page()
        print(f"Navigating to CPPA page: {index_url}")
        page.goto(index_url, timeout=wait_timeout)
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1500)

        entries = collect_pdf_links(page, index_url)
        if entries:
            print(f"✓ Found {len(entries)} unique PDF link(s):")
            for idx, entry in enumerate(entries, start=1):
                kind = entry.get("kind", "unknown")
                print(f"  {idx}. [{kind}] {entry['text']} -> {entry['href']}")
            agg_count = sum(1 for entry in entries if entry.get("kind") == "aggregated_comments")
            print(f"Aggregated comment PDFs identified: {agg_count}")
        else:
            print("✗ No PDF links detected on the page.")

        if perform_downloads:
            download_pdfs(
                playwright,
                entries,
                downloads_dir,
                timeout=wait_timeout,
                max_downloads=max_downloads,
            )

        if dump_json:
            dump_json.parent.mkdir(parents=True, exist_ok=True)
            serialized = [
                {key: value for key, value in entry.items() if key != "raw_href"}
                for entry in entries
            ]
            dump_json.write_text(
                json.dumps(serialized, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Metadata written to {dump_json}")

        if pause_on_complete and not headless:
            try:
                page.pause()
            except Exception:  # noqa: BLE001
                print("⚠️  Playwright pause failed; continuing.")

        context.close()
        browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the CPPA ADMT updates page, list PDF comment links, and optionally download them.",
    )
    parser.add_argument(
        "index_url",
        nargs="?",
        default=INDEX_URL,
        help="Page to visit (default: CPPA ADMT updates listing).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium in headless mode for unattended runs.",
    )
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=250,
        help="Delay (ms) Playwright inserts between operations to aid debugging.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=60000,
        help="Maximum time (ms) to wait for page operations.",
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "downloads" / "cppa",
        help="Directory to store downloaded PDFs (default: <repo>/downloads/cppa).",
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="Optional limit on the number of PDFs to download.",
    )
    parser.add_argument(
        "--download-first",
        type=int,
        default=None,
        help="Download only the first K PDFs (shortcut for --max-downloads).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Collect metadata only; skip downloading PDFs.",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Do not enter Playwright Inspector pause mode on completion.",
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        help="Optional path to write collected metadata as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launch_and_collect(
        args.index_url,
        args.headless,
        args.slow_mo,
        args.wait_timeout,
        downloads_dir=args.downloads_dir.expanduser().resolve(),
        perform_downloads=not args.no_download,
        max_downloads=(
            args.download_first
            if args.download_first is not None
            else args.max_downloads
        ),
        pause_on_complete=not args.no_pause,
        dump_json=args.dump_json.expanduser().resolve() if args.dump_json else None,
    )


if __name__ == "__main__":
    main()
