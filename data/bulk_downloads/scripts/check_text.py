#!/usr/bin/env python3
"""
Analyze parsed PDF text files and flag suspicious pages.

This script looks for markers in the form ``<<PAGE N>>`` and inspects each
page's payload for common failure patterns such as blank pages, extremely short
payloads (likely truncated OCR), or a high ratio of undecodable characters.
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PAGE_MARKER_RE = re.compile(r"<<PAGE\s+(\d+)\s*>>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more files or directories to inspect (directories are scanned recursively for *.txt).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=25,
        help="Minimum number of non-whitespace characters required before a page is considered non-blank.",
    )
    parser.add_argument(
        "--gibberish-threshold",
        type=float,
        default=0.4,
        help=(
            "Maximum fraction of characters that may be non-printable/replacement ('�') "
            "before a page is flagged as gibberish."
        ),
    )
    parser.add_argument(
        "--glob",
        default="*.txt",
        help="Glob used when scanning directories (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-page findings instead of only aggregated stats.",
    )
    return parser.parse_args()


def iter_text_files(paths: Iterable[Path], pattern: str) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from path.rglob(pattern)
        elif path.is_file():
            yield path


def iter_pages(text: str) -> Iterable[Tuple[int, str]]:
    matches = list(PAGE_MARKER_RE.finditer(text))
    if not matches:
        yield 1, text.strip()
        return
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        page_no = int(match.group(1))
        yield page_no, text[start:end].strip()


def classify_page(content: str, min_length: int, gib_threshold: float) -> List[str]:
    flags: List[str] = []
    stripped = content.strip()
    if not stripped or len(stripped) < min_length:
        flags.append("blank")
    printable = sum(ch.isprintable() for ch in stripped)
    total = len(stripped) or 1
    gib_ratio = 1 - (printable / total)
    replacement_ratio = stripped.count("�") / total
    noisy_ratio = max(gib_ratio, replacement_ratio)
    if noisy_ratio >= gib_threshold:
        flags.append("gibberish")
    return flags


def analyze_file(
    path: Path,
    *,
    min_length: int,
    gib_threshold: float,
    verbose: bool,
) -> Dict[str, object]:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"path": path, "error": str(exc)}
    findings: List[Tuple[int, List[str]]] = []
    lengths: List[int] = []
    for page_no, payload in iter_pages(data):
        flags = classify_page(payload, min_length, gib_threshold)
        lengths.append(len(payload))
        if flags:
            findings.append((page_no, flags))
            if verbose:
                print(f"{path} :: page {page_no}: {', '.join(flags)} (len={len(payload)})")
    stats = {
        "path": path,
        "pages": len(lengths),
        "flagged_pages": len(findings),
        "blank_pages": sum("blank" in flags for _, flags in findings),
        "gibberish_pages": sum("gibberish" in flags for _, flags in findings),
        "median_chars": statistics.median(lengths) if lengths else 0,
    }
    if findings and not verbose:
        sample = ", ".join(f"p{page}({','.join(flags)})" for page, flags in findings[:5])
        print(f"{path}: flagged {len(findings)} page(s). Examples: {sample}")
    return stats


def main() -> int:
    args = parse_args()
    files = sorted(set(iter_text_files(args.paths, args.glob)))
    if not files:
        print("No text files matched.")
        return 1
    aggregate = {"flagged": 0, "blank": 0, "gibberish": 0, "pages": 0}
    for file_path in files:
        result = analyze_file(
            file_path,
            min_length=args.min_length,
            gib_threshold=args.gibberish_threshold,
            verbose=args.verbose,
        )
        if "error" in result:
            print(f"{file_path}: {result['error']}")
            continue
        aggregate["pages"] += result["pages"]
        aggregate["flagged"] += result["flagged_pages"]
        aggregate["blank"] += result["blank_pages"]
        aggregate["gibberish"] += result["gibberish_pages"]
    print(
        "Summary: {flagged} flagged pages "
        "(blank={blank}, gibberish={gibberish}) out of {pages} total pages."
    ).format(**aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
