#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Section-scoring pipeline with subsection grouping + tqdm + dry-run

CSV columns (expected):
    id, full_text, abstract, conference, accepted, format, review_comments,
    appropriateness, clarity, impact, meaningful_comparison, originality,
    recommendation, recommendation_unofficial, replicability, reviewer_confidence,
    soundness_correctness, substance

We only require:
    - full_text
    - abstract
    - id (optional but preferred; used as doc_id if present)

Features:
- Splits ALL-CAPS headings (numbered allowed) and groups 3.1/3.2.* under 3 <TITLE>.
- Scores each grouped section with MathProcessRewardModel (unless --dry-run).
- Progress bars via tqdm.
- Dry-run prints extracted section titles per document (no metric calls).
- Writes long-format CSV in non-dry runs.

Usage:
    python section_scoring_grouped.py --input in.csv --output out.csv
      [--full-text-col full_text] [--abstract-col abstract]
      [--min-section-chars 50]
      [--keep-subheadings] [--no-keep-subheadings]
      [--include_preface]
      [--max_rows N]
      [--dry-run]

Examples:
    # Dry-run to inspect section detection only
    python section_scoring_grouped.py --input papers.csv --output /tmp/ignore.csv --dry-run

    # Full run with scoring
    python section_scoring_grouped.py --input papers.csv --output results.csv
"""

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # we'll degrade gracefully


from autometrics.metrics.reference_free.PRMRewardModel import MathProcessRewardModel

# ---------------------------
# Heading detection & parsing
# ---------------------------

_HEADING_LINE_RE = re.compile(
    r"""^
        \s*
        (?:
            (?P<num>\d+(?:\.\d+)*)\s+      # numbering like 3 or 3.1.2   (optional group)
        )?
        (?P<title>[A-Z0-9][A-Z0-9 \-–:()\/,&\.'"]+?)   # ALL CAPS-ish title
        \s*$
    """,
    re.VERBOSE,
)

_MAX_HEADING_LEN = 120


def _is_all_capsish(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > _MAX_HEADING_LEN:
        return False
    if not re.search(r"[A-Z]", s):
        return False
    return s == s.upper()


@dataclass
class Heading:
    raw: str
    number: Optional[str]  # '3', '3.1.2', or None
    level: int             # 1 for top-level; unnumbered => 1
    title: str             # normalized title (upper)
    line_index: int


def parse_heading(line: str, idx: int) -> Optional[Heading]:
    line_stripped = line.strip()
    if not _is_all_capsish(line_stripped):
        return None
    m = _HEADING_LINE_RE.match(line_stripped)
    if not m:
        return None
    num = m.group("num")
    title = re.sub(r"\s+", " ", m.group("title").strip())
    level = 1 if not num else num.count(".") + 1
    return Heading(raw=line_stripped, number=num, level=level, title=title, line_index=idx)


@dataclass
class Section:
    heading: Heading
    body_lines: List[str]


def split_sections(full_text: str) -> List[Section]:
    """Return ordered list of Section objects (each with heading + body)."""
    if not isinstance(full_text, str):
        return []
    lines = full_text.splitlines()

    sections: List[Section] = []
    current_heading: Optional[Heading] = None
    current_body: List[str] = []

    def flush():
        nonlocal current_heading, current_body
        if current_heading is not None:
            # Trim leading/trailing blank lines
            while current_body and not current_body[0].strip():
                current_body.pop(0)
            while current_body and not current_body[-1].strip():
                current_body.pop()
            sections.append(Section(current_heading, current_body))
        current_heading, current_body = None, []

    for i, raw in enumerate(lines):
        h = parse_heading(raw, i)
        if h is not None:
            flush()
            if h.number is None:
                h.level = 1
            current_heading = h
            current_body = []
        else:
            if current_heading is None:
                current_heading = Heading(raw="PREFACE", number=None, level=1, title="PREFACE", line_index=i)
            current_body.append(raw)

    flush()
    return sections


# ---------------------------
# Grouping: roll up 3.1, 3.2 → 3 METHODS
# ---------------------------

def top_number(num: Optional[str]) -> Optional[str]:
    return None if not num else num.split(".")[0]


@dataclass
class GroupedSection:
    parent_number: Optional[str]   # e.g., '3' or None (unnumbered)
    parent_title: str              # e.g., '3 METHOD(S)' or 'ABSTRACT'
    chunks: List[Tuple[Heading, List[str]]]  # (subheading, body_lines)


def group_subsections(sections: List[Section]) -> List[GroupedSection]:
    grouped: List[GroupedSection] = []
    current_group: Optional[GroupedSection] = None
    last_parent_top: Optional[str] = None

    for sec in sections:
        h = sec.heading
        if h.level == 1:  # top-level -> new group
            if current_group is not None:
                grouped.append(current_group)
            pn = h.number
            current_group = GroupedSection(
                parent_number=pn if pn else None,
                parent_title=(f"{pn} {h.title}" if pn else h.title),
                chunks=[(h, sec.body_lines)],
            )
            last_parent_top = top_number(pn)
        else:
            tn = top_number(h.number)
            if current_group is None:
                current_group = GroupedSection(parent_number=tn, parent_title=f"{tn} (UNTITLED PARENT)", chunks=[])
                last_parent_top = tn
            if last_parent_top is None or tn != last_parent_top:
                grouped.append(current_group)
                current_group = GroupedSection(parent_number=tn, parent_title=f"{tn} (UNTITLED PARENT)", chunks=[])
                last_parent_top = tn
            current_group.chunks.append((h, sec.body_lines))

    if current_group is not None:
        grouped.append(current_group)

    return grouped


def render_group_body(group: GroupedSection, keep_subheadings: bool = True) -> str:
    out_lines: List[str] = []
    for heading, body in group.chunks:
        if heading.level == 1:
            out_lines.extend(body)
        else:
            if keep_subheadings:
                out_lines.append("")
                out_lines.append(heading.raw)
            out_lines.extend(body)
    while out_lines and not out_lines[0].strip():
        out_lines.pop(0)
    while out_lines and not out_lines[-1].strip():
        out_lines.pop()
    return "\n".join(out_lines).strip()


# ---------------------------
# Scoring pipeline
# ---------------------------

ID_CANDIDATES = ["id", "doc_id", "paper_id"]

def choose_doc_id(row: pd.Series) -> Optional[str]:
    # Prefer the user's provided 'id' column
    for k in ID_CANDIDATES:
        if k in row and pd.notna(row[k]):
            return str(row[k])
    return None


def print_dry_run(doc_index: int, doc_id: Optional[str], grouped: List[GroupedSection]) -> None:
    """Pretty-print grouped section titles (no metric calls)."""
    header = f"[doc_index={doc_index}"
    if doc_id is not None:
        header += f", id={doc_id}"
    header += "]"
    print(header)
    for g in grouped:
        # Parent title
        print(f"  - {g.parent_title}")
        # Child titles (if any)
        child_titles = [h.raw for h, _ in g.chunks if h.level >= 2]
        for ct in child_titles:
            print(f"      * {ct}")
    print("")  # extra space between docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV (ignored in --dry-run)")
    parser.add_argument("--full-text-col", default="full_text", help="Column containing raw paper text")
    parser.add_argument("--abstract-col", default="abstract", help="Column containing abstract text")
    parser.add_argument("--min-section-chars", type=int, default=50, help="Skip grouped sections shorter than this")
    parser.add_argument("--max_rows", type=int, default=None, help="Optionally limit number of input rows")
    parser.add_argument("--include_preface", action="store_true", help="Include the 'PREFACE' top-level section")
    parser.add_argument("--keep-subheadings", dest="keep_sub", action="store_true", default=True,
                        help="Keep subsection titles inside grouped body (default True)")
    parser.add_argument("--no-keep-subheadings", dest="keep_sub", action="store_false",
                        help="Do not include subsection titles inside grouped body")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not run the metric; only print extracted (grouped) section titles per doc")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].copy()

    # Column checks
    missing = [c for c in [args.full_text_col, args.abstract_col] if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: Missing required columns: {missing}")

    # Prepare metric unless dry-run
    metric = None
    if not args.dry_run:
        try:
            MathProcessRewardModel  # type: ignore
        except NameError:
            print(
                "WARNING: MathProcessRewardModel is not imported yet. "
                "Edit the script to add the correct import before running.",
                file=sys.stderr,
            )
        try:
            metric = MathProcessRewardModel(persistent=True)
        except Exception as e:
            print(
                "ERROR: Failed to construct MathProcessRewardModel. "
                "Check your import path and dependencies.\n"
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    results: List[Dict] = []

    iterable = df.iterrows()
    if tqdm is not None:
        iterable = tqdm(iterable, total=len(df), desc="Processing documents", unit="doc")

    for ridx, row in iterable:
        full_text = row[args.full_text_col]
        abstract_text = row[args.abstract_col]
        doc_index = ridx
        doc_id = choose_doc_id(row)

        if not isinstance(full_text, str) or not full_text.strip():
            # Still print something in dry-run for visibility
            if args.dry_run:
                print(f"[doc_index={doc_index}{', id='+doc_id if doc_id else ''}] (empty full_text)")
            continue

        # 1) Split into raw sections
        raw_sections = split_sections(full_text)
        if not args.include_preface:
            raw_sections = [s for s in raw_sections if s.heading.title != "PREFACE"]

        # 2) Group under parent
        grouped = group_subsections(raw_sections)

        if args.dry_run:
            print_dry_run(doc_index, doc_id, grouped)
            continue

        # 3) Score each grouped section
        for g in grouped:
            body = render_group_body(g, keep_subheadings=args.keep_sub)
            if not body or len(body) < args.min_section_chars:
                continue

            section_title = g.parent_title
            try:
                prm_min, prm_max, prm_mean = metric.calculate(
                    str(abstract_text) if isinstance(abstract_text, str) else "",
                    str(body),
                )
            except Exception as e:
                results.append(
                    {
                        "doc_index": doc_index,
                        "doc_id": doc_id,
                        "section_title": section_title,
                        "section_char_len": len(body),
                        "PRM_min": math.nan,
                        "PRM_max": math.nan,
                        "PRM_mean": math.nan,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                continue

            results.append(
                {
                    "doc_index": doc_index,
                    "doc_id": doc_id,
                    "section_title": section_title,
                    "section_char_len": len(body),
                    "PRM_min": float(prm_min),
                    "PRM_max": float(prm_max),
                    "PRM_mean": float(prm_mean),
                }
            )

    if args.dry_run:
        # No output file—explicitly say so
        print("Dry run complete. No metric calls were made; no CSV was written.")
        return

    # 4) Write results
    out_df = pd.DataFrame(results)
    ordered = [
        "doc_index", "doc_id", "section_title", "section_char_len",
        "PRM_min", "PRM_max", "PRM_mean"
    ]
    extra = [c for c in out_df.columns if c not in ordered]
    out_df = out_df[ordered + extra] if not out_df.empty else pd.DataFrame(columns=ordered + extra)

    out_path = os.path.abspath(args.output)
    out_df.to_csv(out_path, index=False)
    print(f"✓ Wrote {len(out_df)} rows → {out_path}")


if __name__ == "__main__":
    main()

# Example Usage:
# python analysis/misc/prm_iclr.py --input autometrics/dataset/datasets/iclr/train.csv --output analysis/misc/prm_iclr_results.csv --full-text-col full_text --abstract-col abstract --min-section-chars 50 --include_preface --keep-subheadings --dry-run
# python analysis/misc/prm_iclr.py --input autometrics/dataset/datasets/iclr/train.csv --output analysis/misc/prm_iclr_results.csv --full-text-col full_text --abstract-col abstract --min-section-chars 50 --include_preface --keep-subheadings