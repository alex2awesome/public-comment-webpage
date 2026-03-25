#!/usr/bin/env python3
"""Clean AI-generated corpus to remove prompt artifacts before estimate.

Removes:
  - XML/HTML tags echoed from the prompt (<rule>, </notice>, etc.)
  - Metadata echoed from original text (<<COMMENT 1>>, <<PAGE 1>>, dates, true/false)
  - Preamble lines (e.g. "Here is the rewritten rule:", "becomes:")
  - Postamble lines (e.g. "I hope this rewritten version...", "Let me know if...")
  - Placeholder brackets ([your name], [address], [doc ...])
  - Leading/trailing whitespace

Usage:
    python clean_ai_corpus.py --input data/v3/ai_corpus_rule.parquet --output data/v3_cleaned/ai_corpus_rule.parquet
    python clean_ai_corpus.py --input-dir data/v3 --output-dir data/v3_cleaned  # batch mode
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Cleaning patterns
# ---------------------------------------------------------------------------

# XML/HTML tags (including custom ones the LLM echoes back)
_XML_TAG_RE = re.compile(
    r"</?(?:rule|notice|proposed_rule|comment|rewritten_notice|rewritten_rule"
    r"|rewritten_proposed_rule|rewritten|page|date|effective-date|title"
    r"|doc|section|preamble|summary|action|agency|heading)\b[^>]*>",
    re.IGNORECASE,
)

# Metadata markers from original text assembly
_METADATA_RE = re.compile(
    r"<<(?:COMMENT|PAGE)\s*\d*>>",
    re.IGNORECASE,
)

# Standalone date lines echoed from original (e.g. "2018-08-23T04:00Z")
_DATE_LINE_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z?\s*$",
    re.MULTILINE,
)

# Standalone true/false lines echoed from original metadata
_BOOL_LINE_RE = re.compile(
    r"^\s*(?:true|false)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Placeholder brackets: [your name], [your address], [insert ...], [doc ...], etc.
_PLACEHOLDER_RE = re.compile(
    r"\[(?:your|insert|add|doc|name|address|date|title|agency|department"
    r"|organization|city|state|zip|email|phone|signature)[^\]]{0,60}\]",
    re.IGNORECASE,
)

# Preamble patterns (at start of text)
_PREAMBLE_PATTERNS = [
    re.compile(r"^\s*(?:here\s+is|here's)\s+(?:the\s+)?(?:a\s+)?(?:rewritten|rephrased|revised|new)\s+.*?[:\n]", re.IGNORECASE),
    re.compile(r"^\s*(?:the\s+)?(?:rewritten|rephrased|revised)\s+(?:version|text|rule|notice|document)\s*.*?[:\n]", re.IGNORECASE),
    re.compile(r"^\s*becomes\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*(?:sure|okay|ok)[,!.]?\s*(?:here|i(?:'ll| will))\s+.*?[:\n]", re.IGNORECASE),
]

# Postamble patterns (at end of text)
_POSTAMBLE_PATTERNS = [
    re.compile(r"(?:i\s+hope|let\s+me\s+know|please\s+(?:let|note)|don't\s+hesitate|feel\s+free)\s+.*$", re.IGNORECASE),
    re.compile(r"(?:note\s*:\s*(?:i|the|this)\s+).*$", re.IGNORECASE),
    re.compile(r"(?:best|warm|kind)?\s*regards[,.]?\s*$", re.IGNORECASE),
]

# Markdown bold headers that LLM adds (e.g. **Proposed Rule**)
_MARKDOWN_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")

# Repeated newlines
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_ai_text(text: str) -> str:
    """Apply all cleaning steps to a single AI-generated text."""
    if not text or not isinstance(text, str):
        return text

    # Strip leading/trailing whitespace
    text = text.strip()

    # Remove XML tags
    text = _XML_TAG_RE.sub("", text)

    # Remove metadata markers
    text = _METADATA_RE.sub("", text)

    # Remove standalone date lines
    text = _DATE_LINE_RE.sub("", text)

    # Remove standalone bool lines
    text = _BOOL_LINE_RE.sub("", text)

    # Remove placeholders
    text = _PLACEHOLDER_RE.sub("", text)

    # Remove preamble (try each pattern, take first match)
    for pat in _PREAMBLE_PATTERNS:
        text = pat.sub("", text, count=1)

    # Remove postamble (last 500 chars only, to avoid false matches in body)
    if len(text) > 500:
        body, tail = text[:-500], text[-500:]
        for pat in _POSTAMBLE_PATTERNS:
            tail = pat.sub("", tail)
        text = body + tail
    else:
        for pat in _POSTAMBLE_PATTERNS:
            text = pat.sub("", text)

    # Unwrap markdown bold to plain text
    text = _MARKDOWN_BOLD_RE.sub(r"\1", text)

    # Collapse excessive newlines
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # Final strip
    text = text.strip()

    return text


def clean_corpus(input_path: Path, output_path: Path) -> None:
    """Clean a single ai_corpus parquet file."""
    df = pd.read_parquet(input_path)
    n_before = len(df)

    print(f"Cleaning {input_path.name}: {n_before} rows")

    # Clean ai_text column
    df["ai_text"] = df["ai_text"].apply(clean_ai_text)

    # Drop rows where ai_text became empty or too short after cleaning
    min_chars = 50
    df = df[df["ai_text"].str.len() >= min_chars].reset_index(drop=True)
    n_after = len(df)

    # Stats
    print(f"  Kept {n_after}/{n_before} rows ({n_after/n_before*100:.1f}%)")
    print(f"  Mean ai_text len: {df['ai_text'].str.len().mean():.0f} chars")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean AI corpus parquet files")
    parser.add_argument("--input", type=Path, help="Single input parquet file")
    parser.add_argument("--output", type=Path, help="Single output parquet file")
    parser.add_argument("--input-dir", type=Path, help="Directory with ai_corpus_*.parquet files")
    parser.add_argument("--output-dir", type=Path, help="Output directory for cleaned files")
    parser.add_argument(
        "--doc-types",
        nargs="+",
        default=["rule", "notice", "proposed_rule"],
        help="Doc types to clean in batch mode (default: rule notice proposed_rule)",
    )
    args = parser.parse_args()

    if args.input and args.output:
        clean_corpus(args.input, args.output)
    elif args.input_dir and args.output_dir:
        for dt in args.doc_types:
            inp = args.input_dir / f"ai_corpus_{dt}.parquet"
            if not inp.exists():
                print(f"Skipping {inp} (not found)")
                continue
            out = args.output_dir / f"ai_corpus_{dt}.parquet"
            clean_corpus(inp, out)
    else:
        parser.error("Provide either --input/--output or --input-dir/--output-dir")


if __name__ == "__main__":
    main()
