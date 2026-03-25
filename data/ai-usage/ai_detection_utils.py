"""
Shared utilities for AI-usage detection in regulatory documents.

This module consolidates:
  - Cleaning heuristics for AI-rewritten corpora (strip XML tags, meta-commentary,
    self-chat artifacts, etc.)
  - Tokenization (sentence splitting via blingfire or NLTK)
  - Date/time helpers for regulatory document timestamps
  - File I/O helpers for iterating over bulk-download CSVs
  - Log-probability computation (full Bernoulli model, Liang et al. 2025)
  - MLE estimation of AI-written fraction alpha with bootstrap CIs
  - Word-level P/Q distribution building
  - Hierarchical Bayes helpers (empirical Bayes shrinkage of per-agency word probs)
  - Data loading helpers for parallel processing
  - Parallel inference helpers for stratum-level and sweep-level estimation

The cleaning entry point is ``clean_ai_rewrite`` (single text) or
``clean_ai_corpus`` (DataFrame batch).  The estimation entry point is
``MLEEstimator``.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ===================================================================
# Constants
# ===================================================================

DOC_TYPES = ["public_submission", "notice", "rule", "proposed_rule"]
DEFAULT_HUMAN_CUTOFF = "2022-11-30"
DEFAULT_OOV_LOG_PROB = -13.8
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_MIN_HUMAN_COUNT = 5
DEFAULT_MIN_AI_COUNT = 3
DEFAULT_MIN_HUMAN_FRAC = None
DEFAULT_MIN_AI_FRAC = None
DEFAULT_MIN_SENTENCES = 50

# Rough chars-per-token ratio; conservative to avoid exceeding context windows.
# Regulatory text is token-dense (acronyms, numbers), so we use a low ratio.
_CHARS_PER_TOKEN = 2.5


# ===================================================================
# Cleaning section — AI-rewrite corpus heuristics
# ===================================================================

# ---------------------------------------------------------------------------
# Compiled patterns (order matters -- applied sequentially)
# ---------------------------------------------------------------------------

# XML-style wrappers the model sometimes adds
_XML_OPEN_RE = re.compile(
    r"^\s*[_(\">]*<[_]?(?:rewritten[_\s-]*comment[_\s]*\d*|rewritten[_\s-]*response|comment"
    r"|new\s+version\s+of\s+comment|_END_OF_COMMENT_)>[)\"]*\s*",
    re.IGNORECASE,
)
_XML_CLOSE_RE = re.compile(
    r"\s*</(?:rewritten[_\s-]*comment[_\s]*\d*|rewritten[_\s-]*response|comment)>\s*$",
    re.IGNORECASE,
)

# Multi-comment continuation: model writes <<COMMENT N>> or </comment> then
# starts another comment.  We want everything BEFORE the first such boundary.
# Variants found in the wild: <<COMMENT 2>>, <<COMMENT_2>>, <<COMMENT END>>,
# <<COMMENT 1 END>>, << COMMENT 2>>, <<<< COMMENT 2 >>>>, <<COMMENT>>
_MULTI_COMMENT_RE = re.compile(
    r"</comment\s*\d*\s*>"          # </comment>, </comment 1>, </COMMENT 2>
    r"|<comment\s*\d+>"             # <COMMENT 2> (with number = continuation)
    r"|<END\s+OF\s+COMMENT"         # <END OF COMMENT 1>
    r"|<{2,}\s*(?:REWRITTEN[\s_]*)?COMMENT[\s_]*(?:\d+)?[\s_]*(?:END)?[\s_]*>{2,}"
    r"|<rewritten_response>"
    r"|<rewritten_comment>"
    r"|<\|reserved_special_token"      # Llama special tokens
    r"|</rewritten_comment>"           # mid-text closing tag
    r"|<{2,}\s*(?:END\s*)?COMMENT"     # bare <<COMMENT or <<ENDCOMMENT
    r"|<{3,}"                          # <<< (stray angle brackets)
    r"|<{2,}\s*PAGE\s*\d"              # <<PAGE N>> (model paginating)
    r"|-{3,}\s*ENDED?\s+COMMENT"         # ---ENDED COMMENT---
    r"|<-+\s*End\s+of\s+Comment"         # <- End of Comment 1 ->
    r"|<-+\s*COMMENT\s*\d"               # <-COMMENT 1>>
    r"|<new\s+version"                    # <new version of comment>
    r"|</assistant"                       # leaked model closing tag
    r"|<resubmit"                         # <resubmitted comment>, <resubmission>
    r"|<NAME>"                            # placeholder tags
    r"|<DATE>",
    re.IGNORECASE,
)

# Leading junk after XML stripping: model echoing metadata fields
# Patterns: "false\n\n1\n\n", "THISCOMMENT\nfalse\n", "COMMENT 2>>\nfalse\n",
# "No\n\n", "<resco>", ";<essay>", etc.
_LEADING_JUNK_RE = re.compile(
    r"^\s*(?:<\w+>)?"                        # optional stray tag like <resco>
    r"(?:;?\s*<?\w*>?)?"                     # optional ;<essay> etc
    r"\s*(?:COMMENT\s*\d*\s*>{0,2})?"        # optional COMMENT N>>
    r"\s*(?:THISCOMMENT|No)?"                 # optional THISCOMMENT or No
    r"\s*(?:false|true)\s*\n+"               # the "false" or "true" metadata
    r"(?:\s*\d+\s*\n+)?",                    # optional number after
    re.IGNORECASE,
)

# Self-chat: model talks about the rewriting task itself
_SELF_CHAT_RE = re.compile(
    r"^(?:"
    r"(?:Please )?[Rr]ewrite (?:the|this) (?:following |above )?comment"
    r"|[Rr]ewrite the following"
    r"|[Dd]rafting this comment from scratch"
    r"|[Hh]ere is the next comment"
    r"|[Tt]he (?:original|following) comment"
    r"|[Nn]ow,? (?:here is|I will|let me)"
    r"|was rewritten to"
    r"|[Tt]his rewritten comment"
    r"|[Mm]y rewritten comment"
    r"|[Ee]nd of (?:rewritten )?comment"
    r"|the rewritten comment (?:as|for|is|meets|maintains)"
    r"|[Ff]inal [Aa]nswer"
    r"|[Ss]core:\s*\d"
    r"|becomes:?\s*$"
    r"|[Tt]he best answer is"
    r"|I made the following changes"
    r"|I am trying to maintain"
    r"|I (?:also )?(?:changed|kept|made|rewrote|used)"
    r"|_optional_ provide"
    r"|Your (?:rewritten )?comment (?:is|should)"
    r"|rewritten_response:"
    r"|.{0,40}(?:the|to the) original comment"
    r"|.{0,20}rewritten (?:comment|version|response) (?:is|also|maintains|meets|will|that)"
    r"|<end of rewritten"
    r"|is too long and exceeds"
    r"|is not accurate to the"
    r"|was removed (?:by|and)"
    r"|seems to be a? ?corrupted"
    r"|appears to be an? (?:image|scan)"
    r"|[Ll]et\'?s start fresh"
    r"|is removed,? and (?:the|this)"
    r"|was altered to:?"
    r"|this is the rewritten"
    r"|New [Cc]omment:?"
    r"|(?:Here is|Here\'s) (?:the|a) (?:different|new|revised|alternative) (?:version|comment)"
    r"|[Mm]y revised comment"
    r"|[Rr]evised comment:?"
    r"|Invoice:"
    r"|I have rewritten the comment"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# Meta-commentary that the model appends after the actual rewrite.
# These are full-line patterns -- we truncate at the first match.
_META_LINE_STARTS = re.compile(
    r"^(?:"
    r"(?:\()?Note:?\s"
    r"|Note that\s"
    r"|Please (?:ensure|let me know|go ahead|note|provide)"
    r"|I hope this"
    r"|I would (?:be happy|like to)"
    r"|I\'d be happy"
    r"|I\'m (?:happy to|glad)"
    r"|Your rewritten comment"
    r"|Here is (?:the|a) (?:rewritten|new|revised|alternative)"
    r"|Here\'s (?:the|a) (?:rewritten|new|revised)"
    r"|Let me know"
    r"|This is the complete response"
    r"|Therefore,? the response remains"
    r"|Now,? let\'s try another"
    r"|Let\'s try another"
    r"|If you (?:have any|need|would like)"
    r"|You\'re welcome"
    r"|Thank you for (?:the opportunity|your)"
    r"|I look forward"
    r"|I\'ll be in touch"
    r"|I will (?:provide|talk|make sure)"
    r"|Submission of this comment does not"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# "Here is the rewritten comment:" as a PREFIX (model preamble before the actual text)
# Prefix: model preamble before the actual rewrite text.
# Matches "Here is the rewritten comment:", "completely rewritten comment:",
# "public comment rewritten", "and here is the rewritten comment", etc.
# Key insight: must match the FULL phrase including trailing "of the comment:"
# e.g., "Here is a rewritten version of the comment:" -- not just "Here is a rewritten version"
_PREFIX_RE = re.compile(
    r"^\s*(?:and\s+)?(?:Here is|Here\'s)\s+(?:the|a|my)\s+(?:rewritten|new|revised|alternative|different)\s+(?:comment|version|response)(?:\s+of\s+the\s+comment)?\s*:?\s*"
    r"|^\s*(?:completely |public comment )?rewritten\s*(?:comment|version)?\s*:?\s*\n"
    r"|^\s*(?:on\s*)?[Ss]ubmitting a rewritten version of the comment>?\s*\n"
    r"|^\s*(?:version of the comment\s*(?:rewritten|revised)?|of the comment|was (?:not )?provided)\s*:?\s*\n"
    r"|^\s*(?:New [Cc]omment|New [Vv]ersion|Version of [Cc]omment|Here is the (?:comment|new comment)|Revised [Cc]omment)\s*:?\s*\n"
    r"|^\s*public comment submitted by\s+<[^>]+>\s+on\s+<[^>]+>[^:]*:?\s*\n"
    r"|^\s*[Bb]elow is a rewritten version[^\n]*:?\s*\n"
    r"|^\s*SIGNING OFF\s*\n",
    re.IGNORECASE | re.MULTILINE,
)

# Placeholder signatures: [Your Name], [Your Address], [City, State ZIP]
_PLACEHOLDER_SIG_RE = re.compile(
    r"\[Your (?:Name|Address|City|State|Zip|Email|Phone)[^\]]*\]"
    r"|\[(?:NAME|CONTACT INFORMATION|SIGNATURE|ADDRESS|EMAIL|PHONE"
    r"|Attachment:[^\]]*|Cook Group Representative|Removed[^\]]*)\]",
    re.IGNORECASE,
)

# Trailing "Anytown, XX 12345" fake addresses (common Llama artifact)
_FAKE_ADDRESS_RE = re.compile(
    r"\bAnytown,?\s+[A-Z]{2,3}\s+\d{5}\b",
)

# Repeated sign-off artifacts: "Sincerely," appearing more than once
# (model rewrites, then repeats with a different name)
_DOUBLE_SIGNOFF_RE = re.compile(
    r"(Sincerely,.*?)(?=Sincerely,)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------


def clean_ai_rewrite(text: str, max_ratio: float = 3.0,
                     original_len: Optional[int] = None) -> str:
    """Clean a single AI-rewritten text by removing model artifacts.

    Steps applied in order:
      1. Strip XML wrapper tags (<rewritten_comment>...</rewritten_comment>)
      2. Strip "Here is the rewritten comment:" prefix
      3. Truncate at first multi-comment boundary (</comment>, <<COMMENT N>>)
      4. Truncate at first meta-commentary line
      5. Remove placeholder signatures ([Your Name] etc.)
      6. Remove fake addresses (Anytown, XX 12345)
      7. Length-ratio guard: if result is still >max_ratio of original, truncate

    Parameters
    ----------
    text : str
        The raw AI-generated text.
    max_ratio : float
        Maximum allowed length ratio (ai / original). Text beyond this is
        truncated at the nearest paragraph boundary.
    original_len : int, optional
        Character length of the original human text. Used for ratio guard.

    Returns
    -------
    str
        Cleaned text, or empty string if nothing salvageable remains.
    """
    if not text or not isinstance(text, str):
        return ""

    t = text.strip()

    # 1. Strip XML wrappers
    t = _XML_OPEN_RE.sub("", t)
    t = _XML_CLOSE_RE.sub("", t)

    # 1b. Strip leading ">", '"' remnants from partial tag stripping
    t = t.lstrip('>"').strip()

    # 2. Strip prefix ("Here is the rewritten comment:")
    t = _PREFIX_RE.sub("", t)

    # 2b. Strip leading metadata junk ("false\n\n1\n\n")
    t = _LEADING_JUNK_RE.sub("", t)

    # 2c. Strip leading stray characters left over from partial tag/preamble stripping
    # e.g., ":", "**", ")", ",", "[]", single digits, "EUR", "No", ";", "is", '">>',  '"'
    t = re.sub(
        r'^\s*(?:[:\)\],;*\[\.\">]+|EUR|\d{1,2}|No|is)\s*\n',
        "", t, flags=re.IGNORECASE,
    ).strip()

    # 2d. Strip leading URL artifacts like <http://www.regulations.gov>
    t = re.sub(r"^\s*<https?://[^>]+>\s*\n?", "", t).strip()

    # 3. Truncate at first multi-comment boundary
    m = _MULTI_COMMENT_RE.search(t)
    if m:
        t = t[:m.start()]

    # 3b. Truncate at first self-chat line
    m = _SELF_CHAT_RE.search(t)
    if m:
        t = t[:m.start()]

    # 4. Truncate at first meta-commentary line
    m = _META_LINE_STARTS.search(t)
    if m:
        t = t[:m.start()]

    # 5. Remove placeholder signatures
    t = _PLACEHOLDER_SIG_RE.sub("", t)

    # 6. Remove fake addresses
    t = _FAKE_ADDRESS_RE.sub("", t)

    # 7. Clean up trailing whitespace / stray punctuation
    t = t.strip()
    # Remove trailing stray closing tags that might remain
    t = re.sub(r"\s*</?\w+>\s*$", "", t).strip()

    # 7b. Remove trailing metadata/numbers blocks
    # Pattern: line of mostly digits, "false"/"true", and whitespace
    t = re.sub(r"\n\s*(?:[\d\s]+(?:false|true)[\d\s]*)+\s*$", "", t, flags=re.IGNORECASE).strip()

    # 7c. Remove trailing "false" or "true" with optional numbers
    t = re.sub(r"\n\s*(?:false|true)\s*(?:\n\s*\d+\s*)*$", "", t, flags=re.IGNORECASE).strip()

    # 7d. Remove trailing stray single digit, "PAGE N", "end.", "false end."
    t = re.sub(r"\s+(?:PAGE\s*\d+|\d{1,2}|false\s*end\.?|end\.?)$", "", t, flags=re.IGNORECASE).strip()

    # 7e. Remove trailing stray non-alpha junk on last line
    t = re.sub(r"\n\s*[×•·]+\s*$", "", t).strip()

    # 8. Length-ratio guard
    if original_len and original_len > 0 and max_ratio > 0:
        max_chars = int(original_len * max_ratio)
        if len(t) > max_chars:
            # Truncate at nearest paragraph boundary
            cut = t[:max_chars].rfind("\n\n")
            if cut > max_chars * 0.5:
                t = t[:cut].strip()
            else:
                # Fall back to sentence boundary
                cut = t[:max_chars].rfind(". ")
                if cut > max_chars * 0.3:
                    t = t[:cut + 1].strip()
                else:
                    t = t[:max_chars].strip()

    # Final check: if too short after cleaning, it's unsalvageable
    if len(t) < 20:
        return ""

    return t


# ---------------------------------------------------------------------------
# Batch cleaning for a parquet AI corpus
# ---------------------------------------------------------------------------


def clean_ai_corpus(df, text_col: str = "ai_text",
                    orig_col: str = "original_text",
                    max_ratio: float = 3.0) -> "pd.DataFrame":
    """Clean an AI corpus DataFrame in place and drop unsalvageable rows.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns `text_col` and optionally `orig_col`.
    text_col : str
        Column containing AI-generated text.
    orig_col : str
        Column containing original human text (for length-ratio guard).
    max_ratio : float
        Maximum ai/original length ratio.

    Returns
    -------
    pd.DataFrame
        Cleaned copy with bad rows dropped.
    """
    df = df.copy()
    orig_lens = (
        df[orig_col].str.len() if orig_col in df.columns else pd.Series([None] * len(df))
    )
    df[text_col] = [
        clean_ai_rewrite(text, max_ratio=max_ratio, original_len=olen)
        for text, olen in zip(df[text_col], orig_lens)
    ]
    # Drop rows where cleaning produced empty text
    before = len(df)
    df = df[df[text_col].str.len() >= 20].reset_index(drop=True)
    n_dropped = before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} unsalvageable rows ({n_dropped / before * 100:.1f}%)")
    return df


# ===================================================================
# Tokenization
# ===================================================================

_WORD_RE = re.compile(r"\b\w+\b")

try:
    from blingfire import text_to_sentences as _bf_sent_tokenize
    _USE_BLINGFIRE = True
except ImportError:
    _USE_BLINGFIRE = False


def ensure_nltk_punkt():
    """Download the NLTK punkt tokenizer if blingfire is not available."""
    if _USE_BLINGFIRE:
        return
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


# Backward-compat alias
_ensure_nltk_punkt = ensure_nltk_punkt


def sent_split(text: str) -> List[str]:
    """Split text into sentences using blingfire (fast) or NLTK (fallback)."""
    if _USE_BLINGFIRE:
        return _bf_sent_tokenize(text).split("\n")
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)


# Backward-compat alias
_sent_split = sent_split


def tokenize_text(text: str) -> List[List[str]]:
    """Split text into sentences; each sentence is a list of lowercased words."""
    if not text or not isinstance(text, str):
        return []
    sentences = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sent in sent_split(paragraph):
            words = _WORD_RE.findall(sent.lower())
            words = [w for w in words if not w.isdigit()]
            if len(words) > 10:
                sentences.append(words)
    return sentences


# ===================================================================
# Date / time helpers
# ===================================================================


def parse_posted_date(date_str) -> Optional[datetime]:
    """Parse a Posted Date string (ISO-8601, possibly with trailing 'Z')."""
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    try:
        clean = date_str.strip().rstrip("Z")
        return datetime.fromisoformat(clean)
    except ValueError:
        return None


def assign_quarter(date_str) -> Optional[str]:
    """Return a 'YYYYQn' string for the given date, or None."""
    dt = parse_posted_date(date_str)
    if dt is None:
        return None
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{q}"


def assign_half(date_str) -> Optional[str]:
    """Return a 'YYYYHn' string (semi-annual) for the given date, or None."""
    dt = parse_posted_date(date_str)
    if dt is None:
        return None
    h = 1 if dt.month <= 6 else 2
    return f"{dt.year}H{h}"


def assign_year(date_str) -> Optional[int]:
    """Return the year for the given date, or None."""
    dt = parse_posted_date(date_str)
    return dt.year if dt else None


def is_pre_chatgpt(date_str, cutoff: str = DEFAULT_HUMAN_CUTOFF) -> bool:
    """Return True if date_str is before the cutoff (default: 2022-11-30)."""
    dt = parse_posted_date(date_str)
    if dt is None:
        return False
    return dt < datetime.fromisoformat(cutoff)


# ===================================================================
# File I/O helpers
# ===================================================================


def iter_input_files(base_dir: Path, doc_type: str) -> Iterable[Path]:
    """Iterate over ``<agency>/<agency_year>/<doc_type>_all_text.csv[.gz]`` files.

    When both .csv and .csv.gz exist for the same stem, prefers .csv.gz —
    the .gz files are from a later pipeline run that uses clean HTML text,
    while .csv files are from an earlier run with bloated OCR'd PDF text
    (whose source PDFs have since been deleted).
    """
    plain = sorted(base_dir.glob(f"*/*/{doc_type}_all_text.csv"))
    gz = sorted(base_dir.glob(f"*/*/{doc_type}_all_text.csv.gz"))
    # Prefer .csv.gz; deduplicate by stem so .csv is only used as fallback
    seen = set()
    result = []
    for p in gz + plain:
        key = p.with_suffix("").with_suffix("")  # strip .csv.gz or .csv
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def load_dedup_representatives(base_dir: Path, doc_type: str) -> Optional[Set[str]]:
    """Load minhash dedup mappers and return the set of cluster-representative doc IDs.

    For each cluster, we keep the first document ID as the representative.
    Mapper files live at:
      ``<base_dir>/<agency>/<agency_year>/<doc_type>_all_text__dedup_mapper.csv.gz``
    with columns: agency_id, docket_id, document_id, cluster_id, cluster_uid
    """
    mapper_files = sorted(
        list(base_dir.glob(f"*/*/{doc_type}_all_text__dedup_mapper.csv.gz"))
        + list(base_dir.glob(f"*/*/{doc_type}_all_text__dedup_mapper.csv"))
    )
    if not mapper_files:
        return None

    representatives: Set[str] = set()
    n_total = 0
    n_clusters = 0
    for mapper_path in mapper_files:
        try:
            mapper = pd.read_csv(mapper_path, usecols=["document_id", "cluster_uid"])
        except (ValueError, KeyError):
            continue
        n_total += len(mapper)
        # Keep only the first document per cluster as representative
        reps = mapper.groupby("cluster_uid")["document_id"].first()
        representatives.update(reps.values.astype(str))
        n_clusters += len(reps)

    logging.info(
        "%s dedup: %d total docs -> %d clusters -> %d representatives",
        doc_type, n_total, n_clusters, len(representatives),
    )
    return representatives


# ===================================================================
# Text truncation
# ===================================================================


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to approximately fit a token budget."""
    if not text:
        return ""
    return text[:max_chars]


# Backward-compat alias
_truncate_text = truncate_text


# ===================================================================
# Batch helper
# ===================================================================


def batch_iter(iterable: list, size: int) -> Iterable[list]:
    """Yield successive chunks of ``size`` from ``iterable``."""
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


# Backward-compat alias
_batch = batch_iter


# ===================================================================
# Log probability computation
# ===================================================================


def sentence_log_probs_raw(sentences, w2i, delta_p, delta_q, baseline_p, baseline_q):
    """Full Bernoulli log probs from raw arrays (no estimator object needed).

    For each sentence s, computes:
        log P(s) = baseline_p + sum_{w in s} delta_p[w]
        log Q(s) = baseline_q + sum_{w in s} delta_q[w]

    Parameters
    ----------
    sentences : list of list of str
        Each inner list is a tokenized sentence (lowercased words).
    w2i : dict[str, int]
        Word-to-index mapping into the delta arrays.
    delta_p, delta_q : np.ndarray
        Per-word deltas: log P(w) - log(1-P(w)).
    baseline_p, baseline_q : float
        Sum of log(1-P(w)) over all vocabulary words.

    Returns
    -------
    log_p, log_q : np.ndarray
        Per-sentence log-probabilities under human (P) and AI (Q) models.
    """
    n = len(sentences)
    log_p = np.full(n, baseline_p)
    log_q = np.full(n, baseline_q)
    for i, sent in enumerate(sentences):
        indices = []
        for word in set(sent):
            idx = w2i.get(word)
            if idx is not None:
                indices.append(idx)
        if indices:
            idx_arr = np.array(indices, dtype=np.intp)
            log_p[i] += delta_p[idx_arr].sum()
            log_q[i] += delta_q[idx_arr].sum()
    return log_p, log_q


# Backward-compat alias
_sentence_log_probs_raw = sentence_log_probs_raw


# ===================================================================
# MLE Estimation
# ===================================================================


class MLEEstimator:
    """Maximum-likelihood estimator for the AI-written fraction alpha.

    Implements Distributional GPT Detection (Liang et al. 2025) with a
    full Bernoulli word-presence model, MLE via bounded scalar optimization,
    and bootstrap confidence intervals.
    """

    def __init__(self, distribution_path: Path, oov_log_prob: float = DEFAULT_OOV_LOG_PROB):
        dist_df = pd.read_parquet(distribution_path)
        self.vocab = set(dist_df["word"].values)
        self.logP = dict(zip(dist_df["word"], dist_df["logP"]))
        self.logQ = dict(zip(dist_df["word"], dist_df["logQ"]))
        self.log1mP = dict(zip(dist_df["word"], dist_df["log1mP"]))
        self.log1mQ = dict(zip(dist_df["word"], dist_df["log1mQ"]))
        self.oov_log_prob = oov_log_prob

        # Pre-compute vectors for full Bernoulli model (Liang et al. 2025).
        # Each sentence's log-prob includes present words (logP/logQ) and
        # absent words (log(1-P)/log(1-Q)).
        vocab_list = list(self.vocab)
        self._word_to_idx = {w: i for i, w in enumerate(vocab_list)}
        self._logP_arr = np.array([self.logP[w] for w in vocab_list])
        self._logQ_arr = np.array([self.logQ[w] for w in vocab_list])
        self._log1mP_arr = np.array([self.log1mP[w] for w in vocab_list])
        self._log1mQ_arr = np.array([self.log1mQ[w] for w in vocab_list])
        # Baseline: sum of log(1-P(w)) and log(1-Q(w)) over ALL vocab words
        # (the probability of a sentence where no words appear)
        self._baseline_p = float(self._log1mP_arr.sum())
        self._baseline_q = float(self._log1mQ_arr.sum())
        # Delta: log(P(w)) - log(1-P(w)) for each word (added when word is present)
        self._delta_p = self._logP_arr - self._log1mP_arr
        self._delta_q = self._logQ_arr - self._log1mQ_arr

    def sentence_log_probs(
        self, sentences: List[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full Bernoulli log-probabilities (Liang et al. 2025).

        For each sentence: log P(s) = baseline_p + sum_{w present} delta_p[w]
        where baseline_p = sum_{all w} log(1-P(w))
        and delta_p[w] = log P(w) - log(1-P(w))
        """
        return sentence_log_probs_raw(
            sentences, self._word_to_idx, self._delta_p, self._delta_q,
            self._baseline_p, self._baseline_q
        )

    @staticmethod
    def neg_log_likelihood(alpha: float, log_ratio: np.ndarray) -> float:
        """Paper's reparameterized likelihood: -mean(log((1-a) + a * exp(logQ-logP)))."""
        return -np.mean(np.log(np.maximum(
            (1 - alpha) + alpha * np.exp(log_ratio), 1e-300
        )))

    # Keep the old private name working
    _neg_log_likelihood = neg_log_likelihood

    def estimate_alpha(self, log_ratio: np.ndarray) -> float:
        """Find the MLE alpha given pre-computed log-ratios."""
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            self.neg_log_likelihood,
            bounds=(1e-6, 1 - 1e-6),
            args=(log_ratio,),
            method="bounded",
        )
        return float(result.x)

    def bootstrap_ci(
        self,
        log_p: np.ndarray,
        log_q: np.ndarray,
        n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """Return (alpha_point, ci_lower_2.5%, ci_upper_97.5%)."""
        log_ratio = log_q - log_p
        rng = np.random.RandomState(seed)
        n = len(log_ratio)
        alphas = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            alphas.append(self.estimate_alpha(log_ratio[idx]))
        alpha_point = self.estimate_alpha(log_ratio)
        return alpha_point, float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))

    def inference(
        self,
        sentences: List[List[str]],
        n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    ) -> Tuple[float, float, float, int]:
        """Returns (alpha, ci_lower, ci_upper, n_sentences_used)."""
        filtered = [s for s in sentences if set(s) & self.vocab]
        if len(filtered) < 10:
            return float("nan"), float("nan"), float("nan"), len(filtered)
        log_p, log_q = self.sentence_log_probs(filtered)
        alpha, lo, hi = self.bootstrap_ci(log_p, log_q, n_bootstrap)
        return alpha, lo, hi, len(filtered)

    def est_data_tuple(self):
        """Return (w2i, delta_p, delta_q, baseline_p, baseline_q, vocab_set) for parallel tasks."""
        return (
            self._word_to_idx, self._delta_p, self._delta_q,
            self._baseline_p, self._baseline_q, self.vocab,
        )


# ===================================================================
# Distribution building
# ===================================================================


def build_distribution(
    human_counts: Counter,
    n_human: int,
    ai_counts: Counter,
    n_ai: int,
    min_human_count: int = DEFAULT_MIN_HUMAN_COUNT,
    min_ai_count: int = DEFAULT_MIN_AI_COUNT,
    min_human_frac: float = None,
    min_ai_frac: float = None,
    max_vocab: int = None,
) -> Optional[pd.DataFrame]:
    """Build a word-level P/Q distribution from human and AI word counts.

    Returns a DataFrame with columns: word, logP, logQ, log1mP, log1mQ,
    human_count, ai_count, log_odds_ratio.  Returns None if the resulting
    vocabulary is empty.
    """
    human_log_probs = {w: np.log(c / n_human) for w, c in human_counts.items()}
    ai_log_probs = {w: np.log(c / n_ai) for w, c in ai_counts.items()}

    common_vocab = set(human_counts.keys()) & set(ai_counts.keys())
    min_h = int(n_human * min_human_frac) if min_human_frac else min_human_count
    min_a = int(n_ai * min_ai_frac) if min_ai_frac else min_ai_count
    common_vocab = {
        w for w in common_vocab
        if human_counts[w] >= min_h and ai_counts[w] >= min_a
    }
    if max_vocab and len(common_vocab) > max_vocab:
        ranked = sorted(common_vocab, key=lambda w: ai_counts[w], reverse=True)
        common_vocab = set(ranked[:max_vocab])

    if not common_vocab:
        return None

    rows = []
    for word in common_vocab:
        logP = human_log_probs[word]
        logQ = ai_log_probs[word]
        log1mP = np.log1p(-np.exp(logP))
        log1mQ = np.log1p(-np.exp(logQ))
        human_lo = logP - log1mP
        ai_lo = logQ - log1mQ
        lor = human_lo - ai_lo
        if np.isinf(lor) or np.isnan(lor):
            continue
        rows.append({
            "word": word,
            "logP": logP,
            "logQ": logQ,
            "log1mP": log1mP,
            "log1mQ": log1mQ,
            "human_count": human_counts[word],
            "ai_count": ai_counts[word],
            "log_odds_ratio": lor,
        })

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("log_odds_ratio").reset_index(drop=True)


# Backward-compat alias
_build_distribution = build_distribution


# ===================================================================
# Hierarchical Bayes helpers
# ===================================================================


def optimize_kappa(
    pool_freq: Dict[str, float],
    agency_word_counts: Dict[str, Tuple[Counter, int]],
    vocab: Set[str],
    kappa_bounds: Tuple[float, float] = (1.0, 1e6),
) -> float:
    """Find optimal concentration kappa by maximizing Beta-Binomial marginal likelihood.

    For each agency a and word w, the model is:
        P_a(w) ~ Beta(kappa * mu_w, kappa * (1 - mu_w))
    where mu_w = P_pool(w).  The marginal likelihood integrates out P_a(w):
        P(n_{a,w} | kappa) = BetaBinom(n_{a,w}; n_a, kappa * mu_w, kappa * (1 - mu_w))
    We optimize kappa over all agencies and words jointly.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import betaln

    words = sorted(vocab)
    mu = np.array([pool_freq[w] for w in words])
    mu = np.clip(mu, 1e-8, 1 - 1e-8)

    # Pre-collect per-agency count arrays
    agency_arrays = []
    for agency_id, (counts, n_a) in agency_word_counts.items():
        n_aw = np.array([counts.get(w, 0) for w in words], dtype=float)
        agency_arrays.append((n_aw, n_a))

    def neg_marginal_ll(log_kappa):
        kappa = np.exp(log_kappa)
        a_prior = kappa * mu
        b_prior = kappa * (1 - mu)
        prior_term = betaln(a_prior, b_prior)
        ll = 0.0
        for n_aw, n_a in agency_arrays:
            ll += np.sum(
                betaln(a_prior + n_aw, b_prior + (n_a - n_aw))
                - prior_term
            )
        return -ll

    result = minimize_scalar(
        neg_marginal_ll,
        bounds=(np.log(kappa_bounds[0]), np.log(kappa_bounds[1])),
        method="bounded",
    )
    optimal_kappa = float(np.exp(result.x))
    logging.info(
        "Optimal kappa = %.1f (log-marginal-lik = %.2f)", optimal_kappa, -result.fun
    )
    return optimal_kappa


# Backward-compat alias
_optimize_kappa = optimize_kappa


def shrink_p(
    pool_freq: np.ndarray,
    agency_counts_arr: np.ndarray,
    n_agency: int,
    kappa: float,
) -> np.ndarray:
    """Compute posterior-mean P_a(w) = (kappa * mu_w + n_{a,w}) / (kappa + n_a)."""
    mu = np.clip(pool_freq, 1e-10, 1 - 1e-10)
    a_post = kappa * mu + agency_counts_arr
    b_post = kappa * (1 - mu) + (n_agency - agency_counts_arr)
    return np.clip(a_post / (a_post + b_post), 1e-10, 1 - 1e-10)


# Backward-compat alias
_shrink_p = shrink_p


def build_agency_est_data(
    dist_df: pd.DataFrame,
    agency_word_counts: Dict[str, Tuple[Counter, int]],
    kappa: float,
    agency_id: str,
    agency_ai_word_counts: Optional[Dict[str, Tuple[Counter, int]]] = None,
    kappa_q: Optional[float] = None,
) -> Tuple:
    """Build est_data tuple with agency-specific P and Q (both shrunk).

    Returns the same (w2i, delta_p, delta_q, baseline_p, baseline_q, vocab_set)
    tuple expected by ``infer_stratum``.
    """
    vocab_list = list(dist_df["word"].values)
    w2i = {w: i for i, w in enumerate(vocab_list)}
    vocab_set = set(vocab_list)

    # --- Agency-specific P ---
    mu_p = np.exp(dist_df["logP"].values.astype(float))

    if agency_id in agency_word_counts:
        word_counts, n_a = agency_word_counts[agency_id]
        n_aw = np.array([word_counts.get(w, 0) for w in vocab_list], dtype=float)
    else:
        n_aw = np.zeros(len(vocab_list))
        n_a = 0

    p_shrunk = shrink_p(mu_p, n_aw, n_a, kappa)
    logP = np.log(p_shrunk)
    log1mP = np.log(1 - p_shrunk)

    # --- Agency-specific Q (or pooled fallback) ---
    if agency_ai_word_counts is not None and kappa_q is not None and agency_id in agency_ai_word_counts:
        mu_q = np.exp(dist_df["logQ"].values.astype(float))
        ai_counts, n_ai_a = agency_ai_word_counts[agency_id]
        n_ai_aw = np.array([ai_counts.get(w, 0) for w in vocab_list], dtype=float)
        q_shrunk = shrink_p(mu_q, n_ai_aw, n_ai_a, kappa_q)
        logQ = np.log(q_shrunk)
        log1mQ = np.log(1 - q_shrunk)
    else:
        logQ = dist_df["logQ"].values.astype(float)
        log1mQ = dist_df["log1mQ"].values.astype(float)

    baseline_p = float(log1mP.sum())
    baseline_q = float(log1mQ.sum())
    delta_p = logP - log1mP
    delta_q = logQ - log1mQ

    return (w2i, delta_p, delta_q, baseline_p, baseline_q, vocab_set)


# Backward-compat alias
_build_agency_est_data = build_agency_est_data


def load_agency_word_counts(path: Path) -> Dict[str, Tuple[Counter, int]]:
    """Load per-agency word counts from parquet.

    Returns a dict mapping agency_id -> (Counter of word counts, n_sentences).
    """
    df = pd.read_parquet(path)
    result = {}
    for agency_id, grp in df.groupby("agency_id"):
        counts = Counter(dict(zip(grp["word"], grp["count"])))
        n_sents = int(grp["n_sentences"].iloc[0])
        result[agency_id] = (counts, n_sents)
    return result


# Backward-compat alias
_load_agency_word_counts = load_agency_word_counts


# ===================================================================
# Data loading helpers (top-level for multiprocessing pickling)
# ===================================================================


def process_human_csv(args_tuple):
    """Process a single CSV file: read, filter, tokenize, and return streaming counts.

    Must be a top-level function for multiprocessing pickling.

    Parameters
    ----------
    args_tuple : tuple
        (csv_path, human_cutoff, dedup_reps, hierarchical_unused,
         [agencies_filter, [doc_ids_filter]])

    Returns
    -------
    tuple
        (counts: Counter, n_sents: int, n_docs: int, n_dedup_skipped: int,
         agency_accum: dict or None)
    """
    csv_path, human_cutoff, dedup_reps, _hierarchical_unused = args_tuple[:4]
    agencies_filter = args_tuple[4] if len(args_tuple) > 4 else None
    doc_ids_filter = args_tuple[5] if len(args_tuple) > 5 else None
    csv_path = Path(csv_path)

    # Always read Document ID so we can deduplicate across files
    cols = ["Posted Date", "canonical_text", "Agency ID", "Document ID"]
    try:
        df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    except (ValueError, KeyError):
        return Counter(), 0, 0, 0, None, set()

    df = df.dropna(subset=["canonical_text", "Posted Date"])
    df["Document ID"] = df["Document ID"].astype(str)
    df = df[df["Posted Date"].apply(lambda d: is_pre_chatgpt(d, human_cutoff))]
    df = df[df["canonical_text"].str.len() >= 100]
    if agencies_filter:
        df = df[df["Agency ID"].isin(agencies_filter)]

    n_dedup_skipped = 0
    if dedup_reps is not None:
        before_len = len(df)
        df = df[df["Document ID"].isin(dedup_reps)]
        n_dedup_skipped = before_len - len(df)

    if doc_ids_filter is not None:
        df = df[df["Document ID"].isin(doc_ids_filter)]

    # Deduplicate within this file
    df = df.drop_duplicates(subset=["Document ID"], keep="first")

    counts = Counter()
    n_sents = 0
    agency_accum = {}
    doc_ids_seen = set(df["Document ID"])

    for agency_id, text in zip(df["Agency ID"], df["canonical_text"]):
        for sent in tokenize_text(text):
            word_set = set(sent)
            counts.update(word_set)
            n_sents += 1
            acc = agency_accum.setdefault(
                agency_id, {"counts": Counter(), "n_sents": 0}
            )
            acc["counts"].update(word_set)
            acc["n_sents"] += 1

    return counts, n_sents, len(df), n_dedup_skipped, agency_accum, doc_ids_seen


# Backward-compat alias
_process_human_csv = process_human_csv


def process_ai_texts(texts):
    """Process a batch of AI texts: tokenize and return streaming counts.

    Parameters
    ----------
    texts : list of str
        AI-generated text strings.

    Returns
    -------
    tuple
        (counts: Counter, n_sents: int)
    """
    counts = Counter()
    n_sents = 0
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        for sent in tokenize_text(text):
            counts.update(set(sent))
            n_sents += 1
    return counts, n_sents


# Backward-compat alias
_process_ai_texts = process_ai_texts


def load_and_tokenize_file(
    csv_path: Path,
    agencies_filter: Optional[Set[str]],
    dedup_reps: Optional[Set[str]],
) -> List[Dict]:
    """Load one CSV, tokenize, return records.  Top-level for multiprocessing.

    Returns a list of dicts with keys: document_id, agency_id, quarter, year,
    sentences, n_sentences.
    """
    try:
        cols = ["Document ID", "Agency ID", "Posted Date", "canonical_text"]
        df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    except (ValueError, KeyError):
        return []
    df = df.dropna(subset=["canonical_text", "Posted Date"])
    df = df[df["canonical_text"].str.len() >= 50]
    if agencies_filter:
        df = df[df["Agency ID"].isin(agencies_filter)]
    if dedup_reps is not None:
        df = df[df["Document ID"].astype(str).isin(dedup_reps)]
    if df.empty:
        return []
    # Vectorized date parsing instead of per-row apply
    dt = pd.to_datetime(df["Posted Date"], errors="coerce")
    df = df[dt.notna()]
    dt = dt[dt.notna()]
    df["quarter"] = dt.dt.year.astype(str) + "Q" + ((dt.dt.month - 1) // 3 + 1).astype(str)
    df["half"] = dt.dt.year.astype(str) + "H" + (((dt.dt.month - 1) // 6 + 1).astype(str))
    df["year"] = dt.dt.year

    file_records = []
    for doc_id, text, agency, quarter, half, year in zip(
        df["Document ID"], df["canonical_text"], df["Agency ID"],
        df["quarter"], df["half"], df["year"],
    ):
        sents = tokenize_text(text)
        if sents:
            file_records.append(
                {
                    "document_id": str(doc_id),
                    "agency_id": agency,
                    "quarter": quarter,
                    "half": half,
                    "year": int(year),
                    "sentences": sents,
                    "n_sentences": len(sents),
                }
            )
    return file_records


# Backward-compat alias
_load_and_tokenize_file = load_and_tokenize_file


# ===================================================================
# Parallel inference helpers
# ===================================================================


def infer_stratum(task_tuple):
    """Module-level function for parallel stratum inference.

    Parameters
    ----------
    task_tuple : tuple
        (key, n_docs, sentences, est_data, n_bootstrap)
        where est_data = (w2i, delta_p, delta_q, baseline_p, baseline_q, vocab_set)
        -- all picklable.

    Returns
    -------
    tuple
        (key, n_docs, alpha, ci_lo, ci_hi, n_used)
    """
    from scipy.optimize import minimize_scalar

    key, n_docs, sentences, est_data, n_bootstrap = task_tuple
    w2i, delta_p, delta_q, baseline_p, baseline_q, vocab_set = est_data

    # Filter sentences to those with at least one vocab word
    filtered = [s for s in sentences if set(s) & vocab_set]
    n_used = len(filtered)
    if n_used < 10:
        return key, n_docs, float("nan"), float("nan"), float("nan"), n_used

    log_p, log_q = sentence_log_probs_raw(filtered, w2i, delta_p, delta_q, baseline_p, baseline_q)

    # Use paper's reparameterized likelihood: factor out P(sentence),
    # optimize over log((1-alpha) + alpha * exp(logQ - logP))
    log_ratio = log_q - log_p  # logQ(s) - logP(s) for each sentence

    def _neg_ll(alpha, lr):
        return -np.mean(np.log(np.maximum((1 - alpha) + alpha * np.exp(lr), 1e-300)))

    def _est_alpha(lr):
        result = minimize_scalar(
            _neg_ll, bounds=(1e-6, 1 - 1e-6), args=(lr,), method="bounded"
        )
        return float(result.x)

    rng = np.random.RandomState(42)
    n = len(log_ratio)
    alphas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        alphas.append(_est_alpha(log_ratio[idx]))
    alpha_point = _est_alpha(log_ratio)
    ci_lo = float(np.percentile(alphas, 2.5))
    ci_hi = float(np.percentile(alphas, 97.5))
    return key, n_docs, alpha_point, ci_lo, ci_hi, n_used


# Backward-compat alias
_infer_stratum = infer_stratum


def sweep_infer_stratum(task_tuple):
    """Top-level function for parallel sweep inference. Must be picklable.

    Parameters
    ----------
    task_tuple : tuple
        (key, n_docs, sent_indices, log_q, n_used, b_p, d_p, bootstrap_n)

    Returns
    -------
    dict
        {"key", "n_docs", "n_used", "alpha", "ci_lo", "ci_hi"}
    """
    from scipy.optimize import minimize_scalar

    key, n_docs, sent_indices, log_q, n_used, b_p, d_p, bootstrap_n = task_tuple

    # Compute log P per sentence
    log_p = np.full(n_used, b_p)
    for i, idxs in enumerate(sent_indices):
        if len(idxs):
            log_p[i] += d_p[idxs].sum()

    log_ratio = log_q - log_p

    def _neg_ll(alpha, lr):
        return -np.mean(np.log(np.maximum(
            (1 - alpha) + alpha * np.exp(lr), 1e-300
        )))

    def _est_alpha(lr):
        result = minimize_scalar(
            _neg_ll, bounds=(1e-6, 1 - 1e-6), args=(lr,), method="bounded"
        )
        return float(result.x)

    alpha_point = _est_alpha(log_ratio)
    rng = np.random.RandomState(42)
    n = len(log_ratio)
    alphas = []
    for _ in range(bootstrap_n):
        idx = rng.randint(0, n, n)
        alphas.append(_est_alpha(log_ratio[idx]))
    ci_lo = float(np.percentile(alphas, 2.5))
    ci_hi = float(np.percentile(alphas, 97.5))

    return {
        "key": key,
        "n_docs": n_docs,
        "n_used": n_used,
        "alpha": alpha_point,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


# Backward-compat alias
_sweep_infer_stratum = sweep_infer_stratum
