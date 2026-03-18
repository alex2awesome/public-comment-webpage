"""
Estimate LLM-assisted writing prevalence in regulatory documents using
Distributional GPT Detection (Liang et al. 2025).

Three subcommands:
    generate  — Build AI reference corpus by LLM-rewriting pre-ChatGPT docs
    estimate  — Compute word-level P/Q distributions from reference corpora (CPU)
    infer     — MLE estimation of AI fraction per stratum with bootstrap CIs (CPU)

Usage:
    # Step 1: generate AI rewrites from multiple model families
    # vLLM batch (on sk3 GPU — most efficient for open models):
    python scripts/corpus_ai_usage.py generate \\
        --models meta-llama/Llama-3.3-70B-Instruct --sample-per-model 100

    # API models (from any machine):
    python scripts/corpus_ai_usage.py generate \\
        --models gpt-4 gpt-4o gpt-5 claude-sonnet-4-6 --sample-per-model 100

    # Step 2 (CPU):
    python scripts/corpus_ai_usage.py estimate

    # Step 3 (CPU):
    python scripts/corpus_ai_usage.py infer --stratify-by agency quarter
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR.parent  # data/bulk_downloads/
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"

DOC_TYPES = ["public_submission", "notice", "rule", "proposed_rule"]
DEFAULT_HUMAN_CUTOFF = "2022-11-30"

DEFAULT_MODELS = [
    "gpt-4",
    "gpt-4o",
    "gpt-5",
    "claude-sonnet-4-6",
    "meta-llama/Llama-3.3-70B-Instruct",
]
DEFAULT_SAMPLE_PER_MODEL = 100
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8002/v1"

DEFAULT_OOV_LOG_PROB = -13.8
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_MIN_HUMAN_COUNT = 5
DEFAULT_MIN_AI_COUNT = 3
DEFAULT_MIN_SENTENCES = 50

REWRITE_PROMPT_TEMPLATES = {
    "public_submission": (
        "Below is a public comment submitted to a government regulatory agency. "
        "Write a new version of this comment that takes the same position and makes "
        "the same arguments on the same topic, but uses entirely different wording. "
        "Keep a similar length. Just write the comment, do NOT say anything else.\n\n"
        "<comment>{text}</comment>"
    ),
    "notice": (
        "Below is a government notice. Write a new version that covers the same "
        "regulatory content, structure, and topic, but uses entirely different wording. "
        "Keep a similar length. Just write the notice, do NOT say anything else.\n\n"
        "<notice>{text}</notice>"
    ),
    "rule": (
        "Below is a government rule. Write a new version that preserves the same "
        "regulatory requirements, structure, and scope, but uses entirely different "
        "wording. Keep a similar length. Just write the rule, do NOT say anything else.\n\n"
        "<rule>{text}</rule>"
    ),
    "proposed_rule": (
        "Below is a proposed regulatory rule. Write a new version that preserves the "
        "same regulatory requirements, structure, and scope, but uses entirely different "
        "wording. Keep a similar length. Just write the proposed rule, do NOT say anything else.\n\n"
        "<proposed_rule>{text}</proposed_rule>"
    ),
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _detect_backend(model_name: str) -> str:
    """Infer API backend from model name: 'openai', 'anthropic', or 'vllm'."""
    lower = model_name.lower()
    if lower.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if lower.startswith("claude"):
        return "anthropic"
    return "vllm"


def _batch(iterable: list, size: int) -> Iterable[list]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def parse_posted_date(date_str) -> Optional[datetime]:
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    try:
        clean = date_str.strip().rstrip("Z")
        return datetime.fromisoformat(clean)
    except ValueError:
        return None


def assign_quarter(date_str) -> Optional[str]:
    dt = parse_posted_date(date_str)
    if dt is None:
        return None
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{q}"


def assign_year(date_str) -> Optional[int]:
    dt = parse_posted_date(date_str)
    return dt.year if dt else None


def is_pre_chatgpt(date_str, cutoff: str = DEFAULT_HUMAN_CUTOFF) -> bool:
    dt = parse_posted_date(date_str)
    if dt is None:
        return False
    return dt < datetime.fromisoformat(cutoff)


def iter_input_files(base_dir: Path, doc_type: str) -> Iterable[Path]:
    return sorted(base_dir.glob(f"*/*/{doc_type}_all_text.csv"))


def load_dedup_representatives(base_dir: Path, doc_type: str) -> Optional[Set[str]]:
    """Load minhash dedup mappers and return the set of cluster-representative doc IDs.

    For each cluster, we keep the first document ID as the representative.
    Mapper files live at: <base_dir>/<agency>/<agency_year>/<doc_type>_all_text__dedup_mapper.csv
    with columns: agency_id, docket_id, document_id, cluster_id, cluster_uid
    """
    mapper_files = sorted(base_dir.glob(f"*/*/{doc_type}_all_text__dedup_mapper.csv"))
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
        "%s dedup: %d total docs → %d clusters → %d representatives",
        doc_type, n_total, n_clusters, len(representatives),
    )
    return representatives


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b[a-zA-Z]+\b")


def _ensure_nltk_punkt():
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def tokenize_text(text: str) -> List[List[str]]:
    """Split text into sentences; each sentence is a list of lowercased words."""
    from nltk.tokenize import sent_tokenize

    if not text or not isinstance(text, str):
        return []
    sentences = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sent in sent_tokenize(paragraph):
            words = [w.lower() for w in _WORD_RE.findall(sent)]
            if len(words) >= 3:
                sentences.append(words)
    return sentences


# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------

# Rough chars-per-token ratio; conservative to avoid exceeding context windows.
# Regulatory text is token-dense (acronyms, numbers), so we use a low ratio.
_CHARS_PER_TOKEN = 2.5


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to approximately fit a token budget."""
    if not text:
        return ""
    return text[:max_chars]


# ===================================================================
# Subcommand 1: generate
# ===================================================================


def _make_generate_fn_vllm_offline(
    model: str,
    *,
    max_tokens: int,
    temperature: float,
    max_model_len: int,
) -> Tuple[Callable[[List[str]], List[str]], Optional[Callable]]:
    """Return (generate_batch, cleanup_fn) using vLLM batch inference."""
    os.environ.setdefault("VLLM_DISABLE_PROGRESS_BAR", "1")
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected for vLLM offline mode")

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    llm_kwargs = {"max_model_len": max_model_len}
    gpu = torch.cuda.get_device_name(0).upper() if torch.cuda.is_available() else ""
    if any(tag in gpu for tag in ("B200", "H200")):
        llm_kwargs.update({"dtype": "auto", "gpu_memory_utilization": 0.95})
    llm = LLM(model=model, **llm_kwargs)

    def generate_batch(batch: List[str]) -> List[str]:
        outputs = llm.generate(batch, sampling_params)
        return [(o.outputs[0].text or "") if o.outputs else "" for o in outputs]

    def cleanup():
        nonlocal llm
        del llm
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    return generate_batch, cleanup


def _make_generate_fn_vllm_online(
    model: str,
    *,
    max_tokens: int,
    temperature: float,
    base_url: str,
    batch_size: int,
) -> Tuple[Callable[[List[str]], List[str]], None]:
    """Return (generate_batch, None) using a running vLLM server (OpenAI-compatible)."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")

    def _call_one(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return getattr(resp.choices[0].message, "content", "") or ""

    def generate_batch(batch: List[str]) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(batch), 16)) as pool:
            return list(pool.map(_call_one, batch))

    return generate_batch, None


def _make_generate_fn_openai(
    model: str, *, max_tokens: int, temperature: float
) -> Tuple[Callable[[List[str]], List[str]], None]:
    """Return (generate_batch, None) for OpenAI API models (GPT-4, etc.)."""
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY env var

    def _call_one(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return getattr(resp.choices[0].message, "content", "") or ""

    def generate_batch(batch: List[str]) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(batch), 16)) as pool:
            return list(pool.map(_call_one, batch))

    return generate_batch, None


def _make_generate_fn_anthropic(
    model: str, *, max_tokens: int, temperature: float
) -> Tuple[Callable[[List[str]], List[str]], None]:
    """Return (generate_batch, None) for Anthropic API models (Claude)."""
    from anthropic import Anthropic

    client = Anthropic()  # uses ANTHROPIC_API_KEY env var

    def _call_one(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""

    def generate_batch(batch: List[str]) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(batch), 8)) as pool:
            return list(pool.map(_call_one, batch))

    return generate_batch, None


def _make_generate_fn(
    model: str,
    backend: str,
    *,
    max_tokens: int,
    temperature: float,
    max_model_len: int,
    vllm_base_url: str,
    batch_size: int,
    offline: bool,
) -> Tuple[Callable[[List[str]], List[str]], Optional[Callable]]:
    """Dispatch to the right backend."""
    if backend == "openai":
        return _make_generate_fn_openai(model, max_tokens=max_tokens, temperature=temperature)
    if backend == "anthropic":
        return _make_generate_fn_anthropic(model, max_tokens=max_tokens, temperature=temperature)
    # vllm
    if offline:
        return _make_generate_fn_vllm_offline(
            model, max_tokens=max_tokens, temperature=temperature, max_model_len=max_model_len,
        )
    return _make_generate_fn_vllm_online(
        model, max_tokens=max_tokens, temperature=temperature,
        base_url=vllm_base_url, batch_size=batch_size,
    )


def cmd_generate(args: argparse.Namespace) -> None:
    logging.info("=== generate: building AI reference corpus ===")

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Max chars to send as input (conservative: ~6k tokens worth of text).
    max_input_chars = int((args.max_model_len - args.max_tokens) * _CHARS_PER_TOKEN)

    rng = np.random.RandomState(args.seed)

    # --- Load models FIRST to claim GPU before slow CSV reading ---
    model_fns: Dict[str, Tuple[Callable, Optional[Callable]]] = {}
    for model in args.models:
        backend = _detect_backend(model)
        use_offline = args.offline and backend == "vllm"
        logging.info(
            "Loading model %s (backend=%s, offline=%s)...", model, backend, use_offline
        )
        generate_batch, cleanup_fn = _make_generate_fn(
            model,
            backend,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_model_len=args.max_model_len,
            vllm_base_url=args.vllm_base_url,
            batch_size=args.batch_size,
            offline=use_offline,
        )
        model_fns[model] = (generate_batch, cleanup_fn)
        logging.info("Model %s loaded successfully.", model)

    try:
        _generate_with_models(args, base_dir, output_dir, max_input_chars, rng, model_fns)
    finally:
        # Clean up all models
        for model, (_, cleanup_fn) in model_fns.items():
            if cleanup_fn:
                logging.info("Cleaning up model %s...", model)
                cleanup_fn()


def _generate_with_models(
    args: argparse.Namespace,
    base_dir: Path,
    output_dir: Path,
    max_input_chars: int,
    rng: np.random.RandomState,
    model_fns: Dict[str, Tuple[Callable, Optional[Callable]]],
) -> None:
    """Run generation across doc_types using pre-loaded models."""

    for doc_type in args.doc_types:
        output_path = output_dir / f"ai_corpus_{doc_type}.parquet"

        # --- collect pre-ChatGPT source documents ---
        all_rows = []
        csv_files = list(iter_input_files(base_dir, doc_type))
        logging.info("%s: found %d CSV files to scan", doc_type, len(csv_files))
        for i, csv_path in enumerate(csv_files, 1):
            logging.info("  [%d/%d] reading %s", i, len(csv_files), csv_path.name)
            try:
                df = pd.read_csv(
                    csv_path,
                    usecols=["Document ID", "Agency ID", "Posted Date", "canonical_text"],
                    low_memory=False,
                )
            except (ValueError, KeyError):
                logging.info("    skipped (missing columns)")
                continue
            df = df.dropna(subset=["canonical_text", "Posted Date"])
            df = df[df["Posted Date"].apply(lambda d: is_pre_chatgpt(d, args.human_cutoff))]
            df = df[df["canonical_text"].str.len() >= 100]
            logging.info("    %d pre-ChatGPT docs kept", len(df))
            all_rows.append(df)

        if not all_rows:
            logging.warning("No pre-ChatGPT data found for %s; skipping", doc_type)
            continue

        corpus = pd.concat(all_rows, ignore_index=True)
        corpus["Document ID"] = corpus["Document ID"].astype(str)
        logging.info("%s: %d pre-ChatGPT documents available", doc_type, len(corpus))

        # --- dedup if requested ---
        if args.dedup:
            dedup_reps = load_dedup_representatives(base_dir, doc_type)
            if dedup_reps is not None:
                before = len(corpus)
                corpus = corpus[corpus["Document ID"].isin(dedup_reps)]
                logging.info(
                    "%s: dedup %d → %d documents", doc_type, before, len(corpus)
                )
            else:
                logging.warning(
                    "No dedup mapper files for %s; sampling without dedup", doc_type
                )

        # --- load existing results for resume ---
        existing_df = None
        done_pairs: Set[Tuple[str, str]] = set()  # (document_id, model)
        if output_path.exists() and not args.overwrite:
            try:
                existing_df = pd.read_parquet(output_path)
                for _, row in existing_df.iterrows():
                    done_pairs.add((str(row["document_id"]), str(row["model"])))
                logging.info("Resuming: %d (doc, model) pairs already done", len(done_pairs))
            except Exception:
                existing_df = None

        template = REWRITE_PROMPT_TEMPLATES[doc_type]

        # --- loop over models ---
        for model in args.models:
            backend = _detect_backend(model)
            use_offline = args.offline and backend == "vllm"
            generate_batch, _ = model_fns[model]

            # Filter to docs not yet done for this model
            already_done_ids = {doc_id for doc_id, m in done_pairs if m == model}
            available = corpus[~corpus["Document ID"].isin(already_done_ids)]
            logging.info(
                "%s / %s: %d already done, %d available for generation",
                doc_type, model, len(already_done_ids), len(available),
            )

            if args.sample_per_agency:
                # Stratified sampling: proportional to agency size with floor.
                # Each agency gets at least `floor` docs, then remaining budget
                # is allocated proportionally.  Falls back to uniform cap if
                # --sample-proportional is not set.
                cap = args.sample_per_agency
                floor = args.sample_agency_floor
                agency_groups = dict(list(available.groupby("Agency ID")))
                n_agencies = len(agency_groups)

                if args.sample_proportional and n_agencies > 0:
                    # Proportional-with-floor strategy
                    total_available = len(available)
                    total_budget = min(cap * n_agencies, total_available)
                    floor_budget = floor * n_agencies
                    remaining = max(0, total_budget - floor_budget)

                    sampled_parts = []
                    for agency_id, agency_grp in agency_groups.items():
                        # Floor allocation
                        n_floor = min(floor, len(agency_grp))
                        # Proportional allocation of remaining budget
                        prop = len(agency_grp) / total_available
                        n_prop = int(remaining * prop)
                        n_take = min(n_floor + n_prop, len(agency_grp))
                        n_take = max(n_take, 1)
                        sampled_parts.append(
                            agency_grp.sample(n=n_take, random_state=rng)
                        )
                    logging.info(
                        "%s / %s: proportional sample (floor=%d, budget=%d)",
                        doc_type, model, floor, total_budget,
                    )
                else:
                    # Uniform cap per agency (original behavior)
                    sampled_parts = []
                    for agency_id, agency_grp in agency_groups.items():
                        n_take = min(cap, len(agency_grp))
                        sampled_parts.append(
                            agency_grp.sample(n=n_take, random_state=rng)
                        )

                if not sampled_parts:
                    logging.info("%s / %s: nothing to generate (all done)", doc_type, model)
                    continue
                sampled = pd.concat(sampled_parts, ignore_index=True)
                # Shuffle so agencies are interleaved in batches
                sampled = sampled.sample(frac=1, random_state=rng).reset_index(drop=True)
                sample_n = len(sampled)
                logging.info(
                    "%s / %s: stratified sample = %d docs from %d agencies",
                    doc_type, model, sample_n, len(sampled_parts),
                )
            else:
                sample_n = min(args.sample_per_model, len(available))
                if sample_n <= 0:
                    logging.info("%s / %s: nothing to generate (all done)", doc_type, model)
                    continue
                sampled = available.sample(n=sample_n, random_state=rng).reset_index(drop=True)

            prompts = []
            for text in sampled["canonical_text"]:
                truncated = _truncate_text(text, max_input_chars)
                prompts.append(template.replace("{text}", truncated))

            ai_texts: List[str] = []
            batches = list(_batch(prompts, args.batch_size))
            for b in tqdm(batches, desc=f"{doc_type}/{model}"):
                try:
                    ai_texts.extend(generate_batch(b))
                except Exception as e:
                    logging.error("Batch failed (%d prompts): %s", len(b), e)
                    # Fill with empty strings so indices stay aligned
                    ai_texts.extend([""] * len(b))

            # Drop rows where generation failed (empty ai_text)
            ai_texts = ai_texts[: len(sampled)]  # safety trim

            new_rows = pd.DataFrame(
                {
                    "document_id": sampled["Document ID"].values,
                    "agency_id": sampled["Agency ID"].values,
                    "original_text": sampled["canonical_text"].values,
                    "ai_text": ai_texts,
                    "model": model,
                    "doc_type": doc_type,
                }
            )
            # Drop rows with empty AI text (failed generations)
            new_rows = new_rows[new_rows["ai_text"].str.len() > 0].reset_index(drop=True)
            logging.info("Generated %d / %d rewrites successfully", len(new_rows), sample_n)

            # Append to existing
            if existing_df is not None and not existing_df.empty:
                existing_df = pd.concat([existing_df, new_rows], ignore_index=True)
            else:
                existing_df = new_rows

            # Update done pairs
            for doc_id in sampled["Document ID"]:
                done_pairs.add((str(doc_id), model))

            # Write after each model (crash-safe)
            existing_df.to_parquet(output_path, index=False)
            logging.info(
                "Wrote %s (%d total rows after %s)", output_path, len(existing_df), model
            )


# ===================================================================
# Subcommand 2: estimate
# ===================================================================


def _count_binary_word_occurrences(
    sentences: List[List[str]],
) -> Tuple[Counter, int]:
    """Count binary word occurrences (each word counted at most once per sentence)."""
    counts: Counter = Counter()
    for sent in sentences:
        counts.update(set(sent))
    return counts, len(sentences)


def cmd_estimate(args: argparse.Namespace) -> None:
    logging.info("=== estimate: building word distributions ===")
    _ensure_nltk_punkt()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ai_corpus_dir = Path(args.ai_corpus_dir)

    metadata = {
        "human_cutoff": args.human_cutoff,
        "min_human_count": args.min_human_count,
        "min_ai_count": args.min_ai_count,
        "doc_types": args.doc_types,
    }

    for doc_type in args.doc_types:
        logging.info("--- estimating distribution for %s ---", doc_type)

        # --- load dedup representatives if requested ---
        dedup_reps: Optional[Set[str]] = None
        if args.dedup:
            dedup_reps = load_dedup_representatives(base_dir, doc_type)
            if dedup_reps is None:
                logging.warning(
                    "No dedup mapper files found for %s; proceeding without dedup", doc_type
                )

        # --- human sentences ---
        human_sentences: List[List[str]] = []
        n_human_docs = 0
        n_dedup_skipped = 0
        for csv_path in tqdm(
            list(iter_input_files(base_dir, doc_type)),
            desc=f"human {doc_type}",
        ):
            try:
                cols = ["Posted Date", "canonical_text"]
                if dedup_reps is not None:
                    cols.append("Document ID")
                df = pd.read_csv(
                    csv_path,
                    usecols=cols,
                    low_memory=False,
                )
            except (ValueError, KeyError):
                continue
            df = df.dropna(subset=["canonical_text", "Posted Date"])
            df = df[df["Posted Date"].apply(lambda d: is_pre_chatgpt(d, args.human_cutoff))]
            df = df[df["canonical_text"].str.len() >= 100]

            # Filter to cluster representatives only
            if dedup_reps is not None and "Document ID" in df.columns:
                before_len = len(df)
                df = df[df["Document ID"].astype(str).isin(dedup_reps)]
                n_dedup_skipped += before_len - len(df)

            for text in df["canonical_text"]:
                human_sentences.extend(tokenize_text(text))
            n_human_docs += len(df)

        if dedup_reps is not None:
            logging.info(
                "%s: dedup removed %d duplicate docs, kept %d representatives",
                doc_type, n_dedup_skipped, n_human_docs,
            )

        logging.info(
            "%s: %d human docs → %d sentences",
            doc_type, n_human_docs, len(human_sentences),
        )

        # --- AI sentences ---
        ai_parquet = ai_corpus_dir / f"ai_corpus_{doc_type}.parquet"
        if not ai_parquet.exists():
            logging.warning("Missing %s; skipping %s", ai_parquet, doc_type)
            continue
        ai_df = pd.read_parquet(ai_parquet)
        ai_sentences: List[List[str]] = []
        for text in tqdm(ai_df["ai_text"].dropna(), desc=f"ai {doc_type}"):
            ai_sentences.extend(tokenize_text(text))
        logging.info(
            "%s: %d AI docs → %d sentences",
            doc_type, len(ai_df), len(ai_sentences),
        )

        if not human_sentences or not ai_sentences:
            logging.warning("Insufficient data for %s; skipping", doc_type)
            continue

        # --- compute distributions ---
        human_counts, n_human = _count_binary_word_occurrences(human_sentences)
        ai_counts, n_ai = _count_binary_word_occurrences(ai_sentences)

        human_log_probs = {
            w: np.log(c / n_human) for w, c in human_counts.items()
        }
        ai_log_probs = {
            w: np.log(c / n_ai) for w, c in ai_counts.items()
        }

        common_vocab = set(human_counts.keys()) & set(ai_counts.keys())
        common_vocab = {
            w
            for w in common_vocab
            if human_counts[w] >= args.min_human_count
            and ai_counts[w] >= args.min_ai_count
        }
        logging.info("%s: common vocabulary size = %d", doc_type, len(common_vocab))

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
            rows.append(
                {
                    "word": word,
                    "logP": logP,
                    "logQ": logQ,
                    "log1mP": log1mP,
                    "log1mQ": log1mQ,
                    "human_count": human_counts[word],
                    "ai_count": ai_counts[word],
                    "log_odds_ratio": lor,
                }
            )

        dist_df = pd.DataFrame(rows).sort_values("log_odds_ratio").reset_index(drop=True)
        out_path = output_dir / f"distribution_{doc_type}.parquet"
        dist_df.to_parquet(out_path, index=False)
        logging.info(
            "Wrote %s (%d words, human=%d sents, ai=%d sents)",
            out_path, len(dist_df), n_human, n_ai,
        )
        metadata[f"{doc_type}_vocab_size"] = len(dist_df)
        metadata[f"{doc_type}_human_sentences"] = n_human
        metadata[f"{doc_type}_ai_sentences"] = n_ai

    meta_path = output_dir / "distribution_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info("Wrote %s", meta_path)


# ===================================================================
# Subcommand 3: infer
# ===================================================================


class MLEEstimator:
    """Maximum-likelihood estimator for the AI-written fraction alpha."""

    def __init__(self, distribution_path: Path, oov_log_prob: float = DEFAULT_OOV_LOG_PROB):
        dist_df = pd.read_parquet(distribution_path)
        self.vocab = set(dist_df["word"].values)
        self.logP = dict(zip(dist_df["word"], dist_df["logP"]))
        self.logQ = dict(zip(dist_df["word"], dist_df["logQ"]))
        self.log1mP = dict(zip(dist_df["word"], dist_df["log1mP"]))
        self.log1mQ = dict(zip(dist_df["word"], dist_df["log1mQ"]))
        self.oov_log_prob = oov_log_prob

        # Pre-compute vectors for fast scoring (present-only model, matching
        # Liang et al. 2025: sum log(p_w) only for words present in sentence)
        vocab_list = list(self.vocab)
        self._word_to_idx = {w: i for i, w in enumerate(vocab_list)}
        self._logP_arr = np.array([self.logP[w] for w in vocab_list])
        self._logQ_arr = np.array([self.logQ[w] for w in vocab_list])

    def sentence_log_probs(
        self, sentences: List[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sum log-probabilities over present words only (Liang et al. 2025)."""
        n = len(sentences)
        log_p = np.zeros(n)
        log_q = np.zeros(n)
        w2i = self._word_to_idx
        logP = self._logP_arr
        logQ = self._logQ_arr
        oov = self.oov_log_prob
        for i, sent in enumerate(sentences):
            for word in set(sent):
                idx = w2i.get(word)
                if idx is not None:
                    log_p[i] += logP[idx]
                    log_q[i] += logQ[idx]
                else:
                    log_p[i] += oov
                    log_q[i] += oov
        return log_p, log_q

    @staticmethod
    def _neg_log_likelihood(
        alpha: float, log_p: np.ndarray, log_q: np.ndarray
    ) -> float:
        log_mix = np.logaddexp(
            np.log(max(1 - alpha, 1e-15)) + log_p,
            np.log(max(alpha, 1e-15)) + log_q,
        )
        return -np.sum(log_mix)

    def estimate_alpha(self, log_p: np.ndarray, log_q: np.ndarray) -> float:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            self._neg_log_likelihood,
            bounds=(1e-6, 1 - 1e-6),
            args=(log_p, log_q),
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
        rng = np.random.RandomState(seed)
        n = len(log_p)
        alphas = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            alphas.append(self.estimate_alpha(log_p[idx], log_q[idx]))
        alpha_point = self.estimate_alpha(log_p, log_q)
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


def cmd_infer(args: argparse.Namespace) -> None:
    logging.info("=== infer: estimating AI fraction per stratum ===")
    _ensure_nltk_punkt()

    base_dir = Path(args.base_dir)
    dist_dir = Path(args.distribution_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agencies_filter = set(args.agencies) if args.agencies else None

    all_results = []

    for doc_type in args.doc_types:
        dist_path = dist_dir / f"distribution_{doc_type}.parquet"
        if not dist_path.exists():
            logging.warning("Missing distribution %s; skipping %s", dist_path, doc_type)
            continue

        estimator = MLEEstimator(dist_path, oov_log_prob=args.oov_log_prob)
        logging.info(
            "%s: loaded distribution with %d vocab words", doc_type, len(estimator.vocab)
        )

        # --- optionally load dedup representatives ---
        dedup_reps: Optional[Set[str]] = None
        if args.dedup:
            dedup_reps = load_dedup_representatives(base_dir, doc_type)
            if dedup_reps is not None:
                logging.info(
                    "%s: dedup enabled for inference — %d representative docs",
                    doc_type, len(dedup_reps),
                )
            else:
                logging.warning(
                    "%s: --dedup requested but no mapper files found; "
                    "proceeding without dedup", doc_type,
                )

        # --- load and tokenize all documents ---
        records: List[Dict] = []  # each record = {agency_id, quarter, year, sentences}
        input_files = list(iter_input_files(base_dir, doc_type))
        for csv_path in tqdm(input_files, desc=f"load {doc_type}"):
            try:
                cols = ["Document ID", "Agency ID", "Posted Date", "canonical_text"]
                df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
            except (ValueError, KeyError):
                continue
            df = df.dropna(subset=["canonical_text", "Posted Date"])
            df = df[df["canonical_text"].str.len() >= 50]
            if agencies_filter:
                df = df[df["Agency ID"].isin(agencies_filter)]
            if dedup_reps is not None:
                df = df[df["Document ID"].astype(str).isin(dedup_reps)]
            if df.empty:
                continue
            df["quarter"] = df["Posted Date"].apply(assign_quarter)
            df["year"] = df["Posted Date"].apply(assign_year)
            df = df.dropna(subset=["quarter"])

            for _, row in df.iterrows():
                sents = tokenize_text(row["canonical_text"])
                if sents:
                    records.append(
                        {
                            "agency_id": row["Agency ID"],
                            "quarter": row["quarter"],
                            "year": int(row["year"]),
                            "sentences": sents,
                            "n_sentences": len(sents),
                        }
                    )

        if not records:
            logging.warning("No records for %s", doc_type)
            continue

        rec_df = pd.DataFrame(records)
        logging.info("%s: %d documents loaded", doc_type, len(rec_df))

        # --- build strata ---
        group_cols = []
        if "agency" in args.stratify_by:
            group_cols.append("agency_id")
        if "quarter" in args.stratify_by:
            group_cols.append("quarter")
        if "year" in args.stratify_by and "quarter" not in args.stratify_by:
            group_cols.append("year")

        if not group_cols:
            groups = [("all", rec_df)]
        else:
            groups = list(rec_df.groupby(group_cols))

        for key, grp in tqdm(groups, desc=f"infer {doc_type}"):
            all_sents = []
            for sent_list in grp["sentences"]:
                all_sents.extend(sent_list)

            if len(all_sents) < args.min_sentences:
                continue

            alpha, ci_lo, ci_hi, n_used = estimator.inference(
                all_sents, n_bootstrap=args.bootstrap_n
            )

            if not isinstance(key, tuple):
                key = (key,)

            result = {
                "doc_type": doc_type,
                "alpha_estimate": round(alpha, 6) if not np.isnan(alpha) else None,
                "ci_lower": round(ci_lo, 6) if not np.isnan(ci_lo) else None,
                "ci_upper": round(ci_hi, 6) if not np.isnan(ci_hi) else None,
                "n_documents": len(grp),
                "n_sentences": n_used,
                "distribution_file": dist_path.name,
            }
            for col, val in zip(group_cols, key):
                result[col] = val

            # If quarter is a group col, extract year from it
            if "quarter" in args.stratify_by and "year" not in result:
                q_val = result.get("quarter", "")
                if isinstance(q_val, str) and len(q_val) >= 4:
                    result["year"] = int(q_val[:4])

            all_results.append(result)

        # Save after each doc_type completes (crash-safe)
        if not all_results:
            continue

        results_df = pd.DataFrame(all_results)

        # Append to existing results, replacing only doc_types we just processed
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            new_doc_types = set(results_df["doc_type"].unique())
            existing_df = existing_df[~existing_df["doc_type"].isin(new_doc_types)]
            results_df = pd.concat([existing_df, results_df], ignore_index=True)
            logging.info("Appended to existing results (kept %d old rows, added %d new rows)",
                          len(existing_df), len(results_df) - len(existing_df))

        # Reorder columns
        first_cols = ["doc_type"]
        if "agency_id" in results_df.columns:
            first_cols.append("agency_id")
        if "quarter" in results_df.columns:
            first_cols.append("quarter")
        if "year" in results_df.columns:
            first_cols.append("year")
        first_cols.extend(
            ["alpha_estimate", "ci_lower", "ci_upper", "n_documents", "n_sentences", "distribution_file"]
        )
        col_order = [c for c in first_cols if c in results_df.columns]
        col_order += [c for c in results_df.columns if c not in col_order]
        results_df = results_df[col_order].sort_values(
            [c for c in ["doc_type", "agency_id", "quarter"] if c in results_df.columns]
        )
        results_df.to_csv(output_path, index=False)
        logging.info("Wrote %s (%d rows)", output_path, len(results_df))

    if not all_results:
        logging.warning("No results produced")


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate LLM-assisted writing prevalence (Liang et al. 2025)"
    )
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Root of bulk_downloads data (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Generate AI reference corpus via LLM rewriting")
    p_gen.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Model names to generate rewrites with (default: %(default)s)",
    )
    p_gen.add_argument(
        "--sample-per-model", type=int, default=DEFAULT_SAMPLE_PER_MODEL,
        help="Number of documents to rewrite per model (default: %(default)s)",
    )
    p_gen.add_argument(
        "--offline", action="store_true",
        help="Use vLLM batch inference for open models (more efficient on GPU). "
             "Only applies to models detected as vllm backend.",
    )
    p_gen.add_argument(
        "--doc-types", nargs="+", default=DOC_TYPES, choices=DOC_TYPES,
    )
    p_gen.add_argument("--batch-size", type=int, default=32)
    p_gen.add_argument("--max-tokens", type=int, default=2048)
    p_gen.add_argument("--max-model-len", type=int, default=8192)
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument(
        "--vllm-base-url", default=DEFAULT_VLLM_BASE_URL,
        help="Base URL for vLLM server (online mode only, default: %(default)s)",
    )
    p_gen.add_argument(
        "--sample-per-agency", type=int, default=None,
        help="If set, sample up to this many docs per agency (stratified sampling). "
             "Overrides --sample-per-model. E.g. --sample-per-agency 500",
    )
    p_gen.add_argument(
        "--sample-proportional", action="store_true",
        help="With --sample-per-agency, allocate samples proportionally to agency "
             "size (with a floor per agency) instead of a uniform cap. "
             "Keeps Q corpus large while ensuring small agencies are represented.",
    )
    p_gen.add_argument(
        "--sample-agency-floor", type=int, default=20,
        help="Min docs per agency when using --sample-proportional (default: 20)",
    )
    p_gen.add_argument(
        "--dedup", action="store_true",
        help="Use minhash dedup mapper files to sample only from cluster "
             "representatives (one doc per near-duplicate cluster).",
    )
    p_gen.add_argument("--overwrite", action="store_true")
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument(
        "--human-cutoff", default=DEFAULT_HUMAN_CUTOFF,
        help="ISO date cutoff for pre-ChatGPT documents (default: %(default)s)",
    )
    p_gen.add_argument(
        "--output-dir", default=str(DEFAULT_DATA_DIR),
        help="Directory for AI corpus output (default: %(default)s)",
    )

    # --- estimate ---
    p_est = sub.add_parser("estimate", help="Build word-level distributions")
    p_est.add_argument(
        "--doc-types", nargs="+", default=DOC_TYPES, choices=DOC_TYPES
    )
    p_est.add_argument("--human-cutoff", default=DEFAULT_HUMAN_CUTOFF)
    p_est.add_argument("--min-human-count", type=int, default=DEFAULT_MIN_HUMAN_COUNT)
    p_est.add_argument("--min-ai-count", type=int, default=DEFAULT_MIN_AI_COUNT)
    p_est.add_argument(
        "--dedup", action="store_true",
        help="Use minhash dedup mapper files to keep only one representative per "
             "cluster when building the human (P) distribution. Looks for "
             "<doc_type>_all_text__dedup_mapper.csv files alongside the input CSVs.",
    )
    p_est.add_argument(
        "--ai-corpus-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing ai_corpus_*.parquet files (default: %(default)s)",
    )
    p_est.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATA_DIR / "ai_usage_distributions"),
        help="Directory for distribution output (default: %(default)s)",
    )

    # --- infer ---
    p_inf = sub.add_parser("infer", help="Estimate AI fraction per stratum via MLE")
    p_inf.add_argument(
        "--doc-types", nargs="+", default=DOC_TYPES, choices=DOC_TYPES
    )
    p_inf.add_argument(
        "--distribution-dir",
        default=str(DEFAULT_DATA_DIR / "ai_usage_distributions"),
    )
    p_inf.add_argument(
        "--stratify-by",
        nargs="+",
        default=["agency", "quarter"],
        choices=["agency", "quarter", "year", "doc_type"],
    )
    p_inf.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N)
    p_inf.add_argument("--oov-log-prob", type=float, default=DEFAULT_OOV_LOG_PROB)
    p_inf.add_argument("--min-sentences", type=int, default=DEFAULT_MIN_SENTENCES)
    p_inf.add_argument("--agencies", nargs="*", default=None)
    p_inf.add_argument(
        "--dedup",
        action="store_true",
        help="Filter to dedup cluster representatives during inference",
    )
    p_inf.add_argument(
        "--output",
        default=str(DEFAULT_DATA_DIR / "ai_usage_results.csv"),
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "estimate":
        cmd_estimate(args)
    elif args.command == "infer":
        cmd_infer(args)


if __name__ == "__main__":
    main()
