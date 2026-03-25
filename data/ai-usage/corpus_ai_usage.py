"""
Estimate LLM-assisted writing prevalence in regulatory documents using
Distributional GPT Detection (Liang et al. 2025).

Subcommands:
    generate  — Build AI reference corpus by LLM-rewriting pre-ChatGPT docs
    estimate  — Compute word-level P/Q distributions from reference corpora (CPU)
    infer     — MLE estimation of AI fraction per stratum with bootstrap CIs (CPU)
    evaluate  — Sweep κ values for hierarchical model; measure spurious alpha (CPU)

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

from ai_detection_utils import (
    # Constants
    DOC_TYPES,
    DEFAULT_HUMAN_CUTOFF,
    DEFAULT_OOV_LOG_PROB,
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_MIN_HUMAN_COUNT,
    DEFAULT_MIN_AI_COUNT,
    DEFAULT_MIN_HUMAN_FRAC,
    DEFAULT_MIN_AI_FRAC,
    DEFAULT_MIN_SENTENCES,
    _CHARS_PER_TOKEN,
    # Tokenization
    ensure_nltk_punkt,
    tokenize_text,
    # Date/time
    parse_posted_date,
    assign_quarter,
    assign_year,
    is_pre_chatgpt,
    # File I/O
    iter_input_files,
    load_dedup_representatives,
    # Text processing
    truncate_text,
    batch_iter,
    # Core computation
    sentence_log_probs_raw,
    MLEEstimator,
    build_distribution,
    # Hierarchical Bayes
    optimize_kappa,
    shrink_p,
    build_agency_est_data,
    load_agency_word_counts,
    # Data loading
    process_human_csv,
    process_ai_texts,
    load_and_tokenize_file,
    # Parallel inference
    infer_stratum,
    sweep_infer_stratum,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR.parent / "bulk_downloads"  # data/bulk_downloads/
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "ai-usage-generations"  # data/ai-usage-generations/

# ---------------------------------------------------------------------------
# Generation-specific constants
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "gpt-4",
    "gpt-4o",
    "gpt-5",
    "claude-sonnet-4-6",
    "meta-llama/Llama-3.3-70B-Instruct",
]
DEFAULT_SAMPLE_PER_MODEL = 100
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8002/v1"

REWRITE_PROMPT_TEMPLATES = {
    "public_submission": (
        "Below is a public comment submitted to a government regulatory agency. "
        "Write a new version of this comment that takes the same position and makes "
        "the same arguments on the same topic, but uses entirely different wording. "
        "Keep a similar length. Output ONLY the rewritten comment text — no XML tags, "
        "no preamble like 'Here is the rewritten comment', no meta-commentary, no "
        "notes about what you changed, no sign-offs beyond what the original has. "
        "Do NOT write multiple comments or continue with another comment. "
        "Just the single rewritten comment, nothing else.\n\n"
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
# Backend detection
# ---------------------------------------------------------------------------


MODEL_ALIASES = {
    "gpt-5-mini": "gpt-5.4-mini",
}


def _resolve_model(model_name: str) -> str:
    """Resolve model aliases to canonical names."""
    return MODEL_ALIASES.get(model_name, model_name)


def _detect_backend(model_name: str) -> str:
    """Infer API backend from model name: 'openai', 'anthropic', or 'vllm'."""
    lower = _resolve_model(model_name).lower()
    if lower.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if lower.startswith("claude"):
        return "anthropic"
    return "vllm"


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------


def _make_generate_fn_vllm_offline(
    model: str,
    *,
    max_tokens: int,
    temperature: float,
    max_model_len: int,
    tensor_parallel_size: int = 1,
) -> Tuple[Callable[[List[str]], List[str]], Optional[Callable]]:
    """Return (generate_batch, cleanup_fn) using vLLM batch inference."""
    os.environ.setdefault("VLLM_DISABLE_PROGRESS_BAR", "1")
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected for vLLM offline mode")

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    llm_kwargs = {"max_model_len": max_model_len}
    if tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = tensor_parallel_size
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
    model: str, *, max_tokens: int, temperature: float, service_tier: Optional[str] = None,
) -> Tuple[Callable[[List[str]], List[str]], None]:
    """Return (generate_batch, None) for OpenAI Responses API using AsyncOpenAI."""
    import asyncio
    from openai import AsyncOpenAI

    client = AsyncOpenAI()  # uses OPENAI_API_KEY env var

    async def _call_one(prompt: str, semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            kwargs: Dict = dict(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                reasoning={"effort": "none"},
                service_tier=service_tier or "flex",
            )
            resp = await client.responses.create(**kwargs)
            return resp.output_text or ""

    async def _run_batch(batch: List[str]) -> List[str]:
        semaphore = asyncio.Semaphore(30)
        tasks = [_call_one(prompt, semaphore) for prompt in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def generate_batch(batch: List[str]) -> List[str]:
        # Get or create event loop for the current thread
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an event loop — run in a new loop via thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                results = pool.submit(asyncio.run, _run_batch(batch)).result()
        else:
            results = asyncio.run(_run_batch(batch))

        # Replace exceptions with empty strings, but log them
        output = []
        n_errors = 0
        for r in results:
            if isinstance(r, str):
                output.append(r)
            else:
                n_errors += 1
                if n_errors <= 3:
                    logging.error("API call failed: %s: %s", type(r).__name__, r)
                output.append("")
        if n_errors > 3:
            logging.error("... and %d more API errors in this batch", n_errors - 3)
        return output

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
    service_tier: Optional[str] = None,
    tensor_parallel_size: int = 1,
) -> Tuple[Callable[[List[str]], List[str]], Optional[Callable]]:
    """Dispatch to the right backend."""
    if backend == "openai":
        return _make_generate_fn_openai(
            model, max_tokens=max_tokens, temperature=temperature, service_tier=service_tier,
        )
    if backend == "anthropic":
        return _make_generate_fn_anthropic(model, max_tokens=max_tokens, temperature=temperature)
    # vllm
    if offline:
        return _make_generate_fn_vllm_offline(
            model, max_tokens=max_tokens, temperature=temperature,
            max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size,
        )
    return _make_generate_fn_vllm_online(
        model, max_tokens=max_tokens, temperature=temperature,
        base_url=vllm_base_url, batch_size=batch_size,
    )


# ===================================================================
# Subcommand 1: generate
# ===================================================================


def cmd_generate(args: argparse.Namespace) -> None:
    logging.info("=== generate: building AI reference corpus ===")

    # Resolve model aliases
    args.models = [_resolve_model(m) for m in args.models]

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute max_model_len from the two user-facing params
    args.max_model_len = args.max_input_tokens + args.max_tokens
    max_input_chars = int(args.max_input_tokens * _CHARS_PER_TOKEN)

    # Load tokenizer for accurate truncation (API models)
    _tokenizer = None
    _backend_first = _detect_backend(args.models[0])
    if _backend_first == "openai":
        try:
            import tiktoken
            _tokenizer = tiktoken.encoding_for_model(args.models[0])
        except Exception:
            try:
                _tokenizer = tiktoken.get_encoding("o200k_base")
            except Exception:
                pass
        if _tokenizer:
            logging.info("Using tiktoken for input truncation (max %d tokens)", args.max_input_tokens)

    def truncate_for_model(text: str) -> str:
        """Truncate text to max_input_tokens using tokenizer if available, else chars."""
        if _tokenizer is not None:
            tokens = _tokenizer.encode(text, allowed_special="all")
            if len(tokens) > args.max_input_tokens:
                return _tokenizer.decode(tokens[:args.max_input_tokens])
            return text
        return truncate_text(text, max_input_chars)

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
            service_tier=getattr(args, "service_tier", None),
            tensor_parallel_size=getattr(args, "tp", 1),
        )
        model_fns[model] = (generate_batch, cleanup_fn)
        logging.info("Model %s loaded successfully.", model)

    try:
        _generate_with_models(args, base_dir, output_dir, max_input_chars, rng, model_fns,
                              truncate_fn=truncate_for_model, tokenizer=_tokenizer,
                              backend_type=_backend_first)
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
    truncate_fn: Optional[Callable] = None,
    tokenizer=None,
    backend_type: str = "vllm",
) -> None:
    """Run generation across doc_types using pre-loaded models.

    Two-pass approach for API models:
      1. Sample + tokenize all doc_types, compute costs
      2. Show combined cost estimate, ask for confirmation
      3. Generate
    """

    if truncate_fn is None:
        truncate_fn = lambda text: truncate_text(text, max_input_chars)

    # ======================================================================
    # PASS 1: Sample documents, tokenize, compute costs for ALL doc_types
    # ======================================================================
    # Each entry: {doc_type, model, sampled_df, prompts, input_tokens, existing_df, done_pairs}
    generation_plan: List[Dict] = []

    for doc_type in args.doc_types:
        output_path = output_dir / f"ai_corpus_{doc_type}.parquet"

        # --- collect pre-ChatGPT source documents ---
        all_rows = []
        csv_files = list(iter_input_files(base_dir, doc_type))
        logging.info("%s: scanning %d CSV files...", doc_type, len(csv_files))
        for i, csv_path in enumerate(csv_files, 1):
            try:
                df = pd.read_csv(
                    csv_path,
                    usecols=["Document ID", "Agency ID", "Posted Date", "canonical_text"],
                    low_memory=False,
                )
            except (ValueError, KeyError):
                continue
            df = df.dropna(subset=["canonical_text", "Posted Date"])
            df = df[df["Posted Date"].apply(lambda d: is_pre_chatgpt(d, args.human_cutoff))]
            df = df[df["canonical_text"].str.len() >= 100]
            if len(df) > 0:
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

        # --- filter to specific agencies if requested ---
        if args.agencies:
            target_set = set(args.agencies)
            before = len(corpus)
            corpus = corpus[corpus["Agency ID"].isin(target_set)]
            logging.info(
                "%s: agency filter %s → %d of %d docs",
                doc_type, args.agencies, len(corpus), before,
            )
            if corpus.empty:
                logging.warning("%s: no docs for agencies %s; skipping", doc_type, args.agencies)
                continue

        # --- load existing results for resume ---
        existing_df = None
        done_pairs: Set[Tuple[str, str]] = set()
        if output_path.exists() and not args.overwrite:
            try:
                existing_df = pd.read_parquet(output_path)
                for _, row in existing_df.iterrows():
                    done_pairs.add((str(row["document_id"]), str(row["model"])))
                logging.info("Resuming: %d (doc, model) pairs already done", len(done_pairs))
            except Exception:
                existing_df = None

        template = REWRITE_PROMPT_TEMPLATES[doc_type]

        # --- sample + tokenize for each model ---
        for model in args.models:
            already_done_ids = {doc_id for doc_id, m in done_pairs if m == model}
            available = corpus[~corpus["Document ID"].isin(already_done_ids)]
            logging.info(
                "%s / %s: %d already done, %d available for generation",
                doc_type, model, len(already_done_ids), len(available),
            )

            if args.sample_per_agency:
                cap = args.sample_per_agency
                floor = args.sample_agency_floor
                agency_groups = dict(list(available.groupby("Agency ID")))
                n_agencies = len(agency_groups)

                if args.sample_proportional and n_agencies > 0:
                    total_available = len(available)
                    total_budget = min(cap * n_agencies, total_available)
                    floor_budget = floor * n_agencies
                    remaining = max(0, total_budget - floor_budget)

                    sampled_parts = []
                    for agency_id, agency_grp in agency_groups.items():
                        n_floor = min(floor, len(agency_grp))
                        prop = len(agency_grp) / total_available
                        n_prop = int(remaining * prop)
                        n_take = min(n_floor + n_prop, len(agency_grp))
                        n_take = max(n_take, 1)
                        sampled_parts.append(
                            agency_grp.sample(n=n_take, random_state=rng)
                        )
                else:
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
                sampled = sampled.sample(frac=1, random_state=rng).reset_index(drop=True)
                logging.info(
                    "%s / %s: stratified sample = %d docs from %d agencies",
                    doc_type, model, len(sampled), len(sampled_parts),
                )
            else:
                sample_n = min(args.sample_per_model, len(available))
                if sample_n <= 0:
                    logging.info("%s / %s: nothing to generate (all done)", doc_type, model)
                    continue
                sampled = available.sample(n=sample_n, random_state=rng).reset_index(drop=True)

            # Debug mode: limit to 10 docs
            if getattr(args, "debug", False) and len(sampled) > 10:
                sampled = sampled.head(10).reset_index(drop=True)
                logging.info("%s / %s: debug mode, limited to %d docs", doc_type, model, len(sampled))

            # Tokenize and build prompts
            prompts = []
            input_token_count = 0
            text_iter = sampled["canonical_text"]
            if tqdm is not None:
                text_iter = tqdm(
                    text_iter, desc=f"tokenizing {doc_type}/{model}",
                    total=len(sampled), leave=False,
                )
            for text in text_iter:
                truncated = truncate_fn(text)
                prompts.append(template.replace("{text}", truncated))
                if tokenizer:
                    input_token_count += len(tokenizer.encode(truncated, allowed_special="all"))

            generation_plan.append({
                "doc_type": doc_type,
                "model": model,
                "sampled": sampled,
                "prompts": prompts,
                "input_tokens": input_token_count,
                "output_path": output_path,
                "existing_df": existing_df,
                "done_pairs": done_pairs,
            })

    if not generation_plan:
        logging.info("Nothing to generate.")
        return

    # ======================================================================
    # COST CONFIRMATION (API models only)
    # ======================================================================
    if backend_type == "openai" and not getattr(args, "debug", False):
        is_flex = (getattr(args, "service_tier", None) or "flex") == "flex"
        input_price = 0.375 if is_flex else 0.75
        output_price = 2.25 if is_flex else 4.50
        tier_label = "flex" if is_flex else "standard"

        print("\n" + "=" * 70)
        print("  COST ESTIMATE (%s tier)" % tier_label)
        print("=" * 70)
        print("  %-25s %6s %10s %10s %8s" % (
            "Doc Type / Model", "Docs", "Input Tok", "Output Tok", "Cost"))
        print("-" * 70)

        total_cost = 0
        for entry in generation_plan:
            inp = entry["input_tokens"]
            out = inp  # assume output ≈ input
            cost = (inp / 1e6) * input_price + (out / 1e6) * output_price
            total_cost += cost
            label = "%s / %s" % (entry["doc_type"], entry["model"])
            print("  %-25s %6d %9.1fM %9.1fM   $%.2f" % (
                label, len(entry["prompts"]), inp / 1e6, out / 1e6, cost))

        print("-" * 70)
        print("  %-25s %6d %9.1fM %9.1fM   $%.2f" % (
            "TOTAL",
            sum(len(e["prompts"]) for e in generation_plan),
            sum(e["input_tokens"] for e in generation_plan) / 1e6,
            sum(e["input_tokens"] for e in generation_plan) / 1e6,
            total_cost,
        ))
        print("=" * 70)

        confirm = input("\nProceed? [y/N] ")
        if confirm.strip().lower() not in ("y", "yes"):
            logging.info("Aborted by user.")
            return

    # ======================================================================
    # PASS 2: Generate
    # ======================================================================
    for entry in generation_plan:
        doc_type = entry["doc_type"]
        model = entry["model"]
        sampled = entry["sampled"]
        prompts = entry["prompts"]
        output_path = entry["output_path"]
        existing_df = entry["existing_df"]
        done_pairs = entry["done_pairs"]
        generate_batch, _ = model_fns[model]

        logging.info(
            "%s / %s: generating %d rewrites...", doc_type, model, len(prompts)
        )

        ai_texts: List[str] = []
        batches = list(batch_iter(prompts, args.batch_size))
        for b in tqdm(batches, desc=f"{doc_type}/{model}"):
            try:
                ai_texts.extend(generate_batch(b))
            except Exception as e:
                logging.error("Batch failed (%d prompts): %s", len(b), e)
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
        logging.info("Generated %d / %d rewrites successfully", len(new_rows), len(sampled))

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


def cmd_estimate(args: argparse.Namespace) -> None:
    logging.info("=== estimate: building word distributions ===")
    ensure_nltk_punkt()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ai_corpus_dir = Path(args.ai_corpus_dir)
    n_workers = getattr(args, "workers", 1)

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

        # --- load AI doc IDs for matched filtering (if requested) ---
        doc_ids_filter: Optional[Set[str]] = None
        if not getattr(args, "no_matched", False):
            ai_parquet_paths_pre = []
            root_pq_pre = ai_corpus_dir / f"ai_corpus_{doc_type}.parquet"
            if root_pq_pre.exists():
                ai_parquet_paths_pre.append(root_pq_pre)
            for subdir in sorted(ai_corpus_dir.iterdir()):
                if subdir.is_dir():
                    sub_pq = subdir / f"ai_corpus_{doc_type}.parquet"
                    if sub_pq.exists():
                        ai_parquet_paths_pre.append(sub_pq)
            if ai_parquet_paths_pre:
                doc_ids_filter = set()
                for pq_path in ai_parquet_paths_pre:
                    ids = pd.read_parquet(pq_path, columns=["document_id"])["document_id"].astype(str)
                    doc_ids_filter.update(ids)
                logging.info(
                    "%s: --matched mode: filtering human corpus to %d AI document IDs",
                    doc_type, len(doc_ids_filter),
                )
            else:
                logging.warning(
                    "%s: --matched requested but no AI corpus found; using full human corpus",
                    doc_type,
                )

        # --- human sentences (streaming counts, no sentence accumulation) ---
        hierarchical = getattr(args, "hierarchical", False)
        input_files = list(iter_input_files(base_dir, doc_type))

        # Build task args for each CSV file
        agencies_filter = set(args.agencies) if getattr(args, "agencies", None) else None
        task_args = [
            (str(csv_path), args.human_cutoff, dedup_reps, hierarchical,
             agencies_filter, doc_ids_filter)
            for csv_path in input_files
        ]

        human_counts = Counter()
        n_human_sents = 0
        n_human_docs = 0
        n_dedup_skipped = 0
        n_cross_file_dupes = 0
        agency_accum: Dict[str, Dict] = {}
        seen_doc_ids: Set[str] = set()

        def _merge_result(counts, n_s, n_d, n_skip, aa, doc_ids):
            nonlocal human_counts, n_human_sents, n_human_docs
            nonlocal n_dedup_skipped, n_cross_file_dupes, seen_doc_ids
            # Check for cross-file duplicates
            dupes = doc_ids & seen_doc_ids
            if dupes:
                n_cross_file_dupes += len(dupes)
            seen_doc_ids.update(doc_ids)
            # Note: when cross-file dupes exist, their word counts are
            # double-counted. This is acceptable because: (1) dupes are
            # rare (~1%), (2) double-counting a doc's words slightly
            # inflates its contribution to P(w) but doesn't change the
            # vocab or direction of log-odds ratios.
            human_counts += counts
            n_human_sents += n_s
            n_human_docs += n_d
            n_dedup_skipped += n_skip
            if aa:
                for aid, acc in aa.items():
                    existing = agency_accum.setdefault(
                        aid, {"counts": Counter(), "n_sents": 0}
                    )
                    existing["counts"] += acc["counts"]
                    existing["n_sents"] += acc["n_sents"]

        if n_workers > 1 and len(input_files) > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            logging.info(
                "%s: processing %d files with %d workers",
                doc_type, len(input_files), n_workers,
            )
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(process_human_csv, ta): ta[0]
                    for ta in task_args
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"human {doc_type}",
                ):
                    counts, n_s, n_d, n_skip, aa, doc_ids = future.result()
                    _merge_result(counts, n_s, n_d, n_skip, aa, doc_ids)
        else:
            for ta in tqdm(task_args, desc=f"human {doc_type}"):
                counts, n_s, n_d, n_skip, aa, doc_ids = process_human_csv(ta)
                _merge_result(counts, n_s, n_d, n_skip, aa, doc_ids)

        # Correct doc count for cross-file duplicates
        n_human_docs = len(seen_doc_ids)
        if n_cross_file_dupes > 0:
            logging.warning(
                "%s: %d cross-file duplicate documents detected and excluded from doc count",
                doc_type, n_cross_file_dupes,
            )

        n_human = n_human_sents

        if dedup_reps is not None:
            logging.info(
                "%s: dedup removed %d duplicate docs, kept %d representatives",
                doc_type, n_dedup_skipped, n_human_docs,
            )

        logging.info(
            "%s: %d human docs → %d sentences",
            doc_type, n_human_docs, n_human_sents,
        )

        # --- AI sentences (streaming counts) ---
        # Search root dir and all subdirectories for AI corpus parquets
        ai_parquet_paths = []
        root_pq = ai_corpus_dir / f"ai_corpus_{doc_type}.parquet"
        if root_pq.exists():
            ai_parquet_paths.append(root_pq)
        for subdir in sorted(ai_corpus_dir.iterdir()):
            if subdir.is_dir():
                sub_pq = subdir / f"ai_corpus_{doc_type}.parquet"
                if sub_pq.exists():
                    ai_parquet_paths.append(sub_pq)
        if not ai_parquet_paths:
            logging.warning("No AI corpus files found for %s; skipping", doc_type)
            continue
        logging.info("AI corpus files for %s: %s", doc_type, [str(p) for p in ai_parquet_paths])
        ai_frames = []
        for pq_path in ai_parquet_paths:
            _df = pd.read_parquet(pq_path)
            if agencies_filter and "agency_id" in _df.columns:
                _df = _df[_df["agency_id"].isin(agencies_filter)]
            ai_frames.append(_df)
        ai_df = pd.concat(ai_frames, ignore_index=True)
        ai_texts = ai_df["ai_text"].dropna().tolist()

        # Build per-agency AI counts alongside pooled counts
        ai_counts = Counter()
        n_ai = 0
        ai_agency_accum: Dict[str, Dict] = {}
        for agency_id, text in zip(ai_df["agency_id"], ai_df["ai_text"].fillna("")):
            for sent in tokenize_text(text):
                word_set = set(sent)
                ai_counts.update(word_set)
                n_ai += 1
                acc = ai_agency_accum.setdefault(
                    agency_id, {"counts": Counter(), "n_sents": 0}
                )
                acc["counts"].update(word_set)
                acc["n_sents"] += 1

        logging.info(
            "%s: %d AI docs → %d sentences",
            doc_type, len(ai_df), n_ai,
        )

        # --- verify matched filtering ---
        if doc_ids_filter is not None:
            ai_doc_ids = set(ai_df["document_id"].astype(str))
            # AI IDs should exactly match what we used as the filter
            assert ai_doc_ids == doc_ids_filter, (
                f"{doc_type}: AI doc IDs ({len(ai_doc_ids)}) != filter IDs "
                f"({len(doc_ids_filter)}). "
                f"Missing from AI: {len(doc_ids_filter - ai_doc_ids)}, "
                f"Extra in AI: {len(ai_doc_ids - doc_ids_filter)}"
            )
            # Human docs may be slightly fewer (short text filtered out), but
            # should be close. Warn if >5% are missing, fail if >50%.
            n_expected = len(doc_ids_filter)
            n_missing = n_expected - n_human_docs
            if n_missing > 0:
                pct_missing = n_missing / n_expected * 100
                logging.warning(
                    "%s: matched mode — %d/%d (%.1f%%) AI docs not found in "
                    "human corpus (likely filtered by text length or missing fields)",
                    doc_type, n_missing, n_expected, pct_missing,
                )
                assert pct_missing < 50, (
                    f"{doc_type}: matched mode but only {n_human_docs}/{n_expected} "
                    f"AI docs found in human corpus ({pct_missing:.1f}% missing). "
                    f"Check that --base-dir and --ai-corpus-dir point to the same data."
                )
            logging.info(
                "%s: MATCHED — %d human docs / %d AI docs (%.1f%% coverage)",
                doc_type, n_human_docs, n_expected,
                n_human_docs / max(n_expected, 1) * 100,
            )

        if not n_human or not n_ai:
            logging.warning("Insufficient data for %s; skipping", doc_type)
            continue

        dist_df = build_distribution(
            human_counts, n_human, ai_counts, n_ai,
            min_human_count=args.min_human_count,
            min_ai_count=args.min_ai_count,
            min_human_frac=args.min_human_frac,
            min_ai_frac=args.min_ai_frac,
            max_vocab=getattr(args, "max_vocab", None),
        )
        if dist_df is None:
            continue
        out_path = output_dir / f"distribution_{doc_type}.parquet"
        dist_df.to_parquet(out_path, index=False)
        logging.info(
            "Wrote %s (%d words, human=%d sents, ai=%d sents)",
            out_path, len(dist_df), n_human, n_ai,
        )
        metadata[f"{doc_type}_vocab_size"] = len(dist_df)
        metadata[f"{doc_type}_human_sentences"] = n_human
        metadata[f"{doc_type}_ai_sentences"] = n_ai

        # --- per-agency: build separate distribution per agency ---
        per_agency = getattr(args, "per_agency", False)
        min_agency_docs = getattr(args, "min_agency_docs", 100)
        if per_agency and agency_accum and ai_agency_accum:
            # Find agencies present in both human and AI data
            common_agencies = set(agency_accum.keys()) & set(ai_agency_accum.keys())
            for agency_id in sorted(common_agencies):
                h_acc = agency_accum[agency_id]
                a_acc = ai_agency_accum[agency_id]
                h_counts = h_acc["counts"]
                h_n = h_acc["n_sents"]
                a_counts = a_acc["counts"]
                a_n = a_acc["n_sents"]
                if h_n < min_agency_docs or a_n < 10:
                    continue
                a_dist = build_distribution(
                    h_counts, h_n, a_counts, a_n,
                    min_human_count=args.min_human_count,
                    min_ai_count=args.min_ai_count,
                    min_human_frac=args.min_human_frac,
                    min_ai_frac=args.min_ai_frac,
                    max_vocab=getattr(args, "max_vocab", None),
                )
                if a_dist is None or len(a_dist) < 50:
                    logging.info(
                        "  %s/%s: skipped (vocab too small: %d)",
                        doc_type, agency_id, len(a_dist) if a_dist is not None else 0,
                    )
                    continue
                agency_dir = output_dir / agency_id
                agency_dir.mkdir(parents=True, exist_ok=True)
                a_out = agency_dir / f"distribution_{doc_type}.parquet"
                a_dist.to_parquet(a_out, index=False)
                logging.info(
                    "  %s/%s: %d words (human=%d sents, ai=%d sents)",
                    doc_type, agency_id, len(a_dist), h_n, a_n,
                )

        # --- hierarchical: save per-agency counts and optimize kappa ---
        if agency_accum:
            # Recover common_vocab from the pooled distribution we just wrote
            common_vocab = set(dist_df["word"].values)
            agency_word_counts = {
                aid: (acc["counts"], acc["n_sents"])
                for aid, acc in agency_accum.items()
            }
            pool_freq = {w: human_counts[w] / n_human for w in common_vocab}
            optimal_kappa = optimize_kappa(
                pool_freq, agency_word_counts, common_vocab
            )
            metadata[f"{doc_type}_optimal_kappa"] = optimal_kappa
            metadata[f"{doc_type}_n_agencies"] = len(agency_word_counts)

            # Save per-agency word counts (only for common vocab)
            agency_rows = []
            for aid, (counts, n_sents) in agency_word_counts.items():
                for w in common_vocab:
                    agency_rows.append({
                        "agency_id": aid,
                        "word": w,
                        "count": counts.get(w, 0),
                        "n_sentences": n_sents,
                    })
            agency_df = pd.DataFrame(agency_rows)
            agency_path = output_dir / f"agency_counts_{doc_type}.parquet"
            agency_df.to_parquet(agency_path, index=False)
            logging.info(
                "Wrote %s (%d agencies, %d vocab, kappa=%.1f)",
                agency_path, len(agency_word_counts), len(common_vocab),
                optimal_kappa,
            )

        # Save per-agency AI word counts
        if ai_agency_accum:
            ai_agency_word_counts = {
                aid: (acc["counts"], acc["n_sents"])
                for aid, acc in ai_agency_accum.items()
            }
            pool_ai_freq = {w: ai_counts[w] / n_ai for w in common_vocab}
            optimal_kappa_q = optimize_kappa(
                pool_ai_freq, ai_agency_word_counts, common_vocab
            )
            metadata[f"{doc_type}_optimal_kappa_q"] = optimal_kappa_q
            metadata[f"{doc_type}_n_ai_agencies"] = len(ai_agency_word_counts)

            ai_agency_rows = []
            for aid, (counts, n_sents) in ai_agency_word_counts.items():
                for w in common_vocab:
                    ai_agency_rows.append({
                        "agency_id": aid,
                        "word": w,
                        "count": counts.get(w, 0),
                        "n_sentences": n_sents,
                    })
            ai_agency_df = pd.DataFrame(ai_agency_rows)
            ai_agency_path = output_dir / f"agency_ai_counts_{doc_type}.parquet"
            ai_agency_df.to_parquet(ai_agency_path, index=False)
            logging.info(
                "Wrote %s (%d agencies, %d vocab, kappa_q=%.1f)",
                ai_agency_path, len(ai_agency_word_counts), len(common_vocab),
                optimal_kappa_q,
            )

    meta_path = output_dir / "distribution_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info("Wrote %s", meta_path)


# ===================================================================
# Subcommand 3: infer
# ===================================================================


def cmd_infer(args: argparse.Namespace) -> None:
    logging.info("=== infer: estimating AI fraction per stratum ===")
    ensure_nltk_punkt()

    base_dir = Path(args.base_dir)
    dist_dir = Path(args.distribution_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agencies_filter = set(args.agencies) if args.agencies else None

    all_results = []
    all_doc_scores = []

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
        input_files = list(iter_input_files(base_dir, doc_type))

        records: List[Dict] = []
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from functools import partial
            load_fn = partial(
                load_and_tokenize_file,
                agencies_filter=agencies_filter,
                dedup_reps=dedup_reps,
            )
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(load_fn, f): f for f in input_files}
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc=f"load {doc_type}"
                ):
                    records.extend(future.result())
        else:
            for csv_path in tqdm(input_files, desc=f"load {doc_type}"):
                records.extend(
                    load_and_tokenize_file(csv_path, agencies_filter, dedup_reps)
                )

        if not records:
            logging.warning("No records for %s", doc_type)
            continue

        rec_df = pd.DataFrame(records)
        logging.info("%s: %d documents loaded", doc_type, len(rec_df))

        # --- subsample if requested ---
        subsample_frac = getattr(args, "subsample_frac", None)
        if subsample_frac and 0 < subsample_frac < 1:
            seed = getattr(args, "subsample_seed", None)
            before = len(rec_df)
            rec_df = rec_df.sample(frac=subsample_frac, random_state=seed).reset_index(drop=True)
            logging.info(
                "%s: subsampled %.0f%% → %d of %d documents",
                doc_type, subsample_frac * 100, len(rec_df), before,
            )

        # --- build strata ---
        group_cols = []
        if "agency" in args.stratify_by:
            group_cols.append("agency_id")
        if "quarter" in args.stratify_by:
            group_cols.append("quarter")
        elif "half" in args.stratify_by:
            group_cols.append("half")
        elif "year" in args.stratify_by:
            group_cols.append("year")

        if not group_cols:
            groups = [("all", rec_df)]
        else:
            groups = list(rec_df.groupby(group_cols))

        # --- prepare stratum tasks ---
        stratum_tasks = []
        # For per-document scoring: map stratum key -> list of (doc_id, sentences)
        stratum_docs = {}
        for key, grp in groups:
            all_sents = []
            for sent_list in grp["sentences"]:
                all_sents.extend(sent_list)
            if len(all_sents) < args.min_sentences:
                continue
            stratum_tasks.append((key, len(grp), all_sents))
            if args.document_scores:
                stratum_docs[key] = list(zip(grp["document_id"], grp["sentences"]))

        logging.info(
            "%s: %d strata to infer (workers=%d)",
            doc_type, len(stratum_tasks), args.workers,
        )

        # --- build inference tasks (sentence_log_probs + bootstrap run in workers) ---
        per_agency = getattr(args, "per_agency", False)
        hierarchical = getattr(args, "hierarchical", False)

        if per_agency and "agency_id" in group_cols:
            # Per-agency mode: load separate distribution per agency
            agency_est_cache: Dict[str, Tuple] = {}
            inference_tasks = []
            n_agency_found = 0
            n_agency_fallback = 0
            for key, n_docs, sents in stratum_tasks:
                idx = group_cols.index("agency_id")
                agency_id = key[idx] if isinstance(key, tuple) else key
                if agency_id not in agency_est_cache:
                    agency_dist_path = dist_dir / agency_id / f"distribution_{doc_type}.parquet"
                    if agency_dist_path.exists():
                        ae = MLEEstimator(agency_dist_path, oov_log_prob=args.oov_log_prob)
                        agency_est_cache[agency_id] = (
                            ae._word_to_idx, ae._delta_p, ae._delta_q,
                            ae._baseline_p, ae._baseline_q, ae.vocab,
                        )
                        n_agency_found += 1
                    else:
                        # Fall back to pooled distribution
                        agency_est_cache[agency_id] = (
                            estimator._word_to_idx, estimator._delta_p,
                            estimator._delta_q, estimator._baseline_p,
                            estimator._baseline_q, estimator.vocab,
                        )
                        n_agency_fallback += 1
                inference_tasks.append(
                    (key, n_docs, sents, agency_est_cache[agency_id], args.bootstrap_n)
                )
            logging.info(
                "%s: per-agency mode — %d agency distributions loaded, %d fallback to pooled",
                doc_type, n_agency_found, n_agency_fallback,
            )

        elif hierarchical:
            # Load per-agency word counts and build agency-specific est_data
            agency_counts_path = dist_dir / f"agency_counts_{doc_type}.parquet"
            if not agency_counts_path.exists():
                logging.warning(
                    "Missing %s; falling back to pooled P. "
                    "Run `estimate --hierarchical` first.",
                    agency_counts_path,
                )
                hierarchical = False

        if hierarchical and not per_agency:
            agency_word_counts_data = load_agency_word_counts(agency_counts_path)
            # kappa_p: use explicit arg, or fall back to metadata
            kappa = getattr(args, "kappa", None)
            meta_path = dist_dir / "distribution_metadata.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            if kappa is None:
                kappa = meta.get(f"{doc_type}_optimal_kappa")
                if kappa is None:
                    logging.warning("No kappa found; using default 1000")
                    kappa = 1000.0

            # Agency-specific Q: load AI agency counts and kappa_q
            agency_ai_word_counts = None
            kappa_q = None
            ai_agency_counts_path = dist_dir / f"agency_ai_counts_{doc_type}.parquet"
            if ai_agency_counts_path.exists():
                agency_ai_word_counts = load_agency_word_counts(ai_agency_counts_path)
                kappa_q = getattr(args, "kappa_q", None)
                if kappa_q is None:
                    kappa_q = meta.get(f"{doc_type}_optimal_kappa_q")
                if kappa_q is not None:
                    logging.info(
                        "%s: agency-specific Q — kappa_q=%.1f, %d AI agencies",
                        doc_type, kappa_q, len(agency_ai_word_counts),
                    )
                else:
                    logging.info("%s: no kappa_q found; using pooled Q", doc_type)
                    agency_ai_word_counts = None
            else:
                logging.info(
                    "%s: no agency_ai_counts file; using pooled Q", doc_type
                )

            logging.info(
                "%s: hierarchical mode — kappa=%.1f, %d agencies",
                doc_type, kappa, len(agency_word_counts_data),
            )

            dist_df = pd.read_parquet(dist_path)
            agency_est_cache = {}
            inference_tasks = []
            for key, n_docs, sents in stratum_tasks:
                # Extract agency_id from stratum key
                if "agency_id" in group_cols:
                    idx = group_cols.index("agency_id")
                    agency_id = key[idx] if isinstance(key, tuple) else key
                else:
                    agency_id = "__pooled__"
                if agency_id not in agency_est_cache:
                    agency_est_cache[agency_id] = build_agency_est_data(
                        dist_df, agency_word_counts_data, kappa, agency_id,
                        agency_ai_word_counts, kappa_q,
                    )
                inference_tasks.append(
                    (key, n_docs, sents, agency_est_cache[agency_id], args.bootstrap_n)
                )
        elif not per_agency:
            est_data = (
                estimator._word_to_idx,
                estimator._delta_p,
                estimator._delta_q,
                estimator._baseline_p,
                estimator._baseline_q,
                estimator.vocab,
            )
            inference_tasks = [
                (key, n_docs, sents, est_data, args.bootstrap_n)
                for key, n_docs, sents in stratum_tasks
            ]
        logging.info("%s: starting inference (%d strata)...", doc_type, len(inference_tasks))

        def _build_result(key, n_docs, alpha, ci_lo, ci_hi, n_used):
            if not isinstance(key, tuple):
                key = (key,)
            result = {
                "doc_type": doc_type,
                "alpha_estimate": round(alpha, 6) if not np.isnan(alpha) else None,
                "ci_lower": round(ci_lo, 6) if not np.isnan(ci_lo) else None,
                "ci_upper": round(ci_hi, 6) if not np.isnan(ci_hi) else None,
                "n_documents": n_docs,
                "n_sentences": n_used,
                "distribution_file": dist_path.name,
            }
            for col, val in zip(group_cols, key):
                result[col] = val
            # Extract year from quarter or half if not already present
            if "year" not in result:
                for time_col in ["quarter", "half"]:
                    t_val = result.get(time_col, "")
                    if isinstance(t_val, str) and len(t_val) >= 4:
                        result["year"] = int(t_val[:4])
                        break
            return result

        # Map stratum key -> fitted alpha (for per-document scoring)
        stratum_alphas = {}

        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            futures = {}
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for task in inference_tasks:
                    futures[pool.submit(infer_stratum, task)] = task[0]
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc=f"infer {doc_type}"
                ):
                    key, n_docs, alpha, ci_lo, ci_hi, n_used = future.result()
                    stratum_alphas[key] = alpha
                    all_results.append(_build_result(key, n_docs, alpha, ci_lo, ci_hi, n_used))
        else:
            for task in tqdm(inference_tasks, desc=f"infer {doc_type}"):
                key, n_docs, alpha, ci_lo, ci_hi, n_used = infer_stratum(task)
                stratum_alphas[key] = alpha
                all_results.append(_build_result(key, n_docs, alpha, ci_lo, ci_hi, n_used))

        # --- per-document AI scores ---
        if args.document_scores and stratum_docs:
            logging.info("%s: computing per-document AI scores...", doc_type)
            for stratum_key, doc_list in stratum_docs.items():
                alpha = stratum_alphas.get(stratum_key, float("nan"))
                if np.isnan(alpha):
                    continue
                log_alpha = np.log(max(alpha, 1e-15))
                log_1malpha = np.log(max(1 - alpha, 1e-15))

                # Use agency-specific est_data in hierarchical or per-agency mode
                if hierarchical or per_agency:
                    if "agency_id" in group_cols:
                        idx = group_cols.index("agency_id")
                        agency_id = stratum_key[idx] if isinstance(stratum_key, tuple) else stratum_key
                    else:
                        agency_id = "__pooled__"
                    doc_est_data = agency_est_cache.get(agency_id)
                    if doc_est_data is None:
                        logging.warning(
                            "No est_data for agency %s in cache; skipping doc scoring",
                            agency_id,
                        )
                        continue
                    doc_w2i, doc_dp, doc_dq, doc_bp, doc_bq, doc_vs = doc_est_data
                else:
                    doc_w2i = estimator._word_to_idx
                    doc_dp = estimator._delta_p
                    doc_dq = estimator._delta_q
                    doc_bp = estimator._baseline_p
                    doc_bq = estimator._baseline_q
                    doc_vs = estimator.vocab

                for doc_id, doc_sents in doc_list:
                    filtered = [s for s in doc_sents if set(s) & doc_vs]
                    if not filtered:
                        continue
                    log_p, log_q = sentence_log_probs_raw(
                        filtered, doc_w2i, doc_dp, doc_dq, doc_bp, doc_bq,
                    )
                    # Posterior: r_i = alpha*Q(s_i) / ((1-alpha)*P(s_i) + alpha*Q(s_i))
                    log_num = log_alpha + log_q
                    log_den = np.logaddexp(log_1malpha + log_p, log_num)
                    r = np.exp(log_num - log_den)
                    ai_score = float(np.mean(r))
                    doc_score_row = {
                        "document_id": doc_id,
                        "doc_type": doc_type,
                        "ai_score": round(ai_score, 6),
                        "n_sentences": len(filtered),
                        "stratum_alpha": round(alpha, 6),
                    }
                    key_tuple = stratum_key if isinstance(stratum_key, tuple) else (stratum_key,)
                    for col, val in zip(group_cols, key_tuple):
                        doc_score_row[col] = val
                    all_doc_scores.append(doc_score_row)
            logging.info(
                "%s: scored %d documents",
                doc_type, sum(1 for d in all_doc_scores if d["doc_type"] == doc_type),
            )

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

    # --- write per-document scores ---
    if args.document_scores and all_doc_scores:
        doc_scores_df = pd.DataFrame(all_doc_scores)
        doc_scores_path = Path(args.document_scores)
        doc_scores_path.parent.mkdir(parents=True, exist_ok=True)
        doc_scores_df.to_csv(doc_scores_path, index=False)
        logging.info(
            "Wrote per-document scores: %s (%d documents)",
            doc_scores_path, len(doc_scores_df),
        )


# ===================================================================
# Subcommand 4: evaluate (hierarchical kappa sweep)
# ===================================================================


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Sweep kappa values and measure spurious alpha on pre-ChatGPT data per agency.

    This lets you find the kappa that minimises false-positive AI detection
    before committing to a value for full inference.
    """
    from scipy.optimize import minimize_scalar

    logging.info("=== evaluate: sweeping kappa for hierarchical model ===")
    ensure_nltk_punkt()

    base_dir = Path(args.base_dir)
    dist_dir = Path(args.distribution_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kappa_values = list(np.logspace(
        np.log10(args.kappa_min), np.log10(args.kappa_max), args.kappa_steps
    ))
    logging.info("Sweeping kappa: %s", [f"{k:.0f}" for k in kappa_values])

    agencies_filter = set(args.agencies) if args.agencies else None
    all_results = []

    for doc_type in args.doc_types:
        dist_path = dist_dir / f"distribution_{doc_type}.parquet"
        agency_counts_path = dist_dir / f"agency_counts_{doc_type}.parquet"
        if not dist_path.exists():
            logging.warning("Missing %s; skipping", dist_path)
            continue
        if not agency_counts_path.exists():
            logging.warning(
                "Missing %s; run `estimate --hierarchical` first", agency_counts_path
            )
            continue

        dist_df = pd.read_parquet(dist_path)
        agency_word_counts_data = load_agency_word_counts(agency_counts_path)
        vocab_list = list(dist_df["word"].values)
        w2i = {w: i for i, w in enumerate(vocab_list)}
        vocab_set = set(vocab_list)

        # Pooled Q arrays (fallback)
        logQ_arr = dist_df["logQ"].values.astype(float)
        log1mQ_arr = dist_df["log1mQ"].values.astype(float)
        pooled_baseline_q = float(log1mQ_arr.sum())
        pooled_delta_q = logQ_arr - log1mQ_arr
        mu_q = np.exp(logQ_arr)  # pooled Q frequencies as prior for shrinkage

        # Agency-specific Q: load AI agency counts and kappa_q
        agency_ai_word_counts = None
        kappa_q = None
        meta_path = dist_dir / "distribution_metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        ai_agency_counts_path = dist_dir / f"agency_ai_counts_{doc_type}.parquet"
        if ai_agency_counts_path.exists():
            agency_ai_word_counts = load_agency_word_counts(ai_agency_counts_path)
            kappa_q = meta.get(f"{doc_type}_optimal_kappa_q")
            if kappa_q is not None:
                logging.info(
                    "%s: agency-specific Q — kappa_q=%.1f, %d AI agencies",
                    doc_type, kappa_q, len(agency_ai_word_counts),
                )
            else:
                agency_ai_word_counts = None

        pooled_logP_arr = dist_df["logP"].values.astype(float)
        pooled_log1mP_arr = dist_df["log1mP"].values.astype(float)
        pooled_baseline_p = float(pooled_log1mP_arr.sum())
        pooled_delta_p = pooled_logP_arr - pooled_log1mP_arr
        mu = np.exp(pooled_logP_arr)  # pool frequencies as prior

        # --- load dedup reps if requested ---
        dedup_reps = None
        if args.dedup:
            dedup_reps = load_dedup_representatives(base_dir, doc_type)

        # --- load and tokenize pre-ChatGPT documents per agency ---
        max_docs = getattr(args, "max_docs_per_agency", None)
        logging.info(
            "%s: loading pre-ChatGPT documents%s...",
            doc_type,
            f" (max {max_docs} docs/agency)" if max_docs else "",
        )
        # First pass: collect doc records per agency
        agency_records: Dict[str, List] = {}
        for csv_path in tqdm(
            list(iter_input_files(base_dir, doc_type)), desc=f"load {doc_type}"
        ):
            records = load_and_tokenize_file(csv_path, agencies_filter, dedup_reps)
            for rec in records:
                if rec["year"] > 2022:
                    continue
                agency_id = rec["agency_id"]
                agency_records.setdefault(agency_id, []).append(rec)

        # Downsample if requested
        if max_docs:
            rng = np.random.RandomState(42)
            for agency_id in agency_records:
                recs = agency_records[agency_id]
                if len(recs) > max_docs:
                    agency_records[agency_id] = [
                        recs[i] for i in rng.choice(len(recs), max_docs, replace=False)
                    ]

        # Flatten to sentences
        agency_data: Dict[str, List] = {}
        for agency_id, recs in agency_records.items():
            sents = []
            for rec in recs:
                sents.extend(rec["sentences"])
            agency_data[agency_id] = sents
        del agency_records

        if not agency_data:
            logging.warning("No pre-ChatGPT data for %s", doc_type)
            continue

        # Pre-compute per-agency: filter sentences, index words, compute log Q
        agency_prepared: Dict[str, Tuple] = {}
        for agency_id, sents in agency_data.items():
            filtered = [s for s in sents if set(s) & vocab_set]
            if len(filtered) < args.min_sentences:
                continue
            # Pre-compute word indices per sentence (reused across kappa values)
            sent_indices = []
            for s in filtered:
                idxs = np.array(
                    [w2i[w] for w in set(s) if w in w2i], dtype=np.intp
                )
                sent_indices.append(idxs)
            # Compute log Q — agency-specific if available, else pooled
            if agency_ai_word_counts is not None and kappa_q is not None and agency_id in agency_ai_word_counts:
                ai_counts, n_ai_a = agency_ai_word_counts[agency_id]
                n_ai_aw = np.array([ai_counts.get(w, 0) for w in vocab_list], dtype=float)
                q_shrunk = shrink_p(mu_q, n_ai_aw, n_ai_a, kappa_q)
                logQ_agency = np.log(q_shrunk)
                log1mQ_agency = np.log(1 - q_shrunk)
                baseline_q = float(log1mQ_agency.sum())
                delta_q = logQ_agency - log1mQ_agency
            else:
                baseline_q = pooled_baseline_q
                delta_q = pooled_delta_q
            log_q = np.full(len(filtered), baseline_q)
            for i, idxs in enumerate(sent_indices):
                if len(idxs):
                    log_q[i] += delta_q[idxs].sum()
            agency_prepared[agency_id] = (sent_indices, log_q)

        del agency_data  # free memory

        logging.info(
            "%s: %d agencies with enough pre-ChatGPT sentences",
            doc_type, len(agency_prepared),
        )

        # --- sweep kappa values (+ pooled baseline) ---
        for kappa in kappa_values + [float("inf")]:
            kappa_label = f"{kappa:.0f}" if np.isfinite(kappa) else "pooled"

            for agency_id, (sent_indices, log_q) in agency_prepared.items():
                n_sents = len(sent_indices)

                # Build agency-specific P
                if np.isfinite(kappa):
                    if agency_id in agency_word_counts_data:
                        wc, n_a = agency_word_counts_data[agency_id]
                        n_aw = np.array(
                            [wc.get(w, 0) for w in vocab_list], dtype=float
                        )
                    else:
                        n_aw = np.zeros(len(vocab_list))
                        n_a = 0
                    p_shrunk = shrink_p(mu, n_aw, n_a, kappa)
                    logP = np.log(p_shrunk)
                    log1mP = np.log(1 - p_shrunk)
                    b_p = float(log1mP.sum())
                    d_p = logP - log1mP
                else:
                    b_p = pooled_baseline_p
                    d_p = pooled_delta_p

                # Compute log P(s) for each sentence
                log_p = np.full(n_sents, b_p)
                for i, idxs in enumerate(sent_indices):
                    if len(idxs):
                        log_p[i] += d_p[idxs].sum()

                # MLE for alpha (point estimate only, no bootstrap)
                log_ratio = log_q - log_p

                def _neg_ll(alpha, lr=log_ratio):
                    return -np.mean(np.log(np.maximum(
                        (1 - alpha) + alpha * np.exp(lr), 1e-300
                    )))

                res = minimize_scalar(
                    _neg_ll, bounds=(1e-6, 1 - 1e-6), method="bounded"
                )
                alpha = float(res.x)

                all_results.append({
                    "doc_type": doc_type,
                    "agency_id": agency_id,
                    "kappa": kappa if np.isfinite(kappa) else None,
                    "kappa_label": kappa_label,
                    "spurious_alpha": round(alpha, 6),
                    "n_sentences": n_sents,
                })

    if not all_results:
        logging.warning("No results produced")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    logging.info("Wrote %s (%d rows)", output_path, len(results_df))

    # --- Print summary ---
    print("\n=== Spurious Alpha Summary (pre-ChatGPT, should be ~ 0) ===")
    for dt in results_df["doc_type"].unique():
        dt_df = results_df[results_df["doc_type"] == dt]
        print(f"\n--- {dt} ---")
        summary = dt_df.groupby("kappa_label", sort=False).agg(
            mean_alpha=("spurious_alpha", "mean"),
            median_alpha=("spurious_alpha", "median"),
            max_alpha=("spurious_alpha", "max"),
            pct_above_1pct=("spurious_alpha", lambda x: f"{(x > 0.01).mean() * 100:.0f}%"),
            n_agencies=("agency_id", "nunique"),
        )
        print(summary.to_string())
    print()


# ===================================================================
# Subcommand 5: sweep (kappa sweep on post-ChatGPT data)
# ===================================================================


def cmd_sweep(args: argparse.Namespace) -> None:
    """Sweep kappa values on real (post-ChatGPT) data to compare alpha estimates.

    Loads data once, then runs inference at each kappa value plus vanilla (pooled).
    Outputs a single CSV with a kappa_label column.
    """
    from scipy.optimize import minimize_scalar

    logging.info("=== sweep: kappa comparison on post-ChatGPT data ===")
    ensure_nltk_punkt()

    base_dir = Path(args.base_dir)
    dist_dir = Path(args.distribution_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kappa_values = list(np.logspace(
        np.log10(args.kappa_min), np.log10(args.kappa_max), args.kappa_steps
    ))
    logging.info(
        "Sweeping kappa: %s + pooled",
        [f"{k:.0f}" for k in kappa_values],
    )

    agencies_filter = set(args.agencies) if args.agencies else None
    all_results = []

    for doc_type in args.doc_types:
        dist_path = dist_dir / f"distribution_{doc_type}.parquet"
        agency_counts_path = dist_dir / f"agency_counts_{doc_type}.parquet"
        if not dist_path.exists():
            logging.warning("Missing %s; skipping", dist_path)
            continue
        if not agency_counts_path.exists():
            logging.warning(
                "Missing %s; run `estimate --hierarchical` first",
                agency_counts_path,
            )
            continue

        dist_df = pd.read_parquet(dist_path)
        agency_word_counts_data = load_agency_word_counts(agency_counts_path)

        # --- Pooled arrays (constant across kappa) ---
        vocab_list = list(dist_df["word"].values)
        w2i = {w: i for i, w in enumerate(vocab_list)}
        vocab_set = set(vocab_list)

        logQ_pooled = dist_df["logQ"].values.astype(float)
        log1mQ_pooled = dist_df["log1mQ"].values.astype(float)
        pooled_baseline_q = float(log1mQ_pooled.sum())
        pooled_delta_q = logQ_pooled - log1mQ_pooled
        mu_q = np.exp(logQ_pooled)  # pooled Q frequencies as prior for shrinkage

        # Agency-specific Q
        agency_ai_word_counts = None
        kappa_q = None
        meta_path = dist_dir / "distribution_metadata.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        ai_agency_counts_path = dist_dir / f"agency_ai_counts_{doc_type}.parquet"
        if ai_agency_counts_path.exists():
            agency_ai_word_counts = load_agency_word_counts(ai_agency_counts_path)
            kappa_q = meta.get(f"{doc_type}_optimal_kappa_q")
            if kappa_q is not None:
                logging.info(
                    "%s: agency-specific Q — kappa_q=%.1f, %d AI agencies",
                    doc_type, kappa_q, len(agency_ai_word_counts),
                )
            else:
                agency_ai_word_counts = None

        pooled_logP = dist_df["logP"].values.astype(float)
        pooled_log1mP = dist_df["log1mP"].values.astype(float)
        pooled_baseline_p = float(pooled_log1mP.sum())
        pooled_delta_p = pooled_logP - pooled_log1mP
        mu = np.exp(pooled_logP)

        # --- Load dedup reps ---
        dedup_reps = None
        if args.dedup:
            dedup_reps = load_dedup_representatives(base_dir, doc_type)

        # --- Load and tokenize ALL documents ONCE ---
        logging.info("%s: loading documents...", doc_type)
        input_files = list(iter_input_files(base_dir, doc_type))

        records: List[Dict] = []
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            from functools import partial
            load_fn = partial(
                load_and_tokenize_file,
                agencies_filter=agencies_filter,
                dedup_reps=dedup_reps,
            )
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(load_fn, f): f for f in input_files}
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc=f"load {doc_type}"
                ):
                    records.extend(future.result())
        else:
            for csv_path in tqdm(input_files, desc=f"load {doc_type}"):
                records.extend(
                    load_and_tokenize_file(csv_path, agencies_filter, dedup_reps)
                )

        if not records:
            logging.warning("No records for %s", doc_type)
            continue

        rec_df = pd.DataFrame(records)
        logging.info("%s: %d documents loaded", doc_type, len(rec_df))

        # --- Build strata (agency x quarter) ---
        group_cols = []
        if "agency" in args.stratify_by:
            group_cols.append("agency_id")
        if "quarter" in args.stratify_by:
            group_cols.append("quarter")
        elif "half" in args.stratify_by:
            group_cols.append("half")
        elif "year" in args.stratify_by:
            group_cols.append("year")

        if not group_cols:
            groups = [("all", rec_df)]
        else:
            groups = list(rec_df.groupby(group_cols))

        # Pre-collect per-stratum sentences and pre-index words (constant across kappa)
        strata = []
        for key, grp in groups:
            all_sents = []
            for sent_list in grp["sentences"]:
                all_sents.extend(sent_list)
            if len(all_sents) < args.min_sentences:
                continue

            # Pre-compute word indices per sentence (reused across kappa)
            filtered = [s for s in all_sents if set(s) & vocab_set]
            if len(filtered) < 10:
                continue
            sent_indices = []
            for s in filtered:
                idxs = np.array(
                    [w2i[w] for w in set(s) if w in w2i], dtype=np.intp
                )
                sent_indices.append(idxs)

            # Extract agency_id for hierarchical P and Q
            if "agency_id" in group_cols:
                idx = group_cols.index("agency_id")
                agency_id = key[idx] if isinstance(key, tuple) else key
            else:
                agency_id = "__pooled__"

            # Compute log Q — agency-specific if available, else pooled
            if agency_ai_word_counts is not None and kappa_q is not None and agency_id in agency_ai_word_counts:
                ai_counts, n_ai_a = agency_ai_word_counts[agency_id]
                n_ai_aw = np.array([ai_counts.get(w, 0) for w in vocab_list], dtype=float)
                q_shrunk = shrink_p(mu_q, n_ai_aw, n_ai_a, kappa_q)
                logQ_a = np.log(q_shrunk)
                log1mQ_a = np.log(1 - q_shrunk)
                baseline_q = float(log1mQ_a.sum())
                delta_q = logQ_a - log1mQ_a
            else:
                baseline_q = pooled_baseline_q
                delta_q = pooled_delta_q
            log_q = np.full(len(filtered), baseline_q)
            for i, idxs in enumerate(sent_indices):
                if len(idxs):
                    log_q[i] += delta_q[idxs].sum()

            strata.append({
                "key": key,
                "n_docs": len(grp),
                "agency_id": agency_id,
                "sent_indices": sent_indices,
                "log_q": log_q,
                "n_used": len(filtered),
            })

        logging.info(
            "%s: %d strata ready, sweeping %d kappa values + pooled",
            doc_type, len(strata), len(kappa_values),
        )

        # --- Sweep kappa values + pooled ---
        for kappa in kappa_values + [float("inf")]:
            kappa_label = f"{kappa:.0f}" if np.isfinite(kappa) else "pooled"
            logging.info("%s: running kappa=%s ...", doc_type, kappa_label)

            # Build tasks for this kappa
            sweep_tasks = []
            for stratum in strata:
                # Build P for this kappa
                agency_id = stratum["agency_id"]
                if np.isfinite(kappa) and agency_id in agency_word_counts_data:
                    wc, n_a = agency_word_counts_data[agency_id]
                    n_aw = np.array(
                        [wc.get(w, 0) for w in vocab_list], dtype=float
                    )
                    p_shrunk = shrink_p(mu, n_aw, n_a, kappa)
                    logP = np.log(p_shrunk)
                    log1mP = np.log(1 - p_shrunk)
                    b_p = float(log1mP.sum())
                    d_p = logP - log1mP
                else:
                    b_p = pooled_baseline_p
                    d_p = pooled_delta_p

                sweep_tasks.append((
                    stratum["key"], stratum["n_docs"],
                    stratum["sent_indices"], stratum["log_q"],
                    stratum["n_used"], b_p, d_p, args.bootstrap_n,
                ))

            if args.workers > 1:
                from concurrent.futures import ProcessPoolExecutor, as_completed
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = {
                        pool.submit(sweep_infer_stratum, t): t[0]
                        for t in sweep_tasks
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        row = {
                            "doc_type": doc_type,
                            "kappa": kappa if np.isfinite(kappa) else None,
                            "kappa_label": kappa_label,
                            "alpha_estimate": round(result["alpha"], 6),
                            "ci_lower": round(result["ci_lo"], 6),
                            "ci_upper": round(result["ci_hi"], 6),
                            "n_documents": result["n_docs"],
                            "n_sentences": result["n_used"],
                        }
                        key = result["key"]
                        if not isinstance(key, tuple):
                            key = (key,)
                        for col, val in zip(group_cols, key):
                            row[col] = val
                        all_results.append(row)
            else:
                for task in sweep_tasks:
                    result = sweep_infer_stratum(task)
                    row = {
                        "doc_type": doc_type,
                        "kappa": kappa if np.isfinite(kappa) else None,
                        "kappa_label": kappa_label,
                        "alpha_estimate": round(result["alpha"], 6),
                        "ci_lower": round(result["ci_lo"], 6),
                        "ci_upper": round(result["ci_hi"], 6),
                        "n_documents": result["n_docs"],
                        "n_sentences": result["n_used"],
                    }
                    key = result["key"]
                    if not isinstance(key, tuple):
                        key = (key,)
                    for col, val in zip(group_cols, key):
                        row[col] = val
                    all_results.append(row)

    if not all_results:
        logging.warning("No results produced")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    logging.info("Wrote %s (%d rows)", output_path, len(results_df))

    # --- Print summary ---
    print("\n=== Alpha Estimates by kappa (post-ChatGPT) ===")
    for dt in results_df["doc_type"].unique():
        dt_df = results_df[results_df["doc_type"] == dt]
        print(f"\n--- {dt} ---")
        summary = dt_df.groupby("kappa_label", sort=False).agg(
            mean_alpha=("alpha_estimate", "mean"),
            median_alpha=("alpha_estimate", "median"),
            max_alpha=("alpha_estimate", "max"),
            n_strata=("alpha_estimate", "count"),
        )
        print(summary.to_string())
    print()


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
    p_gen.add_argument("--max-output-tokens", type=int, default=2048,
                       dest="max_tokens",
                       help="Max tokens the LLM can generate in its rewrite (default: 2048)")
    p_gen.add_argument("--max-input-tokens", type=int, default=6144,
                       help="Max input tokens for the source document (default: 6144). "
                            "Documents longer than this are truncated.")
    p_gen.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallel size for vLLM offline mode (number of GPUs). Default: 1.",
    )
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
    p_gen.add_argument(
        "--agencies", nargs="+", default=None,
        help="Only generate rewrites for these agencies (e.g. --agencies DOT IRS NOAA). "
             "If not set, generates for all agencies.",
    )
    p_gen.add_argument("--overwrite", action="store_true")
    p_gen.add_argument("--debug", action="store_true",
                       help="Debug mode: only process 10 documents, skip cost confirmation")
    p_gen.add_argument(
        "--service-tier", default=None,
        choices=["auto", "default", "flex", "scale", "priority"],
        help="OpenAI service tier (e.g. 'flex' for cheaper flex processing). "
             "Only applies to OpenAI models.",
    )
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
        "--min-human-frac", type=float, default=DEFAULT_MIN_HUMAN_FRAC,
        help="Min word frequency as fraction of human sentences (e.g. 0.001 = 0.1%%). "
             "Overrides --min-human-count when set.",
    )
    p_est.add_argument(
        "--min-ai-frac", type=float, default=DEFAULT_MIN_AI_FRAC,
        help="Min word frequency as fraction of AI sentences (e.g. 0.001 = 0.1%%). "
             "Overrides --min-ai-count when set.",
    )
    p_est.add_argument(
        "--max-vocab", type=int, default=None,
        help="Cap vocabulary to top K words by AI frequency. "
             "Applied after min count/frac filtering.",
    )
    p_est.add_argument(
        "--dedup", action="store_true",
        help="Use minhash dedup mapper files to keep only one representative per "
             "cluster when building the human (P) distribution. Looks for "
             "<doc_type>_all_text__dedup_mapper.csv.gz files alongside the input CSVs.",
    )
    p_est.add_argument(
        "--no-matched", action="store_true",
        help="By default, the human corpus is filtered to only documents that have "
             "AI rewrites (matched P/Q). Use this flag to disable matching and use "
             "the full human corpus.",
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
    p_est.add_argument(
        "--hierarchical", action="store_true",
        help="Additionally save per-agency word counts and optimise kappa "
             "(Beta-Binomial shrinkage). Required before `infer --hierarchical`.",
    )
    p_est.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for file processing (default: 1). "
             "Set to e.g. 8 or 16 to parallelize CSV reading + tokenization.",
    )
    p_est.add_argument(
        "--agencies", nargs="+", default=None,
        help="Limit estimate to these agency IDs (e.g. --agencies CMS EPA). "
             "If not set, all agencies are included.",
    )
    p_est.add_argument(
        "--per-agency", action="store_true",
        help="Build separate vocabulary and P/Q distributions for each agency. "
             "Writes to {output-dir}/{agency_id}/distribution_{doc_type}.parquet. "
             "Only agencies with >= --min-agency-docs human sentences are included.",
    )
    p_est.add_argument(
        "--min-agency-docs", type=int, default=100,
        help="Minimum human sentences per agency for --per-agency (default: 100)",
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
        choices=["agency", "quarter", "half", "year", "doc_type"],
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
        "--workers", type=int, default=1,
        help="Number of parallel workers for stratum inference (default: 1)",
    )
    p_inf.add_argument(
        "--subsample-frac", type=float, default=None,
        help="Randomly subsample this fraction of documents before inference "
             "(e.g. 0.05 for 5%%). Useful for large corpora like public_submission.",
    )
    p_inf.add_argument(
        "--subsample-seed", type=int, default=None,
        help="Random seed for subsampling (default: None = random).",
    )
    p_inf.add_argument(
        "--document-scores",
        default=None,
        metavar="PATH",
        help="If set, compute per-document AI scores using the stratum alpha "
             "and write to this CSV path. Each document gets the mean posterior "
             "P(AI | sentence) averaged across its sentences.",
    )
    p_inf.add_argument(
        "--output",
        default=str(DEFAULT_DATA_DIR / "ai_usage_results.csv.gz"),
    )
    p_inf.add_argument(
        "--hierarchical", action="store_true",
        help="Use agency-specific P distributions (Beta shrinkage toward pool). "
             "Requires `estimate --hierarchical` to have been run first.",
    )
    p_inf.add_argument(
        "--per-agency", action="store_true",
        help="Use fully separate per-agency P/Q distributions. "
             "Looks for {distribution-dir}/{agency_id}/distribution_{doc_type}.parquet. "
             "Requires `estimate --per-agency` to have been run first. "
             "Falls back to pooled distribution for agencies without their own.",
    )
    p_inf.add_argument(
        "--kappa", type=float, default=None,
        help="Concentration parameter kappa for hierarchical mode. "
             "If not set, uses the optimal kappa from estimate metadata.",
    )

    # --- evaluate ---
    p_eval = sub.add_parser(
        "evaluate",
        help="Sweep kappa values and measure spurious alpha on pre-ChatGPT data",
    )
    p_eval.add_argument(
        "--doc-types", nargs="+", default=DOC_TYPES, choices=DOC_TYPES,
    )
    p_eval.add_argument(
        "--distribution-dir",
        default=str(DEFAULT_DATA_DIR / "ai_usage_distributions"),
    )
    p_eval.add_argument("--kappa-min", type=float, default=10.0)
    p_eval.add_argument("--kappa-max", type=float, default=100000.0)
    p_eval.add_argument("--kappa-steps", type=int, default=10)
    p_eval.add_argument("--min-sentences", type=int, default=DEFAULT_MIN_SENTENCES)
    p_eval.add_argument("--agencies", nargs="*", default=None)
    p_eval.add_argument(
        "--dedup", action="store_true",
        help="Filter to dedup cluster representatives",
    )
    p_eval.add_argument(
        "--output",
        default=str(DEFAULT_DATA_DIR / "hierarchical_eval.csv"),
        help="Output CSV with per-agency spurious alpha at each kappa",
    )
    p_eval.add_argument(
        "--max-docs-per-agency", type=int, default=None,
        help="Randomly sample at most this many docs per agency (for speed).",
    )

    # --- sweep ---
    p_sweep = sub.add_parser(
        "sweep",
        help="Sweep kappa values on post-ChatGPT data to compare alpha estimates. "
             "Loads data once, runs inference at each kappa + pooled baseline.",
    )
    p_sweep.add_argument(
        "--doc-types", nargs="+", default=DOC_TYPES, choices=DOC_TYPES,
    )
    p_sweep.add_argument(
        "--distribution-dir",
        default=str(DEFAULT_DATA_DIR / "ai_usage_distributions"),
    )
    p_sweep.add_argument("--kappa-min", type=float, default=10.0)
    p_sweep.add_argument("--kappa-max", type=float, default=100000.0)
    p_sweep.add_argument("--kappa-steps", type=int, default=10)
    p_sweep.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N)
    p_sweep.add_argument("--min-sentences", type=int, default=DEFAULT_MIN_SENTENCES)
    p_sweep.add_argument("--agencies", nargs="*", default=None)
    p_sweep.add_argument(
        "--stratify-by",
        nargs="+",
        default=["agency", "quarter"],
        choices=["agency", "quarter", "half", "year", "doc_type"],
    )
    p_sweep.add_argument(
        "--dedup", action="store_true",
        help="Filter to dedup cluster representatives",
    )
    p_sweep.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for data loading (default: 1)",
    )
    p_sweep.add_argument(
        "--output",
        default=str(DEFAULT_DATA_DIR / "kappa_sweep_results.csv"),
        help="Output CSV with alpha estimates at each kappa (default: %(default)s)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # Silence noisy HTTP request logs from openai/httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "estimate":
        cmd_estimate(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "sweep":
        cmd_sweep(args)


if __name__ == "__main__":
    main()
