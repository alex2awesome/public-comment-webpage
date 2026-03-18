"""Retrieval-Based Comment–Response Matching Pipeline.

For each agency/year directory, builds three retrieval indexes over
comment clusters (at either claim or comment level):
  1. BM25 (sparse) — for distribution to users without GPUs
  2. nvidia/llama-embed-nemotron-8b (dense) — primary, used for matching
  3. all-mpnet-base-v2 (dense) — for distribution

Retrieves top-k candidates per government response using the primary
dense model, then uses LLM sampling to find an optimal dense-score
threshold for final labeling.

Outputs per agency/year directory:
  - public_submission_all_text__{level}_response_matches.csv
  - public_submission_all_text__{level}_comment_labels.csv
  - .retriv_indexes/ (BM25 + dense indexes for distribution)

Global output:
  - pipeline_log.csv  (one row per directory, tracks strategy/metrics)

Usage:
    python data/bulk_downloads/scripts/match_comments_and_create_indexes.py --level claims --prompt-backend openai --llm-model gpt-5-mini --top-k 10
    python data/bulk_downloads/scripts/match_comments_and_create_indexes.py --level claims --llm-model gpt-5 gpt-5-mini --collect-training-data --training-samples-per-dir 200
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import atexit
import json
import logging
import os
import random
import re
import signal
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import filelock
import numpy as np
import pandas as pd

# Monkey-patch json.JSONEncoder.default so autofaiss can serialize numpy scalars.
_orig_json_default = json.JSONEncoder.default

def _numpy_json_default(self, obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return _orig_json_default(self, obj)

json.JSONEncoder.default = _numpy_json_default  # type: ignore[assignment]
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Ensure notebooks/ is on sys.path so we can import prompt_utils
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent.parent.parent  # regulations-demo/
_NOTEBOOKS_DIR = _PROJECT_ROOT / "notebooks"
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_MODE", "disabled")
if not os.environ.get("OPENAI_API_KEY"):
    key_path = _PROJECT_ROOT.parent / ".openai-salt-lab-key.txt"
    if key_path.exists():
        os.environ["OPENAI_API_KEY"] = key_path.read_text().strip()

import prompt_utils  # noqa: E402

# ---------------------------------------------------------------------------
# robust_json_load copied from utils.py to avoid IPython dependency
# ---------------------------------------------------------------------------


def robust_json_load(value):
    """Best-effort parse for JSON-ish strings."""
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return ast.literal_eval(text)
        except (ValueError, SyntaxWarning, SyntaxError):
            try:
                escaped = text.replace("\\", "\\\\")
                return json.loads(escaped)
            except Exception:
                return []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BULK_DIR = _SCRIPTS_DIR.parent  # data/bulk_downloads/
LOG_FILE = _SCRIPTS_DIR / "pipeline_log.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
for noisy_logger in ("openai", "httpx", "httpcore"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Lock file management (following comments_extract_claims.py pattern)
# ---------------------------------------------------------------------------
_active_processing_files: Set[Path] = set()


def _cleanup_processing_files() -> None:
    for p in list(_active_processing_files):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    _active_processing_files.clear()


def _signal_handler(signum, frame) -> None:
    _cleanup_processing_files()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


atexit.register(_cleanup_processing_files)


# ---------------------------------------------------------------------------
# 1. Load response data
# ---------------------------------------------------------------------------


def load_response_df() -> pd.DataFrame:
    candidates = [
        _NOTEBOOKS_DIR / "2026-02-10__comment-response-cache.csv",
        _PROJECT_ROOT / "2026-02-10__comment-response-cache.csv",
        Path("2026-02-10__comment-response-cache.csv"),
    ]
    response_cache_path = None
    for c in candidates:
        if c.exists():
            response_cache_path = c
            break
    if response_cache_path is None:
        raise FileNotFoundError(
            f"Response cache not found. Tried: {[str(c) for c in candidates]}"
        )

    logger.info("Loading response cache from %s", response_cache_path)
    orig = pd.read_csv(response_cache_path, index_col=0).assign(
        parsed_response=lambda df: df["summarized_response"].apply(robust_json_load)
    ).drop(columns="summarized_response")

    proc = (
        orig.assign(
            parsed_response=lambda df: df["parsed_response"].apply(
                lambda x: x if isinstance(x, list) else [x]
            )
        )
        .loc[lambda df: df["parsed_response"].str.len() > 0]
        .explode("parsed_response")
        .reset_index(drop=True)
        .assign(
            parsed_response=lambda df: df["parsed_response"].apply(
                lambda x: x[0] if isinstance(x, list) else x
            )
        )
        .pipe(
            lambda df: pd.concat(
                [
                    df[["Agency ID", "Docket ID"]],
                    pd.DataFrame(df["parsed_response"].tolist()),
                ],
                axis=1,
            )
        )
        .drop(
            columns=["error", "detail", "commenter_identifiers_Text"],
            errors="ignore",
        )
    )
    logger.info("Loaded %d response rows", len(proc))
    return proc


# ---------------------------------------------------------------------------
# 2. Data loading for claims / comment levels
# ---------------------------------------------------------------------------


def _parse_agency_year(dir_name: str) -> tuple[str, str]:
    """Extract agency and year range from directory name like 'epa_2020_2021'."""
    m = re.match(r"^([a-z]+)_(\d{4}_\d{4})$", dir_name)
    if m:
        return m.group(1), m.group(2)
    return dir_name, ""


def load_claims_data(agency_year_dir: Path) -> Optional[pd.DataFrame]:
    """Load claims.csv for a directory. Returns exploded claim rows or None."""
    claims_path = agency_year_dir / "public_submission_all_text__claims.csv"
    if not claims_path.exists():
        logger.debug("No claims file at %s", claims_path)
        return None

    agency, year_range = _parse_agency_year(agency_year_dir.name)

    claims_df = pd.read_csv(claims_path, low_memory=False)
    rows = []
    for _, row in claims_df.iterrows():
        cluster_uid = str(row.get("cluster_uid", ""))
        if not cluster_uid:
            continue

        # Try claims_parsed_json first, then claims_fix_parsed_json
        claims_list = None
        if row.get("parse_ok") is True or str(row.get("parse_ok", "")).lower() == "true":
            claims_list = robust_json_load(row.get("claims_parsed_json", ""))
        if not claims_list:
            if row.get("fix_parse_ok") is True or str(row.get("fix_parse_ok", "")).lower() == "true":
                claims_list = robust_json_load(row.get("claims_fix_parsed_json", ""))
        if not claims_list:
            continue

        docket_id = str(row.get("docket_id", ""))
        document_id = str(row.get("document_id", ""))

        for i, claim_text in enumerate(claims_list):
            if not isinstance(claim_text, str) or not claim_text.strip():
                continue
            claim_id = f"{agency}__{year_range}__{docket_id}__{document_id}__{i}"
            rows.append({
                "id": claim_id,
                "text": claim_text.strip(),
                "cluster_uid": cluster_uid,
                "docket_id": docket_id,
                "document_id": document_id,
            })

    if not rows:
        logger.warning("No valid claims found in %s", claims_path)
        return None

    return pd.DataFrame(rows)


def load_comment_data(agency_year_dir: Path) -> Optional[pd.DataFrame]:
    """Load comment text data for a directory, one row per cluster."""
    mapper_path = agency_year_dir / "public_submission_all_text__dedup_mapper.csv"
    all_text_path = agency_year_dir / "public_submission_all_text.csv"

    if not mapper_path.exists():
        logger.warning("No dedup mapper at %s", mapper_path)
        return None
    if not all_text_path.exists():
        logger.warning("No all_text file at %s", all_text_path)
        return None

    agency, year_range = _parse_agency_year(agency_year_dir.name)

    mapper = pd.read_csv(mapper_path, low_memory=False)
    all_text = pd.read_csv(
        all_text_path,
        usecols=["Document ID", "Docket ID", "canonical_text"],
        low_memory=False,
    )

    mapper["document_id"] = mapper["document_id"].astype(str)
    all_text["Document ID"] = all_text["Document ID"].astype(str)

    merged = mapper.merge(
        all_text,
        left_on="document_id",
        right_on="Document ID",
        how="inner",
    )
    if merged.empty:
        logger.warning("Empty merge for %s", agency_year_dir)
        return None

    merged["canonical_text"] = merged["canonical_text"].fillna("")
    merged["text_len"] = merged["canonical_text"].str.len()

    # Pick the longest text per cluster as representative
    reps = (
        merged.sort_values("text_len", ascending=False)
        .groupby("cluster_uid", as_index=False)
        .head(1)
    )

    rows = []
    for _, row in reps.iterrows():
        cluster_uid = str(row.get("cluster_uid", ""))
        text = str(row.get("canonical_text", "")).strip()
        if not cluster_uid or not text:
            continue
        docket_id = str(row.get("docket_id", ""))
        document_id = str(row.get("document_id", ""))
        comment_id = f"{agency}__{year_range}__{docket_id}__{document_id}"
        rows.append({
            "id": comment_id,
            "text": text,
            "cluster_uid": cluster_uid,
            "docket_id": docket_id,
            "document_id": document_id,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def build_collection(agency_year_dir: Path, level: str) -> Optional[pd.DataFrame]:
    """Build retriv collection for a directory. Returns DataFrame with id, text, cluster_uid, docket_id."""
    if level == "claims":
        return load_claims_data(agency_year_dir)
    else:
        return load_comment_data(agency_year_dir)


# ---------------------------------------------------------------------------
# 3. Index management
# ---------------------------------------------------------------------------


def _index_exists(base_path: Path, index_name: str, kind: str) -> bool:
    """Check if a retriv index exists on disk."""
    idx_dir = base_path / "collections" / index_name
    if kind == "sparse":
        return (idx_dir / "sr_state.npz").exists()
    else:
        return (idx_dir / "dr_state.npz").exists()


DISTRIBUTION_DENSE_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
]


# Sane max_length caps per model to avoid padding to huge context windows
# (e.g. Llama 3.1's max_position_embeddings=131072 would pad every input
# to 128K tokens, making even 21 short texts take forever).
_MODEL_MAX_LENGTH = {
    "nvidia/llama-embed-nemotron-8b": 4096,  # NVIDIA recommends 4096 in examples
}


# Cache encoders across directories so the 8B model is loaded once and reused.
_encoder_cache: dict[str, object] = {}


def _build_or_load_one_dense(
    index_base: Path,
    index_name: str,
    model_name: str,
    collection: List[dict],
    batch_size: int,
    overwrite: bool,
):
    """Build or load a single dense index. Returns DenseRetriever."""
    from retriv import DenseRetriever

    # Use a suffix derived from the model name for unique index dirs.
    safe_suffix = model_name.replace("/", "_")
    full_name = f"{index_name}_{safe_suffix}"
    max_len = _MODEL_MAX_LENGTH.get(model_name, None)

    if not overwrite and _index_exists(index_base, full_name, "dense"):
        logger.info("Loading existing dense index: %s (model=%s)", full_name, model_name)
        cached = _encoder_cache.get(model_name)
        if cached is not None:
            # Load index without re-loading the encoder, then assign the cached one.
            logger.info("Reusing cached encoder for %s", model_name)
            dr = DenseRetriever.load(full_name, encoder=cached)
        else:
            dr = DenseRetriever.load(full_name)
            # Override max_length on loaded encoder if needed (saved state may
            # have the uncapped value from a previous build).
            if max_len and dr.encoder is not None and dr.encoder.max_length != max_len:
                logger.info("Overriding encoder max_length %s → %s", dr.encoder.max_length, max_len)
                dr.encoder.max_length = max_len
                dr.encoder.tokenizer_kwargs["max_length"] = max_len
            # Ensure encoder is on GPU after loading (retriv may deserialize to CPU).
            # NOTE: Encoder.model is the model NAME (str); the actual AutoModel
            # is stored as Encoder.encoder.
            if dr.encoder is not None and hasattr(dr.encoder, "encoder"):
                import torch
                if torch.cuda.is_available():
                    try:
                        device = next(dr.encoder.encoder.parameters()).device
                        if str(device) == "cpu":
                            logger.info("Moving encoder to CUDA (was on %s)", device)
                            dr.encoder.encoder.to("cuda")
                            dr.encoder.device = "cuda"
                    except StopIteration:
                        pass
            # Cache the encoder for reuse
            if dr.encoder is not None:
                _encoder_cache[model_name] = dr.encoder
                logger.info("Cached encoder for %s", model_name)
    else:
        logger.info(
            "Building dense index: %s (%d docs, model=%s, max_length=%s)",
            full_name,
            len(collection),
            model_name,
            max_len,
        )
        # Reuse cached encoder if available — pass it directly to avoid
        # redundant model loading in the DenseRetriever constructor.
        cached = _encoder_cache.get(model_name)
        if cached is not None:
            logger.info("Building index with cached encoder for %s", model_name)
            dr = DenseRetriever(
                index_name=full_name,
                model=model_name,
                normalize=True,
                use_ann=True,
                max_length=max_len,
                encoder=cached,
            )
        else:
            dr = DenseRetriever(
                index_name=full_name,
                model=model_name,
                normalize=True,
                use_ann=True,
                max_length=max_len,
            )
            if dr.encoder is not None:
                _encoder_cache[model_name] = dr.encoder
                logger.info("Cached encoder for %s", model_name)
        dr.index(collection, batch_size=batch_size, show_progress=True)

    # Fail fast if the ANN index wasn't built (e.g. autofaiss memory error).
    if dr.use_ann and (dr.ann_searcher is None or dr.ann_searcher.faiss_index is None):
        raise RuntimeError(
            f"Dense ANN index build failed for {full_name}. "
            "Delete the .retriv_indexes directory and retry, or check autofaiss logs."
        )
    return dr


def build_or_load_indexes(
    agency_year_dir: Path,
    level: str,
    collection: List[dict],
    primary_embedding_model: str,
    batch_size: int,
    overwrite: bool,
    skip_distribution: bool = False,
):
    """Build or load BM25 + dense indexes for a directory.

    Builds three indexes:
      1. BM25 (sparse) — for distribution to users without GPUs
      2. Primary dense (primary_embedding_model) — used for matching/search
      3. Distribution dense (all-mpnet-base-v2) — for distribution

    Returns the primary DenseRetriever (used for search).
    """
    import retriv
    from retriv import SparseRetriever

    index_base = agency_year_dir / ".retriv_indexes"
    index_base.mkdir(parents=True, exist_ok=True)
    retriv.set_base_path(str(index_base))

    dir_name = agency_year_dir.name  # e.g. "msha_2017_2018"
    index_name = f"{dir_name}_{level}"

    # 1. Sparse (BM25) — for distribution (skip in training-data mode)
    if skip_distribution:
        logger.info("Skipping BM25 index (training data mode)")
    elif not overwrite and _index_exists(index_base, f"{index_name}_bm25", "sparse"):
        logger.info("Loading existing BM25 index: %s", index_name)
        SparseRetriever.load(f"{index_name}_bm25")
    else:
        logger.info("Building BM25 index: %s (%d docs)", index_name, len(collection))
        sr = SparseRetriever(
            index_name=f"{index_name}_bm25",
            model="bm25",
            min_df=1,
        )
        sr.index(collection, show_progress=True)

    # 2. Primary dense index — used for matching/search
    primary_dr = _build_or_load_one_dense(
        index_base, index_name, primary_embedding_model,
        collection, batch_size, overwrite,
    )

    # 3. Distribution dense indexes — built but not used for search
    #    Free each distribution model after building to reclaim GPU memory
    #    so the primary 8B model can be used for query encoding.
    #    Skipped entirely when collecting training data (only the primary
    #    model is needed for retrieval).
    if skip_distribution:
        logger.info("Skipping distribution dense indexes (training data mode)")
        return primary_dr

    for dist_model in DISTRIBUTION_DENSE_MODELS:
        if dist_model == primary_embedding_model:
            continue  # already built above
        try:
            dist_dr = _build_or_load_one_dense(
                index_base, index_name, dist_model,
                collection, batch_size, overwrite,
            )
            # Explicitly free the distribution encoder to reclaim GPU memory.
            if dist_dr.encoder is not None:
                _encoder_cache.pop(dist_model, None)
                del dist_dr.encoder
            del dist_dr
        except Exception as e:
            logger.warning(
                "Failed to build distribution index %s for %s: %s",
                dist_model, index_name, e,
            )

    # Clear GPU cache after building distribution indexes so the primary
    # model has full GPU memory available for query encoding.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return primary_dr


# ---------------------------------------------------------------------------
# 4. Retrieval
# ---------------------------------------------------------------------------


def retrieve_for_response(
    dr,
    response_text: str,
    docket_doc_ids: List[str],
    k: int = 10,
    encoded_query=None,
) -> List[dict]:
    """Query the primary dense index for one response, filtering to docket docs.

    Returns list of dicts with doc_id, dense_score.
    If encoded_query is provided, skip re-encoding for the dense retriever.
    """
    if not docket_doc_ids or not response_text.strip():
        return []

    results = []
    try:
        dense_hits = dr.search(
            query=response_text if encoded_query is None else None,
            encoded_query=encoded_query,
            include_id_list=docket_doc_ids,
            return_docs=False,
            cutoff=k,
        )
        for hit in dense_hits:
            doc_id = hit["id"] if isinstance(hit, dict) else hit
            score = hit.get("score", 0.0) if isinstance(hit, dict) else 0.0
            results.append({"doc_id": doc_id, "dense_score": score})
    except Exception as e:
        logger.warning("Dense search error for %d docs: %s", len(docket_doc_ids), e)

    return results


# ---------------------------------------------------------------------------
# 5. LLM accuracy check
# ---------------------------------------------------------------------------

CLAIMS_MATCHING_PROMPT = """You are an expert legal assistant.
I am analyzing government responses to comments submitted during the notice & comment process.
I will show you a government response and a specific claim extracted from a public comment.
Tell me whether this government response is addressing this claim:
either directly or as part of a larger group of similar comments.
Answer with "yes" or "no". Don't say anything else.

<claim>
{claim}
</claim>

<original_comment_excerpt>
{original_comment}
</original_comment_excerpt>

<response>
{response}
</response>

Your response:
"""

COMMENT_MATCHING_PROMPT = """You are an expert legal assistant.
I am analyzing government responses to comments submitted during the notes & comment process.
I will show you a comment and a government response to comments. You will tell me whether the response is responding to this comment:
either directly an individual comment or as part of a larger group.
Be careful: even comments that are not being responded to are likely to be semantically similar, so really read them carefully.
Ignore any "official"-seeming correlates, like letterhead, signatures, citations of evidence in the comment.
Only look directly at the content of the comment and whether the response is responding to it.
Answer with "yes" or "no". Don't say anything else.

Here is a comment:

<comment>
{comment}
</comment>

<response>
{response}
</response>

Your response:
"""


def _normalize_label(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    raw = str(value).strip().lower()
    # Check for API error JSON — treat as unlabeled
    if raw.startswith("{") and "error" in raw:
        return ""
    # Extract yes/no from verbose responses (e.g. "Yes, this response...")
    if raw in ("yes", "no"):
        return raw
    # Check if first word is yes/no (common with verbose models)
    first_word = raw.split(",")[0].split(".")[0].split()[0] if raw else ""
    if first_word in ("yes", "no"):
        return first_word
    return raw


def sample_pairs_for_llm(all_pairs_df: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    """Stratified sample across dense_score distribution."""
    df = all_pairs_df.copy()
    if len(df) <= n:
        return df

    # Stratified sampling: 10 bins across dense_score
    df["_score_bin"] = pd.cut(df["dense_score"], bins=10, labels=False)
    df["_score_bin"] = df["_score_bin"].fillna(0).astype(int)

    sampled_parts = []
    per_bin = max(1, n // 10)
    for _, bin_df in df.groupby("_score_bin"):
        sample_n = min(len(bin_df), per_bin)
        sampled_parts.append(bin_df.sample(n=sample_n, random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)

    # If we have fewer than n, sample more from the full set
    if len(sampled) < n:
        remaining = df.loc[~df.index.isin(sampled.index)]
        extra = min(n - len(sampled), len(remaining))
        if extra > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=extra, random_state=42)],
                ignore_index=True,
            )

    # If we have more than n, downsample
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=42).reset_index(drop=True)

    sampled = sampled.drop(columns=["_score_bin"], errors="ignore")
    return sampled


def build_matching_prompt(
    level: str,
    response_text: str,
    candidate_text: str,
    original_comment: Optional[str] = None,
) -> str:
    if level == "claims":
        return CLAIMS_MATCHING_PROMPT.format(
            claim=candidate_text[:20000],
            original_comment=(original_comment or "")[:7500],
            response=response_text[:20000],
        )
    else:
        return COMMENT_MATCHING_PROMPT.format(
            comment=candidate_text[:20000],
            response=response_text[:20000],
        )


async def run_llm_check(
    sampled_df: pd.DataFrame,
    level: str,
    backend: str,
    model: str,
    collection_df: pd.DataFrame,
    original_texts: Optional[dict] = None,
) -> pd.DataFrame:
    """Label sampled pairs via LLM. Returns DataFrame with llm_label column."""
    prompts = []
    for _, row in sampled_df.iterrows():
        response_text = str(row.get("response_text", ""))
        candidate_text = str(row.get("candidate_text", ""))
        original_comment = None
        if level == "claims" and original_texts is not None:
            cluster_uid = str(row.get("doc_id", "")).split("::claim_")[0]
            original_comment = original_texts.get(cluster_uid, "")
        prompts.append(
            build_matching_prompt(level, response_text, candidate_text, original_comment)
        )

    logger.info(
        "Querying %d LLM prompts (model=%s, backend=%s)",
        len(prompts),
        model,
        backend,
    )
    raw_labels = await prompt_utils.process_batch(
        prompts=prompts,
        model=model,
        backend=backend,
    )
    sampled_df = sampled_df.copy()
    normalized = [_normalize_label(v) for v in raw_labels]
    sampled_df["llm_label"] = normalized

    # In debug mode, print a sample positive and negative prompt+response
    if logger.isEnabledFor(logging.DEBUG) or os.environ.get("DEBUG"):
        yes_idxs = [i for i, l in enumerate(normalized) if l == "yes"]
        no_idxs = [i for i, l in enumerate(normalized) if l == "no"]
        for tag, idxs in [("POSITIVE (yes)", yes_idxs), ("NEGATIVE (no)", no_idxs)]:
            if idxs:
                i = idxs[0]
                logger.info(
                    "Sample %s prompt:\n%s\nLLM response: %s",
                    tag, prompts[i][:2000], raw_labels[i],
                )

    # Log label distribution and sample non-yes/no responses for debugging
    from collections import Counter
    label_counts = Counter(normalized)
    logger.info("LLM label distribution: %s", dict(label_counts))
    bad_labels = [v for v in raw_labels if _normalize_label(v) not in ("yes", "no")]
    if bad_labels:
        logger.warning(
            "Sample non-yes/no LLM responses (%d total): %s",
            len(bad_labels),
            [str(v)[:200] for v in bad_labels[:5]],
        )

    return sampled_df


async def run_multi_model_llm_check(
    sampled_df: pd.DataFrame,
    level: str,
    backend: str,
    models: List[str],
    collection_df: pd.DataFrame,
    original_texts: Optional[dict] = None,
) -> pd.DataFrame:
    """Label pairs using multiple LLM models with uniform random assignment.

    Each pair is randomly assigned to one of the models. Results are merged
    back with an ``llm_model`` column recording which model labeled each pair.
    """
    if len(models) == 1:
        result = await run_llm_check(
            sampled_df, level, backend, models[0], collection_df, original_texts,
        )
        result["llm_model"] = models[0]
        return result

    # Randomly assign each row to a model (uniform)
    assignments = [random.choice(models) for _ in range(len(sampled_df))]
    sampled_df = sampled_df.copy()
    sampled_df["_assigned_model"] = assignments

    logger.info(
        "Multi-model split: %s",
        {m: assignments.count(m) for m in models},
    )

    parts = []
    for model in models:
        subset = sampled_df.loc[sampled_df["_assigned_model"] == model].copy()
        if subset.empty:
            continue
        labeled = await run_llm_check(
            subset.drop(columns=["_assigned_model"]),
            level, backend, model, collection_df, original_texts,
        )
        labeled["llm_model"] = model
        parts.append(labeled)

    result = pd.concat(parts, ignore_index=True)
    return result


def find_optimal_threshold(labeled_df: pd.DataFrame, score_col: str) -> dict:
    """Grid-search for best F1 threshold on a score column."""
    df = labeled_df.loc[labeled_df["llm_label"].isin(["yes", "no"])].copy()
    y_true = (df["llm_label"] == "yes").astype(int)
    logger.info(
        "Searching threshold for %s using %d labeled pairs", score_col, len(df)
    )

    if len(df) == 0 or y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return {
            "f1": 0.0,
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "report": "No positive or negative samples",
        }

    thresholds = np.arange(
        max(0.0, df[score_col].min()),
        min(1.0, df[score_col].max()) + 0.01,
        0.01,
    )
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (df[score_col] >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    preds = (df[score_col] >= best_t).astype(int)
    report = classification_report(
        y_true, preds, target_names=["no", "yes"], zero_division=0
    )
    logger.info(
        "Threshold search complete for %s: best F1 %.3f @ %.3f",
        score_col,
        best_f1,
        best_t,
    )
    return {
        "f1": best_f1,
        "threshold": best_t,
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "report": report,
    }


# ---------------------------------------------------------------------------
# 6. Training data collection & QA
# ---------------------------------------------------------------------------


def save_training_pairs(
    labeled_df: pd.DataFrame,
    agency_year_dir: Path,
    level: str,
    training_data_dir: Path,
    llm_model: str,
):
    """Append LLM-labeled pairs to a central CSV for cross-encoder training.

    Thread-safe via file lock so multiple pipeline instances can write
    concurrently.
    """
    training_data_dir.mkdir(parents=True, exist_ok=True)
    out_path = training_data_dir / "llm_labeled_pairs.csv"
    lock_path = training_data_dir / "llm_labeled_pairs.csv.lock"

    valid = labeled_df.loc[labeled_df["llm_label"].isin(["yes", "no"])].copy()
    if valid.empty:
        return 0

    valid["source_dir"] = str(agency_year_dir)
    valid["level"] = level
    # Prefer per-row llm_model from multi-model labeling; fall back to arg
    if "llm_model" not in valid.columns:
        valid["llm_model"] = llm_model
    valid["timestamp"] = datetime.now().isoformat()

    cols = [
        "response_text", "candidate_text", "dense_score",
        "llm_label", "agency_id", "docket_id", "doc_id",
        "source_dir", "level", "llm_model", "timestamp",
    ]
    out_df = valid[[c for c in cols if c in valid.columns]]

    with filelock.FileLock(lock_path):
        write_header = not out_path.exists()
        out_df.to_csv(out_path, mode="a", header=write_header, index=False)

    logger.info(
        "Saved %d training pairs from %s to %s",
        len(out_df), agency_year_dir.name, out_path,
    )
    return len(out_df)


async def run_qa_check(
    labeled_df: pd.DataFrame,
    level: str,
    backend: str,
    primary_model: str,
    qa_model: str,
    qa_fraction: float,
    collection_df: pd.DataFrame,
    original_texts: Optional[dict],
    training_data_dir: Path,
):
    """Re-label a subset of pairs with a stronger model for agreement analysis."""
    valid = labeled_df.loc[labeled_df["llm_label"].isin(["yes", "no"])].copy()
    if valid.empty or qa_fraction <= 0:
        return

    n_qa = max(1, int(len(valid) * qa_fraction))
    qa_sample = valid.sample(n=min(n_qa, len(valid)), random_state=42).copy()

    logger.info(
        "Running QA check: re-labeling %d pairs with %s",
        len(qa_sample), qa_model,
    )

    # Re-label with the QA model
    qa_labeled = await run_llm_check(
        qa_sample.drop(columns=["llm_label"]),
        level, backend, qa_model, collection_df, original_texts,
    )

    qa_valid = qa_labeled.loc[qa_labeled["llm_label"].isin(["yes", "no"])].copy()
    if qa_valid.empty:
        logger.warning("QA model returned no valid labels")
        return

    # Compare: primary label (from qa_sample) vs QA label (from qa_labeled)
    qa_sample = qa_sample.rename(columns={"llm_label": "primary_label"})
    merged = qa_sample[["doc_id", "response_text", "primary_label"]].merge(
        qa_valid[["doc_id", "response_text", "llm_label"]].rename(
            columns={"llm_label": "qa_label"}
        ),
        on=["doc_id", "response_text"],
        how="inner",
    )

    if merged.empty:
        logger.warning("QA merge produced 0 rows")
        return

    agreement = (merged["primary_label"] == merged["qa_label"]).mean()
    logger.info(
        "QA agreement (%s vs %s): %.1f%% on %d pairs",
        primary_model, qa_model, agreement * 100, len(merged),
    )

    # Save QA results
    training_data_dir.mkdir(parents=True, exist_ok=True)
    qa_path = training_data_dir / "llm_labeled_pairs_gpt5_qa.csv"
    lock_path = training_data_dir / "llm_labeled_pairs_gpt5_qa.csv.lock"

    merged["primary_model"] = primary_model
    merged["qa_model"] = qa_model
    merged["timestamp"] = datetime.now().isoformat()

    with filelock.FileLock(lock_path):
        write_header = not qa_path.exists()
        merged.to_csv(qa_path, mode="a", header=write_header, index=False)

    # Append agreement stats
    report_path = training_data_dir / "agreement_report.json"
    report_entry = {
        "timestamp": datetime.now().isoformat(),
        "primary_model": primary_model,
        "qa_model": qa_model,
        "n_pairs": len(merged),
        "agreement": round(agreement, 4),
        "primary_yes_rate": round((merged["primary_label"] == "yes").mean(), 4),
        "qa_yes_rate": round((merged["qa_label"] == "yes").mean(), 4),
    }
    with open(report_path, "a") as f:
        f.write(json.dumps(report_entry) + "\n")

    logger.info("QA results saved to %s", qa_path)


# ---------------------------------------------------------------------------
# 7. Logging
# ---------------------------------------------------------------------------

_LOG_COLUMNS = [
    "timestamp",
    "directory",
    "level",
    "total_docs",
    "total_pairs",
    "sampled_pairs",
    "labeled_pairs",
    "label_dist_yes",
    "label_dist_no",
    "dense_f1",
    "dense_threshold",
    "dense_precision",
    "dense_recall",
    "match_rate",
    "status",
    "error",
]


def _init_log():
    if not LOG_FILE.exists():
        pd.DataFrame(columns=_LOG_COLUMNS).to_csv(LOG_FILE, index=False)


def log_result(entry: dict):
    entry["timestamp"] = datetime.now().isoformat()
    if "label_dist" in entry:
        dist = entry.pop("label_dist")
        entry["label_dist_yes"] = dist.get("yes", 0)
        entry["label_dist_no"] = dist.get("no", 0)
    row = pd.DataFrame([entry]).reindex(columns=_LOG_COLUMNS)
    row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    logger.info(json.dumps(entry, indent=2, default=str))


# ---------------------------------------------------------------------------
# 7. Output helpers
# ---------------------------------------------------------------------------


def _output_paths(agency_year_dir: Path, level: str):
    resp_path = agency_year_dir / f"public_submission_all_text__{level}_response_matches.csv"
    comment_path = agency_year_dir / f"public_submission_all_text__{level}_comment_labels.csv"
    return resp_path, comment_path


def _outputs_exist(agency_year_dir: Path, level: str) -> bool:
    resp_path, comment_path = _output_paths(agency_year_dir, level)
    return resp_path.exists() and comment_path.exists()


def save_outputs(all_pairs_df: pd.DataFrame, agency_year_dir: Path, level: str):
    """Save response_matches.csv and comment_labels.csv."""
    resp_path, comment_path = _output_paths(agency_year_dir, level)

    # Build response key
    df = all_pairs_df.copy()
    df["response_key"] = (
        df["agency_id"].astype(str)
        + "|"
        + df["docket_id"].astype(str)
        + "|"
        + df["response_text"].fillna("").astype(str).str[:200]
    )

    # --- response_matches.csv ---
    matches = df.loc[df["final_label"] == "yes"]
    if not matches.empty:
        resp_agg = (
            matches.groupby("response_key")
            .agg(
                agency_id=("agency_id", "first"),
                docket_id=("docket_id", "first"),
                response_content=("response_text", "first"),
                matched_doc_ids=(
                    "doc_id",
                    lambda x: ";".join(sorted(set(x.astype(str)))),
                ),
                match_count=("doc_id", "nunique"),
            )
            .reset_index()
        )
    else:
        resp_agg = pd.DataFrame(
            columns=[
                "response_key",
                "agency_id",
                "docket_id",
                "response_content",
                "matched_doc_ids",
                "match_count",
            ]
        )
    resp_agg.to_csv(resp_path, index=False)
    logger.info("Saved %d response rows to %s", len(resp_agg), resp_path)

    # --- comment_labels.csv ---
    comment_agg = (
        df.groupby("doc_id")
        .agg(
            cluster_uid=("doc_id", "first"),
            docket_id=("docket_id", "first"),
            matched=(
                "final_label",
                lambda x: "yes" if (x == "yes").any() else "no",
            ),
            matched_response_keys=(
                "response_key",
                lambda x: ";".join(
                    sorted(
                        set(
                            x.loc[
                                df.loc[x.index, "final_label"] == "yes"
                            ]
                        )
                    )
                )
                if (df.loc[x.index, "final_label"] == "yes").any()
                else "",
            ),
            num_responses_matched=(
                "final_label",
                lambda x: (x == "yes").sum(),
            ),
            best_dense_score=("dense_score", "max"),
            **({
                "best_cross_encoder_score": ("cross_encoder_score", "max"),
            } if "cross_encoder_score" in df.columns else {}),
        )
        .reset_index()
    )
    comment_agg.to_csv(comment_path, index=False)
    logger.info("Saved %d comment rows to %s", len(comment_agg), comment_path)


# ---------------------------------------------------------------------------
# 8. Load original texts for claims-level prompts
# ---------------------------------------------------------------------------


def load_original_texts(agency_year_dir: Path) -> dict:
    """Load original comment texts keyed by cluster_uid.

    Used to provide context in claims-level LLM prompts.
    """
    mapper_path = agency_year_dir / "public_submission_all_text__dedup_mapper.csv"
    all_text_path = agency_year_dir / "public_submission_all_text.csv"

    if not mapper_path.exists() or not all_text_path.exists():
        return {}

    mapper = pd.read_csv(mapper_path, low_memory=False)
    all_text = pd.read_csv(
        all_text_path,
        usecols=["Document ID", "canonical_text"],
        low_memory=False,
    )
    mapper["document_id"] = mapper["document_id"].astype(str)
    all_text["Document ID"] = all_text["Document ID"].astype(str)

    merged = mapper.merge(
        all_text,
        left_on="document_id",
        right_on="Document ID",
        how="inner",
    )
    merged["canonical_text"] = merged["canonical_text"].fillna("")
    merged["text_len"] = merged["canonical_text"].str.len()

    reps = (
        merged.sort_values("text_len", ascending=False)
        .groupby("cluster_uid", as_index=False)
        .head(1)
    )
    return dict(zip(reps["cluster_uid"].astype(str), reps["canonical_text"]))


# ---------------------------------------------------------------------------
# 9. Main pipeline per directory
# ---------------------------------------------------------------------------


def _all_indexes_exist(agency_year_dir: Path, level: str, primary_model: str) -> bool:
    """Check if all three indexes (BM25 + primary dense + distribution dense) exist."""
    index_base = agency_year_dir / ".retriv_indexes"
    dir_name = agency_year_dir.name
    index_name = f"{dir_name}_{level}"
    # BM25
    if not _index_exists(index_base, f"{index_name}_bm25", "sparse"):
        return False
    # Primary dense
    safe_primary = primary_model.replace("/", "_")
    if not _index_exists(index_base, f"{index_name}_{safe_primary}", "dense"):
        return False
    # Distribution dense
    for dist_model in DISTRIBUTION_DENSE_MODELS:
        if dist_model == primary_model:
            continue
        safe_dist = dist_model.replace("/", "_")
        if not _index_exists(index_base, f"{index_name}_{safe_dist}", "dense"):
            return False
    return True


async def process_directory(
    agency_year_dir: Path,
    response_df: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    """Process one agency/year directory with lock file protection."""
    level = args.level
    log_entry = {"directory": str(agency_year_dir), "level": level}

    # Check if work is already done
    if not args.overwrite:
        if args.index_only:
            if _all_indexes_exist(agency_year_dir, level, args.primary_embedding_model):
                logger.info("All indexes exist for %s, skipping.", agency_year_dir.name)
                log_entry["status"] = "skipped_existing_indexes"
                log_result(log_entry)
                return
        elif _outputs_exist(agency_year_dir, level):
            logger.info("Outputs already exist for %s, skipping.", agency_year_dir.name)
            log_entry["status"] = "skipped_existing_outputs"
            log_result(log_entry)
            return

    # Acquire lock file
    processing_flag = agency_year_dir / ".match-processing"
    try:
        fd = os.open(str(processing_flag), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        logger.info("Skipping %s (another process is working on it)", agency_year_dir.name)
        log_entry["status"] = "skipped_locked"
        log_result(log_entry)
        return
    _active_processing_files.add(processing_flag)

    try:
        await _process_directory_inner(agency_year_dir, response_df, args, log_entry)
    except Exception as e:
        logger.error("Failed on %s: %s", agency_year_dir, e, exc_info=True)
        log_entry["status"] = "error"
        log_entry["error"] = str(e)
        log_result(log_entry)
    finally:
        processing_flag.unlink(missing_ok=True)
        _active_processing_files.discard(processing_flag)


async def _process_directory_inner(
    agency_year_dir: Path,
    response_df: Optional[pd.DataFrame],
    args: argparse.Namespace,
    log_entry: dict,
) -> None:
    level = args.level

    # ── Step 1: Load collection data ──
    collection_df = build_collection(agency_year_dir, level)
    if collection_df is None or collection_df.empty:
        logger.warning("No data for %s at level=%s, skipping.", agency_year_dir.name, level)
        log_entry["status"] = "skipped_no_data"
        log_result(log_entry)
        return

    log_entry["total_docs"] = len(collection_df)
    logger.info(
        "Loaded %d %s-level docs for %s",
        len(collection_df),
        level,
        agency_year_dir.name,
    )

    # Build doc_id → docket_id mapping (skip NaN/empty docket IDs)
    doc_to_docket = {}
    for doc_id, docket_id in zip(
        collection_df["id"].astype(str), collection_df["docket_id"].astype(str)
    ):
        if docket_id and docket_id != "nan":
            doc_to_docket[doc_id] = docket_id
    # Build docket → list of doc IDs
    docket_to_docs: dict[str, list[str]] = {}
    for doc_id, docket_id in doc_to_docket.items():
        docket_to_docs.setdefault(docket_id, []).append(doc_id)

    # Build collection for retriv (list of {"id": ..., "text": ...})
    collection_list = collection_df[["id", "text"]].to_dict("records")

    # ── Step 2: Build/load indexes ──
    dr = build_or_load_indexes(
        agency_year_dir,
        level,
        collection_list,
        args.primary_embedding_model,
        args.batch_size,
        args.overwrite,
        skip_distribution=getattr(args, "collect_training_data", False),
    )

    # If --index-only, stop after building indexes.
    if args.index_only:
        logger.info("Index-only mode: finished building indexes for %s", agency_year_dir.name)
        log_entry["status"] = "indexed"
        log_result(log_entry)
        return

    # ── Step 3: Retrieve candidates ──
    # Filter response_df to dockets present in this directory
    dir_dockets = set(docket_to_docs.keys())
    resp_subset = response_df.loc[
        response_df["Docket ID"].astype(str).isin(dir_dockets)
    ].copy()

    if resp_subset.empty:
        logger.warning("No responses for dockets in %s", agency_year_dir.name)
        log_entry["status"] = "skipped_no_responses"
        log_result(log_entry)
        return

    # Determine effective sample size for later pair sampling.
    effective_sample_size = args.llm_sample_size
    if getattr(args, "collect_training_data", False) and getattr(args, "training_samples_per_dir", None):
        effective_sample_size = args.training_samples_per_dir

    # Build response text
    resp_subset["response_text"] = (
        resp_subset.fillna("")
        .apply(
            lambda r: (
                str(r.get("content_of_comment", ""))
                + " "
                + str(r.get("summarized_content_of_comment", ""))
            ),
            axis=1,
        )
        .str.strip()
    )

    # Build text lookup from collection
    doc_id_to_text = dict(zip(collection_df["id"].astype(str), collection_df["text"]))

    # Batch-encode unique response texts for the dense retriever upfront
    # to avoid re-encoding the 8B model per query (massive speedup).
    response_texts = resp_subset["response_text"].tolist()
    unique_texts = list(dict.fromkeys(response_texts))  # preserves order, deduplicates
    logger.info(
        "Batch-encoding %d unique response queries (%d total) for dense retrieval (%s)...",
        len(unique_texts),
        len(response_texts),
        agency_year_dir.name,
    )
    query_bs = args.query_batch_size or args.batch_size
    # Defensive: ensure max_length is capped before query encoding.
    enc_max = getattr(dr.encoder, "max_length", None)
    model_cap = _MODEL_MAX_LENGTH.get(args.primary_embedding_model)
    if model_cap and enc_max and enc_max != model_cap:
        logger.warning(
            "Encoder max_length is %s, overriding to %s before query encoding",
            enc_max, model_cap,
        )
        dr.encoder.max_length = model_cap
        dr.encoder.tokenizer_kwargs["max_length"] = model_cap
    logger.info(
        "Encoder max_length=%s, query_batch_size=%d",
        getattr(dr.encoder, "max_length", "?"), query_bs,
    )
    if args.debug and unique_texts:
        sample = unique_texts[0]
        logger.info(
            "Sample query (%d chars): %s",
            len(sample), sample[:500],
        )
    # Log device info to verify GPU usage.
    # NOTE: Encoder.model is the model NAME (str); the actual AutoModel is
    # stored as Encoder.encoder.
    if hasattr(dr.encoder, "encoder"):
        enc_model = dr.encoder.encoder  # the actual torch model
        try:
            dev = next(enc_model.parameters()).device
            logger.info("Encoder device: %s", dev)
            if str(dev) == "cpu":
                import torch
                if torch.cuda.is_available():
                    logger.info("Moving encoder to CUDA for query encoding")
                    enc_model.to("cuda")
                    dr.encoder.device = "cuda"
        except StopIteration:
            logger.info("Encoder device: unknown (no parameters)")
    unique_embeddings = dr.encoder(unique_texts, batch_size=query_bs, show_progress=True)
    text_to_embedding = {t: unique_embeddings[i] for i, t in enumerate(unique_texts)}

    all_pairs = []
    skipped_no_docs = 0
    skipped_no_results = 0
    for _, resp_row in tqdm(
        resp_subset.iterrows(),
        total=len(resp_subset),
        desc=f"Retrieving ({agency_year_dir.name})",
    ):
        docket_id = str(resp_row["Docket ID"])
        docket_doc_ids = docket_to_docs.get(docket_id, [])
        if not docket_doc_ids:
            skipped_no_docs += 1
            continue

        response_text = str(resp_row["response_text"])
        results = retrieve_for_response(
            dr, response_text, docket_doc_ids,
            k=args.top_k,
            encoded_query=text_to_embedding[response_text],
        )
        if not results:
            skipped_no_results += 1

        for res in results:
            all_pairs.append({
                "agency_id": str(resp_row.get("Agency ID", "")),
                "docket_id": docket_id,
                "response_text": response_text,
                "doc_id": res["doc_id"],
                "candidate_text": doc_id_to_text.get(res["doc_id"], ""),
                "dense_score": res["dense_score"],
            })

    if not all_pairs:
        logger.warning(
            "No retrieval pairs for %s (skipped_no_docs=%d, skipped_no_results=%d, total_responses=%d)",
            agency_year_dir.name, skipped_no_docs, skipped_no_results, len(resp_subset),
        )
        log_entry["status"] = "skipped_no_pairs"
        log_result(log_entry)
        return

    all_pairs_df = pd.DataFrame(all_pairs)
    log_entry["total_pairs"] = len(all_pairs_df)
    logger.info("Retrieved %d candidate pairs for %s", len(all_pairs_df), agency_year_dir.name)

    # ── Step 3.5: Cross-encoder reranking (if enabled) ──
    score_col = "dense_score"
    if getattr(args, "cross_encoder_model", None):
        from cross_encoder_utils import load_cross_encoder, rerank_pairs, load_optimal_threshold

        ce_model = load_cross_encoder(
            args.cross_encoder_model,
            max_length=args.cross_encoder_max_length,
        )
        logger.info(
            "Reranking %d pairs with cross-encoder (%s)",
            len(all_pairs_df), args.cross_encoder_model,
        )
        ce_scores = rerank_pairs(
            ce_model, all_pairs_df,
            batch_size=args.cross_encoder_batch_size,
        )
        all_pairs_df["cross_encoder_score"] = ce_scores
        score_col = "cross_encoder_score"

        # Fast-path: if a fixed threshold is given, skip LLM labeling entirely
        if args.cross_encoder_threshold is not None:
            ce_threshold = args.cross_encoder_threshold
            logger.info(
                "Applying preset cross-encoder threshold %.3f (skipping LLM)",
                ce_threshold,
            )
            all_pairs_df["final_label"] = np.where(
                all_pairs_df["cross_encoder_score"] >= ce_threshold, "yes", "no"
            )
            match_rate = (all_pairs_df["final_label"] == "yes").mean()
            log_entry["cross_encoder_threshold"] = ce_threshold
            log_entry["match_rate"] = round(match_rate, 4)
            log_entry["status"] = "completed_cross_encoder"
            log_result(log_entry)
            save_outputs(all_pairs_df, agency_year_dir, level)
            return

    # ── Step 4: LLM accuracy check ──
    # When collecting training data, optionally use a smaller per-dir budget
    effective_sample_size = args.llm_sample_size
    if getattr(args, "collect_training_data", False) and getattr(args, "training_samples_per_dir", None):
        effective_sample_size = args.training_samples_per_dir
        logger.info(
            "Training data mode: sampling %d pairs (--training-samples-per-dir) instead of %d",
            effective_sample_size, args.llm_sample_size,
        )

    sampled = sample_pairs_for_llm(all_pairs_df, n=effective_sample_size)
    log_entry["sampled_pairs"] = len(sampled)

    # Load original texts for claims-level prompts
    original_texts = None
    if level == "claims":
        original_texts = load_original_texts(agency_year_dir)

    # args.llm_model is now a list of models; use multi-model labeling
    labeled = await run_multi_model_llm_check(
        sampled,
        level,
        args.prompt_backend,
        args.llm_model,  # list of model names
        collection_df,
        original_texts,
    )
    labeled_valid = labeled.loc[labeled["llm_label"].isin(["yes", "no"])]
    log_entry["labeled_pairs"] = len(labeled_valid)
    log_entry["label_dist"] = labeled_valid["llm_label"].value_counts().to_dict()
    if "llm_model" in labeled_valid.columns:
        log_entry["model_dist"] = labeled_valid["llm_model"].value_counts().to_dict()

    # ── Step 4b: Save training data & QA (if enabled) ──
    if getattr(args, "collect_training_data", False):
        training_data_dir = Path(
            args.training_data_dir or (_SCRIPTS_DIR / "training_data")
        )
        # llm_model column is already set per-row by run_multi_model_llm_check
        save_training_pairs(
            labeled, agency_year_dir, level, training_data_dir,
            llm_model=", ".join(args.llm_model),
        )
        if getattr(args, "qa_model", None):
            await run_qa_check(
                labeled, level, args.prompt_backend,
                args.llm_model[0], args.qa_model, args.qa_sample_fraction,
                collection_df, original_texts, training_data_dir,
            )

    # ── Step 5: Find optimal threshold ──
    # Always compute dense_score threshold for comparison
    dense_eval = find_optimal_threshold(labeled, "dense_score")
    log_entry["dense_f1"] = dense_eval["f1"]
    log_entry["dense_threshold"] = dense_eval["threshold"]
    log_entry["dense_precision"] = dense_eval["precision"]
    log_entry["dense_recall"] = dense_eval["recall"]

    logger.info(
        "Dense F1: %.3f @ %.3f\n%s",
        dense_eval["f1"],
        dense_eval["threshold"],
        dense_eval["report"],
    )

    # If cross-encoder is active, also compute threshold on cross-encoder scores
    if score_col == "cross_encoder_score":
        ce_eval = find_optimal_threshold(labeled, "cross_encoder_score")
        log_entry["ce_f1"] = ce_eval["f1"]
        log_entry["ce_threshold"] = ce_eval["threshold"]
        log_entry["ce_precision"] = ce_eval["precision"]
        log_entry["ce_recall"] = ce_eval["recall"]
        logger.info(
            "Cross-encoder F1: %.3f @ %.3f\n%s",
            ce_eval["f1"],
            ce_eval["threshold"],
            ce_eval["report"],
        )
        threshold_eval = ce_eval
    else:
        threshold_eval = dense_eval

    # ── Step 6: Apply threshold and save ──
    all_pairs_df["final_label"] = np.where(
        all_pairs_df[score_col] >= threshold_eval["threshold"], "yes", "no"
    )
    match_rate = (all_pairs_df["final_label"] == "yes").mean()
    log_entry["match_rate"] = round(match_rate, 4)
    log_entry["status"] = "completed"
    log_result(log_entry)

    save_outputs(all_pairs_df, agency_year_dir, level)


# ---------------------------------------------------------------------------
# 10. Entry point
# ---------------------------------------------------------------------------


def iter_agency_year_dirs(base_dir: Path):
    """Discover all agency/year directories."""
    dirs = []
    for agency_dir in sorted(base_dir.iterdir()):
        if not agency_dir.is_dir() or agency_dir.name == "scripts":
            continue
        for year_dir in sorted(agency_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            dirs.append(year_dir)
    return dirs


async def run_pipeline(args: argparse.Namespace):
    _init_log()

    # Only load the response cache when we actually need it for matching.
    response_df = None
    if not args.index_only:
        response_df = load_response_df()

    dirs = iter_agency_year_dirs(BULK_DIR)
    if not dirs:
        logger.warning("No agency/year directories found under %s", BULK_DIR)
        return

    if args.debug:
        # Pick one directory with ~1000 claims for quick testing.
        # When not index-only, also require matching responses.
        response_dockets = (
            set(response_df["Docket ID"].astype(str)) if response_df is not None else None
        )
        best_dir, best_count = None, None
        best_has_indexes = False
        target = 1000
        for d in dirs:
            # Skip directories that already have outputs
            if not args.overwrite and _outputs_exist(d, args.level):
                continue
            claims_path = d / "public_submission_all_text__claims.csv"
            if not claims_path.exists():
                continue
            collection_df = build_collection(d, args.level)
            if collection_df is None or collection_df.empty:
                continue
            # When doing full matching, require responses for this directory.
            if response_dockets is not None:
                dir_dockets = set(
                    collection_df["docket_id"].astype(str).replace("nan", pd.NA).dropna()
                )
                if not dir_dockets & response_dockets:
                    logger.debug("Debug: %s has no matching responses, skipping", d.name)
                    continue
            n = len(collection_df)
            has_indexes = _all_indexes_exist(d, args.level, args.primary_embedding_model)
            # Prefer dirs with existing indexes to avoid rebuilding
            if best_dir is None:
                best_dir, best_count, best_has_indexes = d, n, has_indexes
            elif has_indexes and not best_has_indexes:
                # Always prefer a dir with indexes over one without
                best_dir, best_count, best_has_indexes = d, n, has_indexes
            elif has_indexes == best_has_indexes and abs(n - target) < abs(best_count - target):
                best_dir, best_count, best_has_indexes = d, n, has_indexes
            if has_indexes and 500 <= n <= 2000:
                break
        if best_dir is None:
            logger.error("Debug mode: no directories with data found.")
            return
        dirs = [best_dir]
        logger.info("Debug mode: selected %s (%d docs)", best_dir.name, best_count)

    if args.dir_order == "shuffle":
        random.shuffle(dirs)
    elif args.dir_order == "name":
        dirs.sort(key=lambda p: str(p))

    logger.info(
        "Processing %d directories (level=%s, order=%s, index_only=%s, pid=%d)",
        len(dirs),
        args.level,
        args.dir_order,
        args.index_only,
        os.getpid(),
    )

    for agency_year_dir in tqdm(dirs, desc="Directories"):
        try:
            await process_directory(agency_year_dir, response_df, args)
        except Exception as e:
            logger.error("Failed on %s: %s", agency_year_dir, e, exc_info=True)
            log_result({
                "directory": str(agency_year_dir),
                "level": args.level,
                "status": "error",
                "error": str(e),
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieval-based comment matching pipeline."
    )
    parser.add_argument(
        "--level",
        choices=["claims", "comment"],
        required=True,
        help="Match at claim level or comment level.",
    )
    parser.add_argument(
        "--primary-embedding-model",
        default="nvidia/llama-embed-nemotron-8b",
        help="Primary dense model used for matching/search (default: nvidia/llama-embed-nemotron-8b).",
    )
    parser.add_argument(
        "--prompt-backend",
        choices=["openai", "vllm"],
        default="openai",
        help="LLM backend for accuracy check (default: openai).",
    )
    parser.add_argument(
        "--llm-model",
        nargs="+",
        default=["gpt-5-mini"],
        help="LLM model(s) for accuracy check (default: gpt-5-mini). "
             "Multiple models are split uniformly at random per pair.",
    )
    parser.add_argument(
        "--llm-sample-size",
        type=int,
        default=1000,
        help="Number of pairs to LLM-label per directory (default: 1000).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Candidates per response from each retriever (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding during indexing (default: 32).",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=None,
        help="Batch size for encoding response queries (default: same as --batch-size). "
             "Use a smaller value if response texts are long and cause OOM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild indexes and overwrite existing outputs.",
    )
    parser.add_argument(
        "--dir-order",
        choices=["shuffle", "name"],
        default="shuffle",
        help="Processing order for directories (default: shuffle).",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only build indexes, skip retrieval and matching.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: pick one directory with ~1000 claims and process only that.",
    )

    # ── Training data collection flags ──
    parser.add_argument(
        "--collect-training-data",
        action="store_true",
        help="Save all LLM-labeled pairs to a central CSV for cross-encoder training.",
    )
    parser.add_argument(
        "--training-samples-per-dir",
        type=int,
        default=None,
        help="When --collect-training-data is set, limit LLM-labeled pairs per directory "
             "to this value (overrides --llm-sample-size) so we spread budget across more "
             "agencies. Default: use --llm-sample-size.",
    )
    parser.add_argument(
        "--training-data-dir",
        type=str,
        default=None,
        help="Directory for centralized training data (default: scripts/training_data/).",
    )
    parser.add_argument(
        "--qa-model",
        type=str,
        default=None,
        help="Secondary LLM model for quality assurance agreement checks (e.g. gpt-5). "
             "When set, a random subset of pairs is re-labeled with this model.",
    )
    parser.add_argument(
        "--qa-sample-fraction",
        type=float,
        default=0.1,
        help="Fraction of labeled pairs to re-label with --qa-model (default: 0.1).",
    )

    # ── Cross-encoder reranking flags ──
    parser.add_argument(
        "--cross-encoder-model",
        type=str,
        default=None,
        help="Path to trained cross-encoder model. When set, reranks bi-encoder "
             "candidates and uses cross-encoder scores for thresholding.",
    )
    parser.add_argument(
        "--cross-encoder-threshold",
        type=float,
        default=None,
        help="When set with --cross-encoder-model, skip LLM labeling and apply "
             "this threshold directly (production mode, no LLM cost).",
    )
    parser.add_argument(
        "--cross-encoder-batch-size",
        type=int,
        default=64,
        help="Batch size for cross-encoder scoring (default: 64).",
    )
    parser.add_argument(
        "--cross-encoder-max-length",
        type=int,
        default=4096,
        help="Max sequence length for cross-encoder (default: 4096).",
    )

    args = parser.parse_args()

    # Register signal handlers for cleanup
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _signal_handler)

    asyncio.run(run_pipeline(args))
