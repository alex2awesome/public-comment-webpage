"""
Usage:
    python data/bulk_downloads/scripts/comments_extract_claims.py --mode offline --batch-size 5120
"""
from __future__ import annotations

import argparse
import ast
import atexit
import json
import logging
import os
import random
import signal
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

PROMPT_TEMPLATE = (
    "Extract all distinct claims and substantive arguments from the following public comment to a "
    "government agency.\n\n"
    "Rules:\n"
    "- Return ONLY a JSON array of strings: [\"claim1\", \"claim2\", ...]\n"
    "- If the comment contains no substantive claims, return exactly: []\n"
    "- Do not include any text before or after the JSON array\n"
    "- Do not include booleans, objects, or any other values\n\n"
    "<comment>{comment}</comment>\n\n"
    "Your response:\n"
)

PROMPT_FIX_TEMPLATE = (
    "This answer contains a poorly formatted list of strings that I cannot parse with json. "
    "Please copy the list and fix it so I can parse it. Just copy and fix the list. "
    "Do not add any extra text. I will be trying to parse this. "
    "<list>{claims}</list>. Your response:"
)

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:8002/v1"
DEFAULT_API_KEY = "EMPTY"

# Track .processing files created by this process so we can clean them up on exit.
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


def iter_input_files(base_dir: Path) -> Iterable[Path]:
    return base_dir.glob("*/*/public_submission_all_text.csv")


def try_extract_json_list(raw: str) -> list | None:
    # Try direct parse first
    try:
        result = json.loads(raw.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find the first complete JSON array via bracket matching
    start = raw.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(raw)):
        c = raw[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\" and in_string:
            escape_next = True
            continue
        if c == "\"" and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                candidate = raw[start : i + 1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    pass
    return None


def _parse_claims(raw: str) -> Tuple[Optional[List[str]], bool]:
    if raw is None:
        return None, False
    raw = raw.strip()
    if not raw:
        return None, False
    extracted = try_extract_json_list(raw)
    if extracted is None:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None, False
    else:
        parsed = extracted
    if not isinstance(parsed, list):
        return None, False
    if not all(isinstance(item, str) for item in parsed):
        return None, False
    return parsed, True


def _gpu_name() -> str:
    try:
        import torch
    except ImportError:
        return ""
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_name(0) or ""


def _select_model(default_model: str, override: Optional[str]) -> str:
    if override:
        return override
    name_upper = _gpu_name().upper()
    if "B200" in name_upper:
        return "meta-llama/Llama-3.3-70B-Instruct"
    if "H200" in name_upper:
        return "nvidia/Llama-3.1-70B-Instruct-FP8"
    return default_model


def _batch(iterable: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(iterable), size):
        yield iterable[idx : idx + size]


def _hard_clamp_prompt(prompt: str, *, tokenizer, max_input_tokens: int) -> str:
    if tokenizer is None:
        return prompt
    try:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
    except Exception:
        return prompt
    if len(tokens) <= max_input_tokens:
        return prompt
    logging.warning(
        "Prompt exceeded token budget (%d > %d); hard-clamping.",
        len(tokens),
        max_input_tokens,
    )
    return tokenizer.decode(tokens[:max_input_tokens], skip_special_tokens=True)


def _fit_comment_to_budget(
    comment: str, *, tokenizer, max_input_tokens: int
) -> str:
    # Estimate prompt template overhead in chars (~400 chars → ~100 tokens).
    # Use ~2 chars/token as a pessimistic ratio for the comment portion.
    template_overhead_chars = len(PROMPT_TEMPLATE) - len("{comment}")
    char_budget = max(500, (max_input_tokens * 2) - template_overhead_chars)
    if tokenizer is None:
        logging.debug("No tokenizer; truncating comment to %d chars", char_budget)
        return (comment or "")[:char_budget]
    comment = comment or ""
    try:
        comment_tokens = tokenizer.encode(comment, add_special_tokens=True)
    except Exception:
        return comment[:char_budget]
    if not comment_tokens:
        return ""

    def prompt_len_for(k: int) -> int:
        candidate = tokenizer.decode(comment_tokens[:k], skip_special_tokens=True)
        prompt = PROMPT_TEMPLATE.format(comment=candidate)
        return len(tokenizer.encode(prompt, add_special_tokens=True))

    # Fast path if already fits
    if prompt_len_for(len(comment_tokens)) <= max_input_tokens:
        return comment

    low, high = 0, len(comment_tokens)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if prompt_len_for(mid) <= max_input_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    if best < len(comment_tokens) and logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(
            "Truncated comment from %d tokens to %d tokens to fit budget.",
            len(comment_tokens),
            best,
        )
    return tokenizer.decode(comment_tokens[:best], skip_special_tokens=True)


def _fit_fix_to_budget(
    claims: str, *, tokenizer, max_input_tokens: int
) -> str:
    # Fix prompts can be large; budget them similarly to comment prompts.
    template_overhead_chars = len(PROMPT_FIX_TEMPLATE) - len("{claims}")
    char_budget = max(200, (max_input_tokens * 2) - template_overhead_chars)
    if tokenizer is None:
        logging.debug("No tokenizer; truncating fix prompt claims to %d chars", char_budget)
        return (claims or "")[:char_budget]
    claims = claims or ""
    try:
        claims_tokens = tokenizer.encode(claims, add_special_tokens=True)
    except Exception:
        return claims[:char_budget]
    if not claims_tokens:
        return ""

    def prompt_len_for(k: int) -> int:
        candidate = tokenizer.decode(claims_tokens[:k], skip_special_tokens=True)
        prompt = PROMPT_FIX_TEMPLATE.format(claims=candidate)
        return len(tokenizer.encode(prompt, add_special_tokens=True))

    if prompt_len_for(len(claims_tokens)) <= max_input_tokens:
        return claims

    low, high = 0, len(claims_tokens)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if prompt_len_for(mid) <= max_input_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    if best < len(claims_tokens) and logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(
            "Truncated fix claims from %d tokens to %d tokens to fit budget.",
            len(claims_tokens),
            best,
        )
    return tokenizer.decode(claims_tokens[:best], skip_special_tokens=True)


def _prepare_prompts(
    texts: List[str], *, tokenizer, max_input_tokens: int
) -> List[str]:
    prompts = []
    for text in texts:
        truncated = _fit_comment_to_budget(
            text or "",
            tokenizer=tokenizer,
            max_input_tokens=max_input_tokens,
        )
        prompt = PROMPT_TEMPLATE.format(comment=truncated)
        prompts.append(
            _hard_clamp_prompt(
                prompt,
                tokenizer=tokenizer,
                max_input_tokens=max_input_tokens,
            )
        )
    return prompts


def _prepare_fix_prompts(
    texts: List[str], *, tokenizer, max_input_tokens: int
) -> List[str]:
    prompts = []
    for text in texts:
        truncated = _fit_fix_to_budget(
            text or "",
            tokenizer=tokenizer,
            max_input_tokens=max_input_tokens,
        )
        prompt = PROMPT_FIX_TEMPLATE.format(claims=truncated)
        prompts.append(
            _hard_clamp_prompt(
                prompt,
                tokenizer=tokenizer,
                max_input_tokens=max_input_tokens,
            )
        )
    return prompts


def process_file(
    csv_path: Path,
    *,
    mode: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    max_input_tokens: int,
    temperature: float,
    generate_batch,
    tokenizer,
    overwrite: bool,
) -> None:
    logging.info("Processing %s", csv_path)
    mapper_path = csv_path.parent / "public_submission_all_text__dedup_mapper.csv"
    if not mapper_path.exists():
        logging.warning("Missing mapper %s; skipping", mapper_path)
        return
    output_path = csv_path.parent / "public_submission_all_text__claims.csv"
    if output_path.exists() and not overwrite:
        # Check if existing claims cover all document IDs in the all_text file.
        # Claims stores one row per cluster, so we expand cluster_uids back to
        # doc IDs via the mapper, and treat any doc not in the mapper as a
        # singleton (covered if its doc ID appears directly in claims).
        try:
            all_text_df = pd.read_csv(csv_path, usecols=["Document ID", "canonical_text"])
            all_text_ids = set(
                all_text_df.dropna(subset=["Document ID", "canonical_text"])["Document ID"]
                .astype(str)
                .unique()
            )
            claims_uids = set(
                pd.read_csv(output_path, usecols=["cluster_uid"])["cluster_uid"]
                .dropna()
                .astype(str)
                .unique()
            )
            mapper_df = pd.read_csv(mapper_path, usecols=["document_id", "cluster_uid"])
            mapper_df["document_id"] = mapper_df["document_id"].astype(str)
            mapper_df["cluster_uid"] = mapper_df["cluster_uid"].astype(str)
            # Doc IDs covered = mapper docs whose cluster is done + singleton docs whose ID is a done cluster_uid
            covered_via_mapper = set(
                mapper_df.loc[mapper_df["cluster_uid"].isin(claims_uids), "document_id"]
            )
            # Docs not in mapper are singletons; covered if their doc ID is in claims as a cluster_uid
            unmapped_ids = all_text_ids - set(mapper_df["document_id"])
            covered_singletons = unmapped_ids & claims_uids
            covered_ids = covered_via_mapper | covered_singletons
            missing = all_text_ids - covered_ids
            if not missing:
                logging.info("Skipping %s (claims already cover all %d doc IDs)", csv_path, len(all_text_ids))
                return
            logging.info(
                "Resuming %s: claims cover %d/%d doc IDs (%d missing)",
                csv_path, len(covered_ids), len(all_text_ids), len(missing),
            )
        except Exception as exc:
            logging.warning("Could not verify claims completeness for %s (%s); re-processing", csv_path, exc)

    # Atomically claim this folder so other processes skip it.
    processing_flag = csv_path.parent / ".processing"
    try:
        fd = os.open(str(processing_flag), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        logging.info("Skipping %s (another process is working on it)", csv_path)
        return
    _active_processing_files.add(processing_flag)

    try:
        _process_file_inner(
            csv_path,
            mapper_path=mapper_path,
            output_path=output_path,
            mode=mode,
            model=model,
            batch_size=batch_size,
            max_tokens=max_tokens,
            max_input_tokens=max_input_tokens,
            temperature=temperature,
            generate_batch=generate_batch,
            tokenizer=tokenizer,
            overwrite=overwrite,
        )
    finally:
        processing_flag.unlink(missing_ok=True)
        _active_processing_files.discard(processing_flag)


def _process_file_inner(
    csv_path: Path,
    *,
    mapper_path: Path,
    output_path: Path,
    mode: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    max_input_tokens: int,
    temperature: float,
    generate_batch,
    tokenizer,
    overwrite: bool = False,
) -> None:
    df = pd.read_csv(csv_path, low_memory=False)
    mapper = pd.read_csv(mapper_path, low_memory=False)

    if "Document ID" not in df.columns or "canonical_text" not in df.columns:
        raise ValueError(f"Missing required columns in {csv_path}")
    if "document_id" not in mapper.columns or "cluster_uid" not in mapper.columns:
        raise ValueError(f"Missing required columns in {mapper_path}")

    df = df.dropna(subset=["Document ID", "canonical_text"]).copy()
    df["Document ID"] = df["Document ID"].astype(str)
    mapper["document_id"] = mapper["document_id"].astype(str)

    merged = df.merge(mapper, left_on="Document ID", right_on="document_id", how="left")
    if merged.empty:
        logging.warning("No merged rows for %s; skipping", csv_path)
        return

    # Docs not in the mapper are treated as singleton clusters.
    unmapped = merged["cluster_uid"].isna()
    if unmapped.any():
        merged.loc[unmapped, "document_id"] = merged.loc[unmapped, "Document ID"]
        merged.loc[unmapped, "cluster_uid"] = merged.loc[unmapped, "Document ID"]
        # Fill agency_id and docket_id from the all_text CSV columns.
        if "Agency ID" in merged.columns:
            merged.loc[unmapped, "agency_id"] = merged.loc[unmapped, "Agency ID"]
        if "Docket ID" in merged.columns:
            merged.loc[unmapped, "docket_id"] = merged.loc[unmapped, "Docket ID"]
        logging.info("%d docs not in mapper; treating as singleton clusters", unmapped.sum())

    merged["canonical_text"] = merged["canonical_text"].fillna("")
    merged["text_len"] = merged["canonical_text"].str.len()

    reps = merged.sort_values("text_len", ascending=False).groupby("cluster_uid", as_index=False).head(1)

    # Resume support: load existing claims and only process new cluster_uids.
    existing_claims: Optional[pd.DataFrame] = None
    if output_path.exists() and not overwrite:
        try:
            existing_claims = pd.read_csv(output_path, low_memory=False)
            done_uids = set(existing_claims["cluster_uid"].dropna().astype(str).unique())
            reps = reps[~reps["cluster_uid"].astype(str).isin(done_uids)].copy()
            logging.info(
                "Resuming %s: %d clusters already done, %d remaining",
                csv_path, len(done_uids), len(reps),
            )
            if reps.empty:
                logging.info("All clusters already processed for %s", csv_path)
                return
        except Exception as exc:
            logging.warning("Could not load existing claims for resume (%s); starting fresh", exc)
            existing_claims = None

    reps = reps.reset_index(drop=True)
    texts = reps["canonical_text"].tolist()
    prompts = _prepare_prompts(texts, tokenizer=tokenizer, max_input_tokens=max_input_tokens)

    raw_outputs: List[str] = []
    for batch in tqdm(list(_batch(prompts, batch_size)), desc=f"{csv_path.parent.name} claims"):
        batch_outputs = generate_batch(batch)
        raw_outputs.extend(batch_outputs)

    parsed_outputs: List[Optional[List[str]]] = []
    parse_ok: List[bool] = []
    for raw in raw_outputs:
        parsed, ok = _parse_claims(raw)
        parsed_outputs.append(parsed)
        parse_ok.append(ok)

    fix_indices = [i for i, ok in enumerate(parse_ok) if not ok]
    fix_raw_outputs: List[str] = [""] * len(raw_outputs)
    fix_parsed_outputs: List[Optional[List[str]]] = [None] * len(raw_outputs)
    fix_ok: List[bool] = [False] * len(raw_outputs)

    if fix_indices:
        fix_prompts = _prepare_fix_prompts(
            [raw_outputs[i] for i in fix_indices],
            tokenizer=tokenizer,
            max_input_tokens=max_input_tokens,
        )
        fix_results: List[str] = []
        for batch in tqdm(list(_batch(fix_prompts, batch_size)), desc=f"{csv_path.parent.name} fixes"):
            batch_outputs = generate_batch(batch)
            fix_results.extend(batch_outputs)

        for idx, raw_fix in zip(fix_indices, fix_results):
            fix_raw_outputs[idx] = raw_fix
            parsed, ok = _parse_claims(raw_fix)
            fix_parsed_outputs[idx] = parsed
            fix_ok[idx] = ok

    new_output = pd.DataFrame(
        {
            "agency_id": reps.get("agency_id"),
            "docket_id": reps.get("docket_id"),
            "cluster_id": reps.get("cluster_id"),
            "cluster_uid": reps.get("cluster_uid"),
            "document_id": reps.get("document_id"),
            "model": model,
            "mode": mode,
            "input_chars": reps["canonical_text"].str.len(),
            "claims_raw": raw_outputs,
            "claims_parsed_json": [json.dumps(p, ensure_ascii=False) if p is not None else "" for p in parsed_outputs],
            "parse_ok": parse_ok,
            "claims_fix_raw": fix_raw_outputs,
            "claims_fix_parsed_json": [
                json.dumps(p, ensure_ascii=False) if p is not None else "" for p in fix_parsed_outputs
            ],
            "fix_parse_ok": fix_ok,
        }
    )

    # Merge with existing claims if resuming.
    if existing_claims is not None and not existing_claims.empty:
        output = pd.concat([existing_claims, new_output], ignore_index=True)
        logging.info("Merged %d existing + %d new = %d total rows", len(existing_claims), len(new_output), len(output))
    else:
        output = new_output

    output.to_csv(output_path, index=False)
    logging.info("Wrote %s (%d rows)", output_path, len(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--safety-margin", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing claims files.",
    )
    parser.add_argument(
        "--vllm-progress",
        action="store_true",
        help="Show vLLM internal progress bars.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model = _select_model(DEFAULT_MODEL, args.model)
    max_input_tokens = args.max_model_len - args.max_tokens - args.safety_margin
    if max_input_tokens <= 0:
        raise ValueError("max-model-len must exceed max-tokens + safety-margin")
    if args.max_tokens <= 0:
        logging.warning(
            "max-tokens is %d; zero output tokens leaves no headroom for off-by-one.",
            args.max_tokens,
        )
    if args.mode == "offline":
        if not args.vllm_progress:
            os.environ.setdefault("VLLM_DISABLE_PROGRESS_BAR", "1")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for offline mode") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected for offline mode")

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vllm is not installed for offline mode") from exc

        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
        llm_kwargs = {"max_model_len": args.max_model_len}
        if any(tag in _gpu_name().upper() for tag in ("B200", "H200")):
            llm_kwargs.update(
                {
                    "dtype": "auto",
                    "gpu_memory_utilization": 0.95,
                }
            )
            logging.info("Using B200/H200 vLLM settings: %s", llm_kwargs)
        llm = LLM(model=model, **llm_kwargs)
        try:
            tokenizer = llm.get_tokenizer()
        except Exception as exc:
            logging.warning(
                "Could not access tokenizer (%s); falling back to char truncation.",
                exc,
            )
            tokenizer = None

        if tokenizer is not None:
            logging.info("Tokenizer loaded successfully: %s", type(tokenizer).__name__)
        else:
            logging.warning(
                "Running WITHOUT tokenizer – using char-based truncation. "
                "Token budget violations may occur for long inputs."
            )

        def generate_batch(batch: List[str]) -> List[str]:
            outputs = llm.generate(batch, sampling_params)
            results = []
            for out in outputs:
                if not out.outputs:
                    results.append("")
                    continue
                results.append(out.outputs[0].text or "")
            return results
    else:
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for online mode token budgeting") from exc
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        def _call_one(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            return getattr(response.choices[0].message, "content", "") or ""

        def generate_batch(batch: List[str]) -> List[str]:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                return list(pool.map(_call_one, batch))

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _signal_handler)

    base_dir = Path("data/bulk_downloads")
    files = list(iter_input_files(base_dir))
    if not files:
        logging.warning("No input files found under %s", base_dir)
        return

    random.shuffle(files)
    logging.info("Processing %d file(s) in random order (pid=%d)", len(files), os.getpid())

    for csv_path in tqdm(files, desc="Files"):
        process_file(
            csv_path,
            mode=args.mode,
            model=model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            max_input_tokens=max_input_tokens,
            temperature=args.temperature,
            generate_batch=generate_batch,
            tokenizer=tokenizer,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
