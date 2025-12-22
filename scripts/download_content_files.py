#!/usr/bin/env python3
"""Download the content payloads referenced by the bulk CSV exports."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def find_repo_root(current: Path) -> Path:
    for candidate in [current.parent] + list(current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current.parent


PROJECT_ROOT = find_repo_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from ai_corpus.utils import pdf_parsing
except ImportError:  # pragma: no cover - optional dependency
    pdf_parsing = None

PDF_PARSER_CHOICES = ("auto", "pdfminer", "fitz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download files listed under the 'Content Files' column for each CSV row "
            "contained inside a bulk download directory."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "bulk_downloads",
        help="Directory that contains agency subdirectories with CSV exports.",
    )
    parser.add_argument(
        "--output-dir-name",
        default="downloaded_content",
        help="Name of the subdirectory (per agency folder) to store downloaded files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of rows processed per CSV (useful for tests).",
    )
    parser.set_defaults(extract_pdf_text=True)
    parser.add_argument(
        "--skip-pdf-text",
        action="store_false",
        dest="extract_pdf_text",
        help="Disable PDF text extraction after downloads complete.",
    )
    parser.add_argument(
        "--skip-pdf-parsing",
        action="store_false",
        dest="extract_pdf_text",
        help="Alias for --skip-pdf-text; disables PDF extraction.",
    )
    parser.add_argument(
        "--skip-pdf-download",
        action="store_true",
        help="Skip downloading PDF files entirely (also skips parsing).",
    )
    parser.add_argument(
        "--pdf-parser",
        choices=PDF_PARSER_CHOICES,
        default="auto",
        help="PDF extraction backend to use (requires ai_corpus.utils.pdf_parsing).",
    )
    parser.add_argument(
        "--pdf-parse-timeout",
        type=float,
        default=None,
        help="Maximum seconds to spend parsing a single PDF before aborting that extraction.",
    )
    parser.add_argument(
        "--pdf-parse-page-timeout",
        type=float,
        default=None,
        help="Maximum seconds to spend parsing each individual PDF page.",
    )
    parser.add_argument(
        "--pdf-progress-threshold",
        type=int,
        default=20,
        help="Show a per-PDF progress bar when the document has at least this many pages.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, only download a handful of files per CSV for quick debugging.",
    )
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=5,
        help="Maximum number of files to download per CSV when --debug is enabled.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of download retries per file before failing.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout (seconds) for each GET request.",
    )
    parser.add_argument(
        "--sleep-base",
        type=float,
        default=1.5,
        help="Base delay (seconds) for exponential backoff during retries.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Continue processing even if a download ultimately fails (errors are logged).",
    )
    parser.add_argument(
        "--skip-pdf-when-html",
        action="store_true",
        help="Do not extract PDF text for rows that already contain an HTML content file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit detailed per-row logs to diagnose stalls.",
    )
    parser.add_argument(
        "--max-download-seconds",
        type=float,
        default=None,
        help="Abort a single file download if it takes longer than this many seconds (best-effort).",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        dest="max_download_seconds",
        help="Alias for --max-download-seconds.",
    )
    parser.add_argument(
        "--cleanup-ocr-pdfs",
        action="store_true",
        help=(
            "Delete any content.pdf files under --base-dir that already have matching "
            "content.pdf.txt payloads before downloading."
        ),
    )
    return parser.parse_args()


def sanitize(value: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or fallback


def split_content_files(cell_value: Optional[str]) -> List[str]:
    if not cell_value:
        return []
    # The CSV already quotes the field, so a simple split works here.
    return [item.strip() for item in cell_value.split(",") if item.strip()]


def iter_csv_files(base_dir: Path) -> Iterable[Path]:
    for agency_dir in sorted(base_dir.iterdir()):
        if not agency_dir.is_dir():
            continue
        yield from sorted(agency_dir.glob("*.csv"))


def delete_ocrd_content_pdfs(base_dir: Path, *, target_name: str = "content.pdf") -> int:
    """
    Remove ``target_name`` PDFs when a ``.txt`` OCR payload already exists.

    Returns the number of deleted files so callers can report how much space was freed.
    """

    deleted = 0
    if not base_dir.exists():
        logging.warning("Base directory %s does not exist; nothing to clean.", base_dir)
        return deleted

    for pdf_path in sorted(base_dir.rglob(target_name)):
        text_path = pdf_path.with_suffix(pdf_path.suffix + ".txt")
        if not text_path.exists():
            continue
        try:
            pdf_path.unlink()
            deleted += 1
            logging.debug("Deleted %s (text already exists).", pdf_path)
        except FileNotFoundError:
            continue
        except OSError as exc:  # pragma: no cover - best effort logging
            logging.warning("Failed to delete %s: %s", pdf_path, exc)
    return deleted


def looks_like_html(url_or_name: str) -> bool:
    suffix = Path(url_or_name.split("?")[0]).suffix.lower()
    return suffix in {".html", ".htm"}


def log_verbose(args: argparse.Namespace, message: str, *fmt: object) -> None:
    if args.verbose:
        logging.info("[verbose] " + message, *fmt)


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def write_metadata(per_row_dir: Path, row: Dict[str, str]) -> None:
    per_row_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = per_row_dir / "metadata.json"
    if metadata_path.exists():
        return
    sanitized_row = {str(key): value for key, value in row.items() if key is not None}
    metadata_path.write_text(
        json.dumps(sanitized_row, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def maybe_extract_pdf_text(
    dest_path: Path,
    *,
    parser_choice: str,
    parse_timeout: Optional[float],
    per_page_timeout: Optional[float],
    progress_threshold: Optional[int],
) -> None:
    if dest_path.suffix.lower() != ".pdf":
        return
    if pdf_parsing is None:
        logging.warning(
            "Skipping PDF text extraction for %s: ai_corpus.utils.pdf_parsing unavailable",
            dest_path,
        )
        return

    text_path = dest_path.with_suffix(dest_path.suffix + ".txt")
    stats_path = dest_path.with_suffix(dest_path.suffix + ".stats.json")
    if text_path.exists() and stats_path.exists():
        logging.debug("PDF text already extracted for %s", dest_path)
        return

    page_count: Optional[int] = None
    progress_bar = None
    progress_callback = None
    if tqdm is not None and progress_threshold and progress_threshold > 0:
        try:
            page_count = pdf_parsing.get_pdf_page_count(str(dest_path))
        except Exception:
            page_count = None
        show_progress = page_count is not None and page_count >= progress_threshold
        if show_progress:
            progress_bar = tqdm(
                total=page_count,
                desc=f"Parse {dest_path.name}",
                unit="page",
                colour="cyan",
                leave=False,
                position=2,
            )
            progress_state = {"stage": None}

            def _progress_callback(event: Dict[str, object]) -> None:
                if progress_bar is None:
                    return
                event_type = event.get("type")
                stage = event.get("stage")
                if event_type == "stage":
                    progress_state["stage"] = stage
                    total = event.get("total")
                    try:
                        progress_bar.reset(total=total)
                    except Exception:
                        if total:
                            progress_bar.total = total
                    if stage:
                        progress_bar.set_description(f"Parse {dest_path.name} [{stage}]")
                    progress_bar.n = 0
                    progress_bar.refresh()
                elif event_type == "page":
                    total = event.get("total")
                    page_number = event.get("page")
                    if total and progress_bar.total != total:
                        try:
                            progress_bar.reset(total=total)
                        except Exception:
                            progress_bar.total = total
                    if isinstance(page_number, int) and page_number >= 0:
                        if page_number > progress_bar.n:
                            progress_bar.n = page_number
                            progress_bar.refresh()

            progress_callback = _progress_callback

    try:
        deadline = None
        if parse_timeout:
            deadline = time.perf_counter() + parse_timeout
        pages, stats = pdf_parsing.extract_pages(
            str(dest_path),
            parser=parser_choice,
            per_page_timeout=per_page_timeout,
            parse_deadline=deadline,
            progress_callback=progress_callback,
            total_pages=page_count,
        )
    except TimeoutError as exc:
        logging.warning("PDF parsing timeout for %s: %s", dest_path, exc)
        return
    except Exception as exc:  # pragma: no cover - best effort
        logging.warning("Failed to parse PDF %s: %s", dest_path, exc)
        return
    finally:
        if progress_bar is not None:
            progress_bar.close()

    joined_pages = []
    for idx, page_text in enumerate(pages, start=1):
        joined_pages.append(f"<<PAGE {idx}>>\n{page_text.strip()}\n")
    text_payload = "\n".join(joined_pages)

    text_path.write_text(text_payload, encoding="utf-8")
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def download_with_retries(
    session: requests.Session,
    url: str,
    dest_path: Path,
    *,
    retries: int,
    timeout: float,
    max_download_seconds: Optional[float],
    sleep_base: float,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        logging.debug("Skipping existing file %s", dest_path)
        return

    for attempt in range(1, retries + 1):
        download_started = time.perf_counter()
        try:
            logging.debug("GET %s (attempt %s/%s)", url, attempt, retries)
            with session.get(url, timeout=timeout, stream=True) as response:
                response.raise_for_status()
                with dest_path.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            fh.write(chunk)
                        if (
                            max_download_seconds is not None
                            and time.perf_counter() - download_started > max_download_seconds
                        ):
                            raise TimeoutError(
                                f"Download exceeded {max_download_seconds} seconds"
                            )
            return
        except Exception as exc:  # noqa: PERF203 - fine for retry loop
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}") from exc
            sleep_time = min(sleep_base * (2 ** (attempt - 1)), 60.0)
            logging.warning(
                "Error downloading %s (attempt %s/%s): %s. Retrying in %.1fs",
                url,
                attempt,
                retries,
                exc,
                sleep_time,
            )
            time.sleep(sleep_time)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.base_dir.exists():
        logging.error("Base directory %s does not exist", args.base_dir)
        return 1

    if args.cleanup_ocr_pdfs:
        removed = delete_ocrd_content_pdfs(args.base_dir)
        logging.info(
            "Deleted %s existing content.pdf files that already had OCR text.",
            removed,
        )

    csv_paths = list(iter_csv_files(args.base_dir))
    if not csv_paths:
        logging.warning("No CSV files found under %s", args.base_dir)
        return 0

    logging.info("Found %s CSV files under %s", len(csv_paths), args.base_dir)

    if tqdm is None:
        logging.info("Install tqdm for progress bars (pip install tqdm).")
        csv_iterator = csv_paths
        colour_cycle = None
    else:
        csv_iterator = tqdm(
            csv_paths,
            desc="CSV files",
            unit="csv",
            colour="cyan",
            leave=True,
        )
        colour_cycle = cycle(["magenta", "yellow", "green", "blue", "white"])

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "regulations-downloader/0.1 (+mailto:spangher@usc.edu)",
            "Accept": "*/*",
        }
    )
    processed_files = 0
    processed_rows = 0
    downloaded_files = 0
    failed_downloads: List[str] = []

    for index, csv_path in enumerate(csv_iterator, start=1):
        agency_dir = csv_path.parent
        csv_name = csv_path.stem
        logging.info("Processing CSV %s/%s: %s", index, len(csv_paths), csv_path)
        per_csv_downloads = 0
        per_csv_rows = 0
        row_bar = None
        row_total = None
        if tqdm is not None:
            try:
                row_total = count_csv_rows(csv_path)
                if args.max_rows:
                    row_total = min(row_total, args.max_rows)
            except Exception as exc:  # pragma: no cover - file read issues
                logging.debug("Could not count rows for %s: %s", csv_path, exc)
            row_colour = next(colour_cycle) if colour_cycle else None
            row_bar = tqdm(
                total=row_total,
                desc=f"{agency_dir.name}/{csv_name}",
                unit="row",
                colour=row_colour,
                leave=False,
                position=1,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader, start=1):
                if args.max_rows and idx > args.max_rows:
                    break
                if args.debug and per_csv_downloads >= args.debug_limit:
                    logging.debug(
                        "Debug limit reached for %s (limit=%s)",
                        csv_path,
                        args.debug_limit,
                    )
                    break
                content_urls = split_content_files(row.get("Content Files"))
                attachment_urls = split_content_files(row.get("Attachment Files"))
                combined_urls = (
                    [(url, "content") for url in content_urls]
                    + [(url, "attachment") for url in attachment_urls]
                )
                if not combined_urls:
                    if row_bar is not None:
                        row_bar.update(1)
                    continue
                row_has_html = any(looks_like_html(url) for url, _ in combined_urls)
                doc_id = row.get("Document ID") or row.get("DocumentId") or ""
                doc_dir_name = sanitize(doc_id, fallback=f"row_{idx:05d}")
                per_row_dir = (
                    agency_dir
                    / args.output_dir_name
                    / csv_name
                    / doc_dir_name
                )
                log_verbose(
                    args,
                    "Row %s doc=%s urls=%s has_html=%s",
                    idx,
                    doc_id or doc_dir_name,
                    len(combined_urls),
                    row_has_html,
                )
                write_metadata(per_row_dir, row)
                used_names: Dict[str, int] = {}
                for url_idx, (url, url_kind) in enumerate(combined_urls, start=1):
                    suffix = Path(url.split("?")[0]).name or f"file_{url_idx}"
                    base_name = sanitize(suffix, fallback=f"{url_kind}_{url_idx}")
                    counter = used_names.get(base_name, 0)
                    if counter:
                        file_name = f"{base_name}_{counter+1}"
                    else:
                        file_name = base_name
                    used_names[base_name] = counter + 1
                    dest_path = per_row_dir / file_name
                    is_pdf = dest_path.suffix.lower() == ".pdf"
                    if args.skip_pdf_download and is_pdf:
                        log_verbose(
                            args,
                            "  skipping PDF download for %s (flag enabled)",
                            dest_path,
                        )
                        continue
                    if is_pdf and dest_path.name == "content.pdf":
                        text_path = dest_path.with_suffix(dest_path.suffix + ".txt")
                        if text_path.exists():
                            log_verbose(
                                args,
                                "  skipping download for %s (text payload present)",
                                dest_path,
                            )
                            continue
                    log_verbose(
                        args,
                        "  fetching url #%s (%s) -> %s",
                        url_idx,
                        suffix,
                        dest_path,
                    )
                    try:
                        download_with_retries(
                            session,
                            url,
                            dest_path,
                            retries=args.retries,
                            timeout=args.timeout,
                            max_download_seconds=args.max_download_seconds,
                            sleep_base=args.sleep_base,
                        )
                    except RuntimeError as exc:
                        error_msg = (
                            f"{csv_path} | row {idx} | {doc_id or doc_dir_name} | {url}: {exc}"
                        )
                        failed_downloads.append(error_msg)
                        logging.error(error_msg)
                        if not args.allow_failures:
                            raise
                        continue
                    log_verbose(
                        args,
                        "  downloaded %s (%s bytes)",
                        dest_path.name,
                        dest_path.stat().st_size if dest_path.exists() else "n/a",
                    )
                    if (
                        args.extract_pdf_text
                        and not (args.skip_pdf_when_html and row_has_html)
                        and not args.skip_pdf_download
                        and is_pdf
                    ):
                        maybe_extract_pdf_text(
                            dest_path,
                            parser_choice=args.pdf_parser,
                            parse_timeout=args.pdf_parse_timeout,
                            per_page_timeout=args.pdf_parse_page_timeout,
                            progress_threshold=args.pdf_progress_threshold,
                        )
                    elif args.skip_pdf_when_html and is_pdf:
                        log_verbose(
                            args,
                            "  skipped PDF extraction for %s (HTML present in row)",
                            dest_path.name,
                        )
                    downloaded_files += 1
                    per_csv_downloads += 1
                    if args.debug and per_csv_downloads >= args.debug_limit:
                        logging.debug(
                            "Debug limit reached for %s (limit=%s)",
                            csv_path,
                            args.debug_limit,
                        )
                        break
                per_csv_rows += 1
                processed_rows += 1
                if row_bar is not None:
                    row_bar.update(1)
        processed_files += 1
        if row_bar is not None:
            row_bar.close()
        logging.info(
            "Finished %s (%s rows with content, %s files downloaded)",
            csv_path.name,
            per_csv_rows,
            per_csv_downloads,
        )

    logging.info(
        "Done. Processed %s CSV files, %s rows with content, downloaded %s files.",
        processed_files,
        processed_rows,
        downloaded_files,
    )
    if failed_downloads:
        logging.warning(
            "Encountered %s failed downloads. See log summary below.",
            len(failed_downloads),
        )
        for entry in failed_downloads[:20]:
            logging.warning("FAILED: %s", entry)
        if len(failed_downloads) > 20:
            logging.warning("... %s additional failures omitted from log", len(failed_downloads) - 20)
        return 1 if not args.allow_failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
