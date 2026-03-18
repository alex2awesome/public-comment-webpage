"""
Shared helpers for extracting text from PDFs, including OCR fallbacks.
"""

from __future__ import annotations

import io
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "AVAILABLE_PARSERS",
    "PARSER_BACKENDS",
    "PartialParseTimeout",
    "extract_pages",
    "extract_pages_auto",
    "extract_with_pdfminer",
    "extract_with_pymupdf",
    "get_pdf_page_count",
    "is_blank",
    "ocr_pages_with_tesseract",
]

ProgressCallback = Callable[[Dict[str, Any]], None]


class PartialParseTimeout(TimeoutError):
    """
    Raised when a parser exceeds its deadline but has already produced partial pages.
    """

    def __init__(self, message: str, pages: Sequence[str], stage: str) -> None:
        super().__init__(message)
        self.partial_pages = list(pages)
        self.stage = stage


def extract_with_pdfminer(
    pdf_path: str,
    *,
    fail_fast_preview: int = 0,
    per_page_timeout: Optional[float] = None,
    deadline: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
    total_pages: Optional[int] = None,
    max_pages: Optional[int] = None,
    page_callback: Optional[Callable[[int, str], None]] = None,
    capture_pages: bool = True,
    start_page: int = 1,
) -> List[str]:
    pages: List[str] = []
    for page_number, text in _iter_pdfminer_pages(
        pdf_path,
        fail_fast_preview=fail_fast_preview,
        per_page_timeout=per_page_timeout,
        deadline=deadline,
        progress_callback=progress_callback,
        total_pages=total_pages,
        max_pages=max_pages,
        start_page=start_page,
    ):
        if capture_pages:
            pages.append(text)
        if page_callback:
            page_callback(page_number, text)
    return pages


def extract_with_pymupdf(
    pdf_path: str,
    *,
    fail_fast_preview: int = 0,
    per_page_timeout: Optional[float] = None,
    deadline: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
    total_pages: Optional[int] = None,
    max_pages: Optional[int] = None,
    page_callback: Optional[Callable[[int, str], None]] = None,
    capture_pages: bool = True,
    start_page: int = 1,
) -> List[str]:
    pages: List[str] = []
    for page_number, text in _iter_pymupdf_pages(
        pdf_path,
        per_page_timeout=per_page_timeout,
        deadline=deadline,
        progress_callback=progress_callback,
        total_pages=total_pages,
        max_pages=max_pages,
        start_page=start_page,
    ):
        if capture_pages:
            pages.append(text)
        if page_callback:
            page_callback(page_number, text)
    return pages


def _iter_pdfminer_pages(
    pdf_path: str,
    *,
    fail_fast_preview: int = 0,
    per_page_timeout: Optional[float],
    deadline: Optional[float],
    progress_callback: Optional[ProgressCallback],
    total_pages: Optional[int],
    max_pages: Optional[int],
    start_page: int,
):
    try:
        from pdfminer.converter import TextConverter
        from pdfminer.layout import LAParams
        from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
        from pdfminer.pdfpage import PDFPage
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "pdfminer.six is required. Install with: pip install pdfminer.six"
        ) from exc

    laparams = LAParams(char_margin=2.0, line_margin=0.1, word_margin=0.1)
    blank_window: List[bool] = []
    if progress_callback:
        progress_callback({"type": "stage", "stage": "pdfminer", "total": total_pages})
    page_limit = max_pages if max_pages and max_pages > 0 else None
    emitted = 0
    with open(pdf_path, "rb") as fh:
        rsrcmgr = PDFResourceManager()
        for idx, page in enumerate(PDFPage.get_pages(fh)):
            page_number = idx + 1
            if page_number < start_page:
                continue
            if deadline and time.perf_counter() > deadline:
                raise PartialParseTimeout(
                    "pdfminer exceeded overall PDF parse timeout",
                    [],
                    "pdfminer",
                )
            output = io.StringIO()
            device = TextConverter(rsrcmgr, output, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            start_time = time.perf_counter()
            interpreter.process_page(page)
            text = output.getvalue().strip()
            device.close()
            output.close()
            elapsed = time.perf_counter() - start_time
            if per_page_timeout and elapsed > per_page_timeout:
                raise PartialParseTimeout(
                    f"pdfminer page {idx + 1} exceeded per-page timeout {per_page_timeout}s",
                    [],
                    "pdfminer",
                )
            blank_window.append(is_blank(text))
            if len(blank_window) > fail_fast_preview > 0:
                blank_window.pop(0)
            if fail_fast_preview and len(blank_window) == fail_fast_preview:
                if all(blank_window):
                    raise ValueError("pdfminer preview blank")
            if progress_callback:
                progress_callback(
                    {"type": "page", "stage": "pdfminer", "page": idx + 1, "total": total_pages}
                )
            yield idx + 1, text
            emitted += 1
            if deadline and time.perf_counter() > deadline:
                raise PartialParseTimeout(
                    "pdfminer exceeded overall PDF parse timeout",
                    [],
                    "pdfminer",
                )
            if page_limit and emitted >= page_limit:
                break


def _iter_pymupdf_pages(
    pdf_path: str,
    *,
    per_page_timeout: Optional[float],
    deadline: Optional[float],
    progress_callback: Optional[ProgressCallback],
    total_pages: Optional[int],
    max_pages: Optional[int],
    start_page: int = 1,
):
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "PyMuPDF is required. Install with: pip install pymupdf"
        ) from exc

    if progress_callback:
        progress_callback({"type": "stage", "stage": "pymupdf", "total": total_pages})
    page_limit = max_pages if max_pages and max_pages > 0 else None
    emitted = 0
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            page_number = idx + 1
            if page_number < start_page:
                continue
            if deadline and time.perf_counter() > deadline:
                raise PartialParseTimeout(
                    "pymupdf exceeded overall PDF parse timeout",
                    [],
                    "pymupdf",
                )
            start_time = time.perf_counter()
            text = page.get_text("text").strip()
            elapsed = time.perf_counter() - start_time
            if per_page_timeout and elapsed > per_page_timeout:
                raise PartialParseTimeout(
                    f"pymupdf page {idx + 1} exceeded per-page timeout {per_page_timeout}s",
                    [],
                    "pymupdf",
                )
            if progress_callback:
                progress_callback(
                    {"type": "page", "stage": "pymupdf", "page": idx + 1, "total": total_pages}
                )
            yield idx + 1, text
            emitted += 1
            if deadline and time.perf_counter() > deadline:
                raise PartialParseTimeout(
                    "pymupdf exceeded overall PDF parse timeout",
                    [],
                    "pymupdf",
                )
            if page_limit and emitted >= page_limit:
                break


def _ocr_single_page(
    doc,
    page_index: int,
    *,
    per_page_timeout: Optional[float],
    deadline: Optional[float],
) -> Optional[str]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Pillow is required for OCR fallback. Install with: pip install Pillow"
        ) from exc
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "pytesseract is required for OCR fallback. Install with: pip install pytesseract"
        ) from exc

    page_start = time.perf_counter()
    page = doc[page_index]
    pix = page.get_pixmap()
    image_bytes = pix.tobytes("png")
    with Image.open(io.BytesIO(image_bytes)) as img:
        text = pytesseract.image_to_string(img, lang="eng", config="--oem 1 --psm 6")
    elapsed = time.perf_counter() - page_start
    if per_page_timeout and elapsed > per_page_timeout:
        raise TimeoutError(
            f"OCR for page {page_index + 1} exceeded per-page timeout {per_page_timeout}s"
        )
    if deadline and time.perf_counter() > deadline:
        raise TimeoutError("OCR exceeded overall PDF parse timeout")
    cleaned = text.strip()
    return cleaned or None


PARSER_BACKENDS: Dict[str, Callable[..., List[str]]] = {
    "pdfminer": extract_with_pdfminer,
    "fitz": extract_with_pymupdf,
}

AVAILABLE_PARSERS: Tuple[str, ...] = tuple(PARSER_BACKENDS.keys())


def is_blank(text: str) -> bool:
    return not text or not text.strip()


def get_pdf_page_count(pdf_path: str) -> Optional[int]:
    try:
        import PyPDF2
    except ImportError:
        PyPDF2 = None

    if PyPDF2 is not None:
        try:
            with open(pdf_path, "rb") as fh:
                reader = PyPDF2.PdfReader(fh)
                return len(reader.pages)
        except Exception:
            pass
    try:
        import fitz  # PyMuPDF
    except ImportError:
        fitz = None
    if fitz is not None:
        try:
            with fitz.open(pdf_path) as doc:
                return doc.page_count
        except Exception:
            pass
    return None


def ocr_pages_with_tesseract(
    pdf_path: str,
    indices: Sequence[int],
    *,
    per_page_timeout: Optional[float] = None,
    deadline: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[int, str]:
    if not indices:
        return {}
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "PyMuPDF is required for OCR fallback. Install with: pip install pymupdf"
        ) from exc
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Pillow is required for OCR fallback. Install with: pip install Pillow"
        ) from exc
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "pytesseract is required for OCR fallback. Install with: pip install pytesseract"
        ) from exc

    results: Dict[int, str] = {}
    total = len(indices)
    if progress_callback:
        progress_callback({"type": "stage", "stage": "ocr", "total": total})
    with fitz.open(pdf_path) as doc:
        for position, index in enumerate(indices, start=1):
            if index >= len(doc):
                continue
            page_start = time.perf_counter()
            page = doc[index]
            pix = page.get_pixmap()
            image_bytes = pix.tobytes("png")
            with Image.open(io.BytesIO(image_bytes)) as img:
                text = pytesseract.image_to_string(
                    img, lang="eng", config="--oem 1 --psm 6"
                )
            results[index] = text.strip()
            elapsed = time.perf_counter() - page_start
            if per_page_timeout and elapsed > per_page_timeout:
                raise TimeoutError(
                    f"OCR for page {index + 1} exceeded per-page timeout {per_page_timeout}s"
                )
            if deadline and time.perf_counter() > deadline:
                raise TimeoutError("OCR exceeded overall PDF parse timeout")
            if progress_callback:
                progress_callback(
                    {"type": "page", "stage": "ocr", "page": position, "total": total}
                )
    return results


def extract_pages_auto(
    pdf_path: str,
    *,
    per_page_timeout: Optional[float] = None,
    parse_deadline: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
    total_pages: Optional[int] = None,
    max_pages: Optional[int] = None,
    page_callback: Optional[Callable[[int, str], None]] = None,
    capture_pages: bool = True,
    start_page: int = 1,
) -> Tuple[List[str], Dict[str, int]]:
    start_time = time.perf_counter()
    pages: List[str] = [] if capture_pages else []
    stats: Dict[str, int] = {
        "total_pages": 0,
        "initial_blanks": 0,
        "pymupdf_filled": 0,
        "pymupdf_attempted": 0,
        "pymupdf_failed": 0,
        "ocr_filled": 0,
        "ocr_attempted": 0,
        "ocr_failed": 0,
        "remaining_blanks": 0,
        "pdfminer_preview_failed": 0,
        "pymupdf_preview_failed": 0,
    }
    timed_out_stage: Optional[str] = None
    timed_out_reason: Optional[str] = None
    blank_count = 0
    remaining_blanks = 0
    pages_processed = 0
    reach_limit = False

    def emit(page_number: int, text: str) -> None:
        if capture_pages:
            pages.append(text)
        if page_callback:
            page_callback(page_number, text)

    fitz_doc = None

    def ensure_fitz_doc():
        nonlocal fitz_doc
        if fitz_doc is None:
            import fitz  # PyMuPDF

            fitz_doc = fitz.open(pdf_path)
        return fitz_doc

    try:
        iterator = _iter_pdfminer_pages(
            pdf_path,
            fail_fast_preview=10,
            per_page_timeout=per_page_timeout,
            deadline=parse_deadline,
            progress_callback=progress_callback,
            total_pages=total_pages,
            max_pages=max_pages,
            start_page=start_page,
        )
        for page_number, text in iterator:
            pages_processed += 1
            blank = is_blank(text)
            if blank:
                blank_count += 1
                stats["initial_blanks"] = blank_count
                try:
                    doc = ensure_fitz_doc()
                    stats["pymupdf_attempted"] += 1
                    fallback = doc[page_number - 1].get_text("text").strip()
                except Exception:
                    fallback = ""
                    stats["pymupdf_failed"] += 1
                else:
                    if fallback:
                        text = fallback
                        blank = False
                        stats["pymupdf_filled"] += 1
                if blank:
                    stats["ocr_attempted"] += 1
                    try:
                        doc = ensure_fitz_doc()
                        fallback = _ocr_single_page(
                            doc,
                            page_number - 1,
                            per_page_timeout=per_page_timeout,
                            deadline=parse_deadline,
                        )
                    except Exception:
                        fallback = None
                        stats["ocr_failed"] += 1
                    else:
                        if fallback:
                            text = fallback
                            blank = False
                            stats["ocr_filled"] += 1
            emit(page_number, text)
            if blank:
                remaining_blanks += 1
            if max_pages and pages_processed >= max_pages:
                reach_limit = True
                break
            if parse_deadline and time.perf_counter() > parse_deadline:
                timed_out_stage = "pdfminer"
                timed_out_reason = "parse deadline exceeded"
                break
    except ValueError:
        stats["pdfminer_preview_failed"] = 1
        pages_processed = 0
    except PartialParseTimeout as exc:
        timed_out_stage = exc.stage or "pdfminer"
        timed_out_reason = str(exc)
    except TimeoutError as exc:
        timed_out_stage = "pdfminer"
        timed_out_reason = str(exc)

    if pages_processed == 0 and stats["pdfminer_preview_failed"] and not timed_out_stage and not reach_limit:
        try:
            for page_number, text in _iter_pymupdf_pages(
                pdf_path,
                per_page_timeout=per_page_timeout,
                deadline=parse_deadline,
                progress_callback=progress_callback,
                total_pages=total_pages,
                max_pages=max_pages,
                start_page=start_page,
            ):
                pages_processed += 1
                blank = is_blank(text)
                emit(page_number, text)
                if blank:
                    remaining_blanks += 1
                if max_pages and pages_processed >= max_pages:
                    reach_limit = True
                    break
                if parse_deadline and time.perf_counter() > parse_deadline:
                    timed_out_stage = "pymupdf"
                    timed_out_reason = "parse deadline exceeded"
                    break
        except PartialParseTimeout as exc:
            timed_out_stage = exc.stage or "pymupdf"
            timed_out_reason = str(exc)
        except TimeoutError as exc:
            timed_out_stage = "pymupdf"
            timed_out_reason = str(exc)

    stats["total_pages"] = pages_processed
    stats["remaining_blanks"] = remaining_blanks
    stats["timed_out"] = 1 if timed_out_stage else 0
    stats["timed_out_stage"] = timed_out_stage or ""
    if timed_out_reason:
        stats["timed_out_reason"] = timed_out_reason
    stats["elapsed_seconds"] = time.perf_counter() - start_time
    if fitz_doc is not None:
        try:
            fitz_doc.close()
        except Exception:
            pass
    return pages, stats


def extract_pages(
    pdf_path: str,
    parser: str = "auto",
    *,
    per_page_timeout: Optional[float] = None,
    parse_deadline: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
    total_pages: Optional[int] = None,
    max_pages: Optional[int] = None,
    page_callback: Optional[Callable[[int, str], None]] = None,
    capture_pages: bool = True,
    start_page: int = 1,
) -> Tuple[List[str], Dict[str, int]]:
    if parser == "auto":
        return extract_pages_auto(
            pdf_path,
            per_page_timeout=per_page_timeout,
            parse_deadline=parse_deadline,
            progress_callback=progress_callback,
            total_pages=total_pages,
            max_pages=max_pages,
            page_callback=page_callback,
            capture_pages=capture_pages,
            start_page=start_page,
        )
    timed_out_stage: Optional[str] = None
    timed_out_reason: Optional[str] = None
    try:
        extractor = PARSER_BACKENDS[parser]
    except KeyError as exc:
        valid = ", ".join(sorted((*AVAILABLE_PARSERS,)))
        raise ValueError(f"Unsupported parser '{parser}'. Choose from: {valid} or 'auto'.") from exc

    try:
        pages = extractor(
            pdf_path,
            per_page_timeout=per_page_timeout,
            deadline=parse_deadline,
            progress_callback=progress_callback,
            total_pages=total_pages,
            max_pages=max_pages,
            page_callback=page_callback,
            capture_pages=capture_pages,
            start_page=start_page,
        )
    except PartialParseTimeout as exc:
        pages = exc.partial_pages
        timed_out_stage = exc.stage or parser
        timed_out_reason = str(exc)
    blank_count = sum(1 for page in pages if is_blank(page))
    stats = {
        "total_pages": len(pages),
        "initial_blanks": blank_count,
        "pymupdf_filled": 0,
        "pymupdf_attempted": 0,
        "pymupdf_failed": 0,
        "ocr_filled": 0,
        "ocr_attempted": 0,
        "ocr_failed": 0,
        "remaining_blanks": blank_count,
    }
    stats["timed_out"] = 1 if timed_out_stage else 0
    stats["timed_out_stage"] = timed_out_stage or ""
    if timed_out_reason:
        stats["timed_out_reason"] = timed_out_reason
    return pages, stats
