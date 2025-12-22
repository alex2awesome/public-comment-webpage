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
    "extract_pages",
    "extract_pages_auto",
    "extract_with_pdfminer",
    "extract_with_pymupdf",
    "get_pdf_page_count",
    "is_blank",
    "ocr_pages_with_tesseract",
]

ProgressCallback = Callable[[Dict[str, Any]], None]


def extract_with_pdfminer(
    pdf_path: str,
    *,
    fail_fast_preview: int = 0,
    per_page_timeout: Optional[float] = None,
    deadline: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
    total_pages: Optional[int] = None,
) -> List[str]:
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
    pages: List[str] = []
    if progress_callback:
        progress_callback({"type": "stage", "stage": "pdfminer", "total": total_pages})
    with open(pdf_path, "rb") as fh:
        rsrcmgr = PDFResourceManager()
        for idx, page in enumerate(PDFPage.get_pages(fh)):
            if deadline and time.perf_counter() > deadline:
                raise TimeoutError("pdfminer exceeded overall PDF parse timeout")
            output = io.StringIO()
            device = TextConverter(rsrcmgr, output, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            start_time = time.perf_counter()
            interpreter.process_page(page)
            text = output.getvalue().strip()
            pages.append(text)
            device.close()
            output.close()
            elapsed = time.perf_counter() - start_time
            if per_page_timeout and elapsed > per_page_timeout:
                raise TimeoutError(
                    f"pdfminer page {idx + 1} exceeded per-page timeout {per_page_timeout}s"
                )
            if fail_fast_preview and (idx + 1) == fail_fast_preview:
                if all(is_blank(value) for value in pages):
                    raise ValueError("pdfminer preview blank")
            if progress_callback:
                progress_callback(
                    {"type": "page", "stage": "pdfminer", "page": idx + 1, "total": total_pages}
                )
    return pages


def extract_with_pymupdf(
    pdf_path: str,
    *,
    fail_fast_preview: int = 0,
    per_page_timeout: Optional[float] = None,
    deadline: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
    total_pages: Optional[int] = None,
) -> List[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "PyMuPDF is required. Install with: pip install pymupdf"
        ) from exc

    pages: List[str] = []
    if progress_callback:
        progress_callback({"type": "stage", "stage": "pymupdf", "total": total_pages})
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            if deadline and time.perf_counter() > deadline:
                raise TimeoutError("pymupdf exceeded overall PDF parse timeout")
            start_time = time.perf_counter()
            pages.append(page.get_text("text").strip())
            elapsed = time.perf_counter() - start_time
            if per_page_timeout and elapsed > per_page_timeout:
                raise TimeoutError(
                    f"pymupdf page {idx + 1} exceeded per-page timeout {per_page_timeout}s"
                )
            if fail_fast_preview and (idx + 1) == fail_fast_preview:
                if all(is_blank(value) for value in pages):
                    raise ValueError("pymupdf preview blank")
            if progress_callback:
                progress_callback(
                    {"type": "page", "stage": "pymupdf", "page": idx + 1, "total": total_pages}
                )
    return pages


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
) -> Tuple[List[str], Dict[str, int]]:
    fail_fast_preview = 10
    pdfminer_preview_failed = False
    try:
        pages = extract_with_pdfminer(
            pdf_path,
            fail_fast_preview=fail_fast_preview,
            per_page_timeout=per_page_timeout,
            deadline=parse_deadline,
            progress_callback=progress_callback,
            total_pages=total_pages,
        )
    except ValueError:
        pdfminer_preview_failed = True
        pages = []
    except TimeoutError:
        pdfminer_preview_failed = True
        pages = []

    def blank_indices(seq: Sequence[str]) -> List[int]:
        return [idx for idx, value in enumerate(seq) if is_blank(value)]

    blanks = blank_indices(pages)
    stats: Dict[str, int] = {
        "total_pages": len(pages),
        "initial_blanks": len(blanks),
        "pymupdf_filled": 0,
        "pymupdf_attempted": 0,
        "pymupdf_failed": 0,
        "ocr_filled": 0,
        "ocr_attempted": 0,
        "ocr_failed": 0,
        "remaining_blanks": len(blanks),
        "pdfminer_preview_failed": int(pdfminer_preview_failed),
        "pymupdf_preview_failed": 0,
    }

    pymupdf_pages = None
    if not pages or blanks:
        stats["pymupdf_attempted"] = 1
        try:
            pymupdf_pages = extract_with_pymupdf(
                pdf_path,
                fail_fast_preview=0,
                per_page_timeout=per_page_timeout,
                deadline=parse_deadline,
                progress_callback=progress_callback,
                total_pages=total_pages,
            )
        except TimeoutError:
            stats["pymupdf_preview_failed"] = 1
            pymupdf_pages = None
        except ImportError:
            stats["pymupdf_failed"] = 1
            pymupdf_pages = None

    if not pages and pymupdf_pages:
        pages = pymupdf_pages
        blanks = blank_indices(pages)
        stats["total_pages"] = len(pages)
        stats["initial_blanks"] = len(blanks)
        stats["pymupdf_filled"] = len(pages) - len(blanks)
        stats["remaining_blanks"] = len(blanks)
    elif blanks and pymupdf_pages:
        before_count = len(blanks)
        for idx in blanks:
            if idx < len(pymupdf_pages) and not is_blank(pymupdf_pages[idx]):
                pages[idx] = pymupdf_pages[idx]
        blanks = blank_indices(pages)
        stats["pymupdf_filled"] = before_count - len(blanks)
        stats["remaining_blanks"] = len(blanks)

    if blanks:
        stats["ocr_attempted"] = 1
        try:
            ocr_text = ocr_pages_with_tesseract(
                pdf_path,
                blanks,
                per_page_timeout=per_page_timeout,
                deadline=parse_deadline,
                progress_callback=progress_callback,
            )
        except ImportError:
            stats["ocr_failed"] = 1
            ocr_text = {}
        except TimeoutError:
            stats["ocr_failed"] = 1
            ocr_text = {}
        else:
            before_count = len(blanks)
            for idx, text in ocr_text.items():
                if not is_blank(text):
                    pages[idx] = text
            blanks = blank_indices(pages)
            stats["ocr_filled"] = before_count - len(blanks)
            stats["remaining_blanks"] = len(blanks)

    stats["remaining_blanks"] = len(blanks)
    return pages, stats


def extract_pages(
    pdf_path: str,
    parser: str = "auto",
    *,
    per_page_timeout: Optional[float] = None,
    parse_deadline: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
    total_pages: Optional[int] = None,
) -> Tuple[List[str], Dict[str, int]]:
    if parser == "auto":
        return extract_pages_auto(
            pdf_path,
            per_page_timeout=per_page_timeout,
            parse_deadline=parse_deadline,
            progress_callback=progress_callback,
            total_pages=total_pages,
        )
    try:
        extractor = PARSER_BACKENDS[parser]
    except KeyError as exc:
        valid = ", ".join(sorted((*AVAILABLE_PARSERS,)))
        raise ValueError(f"Unsupported parser '{parser}'. Choose from: {valid} or 'auto'.") from exc

    pages = extractor(
        pdf_path,
        per_page_timeout=per_page_timeout,
        deadline=parse_deadline,
        progress_callback=progress_callback,
        total_pages=total_pages,
    )
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
    return pages, stats
