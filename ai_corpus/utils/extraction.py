"""
Utilities for extracting text content from PDFs and HTML documents, including
scaffolded fallbacks (PyMuPDF and Tesseract OCR) for PDF pages.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup  # type: ignore

__all__ = [
    "AVAILABLE_PARSERS",
    "PARSER_BACKENDS",
    "extract_pages",
    "extract_pages_auto",
    "extract_with_pdfminer",
    "extract_with_pymupdf",
    "is_blank",
    "ocr_pages_with_tesseract",
    "pdf_to_text",
    "html_to_text",
]

logger = logging.getLogger(__name__)


def extract_with_pdfminer(pdf_path: str) -> List[str]:
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "pdfminer.six is required. Install with: pip install pdfminer.six"
        ) from exc

    laparams = LAParams(char_margin=2.0, line_margin=0.1, word_margin=0.1)
    text = extract_text(pdf_path, laparams=laparams)
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return [page.strip() for page in pages]


def extract_with_pymupdf(pdf_path: str) -> List[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "PyMuPDF is required. Install with: pip install pymupdf"
        ) from exc

    pages: List[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text").strip())
    return pages


PARSER_BACKENDS: Dict[str, Callable[[str], List[str]]] = {
    "pdfminer": extract_with_pdfminer,
    "fitz": extract_with_pymupdf,
}

AVAILABLE_PARSERS: Tuple[str, ...] = tuple(PARSER_BACKENDS.keys())


def is_blank(text: str) -> bool:
    return not text or not text.strip()


def ocr_pages_with_tesseract(pdf_path: str, indices: Sequence[int]) -> Dict[int, str]:
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
    with fitz.open(pdf_path) as doc:
        for index in indices:
            if index >= len(doc):
                continue
            page = doc[index]
            pix = page.get_pixmap()
            image_bytes = pix.tobytes("png")
            with Image.open(io.BytesIO(image_bytes)) as img:
                text = pytesseract.image_to_string(
                    img, lang="eng", config="--oem 1 --psm 6"
                )
            results[index] = text.strip()
    return results


def extract_pages_auto(pdf_path: str) -> Tuple[List[str], Dict[str, int]]:
    pages = extract_with_pdfminer(pdf_path)

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
    }

    if blanks:
        stats["pymupdf_attempted"] = 1
        try:
            pymupdf_pages = extract_with_pymupdf(pdf_path)
        except ImportError:
            stats["pymupdf_failed"] = 1
            pymupdf_pages = None
        else:
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
            ocr_text = ocr_pages_with_tesseract(pdf_path, blanks)
        except ImportError:
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


def extract_pages(pdf_path: str, parser: str = "auto") -> Tuple[List[str], Dict[str, int]]:
    if parser == "auto":
        return extract_pages_auto(pdf_path)
    try:
        extractor = PARSER_BACKENDS[parser]
    except KeyError as exc:
        valid = ", ".join(sorted((*AVAILABLE_PARSERS,)))
        raise ValueError(f"Unsupported parser '{parser}'. Choose from: {valid} or 'auto'.") from exc

    pages = extractor(pdf_path)
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


def pdf_to_text(path: Path) -> Optional[str]:
    """
    Convert a PDF into text using the shared extraction pipeline. Returns None on failure.
    """
    if not path.exists():
        logger.debug("PDF does not exist: %s", path)
        return None
    try:
        pages, _ = extract_pages(str(path), parser="auto")
    except (ImportError, ValueError) as exc:
        logger.warning("Failed to extract text from %s: %s", path, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error extracting %s: %s", path, exc)
        return None
    text = "\n\n".join(page for page in pages if page.strip())
    return text or None


def html_to_text(path: Path) -> Optional[str]:
    """
    Strip HTML into readable text using BeautifulSoup.
    """
    if not path.exists():
        logger.debug("HTML does not exist: %s", path)
        return None
    try:
        data = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read HTML %s: %s", path, exc)
        return None
    soup = BeautifulSoup(data, "html.parser")
    text = soup.get_text(separator="\n")
    return text.strip() or None


def docx_to_text(path: Path) -> Optional[str]:
    """Extract text from a DOCX file using docx2txt."""
    if not path.exists():
        logger.debug("DOCX does not exist: %s", path)
        return None
    try:
        import docx2txt  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        logger.warning(
            "docx2txt is required to process DOCX files. Install with: pip install docx2txt"
        )
        return None
    try:
        text = docx2txt.process(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to extract DOCX %s: %s", path, exc)
        return None
    if not text:
        return None
    return text.strip() or None
