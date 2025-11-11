"""
Content extraction helpers that transform downloaded assets into text blobs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from ai_corpus.utils import extraction as extraction_utils

logger = logging.getLogger(__name__)


def extract_text_from_artifacts(artifacts: Dict[str, object]) -> Optional[str]:
    """
    Given a connector fetch result, attempt to extract text content. Supports PDF
    and HTML inputs; returns the combined text or None if no extraction was
    possible.
    """
    pdf_path = artifacts.get("pdf")
    html_path = artifacts.get("html")
    docx_path = artifacts.get("docx")

    file_path = artifacts.get("file")
    if file_path and isinstance(file_path, str):
        lower = file_path.lower()
        if lower.endswith(".pdf") and not pdf_path:
            pdf_path = file_path
        elif lower.endswith(".html") and not html_path:
            html_path = file_path
        elif lower.endswith(".docx") and not docx_path:
            docx_path = file_path

    text_fragments = []
    if pdf_path:
        text = extraction_utils.pdf_to_text(Path(pdf_path))
        if text:
            text_fragments.append(text)
    if html_path:
        text = extraction_utils.html_to_text(Path(html_path))
        if text:
            text_fragments.append(text)
    if docx_path:
        text = extraction_utils.docx_to_text(Path(docx_path))
        if text:
            text_fragments.append(text)
    if not text_fragments:
        logger.debug("No extractable content found for %s", artifacts.get("doc_id"))
        return None
    return "\n\n".join(text_fragments)
