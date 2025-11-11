"""
Connector for California Privacy Protection Agency ADMT rulemaking comments.

This implementation mirrors the exploratory Playwright helper but conforms to
the standard connector interface (discover → list_documents → fetch). Discovery
enumerates the aggregated comment PDFs, `list_documents` exposes them as
downloadable artifacts, and `fetch` downloads each bundle before delegating to
an OpenAI-powered page pairing workflow that splits the aggregate into
individual submissions.
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup  # type: ignore
from pydantic import BaseModel
from tqdm import tqdm

from ai_corpus.connectors.base import Collection, DocMeta
from ai_corpus.utils.extraction import extract_pages
from ai_corpus.utils.http import RateLimiter, backoff_get, get_http_session, next_user_agent
from ai_corpus.utils.llm_utils import init_backend, request_structured_response

logger = logging.getLogger(__name__)

PDF_CATEGORY_PATTERNS = [
    ("aggregated_comments", re.compile(r"(all[_\s-]*comments|written comments\b|comments? combined|pre[_\s-]*comments|late[_\s-]*comments|comment summaries)", re.IGNORECASE)),
    ("comment_summary_and_response", re.compile(r"comment summaries?\s+and\s+responses?", re.IGNORECASE)),
    ("hearing_transcript", re.compile(r"transcript", re.IGNORECASE)),
    ("notice", re.compile(r"notice", re.IGNORECASE)),
    ("final_statement_of_reasons", re.compile(r"(final statement of reasons|\bfsor\b)", re.IGNORECASE)),
    ("initial_statement_of_reasons", re.compile(r"(initial statement of reasons|\bisor\b)", re.IGNORECASE)),
    ("economic_impact", re.compile(r"(economic|fiscal impact|std[_\s-]*399|assessment)", re.IGNORECASE)),
    ("reg_text_approved", re.compile(r"approved regulations? text", re.IGNORECASE)),
    ("reg_text_modified", re.compile(r"(modified text|mod[_\s-]*txt)", re.IGNORECASE)),
    ("reg_text_proposed", re.compile(r"text of proposed", re.IGNORECASE)),
    ("invitation_preliminary", re.compile(r"invitation.*comment", re.IGNORECASE)),
    ("supporting_appendix", re.compile(r"appendix", re.IGNORECASE)),
]

CPPA_HISTORY_ORDER = [
    "notice",
    "invitation_preliminary",
    "initial_statement_of_reasons",
    "reg_text_proposed",
    "reg_text_modified",
    "reg_text_approved",
    "comment_summary_and_response",
    "final_statement_of_reasons",
    "aggregated_comments",
    "hearing_transcript",
    "supporting_appendix",
    "economic_impact",
    "other",
]

PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are helping split a large PDF of public comments into individual submissions.
    I will show you two consecutive pages. Your task is simply to decide whether the
    text on Page B continues the same comment from Page A.

    Respond with JSON: {{"label": "1"}} if Page B continues the same comment as Page A,
    or {{"label": "0"}} if Page B starts a new comment. No extra fields or explanation.

    [Page A]
    {page_a}

    [Page B]
    {page_b}

    Answer:"""
)


class PagePairJudgment(BaseModel):
    """Structured output schema for OpenAI responses."""

    label: str


@dataclass(slots=True)
class PdfEntry:
    title: str
    url: str
    filename: str
    kind: str
    raw_href: str


class CppaAdmtConnector:
    name = "cppa_admt"

    def __init__(
        self,
        config: Dict[str, Any],
        global_config: Optional[Dict[str, Any]] = None,
        session=None,
    ) -> None:
        self.config = config or {}
        self.global_config = global_config or {}
        self.session = session or get_http_session(
            {"User-Agent": self.global_config.get("user_agent")}
        )
        self.rate_limiter = RateLimiter(min_interval=0.5)
        self.index_url = self.config.get("index_url")
        self.base_url = self.config.get("base_url", "https://cppa.ca.gov").rstrip("/")
        self.openai_model = self.config.get("openai_model", "gpt-4o")
        self.openai_max_output_tokens = int(self.config.get("openai_max_output_tokens", 96))
        self.parser = self.config.get("parser", "auto")
        self.min_segment_chars = int(self.config.get("min_segment_chars", 400))

        self._entries: Optional[List[PdfEntry]] = None
        self._openai_client: Any | None = None
        self._openai_initialized = False

    # ------------------------------------------------------------------ discovery
    def discover(self, **_) -> Iterable[Collection]:
        """
        Enumerate available collections. The CPPA page currently exposes a
        single docket covering the ADMT preliminary comments, but keeping this
        method flexible allows future expansion if additional dockets appear.
        """

        if not self.index_url:
            logger.warning("CPPA ADMT connector requires an index_url configuration.")
            return []

        self._ensure_entries()
        yield Collection(
            source=self.name,
            collection_id="PR-02-2023",
            title="CPPA ADMT Preliminary Comments",
            url=self.index_url,
            jurisdiction="US-CA",
            topic="AI",
            extra={"total_pdfs": len(self._entries or [])},
        )

    # -------------------------------------------------------------- document list
    def list_documents(self, collection_id: str, **_) -> Iterable[DocMeta]:
        """
        Return metadata for each aggregated comment PDF we discovered. These are
        the large bundles that need to be split into individual submissions.
        """

        if collection_id != "PR-02-2023":
            logger.warning("Unknown collection_id '%s' for CPPA ADMT connector", collection_id)
            return []

        entries = self._ensure_entries()
        for entry in entries:
            if entry.kind != "aggregated_comments":
                continue
            yield DocMeta(
                source=self.name,
                collection_id=collection_id,
                doc_id=entry.filename,
                title=entry.title or entry.filename,
                submitter=None,
                submitter_type=None,
                org=None,
                submitted_at=None,
                language="en",
                urls={"pdf": entry.url},
                extra={"kind": entry.kind, "raw_href": entry.raw_href, "is_bundle": True},
            )

    def get_call_document(self, collection_id: str, **_) -> Optional[DocMeta]:
        if collection_id != "PR-02-2023":
            return None
        entries = self._ensure_entries()
        if not entries:
            return None
        call_priority = [
            "notice",
            "invitation_preliminary",
            "initial_statement_of_reasons",
            "comment_summary_and_response",
        ]
        lookup = {entry.kind: entry for entry in entries}
        for label in call_priority:
            if label in lookup:
                entry = lookup[label]
                return DocMeta(
                    source=self.name,
                    collection_id=collection_id,
                    doc_id=entry.filename,
                    title=entry.title or entry.filename,
                    submitter="CPPA",
                    submitter_type="agency",
                    org="California Privacy Protection Agency",
                    submitted_at=None,
                    language="en",
                    urls={"pdf": entry.url},
                    extra={
                        "kind": entry.kind,
                        "raw_href": entry.raw_href,
                        "document_role": "call",
                    },
                    kind="call",
                )
        # Fallback: return the first non-aggregated PDF if available.
        for entry in entries:
            if entry.kind != "aggregated_comments":
                return DocMeta(
                    source=self.name,
                    collection_id=collection_id,
                    doc_id=entry.filename,
                    title=entry.title or entry.filename,
                    submitter="CPPA",
                    submitter_type="agency",
                    org="California Privacy Protection Agency",
                    submitted_at=None,
                    language="en",
                    urls={"pdf": entry.url},
                    extra={
                        "kind": entry.kind,
                        "raw_href": entry.raw_href,
                        "document_role": "call",
                    },
                    kind="call",
                )
        return None

    # --------------------------------------------------------------------- fetch
    def fetch(self, doc: DocMeta, out_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Download an aggregated PDF and split it into per-submission text files
        using the LLM-guided chunker.
        """

        pdf_url = doc.urls.get("pdf")
        result: Dict[str, Any] = {"doc_id": doc.doc_id}
        if not pdf_url:
            logger.warning("Document %s is missing a PDF URL.", doc.doc_id)
            return result

        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / doc.doc_id
        if not pdf_path.suffix:
            pdf_path = pdf_path.with_suffix(".pdf")

        skip_existing = kwargs.get("skip_existing", True)
        if skip_existing and pdf_path.exists() and pdf_path.stat().st_size > 0:
            logger.debug("Reusing existing PDF %s", pdf_path)
            result["pdf"] = str(pdf_path)
        else:
            response = backoff_get(
                pdf_url,
                session=self.session,
                rate_limiter=self.rate_limiter,
                headers={"User-Agent": next_user_agent()},
                raise_for_status=False,
            )
            if response is None or response.status_code != 200:
                logger.warning("Failed to download PDF %s (status %s)", pdf_url, getattr(response, "status_code", "?"))
                return result
            pdf_path.write_bytes(response.content)
            result["pdf"] = str(pdf_path)

        if doc.extra.get("is_bundle"):
            letters = self._split_bundle(pdf_path)
            if letters:
                result["letters"] = letters
        return result

    def iter_rule_versions(self, collection_id: str, **_) -> Iterable[Dict[str, Any]]:
        if collection_id != "PR-02-2023":
            return []
        entries = list(self._ensure_entries())
        if not entries:
            return []
        order = {label: idx for idx, label in enumerate(CPPA_HISTORY_ORDER)}
        entries.sort(key=lambda entry: (order.get(entry.kind, len(order)), entry.filename))
        parent_filename = entries[0].filename if entries else None
        rows: List[Dict[str, Any]] = []
        for idx, entry in enumerate(entries, start=1):
            mentions_comments = entry.kind == "comment_summary_and_response"
            row = {
                "scrape_mode": "connector_rule_history",
                "source": self.name,
                "fr_document_number": entry.filename,
                "title": entry.title or entry.filename,
                "type": entry.kind,
                "publication_date": "",
                "agency": "California Privacy Protection Agency",
                "fr_url": entry.url,
                "docket_id": collection_id,
                "docket_ids": [collection_id],
                "regs_url": "",
                "regs_document_id": "",
                "regs_object_id": "",
                "comment_start_date": "",
                "comment_due_date": "",
                "comment_status": "",
                "comment_active": False,
                "is_rfi_rfc": "TRUE" if entry.kind in {"notice", "invitation_preliminary"} else "FALSE",
                "rfi_rfc_label": "RFI" if entry.kind in {"notice", "invitation_preliminary"} else "",
                "details": entry.kind,
                "abstract": "",
                "action": "",
                "dates": "",
                "supplementary_information": "",
                "history_parent_docket": collection_id,
                "history_parent_fr_doc": parent_filename or entry.filename,
                "history_stage": entry.kind,
                "history_relationship": "seed" if idx == 1 else entry.kind,
                "history_rank": str(idx),
                "mentions_comment_response": mentions_comments,
                "comment_citation_snippet": entry.title if mentions_comments else "",
            }
            rows.append(row)
        return rows

    # ------------------------------------------------------------ internal utils
    def _ensure_entries(self) -> List[PdfEntry]:
        if self._entries is not None:
            return self._entries

        if not self.index_url:
            self._entries = []
            return self._entries

        response = backoff_get(
            self.index_url,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": self.global_config.get("user_agent")},
            raise_for_status=False,
        )
        if response is None or response.status_code != 200:
            logger.warning("Failed to load CPPA index page from %s", self.index_url)
            self._entries = []
            return self._entries

        soup = BeautifulSoup(response.text, "html.parser")
        entries: List[PdfEntry] = []
        seen: set[str] = set()
        for anchor in soup.select("a[href$='.pdf']"):
            href = (anchor.get("href") or "").strip()
            if not href:
                continue
            if href in seen:
                continue
            seen.add(href)
            if href.startswith(("http://", "https://")):
                full_url = href
            else:
                base_ref = self.index_url or (self.base_url + "/")
                full_url = urljoin(base_ref, href)
            parsed = urlparse(full_url)
            filename = self._sanitize_filename(Path(parsed.path).name or "document.pdf")
            kind = self._classify_pdf(anchor.text, href)
            entries.append(
                PdfEntry(
                    title=(anchor.text or filename).strip(),
                    url=full_url,
                    filename=filename,
                    kind=kind,
                    raw_href=href,
                )
            )
        self._entries = entries
        return self._entries

    def _classify_pdf(self, anchor_text: Optional[str], href: str) -> str:
        haystack = " ".join(part.lower() for part in (anchor_text or "", href or "") if part)
        for label, pattern in PDF_CATEGORY_PATTERNS:
            if pattern.search(haystack):
                return label
        return "other"

    def _sanitize_filename(self, name: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".", " ") else "_" for ch in name)
        cleaned = "_".join(cleaned.split())
        return cleaned or "document.pdf"

    # ------------------------------------------------------------- LLM helpers
    def _get_openai_client(self) -> Any | None:
        if self._openai_initialized:
            return self._openai_client
        self._openai_initialized = True
        try:
            clients = init_backend("openai")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI backend unavailable for CPPA splitting: %s", exc)
            self._openai_client = None
        else:
            self._openai_client = clients.sync
        return self._openai_client

    def _split_bundle(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Split a bundle PDF into individual letters using the structured-output
        page pairing flow. Falls back to a single segment if the model is
        unavailable or produces empty output.
        """

        try:
            pages, _stats = extract_pages(str(pdf_path), self.parser)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract pages from %s: %s", pdf_path, exc)
            return []

        pages = [page for page in pages if page is not None]
        if not pages:
            logger.warning("No text extracted from %s", pdf_path)
            return []

        page_count = len(pages)
        if page_count == 1:
            combined = pages[0].strip()
            if not combined:
                return []
            return self._write_segments(pdf_path, [combined])

        client = self._get_openai_client()
        if client is None:
            logger.warning("Skipping LLM chunking for %s because OpenAI client is unavailable.", pdf_path)
            return []

        logger.info("Splitting %s with %d pages using %s", pdf_path.name, page_count, self.openai_model)
        logger.info("Evaluating %d page transitions", page_count - 1)

        judgments: List[PagePairJudgment] = []
        progress = tqdm(
            total=page_count - 1,
            desc=f"Splitting {pdf_path.name}",
            unit="cmp",
            leave=False,
        )
        for index in range(len(pages) - 1):
            page_a = pages[index]
            page_b = pages[index + 1]
            prompt_text = PROMPT_TEMPLATE.format(page_a=page_a, page_b=page_b)
            logger.info("OpenAI call for %s pages %d-%d", pdf_path.name, index + 1, index + 2)
            try:
                judgment = request_structured_response(
                    client,
                    self.openai_model,
                    prompt_text,
                    PagePairJudgment,
                    max_output_tokens=self.openai_max_output_tokens,
                    temperature=None,
                )
                logger.info(
                    "OpenAI result for %s pages %d-%d: label=%s",
                    pdf_path.name,
                    index + 1,
                    index + 2,
                    judgment.label,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Structured judge failed for pages %s-%s of %s: %s; assuming same comment.",
                    index + 1,
                    index + 2,
                    pdf_path.name,
                    exc,
                )
                judgment = PagePairJudgment(label="1", a_is_error=False, b_is_error=False)
            judgments.append(judgment)
            progress.update(1)
        progress.close()

        if not judgments:
            combined = pages[0].strip()
            return self._write_segments(pdf_path, [combined] if combined else [])

        label_values = [1 if judgment.label == "1" else 0 for judgment in judgments]
        segment_indices: List[List[int]] = []
        current: List[int] = [0]
        for idx, label in enumerate(label_values):
            if label == 1:
                current.append(idx + 1)
            else:
                segment_indices.append(current)
                current = [idx + 1]
        segment_indices.append(current)

        segments: List[str] = []
        for indices in segment_indices:
            combined = "\n\n".join(pages[idx].strip() for idx in indices if pages[idx].strip())
            if combined and len(combined) >= self.min_segment_chars:
                segments.append(combined)

        if not segments:
            logger.warning("LLM chunking produced no usable segments for %s", pdf_path.name)
            return []

        logger.info("Produced %d segments for %s", len(segments), pdf_path.name)
        return self._write_segments(pdf_path, segments)

    def _write_segments(self, pdf_path: Path, segments: List[str]) -> List[Dict[str, Any]]:
        letters: List[Dict[str, Any]] = []
        if not segments:
            return letters
        bundle_dir = pdf_path.parent / f"{pdf_path.stem}_letters"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        for idx, segment in enumerate(segments, start=1):
            cleaned = segment.strip()
            if len(cleaned) < self.min_segment_chars:
                continue
            submitter = self._infer_submitter(cleaned)
            submitted_at = self._infer_date(cleaned)
            letter_path = bundle_dir / f"{pdf_path.stem}_letter_{idx}.txt"
            letter_path.write_text(cleaned, encoding="utf-8")
            letters.append(
                {
                    "submitter_name": submitter,
                    "submitted_at": submitted_at,
                    "text_path": str(letter_path),
                }
            )
        return letters

    @staticmethod
    def _infer_submitter(text: str) -> Optional[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        first_line = lines[0]
        if first_line.lower().startswith("re:") and len(lines) > 1:
            return lines[1]
        return first_line

    @staticmethod
    def _infer_date(text: str) -> Optional[str]:
        match = re.search(
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+20\d{2}",
            text,
        )
        return match.group(0) if match else None
