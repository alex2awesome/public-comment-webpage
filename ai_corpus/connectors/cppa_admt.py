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
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse, urlunparse

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

GOVERNMENT_DOC_KINDS = {kind for kind in CPPA_HISTORY_ORDER if kind != "aggregated_comments"}

SECTION_LABELS = {
    "CRP": "completed",
    "PR": "proposed",
    "PRA": "preliminary",
}

MONTH_PATTERN = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
DATE_PATTERNS = [
    re.compile(rf"({MONTH_PATTERN})\s+\d{{1,2}},\s*\d{{4}}", re.IGNORECASE),
    re.compile(rf"({MONTH_PATTERN})\s+\d{{4}}", re.IGNORECASE),
]
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

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


@dataclass(slots=True)
class DetailTarget:
    url: str
    slug: str
    title: str
    status: str
    date_text: Optional[str] = None
    date_value: Optional[date] = None


@dataclass(slots=True)
class DocketRecord:
    collection_id: str
    slug: str
    title: str
    url: str
    status: str
    posted_date_text: Optional[str]
    posted_date: Optional[date]
    entries: List[PdfEntry]


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
        self.listing_url = self.config.get("listing_url") or self.config.get("index_url")
        self.base_url = self.config.get("base_url", "https://cppa.ca.gov").rstrip("/")
        self.openai_model = self.config.get("openai_model", "gpt-4o")
        self.openai_max_output_tokens = int(self.config.get("openai_max_output_tokens", 96))
        self.parser = self.config.get("parser", "auto")
        self.min_segment_chars = int(self.config.get("min_segment_chars", 400))

        self._collections: Optional[Dict[str, DocketRecord]] = None
        self._openai_client: Any | None = None
        self._openai_initialized = False

    # ------------------------------------------------------------------ discovery
    def discover(self, start_date: Optional[str] = None, end_date: Optional[str] = None, **_) -> Iterable[Collection]:
        """
        Enumerate available CPPA dockets by scraping the agency's regulation
        index. The page lists proposed and completed packages, each with its
        own detail page where PDFs (notices, comment bundles, etc.) live.
        """

        records = self._load_collections()
        if not records:
            return []

        start = self._parse_filter_date(start_date)
        end = self._parse_filter_date(end_date)

        def _sort_key(record: DocketRecord) -> tuple:
            rank = record.posted_date or date.min
            return (rank, record.collection_id)

        results: List[Collection] = []
        for record in sorted(records.values(), key=_sort_key, reverse=True):
            posted = record.posted_date
            if (start or end) and not posted:
                logger.debug(
                    "Skipping %s because no posted date is available while filtering by date range.",
                    record.collection_id,
                )
                continue
            if start and posted and posted < start:
                continue
            if end and posted and posted > end:
                continue
            extra = {
                "status": record.status,
                "slug": record.slug,
                "posted_date": posted.isoformat() if posted else record.posted_date_text,
                "total_pdfs": len(record.entries),
            }
            extra = {k: v for k, v in extra.items() if v is not None}
            results.append(
                Collection(
                    source=self.name,
                    collection_id=record.collection_id,
                    title=record.title,
                    url=record.url,
                    jurisdiction="US-CA",
                    topic=None,
                    extra=extra,
                )
            )
        return results

    # -------------------------------------------------------------- document list
    def list_documents(self, collection_id: str, **_) -> Iterable[DocMeta]:
        """
        Return metadata for every PDF listed on the docket detail page. The CPPA
        posts both aggregated public comments and government-authored updates
        (notices, modified text, comment summaries, hearing transcripts, etc.).
        Aggregated bundles are marked as responses and later split into letters,
        while agency PDFs are tagged as call documents so the pipeline
        downloads them via the `download-call` stage by default.
        """

        record = self._load_collections().get(collection_id)
        if not record:
            logger.warning("Unknown collection_id '%s' for CPPA connector", collection_id)
            return []

        for entry in record.entries:
            doc_kind = "call" if entry.kind in GOVERNMENT_DOC_KINDS else "response"
            is_bundle = entry.kind == "aggregated_comments"
            if is_bundle:
                doc_kind = "response"
            submitter = None
            submitter_type = None
            org = None
            extra = {
                "kind": entry.kind,
                "raw_href": entry.raw_href,
            }
            if is_bundle:
                extra["is_bundle"] = True
            else:
                extra["document_role"] = "call"
                extra["government_document"] = True
                submitter = "CPPA"
                submitter_type = "agency"
                org = "California Privacy Protection Agency"
            yield DocMeta(
                source=self.name,
                collection_id=collection_id,
                doc_id=entry.filename,
                title=entry.title or entry.filename,
                submitter=submitter,
                submitter_type=submitter_type,
                org=org,
                submitted_at=None,
                language="en",
                urls={"pdf": entry.url},
                extra=extra,
                kind=doc_kind,
            )

    def get_call_document(self, collection_id: str, **_) -> Optional[DocMeta]:
        record = self._load_collections().get(collection_id)
        if not record:
            return None
        entries = record.entries
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
        record = self._load_collections().get(collection_id)
        if not record:
            return []
        entries = list(record.entries)
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
    def _load_collections(self) -> Dict[str, DocketRecord]:
        if self._collections is not None:
            return self._collections

        targets = self._discover_detail_targets()
        if not targets and self.listing_url:
            logger.info(
                "Falling back to single-detail mode for CPPA connector using %s",
                self.listing_url,
            )
            parsed = urlparse(self.listing_url)
            slug = Path(parsed.path).stem or "cppa"
            targets = [
                DetailTarget(
                    url=self.listing_url,
                    slug=slug,
                    title="CPPA Regulations",
                    status="unknown",
                )
            ]

        records: Dict[str, DocketRecord] = {}
        for target in targets:
            html = self._fetch_html(target.url)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            collection_id = self._infer_collection_id(soup, target.slug, records.keys())
            title_node = soup.find("h1")
            title_text = title_node.get_text(strip=True) if title_node else target.title
            entries = self._extract_pdf_entries(soup, target.url)
            records[collection_id] = DocketRecord(
                collection_id=collection_id,
                slug=target.slug,
                title=title_text or collection_id,
                url=target.url,
                status=target.status,
                posted_date_text=target.date_text,
                posted_date=target.date_value,
                entries=entries,
            )

        self._collections = records
        return self._collections

    def _discover_detail_targets(self) -> List[DetailTarget]:
        if not self.listing_url:
            logger.warning("CPPA connector requires a listing_url or index_url configuration.")
            return []
        html = self._fetch_html(self.listing_url)
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        main = soup.select_one("#main-content") or soup
        dedup: Dict[str, DetailTarget] = {}
        for anchor in main.find_all("a", href=True):
            href = (anchor.get("href") or "").strip()
            if not href or not href.endswith(".html"):
                continue
            if "/regulations/" not in href:
                continue
            detail_url = urljoin(self.base_url + "/", href)
            parsed = urlparse(detail_url)
            slug = Path(parsed.path).stem or "cppa"
            section_label = self._resolve_section_label(anchor)
            context_text = anchor.parent.get_text(" ", strip=True) if anchor.parent else anchor.get_text(strip=True)
            date_text, date_value = self._extract_date_from_text(context_text)
            target = DetailTarget(
                url=detail_url,
                slug=slug,
                title=anchor.get_text(strip=True),
                status=section_label,
                date_text=date_text,
                date_value=date_value,
            )
            existing = dedup.get(detail_url)
            if existing is None or (
                target.date_value and (not existing.date_value or target.date_value > existing.date_value)
            ):
                dedup[detail_url] = target
        return list(dedup.values())

    def _fetch_html(self, url: str) -> Optional[str]:
        response = backoff_get(
            url,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": self.global_config.get("user_agent")},
            raise_for_status=False,
        )
        if response is None or response.status_code != 200:
            logger.warning("Failed to load CPPA page from %s", url)
            return None
        return response.text

    def _infer_collection_id(
        self,
        soup: BeautifulSoup,
        slug: str,
        existing_ids: Iterable[str],
    ) -> str:
        text = soup.get_text(" ", strip=True)
        html_text = soup.decode()
        match = re.search(r"PR-\d{2}-\d{4}", text)
        if not match:
            match = re.search(r"PR-\d{2}-\d{4}", html_text)
        if match:
            base_id = match.group(0)
        else:
            alt = re.search(r"pr[\-_](\d{2})-(\d{4})", text, re.IGNORECASE) or re.search(
                r"pr[\-_](\d{2})-(\d{4})", html_text, re.IGNORECASE
            )
            if alt:
                base_id = f"PR-{alt.group(1)}-{alt.group(2)}"
            else:
                base_id = slug.replace("-", "_").upper() or "CPPA_DOCKET"
        candidate = base_id
        counter = 2
        existing = set(existing_ids)
        while candidate in existing:
            candidate = f"{base_id}__{counter}"
            counter += 1
        return candidate

    def _extract_pdf_entries(self, soup: BeautifulSoup, page_url: str) -> List[PdfEntry]:
        entries: List[PdfEntry] = []
        seen: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            raw_href = (anchor.get("href") or "").strip()
            normalized_href = self._normalize_pdf_href(raw_href)
            if not normalized_href:
                continue
            full_url = urljoin(page_url, normalized_href)
            if full_url in seen:
                continue
            seen.add(full_url)
            parsed = urlparse(full_url)
            filename = self._sanitize_filename(Path(parsed.path).name or "document.pdf")
            kind = self._classify_pdf(anchor.text, raw_href)
            entries.append(
                PdfEntry(
                    title=(anchor.text or filename).strip(),
                    url=full_url,
                    filename=filename,
                    kind=kind,
                    raw_href=raw_href,
                )
            )
        return entries

    def _normalize_pdf_href(self, href: str) -> Optional[str]:
        if not href or ".pdf" not in href.lower():
            return None
        parsed = urlparse(href)
        path = parsed.path or ""
        if ".pdf" not in path.lower():
            return None
        cutoff = path.lower().find(".pdf") + 4
        path = path[:cutoff]
        cleaned = parsed._replace(path=path, fragment="")
        return urlunparse(cleaned)

    def _resolve_section_label(self, anchor) -> str:
        node = anchor
        while node is not None:
            node_id = getattr(node, "get", lambda *_: None)("id") if hasattr(node, "get") else None
            if node_id in SECTION_LABELS:
                return SECTION_LABELS[node_id]
            node = getattr(node, "parent", None)
        return "unknown"

    def _extract_date_from_text(self, text: Optional[str]) -> tuple[Optional[str], Optional[date]]:
        if not text:
            return None, None
        cleaned = text.replace("(", " ").replace(")", " ")
        for pattern in DATE_PATTERNS:
            match = pattern.search(cleaned)
            if match:
                value = match.group(0)
                return value, self._parse_date_string(value)
        match = YEAR_PATTERN.search(cleaned)
        if match:
            value = match.group(0)
            return value, self._parse_date_string(value)
        return None, None

    def _parse_filter_date(self, raw: Optional[str]) -> Optional[date]:
        if not raw:
            return None
        candidate = raw.strip()
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                if fmt == "%Y-%m":
                    value = datetime.strptime(f"{candidate}-01", "%Y-%m-%d")
                elif fmt == "%Y":
                    value = datetime.strptime(f"{candidate}-01-01", "%Y-%m-%d")
                else:
                    value = datetime.strptime(candidate, "%Y-%m-%d")
                return value.date()
            except ValueError:
                continue
        return None

    def _parse_date_string(self, text_value: str) -> Optional[date]:
        if not text_value:
            return None
        cleaned = re.sub(r"\s+", " ", text_value.strip())
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %Y", "%b %Y"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                if "%d" not in fmt:
                    return date(dt.year, dt.month, 1)
                return dt.date()
            except ValueError:
                continue
        if re.fullmatch(r"\d{4}", cleaned):
            return date(int(cleaned), 1, 1)
        return None

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
