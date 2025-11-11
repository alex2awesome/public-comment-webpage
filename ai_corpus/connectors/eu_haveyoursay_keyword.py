"""
Connector that drives the EU \"Have Your Say\" interface via Playwright to
collect feedback metadata for initiatives matched by a keyword search.

This module preserves the exploratory logic from
`scratch/playwright_eu_keyword.py` while adapting it to the standardized
connector interface used by the rest of the pipeline.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

from ai_corpus.connectors.base import Collection, DocMeta, docmeta_to_rule_row

logger = logging.getLogger(__name__)

INITIATIVES_URL = "https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives_en"


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing special characters, extra
    whitespace, and converting to lowercase.
    """

    cleaned = re.sub(r"[^\w\s]", " ", text)
    return " ".join(cleaned.split()).lower()


def ensure_browser_cache() -> None:
    """
    Ensure Playwright stores browser binaries within the repository so system
    caches are not required.
    """

    if "PLAYWRIGHT_BROWSERS_PATH" in os.environ:
        return

    module_path = Path(__file__).resolve()
    cache_dir: Optional[Path] = None
    for parent in module_path.parents:
        candidate = parent / ".playwright"
        if candidate.exists() or (parent / ".git").exists():
            cache_dir = candidate
            break
    if cache_dir is None:
        cache_dir = module_path.parent / ".playwright"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(cache_dir)


def extract_metadata_table(page: Page, *, timeout: int = 5000) -> List[Tuple[str, str]]:
    """Parse the feedback summary description list into `(field, value)` pairs."""

    details_locator = page.locator("dl.ecl-description-list")
    rows: List[Tuple[str, str]] = []

    try:
        details = details_locator.first
        details.wait_for(state="attached", timeout=timeout)
    except Exception:  # noqa: BLE001 - Playwright raises several custom exceptions.
        return rows

    terms = details.locator("dt")
    definitions = details.locator("dd")
    count = min(terms.count(), definitions.count())

    for idx in range(count):
        term_text = (terms.nth(idx).inner_text() or "").strip()
        value_text = (definitions.nth(idx).inner_text() or "").strip()
        if not term_text and not value_text:
            continue
        rows.append((term_text, value_text))

    return rows


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^\w.-]+", "_", name.strip())
    return sanitized or "file"


@dataclass(slots=True)
class InitiativeContext:
    keyword: str
    matched_text: str
    initiative_url: str
    initiative_path_fragment: str
    feedback_links: List[str]
    collection: Collection


class EuHaveYourSayKeywordConnector:
    name = "eu_have_your_say_keyword"

    def __init__(
        self,
        config: Dict[str, Any],
        global_config: Optional[Dict[str, Any]] = None,
        session=None,
    ) -> None:
        self.config = config or {}
        self.global_config = global_config or {}

        self.debug = bool(self.config.get("debug", False))
        if "headless" in self.config:
            self.headless = bool(self.config.get("headless"))
        else:
            self.headless = not self.debug

        default_slow_mo = 250 if self.debug else 0
        self.slow_mo = int(self.config.get("slow_mo", default_slow_mo))
        self.wait_timeout = int(self.config.get("wait_timeout", 60000))
        self.pause_on_complete = bool(self.config.get("pause_on_complete", self.debug))

        raw_keywords = self.config.get("keywords") or [self.config.get("keyword")]
        self.keywords = [kw for kw in raw_keywords if isinstance(kw, str) and kw.strip()]
        if not self.keywords:
            raise ValueError("At least one keyword must be provided for EU Have Your Say scraping.")

        self.topic = self.config.get("topic") or "AI"
        cache_root_default = Path(__file__).resolve().parents[1] / "downloads" / "eu_have_your_say_cache"
        self.cache_root = Path(self.config.get("cache_dir", cache_root_default)).expanduser().resolve()

        self._keyword_cache: Dict[str, InitiativeContext] = {}
        self._collection_cache: Dict[str, InitiativeContext] = {}
        self._documents_cache: Dict[str, List[DocMeta]] = {}

    def discover(self, **_) -> Iterable[Collection]:
        """
        Identify initiatives for the configured keywords. Discovery is limited to
        collecting the initiative metadata and the associated "All feedback"
        listing URLs; no individual comment pages are visited here.
        """

        for keyword in self.keywords:
            context = self._get_initiative_context(keyword)
            if context is None:
                continue
            yield context.collection

    def list_documents(self, collection_id: str, **_) -> Iterable[DocMeta]:
        """
        Visit each feedback listing for the target initiative and enumerate the
        individual feedback entries. This step loads every feedback page to
        extract metadata but defers binary downloads to `fetch`.
        """

        context = self._collection_cache.get(collection_id)
        if context is None:
            context = self._refresh_context_for_collection(collection_id)
        if context is None:
            logger.warning("Unable to find initiative metadata for collection %s", collection_id)
            return []

        documents = self._harvest_documents_for_context(context)
        for doc in documents:
            yield doc

    def get_call_document(self, collection_id: str, **_) -> Optional[DocMeta]:
        context = self._collection_cache.get(collection_id)
        if context is None:
            context = self._refresh_context_for_collection(collection_id)
        if context is None:
            return None
        collection = context.collection
        return DocMeta(
            source=self.name,
            collection_id=collection.collection_id,
            doc_id=f"initiative:{collection.collection_id}",
            title=collection.title,
            submitter="European Commission",
            submitter_type="agency",
            org="European Commission",
            submitted_at=None,
            language="en",
            urls={"html": context.initiative_url},
            extra={
                "keyword": context.keyword,
                "document_role": "call",
                "matched_text": context.matched_text,
            },
            kind="call",
        )

    def fetch(self, doc: DocMeta, out_dir: Path, **kwargs) -> Dict[str, Any]:
        result: Dict[str, Any] = {"doc_id": doc.doc_id}
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / f"{doc.doc_id}.html"
        skip_existing = bool(kwargs.get("skip_existing", True))

        html_exists = html_path.exists() and html_path.stat().st_size > 0
        needs_html = not (skip_existing and html_exists)

        extra = doc.extra if isinstance(doc.extra, dict) else {}
        attachment_expected = bool(extra.get("attachment_present"))
        expected_attachment_name = extra.get("expected_attachment_name")
        attachments_dir = out_dir
        attachment_path = attachments_dir / expected_attachment_name if expected_attachment_name else None
        attachment_exists = (
            attachment_path is not None
            and attachment_path.exists()
            and attachment_path.stat().st_size > 0
        )
        needs_attachments = not (skip_existing and attachment_exists)

        cached_html_path = Path(extra["cached_html_path"]).expanduser() if extra.get("cached_html_path") else None
        cached_attachment_path = Path(extra["cached_attachment_path"]).expanduser() if extra.get("cached_attachment_path") else None

        if needs_html:
            if cached_html_path and cached_html_path.exists():
                html_path.write_text(cached_html_path.read_text(encoding="utf-8"), encoding="utf-8")
            elif cached_html_path:
                logger.warning("Cached HTML not found at %s for %s", cached_html_path, doc.doc_id)

        if not html_path.exists() and cached_html_path and cached_html_path.exists():
            html_path.write_text(cached_html_path.read_text(encoding="utf-8"), encoding="utf-8")

        if html_path.exists():
            result["html"] = str(html_path)
        elif cached_html_path and cached_html_path.exists():
            result["html"] = str(cached_html_path)

        attachments: List[str] = []
        if needs_attachments and cached_attachment_path and cached_attachment_path.exists():
            if attachment_path is None:
                attachment_path = out_dir / cached_attachment_path.name
            attachment_path.write_bytes(cached_attachment_path.read_bytes())
            attachments.append(str(attachment_path))
        elif attachment_exists and attachment_path:
            attachments.append(str(attachment_path))

        if attachments:
            result["attachments"] = attachments

        return result

    def _get_initiative_context(self, keyword: str, *, force_refresh: bool = False) -> Optional[InitiativeContext]:
        if not force_refresh:
            cached = self._keyword_cache.get(keyword)
            if cached is not None:
                return cached

        context = self._discover_initiative(keyword)
        if context is None:
            return None
        self._keyword_cache[keyword] = context
        self._collection_cache[context.collection.collection_id] = context
        return context

    def _refresh_context_for_collection(self, collection_id: str) -> Optional[InitiativeContext]:
        for keyword in self.keywords:
            context = self._keyword_cache.get(keyword)
            if context and context.collection.collection_id == collection_id:
                return context

        for keyword in self.keywords:
            context = self._get_initiative_context(keyword, force_refresh=True)
            if context and context.collection.collection_id == collection_id:
                return context
        return None

    def _discover_initiative(self, keyword: str) -> Optional[InitiativeContext]:
        logger.info("Discovering initiative for keyword: %s", keyword)
        ensure_browser_cache()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self.headless, slow_mo=self.slow_mo)
            page = browser.new_page()
            try:
                page.goto(INITIATIVES_URL, wait_until="domcontentloaded", timeout=self.wait_timeout)
                self._wait_for_network_idle(page)
                input_box = page.wait_for_selector("#facet-keyword", timeout=self.wait_timeout)
                input_box.click()
                input_box.fill(keyword)
                logger.debug("Filled keyword filter with: %s", keyword)

                page.wait_for_timeout(2000)

                result_locator = page.locator("div.ecl-link.ecl-link--standalone")
                try:
                    result_locator.first.wait_for(state="attached", timeout=self.wait_timeout)
                except Exception:  # noqa: BLE001
                    logger.warning("No results rendered for keyword '%s'", keyword)
                    return None

                matched_link = self._match_result_link(result_locator, keyword)
                if matched_link is None:
                    logger.warning("No matching initiative found for keyword '%s'", keyword)
                    return None

                matched_text = matched_link.text_content() or keyword
                logger.info("Matched initiative '%s' for keyword '%s'", matched_text.strip(), keyword)
                matched_link.click()
                self._wait_for_network_idle(page)
                time.sleep(2)

                # Ensure dynamic sections load before we look for feedback links.
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1500)
                time.sleep(2)
                page.evaluate("window.scrollTo(0, 0)")
                page.wait_for_timeout(500)
                time.sleep(1)

                initiative_url = page.url
                feedback_links = self._extract_feedback_links(page)

                initiative_path_fragment = ""
                if initiative_url:
                    initiative_path_fragment = "/".join(
                        urlparse(initiative_url).path.rstrip("/").split("/")[:-1]
                    )

                collection_id_source = initiative_path_fragment or matched_text or keyword
                collection_id = sanitize_filename(collection_id_source) or sanitize_filename(keyword)
                collection = Collection(
                    source=self.name,
                    collection_id=collection_id,
                    title=(matched_text or keyword).strip(),
                    url=initiative_url,
                    jurisdiction="EU",
                    topic=self.topic,
                    extra={
                        "keyword": keyword,
                        "search_url": INITIATIVES_URL,
                        "matched_text": matched_text.strip(),
                        "feedback_links": feedback_links,
                        "initiative_path_fragment": initiative_path_fragment,
                    },
                )

                context = InitiativeContext(
                    keyword=keyword,
                    matched_text=matched_text.strip(),
                    initiative_url=initiative_url,
                    initiative_path_fragment=initiative_path_fragment,
                    feedback_links=feedback_links,
                    collection=collection,
                )
                logger.info(
                    "Discovered initiative '%s' with %d feedback listing(s)",
                    collection.title,
                    len(feedback_links),
                )
                return context
            finally:
                if self.pause_on_complete and not self.headless:
                    try:
                        page.pause()
                    except Exception:  # noqa: BLE001
                        logger.exception("Playwright pause failed during discovery; continuing.")
                browser.close()

    def _extract_feedback_links(self, page: Page) -> List[str]:
        links: List[str] = []
        try:
            feedback_links = page.get_by_role("link", name=re.compile(r"All feedback", re.IGNORECASE)).all()
            if not feedback_links:
                logger.warning("No 'All feedback' links found on initiative page %s", page.url)
                return links
            parts = urlparse(page.url)
            base_url = f"{parts.scheme}://{parts.netloc}"
            for link in feedback_links:
                href = link.get_attribute("href")
                if not href:
                    continue
                links.append(urljoin(base_url, href))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error extracting 'All feedback' links on %s: %s", page.url, exc)
        return links

    def _harvest_documents_for_context(self, context: InitiativeContext) -> List[DocMeta]:
        cached_docs = self._documents_cache.get(context.collection.collection_id)
        if cached_docs and not self.debug:
            return cached_docs
        ensure_browser_cache()
        documents: List[DocMeta] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self.headless, slow_mo=self.slow_mo)
            page = browser.new_page()
            try:
                entries: List[Dict[str, Any]] = []
                for feedback_url in context.feedback_links:
                    entries.extend(
                        self._collect_feedback_entries_from_listing(
                            page,
                            feedback_url,
                            context.initiative_path_fragment,
                        )
                    )

                if not entries:
                    logger.warning(
                        "No feedback entries found for initiative '%s'",
                        context.collection.title,
                    )
                    return []

                documents = self._collect_entry_metadata_and_download(
                    page,
                    context.keyword,
                    context.collection.collection_id,
                    entries,
                    context,
                )
            finally:
                browser.close()
        self._documents_cache[context.collection.collection_id] = documents
        return documents

    def _collect_feedback_entries_from_listing(
        self,
        page: Page,
        feedback_url: str,
        initiative_path_fragment: str,
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        page.goto(feedback_url, wait_until="domcontentloaded", timeout=self.wait_timeout)
        self._wait_for_network_idle(page)
        logger.debug("Navigated to feedback listing: %s", feedback_url)
        time.sleep(5)

        page_number = 1
        while True:
            logger.debug("Collecting feedback entries (page %d)", page_number)
            entry_elements = page.locator("feedback-item").all()
            if not entry_elements:
                logger.debug("No feedback items found on page %d", page_number)
            for entry in entry_elements:
                href = None
                for candidate_href in entry.locator("a").all():
                    href = candidate_href.get_attribute("href")
                    if href:
                        break
                if not href:
                    continue
                if initiative_path_fragment and initiative_path_fragment not in href:
                    continue
                text = (entry.text_content() or "").strip()
                full_url = urljoin(page.url, href)
                entries.append(
                    {
                        "text": text,
                        "href": full_url,
                        "page": page_number,
                        "feedback_link": feedback_url,
                    }
                )

            next_button = page.locator(
                "ecl-pagination li.ecl-pagination__item--next a.ecl-pagination__link"
            )
            if next_button.count() == 0:
                logger.debug("No next page link; stopping pagination.")
                break

            aria_disabled = next_button.first.get_attribute("aria-disabled")
            if aria_disabled == "true":
                logger.debug("Next page disabled; reached final page.")
                break

            logger.debug("Advancing to next page of feedback entries.")
            next_button.first.click()
            self._wait_for_network_idle(page)
            page.wait_for_timeout(1000)
            time.sleep(1)
            page_number += 1

        return entries

    def _collect_entry_metadata_and_download(
        self,
        page: Page,
        keyword: str,
        collection_id: str,
        entries: List[Dict[str, Any]],
        context: InitiativeContext,
    ) -> List[DocMeta]:
        documents: List[DocMeta] = []
        cache_dir = self.cache_root / collection_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            page.goto(entry["href"], wait_until="domcontentloaded", timeout=self.wait_timeout)
            self._wait_for_network_idle(page)
            time.sleep(5)

            metadata_rows = extract_metadata_table(page, timeout=self.wait_timeout)
            metadata_dict = dict(metadata_rows)

            download_locator = page.locator("a.ecl-file__download")
            has_attachment = download_locator.count() > 0
            reference = metadata_dict.get("Feedback reference")

            doc_identifier = reference or entry["href"].rstrip("/").rsplit("/", 1)[-1]
            sanitized_id = sanitize_filename(doc_identifier)

            title = (
                metadata_dict.get("Title")
                or metadata_dict.get("Feedback reference")
                or entry["text"]
                or sanitized_id
            )
            submitter = metadata_dict.get("Submitted by") or metadata_dict.get("Author")
            submitter_type = metadata_dict.get("Organisation type")
            org = metadata_dict.get("Organisation")
            submitted_at = metadata_dict.get("Submitted on") or metadata_dict.get("Submitted")
            language = metadata_dict.get("Language")

            expected_filename = f"{sanitized_id}.pdf"
            html_cache_path = cache_dir / f"{sanitized_id}.html"
            html_cache_path.write_text(page.content(), encoding="utf-8")

            cached_attachment_path: Optional[Path] = None
            if has_attachment:
                download_locator = page.locator("a.ecl-file__download")
                cached_attachment_path = cache_dir / f"{sanitized_id}.pdf"
                if not cached_attachment_path.exists() or cached_attachment_path.stat().st_size == 0:
                    try:
                        page.wait_for_timeout(2000)
                        with page.expect_download(timeout=self.wait_timeout) as download_info:
                            download_locator.first.click()
                        download = download_info.value
                        download.save_as(str(cached_attachment_path))
                    except Exception:  # noqa: BLE001
                        logger.exception("Failed to download attachment for %s", sanitized_id)
                        cached_attachment_path = None

            extra: Dict[str, Any] = {
                "keyword": keyword,
                "result_text": entry["text"],
                "feedback_listing": entry["feedback_link"],
                "page_number": entry["page"],
                "metadata_table": metadata_dict,
                "attachment_present": has_attachment,
                "expected_attachment_name": expected_filename,
                "cached_html_path": str(html_cache_path),
            }
            if cached_attachment_path:
                extra["cached_attachment_path"] = str(cached_attachment_path)

            documents.append(
                DocMeta(
                    source=self.name,
                    collection_id=collection_id,
                    doc_id=sanitized_id,
                    title=title,
                    submitter=submitter,
                    submitter_type=submitter_type,
                    org=org,
                    submitted_at=submitted_at,
                    language=language,
                    urls={"html": entry["href"]},
                    extra=extra,
                )
            )
        return documents

    def _match_result_link(self, result_locator, keyword: str):
        normalized_keyword = normalize_text(keyword)
        links = result_locator.all()
        logger.debug("Found %d initiative link candidates", len(links))

        for link in links:
            link_text = link.text_content()
            if link_text and normalize_text(link_text) == normalized_keyword:
                logger.debug("Found exact match for keyword '%s'", keyword)
                return link

        for link in links:
            link_text = link.text_content()
            if link_text:
                normalized_link = normalize_text(link_text)
                if normalized_keyword in normalized_link:
                    logger.debug("Found partial match for keyword '%s'", keyword)
                    return link

        for link in links:
            link_text = link.text_content()
            if link_text:
                normalized_link = normalize_text(link_text)
                if normalized_link in normalized_keyword:
                    logger.debug("Found reverse partial match for keyword '%s'", keyword)
                    return link
        return None

    def _wait_for_network_idle(
        self,
        page: Page,
        *,
        timeout: Optional[int] = None,
        fallback_delay_ms: int = 2000,
    ) -> None:
        """Wait for network to settle with graceful degradation on timeout."""

        effective_timeout = timeout or self.wait_timeout
        try:
            page.wait_for_load_state("networkidle", timeout=effective_timeout)
        except PlaywrightTimeoutError:
            logger.warning(
                "Timed out waiting for network idle (%sms); continuing after fallback sleep.",
                effective_timeout,
            )
            page.wait_for_timeout(fallback_delay_ms)

    def iter_rule_versions(self, collection_id: str, **_) -> Iterable[Dict[str, Any]]:
        call_doc = self.get_call_document(collection_id)
        if not call_doc:
            return []
        return [docmeta_to_rule_row(self.name, call_doc, history_rank=1)]


__all__ = ["EuHaveYourSayKeywordConnector", "InitiativeContext"]
