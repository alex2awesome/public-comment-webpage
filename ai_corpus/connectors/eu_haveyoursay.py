"""
Connector for EU Commission \"Have Your Say\" consultation feedback.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from bs4 import BeautifulSoup  # type: ignore
from playwright.sync_api import sync_playwright

from ai_corpus.connectors.base import Collection, DocMeta, docmeta_to_rule_row
from ai_corpus.utils.http import RateLimiter, backoff_get, get_http_session, next_user_agent


class EuHaveYourSayConnector:
    name = "eu_have_your_say"

    def __init__(
        self,
        config: Dict,
        global_config: Optional[Dict] = None,
        session=None,
    ) -> None:
        self.config = config or {}
        self.global_config = global_config or {}
        self.session = session or get_http_session(
            {"User-Agent": self.global_config.get("user_agent")}
        )
        self.rate_limiter = RateLimiter(min_interval=0.5)
        self.initiatives = self.config.get("initiatives", [])
        self.max_workers = int(self.config.get("max_workers", 2))
        self.headless = bool(self.config.get("headless", True))

    def discover(self, **_) -> Iterable[Collection]:
        for initiative in self.initiatives:
            initiative_id = initiative.get("id")
            url = initiative.get("url")
            if not initiative_id or not url:
                continue
            yield Collection(
                source=self.name,
                collection_id=initiative_id,
                title=f"EU Feedback: {initiative_id}",
                url=url,
                jurisdiction="EU",
                topic="AI",
                extra={"initiative": initiative},
            )

    def list_documents(
        self,
        collection_id: str,
        max_pages: Optional[int] = None,
        **_,
    ) -> Iterable[DocMeta]:
        initiative = next(
            (it for it in self.initiatives if it.get("id") == collection_id),
            None,
        )
        if not initiative:
            return []
        base_url = initiative.get("url")
        p_id = initiative.get("p_id")
        page_param = initiative.get("pagination", {}).get("param", "page")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            page = browser.new_page()
            for page_num in itertools.count(0):
                if max_pages is not None and page_num >= max_pages:
                    break
                
                # Construct the URL with query parameters for the current page
                params = {page_param: page_num, "p_id": p_id} if page_num > 0 else {"p_id": p_id}
                url = base_url
                if params:
                    url += "?" + "&".join([f"{k}={v}" for k, v in params.items()])

                page.goto(url)
                
                # Wait for the feedback cards to be rendered
                try:
                    page.wait_for_selector("div.br-feedback-card", timeout=10000)
                except:
                    # If the selector doesn't appear, we assume there are no more pages
                    break

                content = page.content()
                soup = BeautifulSoup(content, "html.parser")
                cards = soup.select("div.br-feedback-card")

                if not cards:
                    break

                for card in cards:
                    meta = self._card_to_docmeta(collection_id, card)
                    if meta:
                        yield meta
            browser.close()

    def get_call_document(self, collection_id: str, **_) -> Optional[DocMeta]:
        initiative = next(
            (it for it in self.initiatives if it.get("id") == collection_id),
            None,
        )
        if not initiative:
            return None
        url = initiative.get("url")
        if not url:
            return None
        title = initiative.get("title") or f"EU Feedback Call {collection_id}"
        return DocMeta(
            source=self.name,
            collection_id=collection_id,
            doc_id=f"initiative:{collection_id}",
            title=title,
            submitter="European Commission",
            submitter_type="agency",
            org=initiative.get("directorate") or "European Commission",
            submitted_at=initiative.get("open_date"),
            language=initiative.get("language") or "en",
            urls={"html": url},
            extra={"document_role": "call", "initiative": initiative},
            kind="call",
        )

    def _card_to_docmeta(self, collection_id: str, card) -> Optional[DocMeta]:
        link = card.select_one("a.br-card__link")
        href = link["href"] if link and link.has_attr("href") else None
        if not href:
            return None
        doc_id = href.rstrip("/").rsplit("/", 1)[-1]
        submitter = self._get_text(card, "span.br-entity-name")
        submitter_type = self._get_text(card, "span.br-entity-type")
        country = self._get_text(card, "span.br-country-name")
        language = self._get_text(card, "div.br-language-label")
        date = None
        time_el = card.select_one("time")
        if time_el and time_el.has_attr("datetime"):
            date = time_el["datetime"]
        snippet = self._get_text(card, "div.br-feedback-entry__text")
        urls = {"html": href if href.startswith("http") else f"https://ec.europa.eu{href}"}
        return DocMeta(
            source=self.name,
            collection_id=collection_id,
            doc_id=doc_id,
            title=submitter or f"Feedback {doc_id}",
            submitter=submitter,
            submitter_type=submitter_type,
            org=submitter,
            submitted_at=date,
            language=language,
            urls=urls,
            extra={
                "country": country,
                "snippet": snippet,
            },
        )

    def _get_text(self, root, selector: str) -> Optional[str]:
        node = root.select_one(selector)
        if not node:
            return None
        text = node.get_text(strip=True)
        return text or None

    def fetch(self, doc: DocMeta, out_dir: Path, **kwargs) -> Dict[str, object]:
        html_url = doc.urls.get("html")
        result: Dict[str, object] = {"doc_id": doc.doc_id}
        if not html_url:
            return result
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / f"{doc.doc_id}.html"
        skip_existing = kwargs.get("skip_existing", True)
        if skip_existing and html_path.exists() and html_path.stat().st_size > 0:
            html_text = html_path.read_text(encoding="utf-8")
            result["html"] = str(html_path)
        else:
            response = backoff_get(
                html_url,
                session=self.session,
                rate_limiter=self.rate_limiter,
                headers={"User-Agent": next_user_agent()},
                raise_for_status=False,
            )
            if response is None or response.status_code != 200:
                return result
            html_path.write_text(response.text, encoding="utf-8")
            result["html"] = str(html_path)
            html_text = response.text
        attachments = self._download_attachments(html_text, doc, out_dir, skip_existing=skip_existing)
        if attachments:
            result["attachments"] = attachments
        return result

    def iter_rule_versions(self, collection_id: str, **_) -> Iterable[Dict[str, Any]]:
        call_doc = self.get_call_document(collection_id)
        if not call_doc:
            return []
        return [docmeta_to_rule_row(self.name, call_doc, history_rank=1)]

    def _download_attachments(self, html: str, doc: DocMeta, out_dir: Path, *, skip_existing: bool = True) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        stored: List[str] = []
        for idx, anchor in enumerate(soup.select("a[href]"), start=1):
            href = anchor.get("href") or ""
            if not href.lower().endswith((".pdf", ".docx", ".zip")):
                continue
            url = href if href.startswith("http") else f"https://ec.europa.eu{href}"
            suffix = Path(url).suffix or ".bin"
            path = out_dir / f"{doc.doc_id}_attachment_{idx}{suffix}"
            if skip_existing and path.exists() and path.stat().st_size > 0:
                stored.append(str(path))
                continue
            response = backoff_get(
                url,
                session=self.session,
                rate_limiter=self.rate_limiter,
                headers={"User-Agent": next_user_agent()},
                raise_for_status=False,
            )
            if response is None or response.status_code != 200:
                continue
            path.write_bytes(response.content)
            stored.append(str(path))
        return stored


__all__ = ["EuHaveYourSayConnector"]
