"""
Connector for the Washington State Register issue pages.

- Issue index: https://lawfilesext.leg.wa.gov/law/wsr/WsrByIssue.htm
- Issue filings (HTML/PDF): https://lawfilesext.leg.wa.gov/law/wsr/<year>/<issue>/<issue>.htm
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ai_corpus.connectors.base import BaseConnector, Collection, DocMeta
from ai_corpus.utils.http import RateLimiter, next_user_agent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IssueLink:
    issue_id: str
    url: str


class WashingtonRegisterConnector(BaseConnector):
    name = "washington_register"

    def __init__(self, config: Dict, global_config: Optional[Dict] = None) -> None:
        self.config = config or {}
        self.global_config = global_config or {}
        self.index_url = self.config.get("index_url", "https://lawfilesext.leg.wa.gov/law/wsr/WsrByIssue.htm")
        self.max_issues = int(self.config.get("max_issues", 5))
        self.rate_limiter = RateLimiter(min_interval=float(self.config.get("rate_limit_seconds", 0.25)))

    # --------------------------------------------------------------------- discovery
    def discover(self, **_) -> Iterable[Collection]:
        issues = self._fetch_issue_links()
        for link in issues[: self.max_issues]:
            year = link.issue_id.split("-")[0]
            yield Collection(
                source=self.name,
                collection_id=link.issue_id,
                title=f"Washington State Register {link.issue_id}",
                url=link.url,
                jurisdiction="US-WA",
                topic="State Rulemaking",
                extra={"year": year},
            )

    # ----------------------------------------------------------------- list documents
    def list_documents(self, collection_id: str, **_) -> Iterable[DocMeta]:
        html = self._get_html(self._issue_url(collection_id))
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        table = None
        for candidate in soup.find_all("table"):
            headers = [th.get_text() for th in candidate.find_all("th")]
            if any("Agency" in (header or "") for header in headers):
                table = candidate
                break
        docs: List[DocMeta] = []
        if not table:
            return docs
        rows = table.find_all("tr")[2:]  # skip header rows
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 5:
                continue
            agency = cells[1].get_text(" ", strip=True)
            html_link = cells[2].find("a", href=True)
            pdf_link = cells[3].find("a", href=True)
            filing_type = cells[4].get_text(strip=True)
            if not html_link:
                continue
            doc_id = html_link.get_text(strip=True)
            docs.append(
                DocMeta(
                    source=self.name,
                    collection_id=collection_id,
                    doc_id=doc_id,
                    title=f"{agency} – {doc_id}",
                    submitter=agency,
                    submitter_type="agency",
                    org=agency,
                    submitted_at=None,
                    language="en",
                    urls={
                        "html": html_link["href"],
                        "pdf": pdf_link["href"] if pdf_link else "",
                    },
                    extra={"filing_type": filing_type},
                    kind="notice",
                )
            )
        return docs

    # -------------------------------------------------------------------------- fetch
    def fetch(self, doc: DocMeta, out_dir: Path, **_) -> Dict[str, str]:
        pdf_url = doc.urls.get("pdf") or ""
        target_url = pdf_url or doc.urls.get("html")
        if not target_url:
            raise ValueError("Document missing download URL")
        content = self._download_file(target_url)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".pdf" if target_url.lower().endswith(".pdf") else ".html"
        out_path = out_dir / f"{doc.doc_id}{suffix}"
        if suffix == ".pdf":
            out_path.write_bytes(content)
            return {"pdf": str(out_path)}
        out_path.write_text(content.decode("utf-8", errors="ignore"), encoding="utf-8")
        return {"html": str(out_path)}

    # ---------------------------------------------------------------------- utilities
    def _fetch_issue_links(self) -> List[IssueLink]:
        html = self._get_html(self.index_url)
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        links: List[IssueLink] = []
        for anchor in soup.find_all("a", href=re.compile(r"/law/wsr/\d{4}/\d{2}/\d{2}-\d{2}\.htm", re.IGNORECASE)):
            href = anchor.get("href")
            issue_id = Path(href).stem
            links.append(
                IssueLink(
                    issue_id=issue_id,
                    url=urljoin(self.index_url, href),
                )
            )
        # Remove duplicates while preserving order
        seen = set()
        unique: List[IssueLink] = []
        for link in links:
            if link.issue_id in seen:
                continue
            seen.add(link.issue_id)
            unique.append(link)
        return unique

    def _issue_url(self, issue_id: str) -> str:
        parts = issue_id.split("-")
        if len(parts) >= 2:
            year = "20" + parts[0] if len(parts[0]) == 2 else parts[0]
            return f"https://lawfilesext.leg.wa.gov/law/wsr/{year}/{parts[1]}/{issue_id}.htm"
        return self.index_url

    def _get_html(self, url: str) -> Optional[str]:
        try:
            self.rate_limiter.wait()
            resp = requests.get(url, headers={"User-Agent": next_user_agent()}, timeout=60)
            if resp.status_code != 200:
                logger.warning("Washington Register GET %s failed: %s", url, resp.status_code)
                return None
            resp.encoding = resp.encoding or "utf-8"
            return resp.text
        except requests.RequestException as exc:
            logger.warning("Washington Register request error: %s", exc)
            return None

    def _download_file(self, url: str) -> bytes:
        try:
            self.rate_limiter.wait()
            resp = requests.get(url, headers={"User-Agent": next_user_agent()}, timeout=60)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc


__all__ = ["WashingtonRegisterConnector"]
