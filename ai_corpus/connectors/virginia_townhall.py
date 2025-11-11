"""
Connector for Virginia's Regulatory Town Hall public comment forums.

Documentation / references:
- Forums listing (HTML rendered grid): https://townhall.virginia.gov/L/Forums.cfm
- Stage comment listings: https://townhall.virginia.gov/L/comments.cfm?stageid=<id>
- Individual comment detail pages: https://townhall.virginia.gov/L/viewcomments.cfm?commentid=<id>
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from ai_corpus.connectors.base import BaseConnector, Collection, DocMeta
from ai_corpus.utils.http import RateLimiter, backoff_get, get_http_session, next_user_agent

logger = logging.getLogger(__name__)

DATE_RE = re.compile(r"Closes:\s*(?P<date>\d{1,2}/\d{1,2}/\d{2})", re.IGNORECASE)
COMMENTS_RE = re.compile(r"(?P<count>\d+)\s+comments?", re.IGNORECASE)


@dataclass(slots=True)
class ForumEntry:
    board: str
    chapter: str
    stage_id: str
    stage_label: str
    action_id: Optional[str]
    action_title: Optional[str]
    closes_on: Optional[datetime]
    comment_count: int


class VirginiaTownhallConnector(BaseConnector):
    name = "virginia_townhall"

    def __init__(
        self,
        config: Dict,
        global_config: Optional[Dict] = None,
        session=None,
    ) -> None:
        self.config = config or {}
        self.global_config = global_config or {}
        self.session = session or get_http_session(
            {"User-Agent": self.global_config.get("user_agent") or next_user_agent()}
        )
        self.base_url = self.config.get("base_url", "https://townhall.virginia.gov").rstrip("/")
        self.rate_limiter = RateLimiter(min_interval=float(self.config.get("rate_limit_seconds", 0.5)))

    # --------------------------------------------------------------------- discovery
    def discover(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **_,
    ) -> Iterable[Collection]:
        entries = self._load_forum_entries()
        if not entries:
            return []

        start_dt = self._parse_date_filter(start_date)
        end_dt = self._parse_date_filter(end_date)

        for entry in entries:
            closes_on = entry.closes_on.date() if entry.closes_on else None
            if start_dt and closes_on and closes_on < start_dt:
                continue
            if end_dt and closes_on and closes_on > end_dt:
                continue
            collection_id = entry.stage_id
            title = entry.action_title or entry.stage_label or f"Stage {entry.stage_id}"
            yield Collection(
                source=self.name,
                collection_id=collection_id,
                title=title,
                url=urljoin(self.base_url + "/", f"L/viewstage.cfm?stageid={entry.stage_id}"),
                jurisdiction="US-VA",
                topic="State Rulemaking",
                extra={
                    "board": entry.board,
                    "chapter": entry.chapter,
                    "stage": entry.stage_label,
                    "action_id": entry.action_id,
                    "action_title": entry.action_title,
                    "closes_on": closes_on.isoformat() if closes_on else None,
                    "comment_count": entry.comment_count,
                },
            )

    # ----------------------------------------------------------------- list documents
    def list_documents(
        self,
        collection_id: str,
        **_,
    ) -> Iterable[DocMeta]:
        html = self._fetch_html(f"L/comments.cfm?stageid={collection_id}")
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", attrs={"style": re.compile("background-color", re.IGNORECASE)})
        if not table:
            return []

        documents: List[DocMeta] = []
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) != 3:
                continue
            link = cells[0].find("a", href=True)
            if not link or "commentid" not in link.get("href", ""):
                continue
            comment_id = self._extract_query_value(link["href"], "commentid")
            if not comment_id:
                continue
            title = link.get_text(strip=True)
            commenter = cells[1].get_text(" ", strip=True) or None
            submitted = self._parse_comment_date(cells[2].get_text(" ", strip=True))
            documents.append(
                DocMeta(
                    source=self.name,
                    collection_id=collection_id,
                    doc_id=comment_id,
                    title=title or f"Comment {comment_id}",
                    submitter=commenter,
                    submitter_type=None,
                    org=None,
                    submitted_at=submitted,
                    language="en",
                    urls={
                        "html": urljoin(self.base_url + "/", f"L/viewcomments.cfm?commentid={comment_id}"),
                        "list": urljoin(self.base_url + "/", f"L/comments.cfm?stageid={collection_id}"),
                    },
                    extra={},
                )
            )
        return documents

    # -------------------------------------------------------------------------- fetch
    def fetch(self, doc: DocMeta, out_dir: Path, **_) -> Dict[str, str]:
        html_url = doc.urls.get("html")
        if not html_url:
            raise ValueError("Doc is missing comment detail URL")
        html = self._fetch_html(html_url, absolute=True)
        if not html:
            raise RuntimeError(f"Failed to download comment {doc.doc_id}")
        soup = BeautifulSoup(html, "html.parser")
        comment_box = soup.select_one(".divComment")
        comment_html = comment_box.decode_contents().strip() if comment_box else ""
        comment_text = comment_box.get_text("\n", strip=True) if comment_box else ""

        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / f"{doc.doc_id}.html"
        html_path.write_text(comment_html or html, encoding="utf-8")
        text_path = out_dir / f"{doc.doc_id}.txt"
        text_path.write_text(comment_text, encoding="utf-8")
        return {"html": str(html_path), "text": str(text_path)}

    # ---------------------------------------------------------------------- utilities
    def _load_forum_entries(self) -> List[ForumEntry]:
        html = self._fetch_html("L/Forums.cfm")
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        entries: List[ForumEntry] = []
        current_board = ""

        for row in rows:
            header_cell = row.find("td", attrs={"colspan": "3"})
            if header_cell and "font-weight:bold" in (header_cell.get("style") or ""):
                current_board = header_cell.get_text(strip=True)
                continue

            cells = row.find_all("td", recursive=False)
            if len(cells) < 3:
                continue
            comments_link = cells[0].find("a", href=re.compile("comments.cfm", re.IGNORECASE))
            if not comments_link:
                continue
            stage_id = self._extract_query_value(comments_link.get("href", ""), "stageid")
            if not stage_id:
                continue

            chapter = ""
            div = cells[1].find("div")
            if div:
                chapter = div.get_text(" ", strip=True)
            stage_info = self._parse_stage_info(cells[2])
            entries.append(
                ForumEntry(
                    board=current_board,
                    chapter=chapter.strip(),
                    stage_id=stage_id,
                    stage_label=stage_info.stage_label,
                    action_id=stage_info.action_id,
                    action_title=stage_info.action_title,
                    closes_on=stage_info.close_date,
                    comment_count=stage_info.comment_count,
                )
            )
        return entries

    @dataclass(slots=True)
    class _StageInfo:
        stage_label: str
        action_id: Optional[str]
        action_title: Optional[str]
        close_date: Optional[datetime]
        comment_count: int

    def _parse_stage_info(self, cell) -> _StageInfo:
        stage_label = ""
        action_id = None
        action_title = None
        close_date = None
        comment_count = 0

        action_link = cell.find("a", href=re.compile("viewaction.cfm", re.IGNORECASE))
        if action_link:
            action_id = self._extract_query_value(action_link["href"], "actionid")
            sibling = action_link.find_parent("td").find_next_sibling("td")
            action_title = sibling.get_text(" ", strip=True) if sibling else None

        stage_link = cell.find("a", href=re.compile("viewstage.cfm", re.IGNORECASE))
        if stage_link:
            sibling = stage_link.find_parent("td").find_next_sibling("td")
            stage_label = sibling.get_text(" ", strip=True) if sibling else ""

        info_text = cell.get_text(" ", strip=True)
        close_match = DATE_RE.search(info_text)
        if close_match:
            try:
                close_date = datetime.strptime(close_match.group("date"), "%m/%d/%y")
            except ValueError:
                close_date = None
        comments_match = COMMENTS_RE.search(info_text)
        if comments_match:
            try:
                comment_count = int(comments_match.group("count"))
            except ValueError:
                comment_count = 0

        return self._StageInfo(
            stage_label=stage_label,
            action_id=action_id,
            action_title=action_title,
            close_date=close_date,
            comment_count=comment_count,
        )

    def _parse_date_filter(self, raw: Optional[str]) -> Optional[datetime.date]:
        if not raw:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                if fmt == "%Y":
                    value = datetime.strptime(f"{raw}-01-01", "%Y-%m-%d")
                elif fmt == "%Y-%m":
                    value = datetime.strptime(f"{raw}-01", "%Y-%m-%d")
                else:
                    value = datetime.strptime(raw, fmt)
                return value.date()
            except ValueError:
                continue
        return None

    def _parse_comment_date(self, raw: str) -> Optional[str]:
        cleaned = re.sub(r"[\xa0]+", " ", raw).strip()
        if not cleaned:
            return None
        for fmt in ("%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        return None

    def _fetch_html(self, path: str, *, absolute: bool = False) -> Optional[str]:
        if absolute:
            url = path
        else:
            url = urljoin(self.base_url + "/", path.lstrip("/"))
        response = backoff_get(
            url,
            rate_limiter=self.rate_limiter,
            session=self.session,
            headers={"User-Agent": self.session.headers.get("User-Agent") or next_user_agent()},
            raise_for_status=False,
        )
        if response is None or response.status_code != 200:
            logger.warning("Virginia Town Hall request to %s failed with %s", url, response.status_code if response else "n/a")
            return None
        response.encoding = response.encoding or "utf-8"
        return response.text

    def _extract_query_value(self, href: str, key: str) -> Optional[str]:
        parsed = urlparse(urljoin(self.base_url + "/", href))
        params = parse_qs(parsed.query)
        values = params.get(key)
        if values:
            return values[0]
        # When the link is "comments.cfm?stageid=10600" without '?', parse manually
        if "?" not in href and "=" in href:
            parts = href.split("=")
            if len(parts) == 2 and parts[0].endswith(key):
                return parts[1]
        return None


__all__ = ["VirginiaTownhallConnector"]
