"""
Connector for Pennsylvania DEP's eComment portal.

References:
- Landing page listing active comment periods: https://www.ahs.dep.pa.gov/eComment/
- Comment list + detail view: https://www.ahs.dep.pa.gov/eComment/ViewComments.aspx?enc=<token>
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ai_corpus.connectors.base import BaseConnector, Collection, DocMeta
from ai_corpus.utils.http import RateLimiter, next_user_agent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CommentRow:
    index: int
    first_name: str
    last_name: str
    affiliation: str
    city: str
    state: str
    country: str
    method: str
    received: str
    button_name: str
    attachments: List[str]


class PaDepEcommentConnector(BaseConnector):
    name = "pa_dep_ecomment"

    def __init__(self, config: Dict, global_config: Optional[Dict] = None) -> None:
        self.config = config or {}
        self.global_config = global_config or {}
        self.base_url = self.config.get("base_url", "https://www.ahs.dep.pa.gov/eComment").rstrip("/")
        self.rate_limiter = RateLimiter(min_interval=float(self.config.get("rate_limit_seconds", 0.5)))

    # --------------------------------------------------------------------- discovery
    def discover(self, **_) -> Iterable[Collection]:
        html = self._get_page(self.base_url + "/")
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        collections: List[Collection] = []
        for link in soup.find_all("a", string="View comments"):
            href = link.get("href")
            if not href or "ViewComments.aspx" not in href:
                continue
            enc = href.split("enc=")[-1]
            row = link.find_parent("tr")
            if not row:
                continue
            cells = row.find_all("td")
            if len(cells) < 6:
                continue
            title = cells[0].get_text(" ", strip=True)
            category = cells[1].get_text(" ", strip=True)
            bulletin_date = cells[2].get_text(" ", strip=True)
            start_date = cells[3].get_text(" ", strip=True)
            end_date = cells[4].get_text(" ", strip=True)
            detail_url = cells[0].find("a", href=True)
            collections.append(
                Collection(
                    source=self.name,
                    collection_id=enc,
                    title=title,
                    url=urljoin(self.base_url + "/", href),
                    jurisdiction="US-PA",
                    topic="State Rulemaking",
                    extra={
                        "category": category,
                        "bulletin_date": bulletin_date,
                        "start_date": start_date,
                        "end_date": end_date,
                        "detail_url": detail_url["href"] if detail_url else None,
                    },
                )
            )
        return collections

    # ----------------------------------------------------------------- list documents
    def list_documents(self, collection_id: str, **_) -> Iterable[DocMeta]:
        rows = self._fetch_comment_rows(collection_id)
        docs: List[DocMeta] = []
        for row in rows:
            doc_id = f"{collection_id}-{row.index}"
            submitter = " ".join(filter(None, [row.first_name, row.last_name])).strip()
            submitted_at = self._parse_date(row.received)
            docs.append(
                DocMeta(
                    source=self.name,
                    collection_id=collection_id,
                    doc_id=doc_id,
                    title=f"Comment from {submitter or 'anonymous'}",
                    submitter=submitter or None,
                    submitter_type=None,
                    org=row.affiliation or None,
                    submitted_at=submitted_at,
                    language="en",
                    urls={
                        "html": f"{self.base_url}/ViewComments.aspx?enc={collection_id}",
                    },
                    extra={
                        "button_name": row.button_name,
                        "enc": collection_id,
                        "attachments": row.attachments,
                        "city": row.city,
                        "state": row.state,
                        "country": row.country,
                        "method": row.method,
                    },
                    kind="response",
                )
            )
        return docs

    # -------------------------------------------------------------------------- fetch
    def fetch(self, doc: DocMeta, out_dir: Path, **_) -> Dict[str, str]:
        extra = doc.extra or {}
        enc = extra.get("enc")
        button_name = extra.get("button_name")
        if not enc or not button_name:
            raise ValueError("Document metadata missing enc/button information.")
        text, session = self._fetch_comment_text(enc, button_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        text_path = out_dir / f"{doc.doc_id}.txt"
        text_path.write_text(text or "", encoding="utf-8")
        artifacts = {"text": str(text_path)}

        attachments = extra.get("attachments") or []
        saved_files = []
        for idx, url in enumerate(attachments):
            file_resp = session.get(url, headers={"User-Agent": next_user_agent()}, timeout=60)
            if file_resp.status_code == 200:
                filename = f"{doc.doc_id}_attachment_{idx+1}"
                content_type = file_resp.headers.get("Content-Type", "")
                if "pdf" in content_type:
                    filename += ".pdf"
                elif "msword" in content_type:
                    filename += ".doc"
                else:
                    filename += ".bin"
                file_path = out_dir / filename
                file_path.write_bytes(file_resp.content)
                saved_files.append(str(file_path))
        if saved_files:
            artifacts["attachments"] = saved_files
        return artifacts

    # ---------------------------------------------------------------------- utilities
    def _fetch_comment_rows(self, enc: str) -> List[CommentRow]:
        session = requests.Session()
        headers = {"User-Agent": next_user_agent()}
        url = f"{self.base_url}/ViewComments.aspx?enc={enc}"
        response = self._session_get(session, url, headers=headers)
        if response is None:
            return []
        view_fields = self._extract_form_fields(response.text)
        if not view_fields:
            return []
        form_data = {**view_fields, "ctl00$ContentPlaceHolder1$ListCommentsBtn": "List Comments"}
        response = self._session_post(session, url, data=form_data, headers=headers)
        if response is None:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", id="ContentPlaceHolder1_CommentGrid")
        rows: List[CommentRow] = []
        if not table:
            return rows
        for idx, tr in enumerate(table.find_all("tr")[1:], start=1):
            cells = tr.find_all("td")
            if len(cells) < 10:
                continue
            button = cells[8].find("input", attrs={"type": "submit"})
            button_name = button.get("name") if button else ""
            attachments = [a.get("href") for a in cells[9].find_all("a", href=True)]
            rows.append(
                CommentRow(
                    index=idx,
                    first_name=cells[0].get_text(strip=True),
                    last_name=cells[1].get_text(strip=True),
                    affiliation=cells[2].get_text(strip=True),
                    city=cells[3].get_text(strip=True),
                    state=cells[4].get_text(strip=True),
                    country=cells[5].get_text(strip=True),
                    method=cells[6].get_text(strip=True),
                    received=cells[7].get_text(strip=True),
                    button_name=button_name,
                    attachments=attachments,
                )
            )
        return rows

    def _fetch_comment_text(self, enc: str, button_name: str) -> Tuple[str, requests.Session]:
        session = requests.Session()
        headers = {"User-Agent": next_user_agent()}
        url = f"{self.base_url}/ViewComments.aspx?enc={enc}"
        response = self._session_get(session, url, headers=headers)
        if response is None:
            return "", session
        view_fields = self._extract_form_fields(response.text)
        if not view_fields:
            return "", session
        list_data = {**view_fields, "ctl00$ContentPlaceHolder1$ListCommentsBtn": "List Comments"}
        response = self._session_post(session, url, data=list_data, headers=headers)
        if response is None:
            return "", session
        view_fields = self._extract_form_fields(response.text)
        if not view_fields:
            return "", session
        view_data = {**view_fields, button_name: "View"}
        response = self._session_post(session, url, data=view_data, headers=headers)
        if response is None:
            return "", session
        soup = BeautifulSoup(response.text, "html.parser")
        textarea = soup.find("textarea", id="ContentPlaceHolder1_CommentTextBox")
        text = textarea.get_text("\n", strip=True) if textarea else ""
        return text, session

    def _session_get(self, session: requests.Session, url: str, headers: Dict[str, str]):
        self.rate_limiter.wait()
        try:
            resp = session.get(url, headers=headers, timeout=60)
            if resp.status_code != 200:
                logger.warning("PA eComment GET %s failed: %s", url, resp.status_code)
                return None
            return resp
        except requests.RequestException as exc:
            logger.warning("PA eComment GET error: %s", exc)
            return None

    def _session_post(self, session: requests.Session, url: str, data: Dict[str, str], headers: Dict[str, str]):
        self.rate_limiter.wait()
        try:
            resp = session.post(url, data=data, headers=headers, timeout=60)
            if resp.status_code != 200:
                logger.warning("PA eComment POST %s failed: %s", url, resp.status_code)
                return None
            return resp
        except requests.RequestException as exc:
            logger.warning("PA eComment POST error: %s", exc)
            return None

    def _extract_form_fields(self, html: str) -> Optional[Dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        viewstate = soup.find("input", id="__VIEWSTATE")
        event_validation = soup.find("input", id="__EVENTVALIDATION")
        view_gen = soup.find("input", id="__VIEWSTATEGENERATOR")
        if not (viewstate and event_validation and view_gen):
            return None
        return {
            "__VIEWSTATE": viewstate.get("value", ""),
            "__EVENTVALIDATION": event_validation.get("value", ""),
            "__VIEWSTATEGENERATOR": view_gen.get("value", ""),
        }

    def _get_page(self, url: str) -> Optional[str]:
        try:
            self.rate_limiter.wait()
            resp = requests.get(url, headers={"User-Agent": next_user_agent()}, timeout=60)
            if resp.status_code != 200:
                logger.warning("PA eComment page %s returned %s", url, resp.status_code)
                return None
            resp.encoding = resp.encoding or "utf-8"
            return resp.text
        except requests.RequestException as exc:
            logger.warning("PA eComment request failed: %s", exc)
            return None

    def _parse_date(self, raw: str) -> Optional[str]:
        raw = raw.strip()
        if not raw:
            return None
        for fmt in ("%m/%d/%Y", "%m/%d/%y"):
            try:
                value = datetime.strptime(raw, fmt)
                return value.isoformat()
            except ValueError:
                continue
        return None


__all__ = ["PaDepEcommentConnector"]
