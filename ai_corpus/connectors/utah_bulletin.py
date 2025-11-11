"""
Connector for Utah's State Bulletin announcements published via WordPress.

- REST API: https://rules.utah.gov/wp-json/wp/v2/posts?categories=76
  (category 76 = Publications, including Utah State Bulletin issues)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

from ai_corpus.connectors.base import BaseConnector, Collection, DocMeta
from ai_corpus.utils.http import RateLimiter, backoff_get, get_http_session, next_user_agent

logger = logging.getLogger(__name__)
PDF_RE = re.compile(r"https://rules\.utah\.gov/wp-content/uploads/[^\"']+\.pdf", re.IGNORECASE)


@dataclass(slots=True)
class BulletinPost:
    post_id: int
    title: str
    link: str
    pdf_url: Optional[str]
    published_at: Optional[str]


class UtahBulletinConnector(BaseConnector):
    name = "utah_bulletin"

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
        self.api_url = self.config.get("api_url", "https://rules.utah.gov/wp-json/wp/v2/posts")
        self.category_id = int(self.config.get("category_id", 76))
        self.per_page = int(self.config.get("per_page", 10))
        self.rate_limiter = RateLimiter(min_interval=float(self.config.get("rate_limit_seconds", 0.25)))

    # --------------------------------------------------------------------- discovery
    def discover(self, **_) -> Iterable[Collection]:
        posts = self._fetch_posts()
        for post in posts:
            yield Collection(
                source=self.name,
                collection_id=str(post.post_id),
                title=post.title,
                url=post.link,
                jurisdiction="US-UT",
                topic="State Rulemaking",
                extra={"pdf_url": post.pdf_url, "published_at": post.published_at},
            )

    # ----------------------------------------------------------------- list documents
    def list_documents(self, collection_id: str, **_) -> Iterable[DocMeta]:
        posts = {str(post.post_id): post for post in self._fetch_posts()}
        post = posts.get(collection_id)
        if not post or not post.pdf_url:
            return []
        return [
            DocMeta(
                source=self.name,
                collection_id=collection_id,
                doc_id=f"utah_bulletin_{collection_id}",
                title=post.title,
                submitter="Utah Office of Administrative Rules",
                submitter_type="agency",
                org="Utah Office of Administrative Rules",
                submitted_at=post.published_at,
                language="en",
                urls={"pdf": post.pdf_url},
                extra={"pdf_url": post.pdf_url},
                kind="notice",
            )
        ]

    # -------------------------------------------------------------------------- fetch
    def fetch(self, doc: DocMeta, out_dir: Path, **_) -> Dict[str, str]:
        pdf_url = doc.urls.get("pdf")
        if not pdf_url:
            raise ValueError("Document missing PDF URL")
        response = backoff_get(
            pdf_url,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": self.session.headers.get("User-Agent") or next_user_agent()},
            raise_for_status=True,
        )
        if response is None:
            raise RuntimeError(f"Failed to download {pdf_url}")
        out_dir.mkdir(parents=True, exist_ok=True)
        file_name = pdf_url.rsplit("/", 1)[-1]
        out_path = out_dir / file_name
        out_path.write_bytes(response.content)
        return {"pdf": str(out_path)}

    # ---------------------------------------------------------------------- utilities
    def _fetch_posts(self) -> List[BulletinPost]:
        params = {"categories": self.category_id, "per_page": self.per_page}
        response = backoff_get(
            self.api_url,
            params=params,
            session=self.session,
            rate_limiter=self.rate_limiter,
            headers={"User-Agent": self.session.headers.get("User-Agent") or next_user_agent()},
            raise_for_status=True,
        )
        if response is None:
            return []
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse Utah posts: %s", exc)
            return []

        posts: List[BulletinPost] = []
        for item in data:
            post_id = item.get("id")
            if not post_id:
                continue
            title = (item.get("title") or {}).get("rendered") or f"Utah Bulletin {post_id}"
            link = item.get("link") or urljoin("https://rules.utah.gov/", f"?p={post_id}")
            content = (item.get("content") or {}).get("rendered") or ""
            pdf_url = self._extract_pdf_url(content)
            posts.append(
                BulletinPost(
                    post_id=int(post_id),
                    title=_strip_html(title),
                    link=link,
                    pdf_url=pdf_url,
                    published_at=item.get("date"),
                )
            )
        return posts

    def _extract_pdf_url(self, html: str) -> Optional[str]:
        match = PDF_RE.search(html)
        return match.group(0) if match else None


def _strip_html(raw: str) -> str:
    return re.sub(r"<[^>]+>", "", raw).strip()


__all__ = ["UtahBulletinConnector"]
