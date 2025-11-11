"""
Reusable HTTP helpers (rate limiting, retries) shared across connectors.
"""

from __future__ import annotations

import itertools
import logging
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Mapping, Optional

import requests

logger = logging.getLogger(__name__)
_thread_local = threading.local()

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 13; SM-G991U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36",
]

_ua_lock = threading.Lock()
_ua_cycle = itertools.cycle(random.sample(_USER_AGENTS, k=len(_USER_AGENTS)))


def next_user_agent() -> str:
    with _ua_lock:
        return next(_ua_cycle)


class RateLimiter:
    """Simple thread-safe rate limiter with back-off support."""

    def __init__(self, min_interval: float = 0.0) -> None:
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        """Block until at or after the next permitted request time."""
        while True:
            with self._lock:
                now = time.monotonic()
                wait_for = self._next_allowed - now
                if wait_for <= 0:
                    self._next_allowed = now + self.min_interval
                    return
            if wait_for > 0:
                time.sleep(wait_for)

    def penalize(self, penalty_seconds: float) -> None:
        if penalty_seconds <= 0:
            return
        with self._lock:
            now = time.monotonic()
            target = now + penalty_seconds
            self._next_allowed = max(self._next_allowed, target)


def get_http_session(default_headers: Optional[Mapping[str, str]] = None) -> requests.Session:
    """
    Return a thread-local requests.Session, optionally seeded with default headers.
    """
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    if default_headers:
        sess.headers.update({k: v for k, v in default_headers.items() if v is not None})
    return sess


def _retry_after_seconds(response: requests.Response, default_wait: float) -> float:
    header = response.headers.get("Retry-After")
    if not header:
        return default_wait
    header = header.strip()
    if header.isdigit():
        return max(float(header), default_wait)
    try:
        retry_dt = parsedate_to_datetime(header)
    except Exception:
        return default_wait
    if retry_dt is None:
        return default_wait
    if retry_dt.tzinfo is None:
        retry_dt = retry_dt.replace(tzinfo=timezone.utc)
    delta = (retry_dt - datetime.now(tz=timezone.utc)).total_seconds()
    if delta > 0:
        return max(delta, default_wait)
    return default_wait


def backoff_get(
    url: str,
    params: Any = None,
    *,
    timeout: int = 30,
    max_attempts: int = 5,
    headers: Optional[Dict[str, str]] = None,
    rate_limiter: Optional[RateLimiter] = None,
    default_headers: Optional[Mapping[str, str]] = None,
    session: Optional[requests.Session] = None,
    raise_for_status: bool = True,
) -> Optional[requests.Response]:
    """
    Perform a GET request with exponential backoff, optional rate limiting, and
    Retry-After handling. Returns None after exhausting attempts.
    """
    if params is None:
        params = {}
    for attempt in range(1, max_attempts + 1):
        try:
            if rate_limiter is not None:
                rate_limiter.wait()
            sess = session or get_http_session(default_headers=default_headers)
            local_headers = dict(headers or {})
            local_headers.setdefault("User-Agent", next_user_agent())
            response = sess.get(url, params=params, timeout=timeout, headers=local_headers)
            if response.status_code == 429:
                base_wait = min(120, 2 ** (attempt - 1))
                wait_for = _retry_after_seconds(response, base_wait)
                wait_for = max(wait_for / 10.0, base_wait / 4.0)
                logger.warning(
                    "Received 429 from %s (attempt %d/%d); backing off for %.1fs",
                    url,
                    attempt,
                    max_attempts,
                    wait_for,
                )
                if rate_limiter is not None:
                    rate_limiter.penalize(wait_for)
                else:
                    time.sleep(wait_for)
                continue
            if response.status_code >= 500 or response.status_code == 408:
                raise requests.HTTPError(f"{response.status_code} {response.reason}")
            if raise_for_status:
                response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            if attempt == max_attempts:
                logger.warning("GET failed %s after %d attempts: %s", url, attempt, exc)
                return None
            sleep_s = min(60, (2 ** (attempt - 1)) + 0.1 * attempt)
            logger.debug(
                "GET error on %s (attempt %d/%d): %s; retrying in %.1fs",
                url,
                attempt,
                max_attempts,
                exc,
                sleep_s,
            )
            if rate_limiter is not None:
                rate_limiter.penalize(sleep_s)
            time.sleep(sleep_s)
    return None


def regs_backoff_get(
    path: str,
    *,
    api_key: Optional[str],
    base_url: str = "https://api.regulations.gov/v4",
    params: Any = None,
    timeout: int = 30,
    max_attempts: int = 5,
    rate_limiter: Optional[RateLimiter] = None,
    user_agent: Optional[str] = None,
    accept: str = "application/vnd.api+json",
    extra_headers: Optional[Dict[str, str]] = None,
    raise_for_status: bool = True,
) -> Optional[requests.Response]:
    """
    Convenience wrapper around `backoff_get` for Regulations.gov requests.
    """
    if not api_key:
        logger.debug("No API key provided for Regulations.gov request to %s", path)
        return None
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    headers = {"X-Api-Key": api_key, "Accept": accept}
    if user_agent:
        headers["User-Agent"] = user_agent
    if extra_headers:
        headers.update(extra_headers)
    return backoff_get(
        url,
        params=params,
        timeout=timeout,
        max_attempts=max_attempts,
        headers=headers,
        rate_limiter=rate_limiter,
        raise_for_status=raise_for_status,
    )


__all__ = [
    "RateLimiter",
    "backoff_get",
    "get_http_session",
    "regs_backoff_get",
    "next_user_agent",
]
