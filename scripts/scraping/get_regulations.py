#!/usr/bin/env python3
"""
Combine Federal Register Public Inspection (current.json) and Published Documents (documents.json)
for a given date range, enrich with Regulations.gov metadata (including documentId and objectId),
and write a unified CSV.

New in this version:
- regs_document_id: Regulations.gov documentId (e.g., FMCSA-2025-0622-0001).
- regs_object_id: Regulations.gov objectId for the document (for /v4/comments?filter[commentOnId]=...).
- Uses /v4/documents/{documentId} (no unsupported filter[frDocNum]).
- RFI/RFC detection retained and improved heuristics preserved.
- Optional server-side narrowing with --fr-term.
- PRORULE/NOTICE mapping in PI retained.

Usage examples:
  python scripts/scraping/get_regulations.py --days 7 --regs-key $REGS_API_KEY
  python scripts/scraping/get_regulations.py --start 2025-09-01 --end 2025-10-07 \
    --rfi-rfc-only --fr-term "Request for Information" --fr-term "Request for Comments" \
    --regs-key $REGS_API_KEY -o rfi_rfc.csv
"""

import argparse, csv, os, logging, re, json, sys
import concurrent.futures
import threading
from functools import partial
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import xml.etree.ElementTree as ET
from dateutil import parser as dtparse
try:
    import xmltodict
except ImportError:  # pragma: no cover - optional dependency
    xmltodict = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from api_utils import RateLimiter, backoff_get, parse_doc_id_from_url, regs_backoff_get
from ai_corpus.rules import RULE_VERSION_COLUMNS, normalize_rule_row, detect_comment_citations

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **_: dict):
        return iterable if iterable is not None else []

FR_BASE = "https://www.federalregister.gov/api/v1"
UA = "regulations-merge/1.4 (contact: you@example.org)"
JSON_HEADERS = {"Accept": "application/json", "User-Agent": UA}

logger = logging.getLogger(__name__)


REGS_RATE_LIMITER = RateLimiter(min_interval=1.0)

# --------------------- parsing helpers ---------------------

def parse_iso_date(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    try:
        return date.fromisoformat(d[:10])
    except Exception:
        try:
            return dtparse.isoparse(d).date()
        except Exception:
            return None

def parse_iso_dt(d: Optional[str]) -> Optional[datetime]:
    if not d:
        return None
    try:
        return dtparse.isoparse(d)
    except Exception:
        return None

def join_list(vals: Optional[List[str]]) -> str:
    if not vals:
        return ""
    return "; ".join([str(v) for v in vals if v is not None])

def pick_agency_names(item: Dict[str, Any]) -> List[str]:
    if item.get("agency_names"):
        return list(item["agency_names"])
    agencies = item.get("agencies") or []
    names = [a.get("name") for a in agencies if isinstance(a, dict) and a.get("name")]
    return names

# --------------------- FR XML extraction ---------------------

def extract_info_from_fr_xml(xml_url: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetch and extract text from an FR full-text XML document.
    Returns (xml_dict, concatenated text within <SUPLINF>), either may be None.
    """
    if not xml_url:
        return None, None
    r = backoff_get(
        xml_url,
        timeout=30,
        headers={"Accept": "application/xml", "User-Agent": UA},
        default_headers=JSON_HEADERS,
    )
    if not r or r.status_code != 200:
        logger.debug("XML fetch failed: %s", xml_url)
        return None, None
    xml_dict = None
    if xmltodict is not None:
        try:
            xml_dict = xmltodict.parse(r.content)
        except Exception:
            xml_dict = None
    try:
        root = ET.fromstring(r.content)
        sup_nodes = root.findall(".//SUPLINF")
        if not sup_nodes:
            return xml_dict, None
        text = "\n\n".join(["".join(n.itertext()) for n in sup_nodes]).strip()
        return xml_dict, (text or None)
    except Exception as e:
        logger.debug("XML parse error for %s: %s", xml_url, e)
        return xml_dict, None

# --------------------- RFI/RFC detection ---------------------

RFI_PATTERNS = [
    re.compile(r"\brequest(?:s)?\s+for\s+information\b", re.I),
    re.compile(r"\bRFI\b", re.I)
]
RFC_PATTERNS = [
    re.compile(r"\brequest(?:s)?\s+for\s+comments?\b", re.I),
    re.compile(r"\bRFC\b", re.I)
]
NEGATIVE_PATTERNS = [
    re.compile(r"\bIETF\b", re.I),
    re.compile(r"\bInternet\s+Engineering\s+Task\s+Force\b", re.I),
    re.compile(r"\bRFC\s*\d{3,5}\b", re.I),
    re.compile(r"\brequest\s+for\s+correction\b", re.I),
]

def detect_rfi_rfc_blobs(fields: Dict[str, Optional[str]]) -> Tuple[bool, str, str]:
    is_rfi = False
    is_rfc = False
    hit_fields = set()
    for fname, text in fields.items():
        if not text:
            continue
        if any(p.search(text) for p in NEGATIVE_PATTERNS):
            continue
        if any(p.search(text) for p in RFI_PATTERNS):
            is_rfi = True
            hit_fields.add(fname)
        if any(p.search(text) for p in RFC_PATTERNS):
            is_rfc = True
            hit_fields.add(fname)
    if is_rfi and is_rfc:
        label = "RFI;RFC"
    elif is_rfi:
        label = "RFI"
    elif is_rfc:
        label = "RFC"
    else:
        label = "NONE"
    return (is_rfi or is_rfc), label, ";".join(sorted(hit_fields)) if hit_fields else ""

# --------------------- FR: Published ---------------------

FR_DOC_FIELDS = [
    "title","agencies","agency_names","type","publication_date","html_url",
    "document_number","comments_close_on","regulations_dot_gov_url",
    "regulations_dot_gov_info","docket_id","docket_ids","abstract","action","dates",
    "full_text_xml_url",
]

def fr_query_documents(
    params: List[Tuple[str, Any]],
    executor: Optional[concurrent.futures.Executor] = None,
    log_label: str = "FR documents",
) -> List[Dict[str, Any]]:
    url = f"{FR_BASE}/documents.json"
    results: List[Dict[str, Any]] = []
    first_params = list(params)
    first_params.append(("page", 1))
    first_response = backoff_get(url, first_params, default_headers=JSON_HEADERS)
    if not first_response:
        return results
    first_json = first_response.json() or {}
    first_batch = first_json.get("results") or []
    if not first_batch:
        return results
    results.extend(first_batch)
    total_pages = first_json.get("total_pages") or 1
    logger.info("%s query has %d page(s)", log_label, total_pages)
    logger.debug("Fetched published page 1/%d with %d item(s)", total_pages, len(first_batch))

    if total_pages <= 1:
        return results

    remaining_pages = range(2, total_pages + 1)
    if executor is None:
        for page in remaining_pages:
            page_params = list(params)
            page_params.append(("page", page))
            resp = backoff_get(url, page_params, default_headers=JSON_HEADERS)
            if not resp:
                continue
            js = resp.json() or {}
            batch = js.get("results") or []
            if not batch:
                continue
            results.extend(batch)
            logger.debug("Fetched published page %d/%d with %d item(s)", page, total_pages, len(batch))
        return results

    future_to_page = {}
    for page in remaining_pages:
        page_params = list(params)
        page_params.append(("page", page))
        future = executor.submit(backoff_get, url, page_params, default_headers=JSON_HEADERS)
        future_to_page[future] = page

    for future in concurrent.futures.as_completed(future_to_page):
        page = future_to_page[future]
        try:
            resp = future.result()
        except Exception as exc:
            logger.warning("Published page %d fetch raised %s", page, exc)
            continue
        if not resp:
            continue
        js = resp.json() or {}
        batch = js.get("results") or []
        if not batch:
            continue
        results.extend(batch)
        logger.debug("Fetched published page %d/%d with %d item(s)", page, total_pages, len(batch))
    return results


def fr_get_published(
    start: date,
    end: date,
    include_types: List[str],
    terms: Optional[List[str]] = None,
    executor: Optional[concurrent.futures.Executor] = None,
) -> List[Dict[str, Any]]:
    params: List[Tuple[str, Any]] = [
        ("per_page", 1000),
        ("order", "newest"),
        ("conditions[publication_date][gte]", start.isoformat()),
        ("conditions[publication_date][lte]", end.isoformat()),
    ]
    for t in include_types:
        params.append(("conditions[type][]", t))
    if terms:
        for t in terms:
            if t and t.strip():
                params.append(("conditions[term]", t.strip()))
    for f in FR_DOC_FIELDS:
        params.append(("fields[]", f))
    logger.info(
        "Fetching published documents %s → %s (types=%s, terms=%s)",
        start.isoformat(), end.isoformat(), ",".join(include_types), json.dumps(terms) if terms else "None",
    )
    return fr_query_documents(
        params,
        executor=executor,
        log_label=f"Published {start.isoformat()}→{end.isoformat()}",
    )


def fr_get_docket_documents(
    docket_id: str,
    include_types: List[str],
    executor: Optional[concurrent.futures.Executor] = None,
) -> List[Dict[str, Any]]:
    params: List[Tuple[str, Any]] = [
        ("per_page", 1000),
        ("order", "oldest"),
        ("conditions[term]", docket_id),
    ]
    for t in include_types:
        if t:
            params.append(("conditions[type][]", t))
    for f in FR_DOC_FIELDS:
        params.append(("fields[]", f))
    return fr_query_documents(
        params,
        executor=executor,
        log_label=f"Docket {docket_id}",
    )

# --------------------- FR: Public Inspection ---------------------

PI_TYPE_CANON = {
    "notice": "NOTICE",
    "proposed rule": "PRORULE",
    "rule": "RULE",
    "presidential document": "PRESDOCU",
}

def fr_get_pi_current() -> List[Dict[str, Any]]:
    url = f"{FR_BASE}/public-inspection-documents/current.json"
    logger.info("Fetching Public Inspection current items")
    r = backoff_get(url, {"per_page": 1000}, default_headers=JSON_HEADERS)
    if not r:
        return []
    data = r.json()
    return data.get("results", data) if isinstance(data, dict) else data

def pi_filter_by_pubdate(items: List[Dict[str, Any]], start: date, end: date, include_types: List[str]) -> List[Dict[str, Any]]:
    incl = set(t.upper() for t in include_types)
    out = []
    for it in items:
        p = parse_iso_date(it.get("publication_date"))
        if p and start <= p <= end:
            t_text = (it.get("type") or "").strip().lower()
            canon = PI_TYPE_CANON.get(t_text)
            if not incl or (canon and canon in incl):
                out.append(it)
    return out

# --------------------- Regulations.gov helpers ---------------------
def regs_get_doc_detail_by_id(doc_id: str, api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    r = regs_backoff_get(
        f"documents/{doc_id}",
        api_key=api_key,
        user_agent=UA,
        rate_limiter=REGS_RATE_LIMITER,
    )
    if not r or r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None

def regs_search_by_docket(docket_id: str, api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    # List docs in docket; pick a sensible one (prefer Proposed Rule; else first).
    r = regs_backoff_get(
        "documents",
        api_key=api_key,
        params={"filter[docketId]": docket_id, "page[size]": 250},
        user_agent=UA,
        rate_limiter=REGS_RATE_LIMITER,
    )
    if not r or r.status_code != 200:
        return None
    js = r.json() or {}
    data = js.get("data") or []
    # Prefer Proposed Rule if present
    for d in data:
        attrs = d.get("attributes", {})
        if (attrs.get("documentType") or "").lower() == "proposed rule":
            did = d.get("id")
            if did:
                return regs_get_doc_detail_by_id(did, api_key)
    if data:
        did = data[0].get("id")
        if did:
            return regs_get_doc_detail_by_id(did, api_key)
    return None

def regs_window_and_ids(detail: Optional[Dict[str, Any]]) -> Tuple[Optional[date], Optional[date], Optional[bool], Optional[str], Optional[str]]:
    """
    From a Regulations.gov document detail JSON, pull:
    commentStart, commentDue, openForComment, documentId, objectId
    """
    if not detail:
        return None, None, None, None, None
    data = detail.get("data") or {}
    attrs = data.get("attributes") or {}
    start = parse_iso_date(attrs.get("commentStartDate") or attrs.get("commentOpenDate"))
    due = parse_iso_date(attrs.get("commentDueDate") or attrs.get("commentEndDate"))
    open_flag = attrs.get("openForComment")
    doc_id = data.get("id")
    object_id = attrs.get("objectId")
    return start, due, open_flag, doc_id, object_id

# --------------------- normalization & status ---------------------

def guess_regs_docket_from_pi_numbers(dnums: Optional[List[str]]) -> Optional[str]:
    if not dnums:
        return None
    # First pass: obvious agency patterns
    for d in dnums:
        if "-" in d and any(tag in d for tag in ("-HQ-", "-OSHA-", "-FDA-", "-CFPB-", "-CMS-", "-APHIS-", "-FWS-", "-FERC-", "-EPA-")):
            return d
    # Next: generic ABC-YYYY-NNNN form
    for d in dnums:
        parts = d.split("-")
        if len(parts) >= 3 and parts[0].isalpha():
            return d
    return None

def compute_comment_fields(
    now: date,
    fr_pub_date: Optional[date],
    fr_close: Optional[date],
    regs_start: Optional[date],
    regs_due: Optional[date],
    regs_open_flag: Optional[bool],
) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    start = regs_start or fr_pub_date
    due = regs_due or fr_close
    if regs_open_flag is True:
        status = "open"
    elif regs_open_flag is False:
        status = "closed"
    else:
        if start and now < start:
            status = "scheduled"
        elif due and now > due:
            status = "closed"
        elif start and (not due or now <= due):
            status = "open"
        else:
            status = "unknown"
    active = (status == "open")
    return (
        start.isoformat() if start else None,
        due.isoformat() if due else None,
        status,
        active
    )

def detect_and_attach_rfi_rfc(row: Dict[str, Any]) -> None:
    fields_to_scan = {
        "title": row.get("title"),
        "action": row.get("action"),
        "abstract": row.get("abstract"),
        "dates": row.get("dates"),
        "supplementary_information": row.get("supplementary_information"),
        "details": row.get("details"),  # PI rows only
    }
    is_rr, label, matched = detect_rfi_rfc_blobs(fields_to_scan)
    row["is_rfi_rfc"] = "TRUE" if is_rr else "FALSE"
    row["rfi_rfc_label"] = label
    row["rfi_rfc_matched_in"] = matched


HISTORY_REL_MAP = {
    "PRORULE": "proposal",
    "RULE": "final",
    "NOTICE": "notice",
}


def annotate_comment_citations(row: Dict[str, Any]) -> None:
    mentions, snippet = detect_comment_citations(
        row.get("supplementary_information"),
        row.get("abstract"),
        row.get("action"),
        row.get("dates"),
    )
    row["mentions_comment_response"] = mentions
    row["comment_citation_snippet"] = snippet or ""


def attach_history_metadata(
    row: Dict[str, Any],
    docket_id: Optional[str],
    parent_docnum: Optional[str],
    rank: int,
) -> None:
    doc_type = (row.get("type") or "").strip().upper()
    relationship = HISTORY_REL_MAP.get(doc_type, "related")
    if rank == 1:
        relationship = "seed"
    row["history_parent_docket"] = docket_id or row.get("docket_id") or ""
    row["history_parent_fr_doc"] = parent_docnum or (row.get("fr_document_number") or "")
    row["history_stage"] = row.get("type") or ""
    row["history_relationship"] = relationship
    row["history_rank"] = str(rank)

def ensure_regs_detail_and_ids(
    regs_key: Optional[str],
    regs_url: Optional[str],
    fr_regs_info: Optional[Dict[str, Any]],
    docket_id: Optional[str],
    cache: Dict[str, Dict[str, Any]],
    cache_lock: threading.Lock,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str]]:
    """
    Ensure we have a Regulations.gov document detail and return:
    (detail_json, regs_document_id, regs_object_id, normalized_regs_url)
    Resolution order:
      1) documentId from FR 'regulations_dot_gov_info.document_id'
      2) parse from regs_url if present
      3) search by docketId (fallback)
    """
    doc_id_hint = None
    if isinstance(fr_regs_info, dict):
        doc_id_hint = fr_regs_info.get("document_id")

    # 1) Try explicit document id first
    detail = None
    regs_doc_id = None
    regs_obj_id = None
    normalized_url = regs_url or ""

    def get_from_cache_or_fetch_doc(doc_id: str) -> Optional[Dict[str, Any]]:
        cache_key = f"doc::{doc_id}"
        with cache_lock:
            cached = cache.get(cache_key)
        if cached is not None:
            return cached
        js = regs_get_doc_detail_by_id(doc_id, regs_key)
        with cache_lock:
            cache[cache_key] = js
        return js

    if doc_id_hint:
        detail = get_from_cache_or_fetch_doc(doc_id_hint)
    # 2) Parse from regs_url
    if detail is None and regs_url:
        parsed = parse_doc_id_from_url(regs_url)
        if parsed:
            detail = get_from_cache_or_fetch_doc(parsed)

    # 3) Fallback: by docket
    if detail is None and docket_id:
        cache_key = f"docket::{docket_id}"
        with cache_lock:
            detail = cache.get(cache_key)
        if detail is None:
            detail = regs_search_by_docket(docket_id, regs_key)
            with cache_lock:
                cache[cache_key] = detail

    # Extract ids
    start, due, open_flag, doc_id, object_id = regs_window_and_ids(detail)
    regs_doc_id = doc_id
    regs_obj_id = object_id
    if regs_doc_id and not normalized_url:
        normalized_url = f"https://www.regulations.gov/document/{regs_doc_id}"

    return detail, regs_doc_id, regs_obj_id, normalized_url

# --------------------- Row builders ---------------------

def normalize_published(
    item: Dict[str, Any],
    regs_key: Optional[str],
    cache: Dict[str, Dict[str, Any]],
    cache_lock: threading.Lock,
) -> Dict[str, Any]:
    agencies = pick_agency_names(item)
    fr_docnum = item.get("document_number")
    fr_url = item.get("html_url")
    fr_pub = parse_iso_date(item.get("publication_date"))
    fr_close = parse_iso_date(item.get("comments_close_on"))

    docket = item.get("docket_id") 
    docket_ids = item.get("docket_ids") or [None]
    regs_url = item.get("regulations_dot_gov_url") or ""

    # Pull Regulations.gov detail + IDs
    regs_detail, regs_doc_id, regs_obj_id, regs_url_norm = ensure_regs_detail_and_ids(
        regs_key=regs_key,
        regs_url=regs_url,
        fr_regs_info=item.get("regulations_dot_gov_info"),
        docket_id=docket,
        cache=cache,
        cache_lock=cache_lock,
    )
    regs_url = regs_url_norm

    # Use Regulations.gov window when available
    start, due, open_flag, _, _ = regs_window_and_ids(regs_detail)
    c_start, c_due, c_status, c_active = compute_comment_fields(
        now=date.today(),
        fr_pub_date=fr_pub,
        fr_close=fr_close,
        regs_start=start,
        regs_due=due,
        regs_open_flag=open_flag,
    )

    xml_dict, suppl = extract_info_from_fr_xml(item.get("full_text_xml_url"))
    row = {
        "source": "published",
        "fr_document_number": fr_docnum or "",
        "title": item.get("title") or "",
        "type": item.get("type") or "",
        "publication_date": fr_pub.isoformat() if fr_pub else "",
        "agency": join_list(agencies),
        "fr_url": fr_url or "",
        "docket_id": docket or "",
        "docket_ids": docket_ids or [],
        "regs_url": regs_url or "",
        "regs_document_id": regs_doc_id or "",
        "regs_object_id": regs_obj_id or "",
        "comment_start_date": c_start or "",
        "comment_due_date": c_due or "",
        "comment_status": c_status or "",
        "comment_active": "TRUE" if c_active else "FALSE",
        "abstract": item.get("abstract") or "",
        "action": item.get("action") or "",
        "dates": item.get("dates") or "",
        "supplementary_information": suppl or "",
        "xml_dict": xml_dict or {},
        "details": "",  # publish rows don't have PI 'details'
    }
    detect_and_attach_rfi_rfc(row)
    annotate_comment_citations(row)
    return row

def normalize_pi(
    item: Dict[str, Any],
    regs_key: Optional[str],
    cache: Dict[str, Dict[str, Any]],
    cache_lock: threading.Lock
) -> Dict[str, Any]:
    agencies = pick_agency_names(item)
    fr_docnum = item.get("document_number")
    fr_url = item.get("html_url")
    fr_pub = parse_iso_date(item.get("publication_date"))

    dnums = item.get("docket_numbers") or []
    guessed = guess_regs_docket_from_pi_numbers(dnums)

    # Try to resolve Regulations.gov detail + IDs from guessed docket; PI rarely has a direct regs URL
    regs_detail, regs_doc_id, regs_obj_id, regs_url = ensure_regs_detail_and_ids(
        regs_key=regs_key,
        regs_url=None,
        fr_regs_info=None,
        docket_id=guessed,
        cache=cache,
        cache_lock=cache_lock,
    )

    start, due, open_flag, _, _ = regs_window_and_ids(regs_detail)
    c_start, c_due, c_status, c_active = compute_comment_fields(
        now=date.today(),
        fr_pub_date=fr_pub,
        fr_close=None,
        regs_start=start,
        regs_due=due,
        regs_open_flag=open_flag,
    )

    subjects = [item.get("subject_1"), item.get("subject_2"), item.get("subject_3")]
    subjects = [s for s in subjects if s]
    details = "; ".join(subjects)[:2000]

    row = {
        "source": "public_inspection",
        "fr_document_number": fr_docnum or "",
        "title": item.get("title") or "",
        "type": item.get("type") or "",
        "publication_date": fr_pub.isoformat() if fr_pub else "",
        "agency": join_list(agencies),
        "fr_url": fr_url or "",
        "docket_id": guessed or "",
        "docket_ids": [],
        "regs_url": regs_url or "",
        "regs_document_id": regs_doc_id or "",
        "regs_object_id": regs_obj_id or "",
        "comment_start_date": c_start or "",
        "comment_due_date": c_due or "",
        "comment_status": c_status or "",
        "comment_active": "TRUE" if c_active else "FALSE",
        "details": details,
        "abstract": "",
        "action": "",
        "dates": "",
        "supplementary_information": "",
        "xml_dict": {},
    }
    detect_and_attach_rfi_rfc(row)
    annotate_comment_citations(row)
    return row

# --------------------- concurrency ---------------------

def parallel_map(
    executor: concurrent.futures.Executor,
    func,
    items: List[Any],
    desc: str,
    unit: str,
) -> List[Any]:
    if not items:
        return []
    futures = {executor.submit(func, item): idx for idx, item in enumerate(items)}
    results: List[Any] = [None] * len(items)
    for fut in tqdm(concurrent.futures.as_completed(futures), total=len(items), desc=desc, unit=unit):
        results[futures[fut]] = fut.result()
    return results


def augment_with_docket_history(
    rows: List[Dict[str, Any]],
    *,
    regs_key: Optional[str],
    cache: Dict[str, Dict[str, Any]],
    cache_lock: threading.Lock,
    include_types: List[str],
    history_types: Optional[List[str]] = None,
    history_limit: Optional[int] = None,
) -> None:
    docket_ids = sorted({r.get("docket_id") for r in rows if r.get("docket_id")})
    if not docket_ids:
        logger.info("No docket IDs available for history enrichment.")
        return
    history_type_list = [t.strip().upper() for t in (history_types or include_types) if t and t.strip()]
    if not history_type_list:
        history_type_list = ["PRORULE", "RULE", "NOTICE"]
    logger.info(
        "Fetching docket histories for %d docket(s) (types=%s limit=%s)",
        len(docket_ids),
        ",".join(history_type_list),
        history_limit if history_limit is not None else "None",
    )
    doc_lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        docnum = row.get("fr_document_number")
        if docnum:
            doc_lookup[docnum] = row

    for docket_id in docket_ids:
        history_items = fr_get_docket_documents(docket_id, history_type_list)
        if not history_items:
            continue
        history_items.sort(
            key=lambda item: (
                parse_iso_date(item.get("publication_date")) or date.min,
                item.get("document_number") or "",
            )
        )
        parent_docnum = next(
            (item.get("document_number") for item in history_items if item.get("document_number")),
            None,
        )
        for idx, item in enumerate(history_items, start=1):
            if history_limit and idx > history_limit:
                break
            row = normalize_published(
                item,
                regs_key=regs_key,
                cache=cache,
                cache_lock=cache_lock,
            )
            attach_history_metadata(row, docket_id, parent_docnum, idx)
            docnum = row.get("fr_document_number")
            if docnum and docnum in doc_lookup:
                attach_history_metadata(doc_lookup[docnum], docket_id, parent_docnum, idx)
                doc_lookup[docnum]["mentions_comment_response"] = row.get("mentions_comment_response")
                doc_lookup[docnum]["comment_citation_snippet"] = row.get("comment_citation_snippet")
                continue
            row["source"] = row.get("source") or "history"
            rows.append(row)
            if docnum:
                doc_lookup[docnum] = row

# --------------------- CSV layout ---------------------

CSV_COLUMNS = list(RULE_VERSION_COLUMNS)

# --------------------- orchestrate ---------------------

def run(
    start: date,
    end: date,
    include_types: List[str],
    include_pi: bool,
    regs_key: Optional[str],
    outfile: str,
    rfi_rfc_only: bool = False,
    fr_terms: Optional[List[str]] = None,
    num_workers: int = 4,
    include_docket_history: bool = False,
    history_types: Optional[List[str]] = None,
    history_limit: Optional[int] = None,
) -> None:
    rows: List[Dict[str, Any]] = []
    cache: Dict[str, Dict[str, Any]] = {}
    cache_lock = threading.Lock()

    # If we only want RFIs/RFCs and no explicit terms, hint the server to speed up.
    if rfi_rfc_only and not fr_terms:
        fr_terms = ["Request for Information", "Request for Comments"]

    worker_count = max(1, num_workers or 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        # 1) PI (current) filtered to our range
        if include_pi:
            pi_all = fr_get_pi_current()
            logger.info("PI returned %d item(s) before filtering", len(pi_all))
            pi_items = pi_filter_by_pubdate(pi_all, start, end, include_types)
            logger.info("PI items within range/types: %d", len(pi_items))
            pi_normalizer = partial(normalize_pi, regs_key=regs_key, cache=cache, cache_lock=cache_lock)
            rows.extend(
                parallel_map(
                    executor,
                    pi_normalizer,
                    pi_items,
                    desc="Normalizing Public Inspection",
                    unit="doc",
                )
            )

        # 2) Published documents in range (+ optional server-side term narrowing)
        pub_items = fr_get_published(start, end, include_types, terms=fr_terms, executor=executor)
        logger.info("Published documents fetched: %d", len(pub_items))
        pub_normalizer = partial(normalize_published, regs_key=regs_key, cache=cache, cache_lock=cache_lock)
        rows.extend(
            parallel_map(
                executor,
                pub_normalizer,
                pub_items,
                desc="Normalizing Published",
                unit="doc",
            )
        )

    # 3) Deduplicate by FR document number, prefer published over PI
    by_doc = {}
    for r in tqdm(rows, desc="Deduplicating", unit="doc"):
        key = r.get("fr_document_number") or (r.get("fr_url") or "")
        if key in by_doc:
            if by_doc[key]["source"] == "public_inspection" and r["source"] == "published":
                by_doc[key] = r
        else:
            by_doc[key] = r
    rows = list(by_doc.values())
    logger.info("Unique rows after deduplication: %d", len(rows))

    # 4) Augment with docket history if requested
    if include_docket_history:
        augment_with_docket_history(
            rows,
            regs_key=regs_key,
            cache=cache,
            cache_lock=cache_lock,
            include_types=include_types,
            history_types=history_types,
            history_limit=history_limit,
        )
        logger.info("Rows after history enrichment: %d", len(rows))

    # 5) Optional filter: only keep RFIs/RFCs
    if rfi_rfc_only:
        before = len(rows)
        rows = [r for r in rows if r.get("is_rfi_rfc") == "TRUE"]
        logger.info("Filtered to RFIs/RFCs: %d → %d", before, len(rows))

    # 6) Annotate scrape mode on all rows
    scrape_mode = "rfi_rfc_only" if rfi_rfc_only else "all_documents"
    for r in rows:
        r["scrape_mode"] = scrape_mode

    # 7) Write CSV
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in tqdm(rows, desc="Writing CSV", unit="row"):
            rr = dict(r)
            if isinstance(rr.get("xml_dict"), dict):
                try:
                    rr["xml_dict"] = json.dumps(rr["xml_dict"])[:500000]
                except Exception:
                    rr["xml_dict"] = str(rr["xml_dict"])
            prepared = normalize_rule_row(rr)
            w.writerow(prepared)

    logger.info("Wrote %d rows to %s", len(rows), outfile)

# --------------------- CLI ---------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Merge Federal Register PI and Published docs into a unified CSV, with Regulations.gov IDs and RFI/RFC labeling.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--days", type=int, help="If set, use [today - days, today] as the date range.")
    g.add_argument("--start", type=str, help="Start date (YYYY-MM-DD). Use with --end.")
    ap.add_argument("--end", type=str, help="End date (YYYY-MM-DD). Required if --start is set.")
    ap.add_argument("--include-types", type=str, default="PRORULE,NOTICE",
                    help="Comma-separated FR types to include (e.g., PRORULE,NOTICE,RULE). Default: PRORULE,NOTICE")
    ap.add_argument("--no-pi", action="store_true", help="Skip Public Inspection.")
    ap.add_argument("--regs-key", type=str, default=os.getenv("REGS_API_KEY"),
                    help="Regulations.gov API key (optional but recommended for objectId/documentId resolution).")
    ap.add_argument("--rfi-rfc-only", action="store_true",
                    help="Return only RFIs/RFCs (also sets scrape_mode=rfi_rfc_only).")
    ap.add_argument("--fr-term", action="append",
                    help="Add a Federal Register conditions[term] search term (repeatable). Example: --fr-term 'Request for Information'")
    ap.add_argument("-o", "--outfile", type=str, default=None, help="Output CSV path.")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="Number of concurrent worker threads for API calls. Default: 4")
    ap.add_argument("--include-docket-history", action="store_true",
                    help="Fetch all available Federal Register documents for each docket and annotate history metadata.")
    ap.add_argument("--history-types", action="append",
                    help="Override the Federal Register document types to include when gathering history (default: --include-types).")
    ap.add_argument("--history-limit", type=int, default=None,
                    help="Maximum number of historical documents to keep per docket (default: unlimited).")
    return ap.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.days is not None:
        end = date.today() + timedelta(days=1)
        start = end - timedelta(days=args.days)
    else:
        if not args.start or not args.end:
            raise SystemExit("Please provide --days OR --start and --end.")
        start = parse_iso_date(args.start)
        end = parse_iso_date(args.end)
        if not start or not end:
            raise SystemExit("Could not parse --start/--end; expected YYYY-MM-DD.")
        if start > end:
            start, end = end, start

    include_types = [t.strip().upper() for t in args.include_types.split(",") if t.strip()]
    outfile = args.outfile or f"federal_rulemaking_{start.isoformat()}_{end.isoformat()}.csv"

    logger.info(
        "Start: %s End: %s Types: %s Include PI: %s Output: %s RFI/RFC only: %s Terms: %s History: %s History types: %s History limit: %s",
        start.isoformat(), end.isoformat(), ",".join(include_types), not args.no_pi, outfile,
        args.rfi_rfc_only, json.dumps(args.fr_term) if args.fr_term else "None",
        args.include_docket_history,
        ",".join(args.history_types) if args.history_types else "default",
        args.history_limit if args.history_limit is not None else "None",
    )
    run(
        start=start,
        end=end,
        include_types=include_types,
        include_pi=(not args.no_pi),
        regs_key=args.regs_key,
        outfile=outfile,
        rfi_rfc_only=args.rfi_rfc_only,
        fr_terms=args.fr_term,
        num_workers=args.num_workers,
        include_docket_history=args.include_docket_history,
        history_types=args.history_types,
        history_limit=args.history_limit,
    )

if __name__ == "__main__":
    main()
