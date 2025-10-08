#!/usr/bin/env python3
"""
Harvest Regulations.gov public comments for each row of an input CSV produced by get_regulations.py.

Resolution strategy per row:
  A) If regs_object_id present -> /v4/comments?filter[commentOnId]=<objectId>
  B) Else if regs_document_id present -> resolve objectId via /v4/documents/{documentId},
     then /v4/comments?filter[commentOnId]=<objectId>
  C) (optional) --allow-docket-fallback -> /v4/comments?filter[docketId]=<docketId>&filter[documentType]=Public Submission

Writes one CSV with one row per comment.
Adds 'harvest_method' showing which path was used.

Usage:
  python scripts/harvesting/collect_comments.py \
    --input path/to/federal_rulemaking.csv \
    --regs-key $REGS_API_KEY \
    --output comments.csv \
    --allow-docket-fallback
"""

import argparse, csv, logging, os, re, time, json
from typing import Dict, Any, List, Optional, Tuple
import requests

REGS_BASE = "https://api.regulations.gov/v4"
UA = "regulations-comment-harvester/1.2 (contact: you@example.org)"

logger = logging.getLogger("harvester")

# -------------------- HTTP helpers with backoff --------------------

def backoff_get(url: str, headers: Dict[str, str], params: Dict[str, Any], timeout: int = 30, max_attempts: int = 6) -> Optional[requests.Response]:
    """
    GET with exponential backoff on 5xx/429/408.
    For other 4xx, return the response immediately (so caller can branch).
    Returns None only after exhausting attempts or connection-level failures.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code in (408, 429) or 500 <= r.status_code < 600:
                if attempt >= max_attempts:
                    logger.warning("GET failed after %d attempts: %s %s params=%s", attempt, url, r.status_code, params)
                    return None
                sleep_s = min(60, (2 ** (attempt - 1)) + 0.1 * attempt)
                time.sleep(sleep_s)
                continue
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt >= max_attempts:
                logger.warning("GET failed after %d attempts: %s %s params=%s", attempt, url, e, params)
                return None
            sleep_s = min(60, (2 ** (attempt - 1)) + 0.1 * attempt)
            time.sleep(sleep_s)

def regs_headers(api_key: str) -> Dict[str, str]:
    return {
        "X-Api-Key": api_key,
        "Accept": "application/vnd.api+json",
        "User-Agent": UA,
    }

# -------------------- API calls --------------------

def resolve_object_id_from_document(api_key: str, document_id: str) -> Optional[str]:
    """
    GET /v4/documents/{documentId}?fields[documents]=objectId
    Returns objectId or None.
    """
    headers = regs_headers(api_key)
    url = f"{REGS_BASE}/documents/{document_id}"
    params = {"fields[documents]": "objectId"}
    r = backoff_get(url, headers=headers, params=params)
    if not r:
        return None
    if r.status_code != 200:
        logger.debug("Document fetch %s returned %s", document_id, r.status_code)
        return None
    js = r.json() or {}
    data = js.get("data") or {}
    attrs = data.get("attributes") or {}
    obj = attrs.get("objectId")
    if obj:
        return obj
    # Fallback: try without sparse fields in case some backends ignore them
    r2 = backoff_get(url, headers=headers, params={})
    if not r2 or r2.status_code != 200:
        return None
    js2 = r2.json() or {}
    data2 = js2.get("data") or {}
    attrs2 = data2.get("attributes") or {}
    return attrs2.get("objectId")

def page_through_comments(
    api_key: str,
    base_params: Dict[str, Any],
    max_comments: Optional[int],
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Iterate /v4/comments with the provided filters in base_params.
    Returns (status_code, comments_list). status_code may be 200, 400, 404, etc.
    """
    out: List[Dict[str, Any]] = []
    page = 1
    page_size = min(250, int(base_params.get("page[size]", 250)))
    headers = regs_headers(api_key)
    url = f"{REGS_BASE}/comments"
    last_status = 200

    while True:
        params = dict(base_params)
        params["page[size]"] = page_size
        params["page[number]"] = page
        r = backoff_get(url, headers=headers, params=params)
        if not r:
            return (0, out)  # network exhaustion
        last_status = r.status_code
        if r.status_code != 200:
            return (last_status, out)
        js = r.json() or {}
        data = js.get("data") or []
        if not data:
            break
        out.extend(data)
        if max_comments and len(out) >= max_comments:
            return (last_status, out[:max_comments])
        page += 1
    return (last_status, out)

def fetch_attachments_for_comment(api_key: str, comment_id: str) -> List[str]:
    headers = regs_headers(api_key)
    url = f"{REGS_BASE}/comments/{comment_id}/attachments"
    r = backoff_get(url, headers=headers, params={})
    if not r or r.status_code != 200:
        return []
    js = r.json() or {}
    out: List[str] = []
    for att in js.get("data", []):
        links = att.get("links") or {}
        rel = links.get("self")
        if rel:
            out.append(rel)
    return out

# -------------------- Row -> comments resolution --------------------

DOC_ID_RE = re.compile(r"/document/([A-Z0-9_\-]+)")

def parse_doc_id_from_url(regs_url: Optional[str]) -> Optional[str]:
    if not regs_url:
        return None
    m = DOC_ID_RE.search(regs_url)
    return m.group(1) if m else None

# -------------------- CSV I/O --------------------

COMMENT_COLUMNS = [
    # provenance from input row
    "input_row_index",
    "fr_document_number",
    "fr_url",
    "docket_id",
    "regs_url",
    "regs_document_id",
    "regs_object_id",
    "harvest_method",
    # comment fields
    "comment_id",
    "docketId",
    "commentOn",
    "commentOnDocumentId",
    "postedDate",
    "receiveDate",
    "title",
    "comment",
    "organization",
    "firstName",
    "lastName",
    "category",
    "city",
    "stateProvinceRegion",
    "zip",
    "openForComment",
    "trackingNbr",
    "attachments_count",
    "attachments_urls",
]

def coerce(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, (dict, list)):
        try:
            return json.dumps(s, ensure_ascii=False)
        except Exception:
            return str(s)
    return str(s)

def harvest_for_row(
    row_index: int,
    row: Dict[str, str],
    api_key: str,
    max_comments_per_row: Optional[int],
    include_attachments: bool,
    allow_docket_fallback: bool,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Returns (method_used, list_of_records) for this input row.
    """
    # Common provenance
    base_common = {
        "input_row_index": str(row_index),
        "fr_document_number": row.get("fr_document_number", ""),
        "fr_url": row.get("fr_url", ""),
        "docket_id": row.get("docket_id", ""),
        "regs_url": row.get("regs_url", ""),
        "regs_document_id": row.get("regs_document_id", ""),
        "regs_object_id": row.get("regs_object_id", ""),
    }

    # 1) Prefer objectId directly in the CSV
    object_id = (row.get("regs_object_id") or "").strip() or None

    # 2) If missing, try to resolve via /v4/documents/{documentId}
    document_id = (row.get("regs_document_id") or "").strip() or None
    if not document_id and row.get("regs_url"):
        document_id = parse_doc_id_from_url(row.get("regs_url"))

    if not object_id and document_id:
        object_id = resolve_object_id_from_document(api_key, document_id)
        if object_id:
            logger.debug("[Row %d] Resolved objectId %s from document %s", row_index, object_id, document_id)

    # Try commentOnId if we have it
    if object_id:
        query = {"filter[commentOnId]": object_id, "sort": "postedDate"}
        status, comments = page_through_comments(api_key, query, max_comments=max_comments_per_row)
        logger.log(logging.INFO if comments else logging.DEBUG,
                   "[%s] Row %d object_id: fetched %d comment(s) (status %s)",
                   base_common["fr_document_number"] or "?", row_index, len(comments), status)

        if comments:
            records = flatten_comments(
                comments, base_common, method="object_id", api_key=api_key, include_attachments=include_attachments
            )
            return ("object_id", records)

    # Optional docket-wide fallback (can be noisy)
    docket = (row.get("docket_id") or "").strip() or None
    if allow_docket_fallback and docket:
        query = {"filter[docketId]": docket, "filter[documentType]": "Public Submission", "sort": "postedDate"}
        status, comments = page_through_comments(api_key, query, max_comments=max_comments_per_row)
        logger.log(logging.INFO if comments else logging.DEBUG,
                   "[%s] Row %d docket: fetched %d comment(s) (status %s)",
                   base_common["fr_document_number"] or "?", row_index, len(comments), status)
        if comments:
            records = flatten_comments(
                comments, base_common, method="docket", api_key=api_key, include_attachments=include_attachments
            )
            return ("docket", records)

    # Nothing found
    return ("none", [])

def flatten_comments(
    comments: List[Dict[str, Any]],
    base_common: Dict[str, str],
    method: str,
    api_key: str,
    include_attachments: bool,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    attach_cache: Dict[str, Tuple[int, str]] = {}

    if include_attachments:
        for c in comments:
            cid = c.get("id")
            urls = fetch_attachments_for_comment(api_key, cid) if cid else []
            attach_cache[cid] = (len(urls), ";".join(urls))

    for c in comments:
        attrs = c.get("attributes", {}) or {}
        cid = c.get("id", "")
        att_count, att_urls = attach_cache.get(cid, (0, ""))
        rec = dict(base_common)
        rec.update({
            "harvest_method": method,
            "comment_id": cid,
            "docketId": coerce(attrs.get("docketId")),
            "commentOn": coerce(attrs.get("commentOn")),
            "commentOnDocumentId": coerce(attrs.get("commentOnDocumentId")),  # echoed for reference
            "postedDate": coerce(attrs.get("postedDate")),
            "receiveDate": coerce(attrs.get("receiveDate")),
            "title": coerce(attrs.get("title")),
            "comment": (attrs.get("comment") or "")[:50000],
            "organization": coerce(attrs.get("organization")),
            "firstName": coerce(attrs.get("firstName")),
            "lastName": coerce(attrs.get("lastName")),
            "category": coerce(attrs.get("category")),
            "city": coerce(attrs.get("city")),
            "stateProvinceRegion": coerce(attrs.get("stateProvinceRegion")),
            "zip": coerce(attrs.get("zip")),
            "openForComment": coerce(attrs.get("openForComment")),
            "trackingNbr": coerce(attrs.get("trackingNbr")),
            "attachments_count": str(att_count),
            "attachments_urls": att_urls,
        })
        out.append(rec)
    return out

def run(
    input_csv: str,
    output_csv: str,
    api_key: str,
    max_comments_per_row: Optional[int],
    include_attachments: bool,
    allow_docket_fallback: bool,
) -> None:
    with open(input_csv, newline="", encoding="utf-8") as f_in, \
         open(output_csv, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=COMMENT_COLUMNS)
        writer.writeheader()

        for idx, row in enumerate(reader, start=1):
            has_any_id = any([
                (row.get("regs_object_id") or "").strip(),
                (row.get("regs_document_id") or "").strip(),
                parse_doc_id_from_url(row.get("regs_url") or ""),
                (allow_docket_fallback and (row.get("docket_id") or "").strip()),
            ])
            if not has_any_id:
                logger.warning("[WARN] No regs identifiers in row %d; skipping", idx)
                continue

            method, records = harvest_for_row(
                row_index=idx,
                row=row,
                api_key=api_key,
                max_comments_per_row=max_comments_per_row,
                include_attachments=include_attachments,
                allow_docket_fallback=allow_docket_fallback,
            )

            if not records:
                identifiers = {
                    "fr_document_number": (row.get("fr_document_number") or "").strip(),
                    "regs_document_id": (row.get("regs_document_id") or "").strip(),
                    "regs_object_id": (row.get("regs_object_id") or "").strip(),
                    "docket_id": (row.get("docket_id") or "").strip(),
                    "regs_url": (row.get("regs_url") or "").strip(),
                    "parsed_doc_id": parse_doc_id_from_url(row.get("regs_url") or "") or "",
                }
                present_bits = [f"{key}={val}" for key, val in identifiers.items() if val]
                id_summary = ", ".join(present_bits) if present_bits else "no identifiers present"
                logger.info(
                    "[OK] Row %d: wrote 0 comment(s) [method=%s, %s]",
                    idx,
                    method,
                    id_summary,
                )
            else:
                for rec in records:
                    writer.writerow(rec)
                logger.info("[OK] Row %d: wrote %d comment(s) [%s]", idx, len(records), method)

# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Harvest Regulations.gov comments using objectId, resolving it from documentId when needed; optional docket fallback.")
    ap.add_argument("--input", required=True, help="Input CSV from get_regulations.py")
    ap.add_argument("--output", required=True, help="Output CSV path for comments")
    ap.add_argument("--regs-key", default=os.getenv("REGS_API_KEY"), help="Regulations.gov API key")
    ap.add_argument("--max-comments-per-row", type=int, default=None, help="Cap number of comments per input row (None = no cap)")
    ap.add_argument("--include-attachments", action="store_true", help="Fetch attachments metadata per comment")
    ap.add_argument("--allow-docket-fallback", action="store_true", help="If set, fallback to filter[docketId] when objectId cannot be resolved")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    if not args.regs_key:
        raise SystemExit("Missing Regulations.gov API key. Provide --regs-key or set REGS_API_KEY.")
    run(
        input_csv=args.input,
        output_csv=args.output,
        api_key=args.regs_key,
        max_comments_per_row=args.max_comments_per_row,
        include_attachments=args.include_attachments,
        allow_docket_fallback=args.allow_docket_fallback,
    )

if __name__ == "__main__":
    main()
