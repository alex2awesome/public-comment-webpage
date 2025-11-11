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

import argparse, csv, logging, os, re, json, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from api_utils import RateLimiter, backoff_get, parse_doc_id_from_url, regs_backoff_get

FR_BASE = "https://www.federalregister.gov/api/v1"
UA = "regulations-comment-harvester/1.2 (contact: you@example.org)"
JSON_HEADERS = {"Accept": "application/json", "User-Agent": UA}
REGS_RATE_LIMITER = RateLimiter(min_interval=1.0)

logger = logging.getLogger("harvester")

# -------------------- API calls --------------------

def resolve_object_id_from_document(api_key: str, document_id: str) -> Optional[str]:
    """
    GET /v4/documents/{documentId}?fields[documents]=objectId
    Returns objectId or None.
    """
    r = regs_backoff_get(
        f"documents/{document_id}",
        api_key=api_key,
        params={"fields[documents]": "objectId"},
        user_agent=UA,
        rate_limiter=REGS_RATE_LIMITER,
        raise_for_status=False,
    )
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
    r2 = regs_backoff_get(
        f"documents/{document_id}",
        api_key=api_key,
        user_agent=UA,
        rate_limiter=REGS_RATE_LIMITER,
        raise_for_status=False,
    )
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
    last_status = 200

    while True:
        params = dict(base_params)
        params["page[size]"] = page_size
        params["page[number]"] = page
        r = regs_backoff_get(
            "comments",
            api_key=api_key,
            params=params,
            user_agent=UA,
            rate_limiter=REGS_RATE_LIMITER,
            raise_for_status=False,
        )
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
    r = regs_backoff_get(
        f"comments/{comment_id}/attachments",
        api_key=api_key,
        user_agent=UA,
        rate_limiter=REGS_RATE_LIMITER,
        raise_for_status=False,
    )
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


def fetch_comment_detail(api_key: str, comment_id: str) -> Optional[Dict[str, Any]]:
    r = regs_backoff_get(
        f"comments/{comment_id}",
        api_key=api_key,
        user_agent=UA,
        rate_limiter=REGS_RATE_LIMITER,
        raise_for_status=False,
    )
    if not r or r.status_code != 200:
        return None
    js = r.json() or {}
    data = js.get("data") or {}
    return data.get("attributes") or {}

# -------------------- Row -> comments resolution --------------------

FR_DOCNUM_RE = re.compile(r"/(20\d{2}-\d{4,7})(?:[/?#]|$)")

def parse_fr_docnum_from_url(fr_url: Optional[str]) -> Optional[str]:
    """
    Best-effort extraction of Federal Register document number from an FR URL.
    Most FR HTML URLs do not include the document number, so this may return None.
    """
    if not fr_url:
        return None
    m = FR_DOCNUM_RE.search(fr_url)
    return m.group(1) if m else None

def resolve_regs_from_fr(
    api_key: str,
    fr_document_number: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Resolve Regulations.gov identifiers from a Federal Register document number.
    Returns (regs_document_id, regs_object_id, regs_url, docket_id).
    """
    if not fr_document_number:
        return None, None, None, None
    url = f"{FR_BASE}/documents/{fr_document_number}.json"
    r = backoff_get(
        url,
        {},
        default_headers=JSON_HEADERS,
        raise_for_status=False,
    )
    if not r or r.status_code != 200:
        return None, None, None, None
    try:
        js = r.json() or {}
    except Exception:
        return None, None, None, None

    regs_info = js.get("regulations_dot_gov_info") or {}
    regs_doc_id = regs_info.get("document_id") or None
    regs_url = js.get("regulations_dot_gov_url") or None
    docket_id = js.get("docket_id") or None

    if not regs_doc_id and regs_url:
        regs_doc_id = parse_doc_id_from_url(regs_url)

    regs_obj_id = None
    if regs_doc_id:
        regs_obj_id = resolve_object_id_from_document(api_key, regs_doc_id)
        if not regs_url:
            regs_url = f"https://www.regulations.gov/document/{regs_doc_id}"

    return regs_doc_id, regs_obj_id, regs_url, docket_id

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
    row: pd.Series,
    api_key: str,
    max_comments_per_row: Optional[int],
    include_attachments: bool,
    allow_docket_fallback: bool,
) -> Tuple[str, List[Dict[str, str]], Dict[str, str]]:
    """
    Returns (method_used, list_of_records, base_common) for this input row.
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
        logger.log(
            logging.INFO if comments else logging.DEBUG,
            "[%s] Row %d object_id: fetched %d comment(s) (status %s)",
            base_common["fr_document_number"] or "?", row_index, len(comments), status
        )

        if comments:
            records = flatten_comments(
                comments, base_common, method="object_id", api_key=api_key, include_attachments=include_attachments
            )
            return ("object_id", records, base_common)

    # Attempt FR -> Regulations.gov resolution if we still lack identifiers
    if not object_id and not document_id:
        fr_docnum = (row.get("fr_document_number") or "").strip() or None
        if not fr_docnum and row.get("fr_url"):
            fr_docnum = parse_fr_docnum_from_url(row.get("fr_url"))

        if fr_docnum:
            r_doc_id, r_obj_id, r_url, r_docket = resolve_regs_from_fr(api_key, fr_docnum)
            if r_doc_id:
                document_id = r_doc_id
            if r_obj_id:
                object_id = r_obj_id
            # Update provenance fields for output
            if r_url:
                base_common["regs_url"] = r_url
            if r_doc_id:
                base_common["regs_document_id"] = r_doc_id
            if r_obj_id:
                base_common["regs_object_id"] = r_obj_id
            if r_docket and not (row.get("docket_id") or "").strip():
                base_common["docket_id"] = r_docket

    # Try again via objectId if we managed to resolve it via FR
    if object_id and (method := "object_id"):
        query = {"filter[commentOnId]": object_id, "sort": "postedDate"}
        status, comments = page_through_comments(api_key, query, max_comments=max_comments_per_row)
        logger.log(
            logging.INFO if comments else logging.DEBUG,
            "[%s] Row %d object_id (FR-resolved): fetched %d comment(s) (status %s)",
            base_common["fr_document_number"] or "?", row_index, len(comments), status
        )

        if comments:
            records = flatten_comments(
                comments, base_common, method="object_id", api_key=api_key, include_attachments=include_attachments
            )
            return ("object_id", records, base_common)

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
            return ("docket", records, base_common)

    # Nothing found
    return ("none", [], base_common)

def flatten_comments(
    comments: List[Dict[str, Any]],
    base_common: Dict[str, str],
    method: str,
    api_key: str,
    include_attachments: bool,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    attach_cache: Dict[str, Tuple[int, str]] = {}
    detail_cache: Dict[str, Dict[str, Any]] = {}

    if include_attachments:
        for c in comments:
            cid = c.get("id")
            urls = fetch_attachments_for_comment(api_key, cid) if cid else []
            attach_cache[cid] = (len(urls), ";".join(urls))

    for c in comments:
        cid = c.get("id", "")
        attrs = detail_cache.get(cid)
        if attrs is None and cid:
            attrs = fetch_comment_detail(api_key, cid) or {}
            detail_cache[cid] = attrs
        if not attrs:
            attrs = c.get("attributes", {}) or {}
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
    resolved_id_file: Optional[str] = None,
) -> None:
    with open(input_csv, newline="", encoding="utf-8") as f_in, \
         open(output_csv, "w", newline="", encoding="utf-8") as f_out, \
         (open(resolved_id_file, "w", newline="", encoding="utf-8") if resolved_id_file else open(os.devnull, "w")) as f_resolved:

        INPUT_COLUMNS = ["fr_document_number", "fr_url", "docket_id", "regs_url", "regs_document_id", "regs_object_id"]
        reader = pd.read_csv(f_in, usecols=INPUT_COLUMNS)
        writer = csv.DictWriter(f_out, fieldnames=COMMENT_COLUMNS)
        writer.writeheader()
        resolved_writer = None
        if resolved_id_file:
            resolved_fields = [
                "input_row_index",
                "fr_document_number",
                "fr_url",
                "fr_document_url",
                "docket_id",
                "regs_url",
                "regs_document_id",
                "regs_object_id",
            ]
            resolved_writer = csv.DictWriter(f_resolved, fieldnames=resolved_fields)
            resolved_writer.writeheader()

        for idx, row in reader.iterrows():
            row = row.fillna("")
            has_any_id = any([
                (row.get("regs_object_id") or "").strip(),
                (row.get("regs_document_id") or "").strip(),
                parse_doc_id_from_url(row.get("regs_url") or ""),
                (row.get("fr_document_number") or "").strip(),
                (row.get("fr_url") or "").strip(),
                (allow_docket_fallback and (row.get("docket_id") or "").strip()),
            ])
            if not has_any_id:
                logger.warning("[WARN] No regs identifiers in row %d; skipping", idx)
                continue

            method, records, base_common = harvest_for_row(
                row_index=idx,
                row=row,
                api_key=api_key,
                max_comments_per_row=max_comments_per_row,
                include_attachments=include_attachments,
                allow_docket_fallback=allow_docket_fallback,
            )

            # Write resolved identifiers row if requested
            if resolved_writer is not None:
                resolved_row = dict(base_common)
                resolved_row["fr_document_url"] = base_common.get("fr_url", "")
                resolved_writer.writerow({
                    "input_row_index": resolved_row.get("input_row_index", ""),
                    "fr_document_number": resolved_row.get("fr_document_number", ""),
                    "fr_url": resolved_row.get("fr_url", ""),
                    "fr_document_url": resolved_row.get("fr_document_url", ""),
                    "docket_id": resolved_row.get("docket_id", ""),
                    "regs_url": resolved_row.get("regs_url", ""),
                    "regs_document_id": resolved_row.get("regs_document_id", ""),
                    "regs_object_id": resolved_row.get("regs_object_id", ""),
                })

            if not records:
                identifiers = {
                    "fr_document_number": (row.get("fr_document_number") or base_common["fr_document_number"] or "").strip(),
                    "regs_document_id": (row.get("regs_document_id") or base_common["regs_document_id"] or "").strip(),
                    "regs_object_id": (row.get("regs_object_id") or base_common["regs_object_id"] or "").strip(),
                    "docket_id": (row.get("docket_id") or base_common["docket_id"] or "").strip(),
                    "regs_url": (row.get("regs_url") or base_common["regs_url"] or "").strip(),
                    "parsed_doc_id": parse_doc_id_from_url(row.get("regs_url") or base_common["regs_url"] or "") or "",
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
    ap.add_argument("--resolved-id-file", type=str, default=None, help="Optional CSV path to write resolved identifiers per row (base_common + fr_document_url)")
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
        resolved_id_file=args.resolved_id_file,
    )

if __name__ == "__main__":
    main()
