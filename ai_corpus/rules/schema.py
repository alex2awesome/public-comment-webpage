"""
Shared schema helpers for exporting rule versions across connectors.
"""

from __future__ import annotations

import json
from typing import Any, Dict

RULE_VERSION_BASE_COLUMNS = [
    "scrape_mode",
    "source",
    "fr_document_number",
    "title",
    "type",
    "publication_date",
    "agency",
    "fr_url",
    "docket_id",
    "docket_ids",
    "regs_url",
    "regs_document_id",
    "regs_object_id",
    "comment_start_date",
    "comment_due_date",
    "comment_status",
    "comment_active",
    "is_rfi_rfc",
    "rfi_rfc_label",
    "rfi_rfc_matched_in",
    "details",
    "abstract",
    "action",
    "dates",
    "supplementary_information",
    "xml_dict",
]

RULE_VERSION_HISTORY_COLUMNS = [
    "history_parent_docket",
    "history_parent_fr_doc",
    "history_stage",
    "history_relationship",
    "history_rank",
    "mentions_comment_response",
    "comment_citation_snippet",
]

RULE_VERSION_COLUMNS = RULE_VERSION_BASE_COLUMNS + RULE_VERSION_HISTORY_COLUMNS

BOOL_COLUMNS = {"comment_active", "is_rfi_rfc", "mentions_comment_response"}
LIST_COLUMNS = {"docket_ids"}


def normalize_rule_row(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all rule-version columns exist and serialize complex values so they
    survive CSV/JSON exports.
    """

    row = {col: "" for col in RULE_VERSION_COLUMNS}
    for key, value in data.items():
        if key not in row or value is None:
            continue
        if key in BOOL_COLUMNS and isinstance(value, bool):
            row[key] = "TRUE" if value else "FALSE"
        elif key in LIST_COLUMNS and isinstance(value, list):
            row[key] = "; ".join(str(v) for v in value if v is not None)
        elif isinstance(value, (dict, list)):
            row[key] = json.dumps(value, ensure_ascii=False)
        else:
            row[key] = value
    return row


__all__ = ["RULE_VERSION_COLUMNS", "normalize_rule_row"]
