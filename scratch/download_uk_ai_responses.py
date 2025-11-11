#!/usr/bin/env python3
"""Download published responses to the UK AI regulation consultation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_corpus.connectors.base import DocMeta


@dataclass
class ResponseRecord:
    org: str
    title: str
    url: str
    filename: str
    description: Optional[str] = None
    published_at: Optional[str] = None
    manual: bool = False

RESPONSES: List[ResponseRecord] = [
    ResponseRecord(
        org="Ada Lovelace Institute",
        title="New rules? Response to the UK AI regulation approach",
        url="https://www.adalovelaceinstitute.org/wp-content/uploads/2024/10/Ada-Lovelace-Institute-New-rules.pdf",
        filename="ada_lovelace_new_rules.pdf",
    ),
    ResponseRecord(
        org="Open Rights Group",
        title="Response to Establishing a pro-innovation approach to regulating AI",
        url="https://www.openrightsgroup.org/app/uploads/2022/10/ORG-response-to-AI-policy-paper.pdf",
        filename="open_rights_group_ai_response.pdf",
    ),
    ResponseRecord(
        org="Trades Union Congress",
        title="Consultation response to the AI White Paper",
        url="https://www.tuc.org.uk/sites/default/files/2023-06/TUC%20response%20to%20AI%20White%20Paper.pdf",
        filename="tuc_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="Law Society of England and Wales",
        title="A pro-innovation approach to AI regulation",
        url="https://prdsitecore93.azureedge.net/-/media/files/campaigns/consultation-responses/law-society-response---a-pro-innovation-approach-to-ai-regulation.pdf?rev=e962a9a45c7e4b8888b5ed1e68ca2433&hash=69686E2FB78F2A914CE4364B47D5D111",
        filename="law_society_ai_regulation_response.pdf",
        manual=True,
    ),
    ResponseRecord(
        org="Public Law Project",
        title="Response to the AI White Paper consultation",
        url="https://publiclawproject.org.uk/content/uploads/2023/06/Public-Law-Project-AI-white-paper-consultation-response.pdf",
        filename="public_law_project_ai_response.pdf",
    ),
    ResponseRecord(
        org="Nursing and Midwifery Council",
        title="Response to the Office for Artificial Intelligence consultation",
        url="https://www.nmc.org.uk/globalassets/sitedocuments/consultations/2023/ai-consultation-response.pdf",
        filename="nmc_ai_consultation_response.pdf",
    ),
    ResponseRecord(
        org="Royal Academy of Engineering / National Engineering Policy Centre",
        title="Regulation of artificial intelligence (AI) consultation response",
        url="https://raeng.org.uk/media/eveleecu/ofai-ai-white-paper-consultation-nepc-response.pdf",
        filename="nepc_ai_regulation_response.pdf",
    ),
    ResponseRecord(
        org="Publishers Association",
        title="White paper on AI regulation consultation response",
        url="https://www.publishers.org.uk/wp-content/uploads/2023/06/21-June-2023-White-paper-on-AI-regulation-PA-consultation-response-2.pdf",
        filename="publishers_association_ai_response.pdf",
        manual=True,
    ),
    ResponseRecord(
        org="Association of Chartered Certified Accountants (ACCA)",
        title="Response to the UK AI regulation white paper",
        url="https://www.accaglobal.com/content/dam/ACCA_Global/professional-insights/artificial-intelligence/PI-AI-REGULATION%E2%80%93SUMMARY%20v6.pdf",
        filename="acca_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="Which?",
        title="Which? response to DSIT's A pro-innovation approach to AI white paper",
        url="https://media.product.which.co.uk/prod/files/file/gm-41b54070-d21e-4fce-ada8-bbbb6c4fda83-ai-white-paper-consultation-response.pdf",
        filename="which_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="Professional Standards Authority",
        title="Response to the Government consultation on the White Paper: A pro-innovation approach to AI regulation",
        url="https://www.professionalstandards.org.uk/sites/default/files/attachments/psa-response-to-Government-consultation-on-White-Paper-on-Artificial-Intelligence-regulation.pdf",
        filename="psa_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="UK Finance",
        title="AI Whitepaper – UK Finance response",
        url="https://www.ukfinance.org.uk/system/files/2023-07/AI%20Whitepaper%20%E2%80%93%20UK%20Finance%20response.pdf",
        filename="uk_finance_ai_whitepaper_response.pdf",
    ),
    ResponseRecord(
        org="Information Commissioner's Office",
        title="Response to the Government's AI White Paper",
        url="https://ico.org.uk/media2/migrated/4024792/ico-response-ai-white-paper-20230304.pdf",
        filename="ico_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="Digital Regulation Cooperation Forum (CMA, Ofcom, ICO, FCA)",
        title="DRCF submission to the AI White Paper consultation",
        url="https://www.drcf.org.uk/siteassets/drcf/pdf-files/drcf-ai-white-paper-submission-.pdf",
        filename="drcf_ai_white_paper_submission.pdf",
        manual=True,
    ),
    ResponseRecord(
        org="Office for Nuclear Regulation",
        title="Pro-innovation approach to AI regulation",
        url="https://www.onr.org.uk/media/v45dkpu2/onr-pro-innovation-approach-to-ai-regulation-paper.pdf",
        filename="onr_ai_regulation_response.pdf",
    ),
    ResponseRecord(
        org="Competition and Markets Authority",
        title="CMA response on a pro-innovation approach to AI regulation",
        url="https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1160272/AI_regulation_-_a_pro-innovation_approach.pdf",
        filename="cma_ai_regulation_response.pdf",
    ),
    ResponseRecord(
        org="All-Party Parliamentary Group on AI / Big Innovation Centre",
        title="Feedback on the Government response to the AI White Paper",
        url="https://biginnovationcentre.com/wp-content/uploads/2024/05/Final-26-of-Feb-APPG-AI-WHITE-PAPER_compressed-1.pdf",
        filename="appg_ai_white_paper_feedback.pdf",
    ),
    ResponseRecord(
        org="Centre for Information Policy Leadership",
        title="Response to the DSIT public consultation on the AI framework",
        url="https://www.informationpolicycentre.com/uploads/5/7/1/0/57104281/cipl_response_to_dsit_public_consultation_on_ai_framework_-_21_june_2023.pdf",
        filename="cipl_dsit_ai_framework_response.pdf",
    ),
    ResponseRecord(
        org="BILETA (British and Irish Law, Education and Technology Association)",
        title="Response to the AI regulation white paper",
        url="https://www.research.herts.ac.uk/ws/portalfiles/portal/46728996/BILETA_Response_to_White_Paper_AI_Regulation_A_Proinnovation_Approach.pdf",
        filename="bileta_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="Liberty",
        title="Written submission to 'A pro-innovation approach to AI regulation'",
        url="https://www.libertyhumanrights.org.uk/wp-content/uploads/2023/07/Libertys-Written-Submission-to-AI-White-Paper-19th-June-2023.pdf",
        filename="liberty_ai_white_paper_submission.pdf",
    ),
    ResponseRecord(
        org="Big Brother Watch",
        title="Response to Government White Paper on AI",
        url="https://www.bigbrotherwatch.org.uk/wp-content/uploads/2023/06/Big-Brother-Watch-response-to-Govt-White-Paper-on-AI.pdf",
        filename="big_brother_watch_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="JUSTICE (UK legal reform charity)",
        title="Response to DSIT AI White Paper consultation",
        url="https://cdn.prod.website-files.com/67becde70dae19a9e5ea2bc3/689cdac713c74b694a7ed23f_JUSTICE-DSIT-AI-Consultation-response.pdf",
        filename="justice_dsit_ai_consultation_response.pdf",
        manual=True,
    ),
    ResponseRecord(
        org="Equality and Human Rights Commission",
        title="A pro-innovation approach to AI regulation – consultation response",
        url="https://www.equalityhumanrights.com/sites/default/files/2023/Department%20for%20Science%2C%20Innovation%20and%20Technology%20-%20A%20pro-innovation%20approach%20to%20AI%20regulation%20White%20Paper%20consultation%20response%2C%2023%20June%202023.docx",
        filename="ehrc_ai_white_paper_response.docx",
    ),
    ResponseRecord(
        org="Market Research Society",
        title="Response to AI Regulation White Paper consultation",
        url="https://www.mrs.org.uk/pdf/MRS%20Response%20AI%20Regulation%20White%20Paper%20Consultation.pdf",
        filename="mrs_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="Academy of Medical Sciences",
        title="Response to AI White Paper consultation",
        url="https://acmedsci.ac.uk/file-download/21989734",
        filename="academy_medical_sciences_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="Association of British HealthTech Industries",
        title="ABHI submission to the UK AI consultation",
        url="https://www.abhi.org.uk/media/okaimhbv/abhi-uk-ai-consultation-2023-final.pdf",
        filename="abhi_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="Authors’ Licensing & Collecting Society",
        title="ALCS response to the AI White Paper",
        url="https://d16dqzv7ay57st.cloudfront.net/uploads/2023/06/ALCS-AI-white-paper-response.pdf",
        filename="alcs_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="ICAEW (Institute of Chartered Accountants in England and Wales)",
        title="ICAEW representation on a pro-innovation approach to AI regulation",
        url="https://www.icaew.com/-/media/corporate/files/technical/icaew-representations/2023/icaew-rep-062-23-a-pro-innovation-approach-to-ai-regulation.ashx",
        filename="icaew_ai_regulation_response.pdf",
    ),
    ResponseRecord(
        org="Royal Statistical Society",
        title="RSS response to the Department for Science, Innovation and Technology",
        url="https://rss.org.uk/RSS/media/File-library/Policy/2023/RSS-AI-white-paper-response-v2-2.pdf",
        filename="rss_ai_white_paper_response.pdf",
    ),
    ResponseRecord(
        org="CBI (Confederation of British Industry)",
        title="AI regulation: a pro-innovation approach consultation response",
        url="https://prod.cbi.org.uk/media/bdvlndze/ai-regulation-a-pro-innovation-approach-cbi-consultation-response.pdf",
        filename="cbi_ai_regulation_response.pdf",
        manual=True,
    ),
    ResponseRecord(
        org="Institution of Engineering and Technology (IET)",
        title="AI regulation: a pro-innovation approach consultation response",
        url="https://www.theiet.org/media/11597/s1195-ai-regulation-a-pro-innovation-approach.pdf",
        filename="iet_ai_regulation_response.pdf",
        manual=True,
    ),
]

def iter_responses(include_manual: bool, selection: Iterable[str] | None) -> Iterable[ResponseRecord]:
    for item in RESPONSES:
        if isinstance(item, dict):
            data = dict(item)
            manual = data.pop("requires_manual", data.get("manual", False))
            data.pop("manual", None)
            item = ResponseRecord(manual=manual, **data)
        if not include_manual and item.manual:
            log(f"[skip] {item.org} (manual download required)")
            continue
        if selection:
            needle = {s.lower() for s in selection}
            haystack = f"{item.org} {item.title}".lower()
            if not any(term in haystack for term in needle):
                continue
        yield item


def fetch(session: requests.Session, url: str) -> bytes:
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def log(message: str) -> None:
    print(f"[responses] {message}", file=sys.stderr)


def build_docmeta(record: ResponseRecord, out_dir: Path) -> DocMeta:
    urls: Dict[str, str] = {"pdf": record.url}
    extra: Dict[str, object] = {
        "organization": record.org,
        "title": record.title,
        "local_filename": (out_dir / record.filename).name,
        "local_path": str((out_dir / record.filename).resolve()),
    }
    if record.description:
        extra["description"] = record.description
    if record.published_at:
        extra["published_at"] = record.published_at
    extra["manual_download"] = record.manual

    return DocMeta(
        source="hand_curated_responses",
        collection_id="uk_ai_regulation_responses",
        doc_id=f"response:{record.filename}",
        title=record.title,
        submitter=record.org,
        submitter_type=None,
        org=record.org,
        submitted_at=record.published_at,
        language="en",
        urls=urls,
        extra=extra,
    )


def download_all(target_dir: Path, include_manual: bool, selection: Iterable[str] | None) -> List[DocMeta]:
    target_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    outputs: List[DocMeta] = []
    for item in iter_responses(include_manual, selection):
        dest = target_dir / item.filename
        try:
            data = fetch(session, item.url)
        except Exception as exc:  # noqa: BLE001
            log(f"[fail] {item.org} – {item.url} ({exc})")
            continue
        dest.write_bytes(data)
        log(f"[ok]   {item.org} -> {dest.name} ({len(data)} bytes)")
        outputs.append(build_docmeta(item, target_dir))
    return outputs


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UK AI regulation consultation responses.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/comments/gov_uk/hand_curated_responses/uk_ai_regulation_responses/raw"),
        help="Directory to store downloaded files (default: data/comments/gov_uk/hand_curated_responses/uk_ai_regulation_responses/raw).",
    )
    parser.add_argument(
        "--meta-file",
        type=Path,
        help="Path to write JSONL metadata (default derived from the output directory).",
    )
    parser.add_argument(
        "--include-manual",
        action="store_true",
        help="Attempt to download files that are known to require manual access (may fail with 403).",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        help="Only download responses whose organisation/title matches any of the provided terms.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print collected DocMeta records as JSON to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    docmeta_items = download_all(args.output, args.include_manual, args.filter)
    collection_dir = args.output.parent
    default_meta = collection_dir / f"{collection_dir.name}.meta.jsonl"
    meta_path = args.meta_file or default_meta
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as fh:
        for doc in docmeta_items:
            fh.write(json.dumps(asdict(doc)))
            fh.write("\n")
    log(f"[meta] wrote {len(docmeta_items)} records to {meta_path}")
    if args.print_json:
        json.dump([asdict(item) for item in docmeta_items], sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
