"""
Rule-based classifier for regulatory comments.
Classifies into:
  - definitely_single:              one commenter, one self-contained comment
  - definitely_campaign:            bundled form letters from many people
  - references_other_submission:    this record references or supplements a separate submission
                                    (not self-contained; exclude from substantive text analysis)
  - uncertain:                      needs LLM review

Input: a single raw string with <<PART N>> and <<PAGE N>> markers, plus optional title.
"""

import re
from typing import Optional


# ── Parsing ─────────────────────────────────────────────────────────


def _parse_comment(raw: str) -> dict:
    """
    Parse raw comment text into structured parts.
    Returns {
        'part1_text': str,
        'parts': list[str],          # full text of each part 2+
        'pages': list[list[str]],    # pages[i] = page texts for part i+2
        'num_parts': int,
        'total_pages': int,
    }
    """
    if '<<PART' in raw:
        part_splits = re.split(r"<<PART\s+(\d+)>>", raw)
    else:
        part_splits = re.split(r"<<COMMENT\s+(\d+)>>", raw)
    parts_dict = {}
    i = 1
    while i < len(part_splits) - 1:
        part_num = int(part_splits[i])
        part_text = part_splits[i + 1]
        parts_dict[part_num] = part_text
        i += 2

    part1_text = parts_dict.get(1, "").strip()
    part_texts = []
    all_pages = []
    total_pages = 0

    for pnum in sorted(parts_dict.keys()):
        if pnum == 1:
            continue
        full_text = parts_dict[pnum]
        part_texts.append(full_text.strip())
        page_splits = re.split(r"<<PAGE\s+\d+>>", full_text)
        pages = [p.strip() for p in page_splits if p.strip()]
        all_pages.append(pages)
        total_pages += len(pages)

    return {
        "part1_text": part1_text,
        "parts": part_texts,
        "pages": all_pages,
        "num_parts": len(parts_dict),
        "total_pages": total_pages,
    }


# ── Main classifier ────────────────────────────────────────────────


def classify_comment(text: str, title: Optional[str] = None) -> dict:
    """
    Classify a single regulatory comment.

    Args:
        text:  raw comment string with <<PART N>> / <<PAGE N>> markers
        title: optional title field from the dataset

    Returns {
        'classification': str,
        'reasons': list[str],
        'campaign_count_estimate': int | None,
        'ref_other_info': dict | None,
        'num_parts': int,
        'total_pages': int,
    }
    """
    parsed = _parse_comment(text)
    part1_text = parsed["part1_text"]
    parts = parsed["parts"]
    pages = parsed["pages"]

    reasons = []
    campaign_signals = 0
    single_signals = 0
    ref_other_signals = 0
    campaign_count_estimate = None
    ref_other_info = None

    # ── Clean up part1 ──────────────────────────────────────────────
    part1_clean = part1_text.strip()
    meta_count = _extract_meta_count(part1_clean)
    if meta_count is not None and meta_count > 10:
        campaign_signals += 2
        campaign_count_estimate = meta_count
        reasons.append(f"metadata count field = {meta_count}")

    see_attached = any(
        re.search(
            r"see\s+(?:attached|the\s+attached|our\s+(?:uploaded|attached)|uploaded)",
            line, re.IGNORECASE,
        )
        for line in part1_clean.split("\n")
    )

    part1_body = _strip_metadata(part1_clean)
    part1_word_count = len(part1_body.split())

    # ── Part 1 substantive text ─────────────────────────────────────
    if part1_body and part1_word_count > 20:
        single_signals += 2
        reasons.append(f"PART 1 contains substantive text ({part1_word_count} words)")

    # ── No attachments ──────────────────────────────────────────────
    has_meaningful_parts = any(_has_meaningful_text(p) for p in parts)
    if part1_word_count > 20 and not has_meaningful_parts:
        single_signals += 2
        reasons.append("no meaningful attachment content (text-box-only submission)")

    # ── Short comment fast-track ────────────────────────────────────
    if parsed["total_pages"] <= 2 and campaign_signals == 0 and meta_count in (None, 0):
        single_signals += 1
        reasons.append(f"short comment ({parsed['total_pages']} pages)")

    # ── References-other-submission detection ────────────────────────
    ref_other_result = _detect_references_other(part1_body, parts)
    if ref_other_result:
        ref_other_signals += 3
        ref_other_info = ref_other_result
        reasons.append(f"references other submission: {ref_other_result['indicators']}")

    # ── Campaign cover letter detection ─────────────────────────────
    if parts and ref_other_signals == 0:
        cover_text = parts[0][:3000]
        campaign_phrases = _detect_campaign_phrases(cover_text)
        if campaign_phrases:
            high_signal = [p for p in campaign_phrases if p != "on behalf of"]
            if len(campaign_phrases) >= 2 or high_signal:
                campaign_signals += 3
                reasons.append(f"cover letter campaign language: {campaign_phrases}")
            else:
                reasons.append(f"weak campaign language (only: {campaign_phrases})")

        explicit_count = _extract_explicit_count(cover_text)
        if explicit_count is not None:
            campaign_signals += 3
            campaign_count_estimate = explicit_count
            reasons.append(f"explicit count in cover letter: {explicit_count}")

    # ── Repetition detection across pages ───────────────────────────
    all_page_texts = [p for part_pages in pages for p in part_pages]
    if len(all_page_texts) >= 3:
        repetition_ratio = _detect_page_repetition(all_page_texts)
        if repetition_ratio is not None and repetition_ratio > 0.5:
            campaign_signals += 3
            reasons.append(f"high repetition across pages ({repetition_ratio:.0%} similar)")
        elif repetition_ratio is not None and repetition_ratio < 0.1:
            single_signals += 1
            reasons.append("low repetition across pages")

    # ── OCR garbage detection ───────────────────────────────────────
    for i, part in enumerate(parts):
        if _is_ocr_garbage(part):
            single_signals += 1
            reasons.append(f"PART {i+2} appears to be failed OCR (likely image/handwriting)")
            break

    # ── Screenshot detection ────────────────────────────────────────
    for i, part in enumerate(parts):
        if _is_screenshot(part):
            reasons.append(f"PART {i+2} appears to be a phone screenshot")
            break

    # ── Single substantive attachment ───────────────────────────────
    if see_attached and parts and campaign_signals == 0 and ref_other_signals == 0:
        substantive_parts = [p for p in parts if _has_meaningful_text(p)]
        if substantive_parts:
            has_no_repetition = True
            if len(all_page_texts) >= 3:
                rep = _detect_page_repetition(all_page_texts)
                if rep is not None and rep > 0.5:
                    has_no_repetition = False
            if has_no_repetition:
                single_signals += 2
                reasons.append("attachment(s) contain substantive text with no campaign signals")

    # ── Title-based signals ─────────────────────────────────────────
    if title:
        title_lower = title.strip().lower()
        for pattern in [
            r"mass comment campaign", r"write[\s-]*in campaign",
            r"mass mailing", r"multiple letters", r"form letter", r"mass comment",
        ]:
            if re.search(pattern, title_lower):
                campaign_signals += 3
                reasons.append(f"title matches campaign pattern: '{pattern}'")
                break

        if title_lower in [
            "comment", "anonymous public comment", "anonymous",
            "see attached", "submitted electronically via erulemaking portal",
        ]:
            reasons.append("generic/uninformative title")

    # ── Decision logic ──────────────────────────────────────────────
    if ref_other_signals >= 3:
        classification = "references_other_submission"
    elif campaign_signals >= 3:
        classification = "definitely_campaign"
    elif single_signals >= 2 and campaign_signals == 0:
        classification = "definitely_single"
    else:
        classification = "uncertain"

    return {
        "classification": classification,
        "reasons": reasons,
        "campaign_count_estimate": campaign_count_estimate,
        "ref_other_info": ref_other_info,
        "num_parts": parsed["num_parts"],
        "total_pages": parsed["total_pages"],
    }


# ── Helper functions ────────────────────────────────────────────────

def _extract_meta_count(part1: str) -> Optional[int]:
    lines = [l.strip() for l in part1.split("\n") if l.strip()]
    if len(lines) >= 2:
        try:
            return int(lines[1])
        except ValueError:
            pass
    return None

def _strip_metadata(part1: str) -> str:
    lines = part1.split("\n")
    skip = [r"^\s*false\s*$", r"^\s*true\s*$", r"^\s*\d+\s*$",
            r"^\s*see\s+attached\s*(?:file)?(?:\(s\))?\s*\.?\s*$",
            r"^\s*please\s+see\s+(?:attached|our\s+uploaded)\s*(?:file)?(?:\(s\))?\s*\.?\s*$",
            r"^\s*$"]
    return "\n".join(
        l for l in lines if not any(re.match(p, l, re.IGNORECASE) for p in skip)
    ).strip()

def _has_meaningful_text(text: str) -> bool:
    if not text:
        return False
    return len(re.findall(r"[a-zA-Z]{3,}", text)) > 10

def _detect_campaign_phrases(text: str) -> list[str]:
    patterns = [
        (r"on behalf of", "on behalf of"),
        (r"submitting\s+[\d,]+\s+comments", "submitting N comments"),
        (r"[\d,]+\s+individual\s+(public\s+)?comments", "N individual comments"),
        (r"split across\s+\d+\s+attachments?", "split across N attachments"),
        (r"split across\s+\d+\s+pdf", "split across N PDFs"),
        (r"separated into\s+\d+\s+separate", "separated into N separate"),
        (r"demonstrating opposition", "demonstrating opposition"),
        (r"please confirm.+received all", "confirm receipt of all"),
        (r"the majority read as follows", "majority read as follows"),
        (r"individual comments may differ", "individual comments may differ"),
        (r"sample\s+(attached|letter)", "sample attached/letter"),
    ]
    found = []
    text_lower = text.lower()
    for pattern, label in patterns:
        if re.search(pattern, text_lower):
            found.append(label)
    return found

def _extract_explicit_count(text: str) -> Optional[int]:
    patterns = [
        r"submitting\s+([\d,]+)\s+comments",
        r"([\d,]+)\s+individual\s+(public\s+)?comments",
        r"([\d,]+)\s+comments\s+demonstrating",
        r"attached.+?([\d,]+)\s+comments",
        r"find\s+([\d,]+)\s+individual",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return int(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None

def _detect_page_repetition(page_texts: list[str]) -> Optional[float]:
    normalized = [_normalize_for_comparison(p) for p in page_texts]
    normalized = [n for n in normalized if len(n) > 50]
    if len(normalized) < 3:
        return None
    ref = normalized[0]
    similar = sum(1 for p in normalized[1:] if _simple_similarity(ref, p) > 0.7)
    return similar / (len(normalized) - 1)

def _normalize_for_comparison(text: str) -> str:
    lines = text.split("\n")
    filtered = []
    for line in lines:
        lc = line.strip()
        if len(lc) < 40:
            continue
        if re.match(r"^\d+\s+\w+", lc) and len(lc) < 60:
            continue
        if re.match(r"^[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}", lc):
            continue
        filtered.append(lc.lower())
    return " ".join(filtered)

def _simple_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

def _is_ocr_garbage(text: str) -> bool:
    if not text or len(text) < 20:
        return False
    noise = sum(1 for c in text if c in "~=|><{}[]^`\\«»" or ord(c) > 127)
    alpha = sum(1 for c in text if c.isalpha())
    total = len(text.strip())
    if total == 0:
        return False
    if alpha == 0:
        return True
    noise_ratio = noise / total
    words = text.split()
    if words:
        avg_len = sum(len(w) for w in words) / len(words)
        short_ratio = sum(1 for w in words if len(w) <= 2) / len(words)
        if noise_ratio > 0.1 and short_ratio > 0.4:
            return True
        if avg_len < 2.5 and len(words) > 5:
            return True
    return noise_ratio > 0.15

def _is_screenshot(text: str) -> bool:
    indicators = [
        r"change\.org", r"\bLTE\b", r"\b\d{1,2}:\d{2}\s*(AM|PM)\b",
        r"share this petition", r"share\s+tweet\s+email", r"\bmail\b.*\bLTE\b",
    ]
    hits = sum(1 for p in indicators if re.search(p, text.lower(), re.IGNORECASE))
    return hits >= 2

def _detect_references_other(part1_body: str, parts: list[str]) -> Optional[dict]:
    scan_text = part1_body
    if parts and parts[0]:
        scan_text += "\n" + parts[0][:2000]
    scan_lower = scan_text.lower()

    indicators = []
    parent_id = None

    if re.search(r"(?:submission|upload)\s+\d+\s+(?:of|through)\s+\d+", scan_lower):
        indicators.append("submission N of M")
    if re.search(r"(?:uploaded|submitted|upload)\s+(?:in|via|as)\s+submission", scan_lower):
        indicators.append("uploaded in other submissions")
    if re.search(
        r"remaining\s+(?:exhibits?|attachments?|files?|documents?)\s+"
        r"(?:will\s+be\s+)?(?:uploaded|submitted)", scan_lower,
    ):
        indicators.append("remaining materials in other uploads")
    if re.search(
        r"comments?\s+(?:themselves?\s+)?(?:will\s+be\s+)?(?:uploaded|submitted)\s+"
        r"in\s+submission", scan_lower,
    ):
        indicators.append("comment body in separate submission")

    for pat in [
        r"(?:primary|parent|main)\s+submission\s+(?:id|number|#)\s+"
        r"(?:associated\s+with\s+\S+\s+\S+\s+)?(?:is|:)\s*([a-zA-Z0-9_-]{6,})",
        r"submission\s+(?:id|number|#)\s*(?:is|:)\s*([a-zA-Z0-9_-]{6,})",
    ]:
        m = re.search(pat, scan_lower)
        if m:
            indicators.append("references parent submission ID")
            parent_id = m.group(1)
            break

    if re.search(r"(?:please\s+)?find\s+exhibits?\s+\S+\s+through\s+\S+", scan_lower):
        indicators.append("exhibit batch reference")
    if re.search(
        r"(?:references?\s+cited\s+in|attachments?\s+to)\s+"
        r"(?:the\s+)?(?:\w+\s+)?(?:comments?|submission)", scan_lower,
    ):
        indicators.append("supporting materials for separate comment")

    part1_lower = part1_body.lower() if part1_body else ""
    if re.search(r"\bpart\s+\d+\s+of\s+\d+\b", part1_lower):
        indicators.append("part N of N in submission text")
    if re.search(r"\[part\s+\d+\]", part1_lower):
        indicators.append("[Part N] label")

    if indicators:
        return {"indicators": indicators, "parent_id": parent_id}
    return None


# ── Tests ───────────────────────────────────────────────────────────

if __name__ == "__main__":

    def _test(name, expected, text, title=None):
        result = classify_comment(text, title=title)
        status = "PASS" if result["classification"] == expected else "FAIL"
        print(f"[{status}] {name}")
        print(f"       expected={expected}  got={result['classification']}")
        print(f"       reasons={result['reasons']}")
        if result.get("campaign_count_estimate"):
            print(f"       campaign_count={result['campaign_count_estimate']}")
        if result.get("ref_other_info"):
            print(f"       ref_other={result['ref_other_info']}")
        print(f"       parts={result['num_parts']}  pages={result['total_pages']}")
        print()

    _test("1. Cancer patient", "definitely_single",
        "<<PART 1>>\nfalse\n\n0\n\nI have stage 4 breast cancer......please do "
        "everything you can to prolong my life. I am not ready to die yet so with "
        "your help this can happen. It is so important you be able to contribute.\n"
        "You are in a position to help 1000's of people.....don't waste your chance!"
        "\n<<PART 2>>\n<<PAGE 1>>\n» SEE\n= 4\nLF\n~ x\nwong\n'pa ae\n'en"
        "\n<<PART 3>>\n<<PAGE 1>>\n| ,\n| \"a a\n: ab a ae ar. - \" a\nhe — i' \"+ po 4")

    _test("2. CREDO Action campaign", "definitely_campaign",
        "<<PART 1>>\nfalse\n\n0\n\nSee Attached\n\n"
        "<<PART 2>>\n<<PAGE 1>>\n"
        "Re: Docket ID: APHIS-2012-0030\n\n"
        "Attached please find 100,497 individual public comments collected online by "
        "CREDO Action for submission to the comment period on the freeze tolerant "
        "eucalyptus lines. Individual comments may differ throughout, although the "
        "majority read as follows:\n\n"
        "\"ArborGen's genetically engineered eucalyptus trees would endanger biodiversity\"\n\n"
        "Sincerely,\nJosh Nelson\nDeputy Political Director, CREDO Action\n\n"
        "<<PART 3>>\n<<PAGE 1>>\nRe: Docket ID: APHIS-2012-0030\n"
        "\"ArborGen's genetically engineered eucalyptus trees would endanger biodiversity, "
        "harm local communities\"\nSincerely,\nMarti Ann Cohen-Wolf\n"
        "355-19B South End Ave\nNew York, NY 10280\nCREDO Action\n"
        "<<PAGE 2>>\nRe: Docket ID: APHIS-2012-0030\n"
        "\"ArborGen's genetically engineered eucalyptus trees would endanger biodiversity, "
        "harm local communities\"\nSincerely,\nDavida Kaye\nRiverside, IL 60546\nCREDO Action\n"
        "<<PAGE 3>>\nRe: Docket ID: APHIS-2012-0030\n"
        "\"ArborGen's genetically engineered eucalyptus trees would endanger biodiversity, "
        "harm local communities\"\nSincerely,\nBetsy Armstrong\nBoulder, CO 80303\nCREDO Action")

    _test("3. Friends of the Earth campaign", "definitely_campaign",
        "<<PART 1>>\nfalse\n\n27588\n\nSee Attached\n\n"
        "<<PART 2>>\n<<PAGE 1>>\n"
        "On behalf of Friends of the Earth, I am submitting 27,590 comments "
        "demonstrating opposition to Trump's dirty water rule.\n"
        "These comments are split across 9 attachments.\n\n"
        "<<PART 3>>\n<<PAGE 1>>\n"
        "Fight Trump's Dirty Water Rule\n\"To the EPA: The proposed rule is far weaker.\"\n"
        "Joanne Steele\nSautee, GA 305712931\n"
        "<<PAGE 2>>\nFight Trump's Dirty Water Rule\n\"To the EPA: The proposed rule is far weaker.\"\n"
        "Brenda James\nVero Beach, FL 329603539\n"
        "<<PAGE 3>>\nFight Trump's Dirty Water Rule\n\"To the EPA: The proposed rule is far weaker.\"\n"
        "Jimmie Barrett\nJacksonville, FL 322058379")

    _test("4. Smart Policy Works", "definitely_single",
        "<<PART 1>>\nfalse\n\nSee attached file(s)\n\n"
        "<<PART 2>>\n<<PAGE 1>>\n"
        "December 7th, 2018\n\nSamantha Deshommes\nChief, Regulatory Coordination Division\n\n"
        "RE: DHS Docket No. USCIS-2010-0012\n\n"
        "Smart Policy Works' Comments in Response to Proposed Rulemaking on "
        "Inadmissibility on Public Charge Grounds\n\n"
        "Dear Ms. Deshommes:\nWith a rich 25-year history, Smart Policy Works has been "
        "breaking down systemic barriers to health and well-being. We use our expertise "
        "in health, disability, and veterans' policy to help people and organizations "
        "navigate public systems in Chicago and across the country.\n\n"
        "We write to express strong opposition to the proposed rule on public charge.")

    _test("5. Stephen Bingham", "definitely_single",
        "<<PART 1>>\nfalse\n0\n"
        "I wish to address two issues related to impaired driving which would not be "
        "adequately addressed by advanced technology.\n My daughter was killed 4 months "
        "after college graduation while biking to her new job in Cleveland Ohio. The "
        "driver of a single-unit truck did an unannounced right turn into her path at "
        "an intersection. He had just passed her so should have been fully aware she "
        "was on his right but he was severely impaired, as he had 8 times Ohio's legal "
        "limit for THC metabolite.\n"
        "<<PART 2>>\n<<PAGE 1>>\n\n<<PART 3>>\n<<PAGE 1>>\n")

    _test("6. Change.org screenshot", "definitely_single",
        "<<PART 1>>\nfalse\n0\nSee attached file(s)\n"
        "<<PART 2>>\n<<PAGE 1>>\nCl Mail ,111 LTE\n10:42 AM\n.- change.org\n"
        "I'm writing as a concerned American consumer in opposition to the proposed "
        "Modernization of Swine Slaughter Inspection rule.\nShare this petition\n"
        "<<PART 3>>\n<<PAGE 1>>\nCl Mail ,111 LTE\n10:42 AM\n.- change.org\n"
        "of slaughter plants themselves, while these facilities operate at dangerously "
        "high line speeds\nI) Share W Tweet ~ Email\nShare this petition")

    _test("7. San Juan Mine", "definitely_single",
        "<<PART 1>>\nfalse\n\n0\n\nSee Attached\n\n"
        "<<PART 2>>\n<<PAGE 1>>\nWestmoreland Coal Company\nSan Juan Mine 1\n"
        "February 13, 2017\n\nMs. Sheila McConnell\nDirector, Office of Standards\n\n"
        "Re: RIN 1219-AB78 Proximity Detection Systems for Mobile Machines Underground\n\n"
        "Dear Ms. McConnell:\nSan Juan Mine 1 is pleased to have the opportunity to "
        "provide our comments concerning the referenced Proposed Rule for Proximity "
        "Detection Systems for Mobile Machines in Underground Mines. San Juan Mine has "
        "experience with proximity detection equipment installed on continuous mining "
        "machines and also nearly a year of experience with proximity detection systems "
        "installed on our fleet of battery-powered coal haulers.")

    _test("8. Title-based campaign", "definitely_campaign",
        "<<PART 1>>\nfalse\n0\nSee attached\n"
        "<<PART 2>>\n<<PAGE 1>>\nsome text here about the environment",
        title="Mass Comment Campaign sponsoring organization unknown. Sample attached (web)")

    _test("9. Earthjustice exhibit batch", "references_other_submission",
        "<<PART 1>>\nfalse\n\n1\n\n"
        "Please find exhibits 1 through 1b-xxxix to the Conservation Groups' Comments "
        "attached. Please note that exhibits 1b-xxxviii and 1b-xl will be uploaded at "
        "the link provided by Jennifer Huser. The remaining exhibits will be uploaded "
        "in submissions 2-11, and the comments themselves will be uploaded in "
        "submission 11. Thank you.\n\n"
        "<<PART 2>>\n<<PAGE 1>>\nApril 20, 2015\n\nMr. Guy Donaldson\nChief, Air Planning\n"
        "RE: Docket ID No. EPA-R06-OAR-2014-0754\n\n"
        "Earthjustice, National Parks Conservation Association, and the Sierra Club "
        "respectfully submit the following comments regarding EPA's proposed action.")

    _test("10. Earthjustice refs w/ parent ID", "references_other_submission",
        "<<PART 1>>\nfalse\n\n1\n\n"
        "Please see attached references cited in comments by Earthjustice et al. [Part 2]\n\n"
        "The primary submission ID associated with the comments is mgf-q6na-wkvq.\n\n"
        "<<PART 2>>\n<<PAGE 1>>\nSeptember 2024\nUnited States EPA\n"
        "Draft Considerations and Resources for Assessing Tribal Exposures in TSCA "
        "Risk Evaluations")

    _test("11. NRDC attachments index", "references_other_submission",
        "<<PART 1>>\nfalse\n\n"
        "Please see the attached files containing additional attachments to the joint "
        "comments submitted by Natural Resources Defense Council et al. Some of the "
        "attachments have been split into multiple files because they exceed 10 MB.\n\n"
        "<<PART 2>>\n<<PAGE 1>>\nMASTER INDEX\nEnvironmental and Public Health "
        "Organizations'\nNHTSA CAFE Comments & Attachments\n\n"
        "Index\nBTS - Household Cost of Transportation.pdf\n"
        "BTS - Number of US vehicles.xlsx")

    _test("12. Multistate AGs NEPA", "definitely_single",
        "<<PART 1>>\nfalse\n\nPlease see our uploaded letter and attachments.\n\n"
        "<<PART 2>>\n<<PAGE 1>>\n"
        "COMMENTS OF ATTORNEYS GENERAL OF THE STATES OF WASHINGTON, CALIFORNIA, "
        "NEW YORK, ILLINOIS\n\nJuly 30, 2025\n\nVIA REGULATIONS.GOV\n\n"
        "U.S. Department of Agriculture\n\n"
        "Re: Revision of National Environmental Policy Act Interim Final Rule\n\n"
        "The Attorneys General respectfully submit these comments in opposition to the "
        "United States Department of Agriculture Interim Final Rule revising the "
        "Agency's regulations implementing NEPA. These comments describe how the Rule "
        "harms the States and is arbitrary and capricious.")

    _test("13. JRC / Eckert Seamans", "definitely_single",
        "<<PART 1>>\nfalse\n\nSee Attached\n\n"
        "<<PART 2>>\n<<PAGE 1>>\nECKERT\nEckert Seamans Cherin & Mellott, LLC\n"
        "Boston, MA 02110\n\nMay 20, 2024\n\n"
        "Docket No. FDA-2023-N-3902; Banned Devices; Proposal to Ban Electrical "
        "Stimulation Devices\n\nOn behalf of The Judge Rotenberg Educational Center, "
        "Inc. we are submitting the enclosed records as a comment to and in connection "
        "with JRC's forthcoming comments on the Proposed Rule.\n"
        "<<PART 3>>\n<<PAGE 1>>\n\n<<PART 4>>\n<<PAGE 1>>\n\n"
        "<<PART 5>>\n<<PAGE 1>>\nDear David - Thanks for your response. We are "
        "weighing our options with the client, and appreciate the comment from CDRH "
        "about a potential IDE pathway.")
