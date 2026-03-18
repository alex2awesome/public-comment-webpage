import argparse
import csv
import json
import re
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


DASH_VARIANTS = "-–—−‒―‐‑"
ALLOWED_SUFFIXES = {".htm", ".html", ".txt"}
FILE_PATTERN = rf"(?:SR|S7)[{DASH_VARIANTS}]\s*[A-Z0-9]+(?:[{DASH_VARIANTS}]\s*[A-Z0-9]+)+"
PATTERNS: List[re.Pattern[str]] = [
    re.compile(rf"File\s+No\.?\s*(?P<value>{FILE_PATTERN})", re.IGNORECASE),
    re.compile(rf"File\s+Number\s*(?P<value>{FILE_PATTERN})", re.IGNORECASE),
    re.compile(rf"(?<![A-Z0-9])(?P<value>{FILE_PATTERN})", re.IGNORECASE),
]
WHITESPACE_RE = re.compile(r"\s+")


def generate_sec_urls(file_no: str, publication_date_str: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Generate URL hints for SEC file numbers.

    Mirrors the scenarios outlined by the user:
      - SR-* entries require manual index scraping.
      - S7-YY-NN (old style) have deterministic comment URLs.
      - S7-YYYY-NN (new style) might need a fallback landing page.
    """

    file_no = file_no.strip().upper()

    if file_no.startswith("SR-"):
        return {
            "method": "SRO_INDEX_SEARCH",
            "primary_url": None,
            "fallback_url": None,
            "note": "Exchange rule filings require scraping the SRO comment index.",
        }

    match = re.match(r"S7-(\d{2,4})-(\d{2})", file_no)
    if match:
        part_1, part_2 = match.groups()
        clean_id = f"s7{part_1}{part_2}"
        primary = f"https://www.sec.gov/comments/{file_no.lower()}/{clean_id}.htm"

        if len(part_1) == 4:
            fallback = None
            if publication_date_str:
                try:
                    dt = datetime.fromisoformat(publication_date_str.replace("Z", "+00:00"))
                    fallback = f"https://www.sec.gov/rules-regulations/{dt.year}/{dt.month:02d}/{file_no.lower()}"
                except ValueError:
                    fallback = "Error: Invalid Date Format provided"

            return {
                "method": "NEW_STYLE_AGENCY",
                "primary_url": primary,
                "fallback_url": fallback,
                "note": "Primary link often fails for 2024+ rules. Use fallback if 404.",
            }

        return {
            "method": "OLD_STYLE_AGENCY",
            "primary_url": primary,
            "fallback_url": None,
            "note": "Standard reliable format.",
        }

    return {"method": "UNKNOWN_FORMAT", "primary_url": None, "fallback_url": None, "note": "Unrecognized file number pattern."}


def normalize_identifier(raw: str) -> str:
    """Normalize Unicode dashes and whitespace so SR codes are comparable."""
    cleaned = raw.strip().upper()
    for dash in DASH_VARIANTS:
        cleaned = cleaned.replace(dash, "-")
    cleaned = WHITESPACE_RE.sub("", cleaned)
    cleaned = cleaned.strip("-[](){}<>,.;:'\"")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned if cleaned.startswith(("SR-", "S7-")) else ""


def should_read(path: Path) -> bool:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    return bool(suffixes & ALLOWED_SUFFIXES)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_bytes().decode("utf-8", errors="ignore")


def extract_identifiers(text: str) -> List[str]:
    matches: List[str] = []
    seen = set()
    for pattern in PATTERNS:
        for match in pattern.finditer(text):
            raw_value = match.group("value")
            normalized = normalize_identifier(raw_value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                matches.append(normalized)
    return matches


def iter_files(data_dir: Path) -> Iterable[Path]:
    pattern = str(data_dir / "*/*/*")
    for name in sorted(glob(pattern)):
        path = Path(name)
        if path.is_file() and should_read(path):
            yield path


def load_metadata(
    data_dir: Path, type_name: str, docket_name: str, cache: Dict[Tuple[str, str], Dict[str, str]]
) -> Dict[str, str]:
    key = (type_name, docket_name)
    if key in cache:
        return cache[key]

    metadata_path = data_dir / type_name / docket_name / "metadata.json"
    if metadata_path.exists():
        try:
            cache[key] = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            cache[key] = {}
    else:
        cache[key] = {}

    return cache[key]


def main(data_dir: Path) -> None:
    records = []
    metadata_cache: Dict[Tuple[str, str], Dict[str, str]] = {}
    for path in tqdm(iter_files(data_dir), desc="Processing files"):
        text = read_text(path)
        identifiers = extract_identifiers(text)
        if not identifiers:
            continue

        type_name, docket_name = path.parts[-3], path.parts[-2]
        metadata = load_metadata(data_dir, type_name, docket_name, metadata_cache)
        posted_date = metadata.get("Posted Date") or metadata.get("PostedDate")
        url_info = [generate_sec_urls(identifier, posted_date) for identifier in identifiers]
        primary_urls = [info["primary_url"] for info in url_info if info.get("primary_url")]
        records.append(
            {
                "type": type_name,
                "docket": docket_name,
                "file": path.name,
                "sec_file_id": identifiers,
                "urls": primary_urls,
                "url_info": url_info,
                "posted_date": posted_date,
            }
        )

    with Path("mapper.jsonl").open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False))
            jsonl_file.write("\n")

    with Path("mapper.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["type", "docket", "file", "sec_file_id", "urls", "url_info", "posted_date"])
        for record in records:
            writer.writerow(
                [
                    record["type"],
                    record["docket"],
                    record["file"],
                    json.dumps(record["sec_file_id"], ensure_ascii=False),
                    json.dumps(record["urls"], ensure_ascii=False),
                    json.dumps(record["url_info"], ensure_ascii=False),
                    record.get("posted_date") or "",
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SEC identifiers from downloaded files.")
    parser.add_argument(
        "--data-dir",
        default="downloaded_content",
        help="Directory that contains type/docket/file content (default: downloaded_content)",
    )
    args = parser.parse_args()
    main(Path(args.data_dir))
