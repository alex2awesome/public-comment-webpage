import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.async_api import Browser, Page, async_playwright
from tqdm import tqdm

OLD_STYLE_PATTERN = re.compile(r"^S7-\d{2}-\d{2}$", re.IGNORECASE)
NOT_FOUND_TITLES = ("oops! page not found.", "page not found")


def load_targets(mapper_path: Path) -> List[Dict[str, str]]:
    """Collect unique old-style S7 IDs with valid primary URLs."""
    targets: Dict[str, str] = {}
    with mapper_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            ids = record.get("sec_file_id") or []
            info_list = record.get("url_info") or []
            for sec_id, info in zip(ids, info_list):
                normalized = sec_id.strip().upper()
                if not OLD_STYLE_PATTERN.match(normalized):
                    continue
                primary = (info or {}).get("primary_url")
                if not primary:
                    continue
                targets.setdefault(normalized, primary)
    return [{"file_no": file_no, "url": url} for file_no, url in targets.items()]


def parse_table(html: str, page_url: str) -> List[Dict[str, str]]:
    """Extract rows from the main usa-table into dicts keyed by header plus the comment link."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.usa-table")
    if not table:
        return []

    header_cells = table.select("thead tr th")
    headers = [cell.get_text(strip=True) or f"col_{idx+1}" for idx, cell in enumerate(header_cells)]
    rows_data: List[Dict[str, str]] = []

    for row in table.select("tbody tr"):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        row_dict: Dict[str, str] = {}
        for idx, cell in enumerate(cells):
            key = headers[idx] if idx < len(headers) else f"col_{idx+1}"
            row_dict[key] = " ".join(cell.stripped_strings)
            anchor = cell.find("a", href=True)
            if anchor and "comment_url" not in row_dict:
                row_dict["comment_url"] = urljoin(page_url, anchor["href"])
        rows_data.append(row_dict)
    return rows_data


async def dismiss_feedback_popup(page: Page) -> None:
    """Attempt to close or remove feedback overlays that block interaction."""
    close_selectors = [
        "button[aria-label*='feedback' i]",
        "button[title*='feedback' i]",
        "[role='dialog'] button[aria-label*='close' i]",
        "[role='dialog'] button:has-text('Close')",
        "[id*='feedback' i] button[aria-label*='close' i]",
    ]
    for selector in close_selectors:
        locator = page.locator(selector)
        if await locator.count():
            try:
                await locator.first.click()
                await asyncio.sleep(0.1)
                return
            except Exception:
                continue

    # As a fallback, remove common feedback containers entirely.
    script = """
        for (const el of document.querySelectorAll('[id*="feedback" i], [class*="feedback" i]')) {
            if (el.closest('[role="dialog"]')) {
                el.closest('[role="dialog"]').remove();
            } else {
                el.remove();
            }
        }
    """
    await page.evaluate(script)


async def is_not_found_page(page: Page) -> bool:
    title = (await page.title()).strip().lower()
    if any(token in title for token in NOT_FOUND_TITLES):
        return True
    body = await page.locator("body").get_attribute("class")
    return body is not None and "path--404" in body


def build_candidate_urls(url: str) -> List[str]:
    lower = url.lower()
    if lower.endswith(".htm"):
        base = url[:-4]
        return [base + ".htm", base + ".shtml"]
    if lower.endswith(".shtml"):
        base = url[:-6]
        return [base + ".htm", base + ".shtml"]
    return [url.rstrip("/").rstrip(".")] + [
        f"{url.rstrip('/').rstrip('.')}.htm",
        f"{url.rstrip('/').rstrip('.')}.shtml",
    ]


async def goto_with_fallback(page: Page, base_url: str) -> Tuple[str, Optional[int]]:
    """Load a comment page, retrying with .shtml if the .htm variant fails."""
    candidates = build_candidate_urls(base_url)

    last_status = None
    for target_url in candidates:
        try:
            response = await page.goto(target_url, wait_until="domcontentloaded", timeout=90_000)
            last_status = response.status if response else None
            if await is_not_found_page(page):
                continue
            await page.wait_for_selector("table.usa-table", timeout=30_000)
            return target_url, last_status
        except Exception:
            continue

    raise RuntimeError(f"Failed to load {base_url} or fallback variant (.shtml). Last status={last_status}")


async def collect_table_rows(page: Page, delay: float, max_pages: int = 200) -> Tuple[List[Dict[str, str]], int]:
    """Capture table rows across all pagination pages."""
    rows: List[Dict[str, str]] = []
    page_count = 0

    while page_count < max_pages:
        await dismiss_feedback_popup(page)
        if await is_not_found_page(page):
            break
        await page.wait_for_selector("table.usa-table tbody tr", timeout=30_000)
        await asyncio.sleep(max(delay, 3.0))
        html = await page.content()
        rows.extend(parse_table(html, page.url))
        page_count += 1

        next_button = page.locator("nav.usa-pagination a.usa-pagination__next-page")
        if await next_button.count() == 0:
            break

        next_locator = next_button.first
        href = await next_locator.get_attribute("href")
        if not href or href.strip("#") == "":
            break

        next_url = urljoin(page.url, href)
        try:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.goto(next_url, wait_until="domcontentloaded", timeout=90_000)
            if await is_not_found_page(page):
                break
            await page.wait_for_selector("table.usa-table tbody tr", timeout=30_000)
            await asyncio.sleep(max(delay, 3.0))
        except Exception:
            break

    return rows, page_count


async def scrape_targets(
    targets: Iterable[Dict[str, str]],
    output_dir: Path,
    delay: float,
    headless: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as playwright:
        browser: Browser = await playwright.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()

        for target in tqdm(list(targets), desc="Scraping S7 tables"):
            file_no = target["file_no"]
            url = target["url"]
            out_path = output_dir / f"{file_no}.json"
            start = time.perf_counter()

            try:
                final_url, status = await goto_with_fallback(page, url)
                rows, page_count = await collect_table_rows(page, delay)
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[WARN] Failed to fetch {file_no} ({url}): {exc}")
                continue

            if not rows:
                tqdm.write(f"[WARN] No usa-table rows found for {file_no} ({final_url})")

            payload = {
                "file_no": file_no,
                "url": final_url,
                "http_status": status,
                "row_count": len(rows),
                "page_count": page_count,
                "rows": rows,
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            elapsed = time.perf_counter() - start
            tqdm.write(f"[INFO] Saved {file_no} ({len(rows)} rows, {page_count} pages) in {elapsed:.1f}s -> {out_path}")

            if delay > 0:
                await asyncio.sleep(delay)

        await browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape SEC comment tables for old-style S7 files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory that contains mapper.jsonl and scraped_tables (defaults to <repo>/data/bulk_downloads/sec/sec_2023).",
    )
    parser.add_argument(
        "--mapper",
        type=Path,
        default=None,
        help="Explicit path to mapper.jsonl (defaults to <data-dir>/mapper.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write table JSON files (defaults to <data-dir>/scraped_tables).",
    )
    parser.add_argument("--delay", type=float, default=2.0, help="Sleep (seconds) between requests")
    parser.add_argument("--not_headless", action="store_true", help="Run browser in headed mode for debugging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_data_dir = Path(__file__).parent / "sec_2023"
    raw_data_dir = args.data_dir or default_data_dir
    data_dir = raw_data_dir
    if not data_dir.exists():
        candidate = Path(__file__).parent / raw_data_dir
        if candidate.exists():
            data_dir = candidate
    if not data_dir.exists():
        raise SystemExit(f"Data directory {raw_data_dir} does not exist.")

    mapper_path = args.mapper or (data_dir / "mapper.jsonl")
    if not mapper_path.exists():
        raise SystemExit(f"Mapper file not found at {mapper_path}")
    output_dir = args.output_dir or (data_dir / "scraped_tables")

    targets = load_targets(mapper_path)
    if not targets:
        raise SystemExit("No old-style S7 targets were found in mapper data.")

    headless = not args.not_headless
    asyncio.run(scrape_targets(targets, output_dir, args.delay, headless=headless))


if __name__ == "__main__":
    main()
