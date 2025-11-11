#!/usr/bin/env python3

"""
Temporary Playwright helper for exploring the EU Have Your Say initiatives page.

Launches a non-headless Chromium session, waits for the keyword filter box to
appear, enters the requested search term, and pauses so the session can be
driven manually.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import re
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright


INITIATIVES_URL = (
    "https://ec.europa.eu/info/law/better-regulation/have-your-say/initiatives_en"
)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing special characters,
    extra whitespace, and converting to lowercase.
    """
    # Remove special characters except spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.lower()


def ensure_browser_cache() -> None:
    """
    Make sure Playwright looks for downloaded browser binaries inside the
    project-local `.playwright` directory so installs do not clash with any
    system-wide caches that might require sudo.
    """

    if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ:
        project_root = Path(__file__).resolve().parents[1]
        cache_dir = project_root / ".playwright"
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(cache_dir)


def extract_metadata_table(page, *, timeout: int = 5000) -> list[tuple[str, str]]:
    """Parse the feedback summary description list into (field, value) pairs."""

    details_locator = page.locator("dl.ecl-description-list")
    rows: list[tuple[str, str]] = []

    try:
        details = details_locator.first
        details.wait_for(state="attached", timeout=timeout)
    except Exception:
        return rows

    terms = details.locator("dt")
    definitions = details.locator("dd")
    count = min(terms.count(), definitions.count())

    for idx in range(count):
        term_text = (terms.nth(idx).inner_text() or "").strip()
        value_text = (definitions.nth(idx).inner_text() or "").strip()
        if not term_text and not value_text:
            continue
        rows.append((term_text, value_text))

    return rows


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^\w.-]+", "_", name.strip())
    return sanitized or "file"


def launch_and_search(
    keyword: str,
    slow_mo: int,
    wait_timeout: int,
    *,
    downloads_dir: Path,
    max_pages: int | None,
    perform_downloads: bool,
    pause_on_complete: bool,
    dump_json: Path | None,
) -> None:
    ensure_browser_cache()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False, slow_mo=slow_mo)
        page = browser.new_page()
        page.goto(INITIATIVES_URL)

        # Wait for the keyword facet container to exist before interacting.
        input_box = page.wait_for_selector("#facet-keyword", timeout=wait_timeout)
        input_box.click()
        input_box.fill(keyword)

        # Wait for search results to appear
        print(f"Searching for results matching: '{keyword}'")
        page.wait_for_timeout(2000)
        
        # Find the clickable result that matches our keyword
        # The results appear as divs with class "ecl-link ecl-link--standalone"
        normalized_keyword = normalize_text(keyword)
        print(f"Normalized search term: '{normalized_keyword}'")
        
        # Get all potential result links
        result_links = page.locator("div.ecl-link.ecl-link--standalone").all()
        print(f"Found {len(result_links)} result links")
        
        matched_link = None
        all_feedback_entries: list[dict[str, object]] = []
        # Try exact match first
        for link in result_links:
            link_text = link.text_content()
            if link_text:
                normalized_link = normalize_text(link_text)
                if normalized_link == normalized_keyword:
                    matched_link = link
                    print(f"✓ Found exact match: '{link_text}'")
                    break
        
        # If no exact match, try partial matching (keyword is substring of result)
        if not matched_link:
            for link in result_links:
                link_text = link.text_content()
                if link_text:
                    normalized_link = normalize_text(link_text)
                    if normalized_keyword in normalized_link:
                        matched_link = link
                        print(f"✓ Found partial match: '{link_text}'")
                        break
        
        # If still no match, try the reverse (result is substring of keyword)
        if not matched_link:
            for link in result_links:
                link_text = link.text_content()
                if link_text:
                    normalized_link = normalize_text(link_text)
                    if normalized_link in normalized_keyword:
                        matched_link = link
                        print(f"✓ Found reverse partial match: '{link_text}'")
                        break
        
        if matched_link:
            print("Clicking on the matched result...")
            matched_link.click()
            # Wait for navigation to complete
            page.wait_for_load_state("networkidle")
            print("✓ Initiative page loaded")
            time.sleep(2)

            # Scroll to the bottom to trigger lazy loading of content
            print("Scrolling to trigger lazy loading...")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1500)
            time.sleep(5)

            # Scroll back to top to see everything
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(500)
            time.sleep(5)

            # Now find all "All feedback" links (there may be multiple for different consultations)
            print("Looking for 'All feedback' links...")
            try:
                # Use Playwright's text locator to find ALL links containing "All feedback"
                feedback_links = page.get_by_role("link", name=re.compile(r"All feedback", re.IGNORECASE)).all()
                
                if not feedback_links:
                    print("✗ No 'All feedback' links found")
                else:
                    print(f"✓ Found {len(feedback_links)} feedback link(s)")
                    
                    # we will iterate through all feedback links and collect the links to the individual feedback entries
                    # for now, lets just do the first one
                    parts = urlparse(page.url)
                    base_url = parts.scheme + '://' + parts.netloc
                    all_feedback_links = list(map(lambda link: base_url + link.get_attribute('href'), feedback_links))

                    for link in all_feedback_links:
                        page.goto(link)
                        page.wait_for_load_state("networkidle")
                        print("✓ Feedback page loaded, waiting 10 seconds...")
                        time.sleep(10)
                        print("✓ Feedback page loaded, continuing...")

                        initiative_url = page.url
                        initiative_path_fragment = ""
                        if initiative_url:
                            initiative_path_fragment = '/'.join(
                                urlparse(initiative_url).path.rstrip("/").split("/")[:-1]
                            )

                        page_number = 1
                        while True:
                            if max_pages is not None and page_number > max_pages:
                                print(
                                    f"  Reached max-pages limit ({max_pages}); "
                                    "stopping pagination."
                                )
                                break
                            print(f"\nCollecting feedback entry links (page {page_number})...")
                            entry_links = page.locator("feedback-item").all()
                            for entry in entry_links:
                                href = None
                                candidate_hrefs = entry.locator('a').all()
                                for candidate_href in candidate_hrefs:
                                    href = candidate_href.get_attribute('href')
                                    if href:
                                        break                        

                                if not href:
                                    continue
                                if initiative_path_fragment and initiative_path_fragment not in href:
                                    continue
                                text = (entry.text_content() or "").strip()
                                full_url = urljoin(page.url, href)
                                all_feedback_entries.append({
                                    "text": text,
                                    "href": full_url,
                                    "page": page_number,
                                    "feedback_link": link
                                })

                            next_button = page.locator(
                                "ecl-pagination li.ecl-pagination__item--next a.ecl-pagination__link"
                            )

                            if next_button.count() == 0:
                                print("  No next page link found; stopping pagination.")
                                break

                            aria_disabled = next_button.first.get_attribute("aria-disabled")
                            if aria_disabled == "true":
                                print("  Next page is disabled; reached final page.")
                                break

                            print("  Moving to next page of feedback entries...")
                            next_button.first.click()
                            page.wait_for_load_state("networkidle")
                            page.wait_for_timeout(1000)
                            time.sleep(1)
                            page_number += 1

                        if all_feedback_entries:
                            print(
                                f"Collected {len(all_feedback_entries)} unique feedback entries across {page_number} page(s):"
                            )
                            for entry in all_feedback_entries:
                                print(
                                    f"  Page {entry['page']}: {entry['text']} -> {entry['href']}"
                                )
                        else:
                            print("No feedback entry links found with expected pattern.")

                    # In a full workflow you could iterate through all_feedback_entries here.
                    
            except Exception as e:
                print(f"✗ Error processing 'All feedback' links: {e}")
                # Try to list available links for debugging
                all_links = page.locator("a").all()
                print("Available links on page:")
                for link in all_links[:10]:
                    link_text = link.text_content()
                    if link_text and link_text.strip():
                        print(f"  - '{link_text.strip()}'")
        else:
            print("✗ No matching result found. Available results:")
            for i, link in enumerate(result_links[:5], 1):  # Show first 5
                print(f"  {i}. '{link.text_content()}'")
        
        # Hand control back to the user for manual inspection
        data = []
        for idx, entry in enumerate(all_feedback_entries, start=1):
            page.goto(entry['href'])
            page.wait_for_load_state("networkidle")
            time.sleep(5)
            metadata_rows = extract_metadata_table(page, timeout=wait_timeout)
            metadata_dict = dict(metadata_rows)

            download_path = None
            download_locator = page.locator("a.ecl-file__download")
            if perform_downloads and download_locator.count():
                reference = metadata_dict.get("Feedback reference")
                safe_name = sanitize_filename(reference or entry["href"])
                target_path = downloads_dir / f"{safe_name}.pdf"
                downloads_dir.mkdir(parents=True, exist_ok=True)

                if target_path.exists():
                    print(f"[entry {idx}] Attachment already exists at {target_path}")
                    download_path = str(target_path)
                else:
                    try:
                        with page.expect_download(timeout=wait_timeout) as download_info:
                            download_locator.first.click()
                        download = download_info.value
                        download.save_as(str(target_path))
                        download_path = str(target_path)
                        print(f"[entry {idx}] Downloaded attachment to {target_path}")
                    except Exception as download_error:
                        print(f"[entry {idx}] Failed to download attachment: {download_error}")
            elif perform_downloads:
                print(f"[entry {idx}] No downloadable attachment link detected.")

            output_dict = {
                "text": entry['text'],
                "href": entry['href'],
                "page": entry['page'],
                "feedback_link": entry['feedback_link'],
            }
            if download_path:
                output_dict["download_path"] = download_path
            output_dict.update(metadata_dict)
            data.append(output_dict)

        print(f"\nProcessed {len(data)} feedback entries.")
        if dump_json:
            dump_json.parent.mkdir(parents=True, exist_ok=True)
            dump_json.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Metadata written to {dump_json}")

        if pause_on_complete:
            page.pause()

        browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open EU initiatives page and populate the keyword filter."
    )
    parser.add_argument(
        "keyword",
        nargs="?",
        default="Artificial intelligence ethical and legal requirements",
        help="Keyword to enter into the filter box.",
    )
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=250,
        help="Delay (ms) Playwright inserts between operations to aid debugging.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=60000,
        help="Maximum time (ms) to wait for elements to appear.",
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "downloads",
        help="Directory to store downloaded attachments (default: <repo>/downloads).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional maximum number of feedback listing pages to traverse.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only collect metadata; skip attachment downloads.",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Finish without entering Playwright inspector pause mode.",
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        help="Optional path to write collected metadata as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launch_and_search(
        args.keyword,
        args.slow_mo,
        args.wait_timeout,
        downloads_dir=args.downloads_dir.expanduser().resolve(),
        max_pages=args.max_pages,
        perform_downloads=not args.no_download,
        pause_on_complete=not args.no_pause,
        dump_json=args.dump_json.expanduser().resolve() if args.dump_json else None,
    )


if __name__ == "__main__":
    main()
