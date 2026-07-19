"""
scrapers/reddit_browser.py — Playwright-based Reddit scraper.
Opens Reddit threads in a real browser and extracts posts + comments.
Works without API keys since Reddit can't block a real browser.

Usage:
    python scrapers/reddit_browser.py                      # Scrape all URLs from Data/Reddit Data URLs
    python scrapers/reddit_browser.py --url <single_url>   # Scrape one URL
    python scrapers/reddit_browser.py --output custom.csv  # Custom output path
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reddit_browser")

# ── Config ──
OUTPUT_CSV = config.OUTPUT_DIR / "reddit_scraped.csv"
SCROLL_PAUSE = 2  # seconds between scrolls
MAX_SCROLLS = 15  # max scrolls to load more comments
PAGE_TIMEOUT = 45000  # ms


def _save_progress(rows: list[dict]):
    """Save progress to a temp file so work isn't lost on interrupt."""
    tmp_path = config.OUTPUT_DIR / "reddit_scraped_partial.csv"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(str(tmp_path), index=False)
    logger.info("  [progress saved: %d items → %s]", len(rows), tmp_path)


def _load_urls_from_file() -> list[str]:
    """Load Reddit URLs from the curated file."""
    urls_file = config.REDDIT_URLS_FILE
    if not urls_file.exists():
        logger.error("Reddit URLs file not found: %s", urls_file)
        return []

    urls = []
    with open(urls_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("http"):
                urls.append(line.rstrip("/"))

    # Deduplicate preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    logger.info("Loaded %d unique URLs from %s", len(unique), urls_file)
    return unique


def _extract_subreddit(url: str) -> str:
    """Extract subreddit name from URL."""
    match = re.search(r"/r/([^/]+)", url)
    return match.group(1) if match else "unknown"


def scrape_thread(page, url: str) -> list[dict]:
    """Scrape a single Reddit thread using Playwright page."""
    rows = []
    subreddit = _extract_subreddit(url)

    try:
        # Use old.reddit.com for simpler HTML structure
        old_url = url.replace("www.reddit.com", "old.reddit.com")
        page.goto(old_url, timeout=PAGE_TIMEOUT, wait_until="networkidle")
        time.sleep(1)

        # Check if page loaded properly
        title_text = page.title()
        if "page not found" in title_text.lower():
            logger.warning("  Thread not found or deleted: %s", url)
            return rows

        # Extract post title + body
        title_el = page.query_selector("a.title")
        title = title_el.inner_text().strip() if title_el else ""

        body_el = page.query_selector(".expando .usertext-body")
        body = body_el.inner_text().strip() if body_el else ""

        post_text = f"{title}. {body}".strip() if body else title
        if post_text and len(post_text) > 5:
            # Try to get post date
            time_el = page.query_selector(".tagline time")
            post_date = time_el.get_attribute("datetime")[:10] if time_el else ""

            rows.append({
                "source_raw": f"reddit_{subreddit}",
                "date": post_date,
                "rating": "",
                "text": post_text,
                "url": url,
                "type": "post",
            })

        # Extract all comments
        comments = page.query_selector_all(".comment .usertext-body .md")
        for comment_el in comments:
            try:
                text = comment_el.inner_text().strip()
                if text and len(text) > 10 and text not in ("[deleted]", "[removed]"):
                    rows.append({
                        "source_raw": f"reddit_{subreddit}",
                        "date": "",
                        "rating": "",
                        "text": text,
                        "url": url,
                        "type": "comment",
                    })
            except Exception:
                continue

    except PwTimeout:
        logger.warning("  Timeout loading: %s", url)
    except Exception as e:
        logger.warning("  Error scraping %s: %s", url, e)

    return rows


def scrape_all_urls(urls: list[str] = None, headless: bool = True) -> pd.DataFrame:
    """
    Scrape all Reddit URLs using Playwright.

    Args:
        urls: List of URLs to scrape. If None, loads from Data/Reddit Data URLs.
        headless: Run browser in headless mode (no visible window).

    Returns:
        DataFrame with columns: [source_raw, date, rating, text, url, type]
    """
    if urls is None:
        urls = _load_urls_from_file()

    if not urls:
        logger.error("No URLs to scrape.")
        return pd.DataFrame()

    # Filter to Reddit URLs only
    reddit_urls = [u for u in urls if "reddit.com" in u]
    non_reddit = [u for u in urls if "reddit.com" not in u]
    if non_reddit:
        logger.info("Skipping %d non-Reddit URLs: %s", len(non_reddit), non_reddit)

    logger.info("Scraping %d Reddit threads with Playwright...", len(reddit_urls))

    all_rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/131.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        for i, url in enumerate(reddit_urls):
            logger.info("[%d/%d] %s", i + 1, len(reddit_urls), url)
            rows = scrape_thread(page, url)
            logger.info("  → %d items extracted", len(rows))
            all_rows.extend(rows)

            # Save progress every 5 threads (so we don't lose work)
            if (i + 1) % 5 == 0 and all_rows:
                _save_progress(all_rows)

            # Be nice — don't hammer Reddit
            if i < len(reddit_urls) - 1:
                time.sleep(2)

        browser.close()

    if not all_rows:
        logger.warning("No data extracted from any URL.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Deduplicate on text
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    logger.info(
        "Done: %d unique items from %d threads (deduped from %d).",
        len(df), len(reddit_urls), before,
    )

    return df


def main():
    parser = argparse.ArgumentParser(description="Reddit Browser Scraper (Playwright)")
    parser.add_argument("--url", type=str, help="Scrape a single URL")
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV),
                        help="Output CSV path")
    parser.add_argument("--visible", action="store_true",
                        help="Show the browser window (not headless)")
    args = parser.parse_args()

    if args.url:
        urls = [args.url]
    else:
        urls = None  # load from file

    df = scrape_all_urls(urls=urls, headless=not args.visible)

    if df.empty:
        logger.error("No data scraped.")
        sys.exit(1)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_path), index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Total: {len(df)} items")
    print(f"  Posts: {(df['type'] == 'post').sum()}")
    print(f"  Comments: {(df['type'] == 'comment').sum()}")
    print(f"\nSubreddits:")
    for src, cnt in df["source_raw"].value_counts().items():
        print(f"  {src}: {cnt}")


if __name__ == "__main__":
    main()
