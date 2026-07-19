"""
scrapers/reddit_public.py — Keyless Reddit scraper.
Fetches comments from Reddit thread URLs using the public .json endpoint.
No API keys needed.
"""

import logging
import re
import time

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; Blinkit-Research-Bot/1.0; academic research)",
    "Accept": "application/json",
}

# Rate limit: Reddit blocks if you hit too fast
_DELAY_BETWEEN_REQUESTS = 4  # seconds


def _load_urls() -> list[str]:
    """Load Reddit URLs from the curated file."""
    urls = []
    urls_file = config.REDDIT_URLS_FILE
    if not urls_file.exists():
        logger.warning("Reddit URLs file not found: %s", urls_file)
        return urls

    with open(urls_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("http") and "reddit.com" in line:
                urls.append(line.rstrip("/"))

    logger.info("Loaded %d Reddit URLs from %s", len(urls), urls_file)
    return list(dict.fromkeys(urls))  # deduplicate, preserve order


def _fetch_thread_json(url: str) -> dict | None:
    """Fetch a Reddit thread as JSON via old.reddit.com."""
    # Convert to old.reddit.com which is more permissive
    json_url = url.replace("www.reddit.com", "old.reddit.com").rstrip("/") + ".json"
    try:
        resp = requests.get(json_url, headers=_HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code == 429:
            logger.warning("Rate limited — waiting 30s...")
            time.sleep(30)
            resp = requests.get(json_url, headers=_HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code == 403:
            # Try with oauth token-less access via api subdomain
            api_url = url.replace("www.reddit.com", "api.reddit.com").rstrip("/") + ".json"
            resp = requests.get(api_url, headers=_HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return None


def _extract_comments(data: list, url: str) -> list[dict]:
    """Recursively extract comments from Reddit JSON response."""
    rows = []

    # data[0] = post, data[1] = comments
    if not isinstance(data, list) or len(data) < 2:
        return rows

    # Extract the post itself
    post_data = data[0].get("data", {}).get("children", [])
    if post_data:
        post = post_data[0].get("data", {})
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        created = post.get("created_utc", 0)
        subreddit = post.get("subreddit", "")
        text = f"{title}. {selftext}".strip() if selftext else title

        if text and len(text) > 10:
            rows.append({
                "source_raw": f"reddit_{subreddit}",
                "date": str(pd.Timestamp(created, unit="s").date()) if created else "",
                "rating": "",
                "text": text,
                "url": url,
            })

    # Extract comments recursively
    def _walk_comments(children):
        for child in children:
            if child.get("kind") != "t1":
                continue
            cdata = child.get("data", {})
            body = cdata.get("body", "").strip()
            created = cdata.get("created_utc", 0)
            subreddit = cdata.get("subreddit", "")

            if body and len(body) > 10 and body != "[deleted]" and body != "[removed]":
                rows.append({
                    "source_raw": f"reddit_{subreddit}",
                    "date": str(pd.Timestamp(created, unit="s").date()) if created else "",
                    "rating": "",
                    "text": body,
                    "url": url,
                })

            # Recurse into replies
            replies = cdata.get("replies", "")
            if isinstance(replies, dict):
                reply_children = replies.get("data", {}).get("children", [])
                _walk_comments(reply_children)

    comment_children = data[1].get("data", {}).get("children", [])
    _walk_comments(comment_children)

    return rows


def scrape_reddit_urls() -> pd.DataFrame:
    """
    Scrape all curated Reddit thread URLs using the public JSON endpoint.
    No API keys required.

    Returns DataFrame with columns: [source_raw, date, rating, text, url]
    """
    urls = _load_urls()
    if not urls:
        return pd.DataFrame()

    all_rows = []
    for i, url in enumerate(urls):
        # Skip non-Reddit URLs (e.g., mouthshut.com)
        if "reddit.com" not in url:
            logger.info("Skipping non-Reddit URL: %s", url)
            continue

        logger.info("[%d/%d] Scraping: %s", i + 1, len(urls), url)
        data = _fetch_thread_json(url)
        if data is None:
            continue

        rows = _extract_comments(data, url)
        logger.info("  → %d posts/comments extracted", len(rows))
        all_rows.extend(rows)

        # Be nice to Reddit's servers
        if i < len(urls) - 1:
            time.sleep(_DELAY_BETWEEN_REQUESTS)

    if not all_rows:
        logger.warning("No comments extracted from any Reddit URL.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Deduplicate on text
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    logger.info(
        "Reddit scrape complete: %d unique posts/comments from %d threads (deduped from %d).",
        len(df), len(urls), before,
    )
    return df
