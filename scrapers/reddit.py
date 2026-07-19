"""
scrapers/reddit.py — Pathway B: Reddit thread scraper.
Scrapes comments from curated Reddit thread URLs using PRAW.
"""

import logging
import re
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def _load_reddit_urls() -> list[str]:
    """
    Load Reddit URLs from:
    1. The plain text 'Reddit Data URLs' file
    2. The links-data.docx file (if python-docx is available)
    """
    urls = []

    # From plain text file
    urls_file = config.REDDIT_URLS_FILE
    if urls_file.exists():
        with open(urls_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line.startswith("http"):
                    urls.append(line)

    # From .docx file
    docx_file = config.REDDIT_LINKS_DOCX
    if docx_file.exists():
        try:
            from docx import Document

            doc = Document(str(docx_file))
            for para in doc.paragraphs:
                text = para.text.strip()
                # Extract URLs from text
                found = re.findall(
                    r"https?://(?:www\.)?reddit\.com/r/\S+", text
                )
                urls.extend(found)

            # Also check hyperlinks in the document
            for rel in doc.part.rels.values():
                if "hyperlink" in str(rel.reltype):
                    target = str(rel._target)
                    if "reddit.com" in target:
                        urls.append(target)
        except ImportError:
            logger.warning(
                "python-docx not installed; skipping links-data.docx. "
                "Install with: pip install python-docx"
            )
        except Exception as e:
            logger.warning("Failed to parse links-data.docx: %s", e)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        u_clean = u.rstrip("/")
        if u_clean not in seen:
            seen.add(u_clean)
            unique.append(u_clean)

    logger.info("Loaded %d unique Reddit URLs.", len(unique))
    return unique


def _extract_submission_id(url: str) -> str | None:
    """Extract Reddit submission ID from URL."""
    match = re.search(r"/comments/([a-z0-9]+)", url)
    return match.group(1) if match else None


def scrape_reddit_threads() -> pd.DataFrame:
    """
    Scrape comments from all curated Reddit thread URLs.

    Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET in .env.
    Returns DataFrame with columns: [source_raw, date, rating, text, url]
    """
    import praw

    if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
        logger.error(
            "Reddit API credentials not set. "
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env"
        )
        return pd.DataFrame()

    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
    )

    urls = _load_reddit_urls()
    if not urls:
        logger.warning("No Reddit URLs found to scrape.")
        return pd.DataFrame()

    rows = []
    for url in urls:
        try:
            submission = reddit.submission(url=url)
            submission.comments.replace_more(limit=0)

            # Add the post itself
            rows.append(
                {
                    "source_raw": "reddit_live",
                    "date": str(
                        pd.Timestamp(submission.created_utc, unit="s")
                    ),
                    "rating": "",  # Reddit has no star ratings
                    "text": f"{submission.title}. {submission.selftext}".strip(),
                    "url": url,
                }
            )

            # Add all comments
            for comment in submission.comments.list():
                if hasattr(comment, "body") and comment.body.strip():
                    rows.append(
                        {
                            "source_raw": "reddit_live",
                            "date": str(
                                pd.Timestamp(comment.created_utc, unit="s")
                            ),
                            "rating": "",
                            "text": comment.body.strip(),
                            "url": url,
                        }
                    )
        except Exception as e:
            logger.warning("Failed to scrape %s: %s", url, e)

    df = pd.DataFrame(rows)
    logger.info("Scraped %d Reddit posts/comments from %d threads.", len(df), len(urls))
    return df
