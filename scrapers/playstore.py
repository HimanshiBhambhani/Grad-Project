"""
scrapers/playstore.py — Pathway A: Live Play Store review scraper.
Pulls latest Blinkit reviews via google-play-scraper.
"""

import logging

import pandas as pd
from google_play_scraper import Sort, reviews

import config

logger = logging.getLogger(__name__)


def scrape_playstore_reviews(
    count: int = config.PLAY_STORE_BATCH_SIZE,
    continuation_token=None,
) -> pd.DataFrame:
    """
    Pull a batch of the newest Play Store reviews for Blinkit.

    Returns:
        DataFrame with columns: [source_raw, date, rating, text, url]
        continuation_token stored as df.attrs["continuation_token"].
    """
    logger.info(
        "Fetching %d Play Store reviews for %s ...",
        count,
        config.PLAY_STORE_PACKAGE,
    )

    result, token = reviews(
        config.PLAY_STORE_PACKAGE,
        lang=config.PLAY_STORE_LANG,
        country=config.PLAY_STORE_COUNTRY,
        sort=Sort.NEWEST,
        count=count,
        continuation_token=continuation_token,
    )

    if not result:
        logger.warning("No reviews returned from Play Store.")
        return pd.DataFrame(columns=["source_raw", "date", "rating", "text", "url"])

    rows = []
    for r in result:
        rows.append(
            {
                "source_raw": "play_store_live",
                "date": str(r.get("at", "")),
                "rating": r.get("score", ""),
                "text": r.get("content", ""),
                "url": f"https://play.google.com/store/apps/details?id={config.PLAY_STORE_PACKAGE}",
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["continuation_token"] = token
    logger.info("Fetched %d live Play Store reviews.", len(df))
    return df


def scrape_playstore_all(max_reviews: int = 500) -> pd.DataFrame:
    """Paginate through Play Store reviews up to max_reviews."""
    all_dfs = []
    token = None
    fetched = 0

    while fetched < max_reviews:
        batch_size = min(config.PLAY_STORE_BATCH_SIZE, max_reviews - fetched)
        df = scrape_playstore_reviews(count=batch_size, continuation_token=token)
        if df.empty:
            break
        all_dfs.append(df)
        token = df.attrs.get("continuation_token")
        fetched += len(df)
        if token is None:
            break

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info("Total live Play Store reviews scraped: %d", len(combined))
    return combined
