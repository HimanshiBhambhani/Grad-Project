"""
scrapers/appstore.py — App Store review scraper for Blinkit.
Uses the `app_store_scraper` package to pull iOS reviews.
"""

import logging
from datetime import datetime

import pandas as pd

import config

logger = logging.getLogger(__name__)


def scrape_appstore_reviews(count: int = 200) -> pd.DataFrame:
    """
    Pull the latest App Store reviews for Blinkit (iOS).

    Args:
        count: Maximum number of reviews to fetch.

    Returns:
        DataFrame with columns: [source_raw, date, rating, text, url]
    """
    from app_store_scraper import AppStore

    logger.info(
        "Fetching up to %d App Store reviews for %s (ID: %s) ...",
        count,
        config.APPSTORE_APP_NAME,
        config.APPSTORE_APP_ID,
    )

    app = AppStore(
        country=config.APPSTORE_COUNTRY,
        app_name=config.APPSTORE_APP_NAME,
        app_id=config.APPSTORE_APP_ID,
    )

    try:
        app.review(how_many=count)
    except Exception as e:
        logger.error("App Store scraping failed: %s", e)
        return pd.DataFrame(columns=["source_raw", "date", "rating", "text", "url"])

    if not app.reviews:
        logger.warning("No reviews returned from App Store.")
        return pd.DataFrame(columns=["source_raw", "date", "rating", "text", "url"])

    rows = []
    for r in app.reviews:
        date_val = r.get("date", "")
        if isinstance(date_val, datetime):
            date_val = date_val.strftime("%Y-%m-%d")

        rows.append(
            {
                "source_raw": "appstore_live",
                "date": str(date_val),
                "rating": r.get("rating", ""),
                "text": r.get("review", "") or r.get("title", ""),
                "url": f"https://apps.apple.com/{config.APPSTORE_COUNTRY}/app/{config.APPSTORE_APP_NAME}/id{config.APPSTORE_APP_ID}",
            }
        )

    df = pd.DataFrame(rows)
    # Drop empty text rows
    df = df[df["text"].str.strip().str.len() > 0]
    logger.info("Fetched %d App Store reviews.", len(df))
    return df
