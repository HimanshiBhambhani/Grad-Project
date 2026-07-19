"""
scrape_reviews.py — Standalone weekly scraper for GitHub Actions.

Scrapes latest reviews from Play Store and App Store,
appends them to the cumulative CSV in Output/scrapes/,
and commits the new data back to the repo.

Usage:
    python scrape_reviews.py                    # Scrape both stores
    python scrape_reviews.py --playstore-only   # Play Store only
    python scrape_reviews.py --appstore-only    # App Store only
    python scrape_reviews.py --max-reviews 200  # Limit per store
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scraper")

# ── Output paths ──
CUMULATIVE_CSV = config.SCRAPE_OUTPUT_DIR / "all_scraped_reviews.csv"
LATEST_CSV = config.SCRAPE_OUTPUT_DIR / "latest_scrape.csv"


def _load_existing() -> pd.DataFrame:
    """Load the cumulative CSV if it exists."""
    if CUMULATIVE_CSV.exists():
        df = pd.read_csv(str(CUMULATIVE_CSV), dtype=str, keep_default_na=False)
        logger.info("Loaded %d existing scraped reviews.", len(df))
        return df
    return pd.DataFrame()


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate on text content."""
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    dupes = before - len(df)
    if dupes > 0:
        logger.info("Removed %d duplicate reviews.", dupes)
    return df


def scrape_playstore(max_reviews: int) -> pd.DataFrame:
    """Scrape Play Store reviews."""
    try:
        from scrapers.playstore import scrape_playstore_all
        df = scrape_playstore_all(max_reviews=max_reviews)
        if not df.empty:
            df["scraped_at"] = datetime.utcnow().isoformat()
        return df
    except ImportError:
        logger.error("google-play-scraper not installed. pip install google-play-scraper")
        return pd.DataFrame()
    except Exception as e:
        logger.error("Play Store scrape failed: %s", e)
        return pd.DataFrame()


def scrape_appstore(max_reviews: int) -> pd.DataFrame:
    """Scrape App Store reviews."""
    try:
        from scrapers.appstore import scrape_appstore_reviews
        df = scrape_appstore_reviews(count=max_reviews)
        if not df.empty:
            df["scraped_at"] = datetime.utcnow().isoformat()
        return df
    except ImportError:
        logger.error("app-store-scraper not installed. pip install app-store-scraper")
        return pd.DataFrame()
    except Exception as e:
        logger.error("App Store scrape failed: %s", e)
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Weekly Review Scraper")
    parser.add_argument("--playstore-only", action="store_true")
    parser.add_argument("--appstore-only", action="store_true")
    parser.add_argument("--max-reviews", type=int, default=200,
                        help="Max reviews per store (default: 200)")
    args = parser.parse_args()

    new_dfs = []

    # ── Scrape Play Store ──
    if not args.appstore_only:
        logger.info("=" * 50)
        logger.info("SCRAPING PLAY STORE")
        logger.info("=" * 50)
        ps_df = scrape_playstore(args.max_reviews)
        if not ps_df.empty:
            new_dfs.append(ps_df)
            logger.info("Play Store: %d new reviews.", len(ps_df))
        else:
            logger.warning("Play Store: 0 reviews scraped.")

    # ── Scrape App Store ──
    if not args.playstore_only:
        logger.info("=" * 50)
        logger.info("SCRAPING APP STORE")
        logger.info("=" * 50)
        as_df = scrape_appstore(args.max_reviews)
        if not as_df.empty:
            new_dfs.append(as_df)
            logger.info("App Store: %d new reviews.", len(as_df))
        else:
            logger.warning("App Store: 0 reviews scraped.")

    if not new_dfs:
        logger.error("No reviews scraped from any source.")
        sys.exit(1)

    # ── Combine new scrapes ──
    new_data = pd.concat(new_dfs, ignore_index=True)
    logger.info("Total new reviews this run: %d", len(new_data))

    # Ensure output directory exists
    CUMULATIVE_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Save latest scrape separately (useful for debugging)
    new_data.to_csv(str(LATEST_CSV), index=False)
    logger.info("Saved latest scrape: %s", LATEST_CSV)

    # ── Merge with cumulative data ──
    existing = _load_existing()
    if not existing.empty:
        # Align columns before concat to avoid NaN mismatches
        shared_cols = list(set(existing.columns) & set(new_data.columns))
        combined = pd.concat(
            [existing[shared_cols], new_data[shared_cols]], ignore_index=True
        )
    else:
        combined = new_data

    combined = _deduplicate(combined)
    combined.to_csv(str(CUMULATIVE_CSV), index=False)
    logger.info("Cumulative total: %d reviews saved to %s", len(combined), CUMULATIVE_CSV)

    # ── Summary ──
    logger.info("")
    logger.info("=" * 50)
    logger.info("SCRAPE SUMMARY")
    logger.info("=" * 50)
    logger.info("New reviews scraped:   %d", len(new_data))
    logger.info("Cumulative total:      %d", len(combined))
    if "source_raw" in combined.columns:
        for src, count in combined["source_raw"].value_counts().items():
            logger.info("  %-20s %d", src, count)

    # Always exit 0 if we scraped any reviews
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Unexpected fatal error: %s", e)
        sys.exit(1)
