"""
main.py — Unified Pipeline Orchestrator.

Runs the full data extraction, scraping, filtering, and classification engine.
Combines Pathway A (live scraping) + Pathway B (historical + Reddit)
through the 3-stage cleaning filter and semantic classifier.

Usage:
    python main.py                          # Historical data only (no API keys needed)
    python main.py --live                   # Include live Play Store scraping
    python main.py --live --reddit          # Include live Reddit scraping
    python main.py --classify ai            # Use Groq (Llama 3.3) for classification
    python main.py --classify offline       # Use rule-based classification (default)
    python main.py --max-playstore 200      # Limit Play Store scrape count
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from ingestion import load_all_historical
from pipeline.filters import run_cleaning_pipeline
from pipeline.classifier import classify_dataframe, classify_dataframe_offline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _map_source_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw source labels to output schema labels."""
    df["Source"] = df["source_raw"].map(config.SOURCE_LABELS).fillna("Historical Dump")

    # For pre-classified rows, use their original source label if available
    if "_preclassified_source_label" in df.columns:
        mask = df["_preclassified_source_label"].str.len() > 0
        df.loc[mask, "Source"] = df.loc[mask, "_preclassified_source_label"]

    return df


def _finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    """Shape the DataFrame to match the output schema."""
    df = _map_source_labels(df)
    df["Platform"] = config.PLATFORM
    df["Raw Content"] = df["text"]

    # Select and order output columns
    output = df[config.OUTPUT_COLUMNS].copy()

    # Clean up any remaining empty values
    output = output.fillna("")
    output = output[output["Raw Content"].str.len() > 10]

    return output.reset_index(drop=True)


def run_pipeline(
    include_live_playstore: bool = False,
    include_live_appstore: bool = False,
    include_live_reddit: bool = False,
    classify_mode: str = "offline",
    max_playstore: int = 500,
    max_appstore: int = 200,
) -> pd.DataFrame:
    """
    Execute the full pipeline end-to-end.

    Args:
        include_live_playstore: Whether to scrape live Play Store reviews
        include_live_appstore: Whether to scrape live App Store reviews
        include_live_reddit: Whether to scrape live Reddit threads
        classify_mode: 'ai' for Groq classification, 'offline' for rule-based
        max_playstore: Max Play Store reviews to scrape
        max_appstore: Max App Store reviews to scrape

    Returns:
        Final cleaned, classified DataFrame in output schema format.
    """
    all_dfs = []

    # ──── Pathway B: Historical Data (always loaded) ────
    logger.info("=" * 70)
    logger.info("PHASE 1: INGESTION")
    logger.info("=" * 70)

    historical = load_all_historical()
    if not historical.empty:
        all_dfs.append(historical)
        logger.info("Historical data: %d rows loaded.", len(historical))

    # ──── Pathway A: Live Play Store ────
    if include_live_playstore:
        try:
            from scrapers.playstore import scrape_playstore_all

            live_ps = scrape_playstore_all(max_reviews=max_playstore)
            if not live_ps.empty:
                all_dfs.append(live_ps)
                logger.info("Live Play Store: %d reviews scraped.", len(live_ps))
        except ImportError:
            logger.error(
                "google-play-scraper not installed. "
                "Run: pip install google-play-scraper"
            )
        except Exception as e:
            logger.error("Play Store scraping failed: %s", e)

    # ──── Pathway A: Live App Store ────
    if include_live_appstore:
        try:
            from scrapers.appstore import scrape_appstore_reviews

            live_as = scrape_appstore_reviews(count=max_appstore)
            if not live_as.empty:
                all_dfs.append(live_as)
                logger.info("Live App Store: %d reviews scraped.", len(live_as))
        except ImportError:
            logger.error(
                "app-store-scraper not installed. "
                "Run: pip install app-store-scraper"
            )
        except Exception as e:
            logger.error("App Store scraping failed: %s", e)

    # ──── Pathway B: Live Reddit ────
    if include_live_reddit:
        try:
            from scrapers.reddit import scrape_reddit_threads

            live_reddit = scrape_reddit_threads()
            if not live_reddit.empty:
                all_dfs.append(live_reddit)
                logger.info("Live Reddit: %d posts/comments scraped.", len(live_reddit))
        except ImportError:
            logger.error("praw not installed. Run: pip install praw")
        except Exception as e:
            logger.error("Reddit scraping failed: %s", e)

    # ──── Combine all sources ────
    if not all_dfs:
        logger.error("No data loaded from any source. Exiting.")
        return pd.DataFrame(columns=config.OUTPUT_COLUMNS)

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info("Total combined data: %d rows", len(combined))

    # ──── PHASE 2: CLEANING PIPELINE ────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: CLEANING PIPELINE")
    logger.info("=" * 70)

    cleaned = run_cleaning_pipeline(combined)

    if cleaned.empty:
        logger.warning("All rows were filtered out! Check filter rules.")
        return pd.DataFrame(columns=config.OUTPUT_COLUMNS)

    # ──── PHASE 3: CLASSIFICATION ────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3: CLASSIFICATION (%s mode)", classify_mode.upper())
    logger.info("=" * 70)

    if classify_mode == "ai":
        classified = classify_dataframe(cleaned)
    else:
        classified = classify_dataframe_offline(cleaned)

    # ──── PHASE 4: FINALIZE OUTPUT ────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 4: FINALIZE OUTPUT")
    logger.info("=" * 70)

    output = _finalize_output(classified)

    # ──── Save outputs ────
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    output.to_csv(config.CLEAN_OUTPUT_CSV, index=False)
    logger.info("Saved CSV: %s (%d rows)", config.CLEAN_OUTPUT_CSV, len(output))

    try:
        output.to_parquet(config.CLEAN_OUTPUT_PARQUET, index=False)
        logger.info("Saved Parquet: %s", config.CLEAN_OUTPUT_PARQUET)
    except Exception:
        logger.warning("Parquet save failed (pyarrow may not be installed).")

    # ──── Summary Stats ────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 70)
    logger.info("Total output rows: %d", len(output))
    logger.info("")
    logger.info("Category distribution:")
    for cat, count in output["Target Category"].value_counts().items():
        logger.info("  %-25s %d", cat, count)
    logger.info("")
    logger.info("Friction Pillar distribution:")
    for pillar, count in output["Friction Pillar"].value_counts().items():
        logger.info("  %-30s %d", pillar, count)
    logger.info("")
    logger.info("Source distribution:")
    for src, count in output["Source"].value_counts().items():
        logger.info("  %-30s %d", src, count)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Blinkit Multi-Channel Data Ingestion & Analytics Engine"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Include live Play Store + App Store scraping"
    )
    parser.add_argument(
        "--appstore", action="store_true",
        help="Include live App Store scraping (requires app-store-scraper)"
    )
    parser.add_argument(
        "--reddit", action="store_true",
        help="Include live Reddit scraping (requires praw + API keys in .env)"
    )
    parser.add_argument(
        "--classify", choices=["ai", "offline"], default="offline",
        help="Classification mode: 'ai' (Groq LLM) or 'offline' (rule-based, default)"
    )
    parser.add_argument(
        "--max-playstore", type=int, default=500,
        help="Max Play Store reviews to scrape (default: 500)"
    )

    args = parser.parse_args()

    output = run_pipeline(
        include_live_playstore=args.live,
        include_live_appstore=args.live or args.appstore,
        include_live_reddit=args.reddit,
        classify_mode=args.classify,
        max_playstore=args.max_playstore,
    )

    if output.empty:
        logger.error("Pipeline produced no output.")
        sys.exit(1)

    logger.info("Done! Output saved to %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
