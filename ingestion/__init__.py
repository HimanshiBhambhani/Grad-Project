"""
ingestion/historical.py — Pathway B (continued): Historical CSV & pre-classified data ingestion.
Loads and normalizes reviews_raw.csv, Ankesh-reviews_raw.csv, and the pre-classified Reddit CSV.
"""

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)


def _normalize_category(raw_cat: str) -> list[str]:
    """
    Map a raw category label to canonical label(s).
    Handles aliases from the pre-classified CSV and multi-label splits.
    """
    raw_cat = raw_cat.strip()

    # Check alias map first (pre-classified CSV labels)
    if raw_cat in config.CATEGORY_ALIAS_MAP:
        return [config.CATEGORY_ALIAS_MAP[raw_cat]]

    # Handle multi-category (e.g., "Electronics/Accessories; Beauty/Skincare/Cosmetics")
    if config.MULTI_CATEGORY_SEPARATOR in raw_cat:
        parts = raw_cat.split(config.MULTI_CATEGORY_SEPARATOR)
        cats = []
        for p in parts:
            p = p.strip()
            if p in config.CATEGORY_ALIAS_MAP:
                cats.append(config.CATEGORY_ALIAS_MAP[p])
            else:
                cats.append(p.lower().strip())
        return cats if cats else [raw_cat.lower()]

    return [raw_cat.lower().strip()]


def load_reviews_csv(filepath) -> pd.DataFrame:
    """
    Load a raw reviews CSV (reviews_raw.csv or Ankesh-reviews_raw.csv).

    Input columns: [Reddit(=index), ''(=source), date, rating, text, category, url]
    Output columns: [source_raw, date, rating, text, category, url]
    """
    filepath = str(filepath)
    logger.info("Loading historical reviews from %s ...", filepath)

    df = pd.read_csv(filepath, dtype=str, keep_default_na=False)

    # The unnamed column '' holds the source (Play Store, Reddit, etc.)
    col_map = {}
    for col in df.columns:
        if col == "":
            col_map[col] = "source_raw"
        elif col == "Reddit":
            col_map[col] = "_index"

    df = df.rename(columns=col_map)

    # Drop the numeric index column if present
    if "_index" in df.columns:
        df = df.drop(columns=["_index"])

    # Ensure required columns
    for col in ["source_raw", "date", "rating", "text"]:
        if col not in df.columns:
            df[col] = ""

    if "category" not in df.columns:
        df["category"] = ""
    if "url" not in df.columns:
        df["url"] = ""

    logger.info("Loaded %d rows from %s", len(df), filepath)
    return df


def load_preclassified_csv() -> pd.DataFrame:
    """
    Load the pre-classified Reddit thread CSV
    (SourceThreadTitleSite-PlatformBlinkit-TargetCatego.csv).

    This data is already in near-output-schema format.
    Maps it to the unified intermediate format for pipeline compatibility.
    """
    filepath = config.PRECLASSIFIED_CSV
    if not filepath.exists():
        logger.warning("Pre-classified CSV not found: %s", filepath)
        return pd.DataFrame()

    logger.info("Loading pre-classified data from %s ...", filepath)
    df = pd.read_csv(str(filepath), dtype=str, keep_default_na=False)

    # Map to intermediate columns
    rows = []
    for _, row in df.iterrows():
        source = row.get("Source (Thread Title / Site)", "")
        category_raw = row.get("Target Category", "")
        pillar = row.get("Inferred Friction Pillar", "")
        text = row.get(
            "Core Customer Voice String (Direct quote or tight paraphrase from the complaint)",
            "",
        )

        categories = _normalize_category(category_raw)

        for cat in categories:
            rows.append(
                {
                    "source_raw": "Reddit (post)",  # These are all Reddit sourced
                    "date": "",
                    "rating": "",
                    "text": text,
                    "category": cat,
                    "url": "",
                    "_preclassified_pillar": pillar,
                    "_preclassified_source_label": source,
                }
            )

    result = pd.DataFrame(rows)
    logger.info(
        "Loaded %d rows from pre-classified CSV (expanded from %d).",
        len(result),
        len(df),
    )
    return result


def load_all_historical() -> pd.DataFrame:
    """
    Load and combine all historical data sources into a single DataFrame.
    """
    dfs = []

    # Load both review dump CSVs
    for csv_path in [config.REVIEWS_RAW_CSV, config.ANKESH_REVIEWS_RAW_CSV]:
        if csv_path.exists():
            dfs.append(load_reviews_csv(csv_path))
        else:
            logger.warning("CSV not found: %s", csv_path)

    # Load pre-classified Reddit data
    pre_df = load_preclassified_csv()
    if not pre_df.empty:
        dfs.append(pre_df)

    if not dfs:
        logger.error("No historical data files found!")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate on text content (exact matches)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"], keep="first")
    after = len(combined)
    if before != after:
        logger.info("Deduplicated: %d → %d rows (removed %d dupes).", before, after, before - after)

    logger.info("Total historical data loaded: %d rows.", len(combined))
    return combined
