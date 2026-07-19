"""
engine/themes.py — Theme clustering and pattern identification.
Groups reviews into emergent themes using TF-IDF + simple clustering.
Works offline (no API needed).
"""

import logging
import re
from collections import Counter

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────── Keyword-based Theme Definitions ───────────────────

THEME_PATTERNS = {
    "Counterfeit / Fake Products": [
        r"\bfake\b", r"\bcounterfeit\b", r"\bduplicate\b", r"\bnot\s*original\b",
        r"\bnot\s*genuine\b", r"\bfirst\s*copy\b", r"\bknock\s*off\b",
    ],
    "Warranty & Return Anxiety": [
        r"\bwarranty\b", r"\breturn\b", r"\bno\s*return\b", r"\bexchange\b",
        r"\bnon[\s-]*returnable\b", r"\brefund\s*denied\b", r"\bno\s*replacement\b",
    ],
    "Dark Store Storage Concerns": [
        r"\bstorage\b", r"\bdark\s*store\b", r"\bheat\b", r"\btemperature\b",
        r"\bexpir\w*\b", r"\bdamaged\b", r"\btampered\b", r"\bpackag\w*\b",
    ],
    "Price Premium vs Alternatives": [
        r"\bexpensive\b", r"\boverpriced\b", r"\bcheaper\b", r"\bamazon\b",
        r"\bnykaa\b", r"\bflipkart\b", r"\bprice\b", r"\bMRP\b", r"\bcostl\w*\b",
    ],
    "Category Unawareness": [
        r"\bdidn'?t\s*know\b", r"\bhad\s*no\s*idea\b", r"\bnever\s*knew\b",
        r"\bnot\s*aware\b", r"\bdidn'?t\s*realize\b", r"\bnever\s*saw\b",
    ],
    "Search & Navigation Friction": [
        r"\bcan'?t\s*find\b", r"\bsearch\b", r"\bhidden\b", r"\bnot\s*visible\b",
        r"\bnavigat\w*\b", r"\bhard\s*to\s*find\b", r"\bwhere\s*is\b",
    ],
    "Grocery-Only Habit Lock": [
        r"\bonly\s*(use|buy|order)\b.*\bgrocery\b", r"\bjust\s*for\s*grocery\b",
        r"\broutine\b", r"\bsame\s*order\b", r"\bhabit\b", r"\balways\s*buy\b",
    ],
    "Customer Support Failures": [
        r"\bsupport\b", r"\bcustomer\s*care\b", r"\bcustomer\s*service\b",
        r"\bbot\b", r"\bno\s*response\b", r"\bunhelpful\b", r"\bescalat\w*\b",
    ],
    "Product Quality Issues": [
        r"\bquality\b", r"\bdefective\b", r"\bbroken\b", r"\bnot\s*working\b",
        r"\bfaulty\b", r"\blow\s*quality\b", r"\bbad\s*quality\b",
    ],
    "Trust & Brand Verification": [
        r"\btrust\b", r"\bauthentic\w*\b", r"\bverif\w*\b", r"\blegit\b",
        r"\boriginal\b", r"\bbrand\b", r"\bserial\s*number\b",
    ],
}

# Compile patterns
_COMPILED_THEMES = {
    name: [re.compile(p, re.IGNORECASE) for p in patterns]
    for name, patterns in THEME_PATTERNS.items()
}


def identify_themes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify themes in the cleaned data using keyword pattern matching.

    Returns DataFrame with theme assignments and match counts.
    """
    logger.info("Identifying themes across %d reviews...", len(df))

    theme_assignments = []
    for _, row in df.iterrows():
        text = str(row.get("Raw Content", ""))
        matched_themes = []

        for theme_name, patterns in _COMPILED_THEMES.items():
            matches = sum(1 for p in patterns if p.search(text))
            if matches > 0:
                matched_themes.append((theme_name, matches))

        # Sort by match count, take the top theme
        matched_themes.sort(key=lambda x: x[1], reverse=True)

        if matched_themes:
            theme_assignments.append({
                "primary_theme": matched_themes[0][0],
                "theme_confidence": matched_themes[0][1],
                "all_themes": [t[0] for t in matched_themes],
            })
        else:
            theme_assignments.append({
                "primary_theme": "Uncategorized",
                "theme_confidence": 0,
                "all_themes": [],
            })

    themes_df = pd.DataFrame(theme_assignments)
    df = df.copy()
    df["Primary Theme"] = themes_df["primary_theme"].values
    df["Theme Confidence"] = themes_df["theme_confidence"].values

    # Log summary
    theme_counts = df["Primary Theme"].value_counts()
    logger.info("Theme distribution:")
    for theme, count in theme_counts.items():
        logger.info("  %-35s %d", theme, count)

    return df


def get_theme_summary(df: pd.DataFrame) -> list[dict]:
    """
    Generate a summary of all identified themes with representative quotes.

    Returns list of theme summary dicts.
    """
    if "Primary Theme" not in df.columns:
        df = identify_themes(df)

    summaries = []
    for theme in df["Primary Theme"].unique():
        theme_df = df[df["Primary Theme"] == theme]

        # Get category and pillar distribution for this theme
        categories = theme_df["Target Category"].value_counts().to_dict()
        pillars = theme_df["Friction Pillar"].value_counts().to_dict()

        # Get top quotes (prefer high-confidence matches)
        top_rows = theme_df.nlargest(3, "Theme Confidence")
        quotes = top_rows["Raw Content"].str[:250].tolist()

        summaries.append({
            "theme": theme,
            "count": len(theme_df),
            "percentage": round(len(theme_df) / len(df) * 100, 1),
            "categories": categories,
            "pillars": pillars,
            "representative_quotes": quotes,
        })

    # Sort by count descending
    summaries.sort(key=lambda x: x["count"], reverse=True)
    return summaries


def get_cross_theme_patterns(df: pd.DataFrame) -> dict:
    """
    Identify cross-cutting patterns: which themes co-occur with which categories/pillars.
    """
    if "Primary Theme" not in df.columns:
        df = identify_themes(df)

    patterns = {
        "theme_by_category": {},
        "theme_by_pillar": {},
        "category_theme_matrix": {},
    }

    for cat in df["Target Category"].unique():
        cat_df = df[df["Target Category"] == cat]
        patterns["theme_by_category"][cat] = cat_df["Primary Theme"].value_counts().to_dict()

    for pillar in df["Friction Pillar"].unique():
        pillar_df = df[df["Friction Pillar"] == pillar]
        patterns["theme_by_pillar"][pillar] = pillar_df["Primary Theme"].value_counts().to_dict()

    return patterns
