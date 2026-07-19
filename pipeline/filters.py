"""
pipeline/filters.py — Three-stage cleaning & noise interception pipeline.

Stage 1: Transactional Noise Filter (blacklisted logistics tokens)
Stage 2: Low-Star / High-Friction Focus (drop 5-star praise)
Stage 3: Vertical Category Isolation (keep only expansion categories)
"""

import logging
import re

import pandas as pd

import config

logger = logging.getLogger(__name__)


# ─────────────────────── Pre-compile patterns ────────────────────────

_BLACKLIST_PATTERN = re.compile(
    "|".join(re.escape(tok) for tok in config.BLACKLISTED_TOKENS),
    re.IGNORECASE,
)

_PRAISE_PATTERN = re.compile(
    "|".join(re.escape(p) for p in config.PRAISE_PATTERNS),
    re.IGNORECASE,
)

_GROCERY_NOISE_PATTERN = re.compile(
    "|".join(re.escape(tok) for tok in config.GROCERY_NOISE_TOKENS),
    re.IGNORECASE,
)

# Expansion category keywords for detecting non-grocery content in "general" rows
_EXPANSION_KEYWORDS = re.compile(
    r"\b("
    r"electronics|earbuds|earphones|headphones|charger|cable|phone|laptop|tablet|"
    r"speaker|smartwatch|watch|power\s*bank|adapter|usb|bluetooth|"
    r"beauty|skincare|cosmetic|serum|moisturizer|sunscreen|face\s*wash|shampoo|"
    r"makeup|lipstick|foundation|cream|lotion|perfume|fragrance|"
    r"pet|dog|cat|pet\s*food|pet\s*care|kibble|"
    r"baby|diaper|infant|formula|baby\s*food|stroller|"
    r"cleaning|detergent|dishwash|floor\s*cleaner|mop|"
    r"pharmacy|medicine|tablet|vitamin|supplement|health|"
    r"intimate|condom|sanitary|pad|tampon"
    r")\b",
    re.IGNORECASE,
)

# Cross-shopping / inertia keywords — used for curated Reddit threads that were
# hand-picked for relevance but may not mention a specific product category.
_CROSS_SHOPPING_KEYWORDS = re.compile(
    r"\b("
    r"blinkit|zepto|swiggy\s*instamart|instamart|dunzo|bigbasket|big\s*basket|"
    r"amazon|flipkart|myntra|nykaa|meesho|jiomart|"
    r"quick\s*commerce|instant\s*delivery|10.min|minutes|"
    r"fake|duplicate|counterfeit|original|genuine|authentic|first\s*copy|"
    r"quality|trust|refund|return|replacement|customer\s*care|support|"
    r"expensive|overpriced|costly|price|mrp|cheaper|discount|"
    r"order|deliver|package|cancel|"
    r"switch|alternative|instead|compare|better|worse|prefer|"
    r"habit|loyalty|stick\s*to|go\s*back|stopped\s*using|never\s*again"
    r")\b",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Basic text cleanup: strip markdown junk, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove markdown artifacts
    text = re.sub(r"[#*_~`>]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stage1_transactional_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 1: Drop rows where text primarily discusses logistics / operations.
    Scans for blacklisted tokens.
    """
    before = len(df)

    mask = df["text"].apply(
        lambda t: not bool(_BLACKLIST_PATTERN.search(str(t)))
    )
    df = df[mask].copy()

    dropped = before - len(df)
    logger.info(
        "Stage 1 (Transactional Noise): %d → %d rows (dropped %d)",
        before, len(df), dropped,
    )
    return df


def stage2_rating_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Keep only rows with rating <= 4, or text expressing friction.
    Drop 5-star praise strings.
    """
    before = len(df)

    def _should_keep(row) -> bool:
        text = str(row.get("text", ""))
        rating_str = str(row.get("rating", "")).strip()

        # If no rating (Reddit, etc.), keep based on text content
        if not rating_str or rating_str == "":
            # No rating → keep unless it's pure praise
            return not bool(_PRAISE_PATTERN.search(text))

        try:
            rating = float(rating_str)
        except ValueError:
            return True  # Can't parse rating → keep

        # Drop 5-star reviews that are pure praise
        if rating >= 5:
            # Check if the text has substantive friction content despite 5 stars
            has_friction = any(
                word in text.lower()
                for word in [
                    "but", "however", "although", "issue", "problem",
                    "complaint", "disappointed", "worried", "anxious",
                    "fake", "counterfeit", "doubt", "expensive", "costly",
                    "overpriced", "didn't know", "had no idea", "can't find",
                    "not available", "quality", "damaged", "broken",
                ]
            )
            if has_friction:
                return True
            return False

        # Keep everything rated 4 or below
        return True

    mask = df.apply(_should_keep, axis=1)
    df = df[mask].copy()

    dropped = before - len(df)
    logger.info(
        "Stage 2 (Rating Filter): %d → %d rows (dropped %d)",
        before, len(df), dropped,
    )
    return df


def stage3_category_isolation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 3: Keep only rows tied to expansion categories.
    Drop grocery staples. Inspect 'general' rows contextually.
    """
    before = len(df)

    def _should_keep(row) -> bool:
        category = str(row.get("category", "")).strip().lower()
        text = str(row.get("text", ""))
        source = str(row.get("source", "")).lower()

        # Explicitly in a KEEP category
        if category in config.KEEP_CATEGORIES:
            # But also check it's not actually a grocery complaint
            if _GROCERY_NOISE_PATTERN.search(text):
                return False
            return True

        # Explicitly in a DROP category
        if category in config.DROP_CATEGORIES:
            return False

        # "general" or unknown → contextual inspection
        if category == config.GENERAL_CATEGORY or category == "":
            # Always keep if text references expansion-category products
            if _EXPANSION_KEYWORDS.search(text):
                return True
            # For curated Reddit data: also keep if cross-shopping relevant
            if "reddit" in source:
                return bool(_CROSS_SHOPPING_KEYWORDS.search(text))
            return False

        # Unknown category → try keyword match
        return bool(_EXPANSION_KEYWORDS.search(text))

    mask = df.apply(_should_keep, axis=1)
    df = df[mask].copy()

    dropped = before - len(df)
    logger.info(
        "Stage 3 (Category Isolation): %d → %d rows (dropped %d)",
        before, len(df), dropped,
    )
    return df


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all three cleaning stages sequentially.

    Input: Raw DataFrame with at least [text, rating, category] columns.
    Output: Cleaned DataFrame with only expansion-relevant friction rows.
    """
    logger.info("=" * 60)
    logger.info("CLEANING PIPELINE START — %d input rows", len(df))
    logger.info("=" * 60)

    # Clean text first
    df["text"] = df["text"].apply(clean_text)

    # Drop empty text
    df = df[df["text"].str.len() > 10].copy()
    logger.info("After empty-text removal: %d rows", len(df))

    # Run three stages
    df = stage1_transactional_noise(df)
    df = stage2_rating_filter(df)
    df = stage3_category_isolation(df)

    logger.info("=" * 60)
    logger.info("CLEANING PIPELINE DONE — %d surviving rows", len(df))
    logger.info("=" * 60)

    return df
