"""
pipeline/classifier.py — Semantic classification using Groq (Llama 3.3 70B).

Assigns each surviving row:
  1. A Target Category (if not already classified)
  2. A Friction Pillar (exactly one of 4)
  3. An Opportunity hypothesis
"""

import json
import logging
import time

import pandas as pd

import config

logger = logging.getLogger(__name__)

# ─────────────────── Classification Prompt ───────────────────

SYSTEM_PROMPT = """You are a strategic product analyst for Blinkit (India's quick-commerce platform).

Your job is to classify customer feedback into EXACTLY ONE friction pillar and assign a target category and opportunity hypothesis.

## Target Categories (pick one):
- electronics — Electronics, gadgets, accessories, chargers, earbuds, cables
- personal_care_beauty — Beauty, skincare, cosmetics, makeup, grooming
- pet — Pet food, pet care, pet supplies
- baby — Baby products, diapers, infant care, baby food
- home_cleaning — Cleaning supplies, detergents, home care
- intimate_personal — Intimate care, personal hygiene
- pharmacy_health — Medicines, health supplements, wellness
- general — Only if it clearly discusses non-grocery shopping behavior but doesn't fit above

## Friction Pillars (pick exactly one):
1. **Habitual Tunnel Vision** — User treats Blinkit as grocery-only, never explores other categories. Path-dependency, "surgical" buying behavior.
2. **Quality & Authenticity Risk** — User doubts product quality/authenticity on Blinkit. Fears about storage, counterfeits, warranty, dark-store conditions.
3. **Discovery Blind Spots** — User didn't know Blinkit sells these products. Poor search, hidden navigation, banner blindness, no category awareness.
4. **Immediate Value Disconnection** — User finds Blinkit overpriced vs Amazon/Nykaa/Supertails. Pack sizes too small. Bad unit economics.

## Output Format (strict JSON):
{
  "target_category": "<category>",
  "friction_pillar": "<exact pillar name>",
  "opportunity": "<1-2 sentence PM hypothesis for fixing this friction>"
}"""

USER_PROMPT_TEMPLATE = """Classify this customer feedback:

"{text}"

Context: Source={source}, Rating={rating}
If the category is already known as "{category}", validate or correct it.

Return ONLY the JSON object."""


def _classify_single(text: str, source: str, rating: str, category: str) -> dict:
    """Classify a single text row using Groq API (Llama 3.3 70B)."""
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)

    user_msg = USER_PROMPT_TEMPLATE.format(
        text=text[:1500],  # Truncate very long texts to save tokens
        source=source,
        rating=rating,
        category=category,
    )

    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content.strip()
        result = json.loads(content)

        # Validate pillar
        pillar = result.get("friction_pillar", "")
        if pillar not in config.FRICTION_PILLARS:
            # Fuzzy match
            for fp in config.FRICTION_PILLARS:
                if fp.lower() in pillar.lower() or pillar.lower() in fp.lower():
                    result["friction_pillar"] = fp
                    break

        return result

    except Exception as e:
        logger.warning("Classification failed for text: %.50s... Error: %s", text, e)
        return {
            "target_category": category or "general",
            "friction_pillar": "Quality & Authenticity Risk",
            "opportunity": "Classification failed — manual review needed.",
        }


def _classify_batch(texts: list[str], sources: list[str], ratings: list[str], categories: list[str]) -> list[dict]:
    """Classify a batch of texts. Currently sequential with rate limiting."""
    results = []
    for i, (text, source, rating, cat) in enumerate(zip(texts, sources, ratings, categories)):
        result = _classify_single(text, source, rating, cat)
        results.append(result)

        # Rate limiting: be gentle with the API
        if (i + 1) % 20 == 0:
            logger.info("Classified %d / %d rows...", i + 1, len(texts))
            time.sleep(1)

    return results


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify all rows in the DataFrame.

    For rows that already have pre-classified pillars (from the pre-classified CSV),
    those are preserved. Other rows go through the AI classifier.

    Adds columns: Target Category, Friction Pillar, Opportunity
    """
    if not config.GROQ_API_KEY:
        logger.error(
            "GROQ_API_KEY not set. Cannot run AI classification. "
            "Set it in .env or export as environment variable."
        )
        # Fall back: use existing categories, leave pillars empty
        df["Target Category"] = df.get("category", "general")
        df["Friction Pillar"] = df.get("_preclassified_pillar", "")
        df["Opportunity"] = ""
        return df

    logger.info("Starting AI classification for %d rows...", len(df))

    # Split: pre-classified vs needs-classification
    has_pillar = (
        df.get("_preclassified_pillar", pd.Series(dtype=str)).str.len() > 0
    )

    # Pre-classified rows: use existing labels
    if has_pillar.any():
        preclassified = df[has_pillar].copy()
        preclassified["Target Category"] = preclassified["category"]
        preclassified["Friction Pillar"] = preclassified["_preclassified_pillar"]
        preclassified["Opportunity"] = ""
        logger.info("Using %d pre-classified rows as-is.", len(preclassified))
    else:
        preclassified = pd.DataFrame()

    # Rows needing classification
    needs_class = df[~has_pillar].copy() if has_pillar.any() else df.copy()

    if not needs_class.empty:
        results = _classify_batch(
            texts=needs_class["text"].tolist(),
            sources=needs_class["source_raw"].tolist(),
            ratings=needs_class["rating"].tolist(),
            categories=needs_class.get("category", pd.Series("", index=needs_class.index)).tolist(),
        )

        needs_class["Target Category"] = [r.get("target_category", "general") for r in results]
        needs_class["Friction Pillar"] = [r.get("friction_pillar", "") for r in results]
        needs_class["Opportunity"] = [r.get("opportunity", "") for r in results]

    # Combine
    if not preclassified.empty and not needs_class.empty:
        result = pd.concat([preclassified, needs_class], ignore_index=True)
    elif not preclassified.empty:
        result = preclassified
    else:
        result = needs_class

    logger.info("Classification complete. %d rows classified.", len(result))
    return result


def _infer_category(text: str) -> str:
    """Infer target category from text using keyword matching."""
    import re as _re
    text_lower = text.lower()
    category_rules = [
        ("electronics", _re.compile(
            r"\b(earbuds|earphones|headphones|charger|cable|phone|laptop|tablet|"
            r"speaker|smartwatch|watch|power\s*bank|adapter|usb|bluetooth|"
            r"electronics|gadget|ps5|playstation|gaming|controller|mouse|keyboard)\b",
            _re.IGNORECASE)),
        ("personal_care_beauty", _re.compile(
            r"\b(beauty|skincare|cosmetic|serum|moisturizer|sunscreen|face\s*wash|"
            r"shampoo|makeup|lipstick|foundation|cream|lotion|perfume|fragrance|"
            r"hair\s*oil|conditioner|body\s*wash|deodorant|deo)\b",
            _re.IGNORECASE)),
        ("pharmacy_health", _re.compile(
            r"\b(pharmacy|medicine|vitamin|supplement|health|protein|whey|"
            r"fitness|gym|workout|creatine|bcaa|pre.workout|nutraceutical)\b",
            _re.IGNORECASE)),
        ("pet", _re.compile(
            r"\b(pet|dog|cat|puppy|kitten|pet\s*food|pet\s*care|kibble|"
            r"treats|litter|leash|collar)\b", _re.IGNORECASE)),
        ("baby", _re.compile(
            r"\b(baby|diaper|infant|formula|baby\s*food|stroller|"
            r"newborn|toddler|wipes)\b", _re.IGNORECASE)),
        ("home_cleaning", _re.compile(
            r"\b(cleaning|detergent|dishwash|floor\s*cleaner|mop|"
            r"fabric\s*softener|toilet\s*cleaner|disinfectant)\b", _re.IGNORECASE)),
        ("intimate_personal", _re.compile(
            r"\b(intimate|condom|sanitary|pad|tampon|menstrual|"
            r"personal\s*hygiene)\b", _re.IGNORECASE)),
    ]
    for cat_name, pattern in category_rules:
        if pattern.search(text):
            return cat_name
    return "general"


def classify_dataframe_offline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Offline/rule-based classification fallback (no API needed).
    Uses keyword matching for pillar assignment. Less accurate but free.
    """
    import re

    logger.info("Running OFFLINE rule-based classification for %d rows...", len(df))

    pillar_rules = {
        "Habitual Tunnel Vision": re.compile(
            r"\b(only use for grocery|never knew|just for grocery|only order groceries|"
            r"don't explore|never tried|always buy same|routine order|"
            r"surgical|habit|stuck with grocery)\b",
            re.IGNORECASE,
        ),
        "Discovery Blind Spots": re.compile(
            r"\b(didn't know|had no idea|never saw|can't find|no idea|"
            r"where is|not visible|hidden|search doesn't show|"
            r"banner|couldn't find|not aware|didn't realize)\b",
            re.IGNORECASE,
        ),
        "Immediate Value Disconnection": re.compile(
            r"\b(expensive|overpriced|costly|cheaper on|amazon has|nykaa|"
            r"supertails|price|MRP|discount|pack size|small pack|"
            r"value for money|not worth|too much|comparison)\b",
            re.IGNORECASE,
        ),
        "Quality & Authenticity Risk": re.compile(
            r"\b(fake|counterfeit|duplicate|quality|damaged|broken|expired|"
            r"warranty|authentic|original|defective|tampered|"
            r"storage|dark store|used|refurbished|not genuine)\b",
            re.IGNORECASE,
        ),
    }

    def _classify_text(text: str) -> tuple[str, str]:
        for pillar, pattern in pillar_rules.items():
            if pattern.search(text):
                # Generate simple opportunity
                opp = f"Address {pillar.lower()} by improving product "
                if pillar == "Quality & Authenticity Risk":
                    opp += "verification, warranty display, and return policies."
                elif pillar == "Discovery Blind Spots":
                    opp += "visibility, search indexing, and category navigation."
                elif pillar == "Immediate Value Disconnection":
                    opp += "pricing, pack sizes, and competitive price badges."
                else:
                    opp += "cross-category discovery nudges and personalized recommendations."
                return pillar, opp

        return "Quality & Authenticity Risk", "Needs manual review for pillar assignment."

    pillars = []
    opportunities = []
    for text in df["text"]:
        p, o = _classify_text(str(text))
        pillars.append(p)
        opportunities.append(o)

    df = df.copy()
    df["Target Category"] = df.get("category", "general")
    # Classify empty categories using keyword matching
    df["Target Category"] = df.apply(
        lambda row: _infer_category(str(row.get("text", "")))
        if str(row.get("Target Category", "")).strip() == ""
        else row["Target Category"],
        axis=1,
    )
    df["Friction Pillar"] = pillars
    df["Opportunity"] = opportunities

    # Use pre-classified pillars where available
    if "_preclassified_pillar" in df.columns:
        mask = df["_preclassified_pillar"].str.len() > 0
        df.loc[mask, "Friction Pillar"] = df.loc[mask, "_preclassified_pillar"]

    logger.info("Offline classification complete.")
    return df
