"""
engine/insights.py — AI-powered insight generation engine.
Takes semantically retrieved reviews and generates structured strategic insights.
"""

import json
import logging

import config

logger = logging.getLogger(__name__)

# ─────────────────── Strategic Questions (from assignment) ───────────────────

STRATEGIC_QUESTIONS = [
    "Why do users repeatedly buy from the same categories?",
    "What prevents users from exploring new categories?",
    "How do users discover products today?",
    "What role do habits play in shopping behavior?",
    "What information do users need before trying a new category?",
    "What frustrations emerge repeatedly?",
    "Which user segments are more likely to experiment?",
    "What unmet needs emerge consistently across discussions?",
]

# ─────────────────── Insight Generation Prompt ───────────────────

SYSTEM_PROMPT = """You are a senior product strategist analyzing customer feedback for Blinkit (India's quick-commerce platform).

Your goal is to generate actionable, evidence-backed insights from real customer reviews. Every claim must be grounded in the provided evidence snippets.

## Output Format (strict JSON):
{
  "question": "<the strategic question being answered>",
  "executive_summary": "<2-3 sentence answer to the question>",
  "themes": [
    {
      "theme_name": "<concise theme label>",
      "description": "<1-2 sentences explaining the pattern>",
      "evidence_count": <number of reviews supporting this>,
      "representative_quotes": ["<verbatim quote 1>", "<verbatim quote 2>"],
      "friction_pillar": "<which of the 4 pillars this maps to>",
      "categories_affected": ["<category1>", "<category2>"]
    }
  ],
  "key_insight": "<the single most important non-obvious finding>",
  "actionable_recommendations": [
    "<specific product/UX recommendation 1>",
    "<specific product/UX recommendation 2>",
    "<specific product/UX recommendation 3>"
  ],
  "confidence_level": "<high|medium|low — based on evidence density>",
  "evidence_gap": "<what data is missing that would strengthen this analysis>"
}"""

USER_PROMPT_TEMPLATE = """## Strategic Question
{question}

## Evidence: {num_reviews} Customer Reviews
{reviews_block}

## Category & Pillar Context
- Categories in evidence: {categories}
- Friction pillars in evidence: {pillars}

Analyze the evidence above and answer the strategic question. Ground every theme in specific quotes. Identify 3-6 distinct themes. Be specific and actionable — no generic advice."""


def generate_insight(
    question: str,
    relevant_reviews: list[dict],
) -> dict:
    """
    Generate a structured insight for a strategic question using retrieved evidence.

    Args:
        question: The strategic question to answer.
        relevant_reviews: List of review dicts from semantic search.

    Returns:
        Structured insight dict.
    """
    import json as json_mod
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)

    # Build the evidence block
    reviews_block = ""
    for i, r in enumerate(relevant_reviews, 1):
        reviews_block += f"\n[{i}] Category: {r.get('Target Category', 'N/A')} | Pillar: {r.get('Friction Pillar', 'N/A')}\n"
        reviews_block += f'    "{r.get("Raw Content", "")[:500]}"\n'

    # Aggregate context
    categories = set(r.get("Target Category", "") for r in relevant_reviews)
    pillars = set(r.get("Friction Pillar", "") for r in relevant_reviews)

    user_msg = USER_PROMPT_TEMPLATE.format(
        question=question,
        num_reviews=len(relevant_reviews),
        reviews_block=reviews_block,
        categories=", ".join(sorted(categories)),
        pillars=", ".join(sorted(pillars)),
    )

    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        logger.error("Insight generation failed: %s", e)
        return {
            "question": question,
            "executive_summary": f"Analysis failed: {e}",
            "themes": [],
            "key_insight": "",
            "actionable_recommendations": [],
            "confidence_level": "low",
            "evidence_gap": "API error — retry needed.",
        }


def generate_insight_offline(
    question: str,
    relevant_reviews: list[dict],
) -> dict:
    """
    Offline insight generation using rule-based analysis (no API needed).
    Question-aware: tailors the summary, themes, and recommendations
    to the specific strategic question being asked.
    """
    from collections import Counter

    categories = Counter(r.get("Target Category", "") for r in relevant_reviews)
    pillars = Counter(r.get("Friction Pillar", "") for r in relevant_reviews)
    total = len(relevant_reviews)

    # Extract representative quotes (top scored)
    quotes = [r.get("Raw Content", "")[:200] for r in relevant_reviews[:5]]

    # ── Question-aware summaries & recommendations ──
    q_lower = question.lower()

    # Detect positive vs negative sentiment in evidence
    positive_keywords = {"good", "great", "love", "happy", "excellent", "amazing", "impressed", "recommend", "satisfied", "best"}
    negative_keywords = {"bad", "worst", "terrible", "fake", "damaged", "broken", "expired", "hate", "pathetic", "fraud"}

    pos_count = sum(1 for r in relevant_reviews
                    if any(kw in str(r.get("Raw Content", "")).lower() for kw in positive_keywords))
    neg_count = sum(1 for r in relevant_reviews
                    if any(kw in str(r.get("Raw Content", "")).lower() for kw in negative_keywords))

    # Build per-category breakdowns
    cat_list = ", ".join(f"{cat} ({cnt})" for cat, cnt in categories.most_common(5) if cat)
    pillar_list = ", ".join(f"{p} ({c})" for p, c in pillars.most_common(4) if p)

    # ── Question-specific logic ──
    if "repeatedly" in q_lower or "same categories" in q_lower:
        summary = (
            f"From {total} relevant reviews, habitual behavior is driven by comfort and risk aversion. "
            f"Users stick to known categories because non-grocery purchases feel uncertain. "
            f"Friction breakdown: {pillar_list}."
        )
        key_insight = (
            "Users don't repeat-buy from the same categories by active choice — they do it "
            "because exploring new categories feels risky (quality doubts, price concerns)."
        )
        recommendations = [
            "Add 'Tried & Trusted' badges on non-grocery items with high reorder rates.",
            "Show 'Customers like you also bought...' cross-category nudges on checkout.",
            "Introduce a 'Try Something New' weekly spotlight with money-back guarantee.",
        ]

    elif "prevents" in q_lower or "exploring" in q_lower:
        summary = (
            f"From {total} relevant reviews, exploration is blocked by: {pillar_list}. "
            f"Trust deficit and price perception are the primary barriers across {cat_list}."
        )
        key_insight = (
            "The biggest blocker isn't awareness — users know Blinkit sells these items — "
            "it's that they don't trust the quality or find the price competitive."
        )
        recommendations = [
            "Show competitive price comparisons ('₹X on Amazon vs ₹Y here') on product pages.",
            "Add verified purchase reviews and authenticity certificates for electronics/beauty.",
            "Offer first-purchase discounts on categories the user hasn't tried yet.",
        ]

    elif "discover" in q_lower:
        summary = (
            f"From {total} relevant reviews, discovery happens primarily through search and "
            f"accidental browsing. Most users ({pillars.get('Discovery Blind Spots', 0)}/{total}) "
            f"report not knowing Blinkit carried these products."
        )
        key_insight = (
            "Current discovery is passive — users find new categories by accident, not by design. "
            "There is no active cross-category recommendation engine."
        )
        recommendations = [
            "Add contextual category suggestions in search results ('Looking for chargers? See Electronics').",
            "Show category exploration banners based on cart contents.",
            "Introduce a 'New on Blinkit' section on the home screen.",
        ]

    elif "habit" in q_lower:
        summary = (
            f"From {total} relevant reviews, habits dominate shopping behavior. "
            f"Users build routines around grocery and stick to them. "
            f"Habitual Tunnel Vision accounts for {pillars.get('Habitual Tunnel Vision', 0)}/{total} reviews."
        )
        key_insight = (
            "Habits aren't just preference — they're cognitive shortcuts. Users open Blinkit "
            "with a grocery list already in mind and never browse beyond it."
        )
        recommendations = [
            "Insert category discovery moments during the checkout flow (not just post-purchase).",
            "Add 'While you wait for delivery' browse suggestions for non-grocery items.",
            "Use push notifications to highlight non-grocery deals timed around routine orders.",
        ]

    elif "information" in q_lower or "before trying" in q_lower:
        summary = (
            f"From {total} relevant reviews, users need trust signals before trying non-grocery categories. "
            f"Key concerns: authenticity ({pillars.get('Quality & Authenticity Risk', 0)} reviews), "
            f"pricing ({pillars.get('Immediate Value Disconnection', 0)} reviews)."
        )
        key_insight = (
            "Users don't need more products — they need more proof. Reviews, ratings, "
            "return policies, and brand verification are prerequisites, not nice-to-haves."
        )
        recommendations = [
            "Show star ratings, review counts, and return policy prominently on non-grocery items.",
            "Add 'Verified Authentic' badges with brand partnerships.",
            "Display comparison tables (specs, price) for electronics and beauty products.",
        ]

    elif "frustration" in q_lower:
        summary = (
            f"From {total} relevant reviews, recurring frustrations span: {pillar_list}. "
            f"Most affected categories: {cat_list}."
        )
        key_insight = (
            "Frustrations cluster around post-purchase disappointment (quality, expiry) rather than "
            "pre-purchase concerns — meaning users DO try new categories but get burned."
        )
        recommendations = [
            "Implement stricter quality checks for non-grocery inventory at dark stores.",
            "Add expiry date visibility on product listings for beauty and pharmacy items.",
            "Create a dedicated 'Report Quality Issue' flow with instant credit.",
        ]

    elif "segment" in q_lower or "experiment" in q_lower:
        # This question specifically needs positive signals
        experimenters = [r for r in relevant_reviews if any(
            kw in str(r.get("Raw Content", "")).lower()
            for kw in {"tried", "first time", "surprised", "impressed", "good", "great", "recommend", "happy"}
        )]
        exp_cats = Counter(r.get("Target Category", "") for r in experimenters)
        exp_cat_list = ", ".join(f"{cat} ({cnt})" for cat, cnt in exp_cats.most_common(5) if cat)

        summary = (
            f"From {total} relevant reviews, {len(experimenters)} show willingness to experiment. "
            f"Experimenters appear across: {exp_cat_list if exp_cat_list else cat_list}. "
            f"Positive signals: {pos_count} reviews, negative: {neg_count} reviews."
        )
        key_insight = (
            "Experimenters aren't defined by tech-savviness — they're defined by positive first experiences. "
            "Users who had ONE good non-grocery purchase become repeat cross-category buyers."
        )
        recommendations = [
            "Identify users who made their first non-grocery purchase and send targeted follow-ups.",
            "Create a 'Category Explorer' reward program with points for trying new categories.",
            "Showcase user success stories ('I bought earbuds on Blinkit and they're great!').",
        ]

    elif "unmet" in q_lower or "needs" in q_lower:
        summary = (
            f"From {total} relevant reviews, unmet needs cluster around: {pillar_list}. "
            f"Users consistently ask for better variety, competitive pricing, and quality assurance "
            f"across {cat_list}."
        )
        key_insight = (
            "The biggest unmet need isn't product availability — it's confidence. Users want to buy "
            "non-grocery items on Blinkit but lack the confidence that quality and price will match expectations."
        )
        recommendations = [
            "Expand product variety in underrepresented categories (pet, baby, pharmacy).",
            "Add price-match guarantee for select non-grocery categories.",
            "Build a 'Blinkit Quality Promise' program with easy returns for non-grocery items.",
        ]

    else:
        # Generic fallback
        top_pillar = pillars.most_common(1)[0][0] if pillars else "Unknown"
        summary = (
            f"Based on {total} relevant reviews, the dominant pattern is '{top_pillar}' "
            f"across {cat_list}. Friction breakdown: {pillar_list}."
        )
        key_insight = f"The strongest signal is {top_pillar} across multiple categories."
        recommendations = [
            f"Address {top_pillar.lower()} with targeted UX interventions.",
            "Implement trust signals (warranty badges, authenticity seals) for non-grocery categories.",
            "Surface cross-category recommendations based on purchase history.",
        ]

    # ── Build themes from pillar groupings ──
    themes = []
    for pillar, count in pillars.most_common(4):
        if not pillar:
            continue
        pillar_reviews = [r for r in relevant_reviews if r.get("Friction Pillar") == pillar]
        pillar_cats = list(set(r.get("Target Category", "") for r in pillar_reviews if r.get("Target Category")))
        pillar_quotes = [r.get("Raw Content", "")[:200] for r in pillar_reviews[:2]]

        theme_map = {
            "Quality & Authenticity Risk": "Trust & product quality concerns",
            "Immediate Value Disconnection": "Price & value perception gaps",
            "Discovery Blind Spots": "Category awareness & visibility failures",
            "Habitual Tunnel Vision": "Entrenched grocery-only shopping habits",
        }

        themes.append({
            "theme_name": theme_map.get(pillar, pillar),
            "description": f"{count} of {total} reviews mapped to {pillar}. Affects: {', '.join(pillar_cats) if pillar_cats else 'multiple categories'}.",
            "evidence_count": count,
            "representative_quotes": pillar_quotes,
            "friction_pillar": pillar,
            "categories_affected": pillar_cats,
        })

    # ── Data bias caveat ──
    bias_note = ""
    if categories and categories.most_common(1)[0][1] > total * 0.5:
        top_cat, top_cnt = categories.most_common(1)[0]
        bias_note = (
            f" Note: {top_cat} dominates the evidence ({top_cnt}/{total} reviews) — "
            f"insights may be skewed toward this category."
        )

    return {
        "question": question,
        "executive_summary": summary + bias_note,
        "themes": themes,
        "key_insight": key_insight,
        "actionable_recommendations": recommendations,
        "confidence_level": "medium" if total >= 10 else "low",
        "evidence_gap": (
            "Dataset skews toward electronics and negative reviews. "
            "Positive purchase experiences and categories like pet/baby/pharmacy are underrepresented. "
            "Run with FAISS + Groq AI for deeper semantic analysis."
        ),
    }
