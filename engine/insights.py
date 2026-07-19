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
    Less sophisticated but works without Groq API key.
    """
    from collections import Counter

    categories = Counter(r.get("Target Category", "") for r in relevant_reviews)
    pillars = Counter(r.get("Friction Pillar", "") for r in relevant_reviews)

    # Extract representative quotes (top scored)
    quotes = [r.get("Raw Content", "")[:200] for r in relevant_reviews[:5]]

    # Build themes from pillar groupings
    themes = []
    for pillar, count in pillars.most_common(4):
        pillar_reviews = [r for r in relevant_reviews if r.get("Friction Pillar") == pillar]
        pillar_cats = list(set(r.get("Target Category", "") for r in pillar_reviews))
        pillar_quotes = [r.get("Raw Content", "")[:200] for r in pillar_reviews[:2]]

        theme_map = {
            "Quality & Authenticity Risk": "Trust & product quality concerns dominate",
            "Immediate Value Disconnection": "Price & value perception gaps",
            "Discovery Blind Spots": "Category awareness & visibility failures",
            "Habitual Tunnel Vision": "Entrenched grocery-only shopping habits",
        }

        themes.append({
            "theme_name": theme_map.get(pillar, pillar),
            "description": f"{count} reviews mapped to {pillar}. Users express concerns across {', '.join(pillar_cats)}.",
            "evidence_count": count,
            "representative_quotes": pillar_quotes,
            "friction_pillar": pillar,
            "categories_affected": pillar_cats,
        })

    top_cat = categories.most_common(1)[0][0] if categories else "general"
    top_pillar = pillars.most_common(1)[0][0] if pillars else "Unknown"

    return {
        "question": question,
        "executive_summary": (
            f"Based on {len(relevant_reviews)} relevant reviews, the dominant pattern is "
            f"'{top_pillar}' affecting primarily '{top_cat}'. "
            f"Users express concerns that map across {len(themes)} distinct themes."
        ),
        "themes": themes,
        "key_insight": f"The strongest signal is {top_pillar} in the {top_cat} category.",
        "actionable_recommendations": [
            f"Address {top_pillar.lower()} with targeted UX interventions in {top_cat}.",
            "Implement trust signals (warranty badges, authenticity seals) for non-grocery categories.",
            "Surface cross-category recommendations based on purchase history.",
        ],
        "confidence_level": "medium" if len(relevant_reviews) >= 10 else "low",
        "evidence_gap": "Offline analysis lacks nuanced NLP — run with --classify ai for deeper insights.",
    }
