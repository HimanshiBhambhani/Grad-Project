"""
engine/pii_filter.py — Strip personally identifiable information from review text.

Removes emails, phone numbers, social media handles, IP addresses,
and Aadhaar-pattern numbers before displaying source quotes.
"""

import re

# Compiled patterns for performance
_PATTERNS = [
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    # Phone numbers — Indian (+91), international, and generic
    (re.compile(r"(?:\+91[\s-]?)?(?:\d[\s-]?){10,13}"), "[PHONE]"),
    # Reddit / social media usernames  (u/username, @handle)
    (re.compile(r"(?:u/|@)[A-Za-z0-9_]{2,30}"), "[USER]"),
    # IP addresses (IPv4)
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP]"),
    # Aadhaar-pattern (12 consecutive digits, optionally space/dash separated)
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[ID]"),
    # Order IDs / transaction references (common patterns like OD12345678)
    (re.compile(r"\b(?:OD|ORD|TXN|REF)[A-Z0-9]{6,20}\b", re.IGNORECASE), "[ORDER_ID]"),
]


def strip_pii(text: str) -> str:
    """Remove all PII patterns from text.

    Args:
        text: Raw review text.

    Returns:
        Sanitised text with PII replaced by placeholders.
    """
    if not isinstance(text, str):
        return str(text)

    result = text
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)

    return result


def strip_pii_from_reviews(reviews: list[dict], field: str = "Raw Content") -> list[dict]:
    """Strip PII from a list of review dicts.

    Args:
        reviews: List of review metadata dicts.
        field: The key containing review text.

    Returns:
        Same list with PII stripped from the specified field.
    """
    for r in reviews:
        if field in r:
            r[field] = strip_pii(str(r[field]))
    return reviews
