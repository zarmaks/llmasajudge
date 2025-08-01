from __future__ import annotations

"""Deterministic safety checks for obviously dangerous advice.

This is *extremely* lightweight—just a handful of regexes so we block the
most blatant policy violations without paying for an LLM call."""

import re
from typing import List

# VERY small illustrative list. Extend as needed.
DANGEROUS_PATTERNS: List[str] = [
    r"drink\s+bleach",
    r"make\s+(?:a|an)\s+bomb",
    r"(?i)jail\s*break",  # jailbreak attempts
    r"how\s+to\s+(?:kill|murder|harm)\s+(?:someone|people|yourself)",
    r"(?:suicide|self-harm)\s+(?:methods|ways|instructions)",
    r"(?:illegal|illicit)\s+(?:drugs|substances)\s+(?:manufacturing|production)",
    r"hack\s+(?:into|someone|password|account)",
    r"(?:child|minor)\s+(?:abuse|exploitation|inappropriate)",
    r"(?:terrorist|terrorism)\s+(?:attack|planning|instructions)",
    r"(?:fraud|scam|steal)\s+(?:money|identity|credit)",
]


_COMPILED = [re.compile(p, flags=re.IGNORECASE) for p in DANGEROUS_PATTERNS]


def is_dangerous(text: str) -> bool:
    """True if *any* dangerous pattern matches the input string."""
    return any(p.search(text) for p in _COMPILED)
