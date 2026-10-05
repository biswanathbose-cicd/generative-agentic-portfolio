"""PII redaction used by the pipeline (independent of the live assistant's guardrail so that
logs written by *older* or *misconfigured* assistant versions are still cleaned)."""

from __future__ import annotations

import re

PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
}


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for label, pat in PATTERNS.items():
        text, n = pat.subn(f"[{label.upper()}_REDACTED]", text)
        if n:
            counts[label] = n
    return text, counts


def contains_pii(text: str) -> bool:
    return any(p.search(text) for p in PATTERNS.values())
