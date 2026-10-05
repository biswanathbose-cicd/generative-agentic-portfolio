"""Input and output guardrails.

Input:  PII redaction (before anything is logged or sent to a model) + prompt-injection screen.
Output: grounding checks - prices, order ids and action claims in a reply must be backed by
        tool results from the same turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PII_PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"),
}

INJECTION_PATTERNS = [
    re.compile(r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)", re.I),
    re.compile(r"(reveal|show|print|repeat).{0,30}(system prompt|your instructions)", re.I),
    re.compile(r"you are now\b", re.I),
    re.compile(r"developer mode|jailbreak|do anything now", re.I),
    re.compile(r"disregard (your|the) (rules|policy|guidelines)", re.I),
]

ORDER_ID_RE = re.compile(r"\bWM-\d{5}\b", re.I)
PRICE_RE = re.compile(r"\$(\d+(?:\.\d{1,2})?)")


@dataclass
class InputCheck:
    redacted_text: str
    pii_found: list[str] = field(default_factory=list)
    injection: bool = False


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for label, pat in PII_PATTERNS.items():
        if pat.search(text):
            found.append(label)
            text = pat.sub(f"[{label.upper()}_REDACTED]", text)
    return text, found


def check_input(text: str) -> InputCheck:
    redacted, found = redact_pii(text)
    injection = any(p.search(text) for p in INJECTION_PATTERNS)
    return InputCheck(redacted_text=redacted, pii_found=found, injection=injection)


@dataclass
class OutputCheck:
    ok: bool
    violations: list[str] = field(default_factory=list)


def _collect(obj, out: list) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, out)
    else:
        out.append(obj)


def check_output(response: str, trace: list[dict], user_text: str = "") -> OutputCheck:
    """Reject replies that state facts the tools did not return."""
    violations: list[str] = []
    values: list = []
    _collect([t["result"] for t in trace], values)
    _collect([t["args"] for t in trace], values)

    allowed_prices = {round(float(v), 2) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for m in PRICE_RE.findall(response):
        if round(float(m), 2) not in allowed_prices:
            violations.append(f"ungrounded_price:${m}")

    allowed_ids = {str(v).upper() for v in values if isinstance(v, str)}
    allowed_ids |= {m.upper() for m in ORDER_ID_RE.findall(user_text)}
    for oid in ORDER_ID_RE.findall(response):
        if oid.upper() not in allowed_ids:
            violations.append(f"ungrounded_order_id:{oid}")

    called = {t["tool"] for t in trace}
    low = response.lower()
    if ("return has been" in low or "return label" in low) and "initiate_return" not in called:
        violations.append("claims_return_without_tool")
    return OutputCheck(ok=not violations, violations=violations)
