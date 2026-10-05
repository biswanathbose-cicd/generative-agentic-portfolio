"""LLM backends behind one tiny interface: ``complete(task, system, user) -> str``.

* ``RuleBasedLLM``  - deterministic offline stand-in. It makes this repo runnable (and its
  evaluations reproducible) with no API key. It is NOT a language model; numbers measured
  with it describe the *pipeline*, not LLM quality.
* ``AnthropicLLM`` / ``OpenAICompatLLM`` - thin stdlib-HTTP clients selected with env vars.
  They were not exercised in the environment this repo was built in (no network/API key);
  see README for how to smoke-test them.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import urllib.request

from .config import AssistantConfig

KEYWORDS = {
    "human_handoff": {"human", "agent", "representative", "manager", "person", "supervisor"},
    "return_request": {"return", "returning", "refund", "refunded", "exchange"},
    "order_status": {"order", "track", "tracking", "shipped", "shipping", "delivery",
                     "delivered", "package", "parcel"},
    "product_search": {"buy", "looking", "find", "need", "recommend", "show", "cheap",
                       "under", "best", "want", "shopping"},
}
FRUSTRATION = {"ridiculous", "unacceptable", "worst", "terrible", "useless", "furious", "angry",
               "awful", "scam", "nobody", "pathetic", "disgusted", "fed", "sick"}
_ALL_KEYWORDS = sorted({w for s in KEYWORDS.values() for w in s} | FRUSTRATION)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'-]*", text.lower())


def within_one_edit(a: str, b: str) -> bool:
    """True if ``a`` and ``b`` differ by at most one insert/delete/substitute/adjacent-swap."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        return (len(diffs) == 2 and diffs[1] == diffs[0] + 1
                and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]])
    short, long_ = (a, b) if la < lb else (b, a)
    i = j = 0
    skipped = False
    while i < len(short) and j < len(long_):
        if short[i] == long_[j]:
            i += 1
            j += 1
        elif skipped:
            return False
        else:
            skipped = True
            j += 1
    return True


def product_vocabulary(catalog: list[dict]) -> set[str]:
    vocab: set[str] = set()
    for p in catalog:
        noun = p["name"].lower()
        vocab.update(_tokens(p["category"]))
        # last words of the name are the noun phrase (brand/adjective come first)
        vocab.update(_tokens(noun)[-2:])
    vocab -= {"set", "ultra", "classic", "premium", "compact", "budget", "organic", "durable",
              "everyday", "lightweight", "waterproof", "rechargeable", "family-size"}
    return vocab


class LLM:
    def complete(self, task: str, system: str, user: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class RuleBasedLLM(LLM):
    def __init__(self, config: AssistantConfig, vocab: set[str]):
        self.cfg = config
        self.vocab = vocab

    # ------------------------------------------------------------------ route
    def _correct(self, toks: list[str], corroborated: bool) -> list[str]:
        candidates = _ALL_KEYWORDS + sorted(self.vocab)
        risky = KEYWORDS["order_status"] | KEYWORDS["return_request"]
        fixed = []
        for t in toks:
            if len(t) > 3 and t not in _ALL_KEYWORDS and t not in self.vocab:
                if self.cfg.strict_fuzzy:
                    near = [c for c in candidates if within_one_edit(t, c)]
                    m = difflib.get_close_matches(t, near, n=1, cutoff=0.0)
                    if m and m[0] in risky and not corroborated:
                        m = []  # a lone "trick"->"track" is not evidence of an order question
                else:
                    m = difflib.get_close_matches(t, candidates, n=1, cutoff=0.8)
                if m:
                    t = m[0]
            fixed.append(t)
        return fixed

    def _route(self, message: str, has_active_order: bool = False, has_order_id: bool = False) -> str:
        toks = _tokens(message)
        if self.cfg.fuzzy_routing:
            toks = self._correct(toks, corroborated=has_order_id or has_active_order)
        tokset = set(toks)
        if self.cfg.detect_frustration and tokset & FRUSTRATION:
            return "human_handoff"
        if tokset & KEYWORDS["human_handoff"]:
            return "human_handoff"
        if tokset & KEYWORDS["return_request"]:
            return "return_request"
        if tokset & KEYWORDS["order_status"]:
            return "order_status"
        if (self.cfg.use_memory and has_active_order and tokset & {"it", "this", "that", "them"}
                and tokset & {"where", "when", "status", "arrive", "arriving", "arrived", "eta", "yet"}):
            return "order_status"  # pronoun follow-up about the order already under discussion
        has_vocab = bool(tokset & self.vocab)
        has_kw = bool(tokset & KEYWORDS["product_search"])
        if self.cfg.fuzzy_routing:  # v2 requires an actual product word
            return "product_search" if has_vocab else "out_of_scope"
        return "product_search" if has_kw else "out_of_scope"  # v1: keyword is enough

    # ----------------------------------------------------------------- answer
    @staticmethod
    def _answer(payload: dict) -> str:
        intent, f = payload["intent"], payload["facts"]
        if intent == "clarify_order":
            return "Could you share your order number? It starts with WM- followed by five digits."
        if intent == "order_status":
            o = f["order"]
            if o.get("error"):
                return f"I couldn't find order {o['order_id']}. Could you double-check the number?"
            st, oid = o["status"], o["order_id"]
            if st == "delivered":
                return f"Order {oid} was delivered on {o['delivered_date']} via {o['carrier']}."
            if st == "shipped":
                return (f"Order {oid} has shipped via {o['carrier']} (tracking {o['tracking_number']}) "
                        f"and is expected on {o['eta']}.")
            if st == "processing":
                return f"Order {oid} is being prepared and has not shipped yet."
            if st == "cancelled":
                return f"Order {oid} was cancelled."
            return f"Order {oid} was delivered and has since been returned."
        if intent == "return_request":
            if f.get("error"):
                return f"I couldn't find order {f['order_id']}. Could you double-check the number?"
            if f.get("initiated"):
                tail = f" Please send it back by {f['deadline']}." if f.get("deadline") else ""
                return (f"Your return has been initiated for order {f['order_id']}. "
                        f"Your return label is {f['label']}.{tail}")
            reasons = {
                "already_returned": "it has already been returned",
                "contains_non_returnable_items": "it contains items that can't be returned (e.g. groceries)",
                "return_window_expired": f"the 30-day return window ended on {f.get('deadline')}",
            }
            why = reasons.get(f["reason"], "it is not eligible under our return policy")
            if f["reason"].startswith("order_is_"):
                why = f"the order is still {f['reason'][9:]}, so it can't be returned yet"
            return f"I can't start a return for order {f['order_id']} because {why}."
        if intent == "product_search":
            res = f["results"]
            if not res:
                return "I couldn't find a matching in-stock product. Want to try different words or a higher budget?"
            lines = "; ".join(f"{r['name']} (${r['price']:.2f})" for r in res)
            return f"Here are some options: {lines}."
        if intent == "human_handoff":
            return (f"I'm sorry for the trouble. I've passed this to a human agent "
                    f"(ticket {f['ticket']}); they'll follow up shortly.")
        if intent == "blocked":
            return "I can't help with that request. I can help with order status, returns, or finding products."
        return "I can only help with orders, returns, and product search at the moment."

    def complete(self, task: str, system: str, user: str) -> str:
        payload = json.loads(user)
        if task == "route":
            return json.dumps({"intent": self._route(payload["message"], payload.get("has_active_order", False),
                                                      payload.get("has_order_id", False))})
        return self._answer(payload)


class _HTTPLLM(LLM):
    def _post(self, url: str, headers: dict, body: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https endpoints
            return json.loads(resp.read())


class AnthropicLLM(_HTTPLLM):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

    def complete(self, task: str, system: str, user: str) -> str:
        data = self._post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {"model": self.model, "max_tokens": 300, "system": system,
             "messages": [{"role": "user", "content": user}]},
        )
        return "".join(b.get("text", "") for b in data["content"])


class OpenAICompatLLM(_HTTPLLM):
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def complete(self, task: str, system: str, user: str) -> str:
        data = self._post(
            f"{self.base_url}/chat/completions",
            {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"},
            {"model": self.model, "max_tokens": 300,
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        )
        return data["choices"][0]["message"]["content"]


def get_llm(config: AssistantConfig, catalog: list[dict]) -> LLM:
    """Pick a backend from ``LLM_PROVIDER`` (rule | anthropic | openai). Default: rule."""
    provider = os.environ.get("LLM_PROVIDER", "rule").lower()
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "openai":
        return OpenAICompatLLM()
    return RuleBasedLLM(config, product_vocabulary(catalog))
