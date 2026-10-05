"""Orchestrator: guardrails -> router -> specialist agent (tools) -> answer -> output guardrails.

    user text
       |
       v
  input guardrails  --(injection)--> refuse
       |  (PII redacted)
       v
     router (LLM) --> order_agent | returns_agent | product_agent | handoff_agent | scope refusal
                          |             |               |                |
                          +------- ToolRegistry (traced, step-capped) ----+
                                          |
                                          v
                               answer writer (LLM, facts only)
                                          |
                                          v
                               output guardrails (grounding)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from .config import AssistantConfig, V2_IMPROVED
from .data import load_catalog, load_orders
from .guardrails import ORDER_ID_RE, check_input, check_output
from .llm import LLM, get_llm
from .prompts import ANSWER_SYSTEM, ROUTER_SYSTEM
from .retrieval import ProductIndex
from .tools import StepLimitExceeded, ToolRegistry

VALID_INTENTS = {"order_status", "return_request", "product_search", "human_handoff", "out_of_scope"}
PRICE_CAP_RE = re.compile(r"(?:under|below|less than|within|max(?:imum)?|budget of)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
SAFE_FALLBACK = "Sorry, I couldn't verify that information, so I'm connecting you with a human agent."


@dataclass
class Session:
    session_id: str = "s0"
    last_order_id: str | None = None
    turns: int = 0


@dataclass
class Result:
    response: str
    intent: str
    outcome: str  # answered | clarify | handoff | refused | refused_scope | error
    trace: list[dict] = field(default_factory=list)
    logged_input: str = ""
    guardrail_flags: list[str] = field(default_factory=list)
    llm_calls: int = 0
    prompt_chars: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "response": self.response, "intent": self.intent, "outcome": self.outcome,
            "tools": [t["tool"] for t in self.trace], "guardrail_flags": self.guardrail_flags,
            "llm_calls": self.llm_calls, "latency_ms": round(self.latency_ms, 3),
        }


class SupportAssistant:
    def __init__(self, config: AssistantConfig, llm: LLM, tools: ToolRegistry):
        self.cfg = config
        self.llm = llm
        self.tools = tools

    # -- LLM helpers (count calls so cost can be tracked) ------------------
    def _llm(self, task: str, system: str, payload: dict, acc: dict) -> str:
        user = json.dumps(payload)
        acc["calls"] += 1
        acc["chars"] += len(system) + len(user)
        return self.llm.complete(task, system, user)

    def _route(self, text: str, acc: dict, has_active_order: bool = False) -> str:
        payload = {"message": text, "has_active_order": has_active_order,
                   "has_order_id": bool(ORDER_ID_RE.search(text))}
        raw = self._llm("route", ROUTER_SYSTEM, payload, acc)
        try:
            intent = json.loads(raw)["intent"]
        except (ValueError, KeyError, TypeError):
            return "out_of_scope"
        return intent if intent in VALID_INTENTS else "out_of_scope"

    def _answer(self, intent: str, facts: dict, acc: dict) -> str:
        return self._llm("answer", ANSWER_SYSTEM, {"intent": intent, "facts": facts}, acc)

    # -- specialist agents ----------------------------------------------------
    def _order_agent(self, order_id: str | None, acc: dict) -> tuple[str, str]:
        if not order_id:
            return self._answer("clarify_order", {}, acc), "clarify"
        order = self.tools.call("get_order_status", order_id=order_id)
        return self._answer("order_status", {"order": order}, acc), "answered"

    @staticmethod
    def _return_reason(text: str) -> str:
        low = text.lower()
        for key, label in (("damaged", "damaged"), ("broken", "damaged"), ("defective", "damaged"),
                           ("wrong", "wrong_item"), ("size", "fit"), (" fit", "fit")):
            if key in low:
                return label
        return "unspecified"

    def _returns_agent(self, order_id: str | None, text: str, acc: dict) -> tuple[str, str]:
        if not order_id:
            return self._answer("clarify_order", {}, acc), "clarify"
        reason = self._return_reason(text)
        if self.cfg.check_return_eligibility:
            elig = self.tools.call("check_return_eligibility", order_id=order_id)
            if elig.get("error"):
                return self._answer("return_request", {"error": True, "order_id": order_id}, acc), "answered"
            if not elig["eligible"]:
                facts = {"order_id": order_id, "initiated": False, "reason": elig["reason"],
                         "deadline": elig.get("deadline")}
                return self._answer("return_request", facts, acc), "answered"
            deadline = elig.get("deadline")
        else:
            deadline = None
        res = self.tools.call("initiate_return", order_id=order_id, reason=reason)
        if res.get("error"):
            return self._answer("return_request", {"error": True, "order_id": order_id}, acc), "answered"
        facts = {"order_id": order_id, "initiated": True, "label": res["label"], "deadline": deadline}
        return self._answer("return_request", facts, acc), "answered"

    def _product_agent(self, text: str, acc: dict) -> tuple[str, str]:
        m = PRICE_CAP_RE.search(text)
        max_price = float(m.group(1)) if m else None
        res = self.tools.call("search_products", query=text, k=3, max_price=max_price)
        return self._answer("product_search", {"results": res["results"]}, acc), "answered"

    def _handoff_agent(self, text: str, acc: dict) -> tuple[str, str]:
        ticket = self.tools.call("create_handoff_ticket", summary=text)
        return self._answer("human_handoff", {"ticket": ticket["ticket"]}, acc), "handoff"

    # -- main entry point --------------------------------------------------------
    def handle(self, session: Session, user_text: str) -> Result:
        t0 = time.perf_counter()
        self.tools.reset_trace()
        acc = {"calls": 0, "chars": 0}
        flags: list[str] = []
        text = user_text
        session.turns += 1

        if self.cfg.input_guardrails:
            chk = check_input(user_text)
            text = chk.redacted_text
            flags += [f"pii_redacted:{p}" for p in chk.pii_found]
            if chk.injection:
                flags.append("prompt_injection")
                resp = self._answer("blocked", {}, acc)
                return self._finish(Result(resp, "blocked", "refused", [], text, flags), acc, t0)

        active = bool(self.cfg.use_memory and session.last_order_id)
        intent = self._route(text, acc, has_active_order=active)
        found = ORDER_ID_RE.search(text)
        order_id = found.group(0).upper() if found else (session.last_order_id if self.cfg.use_memory else None)
        if found and self.cfg.use_memory:
            session.last_order_id = order_id

        try:
            if intent == "order_status":
                resp, outcome = self._order_agent(order_id, acc)
            elif intent == "return_request":
                resp, outcome = self._returns_agent(order_id, text, acc)
            elif intent == "product_search":
                resp, outcome = self._product_agent(text, acc)
            elif intent == "human_handoff":
                resp, outcome = self._handoff_agent(text, acc)
            else:
                resp, outcome = self._answer("out_of_scope", {}, acc), "refused_scope"
        except StepLimitExceeded:
            flags.append("step_limit")
            resp, outcome = SAFE_FALLBACK, "error"

        if self.cfg.output_guardrails and outcome in ("answered", "clarify"):
            oc = check_output(resp, self.tools.trace, text)
            if not oc.ok:
                flags += [f"output_blocked:{v}" for v in oc.violations]
                self.tools.call("create_handoff_ticket", summary=f"output blocked: {oc.violations}")
                resp, outcome = SAFE_FALLBACK, "handoff"

        return self._finish(Result(resp, intent, outcome, list(self.tools.trace), text, flags), acc, t0)

    def _finish(self, r: Result, acc: dict, t0: float) -> Result:
        r.llm_calls, r.prompt_chars = acc["calls"], acc["chars"]
        r.latency_ms = (time.perf_counter() - t0) * 1000
        if not r.trace:
            r.trace = list(self.tools.trace)
        return r


def build_assistant(config: AssistantConfig = V2_IMPROVED, llm: LLM | None = None,
                    catalog: list[dict] | None = None, orders: list[dict] | None = None) -> SupportAssistant:
    catalog = catalog if catalog is not None else load_catalog()
    orders = orders if orders is not None else load_orders()
    index = ProductIndex(catalog, hybrid=config.hybrid_retrieval)
    tools = ToolRegistry(catalog, orders, index, max_steps=config.max_steps)
    return SupportAssistant(config, llm or get_llm(config, catalog), tools)
