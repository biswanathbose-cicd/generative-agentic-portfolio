"""Tool layer: plain functions + JSON-schema specs (so a real LLM can call them).

Every call is recorded on the ``ToolRegistry.trace`` so the evaluation framework can
score tool-call accuracy. A hard step cap protects against runaway loops.
"""

from __future__ import annotations

from datetime import date, timedelta

from .data import NON_RETURNABLE_CATEGORIES, REFERENCE_TODAY, RETURN_WINDOW_DAYS
from .retrieval import ProductIndex


class StepLimitExceeded(RuntimeError):
    pass


class ToolRegistry:
    def __init__(self, catalog: list[dict], orders: list[dict], index: ProductIndex,
                 max_steps: int = 6, today: date = REFERENCE_TODAY):
        self.catalog = {p["sku"]: p for p in catalog}
        self.orders = {o["order_id"]: o for o in orders}
        self.index = index
        self.max_steps = max_steps
        self.today = today
        self.trace: list[dict] = []
        self.returns_initiated: list[str] = []
        self.handoff_tickets: list[str] = []

    # -- infrastructure -------------------------------------------------
    def reset_trace(self) -> None:
        self.trace = []

    def call(self, name: str, **kwargs) -> dict:
        if len(self.trace) >= self.max_steps:
            raise StepLimitExceeded(f"more than {self.max_steps} tool calls in one turn")
        fn = getattr(self, f"_tool_{name}", None)
        if fn is None:
            result = {"error": f"unknown tool {name}"}
        else:
            try:
                result = fn(**kwargs)
            except TypeError as e:
                result = {"error": f"bad arguments: {e}"}
        self.trace.append({"tool": name, "args": kwargs, "result": result})
        return result

    # -- tools ------------------------------------------------------------
    def _tool_get_order_status(self, order_id: str) -> dict:
        o = self.orders.get(order_id)
        if not o:
            return {"error": "order_not_found", "order_id": order_id}
        return {k: o[k] for k in ("order_id", "status", "order_date", "delivered_date", "eta",
                                   "carrier", "tracking_number")}

    def _tool_check_return_eligibility(self, order_id: str) -> dict:
        o = self.orders.get(order_id)
        if not o:
            return {"error": "order_not_found", "order_id": order_id}
        if o["status"] == "returned":
            return {"order_id": order_id, "eligible": False, "reason": "already_returned"}
        if o["status"] != "delivered":
            return {"order_id": order_id, "eligible": False, "reason": f"order_is_{o['status']}"}
        cats = {self.catalog[i["sku"]]["category"] for i in o["items"]}
        if cats & NON_RETURNABLE_CATEGORIES:
            return {"order_id": order_id, "eligible": False, "reason": "contains_non_returnable_items"}
        deadline = date.fromisoformat(o["delivered_date"]) + timedelta(days=RETURN_WINDOW_DAYS)
        if self.today > deadline:
            return {"order_id": order_id, "eligible": False, "reason": "return_window_expired",
                    "deadline": deadline.isoformat()}
        return {"order_id": order_id, "eligible": True, "deadline": deadline.isoformat()}

    def _tool_initiate_return(self, order_id: str, reason: str = "unspecified") -> dict:
        # Deliberately does NOT re-check policy: like many real backends it trusts its caller.
        # The agent is responsible for calling check_return_eligibility first.
        if order_id not in self.orders:
            return {"error": "order_not_found", "order_id": order_id}
        self.returns_initiated.append(order_id)
        return {"order_id": order_id, "return_initiated": True, "reason": reason,
                "label": f"RET-{order_id[3:]}"}

    def _tool_search_products(self, query: str, k: int = 3, max_price: float | None = None) -> dict:
        hits = self.index.search(query, k=k, max_price=max_price)
        return {"query": query, "max_price": max_price,
                "results": [{"sku": h["sku"], "name": h["name"], "category": h["category"],
                             "price": h["price"]} for h in hits]}

    def _tool_create_handoff_ticket(self, summary: str) -> dict:
        ticket = f"HO-{len(self.handoff_tickets) + 1:05d}"
        self.handoff_tickets.append(ticket)
        return {"ticket": ticket, "summary": summary[:200]}


TOOL_SPECS = [
    {"name": "get_order_status", "description": "Look up the status of an order.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}},
                      "required": ["order_id"]}},
    {"name": "check_return_eligibility", "description": "Check whether an order can be returned under policy.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"}},
                      "required": ["order_id"]}},
    {"name": "initiate_return", "description": "Create a return label. Only call after eligibility is confirmed.",
     "input_schema": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                       "reason": {"type": "string"}},
                      "required": ["order_id"]}},
    {"name": "search_products", "description": "Search the product catalog.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"},
                                                       "k": {"type": "integer"},
                                                       "max_price": {"type": "number"}},
                      "required": ["query"]}},
    {"name": "create_handoff_ticket", "description": "Escalate the conversation to a human agent.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}},
                      "required": ["summary"]}},
]
