"""Retrieval metrics (recall@k, MRR) for the product-search agent: word TF-IDF vs hybrid."""

from __future__ import annotations

import random

import numpy as np

from . import paths  # noqa: F401
from support_agent.data import load_catalog
from support_agent.retrieval import ProductIndex

from .dataset import _noun_phrase, _typo


def _queries(catalog: list[dict], n: int, typo: bool, seed: int):
    rng = random.Random(seed)
    groups: dict[str, set[str]] = {}
    for p in catalog:
        groups.setdefault(_noun_phrase(p), set()).add(p["sku"])
    out = []
    for p in rng.sample(catalog, n):
        noun = _noun_phrase(p)
        adj = p["name"].split()[1]
        words = f"{adj} {noun}".split()
        if typo:
            words = [_typo(w, rng) for w in words]
        out.append((" ".join(words), groups[noun]))
    return out


def evaluate(index: ProductIndex, queries, k: int = 5) -> dict:
    recalls, rr = [], []
    for q, relevant in queries:
        hits = [h["sku"] for h in index.search(q, k=k, in_stock_only=False)]
        recalls.append(len(relevant & set(hits)) / len(relevant))
        rank = next((i + 1 for i, s in enumerate(hits) if s in relevant), None)
        rr.append(1 / rank if rank else 0.0)
    return {f"recall@{k}": float(np.mean(recalls)), "mrr": float(np.mean(rr))}


def run(n: int = 100, seed: int = 5) -> dict:
    catalog = load_catalog()
    word, hybrid = ProductIndex(catalog, hybrid=False), ProductIndex(catalog, hybrid=True)
    res = {}
    for label, typo in (("clean_queries", False), ("typo_queries", True)):
        qs = _queries(catalog, n, typo, seed)
        res[label] = {"word_tfidf": evaluate(word, qs), "hybrid": evaluate(hybrid, qs)}
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
