"""Product retrieval for the RAG product-search agent.

``hybrid=True`` blends word-level TF-IDF (precise on exact terms) with character n-gram
TF-IDF (robust to typos). ``hybrid=False`` is the word-only baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _doc(p: dict) -> str:
    return f"{p['name']}. {p['category']}. {p['description']}"


class ProductIndex:
    def __init__(self, catalog: list[dict], hybrid: bool = True, char_weight: float = 0.5):
        self.catalog = catalog
        self.hybrid = hybrid
        self.char_weight = char_weight if hybrid else 0.0
        docs = [_doc(p) for p in catalog]
        self._word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._word_m = self._word.fit_transform(docs)
        if hybrid:
            self._char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self._char_m = self._char.fit_transform(docs)

    def scores(self, query: str) -> np.ndarray:
        s = (self._word_m @ self._word.transform([query]).T).toarray().ravel()
        if self.hybrid:
            c = (self._char_m @ self._char.transform([query]).T).toarray().ravel()
            s = (1 - self.char_weight) * s + self.char_weight * c
        return s

    def search(self, query: str, k: int = 3, max_price: float | None = None,
               category: str | None = None, in_stock_only: bool = True) -> list[dict]:
        s = self.scores(query)
        order = np.argsort(-s)
        out: list[dict] = []
        for idx in order:
            if s[idx] <= 0:
                break
            p = self.catalog[idx]
            if max_price is not None and p["price"] > max_price:
                continue
            if category and p["category"] != category:
                continue
            if in_stock_only and not p["in_stock"]:
                continue
            out.append({**p, "score": round(float(s[idx]), 4)})
            if len(out) == k:
                break
        return out
