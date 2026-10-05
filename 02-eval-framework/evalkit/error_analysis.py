"""Error analysis: failure taxonomy + text clustering of failed conversations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def failure_table(df: pd.DataFrame) -> pd.DataFrame:
    fails = df[~df.success]
    return (fails.groupby(["category", "failure"]).size().rename("n").reset_index()
            .sort_values("n", ascending=False).reset_index(drop=True))


def cluster_failures(df: pd.DataFrame, k: int = 5, seed: int = 0, top_terms: int = 5) -> pd.DataFrame:
    """Cluster the *text* of failed final user turns; label each cluster by top terms and its
    dominant (category, failure) pair so a human can read the failure modes quickly."""
    fails = df[~df.success].reset_index(drop=True)
    if len(fails) < k:
        k = max(1, len(fails))
    if fails.empty:
        return pd.DataFrame(columns=["cluster", "n", "top_terms", "dominant_category", "dominant_failure", "example"])
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(fails.final_user_text)
    Xd = X.toarray()
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for c in range(k):
        members = fails[km.labels_ == c]
        centroid = km.cluster_centers_[c]
        idx = np.where(km.labels_ == c)[0]
        dist = ((Xd[idx] - km.cluster_centers_[c]) ** 2).sum(axis=1)
        nearest = fails.final_user_text.iloc[idx[int(np.argmin(dist))]]
        rows.append({
            "cluster": c, "n": len(members),
            "top_terms": ", ".join(terms[np.argsort(-centroid)[:top_terms]]),
            "dominant_category": members.category.mode().iat[0],
            "dominant_failure": members.failure.mode().iat[0],
            "example": nearest[:80],
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
