"""Experimentation statistics: sample size, tests, paired offline comparison, CUPED, peeking."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def sample_size_two_proportions(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm n for a two-sided two-proportion z-test."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> dict:
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se0 = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se0 if se0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"p_control": p1, "p_treatment": p2, "diff": p2 - p1, "z": z, "p_value": p,
            "ci": (p2 - p1 - zc * se, p2 - p1 + zc * se)}


def mcnemar_exact(a_success: np.ndarray, b_success: np.ndarray) -> dict:
    """Paired comparison of two systems on the SAME cases (exact binomial McNemar)."""
    a, b = np.asarray(a_success, bool), np.asarray(b_success, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = float(stats.binomtest(only_b, n, 0.5).pvalue) if n else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0,
                        alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(b, float) - np.asarray(a, float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def cuped(y: np.ndarray, x_pre: np.ndarray) -> dict:
    """Variance reduction using a pre-experiment covariate (CUPED)."""
    y, x = np.asarray(y, float), np.asarray(x_pre, float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    return {"theta": float(theta), "var_before": float(y.var(ddof=1)), "var_after": float(y_adj.var(ddof=1)),
            "variance_reduction": float(1 - y_adj.var(ddof=1) / y.var(ddof=1)), "y_adj": y_adj}


def simulate_ab(p_control: float, p_treat: float, n_per_arm: int, n_sims: int = 2000, seed: int = 0,
                alpha: float = 0.05) -> dict:
    """Monte-Carlo estimate of power (or false-positive rate when p_control == p_treat)."""
    rng = np.random.default_rng(seed)
    x1 = rng.binomial(n_per_arm, p_control, n_sims)
    x2 = rng.binomial(n_per_arm, p_treat, n_sims)
    pool = (x1 + x2) / (2 * n_per_arm)
    se = np.sqrt(np.maximum(pool * (1 - pool) * 2 / n_per_arm, 1e-12))
    z = (x2 - x1) / n_per_arm / se
    return {"reject_rate": float((np.abs(z) > stats.norm.ppf(1 - alpha / 2)).mean())}


def peeking_false_positive_rate(p: float, n_per_arm: int, looks: int, n_sims: int = 2000, seed: int = 0,
                                alpha: float = 0.05) -> float:
    """A/A test; stop at the first look with p < alpha. Shows why peeking inflates Type-I error."""
    rng = np.random.default_rng(seed)
    step = n_per_arm // looks
    zc = stats.norm.ppf(1 - alpha / 2)
    hits = 0
    for _ in range(n_sims):
        a = rng.random(n_per_arm) < p
        b = rng.random(n_per_arm) < p
        for k in range(1, looks + 1):
            n = k * step
            x1, x2 = a[:n].sum(), b[:n].sum()
            pool = (x1 + x2) / (2 * n)
            se = math.sqrt(max(pool * (1 - pool) * 2 / n, 1e-12))
            if abs((x2 - x1) / n / se) > zc:
                hits += 1
                break
    return hits / n_sims
