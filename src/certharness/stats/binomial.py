"""Exact binomial bounds (Clopper-Pearson), dependency-free.

Used by certification so a grader is gated on what the evidence can *support*, not on
a point estimate: 0 misses out of 50 is not evidence of a <2% miss rate.
"""

from __future__ import annotations

import math


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        log_term = (
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * log_p + (n - i) * log_q
        )  # fmt: skip
        total += math.exp(log_term)
    return min(total, 1.0)


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact upper confidence bound on a binomial rate after k events in n.

    The p at which observing <= k events has probability 1 - confidence.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    alpha = 1.0 - confidence
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    lo, hi = k / n, 1.0
    for _ in range(100):  # bisection; binom_cdf is decreasing in p
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def n_required(max_rate: float, confidence: float = 0.95, misses: int = 0) -> int:
    """Smallest n for which `misses` events still give an upper bound <= max_rate.

    With 0 misses at 95% / 2% this is 149 (the "rule of three": 3 / 0.02 = 150).
    """
    if not 0 < max_rate < 1:
        raise ValueError("max_rate must be in (0, 1)")
    if misses == 0:
        return math.ceil(math.log(1.0 - confidence) / math.log1p(-max_rate))
    n = n_required(max_rate, confidence, 0) + misses
    while clopper_pearson_upper(misses, n, confidence) > max_rate:
        n += 1
    return n
