"""Agreement and detection metrics. Pure functions over boolean pass/fail vectors."""

from __future__ import annotations

from collections.abc import Sequence


def accuracy(truth: Sequence[bool], pred: Sequence[bool]) -> float | None:
    _check(truth, pred)
    if not truth:
        return None
    return sum(t == p for t, p in zip(truth, pred, strict=True)) / len(truth)


def cohen_kappa(truth: Sequence[bool], pred: Sequence[bool]) -> float | None:
    """Chance-corrected agreement. None when undefined (empty, or both raters constant
    and identical, so expected agreement is 1)."""
    _check(truth, pred)
    n = len(truth)
    if n == 0:
        return None
    po = sum(t == p for t, p in zip(truth, pred, strict=True)) / n
    pt, pp = sum(truth) / n, sum(pred) / n
    pe = pt * pp + (1 - pt) * (1 - pp)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def false_alarm_rate(truth: Sequence[bool], pred: Sequence[bool]) -> float | None:
    """Fraction of truly passing items the grader failed."""
    _check(truth, pred)
    pos = [p for t, p in zip(truth, pred, strict=True) if t]
    return None if not pos else sum(not p for p in pos) / len(pos)


def false_pass_rate(truth: Sequence[bool], pred: Sequence[bool]) -> float | None:
    """Fraction of truly failing items the grader passed."""
    _check(truth, pred)
    neg = [p for t, p in zip(truth, pred, strict=True) if not t]
    return None if not neg else sum(neg) / len(neg)


def _check(truth: Sequence[bool], pred: Sequence[bool]) -> None:
    if len(truth) != len(pred):
        raise ValueError(f"length mismatch: {len(truth)} labels vs {len(pred)} verdicts")
