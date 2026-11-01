"""Grader certification.

A grader is CERTIFIED only if (a) every gate passes and (b) the evidence is
reportable: human/external labels on real agent or benchmark trajectories. Otherwise
it stays in SHADOW mode — it may run and report, but its verdicts must not gate.

Two evidence sources, matching the two v0.1 research questions:
  RQ1 validity — grader vs. ground-truth labels on original trajectories.
  RQ2 adequacy — grader vs. seeded mutants, which are failing by construction.

Critical false-passes get two gates instead of a point estimate:
  * zero misses on critical mutants tagged *obvious* (blatant faults), and
  * the one-sided Clopper-Pearson upper bound on the critical false-pass rate must be
    <= max_critical_fp_upper. That bound is unreachable below a minimum number of
    critical mutants, even with zero misses; the report says what that n is.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from certharness.certify import metrics
from certharness.core.grader import Grader
from certharness.core.types import (
    REPORTABLE_LABELS,
    REPORTABLE_TRAJECTORIES,
    Label,
    Provenance,
    Trajectory,
    Verdict,
)
from certharness.stats.binomial import clopper_pearson_upper, n_required

_DEFAULT_LABELABLE = {Provenance.SYNTHETIC, Provenance.STUB}


class Gates(BaseModel):
    min_kappa: float = 0.60
    max_false_alarm_rate: float = 0.10
    min_mutation_score: float = 0.80
    max_obvious_critical_misses: int = 0
    max_critical_fp_upper: float = 0.02
    confidence: float = 0.95


class GateResult(BaseModel):
    name: str
    value: float | None
    threshold: float
    passed: bool | None  # None = not applicable (insufficient evidence)
    note: str = ""


class OperatorStats(BaseModel):
    operator: str
    severity: str
    tier: str
    n: int
    caught: int

    @computed_field
    @property
    def catch_rate(self) -> float:
        return self.caught / self.n if self.n else 0.0


class Validity(BaseModel):
    n: int
    label_pass_rate: float | None
    accuracy: float | None
    kappa: float | None
    false_alarm_rate: float | None
    false_pass_rate: float | None


class CertificationReport(BaseModel):
    grader: str
    status: Literal["certified", "shadow"]
    reportable: bool
    shadow_reasons: list[str] = Field(default_factory=list)
    label_provenance: dict[str, int]
    trajectory_provenance: dict[str, int]
    validity: Validity
    n_mutants: int
    mutation_score: float | None
    n_critical: int
    critical_misses: int
    critical_false_pass_rate: float | None
    critical_fp_upper: float | None
    critical_n_required: int  # critical mutants needed for the bound gate with 0 misses
    n_obvious_critical: int
    obvious_critical_misses: int
    operators: list[OperatorStats]
    gates: list[GateResult]
    verdicts: list[Verdict] = Field(default_factory=list, repr=False)


def _gate(name: str, value: float | None, threshold: float, higher_is_better: bool) -> GateResult:
    if value is None:
        return GateResult(name=name, value=None, threshold=threshold, passed=None,
                          note="insufficient evidence")  # fmt: skip
    ok = value >= threshold if higher_is_better else value <= threshold
    return GateResult(name=name, value=value, threshold=threshold, passed=ok)


def certify(
    grader: Grader,
    originals: list[Trajectory],
    mutants: list[Trajectory],
    labels: list[Label] | None = None,
    gates: Gates | None = None,
) -> CertificationReport:
    gates = gates or Gates()

    # Ground truth for originals: explicit labels, or "passing by construction" for
    # synthetic/stub data only. Real trajectories are never assumed correct.
    by_id = {lab.trajectory_id: lab for lab in labels or []}
    labeled: list[tuple[Trajectory, Label]] = []
    for t in originals:
        if t.mutation is not None:
            raise ValueError(f"{t.id} is a mutant; pass it in `mutants`, not `originals`")
        if t.id in by_id:
            labeled.append((t, by_id[t.id]))
        elif t.provenance in _DEFAULT_LABELABLE:
            labeled.append((t, Label(trajectory_id=t.id, passed=True, provenance=t.provenance)))
        else:
            raise ValueError(f"{t.id} ({t.provenance}) has no label; only synthetic/stub "
                             "trajectories may be assumed passing")  # fmt: skip
    for m in mutants:
        if m.mutation is None:
            raise ValueError(f"{m.id} has no mutation record; not a mutant")

    # Graders see blinded copies only.
    orig_v = [grader.grade(t.blinded()) for t, _ in labeled]
    mut_v = [grader.grade(m.blinded()) for m in mutants]

    truth = [lab.passed for _, lab in labeled]
    pred = [v.passed for v in orig_v]
    single_class = len(set(truth)) < 2
    validity = Validity(
        n=len(labeled),
        label_pass_rate=sum(truth) / len(truth) if truth else None,
        accuracy=metrics.accuracy(truth, pred),
        kappa=None if single_class else metrics.cohen_kappa(truth, pred),
        false_alarm_rate=metrics.false_alarm_rate(truth, pred),
        false_pass_rate=metrics.false_pass_rate(truth, pred),
    )

    per_op: dict[str, OperatorStats] = {}
    for m, v in zip(mutants, mut_v, strict=True):
        st = per_op.setdefault(
            m.mutation.operator,
            OperatorStats(
                operator=m.mutation.operator,
                severity=m.mutation.severity,
                tier=m.mutation.tier,
                n=0,
                caught=0,
            ),
        )
        st.n += 1
        st.caught += not v.passed
    critical = [
        (m, v) for m, v in zip(mutants, mut_v, strict=True) if m.mutation.severity == "critical"
    ]
    obvious = [v for m, v in critical if m.mutation.tier == "obvious"]
    crit_misses = sum(v.passed for _, v in critical)
    obvious_misses = sum(v.passed for v in obvious)
    mutation_score = sum(not v.passed for v in mut_v) / len(mut_v) if mut_v else None
    crit_fp = crit_misses / len(critical) if critical else None
    crit_upper = (
        clopper_pearson_upper(crit_misses, len(critical), gates.confidence) if critical else None
    )
    need = n_required(gates.max_critical_fp_upper, gates.confidence)

    gate_results = [
        _gate("kappa", validity.kappa, gates.min_kappa, True),
        _gate("false_alarm_rate", validity.false_alarm_rate, gates.max_false_alarm_rate, False),
        _gate("mutation_score", mutation_score, gates.min_mutation_score, True),
        _gate(
            "obvious_critical_misses",
            obvious_misses if obvious else None,
            gates.max_obvious_critical_misses,
            False,
        ),
        _gate("critical_fp_upper", crit_upper, gates.max_critical_fp_upper, False),
    ]
    if critical and len(critical) < need:
        gate_results[-1].note = (
            f"unachievable at n={len(critical)}: needs >= {need} critical mutants even with "
            f"0 misses ({gates.confidence:.0%} one-sided Clopper-Pearson)"
        )
    if single_class and labeled:
        gate_results[0].note = "labels are single-class; kappa undefined"

    label_prov = Counter(str(lab.provenance) for _, lab in labeled)
    traj_prov = Counter(str(t.provenance) for t in [*originals, *mutants])
    reasons = []
    if not labeled or any(lab.provenance not in REPORTABLE_LABELS for _, lab in labeled):
        reasons.append("labels are not all human/external")
    if any(t.provenance not in REPORTABLE_TRAJECTORIES for t in [*originals, *mutants]):
        reasons.append("trajectories are not all from real agent runs or external benchmarks")
    reportable = not reasons
    for g in gate_results:
        if g.passed is not True:
            reasons.append(f"gate {g.name}: {'n/a' if g.passed is None else 'failed'}")

    return CertificationReport(
        grader=grader.name,
        status="certified" if not reasons else "shadow",
        reportable=reportable,
        shadow_reasons=reasons,
        label_provenance=dict(label_prov),
        trajectory_provenance=dict(traj_prov),
        validity=validity,
        n_mutants=len(mutants),
        mutation_score=mutation_score,
        n_critical=len(critical),
        critical_misses=crit_misses,
        critical_false_pass_rate=crit_fp,
        critical_fp_upper=crit_upper,
        critical_n_required=need,
        n_obvious_critical=len(obvious),
        obvious_critical_misses=obvious_misses,
        operators=sorted(per_op.values(), key=lambda s: s.operator),
        gates=gate_results,
        verdicts=[*orig_v, *mut_v],
    )
