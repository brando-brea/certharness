import pytest

from certharness.certify import certify
from certharness.certify.metrics import (
    accuracy,
    cohen_kappa,
    false_alarm_rate,
    false_pass_rate,
)
from certharness.core.types import Label, Provenance, Verdict
from certharness.graders import ToolSequenceGrader

T, F = True, False


def test_kappa_known_values():
    assert cohen_kappa([T, T, F, F], [T, T, F, F]) == 1.0
    assert cohen_kappa([T, F, T, F], [F, T, F, T]) == -1.0
    # 20 items: po = 0.7, pe = 0.5 -> kappa = 0.4
    truth = [T] * 10 + [F] * 10
    pred = [T] * 7 + [F] * 3 + [F] * 7 + [T] * 3
    assert cohen_kappa(truth, pred) == pytest.approx(0.4)
    assert cohen_kappa([T, T], [T, T]) is None
    assert cohen_kappa([], []) is None


def test_rates():
    truth, pred = [T, T, T, F, F], [T, F, T, T, F]
    assert accuracy(truth, pred) == pytest.approx(0.6)
    assert false_alarm_rate(truth, pred) == pytest.approx(1 / 3)
    assert false_pass_rate(truth, pred) == pytest.approx(0.5)
    assert false_pass_rate([T], [T]) is None
    with pytest.raises(ValueError, match="length mismatch"):
        accuracy([T], [])


def test_synthetic_is_never_reportable(originals, mutants):
    r = certify(ToolSequenceGrader(), originals, mutants)
    assert not r.reportable and r.status == "shadow"
    assert "labels are not all human/external" in r.shadow_reasons
    assert r.validity.kappa is None and r.gates[0].passed is None


def test_operator_stats_match_matrix(originals, mutants):
    r = certify(ToolSequenceGrader(), originals, mutants)
    stats = {s.operator: s for s in r.operators}
    assert stats["forbidden_action"].catch_rate == 1.0
    assert stats["false_success"].catch_rate == 0.0  # outside this grader's view, by design
    assert sum(s.n for s in r.operators) == r.n_mutants == len(mutants)


def test_unlabeled_real_trajectory_is_rejected(originals):
    real = [t.model_copy(update={"provenance": Provenance.AGENT_RUN}) for t in originals[:2]]
    with pytest.raises(ValueError, match="has no label"):
        certify(ToolSequenceGrader(), real, [])


def test_mutant_in_originals_is_rejected(mutants):
    with pytest.raises(ValueError, match="is a mutant"):
        certify(ToolSequenceGrader(), mutants[:1], [])


class _Oracle:
    """Grader that reads the hidden fault record; certify must blind it."""

    name = "oracle"

    def grade(self, traj):
        return Verdict(trajectory_id=traj.id, grader=self.name, passed=traj.mutation is None)


class _Scripted:
    """Grader that fails exactly the given trajectory ids."""

    name = "scripted"

    def __init__(self, fail_ids):
        self.fail_ids = fail_ids

    def grade(self, traj):
        return Verdict(trajectory_id=traj.id, grader=self.name, passed=traj.id not in self.fail_ids)


def test_graders_receive_blinded_trajectories(originals, mutants):
    r = certify(_Oracle(), originals, mutants)
    assert r.mutation_score == 0.0  # the oracle saw no mutation records


def test_certified_requires_reportable_evidence_and_all_gates(originals, mutants):
    real = [t.model_copy(update={"provenance": Provenance.AGENT_RUN}) for t in originals]
    real_muts = [m.model_copy(update={"provenance": Provenance.AGENT_RUN}) for m in mutants]
    # Human labels: first 4 originals judged failing, the rest passing.
    labels = [
        Label(trajectory_id=t.id, passed=i >= 4, provenance=Provenance.HUMAN, rater="r1")
        for i, t in enumerate(real)
    ]
    perfect = _Scripted({t.id for t in real[:4]} | {m.id for m in real_muts})
    r = certify(perfect, real, real_muts, labels)
    assert r.reportable and r.status == "certified", r.shadow_reasons
    assert r.validity.kappa == 1.0 and r.critical_false_pass_rate == 0.0

    # Same evidence, but the grader misses 2 of 72 critical mutants (2.8% > 2% gate).
    critical = [m.id for m in real_muts if m.mutation.severity == "critical"]
    assert len(critical) == 72
    leaky = _Scripted(perfect.fail_ids - set(critical[:2]))
    r = certify(leaky, real, real_muts, labels)
    assert r.reportable and r.status == "shadow"
    assert any("critical_false_pass_rate" in x for x in r.shadow_reasons)
