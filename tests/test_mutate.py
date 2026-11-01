import random

import pytest

from certharness.core.types import Task, Trajectory
from certharness.mutate import OPERATORS, mutate_all, perturb
from certharness.mutate.operators import is_fact


def _dump(trajs):
    return [t.model_dump() for t in trajs]


def test_deterministic(originals):
    assert _dump(mutate_all(originals, seed=3)) == _dump(mutate_all(originals, seed=3))


def test_seed_changes_mutants(originals):
    assert _dump(mutate_all(originals, seed=3)) != _dump(mutate_all(originals, seed=4))


def test_every_operator_fires_on_every_template(mutants, originals):
    templates = {t.id: t.task.metadata["template"] for t in originals}
    fired = {(templates[m.mutation.parent_id], m.mutation.operator) for m in mutants}
    expected = {(tpl, op) for tpl in set(templates.values()) for op in OPERATORS}
    assert fired == expected


def test_mutants_differ_from_parent_and_record_fault(mutants, originals):
    parents = {t.id: t for t in originals}
    for m in mutants:
        p = parents[m.mutation.parent_id]
        assert _dump(m.steps) != _dump(p.steps)
        assert m.provenance == p.provenance and m.task == p.task
        assert m.mutation.severity == OPERATORS[m.mutation.operator].severity
        assert m.mutation.detail


def test_ids_are_opaque_and_unique(mutants):
    ids = [m.id for m in mutants]
    assert len(set(ids)) == len(ids)
    for m in mutants:
        assert m.mutation.operator not in m.id and m.mutation.parent_id not in m.id


def test_originals_not_modified(originals):
    before = _dump(originals)
    mutate_all(originals, seed=11)
    assert _dump(originals) == before


def test_refuses_to_mutate_a_mutant(mutants):
    with pytest.raises(ValueError, match="already a mutant"):
        mutate_all(mutants[:1])


def test_inapplicable_operator_returns_none():
    t = Trajectory(
        id="t", task=Task(id="t", instruction="hi", tools=[]), steps=[], provenance="stub"
    )
    for op in OPERATORS.values():
        assert op.apply(t, random.Random(0)) is None


def test_forbidden_action_never_splits_call_and_result(mutants):
    for m in (m for m in mutants if m.mutation.operator == "forbidden_action"):
        for i, s in enumerate(m.steps):
            if s.role == "assistant" and s.tool_calls:
                assert m.steps[i + 1].role == "tool"
                assert m.steps[i + 1].tool_call_id == s.tool_calls[0].id


def test_hallucination_changes_a_whole_token(mutants):
    for m in (m for m in mutants if m.mutation.operator == "hallucinate_result"):
        assert "2026-11-" in m.final_answer or "°C" in m.final_answer or "_" in m.final_answer


@pytest.mark.parametrize(
    "value",
    [7, 2.5, True, "2026-11-30", "23:30", "ana.ruiz@example.com", "evt_1000", "Lindqvist"],
)
def test_perturb_changes_value_keeps_shape(value):
    for seed in range(20):
        out = perturb(value, random.Random(seed))
        assert out != value and type(out) is type(value)


def test_perturb_date_and_time_stay_valid():
    rng = random.Random(0)
    for _ in range(50):
        d = perturb("2026-12-31", rng)
        assert len(d) == 10 and d[4] == "-" and d[7] == "-"
        hh, mm = perturb("23:30", rng).split(":")
        assert 0 <= int(hh) < 24 and mm == "30"


def test_is_fact():
    assert is_fact(3) and is_fact("evt_12") and is_fact("a@b.co")
    assert not is_fact("Gus Lindqvist") and not is_fact(True)


def test_forbidden_calls_have_schema_valid_args(mutants):
    for m in (m for m in mutants if m.mutation.operator == "forbidden_action"):
        [call] = [c for _, c in m.tool_calls() if c.name in m.task.forbidden_tools]
        schema = m.task.tool_schemas[call.name]
        assert set(schema["required"]) <= set(call.args)
        assert all(v not in ("", None) for v in call.args.values())
        assert call.id != "cx"
