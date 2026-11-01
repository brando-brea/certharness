import json

from certharness.mutate import mutate_all
from certharness.mutate.replay import can_replay, replay
from certharness.synthetic import generate


def _dump(t):
    return [s.model_dump() for s in t.steps]


def test_replay_of_unmutated_trajectory_is_identical():
    for t in generate(16, seed=11):
        assert _dump(replay(t)) == _dump(t)


def _by_op(mutants, op):
    return [m for m in mutants if m.mutation.operator == op]


def test_replayed_typo_search_now_fails():
    parents = generate(24, seed=12)
    muts = mutate_all(parents, seed=12, operators=["corrupt_arg"], replay=True)
    typo = [m for m in muts if m.mutation.detail.startswith("search_contacts.name")]
    assert typo
    for m in typo:
        call = next(c for _, c in m.tool_calls() if c.name == "search_contacts")
        idx = m.result_index(call.id)
        assert m.steps[idx].is_error and "no contact matching" in m.steps[idx].content
        assert m.mutation.replayed


def test_static_typo_search_keeps_stale_result():
    parents = generate(24, seed=12)
    muts = mutate_all(parents, seed=12, operators=["corrupt_arg"], replay=False)
    typo = [m for m in muts if m.mutation.detail.startswith("search_contacts.name")]
    assert typo and not any(m.mutation.replayed for m in typo)
    assert not any(any(s.is_error for s in m.steps) for m in typo)


def test_replay_keeps_planted_error_pinned():
    muts = mutate_all(generate(16, seed=13), seed=13, operators=["false_success"], replay=True)
    for m in muts:
        assert m.steps[m.mutation.step_index].is_error


def test_swapped_tool_gets_its_own_result():
    muts = mutate_all(generate(16, seed=14), seed=14, operators=["swap_tool"], replay=True)
    stale = 0
    for m in muts:
        static = mutate_all([p for p in generate(16, seed=14) if p.id == m.mutation.parent_id],
                            seed=14, operators=["swap_tool"])[0]  # fmt: skip
        idx = m.mutation.step_index + 1
        stale += m.steps[idx].content == static.steps[idx].content
    assert stale < len(muts)  # replay changed the swapped call's result in most cases


def test_false_success_tier_depends_on_final_answer():
    muts = mutate_all(generate(32, seed=15), seed=15, operators=["false_success"])
    tiers = {m.mutation.tier for m in muts}
    assert tiers == {"obvious", "subtle"}
    for m in muts:
        assert ("does not feed" in m.mutation.detail) == (m.mutation.tier == "subtle")


def test_replay_ids_differ_from_static_and_no_world_falls_back():
    parents = generate(4, seed=16)
    static = {m.id for m in mutate_all(parents, seed=16)}
    replayed = {m.id for m in mutate_all(parents, seed=16, replay=True)}
    assert not static & replayed
    bare = [
        p.model_copy(update={"task": p.task.model_copy(update={"metadata": {}})}) for p in parents
    ]
    assert not can_replay(bare[0])
    assert not any(m.mutation.replayed for m in mutate_all(bare, seed=16, replay=True))


def test_forbidden_call_result_comes_from_world():
    muts = mutate_all(generate(16, seed=17), seed=17, operators=["forbidden_action"], replay=True)
    for m in muts:
        idx = m.mutation.step_index + 1
        assert json.loads(m.steps[idx].content) != {"status": "ok"}
