import json

from certharness.core.types import Provenance
from certharness.graders import OutcomeGrader, ToolSequenceGrader
from certharness.llm import Budget, LLMClient
from certharness.synthetic import generate
from certharness.toyagent import World, run_agent, run_episode
from certharness.toyagent.tools import TRANSIENT_ERROR


def test_world_reproduces_reference_trajectories():
    # The simulated backend must return exactly what the synthetic reference shows.
    for t in generate(16, seed=2):
        world = World(t.task.metadata["world"])
        for _, call in t.tool_calls():
            ref = t.steps[t.result_index(call.id)]
            content, is_error = world.call(call.name, call.args)
            assert (json.loads(content), is_error) == (json.loads(ref.content), ref.is_error), t.id


def test_world_errors():
    w = World(
        {
            "contacts": [{"name": "Ana Ruiz", "email": "a@x.com"}],
            "transient_errors": ["search_contacts"],
        }
    )
    assert w.call("search_contacts", {"name": "Ana"}) == (
        json.dumps({"error": TRANSIENT_ERROR}),
        True,
    )
    assert w.call("search_contacts", {"name": "Ana"})[1] is False  # succeeds on retry
    assert w.call("search_contacts", {"name": "Zed"})[1] is True
    assert w.call("search_contacts", {})[1] is True
    assert w.call("rm_rf", {})[1] is True


def _scripted(reference):
    """Fake transport that replays a reference trajectory's tool calls, one per turn."""
    calls = [c for _, c in reference.tool_calls()]

    def transport(params):
        turn = sum(1 for m in params["messages"] if m["role"] == "assistant")
        usage = {"input_tokens": 500, "output_tokens": 50}
        if turn < len(calls):
            c = calls[turn]
            content = [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "tool_use", "id": f"toolu_{turn}", "name": c.name, "input": c.args},
            ]
            return {
                "model": params["model"],
                "stop_reason": "tool_use",
                "usage": usage,
                "content": content,
            }
        return {
            "model": params["model"],
            "stop_reason": "end_turn",
            "usage": usage,
            "content": [{"type": "text", "text": reference.final_answer}],
        }

    return transport


def test_episode_records_a_faithful_trajectory(tmp_path):
    ref = next(t for t in generate(4, seed=3) if t.task.metadata["template"] == "email_file")
    client = LLMClient(Budget(5.0, tmp_path / "l.jsonl"), tmp_path / "c", _scripted(ref))
    traj = run_episode(client, ref.task, "run-x")
    assert traj.provenance is Provenance.AGENT_RUN
    assert [c.name for _, c in traj.tool_calls()] == [c.name for _, c in ref.tool_calls()]
    assert traj.final_answer == ref.final_answer
    assert traj.task.metadata["agent"]["stop"] == "end_turn"
    assert ToolSequenceGrader().grade(traj).passed and OutcomeGrader().grade(traj).passed


def test_echoed_history_keeps_thinking_and_drops_extras(tmp_path):
    ref = generate(1, seed=3)[0]
    seen = []
    inner = _scripted(ref)

    def transport(params):
        seen.append(params)
        return inner(params)

    client = LLMClient(Budget(5.0, tmp_path / "l.jsonl"), tmp_path / "c", transport)
    run_episode(client, ref.task, "run-y")
    assistant = [m for m in seen[-1]["messages"] if m["role"] == "assistant"][0]
    assert assistant["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig"}
    assert seen[0]["output_config"] == {"effort": "low"} and seen[0]["model"] == "claude-sonnet-5"
    assert "fallbacks" not in seen[0] and "betas" not in seen[0]  # single-model runs


def test_injected_error_is_deterministic_and_recorded(tmp_path):
    refs = generate(8, seed=4)
    by_id = {t.task.id: t for t in refs}

    def transport(params):
        task = next(t for t in refs if t.task.instruction == params["messages"][0]["content"])
        return _scripted(by_id[task.task.id])(params)

    client = LLMClient(Budget(5.0, tmp_path / "l.jsonl"), tmp_path / "c", transport)
    a = run_agent(client, [t.task for t in refs], error_rate=1.0, seed=1, workers=2)
    b = run_agent(client, [t.task for t in refs], error_rate=1.0, seed=1, workers=2)
    assert [t.task.metadata["agent"]["injected_errors"] for t in a] == [
        t.task.metadata["agent"]["injected_errors"] for t in b
    ]
    assert all(len(t.task.metadata["agent"]["injected_errors"]) == 1 for t in a)
    # The scripted agent never retries, so the injected failure stays in the trajectory.
    assert all(any(s.is_error for s in t.steps) for t in a)


def test_refusal_stops_episode(tmp_path):
    def transport(params):
        return {"model": "claude-opus-5", "stop_reason": "refusal", "usage": {}, "content": []}

    t = generate(1)[0]
    traj = run_episode(
        LLMClient(Budget(5.0, tmp_path / "l"), tmp_path / "c", transport), t.task, "r"
    )
    assert traj.task.metadata["agent"]["stop"] == "refusal" and traj.final_answer == ""


def test_tool_errors_record_injected_vs_natural(tmp_path):
    ref = next(t for t in generate(4, seed=8) if t.task.metadata["template"] == "weather")
    calls = [c for _, c in ref.tool_calls()]

    def transport(params):
        # turn 0: first call hits the injected 503 (whatever the args); turn 1: wrong
        # city (natural error); turn 2: right call (succeeds); then answer.
        turn = sum(1 for m in params["messages"] if m["role"] == "assistant")
        usage = {"input_tokens": 100, "output_tokens": 10}
        script = [calls[0].args, {**calls[0].args, "city": "Atlantis"}, calls[0].args]
        if turn < 3:
            use = {
                "type": "tool_use",
                "id": f"t{turn}",
                "name": "get_weather",
                "input": script[turn],
            }
            return {"model": "m", "stop_reason": "tool_use", "usage": usage, "content": [use]}
        return {"model": "m", "stop_reason": "end_turn", "usage": usage,
                "content": [{"type": "text", "text": ref.final_answer}]}  # fmt: skip

    client = LLMClient(Budget(5.0, tmp_path / "l"), tmp_path / "c", transport)
    traj = run_episode(client, ref.task, "r", transient_errors=["get_weather"])
    errs = traj.task.metadata["agent"]["tool_errors"]
    assert [e["injected"] for e in errs] == [True, False]
    assert all(traj.steps[e["step"]].is_error for e in errs)
