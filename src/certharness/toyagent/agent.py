"""A small tool-using agent (Claude Messages API) run against the simulated World.

Manual agentic loop rather than the SDK tool runner, so that every request goes
through certharness's budgeted, disk-cached client and every step is recorded.
Produces Trajectories with provenance AGENT_RUN. Single-model by design: no
server-side fallbacks, so every trajectory comes from the pinned model.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from certharness.config import AGENT_EFFORT, AGENT_MODEL
from certharness.core.types import Provenance, Step, Task, ToolCall, Trajectory
from certharness.llm.client import LLMClient
from certharness.mutate.engine import stable_seed
from certharness.toyagent.tools import TOOL_SPECS, World

DEFAULT_MODEL = AGENT_MODEL
DEFAULT_EFFORT = AGENT_EFFORT
MAX_TURNS = 10

SYSTEM = (
    "You are a personal assistant with tool access to the user's contacts, calendar, "
    "email, files and a weather service. Today is 2026-11-01.\n"
    "Complete the user's request using the tools and follow any constraints in the "
    "request exactly. Only state facts (ids, dates, values) that came from tool results. "
    "If a tool fails you may retry it; if you cannot complete the request, say so plainly.\n"
    "In your final reply use ISO dates (YYYY-MM-DD) and 24-hour times, and include any "
    "ids returned by actions you took (event id, message id)."
)

# Fields to keep when echoing assistant content blocks back to the API.
_ECHO_FIELDS = {
    "text": ("type", "text"),
    "tool_use": ("type", "id", "name", "input"),
    "thinking": ("type", "thinking", "signature"),
    "redacted_thinking": ("type", "data"),
}


def _echo(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for block in content:
        fields = _ECHO_FIELDS.get(block.get("type"))
        if fields:
            out.append({k: block[k] for k in fields if k in block})
    return out


def agent_request(model: str, effort: str, tools: list[dict], messages: list[dict]) -> dict:
    """The exact Messages API request the agent sends each turn."""
    return {
        "model": model,
        "max_tokens": 4096,
        "system": SYSTEM,
        "tools": tools,
        "messages": messages,
        "output_config": {"effort": effort},
        "cache_control": {"type": "ephemeral"},
    }


def tools_for(task: Task) -> list[dict[str, Any]]:
    return [{"name": t, **TOOL_SPECS[t]} for t in task.tools]


def run_episode(
    client: LLMClient,
    task: Task,
    traj_id: str,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    transient_errors: list[str] | None = None,
) -> Trajectory:
    world_spec = {**task.metadata.get("world", {}), "transient_errors": transient_errors or []}
    world = World(world_spec)
    tools = tools_for(task)
    messages: list[dict[str, Any]] = [{"role": "user", "content": task.instruction}]
    steps = [Step(role="user", content=task.instruction)]
    # tool_errors records every error result and whether it was injected or natural.
    meta: dict[str, Any] = {
        "model": model,
        "effort": effort,
        "injected_errors": transient_errors or [],
        "tool_errors": [],
    }

    for _ in range(MAX_TURNS):
        response = client.create("agent", **agent_request(model, effort, tools, messages))
        content = response.get("content") or []
        text = "\n".join(b["text"] for b in content if b.get("type") == "text").strip()
        uses = [b for b in content if b.get("type") == "tool_use"]

        if response.get("stop_reason") == "refusal":
            meta["stop"] = "refusal"
            steps.append(Step(role="assistant", content=text))
            break
        if not uses:
            meta["stop"] = response.get("stop_reason")
            steps.append(Step(role="assistant", content=text))
            break

        steps.append(
            Step(
                role="assistant",
                content=text,
                tool_calls=[ToolCall(id=u["id"], name=u["name"], args=u["input"]) for u in uses],
            )
        )
        results = []
        for u in uses:
            out, is_error = world.call(u["name"], u["input"])
            if is_error:
                meta["tool_errors"].append(
                    {"step": len(steps), "tool": u["name"], "injected": world.last_injected}
                )
            steps.append(Step(role="tool", tool_call_id=u["id"], content=out, is_error=is_error))
            results.append({"type": "tool_result", "tool_use_id": u["id"], "content": out,
                            **({"is_error": True} if is_error else {})})  # fmt: skip
        messages.append({"role": "assistant", "content": _echo(content)})
        messages.append({"role": "user", "content": results})
    else:
        meta["stop"] = "max_turns"

    return Trajectory(
        id=traj_id,
        task=task.model_copy(update={"metadata": {**task.metadata, "agent": meta}}),
        steps=steps,
        provenance=Provenance.AGENT_RUN,
        source=f"agent:{model}:effort={effort}",
    )


def run_agent(
    client: LLMClient,
    tasks: list[Task],
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    error_rate: float = 0.0,
    seed: int = 0,
    workers: int = 4,
) -> list[Trajectory]:
    """Run one episode per task. With probability error_rate a task gets one transient
    error injected into one of its required tools (fails once, succeeds on retry)."""

    def one(task: Task) -> Trajectory:
        rng = random.Random(stable_seed(seed, task.id, "inject"))
        required = sorted({c.name for c in task.expected_calls})
        inject = [rng.choice(required)] if required and rng.random() < error_rate else []
        return run_episode(client, task, f"run-{task.id}", model, effort, inject)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, tasks))
