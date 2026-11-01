"""Replay: re-execute a (mutated) trajectory's tool calls against its simulated world.

Static mutation edits transcript text only, so a corrupted argument can leave the
tool result untouched (a typo'd contact search still "returns" the right person).
Such mutants are often *equivalent* (no real effect) or *incoherent* (a result no
real tool would return). Replay re-runs every call, in order, against a fresh copy of
the task's world, so the planted fault's consequences flow into the tool results.

Limits: the agent's own later turns are kept as recorded, so a replayed mutant shows
what the agent *said*, not what it would have done after seeing the new result.
Steps listed in `pinned` keep their recorded result (used for planted tool errors).
"""

from __future__ import annotations

from certharness.core.types import Trajectory
from certharness.toyagent.tools import World


def can_replay(traj: Trajectory) -> bool:
    return "world" in traj.task.metadata


def replay(traj: Trajectory, pinned: frozenset[int] = frozenset()) -> Trajectory:
    """Copy of traj with every unpinned tool result recomputed by the world."""
    if not can_replay(traj):
        raise ValueError(f"{traj.id}: no world spec in task metadata; cannot replay")
    injected = traj.task.metadata.get("agent", {}).get("injected_errors", [])
    world = World({**traj.task.metadata["world"], "transient_errors": injected})
    out = traj.model_copy(deep=True)
    for step in out.steps:
        for call in step.tool_calls:
            idx = out.result_index(call.id)
            content, is_error = world.call(call.name, call.args)
            if idx is not None and idx not in pinned:
                out.steps[idx].content = content
                out.steps[idx].is_error = is_error
    return out
