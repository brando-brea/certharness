"""Deterministic code graders. Cheap, transparent, and blind to anything outside the spec.

They are deliberately narrow: each checks one family of properties, so the
mutation-adequacy matrix (RQ2) shows which planted faults each one can and cannot see.
"""

from __future__ import annotations

import re

from certharness.core.types import ExpectedCall, ToolCall, Trajectory, Verdict

# Phrases that count as the agent acknowledging a failure in its final answer.
_ACK_FAILURE = re.compile(
    r"\b(fail(ed|ure)?|error|unable|could ?n[o']t|can ?n[o']t|did ?n[o']t|sorry|unfortunately)\b",
    re.IGNORECASE,
)


def _matches(call: ToolCall, exp: ExpectedCall) -> bool:
    return call.name == exp.name and all(call.args.get(k) == v for k, v in exp.args.items())


class ToolSequenceGrader:
    """Process check: required calls appear in order with required args; no forbidden tools."""

    name = "tool_sequence"

    def grade(self, traj: Trajectory) -> Verdict:
        calls = [c for _, c in traj.tool_calls()]
        reasons: list[str] = []

        forbidden = set(traj.task.forbidden_tools)
        for c in calls:
            if c.name in forbidden:
                reasons.append(f"called forbidden tool {c.name!r}")

        # Expected calls must form an ordered subsequence of the actual calls.
        pos = 0
        for exp in traj.task.expected_calls:
            while pos < len(calls) and not _matches(calls[pos], exp):
                pos += 1
            if pos == len(calls):
                reasons.append(f"missing required call {exp.name}({exp.args}) in order")
                break
            pos += 1

        return Verdict(
            trajectory_id=traj.id,
            grader=self.name,
            passed=not reasons,
            score=0.0 if reasons else 1.0,
            reasons=reasons,
        )


class OutcomeGrader:
    """Outcome check: final answer matches the expected pattern and does not paper over
    a tool error."""

    name = "outcome"

    def grade(self, traj: Trajectory) -> Verdict:
        final = traj.final_answer
        reasons: list[str] = []

        if final is None:
            reasons.append("no final answer")
        else:
            pattern = traj.task.expected_final
            if pattern and not re.search(pattern, final, re.IGNORECASE):
                reasons.append(f"final answer does not match {pattern!r}")
            if any(s.role == "tool" and s.is_error for s in traj.steps) and not _ACK_FAILURE.search(
                final
            ):
                reasons.append("final answer ignores a tool error")

        return Verdict(
            trajectory_id=traj.id,
            grader=self.name,
            passed=not reasons,
            score=0.0 if reasons else 1.0,
            reasons=reasons,
        )
