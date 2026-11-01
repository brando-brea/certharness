from __future__ import annotations

from typing import Protocol, runtime_checkable

from certharness.core.types import Trajectory, Verdict


@runtime_checkable
class Grader(Protocol):
    """Anything that turns a trajectory into a pass/fail verdict.

    Graders receive blinded trajectories (no mutation metadata) during certification.
    """

    name: str

    def grade(self, traj: Trajectory) -> Verdict: ...
