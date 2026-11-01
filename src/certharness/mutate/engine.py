from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

from certharness.core.types import Mutation, Trajectory
from certharness.mutate.operators import OPERATORS


def stable_seed(*parts: object) -> int:
    """Process-independent seed (Python's hash() is salted per process)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16)


def mutant_id(parent_id: str, operator: str, seed: int) -> str:
    """Opaque id, so neither the operator nor the parent leaks to a grader via the id."""
    return "m-" + hashlib.sha256(f"{parent_id}|{operator}|{seed}".encode()).hexdigest()[:12]


def mutate_all(
    trajs: Iterable[Trajectory], seed: int = 0, operators: Iterable[str] | None = None
) -> list[Trajectory]:
    """Apply every operator to every trajectory, at most one mutant per pair.

    Deterministic: the same inputs and seed always produce identical mutants.
    """
    ops = [OPERATORS[name] for name in (operators or OPERATORS)]
    out: list[Trajectory] = []
    for parent in trajs:
        if parent.mutation is not None:
            raise ValueError(f"{parent.id} is already a mutant; mutate originals only")
        for op in ops:
            op_seed = stable_seed(seed, parent.id, op.name)
            applied = op.apply(parent, random.Random(op_seed))
            if applied is None:
                continue
            if [s.model_dump() for s in applied.traj.steps] == [
                s.model_dump() for s in parent.steps
            ]:
                continue  # equivalent mutant: nothing changed
            out.append(
                applied.traj.model_copy(
                    update={
                        "id": mutant_id(parent.id, op.name, seed),
                        "mutation": Mutation(
                            operator=op.name,
                            severity=op.severity,
                            parent_id=parent.id,
                            seed=op_seed,
                            step_index=applied.step_index,
                            detail=applied.detail,
                        ),
                    }
                )
            )
    return out
