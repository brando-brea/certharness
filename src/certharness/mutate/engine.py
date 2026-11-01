from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

from certharness.core.seeding import stable_seed
from certharness.core.types import Mutation, Trajectory
from certharness.mutate.operators import OPERATORS
from certharness.mutate.replay import can_replay
from certharness.mutate.replay import replay as replay_traj


def mutant_id(parent_id: str, operator: str, seed: int, replayed: bool = False) -> str:
    """Opaque id, so neither the operator nor the parent leaks to a grader via the id."""
    key = f"{parent_id}|{operator}|{seed}" + ("|replay" if replayed else "")
    return "m-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def mutate_all(
    trajs: Iterable[Trajectory],
    seed: int = 0,
    operators: Iterable[str] | None = None,
    replay: bool = False,
) -> list[Trajectory]:
    """Apply every operator to every trajectory, at most one mutant per pair.

    With replay=True, each mutant's tool calls are re-executed against the task's
    simulated world (when it has one), so the planted fault changes downstream tool
    results instead of only the transcript text. Deterministic for a given seed.
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
            mutated, replayed = applied.traj, False
            if replay and can_replay(mutated):
                mutated, replayed = replay_traj(mutated, applied.pinned), True
            if [s.model_dump() for s in mutated.steps] == [s.model_dump() for s in parent.steps]:
                continue  # equivalent mutant: nothing changed
            out.append(
                mutated.model_copy(
                    update={
                        "id": mutant_id(parent.id, op.name, seed, replayed),
                        "mutation": Mutation(
                            operator=op.name,
                            severity=op.severity,
                            tier=applied.tier or op.tier,
                            parent_id=parent.id,
                            seed=op_seed,
                            step_index=applied.step_index,
                            detail=applied.detail,
                            replayed=replayed,
                        ),
                    }
                )
            )
    return out
