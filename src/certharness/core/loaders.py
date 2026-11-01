"""Trajectory and label I/O, plus the hook for external benchmark loaders.

External sources (e.g. a SWE-bench Lite subset, tau-bench) plug in by registering a
loader that converts the benchmark's native format into `Trajectory` objects:

    @register_loader("tau-bench")
    def load_tau_bench(path: Path) -> list[Trajectory]: ...

Loaders for external data must set `provenance=Provenance.EXTERNAL` and
`source="external:<name>"`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from certharness.core.types import Label, Trajectory

TrajectoryLoader = Callable[[Path], list[Trajectory]]

_LOADERS: dict[str, TrajectoryLoader] = {}


def register_loader(name: str) -> Callable[[TrajectoryLoader], TrajectoryLoader]:
    def deco(fn: TrajectoryLoader) -> TrajectoryLoader:
        if name in _LOADERS:
            raise ValueError(f"loader {name!r} already registered")
        _LOADERS[name] = fn
        return fn

    return deco


def get_loader(name: str) -> TrajectoryLoader:
    try:
        return _LOADERS[name]
    except KeyError:
        raise KeyError(f"unknown loader {name!r}; registered: {sorted(_LOADERS)}") from None


def available_loaders() -> list[str]:
    return sorted(_LOADERS)


@register_loader("jsonl")
def load_jsonl(path: Path) -> list[Trajectory]:
    """certharness's native format: one Trajectory JSON object per line."""
    with Path(path).open(encoding="utf-8") as f:
        return [Trajectory.model_validate_json(line) for line in f if line.strip()]


def write_jsonl(trajs: Iterable[Trajectory], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for t in trajs:
            f.write(t.model_dump_json() + "\n")
            n += 1
    return n


def load_labels(path: Path) -> list[Label]:
    """Labels file: a JSON list of Label objects."""
    return [Label.model_validate(x) for x in json.loads(Path(path).read_text(encoding="utf-8"))]
