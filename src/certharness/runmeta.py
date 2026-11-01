"""Run provenance: grader-freeze enforcement and per-run metadata files."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from certharness.config import FREEZE_TAG, FROZEN_PATHS, MODEL_IDS_AS_OF
from certharness.llm.pricing import PRICES_AS_OF

REPO_ROOT = Path(__file__).resolve().parents[2]


class FreezeError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def freeze_status(
    tag: str = FREEZE_TAG, paths: tuple[str, ...] = FROZEN_PATHS, root: Path = REPO_ROOT
) -> dict[str, Any]:
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        return {"tag": tag, "ok": False, "reason": "not a git checkout"}
    tag_commit = _git(root, "rev-list", "-n", "1", tag)
    status: dict[str, Any] = {
        "tag": tag,
        "head": head.stdout.strip(),
        "tag_commit": tag_commit.stdout.strip() or None,
        "frozen_paths": list(paths),
    }
    if tag_commit.returncode != 0:
        return {**status, "ok": False, "reason": f"tag {tag!r} does not exist"}
    # Compare the tag against the working tree (catches committed AND uncommitted edits),
    # and look for new untracked files under the frozen paths.
    changed = _git(root, "diff", "--name-only", tag, "--", *paths).stdout.split()
    untracked = _git(
        root, "ls-files", "--others", "--exclude-standard", "--", *paths
    ).stdout.split()
    if changed or untracked:
        files = ", ".join(changed + untracked)
        return {**status, "ok": False, "reason": f"frozen paths differ from {tag}: {files}"}
    return {**status, "ok": True, "reason": ""}


def require_frozen(tag: str = FREEZE_TAG, root: Path = REPO_ROOT) -> dict[str, Any]:
    status = freeze_status(tag, root=root)
    if not status["ok"]:
        raise FreezeError(
            f"grader freeze check failed: {status['reason']}. Graders, expected answers and "
            f"gates must match tag {tag!r} for runs on real data; changing them needs a new "
            "tag and a threats-to-validity note."
        )
    return status


def write_meta(path: Path, freeze: dict[str, Any] | None, **fields: Any) -> None:
    import anthropic

    meta = {
        "ts": datetime.now(UTC).isoformat(),
        "freeze": freeze,
        "model_ids_as_of": MODEL_IDS_AS_OF,
        "prices_as_of": PRICES_AS_OF,
        "anthropic_sdk": anthropic.__version__,
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
