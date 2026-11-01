import json

import pytest

from certharness.core.loaders import (
    available_loaders,
    get_loader,
    load_jsonl,
    load_labels,
    register_loader,
    write_jsonl,
)
from certharness.core.types import Mutation, Provenance


def test_final_answer_and_tool_calls(originals):
    t = next(t for t in originals if t.task.metadata["template"] == "email_file")
    assert t.final_answer.startswith("Sent the contents")
    assert [c.name for _, c in t.tool_calls()] == ["read_file", "search_contacts", "send_email"]
    for idx, call in t.tool_calls():
        assert t.result_index(call.id) == idx + 1


def test_blinded_strips_mutation(originals):
    t = originals[0].model_copy(
        update={
            "mutation": Mutation(
                operator="x", severity="major", tier="subtle", parent_id="p", seed=1
            )
        }
    )
    assert t.blinded().mutation is None
    assert t.mutation is not None


def test_jsonl_roundtrip(tmp_path, originals):
    path = tmp_path / "t.jsonl"
    assert write_jsonl(originals, path) == len(originals)
    assert load_jsonl(path) == originals


def test_labels_roundtrip(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([
        {"trajectory_id": "a", "passed": True, "provenance": "human", "rater": "r1", "session": 1},
    ]))  # fmt: skip
    [lab] = load_labels(path)
    assert lab.provenance is Provenance.HUMAN and lab.session == 1


def test_external_loader_hook(tmp_path, originals):
    @register_loader("test-bench")
    def _load(path):
        return [t.model_copy(update={"provenance": Provenance.EXTERNAL}) for t in originals[:2]]

    assert "test-bench" in available_loaders()
    assert all(t.provenance is Provenance.EXTERNAL for t in get_loader("test-bench")(tmp_path))
    with pytest.raises(ValueError, match="already registered"):
        register_loader("test-bench")(_load)


def test_unknown_loader():
    with pytest.raises(KeyError, match="unknown loader"):
        get_loader("nope")
