import json

from typer.testing import CliRunner

from certharness.cli import app
from certharness.core.loaders import write_jsonl

runner = CliRunner()


def test_synth_mutate_certify(tmp_path):
    synth, muts, out = tmp_path / "s.jsonl", tmp_path / "m.jsonl", tmp_path / "reports"

    r = runner.invoke(app, ["synth", "--out", str(synth), "--n", "8", "--seed", "1"])
    assert r.exit_code == 0, r.output

    r = runner.invoke(app, ["mutate", str(synth), "--out", str(muts), "--seed", "1"])
    assert r.exit_code == 0, r.output
    assert "wrote 48 mutants from 8 trajectories" in r.output

    r = runner.invoke(
        app, ["certify", "--originals", str(synth), "--mutants", str(muts), "--out-dir", str(out)]
    )
    assert r.exit_code == 0, r.output
    assert "NOT REPORTABLE" in r.output
    assert "catch rate by operator" in r.output
    report = json.loads((out / "certify_tool_sequence.json").read_text())
    assert report["status"] == "shadow" and report["reportable"] is False
    assert {"operator", "catch_rate"} <= set(report["operators"][0])


def test_mutate_rejects_unknown_operator(tmp_path):
    synth = tmp_path / "s.jsonl"
    runner.invoke(app, ["synth", "--out", str(synth), "--n", "2"])
    r = runner.invoke(app, ["mutate", str(synth), "--out", str(tmp_path / "m"), "--operator", "x"])
    assert r.exit_code != 0


def test_list():
    r = runner.invoke(app, ["list"])
    assert r.exit_code == 0 and "hallucinate_result" in r.output


def test_label_provisional_keeps_rejected_runs(tmp_path, monkeypatch):
    import certharness.cli as cli
    from certharness.core.loaders import load_jsonl
    from certharness.core.types import Provenance
    from certharness.synthetic import generate

    monkeypatch.setattr(
        cli, "require_frozen", lambda tag: {"tag_commit": "f" * 40, "head": "a" * 40}
    )
    trajs = [t.model_copy(update={"provenance": Provenance.AGENT_RUN}) for t in generate(4)]
    broken = trajs[0].model_copy(deep=True)
    broken.steps[-1].content = "Done."  # fails the outcome grader
    trajs[0] = broken
    src = tmp_path / "agent.jsonl"
    write_jsonl(trajs, src)
    args = ["label-provisional", str(src), "--labels-out", str(tmp_path / "labels.json"),
            "--passing-out", str(tmp_path / "pass.jsonl"),
            "--rejected-out", str(tmp_path / "rej.jsonl")]  # fmt: skip
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output
    rejected = load_jsonl(tmp_path / "rej.jsonl")
    assert [t.id for t in rejected] == [broken.id]
    assert rejected[0].task.metadata["pool"] == "natural_failure"
    assert len(load_jsonl(tmp_path / "pass.jsonl")) == 3
    assert (tmp_path / "labels.meta.json").exists()


def test_real_data_requires_freeze(tmp_path, monkeypatch):
    import certharness.cli as cli
    from certharness.core.types import Provenance
    from certharness.runmeta import FreezeError
    from certharness.synthetic import generate

    def refuse(tag):
        raise FreezeError("frozen paths differ")

    monkeypatch.setattr(cli, "require_frozen", refuse)
    trajs = [t.model_copy(update={"provenance": Provenance.AGENT_RUN}) for t in generate(2)]
    src = tmp_path / "agent.jsonl"
    write_jsonl(trajs, src)
    r = runner.invoke(app, ["certify", "--originals", str(src), "--mutants", str(src)])
    assert r.exit_code == 3 and "REFUSED" in r.output
