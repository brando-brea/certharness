import json

from typer.testing import CliRunner

from certharness.cli import app

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
