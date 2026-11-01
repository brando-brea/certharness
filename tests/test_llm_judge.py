import json

import pytest

from certharness.certify import certify
from certharness.graders.llm_judge import RUBRIC, JudgeError, LLMJudge, render_transcript
from certharness.llm import Budget, LLMClient
from certharness.mutate import mutate_all
from certharness.synthetic import generate


def test_transcript_hides_ids_and_metadata(mutants):
    for m in mutants[:30]:
        text = render_transcript(m)
        assert m.id not in text and m.mutation.operator not in text
        for _, c in m.tool_calls():
            assert c.id not in text


def test_transcript_marks_errors(mutants):
    m = next(m for m in mutants if m.mutation.operator == "false_success")
    assert "[ERROR]" in render_transcript(m)


def _judge(tmp_path, verdict_fn, stop="end_turn"):
    def transport(params):
        assert params["system"][0]["text"] == RUBRIC
        assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
        body = params["messages"][0]["content"]
        out = {"analysis": "checked", "passed": verdict_fn(body)}
        return {"model": params["model"], "stop_reason": stop,
                "usage": {"input_tokens": 900, "output_tokens": 40},
                "content": [{"type": "text", "text": json.dumps(out)}]}  # fmt: skip

    return LLMJudge(LLMClient(Budget(5.0, tmp_path / "l.jsonl"), tmp_path / "c", transport))


def test_judge_grades_and_certifies(tmp_path):
    originals = generate(8, seed=6)
    mutants = mutate_all(originals, seed=6)
    judge = _judge(tmp_path, lambda body: "[ERROR]" not in body)  # catches only false_success
    judge.prefetch([*originals, *mutants], workers=4)
    r = certify(judge, originals, mutants)
    stats = {s.operator: s.catch_rate for s in r.operators}
    assert stats["false_success"] == 1.0 and stats["drop_step"] == 0.0
    assert r.validity.false_alarm_rate == 0.0


def test_judge_rejects_truncated_output(tmp_path):
    judge = _judge(tmp_path, lambda body: True, stop="max_tokens")
    with pytest.raises(JudgeError, match="max_tokens"):
        judge.grade(generate(1)[0])


def test_judge_request_has_no_sampling_params():
    # claude-haiku-4-5 rejects temperature ("Extra inputs are not permitted"); see v0.1.1 note.
    params = LLMJudge(client=None).request(generate(1)[0])
    assert not {"temperature", "top_p", "top_k"} & set(params)
