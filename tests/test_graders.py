import json

from certharness.core.types import Step
from certharness.graders import OutcomeGrader, ToolSequenceGrader, get_grader


def test_all_synthetic_originals_pass_both_graders(originals):
    # The generator is correct by construction; a grader failing one is a false alarm.
    for g in (ToolSequenceGrader(), OutcomeGrader()):
        failed = [(t.id, v.reasons) for t in originals if not (v := g.grade(t)).passed]
        assert not failed, f"{g.name} false alarms: {failed}"


def test_tool_sequence_rejects_out_of_order(originals):
    t = next(t for t in originals if t.task.metadata["template"] == "schedule_meeting")
    s = t.model_copy(deep=True)
    s.steps[1:3], s.steps[3:5] = s.steps[3:5], s.steps[1:3]  # create_event before lookup
    v = ToolSequenceGrader().grade(s)
    assert not v.passed and "missing required call" in v.reasons[0]


def test_tool_sequence_ignores_extra_allowed_calls(originals):
    t = next(t for t in originals if t.task.metadata["template"] == "weather")
    s = t.model_copy(deep=True)
    extra = s.steps[1].model_copy(deep=True)
    extra.tool_calls[0].id = "c9"
    s.steps[1:1] = [extra, Step(role="tool", tool_call_id="c9", content="{}")]
    assert ToolSequenceGrader().grade(s).passed


def test_outcome_accepts_acknowledged_error(originals):
    t = next(t for t in originals if t.task.metadata["template"] == "weather")
    s = t.model_copy(deep=True)
    s.steps[2].is_error = True
    s.steps[2].content = json.dumps({"error": "unavailable"})
    s.steps[-1].content = "Sorry, the weather service failed, so I couldn't get the forecast."
    v = OutcomeGrader().grade(s)
    # Acknowledging the error is honest, but the expected value is still missing.
    assert v.reasons == [f"final answer does not match {t.task.expected_final!r}"]


def test_outcome_requires_final_answer(originals):
    s = originals[0].model_copy(deep=True)
    s.steps = s.steps[:-1]
    assert OutcomeGrader().grade(s).reasons == ["no final answer"]


def test_get_grader():
    assert get_grader("outcome").name == "outcome"
