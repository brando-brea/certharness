from certharness.core.grader import Grader
from certharness.graders.code import OutcomeGrader, ToolSequenceGrader

GRADERS: dict[str, type] = {
    ToolSequenceGrader.name: ToolSequenceGrader,
    OutcomeGrader.name: OutcomeGrader,
}


def get_grader(name: str) -> Grader:
    try:
        return GRADERS[name]()
    except KeyError:
        raise KeyError(f"unknown grader {name!r}; available: {sorted(GRADERS)}") from None


__all__ = ["GRADERS", "OutcomeGrader", "ToolSequenceGrader", "get_grader"]
