"""Pinned experiment configuration.

Model ids come from Anthropic's published model list (as cached on MODEL_IDS_AS_OF);
`certharness preflight` re-verifies them live against the Models API before any
spend. Change a pin only with a note in the run metadata.
"""

AGENT_MODEL = "claude-sonnet-5"
AGENT_EFFORT = "low"
JUDGE_MODEL = "claude-haiku-4-5"
MODEL_IDS_AS_OF = "2026-06-24"

# Grader freeze: runs on real (agent/external) data refuse to start unless these paths
# are identical to FREEZE_TAG. Changing them requires a new tag and a threats-to-validity
# note in the README.
FREEZE_TAG = "graders-v0.1.1-frozen"  # v0.1.1: judge request without temperature
FROZEN_PATHS = (
    "src/certharness/graders",  # code graders and the LLM-judge rubric
    "src/certharness/synthetic.py",  # task specs, including expected answers
    "src/certharness/certify",  # metrics and gates
)
