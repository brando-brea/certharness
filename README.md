# certharness

**Regression testing for LLM agents, where the graders have to earn trust first.**

Most agent-eval setups treat their grader (an LLM judge or a scoring script) as ground
truth. But a grader is untested code. certharness *certifies* each grader before its
verdicts are allowed to gate anything:

1. **Validity (RQ1).** Does the grader agree with human labels? (accuracy, Cohen's κ,
   false-alarm and false-pass rates)
2. **Mutation adequacy (RQ2).** Does it catch faults planted on purpose? Seeded
   mutation operators inject known failures into passing trajectories, such as a skipped
   step, a wrong argument, a forbidden action, claiming success after a tool error, or a
   hallucinated result. A grader's *mutation score* is the fraction it catches.
   Critical false-passes (unsafe or dishonest behaviour graded as passing) have their
   own gate.

A grader that fails any gate stays in **shadow mode**: it runs and reports, but does
not gate.

> **Status: pre-release (v0.1 in development).** No results have been published yet.

## Provenance rule

Every trajectory and label records where it came from. A certification is
**reportable** only when the labels come from humans (or a public benchmark's own
labels) *and* the trajectories come from real agent runs or a public benchmark.
Synthetic and stub data exercise the pipeline, and the CLI prints a NOT REPORTABLE
banner on any result built from them. This is enforced in code, so a development
number can't be mistaken for a finding.

## Quickstart

```bash
uv sync
uv run certharness list                                    # graders, operators, loaders
uv run certharness synth  --out runs/dev/synth.jsonl       # passing toy trajectories
uv run certharness mutate runs/dev/synth.jsonl --out runs/dev/mutants.jsonl
uv run certharness certify --originals runs/dev/synth.jsonl \
                           --mutants runs/dev/mutants.jsonl --out-dir runs/dev
```

`certify` prints one report per grader plus an operator × grader catch-rate matrix.

## Layout

```
src/certharness/
  core/       Task, Trajectory, Verdict, Label, Grader protocol, loaders (+ external hook)
  graders/    code graders (tool_sequence, outcome); LLM judges come next
  mutate/     seeded, deterministic mutation operators + engine
  certify/    metrics and the certification report and gates
  synthetic   toy tool-agent domain for development data
  stats/      v0.2: bootstrap CIs, McNemar, power analysis      (stub)
  cost/       v0.2: cost per certified verdict                  (stub)
  runner/     regression runs against a baseline                (stub)
```

External benchmarks plug in through `core.loaders.register_loader`. A loader converts the
benchmark's native format into `Trajectory` objects with `provenance="external"`.

## Roadmap

| Release | Scope | Done when |
|---|---|---|
| **v0.1** | RQ1 grader validity + RQ2 mutation adequacy: code graders and LLM judges, ~50 human-labeled trajectories from two blind labeling sessions | Published with a write-up |
| **v0.2** | RQ3 statistical power (how many tasks to detect a regression of X points) + RQ4 eval cost efficiency (cascades, judge size, caching), external validity on a public benchmark subset (e.g. τ-bench or SWE-bench Lite) | Published with a second write-up |

## Threats to validity (tracked from day one)

- **Equivalent mutants:** a mutation that doesn't actually break the task inflates
  the apparent miss rate. Operators target only spec-relevant parts of a trajectory,
  and identical mutants are discarded. The residual rate gets estimated by hand.
- **Operator realism:** planted faults may be easier or harder to catch than real
  agent failures. The real failures among the human-labeled set are the check on this.
- **Single rater:** labels come from one person. Intra-rater agreement across two
  blind sessions is reported as a limitation.
- **Toy domain:** the development domain is small. v0.2 adds an external benchmark.

## Independence

This is a personal project, built entirely on my own time and equipment. It contains
no employer data, no employer code, and uses no employer credentials or API keys.
All data in this repository is synthetic or comes from public sources.

## License

MIT. See [LICENSE](LICENSE).
