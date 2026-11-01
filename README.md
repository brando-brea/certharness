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

### Real agent runs and the LLM judge (calls the Claude API)

```bash
export ANTHROPIC_API_KEY=...       # or: key file at ~/.config/certharness/anthropic_api_key
                                   # (chmod 600, outside the repo; used when the env var is unset)
uv run certharness preflight --smoke   # key, git-ignores, no secrets on disk, freeze,
                                       # model ids, and one tiny real request per role
uv run certharness run-agent --out runs/m2/agent.jsonl
uv run certharness label-provisional runs/m2/agent.jsonl \
    --labels-out runs/m2/labels_provisional.json \
    --passing-out runs/m2/passing.jsonl --rejected-out runs/m2/natural_failures.jsonl
uv run certharness mutate runs/m2/passing.jsonl --out runs/m2/mutants.jsonl
uv run certharness certify --originals runs/m2/passing.jsonl --mutants runs/m2/mutants.jsonl \
    --labels runs/m2/labels_provisional.json \
    --grader tool_sequence --grader outcome --grader llm_judge --out-dir runs/m2
uv run certharness spend           # tokens, cache use, cost, latency per purpose/model
```

- **Hard spend cap.** Every API call first reserves its worst-case cost against
  `--budget-usd` (default $20). If that would exceed the cap, the call is never sent
  and the run aborts. The cap covers the whole ledger (`runs/ledger.jsonl`), not just
  one process.
- **Everything is cached to disk** (`runs/cache/`), keyed by the full request. Reruns
  are free and reproduce the same results, even for models that can't be run
  deterministically.
- **No secrets on disk.** The API key stays inside the SDK client and never appears
  in request params. Only a whitelist of response-body fields is cached (no headers),
  and the ledger stores counts and costs. Caches and ledgers are git-ignored wherever
  they're written. `preflight` checks all of this.
- **Pinned models** live in `src/certharness/config.py`: the agent is
  `claude-sonnet-5` at effort `low` and the judge is `claude-haiku-4-5`. `preflight`
  confirms both ids against the Models API before any tokens are spent.
- **The toy agent** works against a simulated contacts/calendar/email/files/weather
  backend. Runs are single-model, with no server-side fallbacks. `--error-rate`
  (default 0.25) injects one transient tool failure into that fraction of tasks. Every
  error result is recorded per trajectory as `injected` or `natural`.
- **The judge** sees only the instruction and a transcript, never ids, the task spec,
  or mutation records.
- **Provisional labels** come from code-grader consensus and are tagged `stub`.
  Passing runs become mutation parents. Rejected runs are kept, never mutated, as
  *natural failures* for human labeling (RQ1). The labels are circular for the code
  graders and never reportable.
- **Run metadata.** Every live command writes a `*.meta.json` / `run_meta.json`
  recording the freeze tag and commit, model ids, prices date, SDK version, settings,
  and spend.

## Grader freeze

Graders, the LLM-judge rubric and request, task expected answers and certification
gates are frozen at a git tag (paths listed in `src/certharness/config.py`). Any command
that touches real agent or benchmark data refuses to run if those paths differ from the
current tag, whether through committed or uncommitted edits. A change needs a new tag
and an entry in the log below.

| Tag | What changed | Data affected |
|---|---|---|
| `graders-v0.1-frozen` (`a20be01`) | Initial freeze | — |
| `graders-v0.1.1-frozen` | Removed `temperature=0` from the judge request. The API rejects sampling parameters for `claude-haiku-4-5` ("Extra inputs are not permitted"). | None. Made before any judge verdict existed. Code graders are byte-identical to v0.1, so the agent run, provisional labels and mutants made under v0.1 stay valid. |

**Pre-freeze tuning, made before any agent data existed:** two task expected-answer
patterns were loosened after looking at synthetic data, because they rejected
harmless formatting. `weather` accepts `14 °C` / `14 degrees`, not only `14°C`.
`next_meeting` requires the date and time to appear anywhere in the reply, not as the
literal phrase "DATE at TIME". The agent's system prompt also asks for ISO dates,
24-hour times and returned ids, as a product spec. The judge rubric was written once,
before the freeze, without seeing any agent output.

## Research notes

- **RQ4: judge caching depends on model.** The judge rubric is marked for prompt
  caching, but `claude-haiku-4-5` only caches prefixes of at least 4096 tokens, and
  the rubric is far shorter, so on Haiku it never caches. `claude-sonnet-5` caches
  from 1024 tokens. Haiku without caching is still cheaper per verdict than Sonnet
  with caching, so cost per certified verdict has to be measured, not assumed from
  cache hit rates. `certharness spend` records the actual cache reads.

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
- **Pre-freeze tuning:** see [Grader freeze](#grader-freeze). Expected-answer patterns
  were adjusted on synthetic data before freezing. No changes after the freeze go
  untagged.
- **Judge verdicts are sampled, not greedy (post-freeze change 1, v0.1.1).** The judge
  model doesn't accept `temperature`, so each verdict is one sample from its default
  distribution. Results are reproducible because verdicts are cached, but a re-run
  without the cache could flip borderline verdicts. Verdict stability will be measured
  by re-grading a sample with the cache bypassed.
- **Same-family judging:** a Claude judge grading a Claude agent may share its blind
  spots. A cross-family judge is a planned comparison.
- **Judge–operator leakage:** the judge rubric describes task success in general terms
  and isn't written around the mutation operators. Tuning it to the planted faults
  would inflate catch rates. Changes to the rubric are tracked in git.
- **Parent selection bias:** until human labels exist, the trajectories that get
  mutated are the ones the code graders agree passed. That skews the parent set toward
  trajectories code graders like.
- **Point estimates overstate small samples (a preview of RQ3).** "0 misses out of 50"
  doesn't show a miss rate under 2%. So the critical-false-pass gate uses the one-sided
  95% Clopper–Pearson *upper bound*, which must be ≤ 2%. That bound is out of reach
  below **149 critical mutants even with zero misses** (the rule of three: 3/0.02 ≈ 150).
  One miss needs 236, two need 313. `certify` prints the required n and marks the gate
  unachievable below it, so suite size becomes a design decision instead of an
  afterthought. A second, hard gate requires **zero misses on *obvious* critical
  mutants** (faults visible in a single step, such as calling a forbidden tool, or
  claiming success right after a tool error). Below 236 critical mutants that gate is
  redundant, because one miss of any kind already breaks the bound. Above 236 it's
  what stops a grader from "spending" its allowed miss on a blatant fault. One caveat:
  the bound assumes independent trials, but mutants that share a parent trajectory are
  correlated, so the true uncertainty is somewhat larger than reported.

## Independence

This is a personal project, built entirely on my own time and equipment. It contains
no employer data, no employer code, and uses no employer credentials or API keys.
All data in this repository is synthetic or comes from public sources.

## License

MIT. See [LICENSE](LICENSE).
