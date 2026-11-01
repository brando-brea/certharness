"""certharness CLI: (synth | run-agent) -> mutate -> certify."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from certharness import config
from certharness.certify import CertificationReport, certify
from certharness.core.loaders import available_loaders, get_loader, load_labels, write_jsonl
from certharness.core.types import REPORTABLE_TRAJECTORIES, Label, Provenance
from certharness.graders import GRADERS, OutcomeGrader, ToolSequenceGrader, get_grader
from certharness.graders.llm_judge import DEFAULT_JUDGE_MODEL, LLMJudge
from certharness.llm import PRICES_AS_OF, Budget, BudgetExceeded, LLMClient, read_ledger
from certharness.mutate import OPERATORS, mutate_all
from certharness.runmeta import FreezeError, freeze_status, require_frozen, write_meta
from certharness.synthetic import generate
from certharness.toyagent import DEFAULT_EFFORT, DEFAULT_MODEL, run_agent

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

BudgetUSD = Annotated[float, typer.Option(help="Hard spend cap, enforced across runs.")]
Ledger = Annotated[Path, typer.Option(help="Spend ledger (JSONL); the cap covers its total.")]
CacheDir = Annotated[Path, typer.Option(help="Disk cache of API responses.")]
FreezeTag = Annotated[str, typer.Option(help="Grader-freeze tag required for real-data runs.")]

NOT_REPORTABLE = (
    "NOT REPORTABLE: {reasons}.\n"
    "These numbers exercise the pipeline only. Never quote them as a grader-validity result."
)


@app.command()
def synth(
    out: Annotated[Path, typer.Option(help="Output JSONL of passing trajectories.")],
    n: Annotated[int, typer.Option(min=1)] = 64,
    seed: int = 0,
) -> None:
    """Generate synthetic passing trajectories (development data, never reportable)."""
    count = write_jsonl(generate(n, seed), out)
    typer.echo(f"wrote {count} synthetic trajectories -> {out}")


def _frozen(tag: str) -> dict:
    try:
        status = require_frozen(tag)
    except FreezeError as e:
        typer.secho(f"REFUSED: {e}", fg="red", bold=True)
        raise typer.Exit(3) from e
    typer.echo(f"grader freeze OK: {tag} ({status['tag_commit'][:10]}), HEAD {status['head'][:10]}")
    return status


def _client(budget_usd: float, ledger: Path, cache_dir: Path) -> LLMClient:
    budget = Budget(budget_usd, ledger)
    typer.echo(f"budget: ${budget.spent_usd:.4f} of ${budget_usd:.2f} spent so far ({ledger})")
    return LLMClient(budget, cache_dir)


@app.command("run-agent")
def run_agent_cmd(
    out: Annotated[Path, typer.Option(help="Output JSONL of agent trajectories.")],
    n: Annotated[int, typer.Option(min=1, help="Number of tasks.")] = 96,
    seed: int = 0,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    error_rate: Annotated[
        float, typer.Option(help="Fraction of tasks with one injected transient tool error.")
    ] = 0.25,
    workers: int = 4,
    budget_usd: BudgetUSD = 20.0,
    ledger: Ledger = Path("runs/ledger.jsonl"),
    cache_dir: CacheDir = Path("runs/cache"),
    freeze_tag: FreezeTag = config.FREEZE_TAG,
) -> None:
    """Run the toy agent on n tasks (calls the Claude API; budgeted and cached)."""
    freeze = _frozen(freeze_tag)
    client = _client(budget_usd, ledger, cache_dir)
    tasks = [t.task for t in generate(n, seed)]
    try:
        trajs = run_agent(client, tasks, model, effort, error_rate, seed, workers)
    except BudgetExceeded as e:
        typer.secho(f"ABORTED: {e}", fg="red", bold=True)
        raise typer.Exit(2) from e
    write_jsonl(trajs, out)
    stops: dict[str, int] = {}
    errors = {"injected": 0, "natural": 0}
    for t in trajs:
        agent = t.task.metadata["agent"]
        stop = str(agent.get("stop"))
        stops[stop] = stops.get(stop, 0) + 1
        for e in agent["tool_errors"]:
            errors["injected" if e["injected"] else "natural"] += 1
    meta_path = out.with_suffix(".meta.json")
    write_meta(meta_path, freeze, command="run-agent", n=n, seed=seed, model=model,
               effort=effort, error_rate=error_rate, budget_usd=budget_usd,
               spent_usd_after=client.budget.spent_usd, stop_reasons=stops,
               tool_errors=errors)  # fmt: skip
    typer.echo(f"wrote {len(trajs)} agent trajectories -> {out}  (stop reasons: {stops})")
    typer.echo(f"tool errors: {errors}; run metadata -> {meta_path}")
    typer.echo(f"spent so far: ${client.budget.spent_usd:.4f}")


@app.command("label-provisional")
def label_provisional(
    trajectories: Annotated[Path, typer.Argument(help="Agent trajectories.")],
    labels_out: Annotated[Path, typer.Option(help="Provisional labels JSON.")],
    passing_out: Annotated[Path, typer.Option(help="JSONL of provisionally passing runs.")],
    rejected_out: Annotated[
        Path, typer.Option(help="JSONL of rejected runs (natural failures, for human labels).")
    ],
    freeze_tag: FreezeTag = config.FREEZE_TAG,
) -> None:
    """Split agent runs by code-grader consensus (STUB labels, never reportable).

    Passing runs become mutation parents. Rejected runs are kept, unmutated, as
    "natural failure" trajectories for human labeling (RQ1). Because the code graders
    produced these labels, their own validity scores on them are circular.
    """
    freeze = _frozen(freeze_tag)
    trajs = get_loader("jsonl")(trajectories)
    graders = [ToolSequenceGrader(), OutcomeGrader()]
    labels, passing, rejected = [], [], []
    for t in trajs:
        verdicts = [g.grade(t) for g in graders]
        ok = all(v.passed for v in verdicts)
        injected = bool(t.task.metadata.get("agent", {}).get("injected_errors"))
        why = "; ".join(f"{v.grader}: {', '.join(v.reasons)}" for v in verdicts if not v.passed)
        note = "provisional: code-grader consensus" + (f" (rejected: {why})" if why else "")
        labels.append(Label(trajectory_id=t.id, passed=ok, provenance=Provenance.STUB, note=note))
        pool = {"pool": "provisional_pass" if ok else "natural_failure",
                "injected_error_present": injected}  # fmt: skip
        tagged = t.model_copy(update={"task": t.task.model_copy(
            update={"metadata": {**t.task.metadata, **pool}})})  # fmt: skip
        (passing if ok else rejected).append(tagged)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.write_text("[\n" + ",\n".join(lab.model_dump_json() for lab in labels) + "\n]\n")
    write_jsonl(passing, passing_out)
    write_jsonl(rejected, rejected_out)
    n_inj = sum(t.task.metadata["injected_error_present"] for t in rejected)
    write_meta(labels_out.with_suffix(".meta.json"), freeze, command="label-provisional",
               n=len(trajs), passing=len(passing), rejected=len(rejected),
               rejected_with_injected_error=n_inj)  # fmt: skip
    typer.echo(f"{len(passing)}/{len(trajs)} provisionally passing -> {passing_out}")
    typer.echo(
        f"{len(rejected)} natural failures ({n_inj} with an injected error) -> {rejected_out}"
    )
    typer.echo(f"labels -> {labels_out}")


@app.command()
def preflight(
    ledger: Ledger = Path("runs/ledger.jsonl"),
    cache_dir: CacheDir = Path("runs/cache"),
    freeze_tag: FreezeTag = config.FREEZE_TAG,
    check_models: Annotated[bool, typer.Option(help="Verify pinned model ids live.")] = True,
    smoke: Annotated[
        bool,
        typer.Option(
            help="Send one real agent and one real judge request (max_tokens=16) "
            "to validate the exact request shapes. Costs a fraction of a cent."
        ),
    ] = False,
    budget_usd: BudgetUSD = 20.0,
) -> None:
    """Checks before a live run: key present, outputs git-ignored, no secrets on disk,
    grader freeze intact, pinned model ids valid (Models API; no token spend)."""
    import subprocess

    from certharness.runmeta import REPO_ROOT

    failures = 0

    def report(ok: bool, msg: str) -> None:
        nonlocal failures
        failures += not ok
        typer.secho(f"  [{'ok' if ok else 'FAIL'}] {msg}", fg=None if ok else "red")

    from certharness.llm.credentials import client_kwargs, key_file_private, key_source

    source = key_source()
    report(source is not None, f"API key available from {source or 'nowhere'} (value not shown)")
    if source not in (None, "env"):
        report(key_file_private(), f"key file {source} is owner-only (chmod 600)")
        looks_ok = client_kwargs().get("api_key", "").startswith("sk-ant-api")
        report(looks_ok, "key file holds an API key (sk-ant-api...)")
        git_ls = ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", source]
        tracked = subprocess.run(git_ls, capture_output=True).returncode == 0
        report(not tracked, "key file is not tracked by git")
    for target in (cache_dir / "ab" / "x.json", ledger, Path("runs/m2/agent.meta.json")):
        ignored = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", str(target)])
        report(ignored.returncode == 0, f"{target} is git-ignored")
    needles = ("sk-ant", "x-api-key", "authorization", "api_key")
    leaks = [
        str(f) for f in [ledger, *cache_dir.rglob("*.json")] if f.exists()
        and any(n in f.read_text(encoding="utf-8").lower() for n in needles)
    ]  # fmt: skip
    report(not leaks, f"no credentials/headers in ledger or cache ({len(leaks)} suspicious files)")
    fs = freeze_status(freeze_tag)
    report(fs["ok"], f"grader freeze {freeze_tag}: {'intact' if fs['ok'] else fs['reason']}")
    if check_models:
        try:
            import anthropic

            client = anthropic.Anthropic(**client_kwargs())
            for model in (config.AGENT_MODEL, config.JUDGE_MODEL):
                info = client.models.retrieve(model)
                # An alias may resolve to a dated snapshot id (e.g. claude-haiku-4-5-2025...).
                resolved = info.id == model or info.id.startswith(model + "-")
                shown = model if info.id == model else f"{model} -> {info.id}"
                report(resolved, f"model {shown} exists ({info.display_name})")
        except Exception as e:  # noqa: BLE001 - surface any failure as a failed check
            report(False, f"model check failed: {type(e).__name__}: {e}")
    if smoke:
        from certharness.toyagent.agent import agent_request, tools_for

        client = LLMClient(Budget(budget_usd, ledger), cache_dir)
        sample = generate(1, seed=0)[0]
        msgs = [{"role": "user", "content": sample.task.instruction}]
        tools = tools_for(sample.task)
        requests = {
            "agent": agent_request(config.AGENT_MODEL, config.AGENT_EFFORT, tools, msgs),
            "judge": LLMJudge(client, config.JUDGE_MODEL).request(sample),
        }
        for purpose, params in requests.items():
            try:
                r = client.create(f"smoke-{purpose}", **{**params, "max_tokens": 16})
                report(True, f"smoke {purpose} request accepted ({r.get('stop_reason')})")
            except Exception as e:  # noqa: BLE001 - surface any failure as a failed check
                report(False, f"smoke {purpose} request rejected: {type(e).__name__}: {e}")
    if failures:
        typer.secho(f"{failures} check(s) failed", fg="red", bold=True)
        raise typer.Exit(1)
    typer.secho("all preflight checks passed", fg="green", bold=True)


_SPEND_FIELDS = ["live", "cached", "in", "out", "cache_w", "cache_r", "cost", "latency"]


@app.command()
def spend(ledger: Ledger = Path("runs/ledger.jsonl")) -> None:
    """Summarize the spend ledger: calls, tokens, cache use and cost by purpose/model."""
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for e in read_ledger(ledger):
        r = rows.setdefault((e["purpose"], e["model"]), dict.fromkeys(_SPEND_FIELDS, 0.0))
        r["cached" if e.get("cached") else "live"] += 1
        r["in"] += e.get("input_tokens", 0)
        r["out"] += e.get("output_tokens", 0)
        r["cache_w"] += e.get("cache_creation_input_tokens", 0)
        r["cache_r"] += e.get("cache_read_input_tokens", 0)
        r["cost"] += e.get("cost_usd", 0.0)
        r["latency"] += 0 if e.get("cached") else e.get("latency_s", 0.0)
    typer.echo(f"prices as of {PRICES_AS_OF}")
    head = f"{'purpose':<8}{'model':<20}{'live':>6}{'cached':>7}{'in':>9}{'out':>8}"
    typer.echo(head + f"{'cache_w':>9}{'cache_r':>9}{'cost $':>10}{'s/call':>8}")
    total = 0.0
    for (purpose, model), r in sorted(rows.items()):
        per = r["latency"] / r["live"] if r["live"] else 0.0
        typer.echo(
            f"{purpose:<8}{model:<20}{r['live']:>6.0f}{r['cached']:>7.0f}{r['in']:>9.0f}"
            f"{r['out']:>8.0f}{r['cache_w']:>9.0f}{r['cache_r']:>9.0f}{r['cost']:>10.4f}{per:>8.2f}"
        )
        total += r["cost"]
    typer.echo(f"total: ${total:.4f}")


@app.command()
def mutate(
    originals: Annotated[Path, typer.Argument(help="Trajectories to mutate.")],
    out: Annotated[Path, typer.Option(help="Output JSONL of mutants.")],
    seed: int = 0,
    operator: Annotated[
        list[str] | None, typer.Option(help="Restrict to these operators (repeatable).")
    ] = None,
    loader: Annotated[str, typer.Option(help="Trajectory loader for the input.")] = "jsonl",
    replay: Annotated[
        bool, typer.Option(help="Re-run mutated tool calls against the simulated world.")
    ] = True,
) -> None:
    """Plant one seeded fault per (trajectory, operator) pair."""
    for name in operator or []:
        if name not in OPERATORS:
            raise typer.BadParameter(f"unknown operator {name!r}; available: {sorted(OPERATORS)}")
    trajs = get_loader(loader)(originals)
    mutants = mutate_all(trajs, seed=seed, operators=operator, replay=replay)
    write_jsonl(mutants, out)
    by_op: dict[str, int] = {}
    for m in mutants:
        by_op[m.mutation.operator] = by_op.get(m.mutation.operator, 0) + 1
    n_replayed = sum(m.mutation.replayed for m in mutants)
    typer.echo(f"wrote {len(mutants)} mutants from {len(trajs)} trajectories -> {out}"
               f" ({n_replayed} replayed against the simulated world)")  # fmt: skip
    for name, k in sorted(by_op.items()):
        typer.echo(f"  {name:<20} {k}")


@app.command("certify")
def certify_cmd(
    originals: Annotated[Path, typer.Option(help="Original (unmutated) trajectories.")],
    mutants: Annotated[Path, typer.Option(help="Mutants from `certharness mutate`.")],
    grader: Annotated[
        list[str] | None, typer.Option(help="Grader(s) to certify (repeatable). Default: all.")
    ] = None,
    labels: Annotated[Path | None, typer.Option(help="Ground-truth labels JSON.")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Write one report JSON per grader.")] = None,
    loader: Annotated[str, typer.Option(help="Trajectory loader for --originals.")] = "jsonl",
    judge_model: Annotated[str, typer.Option(help="Model for --grader llm_judge.")] = (
        DEFAULT_JUDGE_MODEL
    ),
    workers: int = 8,
    budget_usd: BudgetUSD = 20.0,
    ledger: Ledger = Path("runs/ledger.jsonl"),
    cache_dir: CacheDir = Path("runs/cache"),
    freeze_tag: FreezeTag = config.FREEZE_TAG,
) -> None:
    """Certify graders against labels (RQ1) and seeded mutants (RQ2).

    Code graders run by default; add `--grader llm_judge` for the LLM judge (calls the
    Claude API; budgeted and cached).
    """
    origs = get_loader(loader)(originals)
    muts = get_loader("jsonl")(mutants)
    labs = load_labels(labels) if labels else None
    real = any(t.provenance in REPORTABLE_TRAJECTORIES for t in [*origs, *muts])
    freeze = _frozen(freeze_tag) if real else None
    if "llm_judge" in (grader or []) and out_dir is None:
        raise typer.BadParameter("--out-dir is required with --grader llm_judge")
    graders = []
    for g in grader or sorted(GRADERS):
        if g == "llm_judge":
            judge = LLMJudge(_client(budget_usd, ledger, cache_dir), judge_model)
            try:
                judge.prefetch([*origs, *muts], workers)
            except BudgetExceeded as e:
                typer.secho(f"ABORTED: {e}", fg="red", bold=True)
                raise typer.Exit(2) from e
            typer.echo(f"judge spend so far: ${judge.client.budget.spent_usd:.4f}")
            graders.append(judge)
        else:
            graders.append(get_grader(g))
    reports = [certify(g, origs, muts, labs) for g in graders]

    if not reports[0].reportable:
        reasons = [r for r in reports[0].shadow_reasons if not r.startswith("gate ")]
        typer.secho(NOT_REPORTABLE.format(reasons="; ".join(reasons)), fg="yellow", bold=True)
        typer.echo()

    for r in reports:
        _print_report(r)
    _print_matrix(reports)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in reports:
            (out_dir / f"certify_{r.grader}.json").write_text(r.model_dump_json(indent=2))
        write_meta(out_dir / "run_meta.json", freeze, command="certify",
                   graders=[g.name for g in graders], originals=str(originals),
                   mutants=str(mutants), labels=str(labels) if labels else None,
                   judge_model=judge_model if "llm_judge" in (grader or []) else None,
                   reportable=reports[0].reportable)  # fmt: skip
        typer.echo(f"\nreports -> {out_dir}")


@app.command()
def grade(
    trajectories: Annotated[Path, typer.Argument(help="Trajectories to grade.")],
    out: Annotated[Path, typer.Option(help="Output JSONL of verdicts.")],
    grader: Annotated[list[str] | None, typer.Option(help="Grader(s) (repeatable).")] = None,
    judge_model: Annotated[str, typer.Option(help="Model for --grader llm_judge.")] = (
        DEFAULT_JUDGE_MODEL
    ),
    workers: int = 8,
    budget_usd: BudgetUSD = 20.0,
    ledger: Ledger = Path("runs/ledger.jsonl"),
    cache_dir: CacheDir = Path("runs/cache"),
    freeze_tag: FreezeTag = config.FREEZE_TAG,
) -> None:
    """Run graders over trajectories and write verdicts (no labels, no certification)."""
    trajs = get_loader("jsonl")(trajectories)
    real = any(t.provenance in REPORTABLE_TRAJECTORIES for t in trajs)
    freeze = _frozen(freeze_tag) if real else None
    names = grader or sorted(GRADERS)
    verdicts = []
    for g in names:
        if g == "llm_judge":
            judge = LLMJudge(_client(budget_usd, ledger, cache_dir), judge_model)
            try:
                judge.prefetch(trajs, workers)
            except BudgetExceeded as e:
                typer.secho(f"ABORTED: {e}", fg="red", bold=True)
                raise typer.Exit(2) from e
            impl = judge
        else:
            impl = get_grader(g)
        vs = [impl.grade(t.blinded()) for t in trajs]
        verdicts += vs
        typer.echo(f"{impl.name}: passed {sum(v.passed for v in vs)}/{len(vs)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(v.model_dump_json() + "\n" for v in verdicts))
    write_meta(out.with_suffix(".meta.json"), freeze, command="grade", graders=names,
               trajectories=str(trajectories), judge_model=judge_model)  # fmt: skip
    typer.echo(f"verdicts -> {out}")


@app.command("list")
def list_cmd() -> None:
    """Show available graders, mutation operators and trajectory loaders."""
    typer.echo("graders:   " + ", ".join([*sorted(GRADERS), "llm_judge"]))
    typer.echo("loaders:   " + ", ".join(available_loaders()))
    typer.echo("operators:")
    for op in OPERATORS.values():
        typer.echo(f"  {op.name:<20} {op.severity:<9} {op.tier:<8} {op.description}")


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _print_report(r: CertificationReport) -> None:
    color = "green" if r.status == "certified" else "yellow"
    typer.secho(f"== {r.grader}: {r.status.upper()}", fg=color, bold=True)
    v = r.validity
    typer.echo(
        f"  validity (n={v.n}): acc={_fmt(v.accuracy)} kappa={_fmt(v.kappa)} "
        f"false_alarm={_fmt(v.false_alarm_rate)} false_pass={_fmt(v.false_pass_rate)}"
    )
    typer.echo(
        f"  adequacy (n={r.n_mutants}): mutation_score={_fmt(r.mutation_score)}; "
        f"critical missed {r.critical_misses}/{r.n_critical} "
        f"(point={_fmt(r.critical_false_pass_rate)}, CP upper={_fmt(r.critical_fp_upper)}, "
        f"n required at 0 misses={r.critical_n_required}); "
        f"obvious missed {r.obvious_critical_misses}/{r.n_obvious_critical}"
    )
    for g in r.gates:
        mark = {True: "pass", False: "FAIL", None: "n/a "}[g.passed]
        note = f"  ({g.note})" if g.note else ""
        typer.echo(f"  [{mark}] {g.name:<26} {_fmt(g.value):>6} vs {g.threshold}{note}")
    typer.echo()


def _print_matrix(reports: list[CertificationReport]) -> None:
    """Operator x grader catch rates: which planted faults each grader can see (RQ2)."""
    ops = sorted({s.operator for r in reports for s in r.operators})
    if not ops:
        return
    width = max(len(r.grader) for r in reports) + 2
    typer.echo("catch rate by operator:")
    typer.echo(f"  {'operator':<20}{'sev':<10}" + "".join(f"{r.grader:>{width}}" for r in reports))
    for op in ops:
        stats = [{s.operator: s for s in r.operators}.get(op) for r in reports]
        sev = next(f"{s.severity[:4]}/{s.tier[:3]}" for s in stats if s)
        cells = "".join(f"{(f'{s.catch_rate:.2f} ({s.n})' if s else '-'):>{width}}" for s in stats)
        typer.echo(f"  {op:<20}{sev:<10}{cells}")


if __name__ == "__main__":
    app()
