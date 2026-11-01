"""certharness CLI: (synth | run-agent) -> mutate -> certify."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from certharness.certify import CertificationReport, certify
from certharness.core.loaders import available_loaders, get_loader, load_labels, write_jsonl
from certharness.core.types import Label, Provenance
from certharness.graders import GRADERS, OutcomeGrader, ToolSequenceGrader, get_grader
from certharness.graders.llm_judge import DEFAULT_JUDGE_MODEL, LLMJudge
from certharness.llm import PRICES_AS_OF, Budget, BudgetExceeded, LLMClient, read_ledger
from certharness.mutate import OPERATORS, mutate_all
from certharness.synthetic import generate
from certharness.toyagent import DEFAULT_EFFORT, DEFAULT_MODEL, run_agent

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

BudgetUSD = Annotated[float, typer.Option(help="Hard spend cap, enforced across runs.")]
Ledger = Annotated[Path, typer.Option(help="Spend ledger (JSONL); the cap covers its total.")]
CacheDir = Annotated[Path, typer.Option(help="Disk cache of API responses.")]

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


def _client(budget_usd: float, ledger: Path, cache_dir: Path) -> LLMClient:
    budget = Budget(budget_usd, ledger)
    typer.echo(f"budget: ${budget.spent_usd:.4f} of ${budget_usd:.2f} spent so far ({ledger})")
    return LLMClient(budget, cache_dir)


@app.command("run-agent")
def run_agent_cmd(
    out: Annotated[Path, typer.Option(help="Output JSONL of agent trajectories.")],
    n: Annotated[int, typer.Option(min=1, help="Number of tasks.")] = 64,
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
) -> None:
    """Run the toy agent on n tasks (calls the Claude API; budgeted and cached)."""
    client = _client(budget_usd, ledger, cache_dir)
    tasks = [t.task for t in generate(n, seed)]
    try:
        trajs = run_agent(client, tasks, model, effort, error_rate, seed, workers)
    except BudgetExceeded as e:
        typer.secho(f"ABORTED: {e}", fg="red", bold=True)
        raise typer.Exit(2) from e
    write_jsonl(trajs, out)
    stops: dict[str, int] = {}
    for t in trajs:
        stop = str(t.task.metadata["agent"].get("stop"))
        stops[stop] = stops.get(stop, 0) + 1
    typer.echo(f"wrote {len(trajs)} agent trajectories -> {out}  (stop reasons: {stops})")
    typer.echo(f"spent so far: ${client.budget.spent_usd:.4f}")


@app.command("label-provisional")
def label_provisional(
    trajectories: Annotated[Path, typer.Argument(help="Agent trajectories.")],
    labels_out: Annotated[Path, typer.Option(help="Provisional labels JSON.")],
    passing_out: Annotated[Path, typer.Option(help="JSONL of provisionally passing ones.")],
) -> None:
    """Label agent runs by code-grader consensus (STUB provenance, never reportable).

    A stand-in until human labels exist: it selects parents for mutation. Because the
    code graders produced these labels, their own validity scores on them are circular.
    """
    trajs = get_loader("jsonl")(trajectories)
    graders = [ToolSequenceGrader(), OutcomeGrader()]
    labels, passing = [], []
    for t in trajs:
        ok = all(g.grade(t).passed for g in graders)
        labels.append(Label(trajectory_id=t.id, passed=ok, provenance=Provenance.STUB,
                            note="provisional: code-grader consensus"))  # fmt: skip
        if ok:
            passing.append(t)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.write_text("[\n" + ",\n".join(lab.model_dump_json() for lab in labels) + "\n]\n")
    write_jsonl(passing, passing_out)
    typer.echo(f"{len(passing)}/{len(trajs)} provisionally passing -> {passing_out}")
    typer.echo(f"labels -> {labels_out}")


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
) -> None:
    """Plant one seeded fault per (trajectory, operator) pair."""
    for name in operator or []:
        if name not in OPERATORS:
            raise typer.BadParameter(f"unknown operator {name!r}; available: {sorted(OPERATORS)}")
    trajs = get_loader(loader)(originals)
    mutants = mutate_all(trajs, seed=seed, operators=operator)
    write_jsonl(mutants, out)
    by_op: dict[str, int] = {}
    for m in mutants:
        by_op[m.mutation.operator] = by_op.get(m.mutation.operator, 0) + 1
    typer.echo(f"wrote {len(mutants)} mutants from {len(trajs)} trajectories -> {out}")
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
) -> None:
    """Certify graders against labels (RQ1) and seeded mutants (RQ2).

    Code graders run by default; add `--grader llm_judge` for the LLM judge (calls the
    Claude API; budgeted and cached).
    """
    origs = get_loader(loader)(originals)
    muts = get_loader("jsonl")(mutants)
    labs = load_labels(labels) if labels else None
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
        typer.echo(f"\nreports -> {out_dir}")


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
