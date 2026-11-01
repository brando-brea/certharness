"""certharness CLI: synth -> mutate -> certify."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from certharness.certify import CertificationReport, certify
from certharness.core.loaders import available_loaders, get_loader, load_labels, write_jsonl
from certharness.graders import GRADERS, get_grader
from certharness.mutate import OPERATORS, mutate_all
from certharness.synthetic import generate

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

NOT_REPORTABLE = (
    "NOT REPORTABLE: {reasons}.\n"
    "These numbers exercise the pipeline only. Never quote them as a grader-validity result."
)


@app.command()
def synth(
    out: Annotated[Path, typer.Option(help="Output JSONL of passing trajectories.")],
    n: Annotated[int, typer.Option(min=1)] = 48,
    seed: int = 0,
) -> None:
    """Generate synthetic passing trajectories (development data, never reportable)."""
    count = write_jsonl(generate(n, seed), out)
    typer.echo(f"wrote {count} synthetic trajectories -> {out}")


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
) -> None:
    """Certify graders against labels (RQ1) and seeded mutants (RQ2)."""
    origs = get_loader(loader)(originals)
    muts = get_loader("jsonl")(mutants)
    labs = load_labels(labels) if labels else None
    reports = [certify(get_grader(g), origs, muts, labs) for g in (grader or sorted(GRADERS))]

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
    typer.echo("graders:   " + ", ".join(sorted(GRADERS)))
    typer.echo("loaders:   " + ", ".join(available_loaders()))
    typer.echo("operators:")
    for op in OPERATORS.values():
        typer.echo(f"  {op.name:<20} {op.severity:<9} {op.description}")


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
        f"  adequacy (n={r.n_mutants}): mutation_score={_fmt(r.mutation_score)} "
        f"critical_false_pass={_fmt(r.critical_false_pass_rate)} (n_critical={r.n_critical})"
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
        sev = next(s.severity for s in stats if s)
        cells = "".join(f"{(f'{s.catch_rate:.2f} ({s.n})' if s else '-'):>{width}}" for s in stats)
        typer.echo(f"  {op:<20}{sev:<10}{cells}")


if __name__ == "__main__":
    app()
