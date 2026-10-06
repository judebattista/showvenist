"""Command-line interface for showvenist."""

import importlib.metadata
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import typer

TMDB_NOTICE = "This product uses the TMDB API but is not endorsed or certified by TMDB."

app = typer.Typer()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        try:
            ver = importlib.metadata.version("showvenist")
        except importlib.metadata.PackageNotFoundError:
            ver = "0.1.0"
        typer.echo(f"showvenist {ver}")
        typer.echo(TMDB_NOTICE)
        raise typer.Exit(0)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        help="Print version and exit",
    )
) -> None:
    """Identifies season and episode of TV disc rips against a reference bundle."""
    pass


bundle_app = typer.Typer(help="Inspect and edit reference bundles.")
app.add_typer(bundle_app, name="bundle")


def _tmdb():
    from showvenist.adapters.tmdb import TmdbAdapter
    from showvenist.config import load_config

    key = load_config().tmdb.api_key
    if not key:
        typer.echo("No TMDB key: set [tmdb] api_key in config.toml or SHOWVENIST_TMDB_API_KEY.",
                   err=True)
        raise typer.Exit(1)
    return TmdbAdapter(key)


def _pick(cands: list[dict]) -> int:
    for i, c in enumerate(cands[:8], 1):
        typer.echo(f"  {i}. {c['title']} ({c['year']})  tmdb:{c['tmdb_id']}")
    n = typer.prompt("Which one", type=typer.IntRange(1, min(len(cands), 8)))
    return cands[n - 1]["tmdb_id"]


@app.command()
def ingest(
    show: str = typer.Argument(..., help="Show title"),
    ref: list[str] = typer.Option([], "--ref", help="TMDB or IMDb URL/ID"),
    subs: Path | None = typer.Option(None, "--subs", help="Folder of reference .srt files"),
    year: int | None = typer.Option(None, "--year"),
    tmdb_id: int | None = typer.Option(None, "--tmdb-id"),
    refresh: bool = typer.Option(False, "--refresh", help="Replace an existing bundle"),
    out: Path | None = typer.Option(None, "-o", help="Bundle root directory"),
) -> None:
    """Build a show's reference bundle and print its data-quality report."""
    from showvenist.adapters.tmdb import TmdbError
    from showvenist.bundle.ingest import IngestError
    from showvenist.bundle.ingest import ingest as run_ingest
    from showvenist.bundle.quality import quality_report, render

    try:
        bundle, bdir, scan = run_ingest(
            _tmdb(), show, year=year, tmdb_id=tmdb_id, refs=tuple(ref), subs_dir=subs,
            root=out, pick=_pick if sys.stdin.isatty() else None, refresh=refresh,
        )
    except (IngestError, TmdbError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Wrote {bdir / 'show.json'} ({len(bundle.episodes)} episodes, "
               f"{len(bundle.orderings)} orderings)")
    if scan is not None:
        typer.echo(f"Subtitles: {len(scan.matched)} matched, {len(scan.missing)} episodes without, "
                   f"{len(scan.unmatched)} unmatched files, {len(scan.duplicates)} duplicates")
    typer.echo(render(quality_report(bundle, bdir)))


@bundle_app.command("check")
def bundle_check(
    show: str = typer.Argument(..., help="Bundle slug"),
    root: Path | None = typer.Option(None, "-o", help="Bundle root directory"),
) -> None:
    """Print the bundle's data-quality report."""
    from showvenist.bundle.overrides import BundleError
    from showvenist.bundle.quality import quality_report, render
    from showvenist.bundle.store import bundle_path, load_bundle, slugify

    if not (bundle_path(show, root) / "show.json").exists():
        show = slugify(show)  # accept a title as well as a slug
    try:
        bundle = load_bundle(show, root)
    except BundleError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from None
    typer.echo(render(quality_report(bundle, bundle_path(show, root))))


@app.command("inspect")
def inspect_cmd(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="Media file"),
    show: str | None = typer.Option(None, "--show", help="Bundle slug (vocabulary, language)"),
    language: str = typer.Option("en", "--language"),
    root: Path | None = typer.Option(None, "-o", help="Bundle root directory"),
) -> None:
    """Dump the extracted dialogue and signals for one file (debugging)."""
    from showvenist.analysis.cache import AnalysisCache
    from showvenist.analysis.inspect import inspect_file
    from showvenist.bundle.store import load_bundle
    from showvenist.config import cache_dir, load_config

    vocab: set[str] = set()
    if show:
        from showvenist.analysis.subtitles import normalize

        b = load_bundle(show, root)
        language = b.show.language
        for e in b.episodes:
            names = [*e.directors, *e.writers]
            for c in e.guest_cast:
                names += [c.actor, c.character]
            vocab.update(
                normalize(" ".join([e.title, *e.alt_titles, *(s.text for s in e.synopses), *names]))
                .split()
            )
    cache_dir().mkdir(parents=True, exist_ok=True)
    cache = AnalysisCache(cache_dir() / "analysis.db")
    typer.echo(inspect_file(file, language, vocab, load_config().analysis, cache))


def _expand(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out += sorted(p.rglob("*.mkv")) if p.is_dir() else [p]
    return out


def _resolve_bundle(show: str, root: Path | None):
    from showvenist.bundle.overrides import BundleError
    from showvenist.bundle.store import bundle_path, load_bundle, slugify

    slug = show if (bundle_path(show, root) / "show.json").exists() else slugify(show)
    try:
        return slug, load_bundle(slug, root)
    except BundleError as e:
        typer.echo(f"{e}\nRun: showvenist ingest {show!r}", err=True)
        raise typer.Exit(1) from None


SCORER_NAMES = ("duration", "title", "refsubs", "synopsis", "guests", "cues", "llm")
LLM_MODES = ("off", "auto", "all")
# Builds the Anthropic client for the LLM scorer. None = the real client (lazy `anthropic`
# import, only once a call is approved). Tests inject a fake here; nothing else should.
LLM_CLIENT_FACTORY: Callable[[], Any] | None = None


def _fail(msg: str) -> NoReturn:
    typer.echo(msg, err=True)
    raise typer.Exit(1)


def _parse_signals(text: str | None) -> set[str]:
    """`--signals duration,refsubs` keeps only those scorers; `--signals -cues,-title` drops
    those. Names: duration, title, refsubs, synopsis, guests, cues, llm."""
    if text is None:
        return set(SCORER_NAMES)
    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not items:
        _fail("--signals: empty list")
    neg = [x.startswith("-") for x in items]
    if any(neg) and not all(neg):
        _fail("--signals: either list the scorers to keep or the ones to drop (-name), not both")
    names = [x.lstrip("-") for x in items]
    unknown = sorted(set(names) - set(SCORER_NAMES))
    if unknown:
        _fail(f"--signals: unknown scorer(s) {', '.join(unknown)} "
              f"(known: {', '.join(SCORER_NAMES)})")
    return set(SCORER_NAMES) - set(names) if all(neg) else set(names)


def _scorers(signals: set[str]):
    from showvenist.session import default_scorers

    return [s for s in default_scorers() if s.name in signals]


_EP_RE = re.compile(r"^s(\d{1,3})e(\d{1,4})$", re.IGNORECASE)
_RANGE_RE = re.compile(r"^(\d{1,4})(?:\s*-\s*(\d{1,4}))?$")


def _parse_episodes(text: str, season: int | None) -> set[tuple[int, int]]:
    """`--episodes 5-8` (numbers within --season), `--episodes S02E05,S02E09`, or a mix of
    comma-separated items. Raises ValueError with a readable message."""
    out: set[tuple[int, int]] = set()
    for item in (x.strip() for x in text.split(",")):
        if not item:
            continue
        m = _EP_RE.match(item)
        if m:
            out.add((int(m[1]), int(m[2])))
            continue
        m = _RANGE_RE.match(item)
        if not m:
            raise ValueError(f"--episodes: can't read {item!r} (use 5-8 or S02E05,S02E09)")
        if season is None:
            raise ValueError(f"--episodes {item}: a bare episode number or range needs --season")
        lo, hi = int(m[1]), int(m[2] or m[1])
        if hi < lo:
            raise ValueError(f"--episodes: range {item!r} runs backwards")
        out |= {(season, e) for e in range(lo, hi + 1)}
    if not out:
        raise ValueError("--episodes: no episodes given")
    return out


def _llm_mode(value: str | None, cfg, signals: set[str]) -> str:
    mode = (value or cfg.llm.mode or "off").lower()
    if mode not in LLM_MODES:
        _fail(f"--llm must be one of {', '.join(LLM_MODES)} (got {mode!r})")
    return mode if "llm" in signals else "off"


def _llm_scorer(cfg, cache):
    from showvenist.scoring.llm import LlmScorer, default_client_factory

    return LlmScorer(
        client_factory=LLM_CLIENT_FACTORY or default_client_factory,
        model=cfg.llm.model,
        effort=cfg.llm.effort,
        send_both_orders=cfg.llm.send_both_orders,
        cache=cache,
    )


def _has_reference_subtitles(bundle, bundle_dir: Path) -> bool:
    return any((bundle_dir / s.path).is_file() for e in bundle.episodes for s in e.subtitles)


@app.command()
def identify(
    paths: list[Path] = typer.Argument(..., exists=True, help="Files or directories"),
    show: str = typer.Option(..., "--show", help="Bundle slug or show title"),
    season: int | None = typer.Option(None, "--season"),
    episodes: str | None = typer.Option(
        None, "--episodes", help="Manual scope: 5-8 (within --season) or S02E05,S02E09"
    ),
    order: str | None = typer.Option(
        None, "--order",
        help="Starting ordering in the bundle. Default: the show's naming_order; bundles and "
        "marker files don't record one yet, so 'aired'",
    ),
    include_specials: bool = typer.Option(False, "--include-specials"),
    signals: str | None = typer.Option(
        None, "--signals",
        help="Scorers to use, e.g. duration,refsubs, or -cues to drop one "
        f"({','.join(SCORER_NAMES)})",
    ),
    llm: str | None = typer.Option(
        None, "--llm", help="off|auto|all (default: llm.mode in config.toml, normally off)"
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Allow the paid LLM calls whose estimated cost is printed"
    ),
    as_report: bool = typer.Option(
        False, "--report", help="Print the report (the default: no review TUI yet)"
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the versioned JSON report"),
    root: Path | None = typer.Option(None, "-o", help="Bundle root directory"),
) -> None:
    """Identify files against a show's bundle (report mode; never touches the media)."""
    from showvenist.analysis.cache import AnalysisCache
    from showvenist.bundle.store import bundle_path
    from showvenist.config import cache_dir, load_config
    from showvenist.report import exit_code, render_tables, to_json
    from showvenist.session import Session

    if as_report and as_json:
        _fail("--report and --json are exclusive")
    slug, bundle = _resolve_bundle(show, root)
    cfg = load_config()
    sig = _parse_signals(signals)
    mode = _llm_mode(llm, cfg, sig)
    try:
        wanted = _parse_episodes(episodes, season) if episodes else None
    except ValueError as e:
        _fail(str(e))
    cache_dir().mkdir(parents=True, exist_ok=True)
    cache = AnalysisCache(cache_dir() / "analysis.db")
    try:
        sess = Session(
            bundle, bundle_path(slug, root), cfg, ordering=order or "aired", season=season,
            episodes=wanted, include_specials=include_specials, cache=cache,
            scorers=_scorers(sig),
        )
    except (ValueError, KeyError) as e:
        _fail(f"identify: {e}")
    files = _expand(paths)

    def progress(done, total, path, error):
        typer.echo(f"[{done}/{total}] {path.name}" + (f"  ERROR {error}" if error else ""),
                   err=True)

    sess.analyze(files, progress=progress)
    if not sess.files:
        _fail("identify: no readable media files")
    sess.score()
    sess.solve()
    if mode != "off":
        scorer = _llm_scorer(cfg, cache)
        scorer.send = set(sess.llm_targets(mode))
        if scorer.send:
            est = scorer.estimate_cost(sess.files, sess.context())
            typer.echo(f"LLM ({mode}): {len(scorer.send)} file(s) to send, estimated "
                       f"${est:.4f} (cached responses are free)", err=True)
            if est > 0 and not yes:
                _fail("Not sending: re-run with --yes to allow these paid API calls.")
            sess.apply_llm(scorer)
            typer.echo(f"LLM spend: ${scorer.total_cost_usd:.4f}", err=True)
    report = sess.report()
    if mode == "off" and not _has_reference_subtitles(bundle, bundle_path(slug, root)):
        report.notices.append(
            "No reference subtitles and the LLM is off: nearly every file will need review. "
            "Add reference subtitles with `showvenist ingest --subs DIR`, or use "
            "`--llm auto --yes` (the estimated cost is printed first)."
        )
    typer.echo(to_json(report) if as_json else render_tables(report))
    raise typer.Exit(exit_code(report))


@app.command("eval")
def eval_cmd(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    root: Path | None = typer.Option(None, "-o", help="Bundle root directory"),
    library: Path | None = typer.Option(
        None, "--library", help="Prefix for relative manifest paths (read-only library root)"
    ),
    signals: str | None = typer.Option(
        None, "--signals", help=f"Scorers to use or -drop ({','.join(SCORER_NAMES)})"
    ),
    llm: str | None = typer.Option(
        None, "--llm", help="off|auto|all (default: llm.mode in config.toml, normally off)"
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Allow the paid LLM calls whose estimated cost is printed"
    ),
    show_errors: bool = typer.Option(
        False, "--show-errors", help="List the files that were wrong at high confidence"
    ),
    ablate: bool = typer.Option(
        False, "--ablate", help="Metrics with each scorer removed in turn (re-solves stored "
        "evidence; nothing is re-analyzed)"
    ),
    fit: bool = typer.Option(False, "--fit", help="Fit weights (leave-one-season-out)"),
    lam: float = typer.Option(1.0, "--lam", help="Ridge strength toward the defaults"),
    extras_share: float | None = typer.Option(
        None, "--extras-share",
        help="Weight extras up to this share of the fit (default 0.15 = π_x when the manifest "
        "has labeled extras; 0 = unweighted)",
    ),
    write: bool = typer.Option(False, "--write", help="Write fitted weights to config.toml"),
) -> None:
    """Accuracy harness and weight fitting over a labeled manifest (LLM off unless asked)."""
    from showvenist.analysis.cache import AnalysisCache
    from showvenist.bundle.store import bundle_path, load_bundle
    from showvenist.config import cache_dir, config_dir, load_config
    from showvenist.eval.fit import fit_loso, format_fit_report, to_weights_config, write_weights
    from showvenist.eval.manifest import read_manifest
    from showvenist.eval.render import (
        fit_items,
        format_ablation,
        format_eval,
        format_heldout,
        format_times,
        heldout_metrics,
        solve_items,
    )
    from showvenist.eval.runner import GroupRun, evidence_table, resolve_metrics, run_eval
    from showvenist.eval.types import ManifestRow
    from showvenist.scoring.types import PI_X_DEFAULT

    if extras_share is not None and not 0.0 <= extras_share < 1.0:
        _fail("--extras-share must be in [0, 1)")
    cfg = load_config()
    sig = _parse_signals(signals)
    mode = _llm_mode(llm, cfg, sig)
    rows = read_manifest(manifest)
    if library is not None:
        rows = [ManifestRow(str(library / r.path), r.show, r.label, r.order, r.disc) for r in rows]
    slugs = sorted({r.show for r in rows})
    bundles, dirs = {}, {}
    for s in slugs:
        try:
            bundles[s], dirs[s] = load_bundle(s, root), bundle_path(s, root)
        except Exception as e:  # missing bundle: the rows become Problems in the harness
            typer.echo(f"bundle {s}: {e}", err=True)
    cache_dir().mkdir(parents=True, exist_ok=True)
    cache = AnalysisCache(cache_dir() / "analysis.db")
    groups: list[GroupRun] = []
    base_items: dict[str, list] = {}
    llm_items: dict[str, list] = {}
    as_solved: dict[str, list] = {}

    def on_group(g: GroupRun) -> None:
        groups.append(g)
        if fit:
            table = evidence_table(g.solution, g.labels)
            eligible = [lv != "high" for lv in g.first_levels]
            items = fit_items(table, g.key, llm_eligible=eligible)
            base_items[g.key] = [i for i in items if i.regime == "base"]
            llm_items[g.key] = [i for i in items if i.regime == "llm"]
            as_solved[g.key] = solve_items(table, g.key)

    def progress(done, total, path, error):
        if done % 10 == 0 or done == total or error:
            typer.echo(f"[{done}/{total}] {path.name}" + (f"  ERROR {error}" if error else ""),
                       err=True)

    def approve(est: float) -> bool:
        return yes or est == 0

    all_decisions, result = run_eval(
        rows, bundles, cfg, dirs, cache, progress=progress, on_group=on_group,
        scorers=_scorers(sig), llm_mode=mode,
        llm_factory=(lambda: _llm_scorer(cfg, cache)) if mode != "off" else None,
        approve=approve,
    )
    typer.echo(format_eval(result))
    if show_errors:
        from showvenist.eval.harness import resolve_labels
        from showvenist.eval.render import format_high_errors

        resolved, _ = resolve_labels(rows, bundles)
        typer.echo("\n" + format_high_errors(resolved, all_decisions, bundles))
    typer.echo("\n" + format_times(groups))
    refused = [g for g in groups if g.llm_refused]

    if ablate and groups:
        names = sorted({k for g in groups for k in g.solution.results})
        abl = [("all signals", resolve_metrics(groups, bundles).overall)]
        abl += [(f"without {n}", resolve_metrics(groups, bundles, drop={n}).overall)
                for n in names]
        typer.echo("\n" + format_ablation(abl))

    if fit:
        flat = [i for v in base_items.values() for i in v]
        has_extras = any(i.target == "extra" for i in flat)
        share = (PI_X_DEFAULT if has_extras else None) if extras_share is None else (
            extras_share or None)
        try:
            loso = fit_loso(flat, "base", lam=lam, extras_share=share)
        except ValueError as e:
            _fail(f"fit: {e}")
        typer.echo("\n" + format_fit_report(loso))
        default_w = to_weights_config(loso.defaults, {})
        per_fold = {f.season_key: to_weights_config(f.weights, {}) for f in loso.folds}
        typer.echo("\n" + format_heldout(
            "held-out seasons, base weights (re-solved without the LLM):",
            heldout_metrics(base_items, default_w), heldout_metrics(base_items, per_fold),
        ))
        llm_w = None
        llm_flat = [i for v in llm_items.values() for i in v]
        if len({i.season_key for i in llm_flat}) >= 2:
            loso_llm = fit_loso(llm_flat, "llm", lam=lam)
            llm_w = loso_llm.insample_weights
            typer.echo("\n" + format_fit_report(loso_llm))
            llm_fold = {f.season_key: f.weights for f in loso_llm.folds}
            both = {
                k: to_weights_config(per_fold[k].model_dump(exclude={"llm"}),
                                     llm_fold.get(k, loso_llm.defaults))
                for k in per_fold
            }
            typer.echo("\n" + format_heldout(
                "held-out seasons, both regimes as solved:",
                heldout_metrics(as_solved, to_weights_config(loso.defaults, loso_llm.defaults)),
                heldout_metrics(as_solved, both),
            ))
        elif mode != "off":
            typer.echo("LLM-regime fit skipped: needs LLM-scored files from two seasons.")
        if write:
            path = config_dir() / "config.toml"
            write_weights(path, loso.insample_weights, llm_w)
            typer.echo(f"wrote weights to {path}"
                       + ("" if llm_w else " ([weights.llm] left unchanged)"))
    if refused:
        est = sum(g.llm_estimate_usd for g in refused)
        _fail(f"LLM not run for {len(refused)} season(s): estimated ${est:.4f}. "
              "Re-run with --yes to allow these paid API calls.")


if __name__ == "__main__":
    app()

