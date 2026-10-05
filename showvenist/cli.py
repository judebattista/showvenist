"""Command-line interface for showvenist."""

import importlib.metadata
import sys
from pathlib import Path

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


@app.command()
def identify(
    paths: list[Path] = typer.Argument(..., exists=True, help="Files or directories"),
    show: str = typer.Option(..., "--show", help="Bundle slug or show title"),
    season: int | None = typer.Option(None, "--season"),
    episodes: str | None = typer.Option(None, "--episodes", help="e.g. S02E05,S02E09"),
    order: str = typer.Option("aired", "--order", help="Matching ordering in the bundle"),
    include_specials: bool = typer.Option(False, "--include-specials"),
    as_json: bool = typer.Option(False, "--json", help="Emit the versioned JSON report"),
    root: Path | None = typer.Option(None, "-o", help="Bundle root directory"),
) -> None:
    """Identify files against a show's bundle (report mode; never touches the media)."""
    import re

    from showvenist.analysis.cache import AnalysisCache
    from showvenist.bundle.store import bundle_path
    from showvenist.config import cache_dir, load_config
    from showvenist.report import exit_code, render_tables, to_json
    from showvenist.session import Session

    slug, bundle = _resolve_bundle(show, root)
    cfg = load_config()
    wanted = None
    if episodes:
        wanted = {(int(a), int(b)) for a, b in re.findall(r"[sS](\d+)[eE](\d+)", episodes)}
    cache_dir().mkdir(parents=True, exist_ok=True)
    sess = Session(
        bundle, bundle_path(slug, root), cfg, ordering=order, season=season, episodes=wanted,
        include_specials=include_specials, cache=AnalysisCache(cache_dir() / "analysis.db"),
    )
    files = _expand(paths)

    def progress(done, total, path, error):
        typer.echo(f"[{done}/{total}] {path.name}" + (f"  ERROR {error}" if error else ""),
                   err=True)

    sess.analyze(files, progress=progress)
    sess.score()
    sess.solve()
    report = sess.report()
    typer.echo(to_json(report) if as_json else render_tables(report))
    raise typer.Exit(exit_code(report))


@app.command("eval")
def eval_cmd(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    root: Path | None = typer.Option(None, "-o", help="Bundle root directory"),
    library: Path | None = typer.Option(
        None, "--library", help="Prefix for relative manifest paths (read-only library root)"
    ),
    fit: bool = typer.Option(False, "--fit", help="Fit weights (leave-one-season-out)"),
    lam: float = typer.Option(1.0, "--lam", help="Ridge strength toward the defaults"),
    write: bool = typer.Option(False, "--write", help="Write fitted weights to config.toml"),
) -> None:
    """Accuracy harness and weight fitting over a labeled manifest (no LLM, no paid calls)."""
    from showvenist.analysis.cache import AnalysisCache
    from showvenist.bundle.store import bundle_path, load_bundle
    from showvenist.config import cache_dir, config_dir, load_config
    from showvenist.eval.fit import fit_loso, format_fit_report, write_weights
    from showvenist.eval.manifest import read_manifest
    from showvenist.eval.render import fit_items, format_eval
    from showvenist.eval.runner import run_eval
    from showvenist.eval.types import ManifestRow

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
    items = []

    def on_group(g):
        from showvenist.eval.runner import evidence_table

        key = f"{g.show}-S{g.season or 0:02d}"
        items.extend(fit_items(evidence_table(g.solution, g.labels), key))

    def progress(done, total, path, error):
        if done % 10 == 0 or done == total or error:
            typer.echo(f"[{done}/{total}] {path.name}" + (f"  ERROR {error}" if error else ""),
                       err=True)

    _, result = run_eval(
        rows, bundles, load_config(), dirs, AnalysisCache(cache_dir() / "analysis.db"),
        progress=progress, on_group=on_group,
    )
    typer.echo(format_eval(result))
    if fit:
        base = [i for i in items if i.regime == "base"]
        loso = fit_loso(base, "base", lam=lam)
        typer.echo("\n" + format_fit_report(loso))
        if write:
            write_weights(config_dir() / "config.toml", loso.insample_weights, {})
            typer.echo(f"wrote weights to {config_dir() / 'config.toml'}")


if __name__ == "__main__":
    app()

