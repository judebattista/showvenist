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


if __name__ == "__main__":
    app()

