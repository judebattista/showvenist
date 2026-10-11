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
    card: bool = typer.Option(
        False, "--card", help="Print the OCR'd text of every sampled frame in the show's "
        "title_card windows (needs --show, or --from/--skip/--length)"),
    card_from: str | None = typer.Option(
        None, "--from", help="--card: window edge, start or end (with --skip and --length)"),
    skip: float | None = typer.Option(None, "--skip", help="--card: seconds to skip from the edge"),
    length: float | None = typer.Option(None, "--length", help="--card: window length, seconds"),
    dark: bool = typer.Option(
        False, "--dark", help="--card, ad-hoc windows: only mostly-black frames (default: any)"),
) -> None:
    """Dump the extracted dialogue and signals for one file (debugging)."""
    from showvenist.analysis.cache import AnalysisCache
    from showvenist.analysis.inspect import inspect_card, inspect_file
    from showvenist.bundle.store import load_bundle
    from showvenist.config import cache_dir, load_config

    adhoc = (card_from, skip, length)
    if not card and (any(x is not None for x in adhoc) or dark):
        _fail("--from, --skip, --length and --dark need --card")
    if card:
        from showvenist.bundle.schema import TitleCardConfig

        if any(x is not None for x in adhoc):
            if not all(x is not None for x in adhoc):
                _fail("--from, --skip and --length go together")
            if card_from not in ("start", "end"):
                _fail("--from must be start or end")
            try:
                tc = TitleCardConfig.model_validate({
                    "windows": [{"from": card_from, "skip_seconds": skip,
                                 "length_seconds": length}],
                    "pattern": "(?P<ref>.)", "kind": "title",
                    "background": "dark" if dark else "any"})
            except ValueError as e:
                _fail(f"invalid window: {e}")
        else:
            tc = None
            if show:
                try:
                    tc = load_bundle(show, root).title_card
                except Exception as e:
                    _fail(str(e))
            if tc is None:
                _fail("no title_card configured" + (f" for {show}" if show else "")
                      + ": give --show NAME with a title_card in its overrides.yaml, or "
                      "--from, --skip and --length")
        cache_dir().mkdir(parents=True, exist_ok=True)
        cache = AnalysisCache(cache_dir() / "analysis.db")
        typer.echo(inspect_card(file, tc, cache))
        return

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
    from showvenist.hierarchy.parse import walk_media

    out: list[Path] = []
    for p in paths:
        out += walk_media(p) if p.is_dir() else [p]
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


SCORER_NAMES = ("duration", "title", "titlecard", "refsubs", "synopsis", "guests", "cues", "llm")
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


# ---- identify: directory input (spec §4.3, §5) ------------------------------------------------

# Readiness (§5.2): seconds between the two stat() calls of the batch. Tests set it to 0.
READINESS_SETTLE_SECONDS = 1.0


def _interactive() -> bool:
    """Prompts only with a TTY on both ends: a non-TTY stdout is report mode (§4.5), which
    never prompts; non-interactive runs fail instead of asking (§5.1). identify also turns
    prompts off for --report / --json."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _use_tui() -> bool:
    """The review TUI needs a real terminal on both ends (checked on its own, so tests that
    pretend to be interactive for the startup prompts still get the report)."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _tag_reader(cache) -> Callable[[Path], str | None]:
    """The MKV title tag (§5.1) via the cached probe; None when the file can't be probed."""
    from showvenist.analysis.probe import probe

    memo: dict[Path, str | None] = {}

    def tag_of(path: Path) -> str | None:
        if path not in memo:
            try:
                memo[path] = probe(path, cache).title_tag
            except Exception:  # noqa: BLE001 - unreadable files are reported by the analysis
                memo[path] = None
        return memo[path]

    return tag_of


def _files_tree(files: list[Path], hcfg, tag_of):
    """Tree for files named one by one: each file's disc from its folder (else its title
    tag), its season from the season folder above (else the disc label, else the tag)."""
    from dataclasses import replace

    from showvenist.hierarchy.parse import parse_disc, parse_named, parse_season, parse_tag_season
    from showvenist.hierarchy.types import DiscNode, FileNode, SeasonNode, ShowNode

    show = ShowNode(root=files[0].parent)
    seasons: dict[int | None, SeasonNode] = {}
    discs: dict[tuple[int | None, str], DiscNode] = {}
    for f in files:
        st = f.stat()
        tag = tag_of(f)
        node = FileNode(f, st.st_size, st.st_mtime, title_tag=tag,
                        named=parse_named(f.name, hcfg.named_patterns), explicit=True)
        tref = parse_disc(tag, hcfg.disc_patterns) if tag else None
        if tref is not None:
            node.tag_disc = replace(tref, source="tag")
        folder = f.parent
        fref = parse_disc(folder.name, hcfg.disc_patterns)
        season = parse_season((folder.parent if fref else folder).name, hcfg.season_patterns)
        if season is None and fref is not None:
            season = fref.season
        if season is None and tag:
            season = parse_tag_season(tag, hcfg.season_patterns)
        sn = seasons.setdefault(season, SeasonNode(season, None))
        if fref is not None and tref is not None and None not in (fref.number, tref.number) \
                and fref.number != tref.number:
            msg = (f"{folder.name}: folder says disc {fref.number}, title tag says disc "
                   f"{tref.number} ({tag!r})")
            if msg not in show.disagreements:
                show.disagreements.append(msg)
        ref, key, where = (fref, folder.name, folder) if fref else (
            node.tag_disc, f"tag:{' '.join(tref.label.split())}" if tref else "", None)
        if ref is None:
            sn.loose.append(node)
            continue
        d = discs.get((season, key))
        if d is None:
            d = discs[(season, key)] = DiscNode(
                key, ref, where, order_source="number" if ref.number is not None else "name")
            sn.discs.append(d)
        d.files.append(node)
    show.seasons = list(seasons.values())
    _mark_duplicates(show)
    return show


def _mark_duplicates(tree) -> None:
    """Same disc number twice in one season (§5.1: identical labels): see
    hierarchy.parse.mark_duplicate_discs."""
    from showvenist.hierarchy.parse import mark_duplicate_discs

    for s in tree.seasons:
        for p in mark_duplicate_discs(s, _season_title(s.number)):
            if p not in tree.problems:
                tree.problems.append(p)


def _input_tree(paths: list[Path], hcfg, tag_of):
    """One tree over every PATH: directories through hierarchy.parse.build_tree, plain files
    through `_files_tree`; seasons with the same number are merged."""
    from showvenist.hierarchy.parse import build_tree
    from showvenist.hierarchy.types import ShowNode

    trees = [build_tree(p, config=hcfg, tag_of=tag_of) for p in paths if p.is_dir()]
    files = [p for p in paths if not p.is_dir()]
    if files:
        trees.append(_files_tree(files, hcfg, tag_of))
    if len(trees) == 1:
        return trees[0]
    out = ShowNode(root=trees[0].root, name=next((t.name for t in trees if t.name), None))
    by_num: dict = {}
    for t in trees:
        out.problems += t.problems
        out.disagreements += t.disagreements
        out.ignored += t.ignored
        for s in t.seasons:
            have = by_num.get(s.number)
            if have is None:
                by_num[s.number] = s
                out.seasons.append(s)
                continue
            seen = {d.key: d for d in have.discs}
            for d in s.discs:
                if d.key in seen:
                    known = {f.path for f in seen[d.key].files}
                    seen[d.key].files += [f for f in d.files if f.path not in known]
                else:
                    have.discs.append(d)
            known = {f.path for f in have.loose}
            have.loose += [f for f in s.loose if f.path not in known]
    from showvenist.hierarchy.parse import disc_sort_key

    _mark_duplicates(out)
    for s in out.seasons:
        s.discs.sort(key=disc_sort_key)
    out.seasons.sort(key=lambda s: (s.number is None, s.number or 0))
    return out


def _show_dir(path: Path, hcfg) -> Path:
    """The show folder above PATH: up past a disc folder, then past a season folder."""
    from showvenist.hierarchy.parse import parse_disc, parse_season

    d = path if path.is_dir() else path.parent
    if parse_disc(d.name, hcfg.disc_patterns) is not None:
        d = d.parent
    if parse_season(d.name, hcfg.season_patterns) is not None:
        d = d.parent
    return d


def _mark_readiness(tree, stable_seconds: float) -> None:
    from showvenist.hierarchy.readiness import ReadinessChecker

    files = tree.all_files()
    if not files:
        return
    checker = ReadinessChecker(stable_seconds, settle_seconds=READINESS_SETTLE_SECONDS)
    got = checker.check([f.path for f in files])
    for f in files:
        r = got[f.path]
        f.ready, f.not_ready_reason = r.ready, r.reason
    for s in tree.seasons:
        for d in s.discs:
            d.incomplete = any(not f.ready for f in d.files)


def _season_title(n: int | None) -> str:
    if n is None:
        return "Season ? (not parsed)"
    return "Specials (Season 0)" if n == 0 else f"Season {n}"


def format_tree(tree, slug: str | None = None, show_name: str | None = None) -> str:
    """The parsed tree (§5.1): show / season / disc / number of files / files not ready, then
    the problems and the folder-vs-tag disagreements. Always shown before analysis."""
    head = f"Show: {show_name or tree.name or tree.root.name}"
    if slug:
        head += f"  (bundle {slug})"
    lines = [head]

    def files_line(fs) -> str:
        n = len(fs)
        bad = sum(not f.ready for f in fs)
        named = sum(f.named is not None for f in fs)
        out = f"{n} file{'s' if n != 1 else ''}"
        if named:
            out += f", {named} named"
        if bad:
            out += f", {bad} not ready"
        return out

    for s in tree.seasons:
        allf = [f for d in s.discs for f in d.files] + s.loose
        lines.append(f"  {_season_title(s.number)}: {files_line(allf)}")
        for d in s.discs:
            num = f"disc {d.number}" if d.number is not None else "disc ?"
            if d.folder is not None:
                src = d.folder.name
            else:
                src = f"title tag {d.ref.label!r}" if d.ref else d.key
            order = "" if d.order_source == "number" else f" [order: {d.order_source}]"
            lines.append(f"    {num:<8} {src}: {files_line(d.files)}{order}")
        if s.loose:
            lines.append(f"    no disc  {files_line(s.loose)}")
        for f in allf:
            if not f.ready:
                lines.append(f"      not ready: {f.path.name} ({f.not_ready_reason})")
    for ig in tree.ignored:
        lines.append(f"Skipping {ig.name}: it has a .showvenist-ignore file")
    if tree.problems:
        lines.append("Problems:")
        lines += [f"  {p}" for p in tree.problems]
    if tree.disagreements:
        lines.append("Folder and title tag disagree (the folder wins):")
        lines += [f"  {p}" for p in tree.disagreements]
    return "\n".join(lines)


def _resolve_show_choice(show_dir: Path, root: Path | None, interactive: bool):
    from showvenist.hierarchy.marker import ShowResolutionError, resolve_show

    def pick(cands: list[str]) -> str:
        for i, c in enumerate(cands, 1):
            typer.echo(f"  {i}. {c}", err=True)
        n = typer.prompt("Which bundle", type=typer.IntRange(1, len(cands)), err=True)
        return cands[n - 1]

    from showvenist.config import bundles_dir

    try:
        return resolve_show(show_dir, root or bundles_dir(), interactive=interactive,
                            pick=pick if interactive else None)
    except (ShowResolutionError, ValueError) as e:
        _fail(f"identify: {e}")


def _ordering_key(bundle, name: str) -> str | None:
    from showvenist.session import ordering_key

    return ordering_key(bundle, name)


def _check_disc_order(tree, interactive: bool) -> None:
    """§5.1 discs without numbers (or with the same number twice): suggest an order by folder
    creation time and ask; non-interactive runs fail. Side-lettered labels (disc1a, disc1b,
    disc2a) are ordered by (number, side) without asking: that is no guess."""
    from showvenist.hierarchy.parse import sides_order, suggest_disc_order

    for s in tree.seasons:
        if len(s.discs) < 2 or all(d.number is not None for d in s.discs):
            continue
        by_side = sides_order(s.discs)
        if by_side is not None:
            s.discs = by_side
            continue
        ordered = suggest_disc_order(list(s.discs))
        names = ", ".join(d.folder.name if d.folder else d.key for d in ordered)
        msg = (f"{_season_title(s.number)}: discs without usable numbers (or with the same "
               f"number twice); suggested order by folder creation time: {names}")
        if not interactive:
            _fail(f"identify: {msg}. Rename the disc folders (Disc 1, Disc 2, ...); "
                  "non-interactive runs never guess.")
        typer.echo(msg, err=True)
        if not typer.confirm("Use this order?", default=False, err=True):
            _fail("identify: disc order not confirmed")
        s.discs = ordered


@app.command()
def identify(
    paths: list[Path] = typer.Argument(..., exists=True, help="Files or directories"),
    show: str | None = typer.Option(
        None, "--show", help="Bundle slug or show title (default: the marker file, a Jellyfin "
        "[tmdbid-N] hint or the show folder's name)"
    ),
    season: int | None = typer.Option(None, "--season", help="Override the season folder"),
    episodes: str | None = typer.Option(
        None, "--episodes", help="Manual scope: 5-8 (within --season) or S02E05,S02E09"
    ),
    order: str | None = typer.Option(
        None, "--order",
        help="Starting ordering in the bundle (default: the marker's naming_order, else "
        "'aired'). Ordering detection may still switch to a better-fitting one",
    ),
    no_blocks: bool = typer.Option(
        False, "--no-blocks", help="Solve each season without the disc block model"
    ),
    include_specials: bool = typer.Option(False, "--include-specials"),
    allow_duplicates: bool = typer.Option(
        False, "--allow-duplicates",
        help="Keep near-duplicate rips (same dialogue, different bytes) as other versions of "
        "their episode instead of duplicates to review. Exact copies stay duplicates",
    ),
    signals: str | None = typer.Option(
        None, "--signals",
        help="Scorers to use, e.g. duration,refsubs, or -cues to drop one "
        f"({','.join(SCORER_NAMES)})",
    ),
    llm: str | None = typer.Option(
        None, "--llm", help="off|auto|all (default: llm.mode in config.toml, normally off)"
    ),
    budget: float = typer.Option(
        0.0, "--budget",
        help="Most you allow the paid LLM calls to cost, in USD (default 0: nothing is sent; "
        "the estimate is printed first)",
    ),
    reanalyze: bool = typer.Option(
        False, "--reanalyze", help="Ignore files already confirmed in history (analyze them again)"
    ),
    as_report: bool = typer.Option(
        False, "--report", help="Print the report instead of opening the review TUI"
    ),
    as_json: bool = typer.Option(
        False, "--json",
        help="Emit the versioned JSON report (a JSON array of reports when several seasons "
        "are solved)",
    ),
    as_plan: bool = typer.Option(
        False, "--plan", help="Print the exact move list after the report; writes nothing"
    ),
    do_apply: bool = typer.Option(
        False, "--apply",
        help="Print the plan, then move the files after you type yes (without a TTY only "
        "with --confirm)",
    ),
    confirm: str | None = typer.Option(
        None, "--confirm",
        help="With --apply: move without typing yes if the plan hashes to this token "
        "(printed by --plan), or 'last' for the most recent --plan's token",
    ),
    root: Path | None = typer.Option(None, "-o", help="Bundle root directory"),
) -> None:
    """Identify files against a show's bundle (report mode never touches the media).

    PATHS may be files or Show / Season / Disc folders (§5). The parsed tree is printed to
    stderr before analysis. On a terminal (without --report, --json, --plan or --apply) the
    review TUI opens. Each season in the tree is solved on its own and reported.
    --plan adds the move list (§13) and writes nothing; --apply moves the files after a typed
    yes (or with --confirm), with a move log that `showvenist undo` reverses."""
    import json as _json
    from dataclasses import replace

    from showvenist.analysis.cache import AnalysisCache
    from showvenist.bundle.store import bundle_path
    from showvenist.config import cache_dir, load_config
    from showvenist.hierarchy.marker import jellyfin_hint, read_marker, suggest_folder_name
    from showvenist.hierarchy.parse import is_duplicate_disc_problem
    from showvenist.report import exit_code, render_tables, to_json
    from showvenist.session import Session
    from showvenist.state.db import SchemaOutdated, StateDb, state_db_path
    from showvenist.state.history import History

    if sum((as_report, as_json, as_plan, do_apply)) > 1:
        _fail("--report, --json, --plan and --apply are exclusive")
    if confirm is not None and not do_apply:
        _fail("--confirm only goes with --apply")
    if budget < 0:
        _fail("--budget must not be negative")
    confirm_token = None
    if confirm is not None:
        from showvenist.rename.token import normalize_token, read_last_token

        if confirm.strip().lower() == "last":
            saved = read_last_token()
            if saved is None:
                _fail("no saved plan token: run --plan first")
            confirm = saved
        try:
            confirm_token = normalize_token(confirm)
        except ValueError as e:
            _fail(str(e))
    if do_apply and confirm_token is None and not _interactive():
        _fail("identify --apply: no TTY, so nothing is moved; use --plan to see the moves, or "
              "add --confirm TOKEN (from --plan) or --confirm last to apply them from a script")
    cfg = load_config()
    hcfg = cfg.hierarchy
    sig = _parse_signals(signals)
    mode = _llm_mode(llm, cfg, sig)
    try:
        wanted = _parse_episodes(episodes, season) if episodes else None
    except ValueError as e:
        _fail(str(e))
    # report mode and --plan never prompt
    interactive = _interactive() and not (as_report or as_json or as_plan)
    paths = [Path(p) for p in paths]

    # --- show, marker ---
    show_dirs = list(dict.fromkeys(_show_dir(p, hcfg) for p in paths))
    show_dir = show_dirs[0] if len(show_dirs) == 1 else None
    if show is None:
        if show_dir is None:
            _fail("identify: the paths belong to different show folders "
                  f"({', '.join(str(d) for d in show_dirs)}); pass --show")
        choice = _resolve_show_choice(show_dir, root, interactive)
        slug, source = choice.slug, choice.source
        _, bundle = _resolve_bundle(slug, root)
    else:
        slug, bundle = _resolve_bundle(show, root)
        source = "--show"
    marker = None
    if show_dir is not None:
        try:
            marker = read_marker(show_dir)
        except ValueError as e:
            _fail(f"identify: {e}")
    if marker is not None and marker.bundle and marker.bundle != slug:
        typer.echo(f"marker {show_dir}/.showvenist.toml names bundle {marker.bundle!r}, not "
                   f"{slug!r}: its settings are ignored", err=True)
        marker = None
    if order is None:
        order = "aired"
        if marker is not None:
            key = _ordering_key(bundle, marker.naming_order)
            if key is None:
                typer.echo(f"marker naming_order {marker.naming_order!r} is not in the bundle; "
                           "starting from 'aired'", err=True)
            else:
                order = key

    # --- tree, readiness: printed before analysis ---
    cache_dir().mkdir(parents=True, exist_ok=True)
    cache = AnalysisCache(cache_dir() / "analysis.db")
    tree = _input_tree(paths, hcfg, _tag_reader(cache))
    tree.marker = marker
    _mark_readiness(tree, cfg.analysis.readiness_stable_seconds)
    shown = format_tree(tree, slug, show_dir.name if show_dir else None)
    typer.echo(f"{shown}\n  show resolved by: {source}", err=True)
    if not tree.all_files():
        _fail("identify: no media files found")
    wrote_marker = None
    if (interactive and marker is None and show_dir is not None
            and source in ("name", "pick", "tmdbid-hint")):
        wrote_marker = _offer_marker(show_dir, slug, bundle, order)
    hard = [p for p in tree.problems if not is_duplicate_disc_problem(p)]
    if hard:
        if not interactive:
            _fail("identify: some folders could not be parsed (see Problems above); "
                  "non-interactive runs never guess. Rename them, or pass the files on their "
                  "own.")
        if not typer.confirm("Continue with those files as loose files (no disc)?",
                             default=False, err=True):
            _fail("identify: stopped")
    _check_disc_order(tree, interactive)

    # --- which seasons to solve ---
    if season is not None:
        picked = [s for s in tree.seasons if s.number == season]
        if not picked:
            if len(tree.seasons) != 1:
                _fail(f"identify: no season {season} in the tree")
            picked = tree.seasons
        jobs = [(s, season) for s in picked]
    else:
        jobs = []
        for s in tree.seasons:
            n = s.number
            if n is None and wanted is None:
                if not interactive:
                    _fail(f"identify: can't tell the season of {tree.root} (no season folder or "
                          "title tag); pass --season. Non-interactive runs never guess.")
                n = typer.prompt(f"Season of the files under {tree.root}", type=int, err=True)
            jobs.append((s, n))

    def progress(done, total, path, error):
        typer.echo(f"[{done}/{total}] {path.name}" + (f"  ERROR {error}" if error else ""),
                   err=True)

    has_refs = _has_reference_subtitles(bundle, bundle_path(slug, root))
    folder_hint = None
    if show_dir is not None and jellyfin_hint(show_dir.name) is None and bundle.show.ids.tmdb:
        folder_hint = suggest_folder_name(bundle.show.title, bundle.show.year,
                                          bundle.show.ids.tmdb)
    hist, hist_db = None, None
    if state_db_path().exists():  # report mode never creates or writes state.db
        try:
            hist_db = (StateDb.open_default() if do_apply else StateDb.open_readonly())
        except SchemaOutdated as e:
            typer.echo(f"identify: history ignored: {e}", err=True)
        if hist_db is not None:
            hist = History(hist_db)
    def make_session(snode, num):
        sub = replace(tree, seasons=[snode])
        try:
            sess = Session(
                bundle, bundle_path(slug, root), cfg, ordering=order, season=num,
                episodes=wanted, include_specials=include_specials or num == 0, cache=cache,
                scorers=_scorers(sig), tree=sub, use_blocks=not no_blocks, marker=marker,
                allow_duplicates=allow_duplicates, history=hist, reanalyze=reanalyze,
            )
        except (ValueError, KeyError) as e:
            _fail(f"identify: {e}")
        if marker is not None:
            sess.language = marker.language  # the rips' subtitle language (§5.4)
        return sess

    def extra_notices(sess, snode) -> list[str]:
        """The notices identify adds to each season's report (TUI and report mode alike)."""
        out = []
        if mode == "off" and not has_refs:
            out.append(
                "No reference subtitles and the LLM is off: nearly every file will need "
                "review. Add reference subtitles with `showvenist ingest --subs DIR`, or use "
                "`--llm auto --budget N` (the estimated cost is printed first)."
            )
        if allow_duplicates and not sess.near:
            out.append(
                "--allow-duplicates: no near duplicates (same dialogue, different bytes) found; "
                "exact copies are reported as duplicates of their first copy either way."
            )
        if folder_hint is not None:
            out.append(
                f"Suggestion (not applied): rename the show folder {show_dir.name!r} to "
                f"{folder_hint!r} so Jellyfin uses the same TMDB entry."
            )
        if wrote_marker is not None:
            out.append(f"Wrote the marker file {wrote_marker}.")
        elif marker is None and source in ("name", "pick", "tmdbid-hint"):
            out.append(
                f"Show resolved from the folder ({source}) as {slug!r}; no marker file was "
                "written."
            )
        out += [f"Folder and title tag disagree (the folder wins): {d}"
                for d in tree.disagreements]
        untagged = [f for f in snode.loose if f.title_tag and f.tag_disc is None]
        if untagged:
            n = len(untagged)
            out.append(
                f"{n} file{'s have' if n != 1 else ' has'} a title tag that names no disc "
                f"(e.g. {untagged[0].title_tag!r}); they are solved without disc blocks."
            )
        return out

    if interactive and not do_apply and _use_tui():  # §12.2: a TTY, no output flag
        try:
            code = _review_tui(
                [(_season_title(num), make_session(snode, num),
                  lambda s, sn=snode: extra_notices(s, sn)) for snode, num in jobs],
                bundle=bundle, cfg=cfg, cache=cache, marker=marker, hist=hist, budget=budget,
                mode=mode, show_title=bundle.show.title)
        finally:
            if hist_db is not None:
                hist_db.close()
        raise typer.Exit(code)

    reports = []
    sessions = []
    for snode, num in jobs:
        sess = make_session(snode, num)
        if len(jobs) > 1:
            typer.echo(f"-- {_season_title(num)}", err=True)
        sess.analyze(None, progress=progress)
        idle = not sess.files and not sess.named_rows
        if idle:  # every file named, not ready or unreadable: reported, nothing solved
            typer.echo(f"{_season_title(num)}: nothing to solve ({len(sess.not_ready)} not "
                       f"ready, {len(sess.named)} named, {len(sess.errors)} unreadable)",
                       err=True)
        else:
            sess.score()
            sess.solve()
        if not idle and mode != "off" and sess.files:
            scorer = _llm_scorer(cfg, cache)
            scorer.send = set(sess.llm_targets(mode))
            if scorer.send:
                est = scorer.estimate_cost(sess.files, sess.context())
                typer.echo(f"LLM ({mode}): {len(scorer.send)} file(s) to send, estimated "
                           f"${est:.4f} (cached responses are free)", err=True)
                if est > budget:
                    _fail(_over_budget(est, budget))
                sess.apply_llm(scorer)
                typer.echo(f"LLM spend: ${scorer.total_cost_usd:.4f}", err=True)
        report = sess.report()
        report.notices += extra_notices(sess, snode)
        reports.append(report)
        sessions.append(sess)

    if as_plan or do_apply:
        try:
            code = _plan_and_apply(reports, bundle, cfg, marker, hist, hist_db, do_apply,
                                   confirm_token, interactive, exit_code, render_tables,
                                   sessions=sessions, show_slug=slug)
        finally:
            if hist_db is not None:
                hist_db.close()
        raise typer.Exit(code)
    if hist_db is not None:
        hist_db.close()
    done = reports
    if as_json:
        if len(done) == 1:
            typer.echo(to_json(done[0]))
        else:
            typer.echo(_json.dumps([_json.loads(to_json(r)) for r in done], indent=2))
    else:
        for r in done:
            if len(done) > 1:
                typer.echo(f"== {_season_title(r.season)} ==")
            typer.echo(render_tables(r))
    raise typer.Exit(_final_exit_code(done, exit_code))


def _review_tui(seasons, *, bundle, cfg, cache, marker, hist, budget: float, mode: str,
                show_title: str) -> int:
    """Open the review TUI (§12.2) on the seasons' sessions, not yet analyzed: the TUI
    analyzes them in the background. Startup questions were asked before this."""
    from showvenist.rename.types import PlanOptions
    from showvenist.tui.app import ReviewOptions, SeasonInput, run_review

    opts = ReviewOptions(
        bundle=bundle, show_title=show_title,
        plan_options=PlanOptions(
            naming_order=marker.naming_order if marker is not None else None,
            template=cfg.naming.template, move_extras=cfg.naming.move_extras),
        history=hist, budget=budget, llm_mode=mode,
        llm_factory=lambda: _llm_scorer(cfg, cache), command_args=" ".join(sys.argv[1:]),
    )
    return run_review([SeasonInput(t, s, n) for t, s, n in seasons], opts)


def _over_budget(est: float, budget: float) -> str:
    import math

    need = math.ceil(est * 100 - 1e-9) / 100
    return (f"Not sending: estimated ${est:.4f} exceeds --budget {budget:g}. "
            f"Re-run with --budget {need:.2f} to allow these paid API calls.")


def _typed_yes(question: str) -> bool:
    return typer.prompt(f"{question} Type yes to go ahead", default="",
                        show_default=False).strip() == "yes"


def _plan_and_apply(reports, bundle, cfg, marker, hist, hist_db, do_apply: bool,
                    confirm_token: str | None,
                    interactive: bool, exit_code, render_tables, *, sessions=(),
                    show_slug: str | None = None) -> int:
    """identify --plan / --apply (§4.3, decision 58): one plan per solved season, built by
    rename.plan.build_plan, printed in full; --apply moves through rename.movelog.apply_plan.
    --plan writes nothing: state.db is read only when it already exists (decision 60: it prints
    a plan token and saves it; --apply --confirm needs the rebuilt plan to hash to the token).

    --apply first recovers leftovers of an interrupted run (§13.4), whatever happens next, then
    asks once for all the plans and records ONE run for the invocation (§7.2), so `undo` reverses
    every season it moved."""
    from showvenist.rename.movelog import PlanRefused, apply_plan, recover
    from showvenist.rename.plan import build_plan
    from showvenist.rename.render import format_apply, format_plan, format_recovery
    from showvenist.rename.token import normalize_token, plan_token, save_last_token
    from showvenist.rename.types import PlanOptions
    from showvenist.session import record_overrides
    from showvenist.state.db import StateDb, state_db_path
    from showvenist.state.history import History

    opts = PlanOptions(
        naming_order=marker.naming_order if marker is not None else None,
        template=cfg.naming.template, move_extras=cfg.naming.move_extras,
    )
    codes = [_final_exit_code(reports, exit_code)]
    db = None  # the writable state.db (--apply only)
    try:
        if do_apply and state_db_path().exists():  # no state.db: nothing can be left over
            db = (hist_db if hist_db is not None and not getattr(hist_db, "readonly", False)
                  else StateDb.open_default())
            rec = format_recovery(recover(db))
            if rec:
                typer.echo(rec, nl=False, err=True)
        plans = []
        for r in reports:
            if len(reports) > 1:
                typer.echo(f"== {_season_title(r.season)} ==")
            typer.echo(render_tables(r))
            try:
                plan = build_plan(r, bundle, opts, history=hist)
            except ValueError as e:
                _fail(f"identify: {e}")
            typer.echo(format_plan(plan))
            plans.append(plan)
            if not plan.ok:
                codes.append(1)
        token = plan_token(plans)
        if not do_apply:
            if any(p.moves for p in plans):
                save_last_token(token)
                typer.echo(f"Plan token: {token}  (apply with --apply --confirm {token}, or "
                           "--confirm last)")
            return max(codes)
        if any(not p.ok for p in plans):
            typer.echo("Not applied: the plan has errors (nothing was moved).", err=True)
            return 1
        if not any(p.moves for p in plans):
            typer.echo("Nothing to move.")
            return max(codes)
        if confirm_token is not None and normalize_token(token) != confirm_token:
            typer.echo(f"Not applied: the plan changed since the token was made (token now "
                       f"{token}). Nothing was moved.", err=True)
            return 1
        n_moves = sum(len(p.moves) for p in plans)
        if confirm_token is None and not _typed_yes(f"Move {n_moves} file(s) as listed above?"):
            typer.echo("Nothing was moved.")
            return 1
        if db is None:
            db = StateDb.open_default()
        # one run row for the whole invocation, after consent (decision 59: nothing proposed)
        run = History(db).begin_run("identify --apply", " ".join(sys.argv[1:]))
        # pins that overrode the tool's choice are negative labels (decision 61); the CLI
        # has no pins, so this records nothing today
        for sess in sessions:
            if sess.pins:
                record_overrides(History(db), sess, run.id, show_slug or sess.show_slug)

        def confirm_cross(ms) -> bool:
            # crossing filesystems copies and removes: always typed, never by --confirm alone
            return interactive and _typed_yes(
                f"{len(ms)} move(s) cross filesystems (copy, check, then remove the "
                "original).")

        for plan in plans:
            if not plan.moves:
                continue
            try:
                res = apply_plan(plan, db, command="identify --apply", args="",
                                 confirm=lambda p: True, confirm_cross_fs=confirm_cross,
                                 run_id=run.id)
            except PlanRefused as e:
                typer.echo(f"Not applied: {e}", err=True)
                return 1
            typer.echo(format_apply(res))
            if res.cancelled or res.failed:
                codes.append(1)
    finally:
        if db is not None and db is not hist_db:
            db.close()
    return 1 if 1 in codes else max(codes)


def _final_exit_code(reports, exit_code) -> int:
    """§4.5: report.exit_code per season (0 / 2), widened for unreadable files: any of them
    needs review (at least 2); when every file of the run was unreadable it's an error (1)."""
    from showvenist.session import FLAG_UNREADABLE

    rows = [f for r in reports for f in r.files]
    bad = [f for f in rows if FLAG_UNREADABLE in f.flags]
    if rows and len(bad) == len(rows):
        return 1
    codes = [exit_code(r) for r in reports]
    return 2 if bad or 2 in codes else 0


def _offer_marker(show_dir: Path, slug: str, bundle, order: str) -> Path | None:
    """§5.4: the marker file is written on the first run after the user confirms the show
    resolved from the folder (name, Jellyfin hint or pick). Never overwrites."""
    from showvenist.hierarchy.marker import MARKER_NAME, write_marker
    from showvenist.hierarchy.types import Marker

    title = bundle.show.title
    if not typer.confirm(f"Write {MARKER_NAME} for {title} in {show_dir}?", default=False,
                         err=True):
        return None
    m = Marker(show=title, tmdb_id=bundle.show.ids.tmdb, bundle=slug,
               language=bundle.show.language, naming_order=order)
    try:
        return write_marker(show_dir, m)
    except (FileExistsError, OSError) as e:
        typer.echo(f"marker not written: {e}", err=True)
        return None


def _state_file(file: Path) -> tuple[str, int]:
    from showvenist.state.fingerprint import fingerprint

    try:
        return fingerprint(file), file.stat().st_size
    except OSError as e:
        _fail(f"cannot read {file}: {e}")


@app.command()
def assign(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="Media file"),
    episode: str | None = typer.Argument(None, help="Episode, e.g. S02E05"),
    show: str | None = typer.Option(
        None, "--show", help="Bundle slug or show title (default: the show folder's marker file)"),
    order: str | None = typer.Option(
        None, "--order", help="Ordering the episode is written in (default: the show's "
        "naming_order)"),
    extra: bool = typer.Option(False, "--extra", help="Record the file as a confirmed extra"),
    rename: bool = typer.Option(
        False, "--rename", help="Also rename/move the file as the plan rules say (§13): shows "
        "the planned move and asks y/N"),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before writing (or moving)"),
    root: Path | None = typer.Option(None, "-o", help="Bundle root directory"),
) -> None:
    """Record a confirmed match by hand (source manual). Moves or renames the file only with
    --rename."""
    from showvenist.config import bundles_dir, load_config
    from showvenist.eval.harness import resolve_labels
    from showvenist.eval.types import ManifestRow
    from showvenist.hierarchy.marker import ShowResolutionError, read_marker, resolve_show
    from showvenist.state.db import StateDb
    from showvenist.state.history import History

    if extra == (episode is not None):
        _fail("assign: give an episode (S02E05) or --extra, not both and not neither")
    hcfg = load_config().hierarchy
    show_dir = _show_dir(file.resolve(), hcfg)
    try:
        marker = read_marker(show_dir)
    except ValueError as e:
        _fail(f"assign: {e}")
    if show is None:
        try:
            slug = resolve_show(show_dir, root or bundles_dir(), interactive=False).slug
        except (ShowResolutionError, ValueError) as e:
            _fail(f"assign: {e}; pass --show")
        _, bundle = _resolve_bundle(slug, root)
    else:
        slug, bundle = _resolve_bundle(show, root)
    if marker is not None and marker.bundle and marker.bundle != slug:
        marker = None
    ordering = order or (marker.naming_order if marker is not None else "aired")
    okey = _ordering_key(bundle, ordering)
    if okey is None:
        _fail(f"assign: unknown ordering {ordering!r} (the bundle has: "
              f"{', '.join(sorted(bundle.orderings))})")
    eid = season = number = title = None
    what = "a confirmed extra"
    if not extra:
        labels, problems = resolve_labels(
            [ManifestRow(str(file), slug, episode or "", okey)], {slug: bundle})
        if problems or not labels or labels[0].kind != "episode":
            _fail(f"assign: {problems[0].reason if problems else f'cannot read {episode!r}'}")
        eid = labels[0].episode_id
        ent = next(e for e in bundle.orderings[okey].entries if e.episode_id == eid)
        season, number = ent.season, ent.episode
        title = bundle.episode_by_id()[eid].title
        what = f"S{season:02d}E{number:02d} {title!r}"
    fp, size = _state_file(file)
    abs_path = str(file.resolve())
    if rename:
        naming = marker.naming_order if marker is not None else None
        _assign_rename(Path(abs_path), fp, slug, bundle, eid, naming, yes)
        return
    with StateDb.open_default() as db:
        hist = History(db)
        typer.echo(f"{file.name}: {what} in {slug} (ordering {okey}), source manual")
        for m in hist.confirmed_for_file(fp):
            if m.show_slug == slug and m.episode_id != eid:
                was = ("extra" if m.episode_id is None
                       else f"S{m.season:02d}E{m.episode:02d}")
                typer.echo(f"  its confirmed match {was} will become rejected")
        if eid is not None:
            for m in hist.confirmed_for_episode(slug, eid):
                other = hist.get_file(m.fingerprint)
                if m.fingerprint != fp:
                    typer.echo(f"  note: already confirmed for "
                               f"{other.last_path if other else m.fingerprint}")
        if not yes and not typer.confirm("Record this?", default=False):
            typer.echo("Nothing written.")
            raise typer.Exit(1)
        hist.touch_file(fp, size=size, path=abs_path, show_slug=slug)
        run = hist.begin_run("assign", f"{abs_path} {'--extra' if extra else episode}")
        hist.confirm(fp, slug, episode_id=eid, season=season, episode=number, title=title,
                     source="manual", run_id=run.id)
    typer.echo(f"Recorded (run {run.id}). Nothing was moved or renamed.")


def _assign_rename(file: Path, fp: str, slug: str, bundle, eid: str | None,
                   naming_order: str | None, yes: bool) -> None:
    """assign --rename: a one-file plan (pinned, so the match is recorded with source manual
    by the move itself), shown in full, applied after y/N. When the file needs no move, the
    match is recorded as a plain assign."""
    from showvenist.config import load_config
    from showvenist.hierarchy.parse import parse_disc
    from showvenist.rename.movelog import PlanRefused, apply_plan, recover
    from showvenist.rename.plan import build_plan
    from showvenist.rename.render import format_apply, format_plan, format_recovery
    from showvenist.rename.types import PlanOptions
    from showvenist.report import format_odds
    from showvenist.report_schema import (
        Confidence,
        Decision,
        EpisodeRef,
        FileReport,
        Report,
        Summary,
    )
    from showvenist.state.db import StateDb
    from showvenist.state.history import History

    cfg = load_config()
    ref = None
    if eid is not None:
        ep = bundle.episode_by_id()[eid]
        ent = next(e for o in bundle.orderings.values() for e in o.entries
                   if e.episode_id == eid)
        ref = EpisodeRef(episode_id=eid, season=ent.season, episode=ent.episode, title=ep.title)
    disc = file.parent.name if parse_disc(file.parent.name, cfg.hierarchy.disc_patterns) else None
    row = FileReport(
        path=str(file), fingerprint=fp, disc=disc, duration_s=0.0, regime="base",
        decision=Decision(kind="episode" if ref else "extra", episode=ref),
        confidence=Confidence(level="high", delta=None, odds=format_odds(None)),
        flags=["pinned"],
    )
    report = Report(tool_version="", show_slug=slug, bundle_revision=bundle.bundle_revision,
                    ordering=naming_order or "aired", files=[row],
                    summary=Summary(files=1, high=1, medium=0, low=0, extras=int(ref is None)))
    opts = PlanOptions(naming_order=naming_order, template=cfg.naming.template)
    with StateDb.open_default() as db:
        rec = format_recovery(recover(db))  # §13.4: before anything else, whatever follows
        if rec:
            typer.echo(rec, nl=False, err=True)
        hist = History(db)
        try:
            plan = build_plan(report, bundle, opts, history=hist)
        except ValueError as e:
            _fail(f"assign: {e}")
        typer.echo(format_plan(plan))
        if not plan.ok:
            _fail("assign: not applied, the plan has errors (nothing was written)")
        if not plan.moves:
            if not yes and not typer.confirm("Nothing to move. Record the match anyway?",
                                             default=False):
                typer.echo("Nothing written.")
                raise typer.Exit(1)
            hist.touch_file(fp, size=file.stat().st_size, path=str(file), show_slug=slug)
            run = hist.begin_run("assign", f"{file} {eid or '--extra'}")
            ent = (next(e for e in bundle.orderings[plan.naming_order].entries
                        if e.episode_id == eid) if eid else None)
            hist.confirm(fp, slug, episode_id=eid, season=ent.season if ent else None,
                         episode=ent.episode if ent else None,
                         title=bundle.episode_by_id()[eid].title if eid else None,
                         source="manual", run_id=run.id)
            typer.echo(f"Recorded (run {run.id}). Nothing was moved.")
            return

        def confirm(p) -> bool:
            return yes or typer.confirm("Move it and record the match?", default=False)

        try:
            res = apply_plan(plan, db, command="assign --rename",
                             args=f"{file} {eid or '--extra'}", confirm=confirm,
                             confirm_cross_fs=lambda ms: typer.confirm(
                                 "This move crosses filesystems (copy, check, then remove the "
                                 "original). Go ahead?", default=False))
        except PlanRefused as e:
            _fail(f"assign: {e}")
        typer.echo(format_apply(res))
        if res.cancelled or res.failed:
            raise typer.Exit(1)


@app.command()
def undo(
    run_id: int | None = typer.Argument(None, help="Run to reverse (default: the latest)"),
    last: bool = typer.Option(False, "--last", help="Reverse the latest run with moves"),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before moving files back"),
) -> None:
    """Reverse an applied rename run: shows its moves, asks y/N, moves files back in reverse
    order; the run's matches become undone (§13.5). Never overwrites; folders stay."""
    from showvenist.rename.render import format_ops, format_recovery, format_undo
    from showvenist.rename.undo import undo_run
    from showvenist.state.db import StateDb, state_db_path

    if last and run_id is not None:
        _fail("undo: give --last or a RUN_ID, not both")
    if not state_db_path().exists():
        _fail("undo: no runs recorded (no state.db)")

    def confirm(ops) -> bool:
        typer.echo(format_ops(ops))
        return yes or typer.confirm(f"Move {len(ops)} file(s) back?", default=False)

    with StateDb.open_default() as db:
        res = undo_run(db, run_id, confirm=confirm)  # recovers leftovers first (§13.4)
    if res.run_id is None or (not res.undone and not res.skipped and not res.cancelled):
        rec = format_recovery(res.recovered)
        if rec:
            typer.echo(rec, nl=False, err=True)
        _fail("undo: nothing to undo" + (f" in run {run_id}" if run_id is not None else ""))
    typer.echo(format_undo(res))
    if res.cancelled or res.skipped:
        raise typer.Exit(1)


@app.command("history")
def history_cmd(
    show: str | None = typer.Option(
        None, "--show", help="Bundle slug or show title (default: every show in history)"),
    season: int | None = typer.Option(None, "--season", help="One season of the ordering"),
    order: str | None = typer.Option(None, "--order", help="Ordering to list in"),
    root: Path | None = typer.Option(None, "-o", help="Bundle root directory"),
) -> None:
    """Per season: confirmed episodes with their files, missing episodes, confirmed extras."""
    from showvenist.state.db import SchemaOutdated, StateDb, StateDbMissing
    from showvenist.state.history import History

    try:
        db = StateDb.open_readonly()
    except StateDbMissing:
        typer.echo("History is empty.")
        return
    except SchemaOutdated as e:
        _fail(f"history: {e}")
    with db:
        hist = History(db)
        slugs = [_resolve_bundle(show, root)[0]] if show else hist.shows()
        if not slugs:
            typer.echo("History is empty.")
            return
        for slug in slugs:
            _, bundle = _resolve_bundle(slug, root)
            _print_history(hist, slug, bundle, season, order)


def _print_history(hist, slug: str, bundle, season: int | None, order: str | None) -> None:
    name = order or "aired"
    okey = _ordering_key(bundle, name)
    if okey is None:
        _fail(f"history: unknown ordering {name!r} for {slug}")
    entries = bundle.orderings[okey].entries
    by_ep = bundle.episode_by_id()
    got: dict[str, list[str]] = {}
    extras: list[str] = []
    for m in hist.confirmed(slug):
        f = hist.get_file(m.fingerprint)
        path = f.last_path if f else m.fingerprint
        if m.episode_id is None:
            extras.append(path)
        else:
            got.setdefault(m.episode_id, []).append(path)
    typer.echo(f"{slug} (ordering {okey})")
    seasons = sorted({e.season for e in entries if season is None or e.season == season})
    for sn in seasons:
        mine = [e for e in entries if e.season == sn]
        have = [e for e in mine if e.episode_id in got]
        miss = [e for e in mine if e.episode_id not in got]
        typer.echo(f"  Season {sn}: {len(have)} of {len(mine)} confirmed")
        for e in have:
            for path in got[e.episode_id]:
                typer.echo(f"    S{e.season:02d}E{e.episode:02d} {by_ep[e.episode_id].title}"
                           f"  {path}")
        if miss:
            typer.echo("    missing: " + ", ".join(f"E{e.episode:02d}" for e in miss))
    if season is None:
        if extras:
            typer.echo(f"  Confirmed extras ({len(extras)})")
            for path in extras:
                typer.echo(f"    {path}")
        undone = hist.undone(slug)
        if undone:
            typer.echo(f"  Undone (not counted): {len(undone)}")


@app.command()
def forget(
    target: str = typer.Argument(..., help="A file path or a fingerprint"),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before removing"),
) -> None:
    """Remove every record of a file from state.db (its matches; the file is not touched)."""
    from showvenist.state.db import StateDb
    from showvenist.state.history import History

    p = Path(target)
    fp = _state_file(p)[0] if p.is_file() else target
    with StateDb.open_default() as db:
        hist = History(db)
        f = hist.get_file(fp)
        matches = hist.matches_for_file(fp)
        if f is None and not matches:
            _fail(f"forget: no record of {target}")
        typer.echo(f"{fp}" + (f"  {f.last_path}" if f else ""))
        for m in matches:
            what = "extra" if m.episode_id is None else f"S{m.season:02d}E{m.episode:02d}"
            typer.echo(f"  {m.status} {what} ({m.source}, {m.show_slug})")
        typer.echo(f"Would remove the file record and {len(matches)} match(es); the move log "
                   "keeps its text.")
        if not yes and not typer.confirm("Forget this file?", default=False):
            typer.echo("Nothing removed.")
            raise typer.Exit(1)
        hist.forget(fp)
    typer.echo("Forgotten.")


@app.command("eval")
def eval_cmd(
    manifest: Path | None = typer.Argument(None, exists=True, dir_okay=False),
    from_history: bool = typer.Option(
        False, "--from-history",
        help="Labels from state.db instead of a manifest: confirmed matches are positive "
        "labels, rejected matches negative ones",
    ),
    show: str | None = typer.Option(
        None, "--show", help="Limit to one show (bundle slug or title)"),
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
    budget: float = typer.Option(
        0.0, "--budget",
        help="Most you allow the paid LLM calls to cost, in USD (default 0: nothing is sent)",
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
    no_blocks: bool = typer.Option(
        False, "--no-blocks", help="Solve every season without the disc block model"
    ),
    compare_blocks: bool = typer.Option(
        False, "--compare-blocks",
        help="Also run without blocks and print both metric blocks and the per-season delta",
    ),
    start_order: str = typer.Option(
        "aired", "--start-order",
        help="Ordering every season starts from, as identify's --order (default 'aired': "
        "ordering detection must find the release order itself). 'labels' starts from the "
        "labels' own ordering, which hands detection the answer",
    ),
) -> None:
    """Accuracy harness and weight fitting over a labeled manifest (LLM off unless asked).

    Discs come from the manifest's disc column, the file's disc folder or its MKV title tag;
    seasons with discs are solved with the block model unless --no-blocks. Labeled files are
    analyzed as unlabeled input (a name like show_s01e02 is never a pin here)."""
    from showvenist.analysis.cache import AnalysisCache
    from showvenist.bundle.store import bundle_path, load_bundle
    from showvenist.config import cache_dir, config_dir, load_config
    from showvenist.eval.fit import fit_loso, format_fit_report, to_weights_config, write_weights
    from showvenist.eval.manifest import read_manifest
    from showvenist.eval.render import (
        eval_caveats,
        fit_items,
        format_ablation,
        format_compare_blocks,
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
    if no_blocks and compare_blocks:
        _fail("--no-blocks and --compare-blocks are exclusive (the comparison runs both)")
    cfg = load_config()
    sig = _parse_signals(signals)
    mode = _llm_mode(llm, cfg, sig)
    if (manifest is None) == (not from_history):
        _fail("eval: give a manifest CSV or --from-history, not both and not neither")
    hl = None
    if from_history:
        from showvenist.eval.history import history_labels
        from showvenist.state.db import SchemaOutdated, StateDb, StateDbMissing
        from showvenist.state.history import History

        slug_only = _resolve_bundle(show, root)[0] if show else None
        try:
            hdb = StateDb.open_readonly()
        except StateDbMissing:
            _fail("eval: no history (no state.db)")
        except SchemaOutdated as e:
            _fail(f"eval: {e}")
        with hdb:
            hl = history_labels(History(hdb), [slug_only] if slug_only else None)
        rows = hl.rows
        if not rows:
            _fail("eval: history has no confirmed matches with files on disk"
                  + (f" for {slug_only}" if slug_only else ""))
    else:
        rows = read_manifest(manifest)
        if show:
            slug_only = _resolve_bundle(show, root)[0]
            rows = [r for r in rows if r.show == slug_only]
    if library is not None and not from_history:
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
        return est <= budget

    llm_factory = (lambda: _llm_scorer(cfg, cache)) if mode != "off" else None
    all_decisions, result = run_eval(
        rows, bundles, cfg, dirs, cache, progress=progress, on_group=on_group,
        scorers=_scorers(sig), llm_mode=mode, llm_factory=llm_factory, approve=approve,
        use_blocks=not no_blocks, starting_order=start_order,
    )
    if hl is not None:
        result.problems += hl.problems
    if compare_blocks:
        # the second run re-reads the analysis cache the first one just filled
        typer.echo("comparison run without blocks", err=True)
        _, plain = run_eval(
            rows, bundles, cfg, dirs, cache,
            scorers=_scorers(sig), llm_mode=mode, llm_factory=llm_factory, approve=approve,
            use_blocks=False, starting_order=start_order,
        )
        typer.echo(format_compare_blocks(result, plain, {g.key: g.n_discs for g in groups}))
    else:
        typer.echo(format_eval(result))
    if hl is not None:
        from showvenist.eval.history import negative_line

        typer.echo(negative_line(hl.negatives, all_decisions))
    typer.echo(f"\nstarting order: {start_order}"
               + (" (the labels' ordering: detection is handed the answer)"
                  if start_order == "labels" else ""))
    for line in eval_caveats(groups, blocks=not no_blocks):
        typer.echo(line)
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
        est = max(g.llm_estimate_usd for g in refused)  # the budget is checked per season
        _fail(f"LLM not run for {len(refused)} season(s). "
              + _over_budget(est, budget).replace("Not sending: ", "", 1))


if __name__ == "__main__":
    app()

