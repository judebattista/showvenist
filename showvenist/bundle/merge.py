"""Merging secondary sources into a bundle (spec §6.5, decision 62).

The primary (TMDB) decides which episodes exist, their ids, numbering and orderings. A secondary
source only adds: alternative titles, synopses, gaps in runtime / air date / production code,
credits and guest characters. Secondary episodes are lined up with the primary's by content (an
order-preserving sequence alignment on title similarity, air-date proximity and position), never
by number. What can't be lined up becomes a conflict in `provenance[].conflict`; nothing is ever
added automatically. Pure: no network, no clock, inputs are not mutated.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

import numpy as np
import yaml
from rapidfuzz import fuzz, process

from showvenist.bundle.schema import (
    Bundle,
    CastEntry,
    Episode,
    Ordering,
    PartialBundle,
    Provenance,
    Synopsis,
)

# Pair score (sum of terms below). A pair is merged only at MATCH_MIN or above.
MATCH_MIN = 2.0
# Title: (ratio - 60) / 10, clamped. 100 -> +4, 80 -> +2, 60 -> 0, <=30 -> -3. Zero when either
# title is uninformative (Roman numeral, "Episode N", generic). At -3, two unrelated real titles
# need more than a same-day air date (+4) to pair: also the same number (+1).
TITLE_PIVOT, TITLE_MIN, TITLE_MAX = 60.0, -3.0, 4.0
# Two real titles this far apart are different episodes whatever the date and number say (two
# episodes aired the same night). "Pilot" is not a real title (see _UNINFORMATIVE).
TITLE_VETO = 45.0
# Air date: |days| <= 1: +4, <= 3: +3, <= 10: +1, <= 45: 0, farther: -3. Missing: 0.
DATE_STEPS = ((1, 4.0), (3, 3.0), (10, 1.0), (45, 0.0))
DATE_FAR = -3.0
POSITION_BONUS = 1.0  # same season and episode number
PRODCODE_BONUS = 3.0  # same production code
# Pairs across a season-0 boundary (specials placed in a season by one source) need a strong,
# informative title match and this total.
SPECIAL_MIN = 4.0
SPECIAL_TITLE_MIN = 85.0
AIR_DATE_CONFLICT_DAYS = 3
SPLIT_TITLE_MIN = 80.0  # primary title vs the parts' shared base
SPLIT_RUNTIME_TOL = 0.1  # parts' runtimes sum to the primary's within 10% (at least 2 min)
SPLIT_ORDERING = "aired-split"

# uppercase only: "Mix" or "Liv" is a word, "MIX" on TMDB is a numeral
_ROMAN = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_NUMWORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
)
_UNINFORMATIVE = [
    re.compile(p, re.I)
    for p in (
        r"^(episode|ep|chapter|part|pt|no|number|show|program|programme)[\s.#:_-]*"
        rf"(\d+([.x]\d+)?|[ivxlcdm]+|{_NUMWORDS})$",
        r"^#?\s*\d+([.x]\d+)?$",
        r"^s\d+\s*e\d+$",
        r"^\d+x\d+$",
        r"^(untitled|unknown|tba|tbd|tba episode|untitled episode|special|episode|"
        r"(unaired )?pilot( episode)?)$",
    )
]
_PART_NUM = rf"\d+|[ivx]+|{_NUMWORDS}"
_PART = re.compile(
    r"^(?P<base>.+?)[\s,:;.–—-]*"
    rf"(?:\(\s*(?:part|pt\.?)?\s*(?P<n1>{_PART_NUM})(?:\s*of\s*\d+)?\s*\)"
    rf"|\b(?:part|pt\.?)\s+(?P<n2>{_PART_NUM})(?:\s*of\s*\d+)?)\s*$",
    re.I,
)
_WORDNUM = {w: i + 1 for i, w in enumerate(_NUMWORDS.split("|"))}


def normalize_title(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).casefold().replace("&", " and ")
    t = re.sub(r"[^0-9a-z]+", " ", t).strip()
    return re.sub(r"^(the|a|an) (?=\S)", "", t)  # "The Birth of Evil" ~ "Birth of Evil"


def is_uninformative(title: str | None, show_title: str | None = None) -> bool:
    """A title that can't identify an episode: a Roman numeral, 'Episode N', 'TBA', ..."""
    t = (title or "").strip()
    if not t or (show_title and normalize_title(t) == normalize_title(show_title)):
        return True
    if _ROMAN.match(t):
        return True
    return any(p.match(t) for p in _UNINFORMATIVE)


def _part_number(tok: str) -> int | None:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    if tok in _WORDNUM:
        return _WORDNUM[tok]
    vals = {"i": 1, "v": 5, "x": 10}
    total = 0
    for i, c in enumerate(tok):
        v = vals[c]
        total += -v if i + 1 < len(tok) and vals[tok[i + 1]] > v else v
    return total


def split_part(title: str) -> tuple[str, int] | None:
    """'The Birth of Evil (1)' -> ('the birth of evil', 1); None without a part marker."""
    m = _PART.match((title or "").strip())
    if not m:
        return None
    base = normalize_title(m.group("base"))
    if not base:
        return None
    return base, _part_number(m.group("n1") or m.group("n2"))


def _ordinal(d: str | None) -> int | None:
    try:
        return date.fromisoformat(d).toordinal() if d else None
    except ValueError:
        return None


@dataclass
class _Ep:
    ep: Episode
    season: int
    number: int
    informative: bool
    titles: list[str]  # normalized: title (+ alt titles on the primary side; + base on the other)
    part: tuple[str, int] | None
    day: int | None


def _entries(ordering: Ordering | None, by_id: dict[str, Episode], show: str) -> list[_Ep]:
    out: list[_Ep] = []
    if ordering is None:
        return out
    for e in sorted(ordering.entries, key=lambda e: (e.season, e.episode, e.episode_id)):
        ep = by_id.get(e.episode_id)
        if ep is None:
            continue
        names = [ep.title, *ep.alt_titles]
        titles = [normalize_title(t) for t in names if not is_uninformative(t, show)]
        part = split_part(ep.title)
        if part:
            titles.append(part[0])
        out.append(
            _Ep(ep, e.season, e.episode, bool(titles), list(dict.fromkeys(titles)), part,
                _ordinal(ep.air_date))
        )
    return out


def _owner_max(m: np.ndarray, owners: list[int], axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Max over consecutive rows (axis 0) or columns (axis 1) that belong to the same owner."""
    o = np.asarray(owners)
    starts = np.flatnonzero(np.r_[True, o[1:] != o[:-1]])
    return np.maximum.reduceat(m, starts, axis=axis), o[starts]


def _title_matrix(ps: list[_Ep], ss: list[_Ep]) -> np.ndarray:
    """Best fuzz.ratio between any informative title of each pair; NaN when either has none."""
    out = np.full((len(ps), len(ss)), np.nan)
    pv = [(i, t) for i, p in enumerate(ps) for t in p.titles]
    sv = [(j, t) for j, s in enumerate(ss) for t in s.titles]
    if not pv or not sv:
        return out
    m = process.cdist([t for _, t in pv], [t for _, t in sv], scorer=fuzz.ratio,
                      dtype=np.float64)
    m, rows = _owner_max(m, [i for i, _ in pv], 0)
    m, cols = _owner_max(m, [j for j, _ in sv], 1)
    out[np.ix_(rows, cols)] = m
    return out


def _scores(ps: list[_Ep], ss: list[_Ep]) -> tuple[np.ndarray, np.ndarray]:
    """Pair scores (primary x secondary) and the raw title ratios (NaN = no title evidence)."""
    titles = _title_matrix(ps, ss)
    sc = np.where(
        np.isnan(titles), 0.0,
        np.clip((np.nan_to_num(titles) - TITLE_PIVOT) / 10.0, TITLE_MIN, TITLE_MAX),
    )
    if not ps or not ss:
        return sc, titles
    veto = np.nan_to_num(titles, nan=100.0) < TITLE_VETO
    pd = np.array([np.nan if p.day is None else p.day for p in ps], dtype=float)
    sd = np.array([np.nan if s.day is None else s.day for s in ss], dtype=float)
    d = np.abs(pd[:, None] - sd[None, :])
    dterm = np.full(d.shape, DATE_FAR)
    with np.errstate(invalid="ignore"):
        for limit, val in DATE_STEPS[::-1]:
            dterm[d <= limit] = val
    sc += np.where(np.isnan(d), 0.0, dterm)
    pnum = np.array([(p.season, p.number) for p in ps]).reshape(len(ps), 2)
    snum = np.array([(s.season, s.number) for s in ss]).reshape(len(ss), 2)
    same = (pnum[:, None, :] == snum[None, :, :]).all(axis=2)
    sc += np.where(same, POSITION_BONUS, 0.0)
    scodes: dict[str, list[int]] = {}
    for j, s in enumerate(ss):
        c = (s.ep.production_code or "").strip().casefold()
        if c:
            scodes.setdefault(c, []).append(j)
    for i, p in enumerate(ps):
        c = (p.ep.production_code or "").strip().casefold()
        for j in scodes.get(c, []) if c else []:
            sc[i, j] += PRODCODE_BONUS
    sc[veto] = np.minimum(sc[veto], MATCH_MIN - 1.0)
    return sc, titles


def _align(sc: np.ndarray) -> list[tuple[int, int]]:
    """Order-preserving alignment maximizing the summed pair score; gaps cost nothing and only
    pairs at MATCH_MIN or above may be matched. Ties prefer the diagonal, then a primary gap."""
    n, m = sc.shape
    if n == 0 or m == 0:
        return []
    rows = sc.tolist()
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row, prev, r = dp[i], dp[i - 1], rows[i - 1]
        for j in range(1, m + 1):
            best = prev[j] if prev[j] >= row[j - 1] else row[j - 1]
            s = r[j - 1]
            if s >= MATCH_MIN and prev[j - 1] + s >= best:
                best = prev[j - 1] + s
            row[j] = best
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        s = rows[i - 1][j - 1]
        if s >= MATCH_MIN and dp[i][j] == dp[i - 1][j - 1] + s:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


@dataclass
class _Split:
    p: _Ep
    parts: list[_Ep]
    base: str | None  # the parts' shared title (display form) when they have one


def _runtime_sum_ok(p: _Ep, parts: list[_Ep]) -> bool:
    rt = p.ep.runtime_min
    rts = [s.ep.runtime_min for s in parts]
    if not rt or any(not r for r in rts):
        return False
    if any(r > 0.75 * rt for r in rts):
        return False
    return abs(sum(rts) - rt) <= max(2.0, SPLIT_RUNTIME_TOL * rt)


def _near(p: _Ep, parts: list[_Ep]) -> bool:
    return p.day is not None and any(
        s.day is not None and abs(s.day - p.day) <= AIR_DATE_CONFLICT_DAYS for s in parts
    )


def _split_ok(p: _Ep, parts: list[_Ep]) -> str | None:
    """'base' or 'runtime' when `parts` look like the pieces of `p`, else None."""
    bases = [s.part for s in parts]
    if all(bases) and len({b[0] for b in bases}) == 1:
        nums = [b[1] for b in bases]
        if nums == list(range(nums[0], nums[0] + len(nums))):
            base = bases[0][0]
            if p.informative:
                if max(fuzz.ratio(t, base) for t in p.titles) >= SPLIT_TITLE_MIN:
                    return "base"
            elif _near(p, parts) or _runtime_sum_ok(p, parts):
                return "base"
    if _near(p, parts) and _runtime_sum_ok(p, parts):
        return "runtime"
    return None


def _find_split(p: _Ep, run: list[int], anchor: int | None, ss: list[_Ep]) -> list[int] | None:
    """Consecutive secondary indices from `run` (containing `anchor` if given) that split `p`;
    the longest window wins, then the earliest."""
    best: list[int] | None = None
    for length in range(len(run), 1, -1):
        for start in range(0, len(run) - length + 1):
            win = run[start:start + length]
            if anchor is not None and anchor not in win:
                continue
            if _split_ok(p, [ss[k] for k in win]):
                best = win
                break
        if best:
            break
    return best


def _detect_splits(
    ps: list[_Ep], ss: list[_Ep], pairs: list[tuple[int, int]]
) -> tuple[list[tuple[int, list[int]]], list[tuple[int, int]]]:
    """Primary episodes matched by two or more consecutive secondary episodes."""
    p2s = dict(pairs)
    s_used = {j for _, j in pairs}
    consumed: set[int] = set()
    splits: list[tuple[int, list[int]]] = []
    for i, p in enumerate(ps):
        if i in p2s:
            j = p2s[i]
            lo = j
            while lo - 1 >= 0 and lo - 1 not in s_used and lo - 1 not in consumed:
                lo -= 1
            hi = j
            while hi + 1 < len(ss) and hi + 1 not in s_used and hi + 1 not in consumed:
                hi += 1
            run, anchor = list(range(lo, hi + 1)), j
        else:
            before = [p2s[k] for k in range(i) if k in p2s]
            after = [p2s[k] for k in range(i + 1, len(ps)) if k in p2s]
            lo = before[-1] + 1 if before else 0
            hi = after[0] - 1 if after else len(ss) - 1
            run = [k for k in range(lo, hi + 1) if k not in s_used and k not in consumed]
            # keep only a consecutive stretch: windows are taken from it
            anchor = None
        if len(run) < 2:
            continue
        win = _find_split(p, run, anchor, ss)
        if win and all(b - a == 1 for a, b in zip(win, win[1:], strict=False)):
            splits.append((i, win))
            consumed.update(win)
    split_p = {i for i, _ in splits}
    kept = [(i, j) for i, j in pairs if i not in split_p and j not in consumed]
    return splits, kept


def _unordered_splits(
    ps: list[_Ep], ss: list[_Ep], done_p: set[int], done_s: set[int]
) -> list[tuple[int, list[int]]]:
    """Splits the ordered alignment can't reach because the sources order them differently
    (e.g. the parts sit at the end of the previous season in one source). Only the strong form
    counts here: consecutive unpaired secondary episodes whose titles share a part base that
    matches the primary's title (or, for an uninformative primary title, aired within days)."""
    out: list[tuple[int, list[int]]] = []
    used: set[int] = set()
    for i, p in enumerate(ps):
        if i in done_p:
            continue
        best: list[int] | None = None
        j = 0
        while j < len(ss):
            if j in done_s or j in used or ss[j].part is None:
                j += 1
                continue
            k = j
            while (k + 1 < len(ss) and k + 1 not in done_s and k + 1 not in used
                   and ss[k + 1].part is not None and ss[k + 1].part[0] == ss[j].part[0]):
                k += 1
            win = list(range(j, k + 1))
            if len(win) >= 2 and _split_ok(p, [ss[x] for x in win]) == "base":
                best = win
                break
            j = k + 1
        if best:
            out.append((i, best))
            used.update(best)
    return out


def _adapter_name(sec: PartialBundle) -> str:
    for pr in sec.provenance:
        if pr.adapter:
            return pr.adapter
    for key in sorted(sec.orderings):
        return sec.orderings[key].source
    return "secondary"


def _secondary_ordering(sec: PartialBundle) -> Ordering | None:
    aired = [k for k in sorted(sec.orderings) if sec.orderings[k].type == "aired"]
    if aired:
        return sec.orderings[aired[0]]
    return sec.orderings[sorted(sec.orderings)[0]] if sec.orderings else None


def _fold(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def _merge_episode(
    p: Episode, s: Episode, adapter: str, show: str
) -> tuple[Episode, int | None]:
    """Primary episode with the secondary's data added; also the air-date gap in days when it
    exceeds the conflict limit (the date is then left alone)."""
    upd: dict = {}
    known = {normalize_title(t) for t in [p.title, *p.alt_titles]}
    alts = list(p.alt_titles)
    for t in [s.title, *s.alt_titles]:
        n = normalize_title(t)
        if n and n not in known and not is_uninformative(t, show):
            alts.append(t.strip())
            known.add(n)
    upd["alt_titles"] = alts
    texts = {_fold(x.text) for x in p.synopses}
    syn = list(p.synopses)
    for x in s.synopses:
        f = _fold(x.text)
        if f and f not in texts:
            syn.append(Synopsis(source=adapter, text=x.text.strip()))
            texts.add(f)
    upd["synopses"] = syn
    for field in ("runtime_min", "production_code"):
        if getattr(p, field) in (None, "") and getattr(s, field) not in (None, ""):
            upd[field] = getattr(s, field)
    gap = None
    if p.air_date is None and s.air_date is not None:
        upd["air_date"] = s.air_date
    elif p.air_date and s.air_date:
        a, b = _ordinal(p.air_date), _ordinal(s.air_date)
        if a is not None and b is not None and abs(a - b) > AIR_DATE_CONFLICT_DAYS:
            gap = abs(a - b)
    for field in ("directors", "writers"):
        cur = list(getattr(p, field))
        seen = {_fold(x) for x in cur}
        for x in getattr(s, field):
            if x.strip() and _fold(x) not in seen:
                cur.append(x.strip())
                seen.add(_fold(x))
        upd[field] = cur
    cast = [c.model_copy() for c in p.guest_cast]
    by_actor = {_fold(c.actor): k for k, c in enumerate(cast)}
    for c in s.guest_cast:
        k = by_actor.get(_fold(c.actor))
        if k is None:
            if c.actor.strip():
                by_actor[_fold(c.actor)] = len(cast)
                cast.append(CastEntry(actor=c.actor.strip(), character=c.character))
        elif not cast[k].character.strip() and c.character.strip():
            cast[k] = cast[k].model_copy(update={"character": c.character})
    upd["guest_cast"] = cast
    return p.model_copy(update=upd), gap


def _label(e: _Ep) -> str:
    return f"S{e.season:02d}E{e.number:02d}"


def _custom_id(ep_id: str) -> str:
    return "custom:" + re.sub(r"[^0-9A-Za-z_-]+", "-", ep_id)


def _split_fix(
    season: int,
    season_splits: list[_Split],
    primary_aired: Ordering,
    aired_key: str,
    adapter: str,
) -> str:
    """Paste-ready overrides.yaml block (§6.6): the parts as new episodes and an ordering that
    inherits `aired_key` and replaces this season with the parts in place of the whole."""
    new_eps: dict[str, dict] = {}
    by_p = {sp.p.ep.id: sp for sp in season_splits}
    for sp in season_splits:
        for s in sp.parts:
            fields: dict = {"title": s.ep.title}
            if s.ep.runtime_min:
                fields["runtime_min"] = s.ep.runtime_min
            if s.ep.air_date:
                fields["air_date"] = s.ep.air_date
            if s.ep.production_code:
                fields["production_code"] = s.ep.production_code
            syn = [{"source": adapter, "text": x.text} for x in s.ep.synopses if x.text.strip()]
            if syn:
                fields["synopses"] = syn
            new_eps[_custom_id(s.ep.id)] = fields
    entries = []
    for e in sorted(
        (e for e in primary_aired.entries if e.season == season), key=lambda e: e.episode
    ):
        ids = [_custom_id(s.ep.id) for s in by_p[e.episode_id].parts] \
            if e.episode_id in by_p else [e.episode_id]
        for ep_id in ids:
            entries.append({"season": season, "episode": len(entries) + 1, "episode_id": ep_id})
    doc = {
        "new_episodes": new_eps,
        "orderings": {SPLIT_ORDERING: {"base": aired_key, "entries": entries}},
    }
    head = "".join(
        f"# {adapter} has {len(sp.parts)} episodes for {_label(sp.p)} {sp.p.ep.id} "
        f"{sp.p.ep.title!r}: " + ", ".join(repr(s.ep.title) for s in sp.parts) + "\n"
        for sp in season_splits
    )
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)
    return head + body


def _conflict(adapter: str, url: str | None, fetched_at: str | None, **kw: str) -> Provenance:
    return Provenance(adapter=adapter, url=url, fetched_at=fetched_at,
                      conflict={"kind": kw.pop("kind"), **{k: str(v) for k, v in kw.items()}})


def _merge_one(
    bundle: Bundle, sec: PartialBundle, fetched_at: str | None
) -> tuple[Bundle, list[Provenance]]:
    adapter = _adapter_name(sec)
    url = next((pr.url for pr in sec.provenance if pr.url), None)
    when = fetched_at or next((pr.fetched_at for pr in sec.provenance if pr.fetched_at), None)
    show = bundle.show.title
    aired_key = "aired" if "aired" in bundle.orderings else next(
        (k for k in sorted(bundle.orderings) if bundle.orderings[k].type == "aired"), None
    )
    conflicts: list[Provenance] = []
    if aired_key is None:
        return bundle, conflicts
    primary_aired = bundle.orderings[aired_key]
    P = _entries(primary_aired, bundle.episode_by_id(), show)
    S = _entries(_secondary_ordering(sec), {e.id: e for e in sec.episodes}, show)

    pairs: list[tuple[_Ep, _Ep]] = []
    splits: list[_Split] = []
    left_p: list[_Ep] = []
    left_s: list[_Ep] = []
    for special in (False, True):
        ps = [p for p in P if (p.season == 0) == special]
        ss = [s for s in S if (s.season == 0) == special]
        sc, _ = _scores(ps, ss)
        found = _align(sc)
        sp, found = _detect_splits(ps, ss, found) if not special else ([], found)
        splits += [_Split(ps[i], [ss[k] for k in win], None) for i, win in sp]
        pairs += [(ps[i], ss[j]) for i, j in found]
        done_p = {i for i, _ in found} | {i for i, _ in sp}
        done_s = {j for _, j in found} | {k for _, w in sp for k in w}
        if not special:
            more = _unordered_splits(ps, ss, done_p, done_s)
            splits += [_Split(ps[i], [ss[k] for k in win], None) for i, win in more]
            done_p |= {i for i, _ in more}
            done_s |= {k for _, w in more for k in w}
        left_p += [p for i, p in enumerate(ps) if i not in done_p]
        left_s += [s for j, s in enumerate(ss) if j not in done_s]

    # Episodes one source places elsewhere (specials inside a season, a moved episode):
    # unordered, strong informative title match only.
    if left_p and left_s:
        sc, titles = _scores(left_p, left_s)
        cands = sorted(
            (-float(sc[i, j]), i, j)
            for i, p in enumerate(left_p)
            for j, s in enumerate(left_s)
            if sc[i, j] >= SPECIAL_MIN
            and not np.isnan(titles[i, j]) and titles[i, j] >= SPECIAL_TITLE_MIN
        )
        used_p: set[int] = set()
        used_s: set[int] = set()
        for _, i, j in cands:
            if i in used_p or j in used_s:
                continue
            used_p.add(i)
            used_s.add(j)
            pairs.append((left_p[i], left_s[j]))
        left_p = [p for i, p in enumerate(left_p) if i not in used_p]
        left_s = [s for j, s in enumerate(left_s) if j not in used_s]

    merged = {e.id: e for e in bundle.episodes}
    for p, s in pairs:
        new, gap = _merge_episode(merged[p.ep.id], s.ep, adapter, show)
        merged[p.ep.id] = new
        if gap is not None:
            conflicts.append(_conflict(
                adapter, url, when, kind="air_date", season=p.season, primary=p.ep.id,
                secondary=s.ep.id,
                detail=f"{_label(p)} {p.ep.title!r} aired {p.ep.air_date} in the primary, "
                       f"{s.ep.air_date} in {adapter} ({gap} days apart); kept the primary's",
            ))

    # numbering disagreements: one per primary season
    by_season: dict[int, list[tuple[_Ep, _Ep]]] = {}
    for p, s in pairs:
        if p.season != 0 and s.season != 0:
            by_season.setdefault(p.season, []).append((p, s))
    for season in sorted(by_season):
        diff = [(p, s) for p, s in by_season[season]
                if (p.season, p.number) != (s.season, s.number)]
        if diff:
            p, s = min(diff, key=lambda x: x[0].number)
            conflicts.append(_conflict(
                adapter, url, when, kind="numbering", season=season, primary=p.ep.id,
                secondary=s.ep.id,
                detail=f"{len(diff)} of {len(by_season[season])} lined-up episodes are numbered "
                       f"differently in {adapter} (e.g. {_label(p)} is {_label(s)})",
            ))

    for season in sorted({sp.p.season for sp in splits}):
        group = [sp for sp in splits if sp.p.season == season]
        fix = _split_fix(season, group, primary_aired, aired_key, adapter)
        for sp in group:
            base = sp.parts[0].part
            if base and all(s.part and s.part[0] == base[0] for s in sp.parts):
                title = _PART.match(sp.parts[0].ep.title.strip()).group("base").strip(" ,:;.-")
                ep = merged[sp.p.ep.id]
                if normalize_title(title) not in {normalize_title(t)
                                                  for t in [ep.title, *ep.alt_titles]}:
                    merged[sp.p.ep.id] = ep.model_copy(
                        update={"alt_titles": [*ep.alt_titles, title]})
            conflicts.append(_conflict(
                adapter, url, when, kind="split", season=season, primary=sp.p.ep.id,
                secondary=", ".join(s.ep.id for s in sp.parts),
                detail=f"{_label(sp.p)} {sp.p.ep.title!r} is {len(sp.parts)} episodes in "
                       f"{adapter}: " + ", ".join(f"{_label(s)} {s.ep.title!r}"
                                                  for s in sp.parts),
                fix=fix,
            ))

    for p in left_p:
        if p.season != 0:
            conflicts.append(_conflict(
                adapter, url, when, kind="unaligned_primary", season=p.season, primary=p.ep.id,
                secondary="", detail=f"{_label(p)} {p.ep.title!r} has no partner in {adapter}",
            ))
    for s in left_s:
        if s.season != 0:
            conflicts.append(_conflict(
                adapter, url, when, kind="unaligned_secondary", season=s.season, primary="",
                secondary=s.ep.id,
                detail=f"{adapter} {_label(s)} {s.ep.title!r} has no partner in the primary; "
                       "not added (use overrides.yaml new_episodes)",
            ))

    show_info = bundle.show
    if sec.show is not None:
        fill = {k: v for k, v in sec.show.ids.model_dump().items()
                if v is not None and getattr(show_info.ids, k) is None}
        if fill:
            show_info = show_info.model_copy(
                update={"ids": show_info.ids.model_copy(update=fill)})
    out = bundle.model_copy(update={
        "show": show_info,
        "episodes": [merged[e.id] for e in bundle.episodes],
    })
    return out, conflicts


def merge_sources(
    primary: Bundle, secondaries: list[PartialBundle], *, fetched_at: str | None = None
) -> Bundle:
    """Primary bundle with each secondary's data added (spec §6.5). Returns a new Bundle with a
    fresh revision; earlier provenance from the secondaries' adapters is replaced, so merging the
    same source twice gives the same result."""
    bundle = primary.model_copy(deep=True)
    names = {_adapter_name(sec) for sec in secondaries}
    prov = [pr for pr in bundle.provenance if pr.adapter not in names]
    for sec in secondaries:
        bundle, conflicts = _merge_one(bundle, sec, fetched_at)
        prov += [pr.model_copy(deep=True) for pr in sec.provenance] + conflicts
    return bundle.model_copy(update={"provenance": prov}).with_revision()
