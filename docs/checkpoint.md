# showvenist: session checkpoint (2026-10-14)

Written so a fresh context can resume. The spec (`docs/spec.md`, decision log §19, decisions 38–68) stays the
source of truth; Beads (`bd ready`, `bd list`) holds the tasks; this file is the narrative state and the
operating notes. Read `CLAUDE.md`, then this, then `bd ready`.

## 1. Where the code is

- Branch **`main`** (head: see `git log`; everything through decision 68 is merged and pushed by the user).
  **Never push; the user pushes.** Subagents never commit. Topic branches (`m7-adapters`, `title-order`,
  `title-order-majority`, `series-cast-main`, `unrecognized-folders`, `not-found-report`) are merged and can be
  deleted.
- 1116 tests pass; `ruff check .` clean. Python is **`.venv/bin/python`**, `.venv/bin/showvenist`, `.venv/bin/ruff`
  (`uv` is no longer on the PATH; do not install anything).
- Milestones: **M0–M7 done** and reviewed. M7 (decision 62): TVmaze secondary source merged at ingest
  (`adapters/tvmaze.py`, `bundle/merge.py`: alignment, conflicts in `provenance[].conflict`, split detection with
  paste-ready overrides fix), `ingest --refresh [--diff]` by slug/stored id with diff + history impact and
  `show.json.prev`, `bundle edit [--apply-fix N]`, staleness notice. `--refresh` aborts when the old bundle had
  `ids.tvmaze` and TVmaze fails (`--source tmdb` drops it on purpose). Wikipedia skipped (`sv-opu.2`).
- **Since M7 (decisions 63–66), all driven by real media (Burn Notice S01/S02 copies, then all 13 bundles):**
  title-order tie-breaker with a majority-rule verdict (63, 65), two-episode files `S01E11+S01E12` (64, schema
  v3 `episode_end`; schema now **v4**, `rename_op.created_dir`), parser knows extras folders, confirmed extras
  move, copies of a confirmed episode become `_dupe_N`, **series cast is main cast only (66)**, a folder under the show that is not a season, disc or extras folder is
  flagged and skipped (67), and each season's report lists the episodes **not found** (68).
- Beads data is local (Dolt DB is gitignored); backup lives in `data/beads-backup/` (`bd backup sync`).

## 2. What exists (package `showvenist/`)

bundle (schema, store, overrides incl. `new_episodes`, inherited orderings and `title_card`, quality, ingest) ·
adapters (tmdb, localsubs, opensubtitles [mocked only, parked], llm_cues) · analysis (probe, tracks, subtitles,
ocr, windows, pipeline, duplicates, inspect, cache, **cardframes**) · scoring (duration, title, **titlecard**,
refsubs, synopsis, characters, cues, llm, candidates) · fusion · assign (basic, confidence, blocks, ordering, **titleorder**) ·
hierarchy (parse, marker, readiness) · session · report (+ **edgeflags**: advisory `play_all`,
`possible_two_parter`) · eval (harness, manifest, runner, fit, render, labelsheet, mkmanifest) · cli (`ingest`,
`bundle check`, `inspect [--card]`, `identify [--plan|--apply]`, `eval [--from-history]`, `assign`, `history`,
`forget`, `undo`) · **state** (migrations: schema v4, db with read-only open, history) · **rename** (types, naming,
plan, sidecars, movelog: journaled apply and crash recovery, undo, token, render). TUI (`tui/`). TVmaze + merge (M7).

**M5 (decisions 56–60).** Duplicates are never moved or deleted: the primary is renamed, other copies are renamed
in place `<episode>_dupe_N`. Extras are always prefixed `disc2__<name>` with `_2`, `_3` collision suffixes, and
nothing is overwritten (`renameat2(RENAME_NOREPLACE)`; cross-filesystem moves copy, fsync, verify the fingerprint,
link, then remove, and need a typed `yes`). `state.db` records only what is decided or executed (confirmed,
rejected, undone). The move log journals `planned`, then moves, then `done` plus the match in one transaction;
`recover` runs at the start of every moving command and confirms the match from the row. One `identify --apply` is
one run, so `undo --last` reverses it all. **Consent (decision 60):** `--budget USD` (default 0) is the only way to
allow paid LLM calls (`identify`, `eval`); `identify --plan` prints a plan token (`7f3a-9c1e`) and saves it;
`--apply` on a TTY asks for a typed `yes`, otherwise needs `--apply --confirm TOKEN|last`. `--yes` exists only on
`assign --rename`, `forget` and `undo`. Report, `--plan`, `history` and `eval --from-history` open `state.db`
read-only.

**Title card (decision 54/55).** A show describes an on-screen title card in `overrides.yaml` (`title_card`:
`windows` from either end, `pattern` with a named group `ref`, `kind` roman|number|title, `background`
dark|any). The tool never guesses; no config means no video is touched. Frames are sampled at ~1 fps in the
window, mostly-black ones are OCR'd, results cached in `analysis.db` (extractor `cardframes`, keyed by windows +
background, not by the pattern). The scorer matches the `ref` against candidate titles as a closed set, needs two
agreeing frames or one exact match, is for-only (+8, never negative) and is summed in both regimes. A run of
exactly four I is read as three (OCR over-counts a trailing III). `inspect FILE --show SLUG --card` prints the OCR
text per frame (ad hoc windows with `--from/--skip/--length`). Eval adds a Problem when a card disagrees with a
label (an audit; metrics unchanged). Samurai Jack is configured (end window 180 s, `EPISODE\s+(?P<ref>[IVXLC]+)`).

## 3. Key measured results (all on the user's library, `/mnt/storage/tv`, read-only)

**Important correction (2026-10-09, `sv-cn7`):** the eval used to leak file order (ties fall back to file order,
and labeled files are listed in episode order). Eval now sorts and shuffles each (show, season) group with a
seed. Coverage and errors at high were never affected (ties only reach low confidence); **top-1 numbers quoted
in earlier checkpoints were inflated**, most for LLM-off runs.

- **Latest (2026-10-13, `PYTHONHASHSEED=0`, LLM off, 538 files, all bundles refreshed with TVmaze and the
  main-cast filter):** top-1 **76.4%**, coverage at high **24.9%**, errors at high **0/132 (≤2.2%)**, unmatched
  precision 90.0% / recall 83.7%; reliability high 100%, medium 95.3%, low 56.2%. Steps (files in `data/`):
  `eval-before-refresh.txt` 51.5% / 8.0% (Burn Notice already on TVmaze) → `eval-after-refresh.txt` 58.6% / 8.2%
  (TVmaze for the other 12 shows) → `eval-after-castfix.txt` 76.4% / 24.9% (decision 66). Per show top-1: West
  Wing 88.9, Hulk 85.4, Firefly 92.9, Samurai Jack 83.3, Babylon 5 78.4, Supernatural 73.8, Wire 68.8, Burn
  Notice 65.9, Expanse 60.0, Invisible Man 50.0, Farscape 18.2, White Shadow 0 (3 files). **Costs:** extras
  recall fell 95.3% → 83.7% (Burn Notice 95% → 70%) and medium-confidence accuracy 100% → 95.3%; the Burn Notice
  extras that flipped are not yet examined. The 15%/floor 3 threshold was a guess and has not been swept.
- **Cause of the cast fix:** `series_cast` held TMDB's whole aggregate credits (526 people for Babylon 5, 2,167
  for Supernatural), so guest-name and synopsis evidence was thrown away for many guests, and a refresh that
  swapped a few entries flipped results (Babylon 5 S03E18). Now only people with `episodes` ≥ max(3, 15% of the
  episodes) count; old bundles without a count keep every entry until refreshed. Eval on Babylon 5 after the
  TVmaze refresh alone showed the other effect of a refresh: S01 runtimes all 45 → 44, so low-confidence files
  reshuffle (noise).
- **Library-wide plan-only run (2026-10-13/14, `identify SHOW --plan`, LLM off, nothing applied):** reports in
  `data/logs/library-plan/<slug>.txt` (+ `.err`, `json/<slug>.json`), driven by `run-all.sh`/`run-two.sh`/
  `run-json.sh` through `/home/jude/realrun/run.sh`; `check.py SLUG` compares a plan with the labeled manifest and
  `spot.py` lists the high decisions. Supernatural 758 files took 3,118 s cold; Babylon 5 938 s, Farscape 880 s,
  Samurai Jack 852 s, Expanse 459 s. Counts (high/medium/low): Supernatural 369/154/183 (332 moves), Samurai Jack
  62/11/5, Expanse 14/2/8, Babylon 5 53/28/28, Farscape 29/14/34; Burn Notice 214 files (S01 is not blind: 42
  files `[confirmed]` from the copy-run history); Hulk, Invisible Man, West Wing, Wire and Firefly are already
  hand-named. All 70 labeled files at high are right. **Spot check** of ~570 non-named, non-confirmed high
  decisions: every episode decision has content evidence (≥ +2 from synopsis, guests or title) except 5 (2 are
  exact copies of confirmed Burn Notice files; Supernatural S04E18 and two S13E12 files at +1 to +2); no episode is
  claimed twice at high by non-duplicates. About 25 Supernatural extras of 18–75 min are called extras at high
  with every signal 0.0 (worth a look), plus a few elsewhere (Burn S04 disc4 `C5_t06`, Expanse S03 disc3 `t08`,
  Babylon 5 S05 disc5 `t00`).
- **Gaps found by that run (episodes no file was assigned to):** Supernatural S05 E06/E10/E22 (failed rips; every
  other Supernatural season is complete); Farscape S01 and S02 E18–E22 (their `disc5` folders are **empty**, so
  those discs were never ripped) and S03 E04 (disc 1 has `t00`, `t02`–`t04`; `t00` is E03 or E04 at 1.0:1 with no
  signal separating the two halves of "Self-Inflicted Wounds", so check it by eye and `assign`); Babylon 5 S02
  E14/E20 (the `Season02/temp` folder was a failed-rip re-do; the user deletes such folders); Expanse S01 E10;
  Burn Notice S02 E16 (the known disc-4 miss, not a missing rip) and S07 (empty folder). Samurai Jack's
  first plan used `aired` by mistake (identify needs `--order disk`: the files carry no disc, so ordering detection
  cannot switch); re-run with `--order disk` (`data/logs/library-plan/samurai-jack-disk.*`): nothing not found, S03
  11/2/3 high/medium/low, S04 10/0/4 low; the lows are the Birth of Evil and Scotsman Saves Jack halves, whose
  custom episodes carry identical synopses. Incomplete seasons are not modeled beyond one skipped title per
  disc (slack 1, `u` = 2): two or more missing on a disc, or a whole missing disc, may pull weak files into empty
  slots. Not measured.
- **Remaining refresh conflicts (read 2026-10-15; no bundle edits needed):** Expanse (a special's air date, 12 days)
  and West Wing (S07E08 air date, 21 days) are date-only. Invisible Man S02: two air dates plus a numbering swap
  (TVmaze has TMDB's E15 as E16); the library holds only `s02e01`, so no effect yet; revisit if E15–E17 are ripped
  and come out wrong. Hulk S01: TVmaze lists "Death in the Family"; TMDB has no such title and its E03 is "Final
  Round" dated 1978-03-10 (possibly the date of "Death in the Family"; unconfirmed). 12 TMDB episodes, 12 hand-named
  files, nothing unmatched. Do not add it via `new_episodes` before the real Season 1 list is known (it would shift
  later numbers); check the early regular episodes by eye if Hulk raw rips appear.
- **Burn Notice real runs (copies in `/home/jude/realrun/`, `run.sh` isolates state/cache/config):** S01 (66
  files, 12 episodes) and S02 (26 files, 16 episodes) with a ground-truth manifest each
  (`data/manifest-burn-s01.csv`, `-s02.csv`). The tool never made a confident mistake. S01 plan/apply/undo on the
  copy: 44 moves, 0 failed, undo restored 66 files. S02 through the real identify path with S01 history:
  13/16 right (the three wrong are all on disc 4, which is not in title order). Combined redundant files are
  called extras (accepted failure mode). TMDB runtimes for S02–S07 are placeholders.
- **LLM off, 538 files (409 hand-named + labeled raw rips), shuffled (OLD, 2026-10-09):** top-1 **50.2%**, coverage at high 8.0%,
  errors at high 0/43 (≤6.7%), unmatched precision 82.0% / recall 95.3%. (Before the fix: 70.6%.)
- **LLM on (cached answers, season scope), 401 hand-named files, blocks on:** top-1 ≈ 91% shuffled (94.8%
  unshuffled), coverage at high **71.8%** (288 files), **2 errors at high** (bound 2.2%; the gate bar is 1%),
  both the Wire S03E05/E06 swap inside one disc at Δ 8.2 and 7.5. The Δ band 4.6–6.3 now holds 48 files with
  0 wrong (it was ~17% wrong at the gate; the block model fixed the Wire S01 cases). Raising `HIGH_DELTA` to 6.3
  would cost 12 points of coverage and fix nothing, so that fallback was dropped. Three seasons (Hulk S05, West
  Wing S03/S04) have no cached LLM answers for the current settings and were scored without the LLM in this check.
- **Samurai Jack with cards (12 labeled files):** LLM off 83.3% top-1, 66.7% coverage at high, 0/8 errors; cached
  LLM 83.3%, 75.0%, 0/9. Before cards: 25% top-1 LLM off. The card was read in 9 of 10 episode files (the tenth is
  half of a split two-parter and has none). The two earlier "high-confidence errors" were label errors (aired
  numbering under the `disk` ordering); the two labels were corrected in the user's sheet.
- **Weak shows with the LLM (60 files: Babylon 5, Farscape, Samurai Jack):** $0.65 spent, top-1 90%, 0/46 errors
  at high. Farscape's and Babylon 5's low LLM-off scores were missing evidence, not bugs.
- **Weight fit** (`eval --fit`) would double coverage at high but add 3 errors in 65: not adopted.
- **Ablation** (538 files, LLM off): removing duration kills accuracy; guests and titlecard help; cues change
  nothing because no bundle has generated cues (generation is paid). The apparent "synopsis hurts" result was the
  order leak.
- **`u` (unclaimed penalty)** swept 1–4: flat; `u = 2` kept (`sv-o2p.8` closed).
- **LLM cost:** about $0.0174 per file, $0.42 per 24-episode season; every response is cached in
  `~/.cache/showvenist/analysis.db` (re-runs are free). Total LLM spend so far ≈ $9.3.
- **Performance:** a 22-file Blu-ray season with sparse subtitles takes ~194 s cold on local disks (meets §8.6),
  ~7 min at gigabit speed (misses it; deferred, `sv-o2p.7`). Card reads cost ~5.5 s per file cold, <1 s warm.
- **Extras:** about 27% of raw rips, almost all short clips and play-alls; zero near-episode-length extras among
  69 labeled near-length rows. The 12 labeled play-alls (`extra-multi`) are decided as extras 11 of 12 times, 0 taken
  for an episode at high. `play_all` (needs reference-subtitle window scores) does not fire in this library.

## 4. Decisions (spec §19; 38–68, read them there)

38 M0 outcome 3 · 39 `wordfreq` OCR word check · 40 LLM evidence = `logit(p)+log(n−1)` · 41 `episode?` labels ·
42 duration scorer · 43 OpenSubtitles live use parked · 44 M3 gate passed with a documented deviation · 45 gate
wording; `llm.mode` stays `off` · 46 ordering detection solves every ordering · 47 ordering threshold 15 · 48 block
params (slack 1, u=2, margin 4, threshold 15) · 49 `extra-multi` label · 50 two-part label `S01E01+S01E02` · 51
`.showvenist-ignore` folders · 52 near-duplicate detection by dialogue overlap · 53 overrides `new_episodes` and
inherited orderings · **54 title-card scorer (opt-in per show via `overrides.yaml`, closed-set match, for-only)** ·
**55 a run of exactly four I in a Roman reading is read as three** · 56 duplicates renamed in place (`_dupe_N`) ·
57 extras prefix and collision suffix · 58 CLI plan/apply before the TUI · 59 `state.db` records decided/executed only ·
**60 `--budget` and `--confirm` replace `--yes`** · 61 M6 TUI first version · 62 M7 order (TVmaze, refresh,
`bundle edit`) · **63 title-order tie-breaker** (0.3·min(d²,4) per displaced rank, only when evidence supports it;
`[blocks] title_order_weight`, 0 disables) · **64 two-episode files** (`assign S01E11+S01E12`, named `s01e11-e12`) ·
**65 title-order verdict is a majority rule** (≥ 2/3 of discs with pairs concordant; copies are not evidence) ·
**66 series cast is main cast only** (stored episode counts; ≥ max(3, 15% of episodes)) ·
**67 unrecognized show-level folders are flagged and skipped** (not a parse problem) · **68 the report lists
episodes not found per season** (`Summary.not_found`, informational only). Also in the spec now: §16.1 file-order rule
(eval shuffles), §14 note on the advisory `play_all`/`possible_two_parter` flags.

## 5. Standing rules and operating notes

- CLAUDE.md rules: commit each integrated wave on the milestone branch without asking; **never push**; **no
  money without permission** (give the estimate; the user approved specific budgets each time); **ask before
  installing anything**; `/mnt/storage/tv` is READ-ONLY (the user creates files there, e.g. `.showvenist-ignore`);
  no copyrighted media in tests/git; `data/` is gitignored.
- The user wants a partner: recommend, question, one topic at a time; a rejected prompt means "let me talk".
- Paid LLM runs: key is in `~/.config/showvenist/claude.api.key`; run with
  `ANTHROPIC_API_KEY="$(cat ~/.config/showvenist/claude.api.key)" .venv/bin/showvenist eval … --llm auto --budget N`
  (never print the key). **The auto-mode classifier blocks any Bash command that puts the key in the environment,
  even for a cached-only run; ask the user to run such a command with `!`.** Without `--budget`, eval uses the LLM
  only where every answer is cached, so a re-run costs nothing; `LlmScorer` can be built without a key for that.
  TMDB token: `~/.config/tmdb/api.token` → `SHOWVENIST_TMDB_API_KEY="$(cat …)"` for `ingest`.
- Env overrides: `SHOWVENIST_<SECTION>_<FIELD>` (e.g. `SHOWVENIST_LLM_SCOPE=season`,
  `SHOWVENIST_BLOCKS_UNCLAIMED_PENALTY=3`). Cached LLM answers match with `SHOWVENIST_LLM_SCOPE=season`.
- The classifier also blocked editing the user's label sheet once; the user edits their own data files.
- Shell gotchas: never `pgrep -f`/`pkill -f` a pattern that appears in your own command line; the safety check
  blocks `rm` inside loops; long evals go in the background with `setsid nohup … &` and a bounded `for` wait
  (a foreground command times out at 10 minutes). `bd create --silent` does not print the new ID; create the epic
  first, read its ID with `bd list`, then pass `--parent`.
- Subagent method (docs/plan.md §1): orchestrator writes the contract; parallel agents on disjoint files (Opus
  for subtle logic, Sonnet for features, Haiku for mechanical); one Opus review per milestone; reports ≤ 15 lines.
  Review the diff and the full suite yourself before committing a wave.
- Scratch scripts live in the session scratchpad (`perfile.py` prints per-file results for a show, `sweep.py`
  collects Δ and correctness per file for threshold sweeps, `flags.py`, `card.py`); recreate if needed.

## 6. Data kept in `data/` (gitignored; recreate from the commands in §7)

`bundles/<slug>/show.json` (each refreshed 2026-10-13, `show.json.prev` kept; a pre-cast-fix copy is in
`/home/jude/realrun/bundles-before-castfix`) for 13 shows (babylon-5-1994, burn-notice-2007, cowboy-bebop-1998, farscape-1999,
firefly-2002, samurai-jack-2001, supernatural-2005, the-expanse-2015, the-incredible-hulk-1977,
the-invisible-man-2000, the-west-wing-1999, the-white-shadow-1978, the-wire-2002); Samurai Jack has an
`overrides.yaml` (custom ordering `disk`; Birth of Evil = S03E11/E12, Scotsman Saves Jack = S04E06/E07; and the
`title_card` block; backup of the earlier file: `overrides.yaml.bak-before-card`). Manifests: `manifest-named.csv`
(410 hand-named files), `manifest-all.csv` (538 rows incl. the labeled sample, regenerated from the corrected
labels). Sheets: `extras-sheet.csv` (1,314 unnamed rips), `extras-sheet-sample.csv` (175 stratified),
`extras-sheet-sample-labelled.csv` (the user's labels, 108; Babylon 5 S05 and two Samurai Jack rows corrected),
`extras-sample-priority.csv`. Eval outputs: `eval-*.txt` (latest: `eval-rebase-off.txt`, `eval-sj-card2-*.txt`,
`eval-weak-llm2.txt`), older `thr*`, `final-*`, `cmp-*`. Real-library facts for parser tests: `title-tags.tsv`,
`library-tree.txt`. Beads backup: `beads-backup/`. Also `manifest-burn-s01.csv`, `manifest-burn-s02.csv`, `eval-before-refresh.txt`,
`eval-after-refresh.txt`, `eval-after-castfix.txt`, `logs/refresh-all-20261013.txt` (the refresh loop output), `logs/library-plan/` (the
plan-only library run).

## 7. Commands that reproduce the current state

```
# manifest from the user's labels (sets order=disk for Samurai Jack, blu-ray-order for West Wing S03,
# intended-order for Firefly automatically)
.venv/bin/python -m showvenist.eval.mkmanifest /mnt/storage/tv --sheet data/extras-sheet.csv \
  --sample data/extras-sheet-sample-labelled.csv --bundles data/bundles -o data/manifest-all.csv
# free eval (LLM off), blocks on, files shuffled; add --compare-blocks, --show-errors, --fit, --ablate
.venv/bin/showvenist eval data/manifest-all.csv -o data/bundles --library /mnt/storage/tv --show-errors
# refresh a bundle (needs `export SHOWVENIST_TMDB_API_KEY=...`; keeps show.json.prev); --diff previews
.venv/bin/showvenist ingest SLUG --refresh --yes -o data/bundles
# reproducible eval: ties are shuffled with a seed derived from a string hash (sv-tz9), so pin it
PYTHONHASHSEED=0 .venv/bin/showvenist eval data/manifest-all.csv -o data/bundles --library /mnt/storage/tv
# cached-only LLM (free)
SHOWVENIST_LLM_SCOPE=season .venv/bin/showvenist eval data/manifest-named.csv -o data/bundles \
  --library /mnt/storage/tv --llm auto
```

## 8. Next steps (in this order; the user decides)

1. **Burn Notice extras recall** (95% → 70% after decision 66). Cause found: 5 of 8 combined/play-all
   (`extra-multi`) files are now decided as episodes (3 low, 2 medium, none high), because a combined file really
   hears the guests of the episodes it holds and the duration scorer treats a double-length file as "episode +
   next". A possible rule, not built: a file whose length fits an adjacent pair is an extra when both episodes
   are already held by confident single files. Sweep the cast share (10/15/20%) only if this does not settle it.
2. **Review the library plan** before any apply: Farscape S03 disc 1 `t00` by eye; the ~25 Supernatural extras of
   18–75 min with no signal; Samurai Jack's proposed S04E04 split (do not run `--apply-fix 3` blindly). The Expanse, West Wing,
   Invisible Man and Hulk refresh conflicts are read (see §3). Then decide on apply, on copies first.
   Optional: measure gap pressure on a season with known holes (Supernatural S05, Babylon 5 S02); a new M3-gate run.
3. Small: `sv-tz9` (eval tie shuffle depends on `PYTHONHASHSEED`), `sv-36h` (trailing `_S01` in a folder name).
   Parked: Wikipedia (`sv-opu.2`), OCR survival eval (`sv-opu.5`, no reference subtitles; keeps epic `sv-opu`
   open), OpenSubtitles (decision 43). Older: `sv-60p.13`, `sv-60p.15`, `sv-31q`, `sv-o2p.7`, `sv-o2p.9`.
4. Known gaps: a newer run's move links to the older run's match; no CLI test applies two seasons in one
   invocation; byte-identical copies share a fingerprint; near-duplicates of a confirmed file are not found (only
   exact content-key copies); a file already named `s01e11-e12` is read as E11 alone without history; `assign` of
   another file to a held second episode gives a note, not a refusal; the bundle diff's history impact compares
   only the first episode's number; a v2 `state.db` upgrades only on the first writing command; runs applied
   before schema v4 have no `created_dir` (undo removes no folders for them); the staleness notice lacks
   `-o ROOT`; `--apply-fix` with an indentless YAML list and trailing comment; `report.render_tables` may
   mishandle bracketed names; TUI `L` deferred. S02 disc 4 of Burn Notice (out of title order) is a known miss.
5. Optional: a season with a dedicated extras disc (the one disc shape not yet seen); a paid eval with the LLM
   (estimate first, ask for the budget); generalize title cards to other shows (config only).

## 9. Waiting on the user

- **Applying to the real library (2026-10-15):** the read-only rule on `/mnt/storage/tv` is lifted one show at a
  time, by the user naming the show, and returns when all are done. Procedure: plan with the real state dir and the
  warm cache (`XDG_CACHE_HOME=/home/jude/realrun/cache`, no `run.sh`), show the move list, get the user's go, apply
  with `--confirm TOKEN`; then **repeat plan + apply on the same show until a plan has no moves** (pinned files
  clear others: The Expanse took 3 passes, 20/4/1 moves; stop on any failure or new kind of change). Only high
  confidence moves; lows stay. Done: The Expanse. Next: Samurai Jack (`--order disk`), Babylon 5, Farscape,
  Supernatural, then Burn Notice. Paid LLM pass afterwards, total budget $29, estimate first (Season 9 pilot: $0.30
  for 18 files, all 34 picks unchanged and all high).

- Real-media runs need the user's shell for anything TMDB: my shell has no `SHOWVENIST_TMDB_API_KEY`, `run.sh`
  isolates XDG config, and the variable must be `export`ed (the user runs `ingest` with `!`).
- Farscape: the user moved `Peacekeeper_Wars` out of the show; Farscape S01/S02 `disc5` are empty (not ripped).
  Babylon 5 `temp` folders are the user's to delete (decision 67 now skips any such folder with a flag).
- The Wire S03E05/E06 swap inside one disc is left as a manual fix (the user's decision; revisit if the error
  turns out to be general).
- The user pushes branches; ask before assuming the remote state.
