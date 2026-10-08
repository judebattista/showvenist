# showvenist: session checkpoint (2026-10-08)

Written so a fresh context can resume. The spec (`docs/spec.md`, decision log §19, decisions 38–53 were added
in this stretch) stays the source of truth; Beads (`bd ready`, `bd list`) holds the tasks; this file is the
narrative state and the operating notes. Read `CLAUDE.md`, then this, then `bd ready`.

## 1. Where the code is

- Branch **`m4-hierarchy`** (head `f811951`). Earlier branches are linear ancestors: `m0-spike` → `m1-bundle` →
  `m3-scoring` → `m4-hierarchy`. **Nothing has been pushed; the user pushes.** Subagents never commit.
- 750 tests pass; `ruff check .` clean. Python is **`.venv/bin/python`**, `.venv/bin/showvenist`, `.venv/bin/ruff`
  (`uv` is no longer on the PATH; do not install anything).
- Milestones: **M0–M4 done** and reviewed (one Opus conformance review each, findings fixed). **M5 (state and
  renames) is next and ON HOLD** by the user ("hold off on M5 for now"). M6 (TUI) and M7 (more adapters) follow.
  M5/M6/M7 epics exist in Beads (`sv-o4f`, `sv-zrm`, `sv-opu`).
- Beads data is local (Dolt DB is gitignored); backup lives in `data/beads-backup/` (`bd backup sync`).

## 2. What exists (package `showvenist/`)

bundle (schema, store, overrides, quality, ingest) · adapters (tmdb, localsubs, opensubtitles [mocked only, parked],
llm_cues) · analysis (probe, tracks, subtitles, ocr, windows, pipeline, duplicates, inspect, cache) ·
scoring (duration, title, refsubs, synopsis, characters, cues, llm, candidates) · fusion · assign (basic,
confidence, blocks, ordering) · hierarchy (parse, marker, readiness) · session · report · eval (harness,
manifest, runner, fit, render, labelsheet, mkmanifest) · cli (`ingest`, `bundle check`, `inspect`, `identify`,
`eval`). Not built: state.db/history, rename plan/move log/undo (M5), TUI (M6), TVmaze/Wikipedia/merge (M7).

## 3. Key measured results (all on the user's library, `/mnt/storage/tv`, read-only)

- **No LLM, no refsubs, 409 hand-named files:** top-1 57%, 1 file reaches "high". With disc blocks: 77%.
- **LLM on (cached answers, season scope), 409 files:** top-1 94.9% with blocks (86.8% without), coverage at high
  72.4% (61.6%), errors at high 2/280 (bound 2.2%). The 2 errors are the Wire S03E05/E06 swap inside one disc
  (disc structure cannot catch swaps within a disc). Known calibration gap recorded (decision 44).
- **Latest run, `data/eval-all-off.txt` (LLM off, 538 files incl. the labeled raw rips):** top-1 69.1%, coverage at
  high 6.5%, errors at high 0/35, unmatched precision 82.0% / recall 95.3%, `extra-multi`: 12 files, 11 decided
  extra, 0 taken for an episode at high. Weak shows in that run: Farscape (11 labeled rows, top-1 18%) and Samurai
  Jack (12 rows, top-1 25%): NOT yet investigated (likely the `disk` ordering / overrides, season-less flat
  folders, or thin synopses). Two Babylon 5 S05 rows have a label-vs-folder season mismatch (labels S04E11 and
  S04E20 under `Season05`): ask the user or fix the labels.
- **LLM cost:** about $0.0174 per file, $0.42 per 24-episode season; the whole-library run cost $6.74; all
  responses are cached in `~/.cache/showvenist/analysis.db` (re-runs are free). Total LLM spend so far ≈ $8.6.
- **Performance:** a 22-file Blu-ray season with sparse subtitles takes ~194 s cold on local disks (meets §8.6),
  ~7 min at gigabit speed (misses it; deferred, task in Beads).
- Extras share of raw rips (user-labeled sample, 108 labeled of 157 usable): about **27%**, almost all short
  clips and play-alls. **Zero near-episode-length extras** among 69 labeled near-length rows; the hard cases are
  the 19 "long" rows (play-all `extra-multi` vs genuine two-part files).

## 4. Decisions made in this stretch (spec §19, 38–53; read them there)

38 M0 outcome 3 · 39 `wordfreq` OCR word check · 40 LLM evidence = `logit(p)+log(n−1)`, p clipped [0.01,0.99],
`p_none` not evidence · 41 `episode?` labels, provable-error accounting · 42 duration scorer (batch-calibrated
scale, slot component, double-length component) · 43 OpenSubtitles live use parked (5–20 downloads/day) ·
44 M3 gate passed with a documented deviation (adjacent-episode errors at high) · 45 gate wording; `llm.mode`
stays `off` (spending is a conscious choice, every paid call needs `--yes`) · 46 ordering detection solves every
ordering · 47 ordering threshold 15 · 48 block params: slack 1, unclaimed penalty u=2, margin 4, threshold 15 ·
49 `extra-multi` label (play-all, redundant with the episodes, scored as extra, moved to `extras/`, never
deleted) · 50 two-part label `S01E01+S01E02` · 51 `.showvenist-ignore` folders · 52 near-duplicate detection
by dialogue overlap (containment ≥ 0.5, ≥ 100 phrases; never by duration) · 53 overrides `new_episodes` and
inherited orderings.

## 5. Standing rules and operating notes

- CLAUDE.md rules: commit each integrated wave on the milestone branch without asking; **never push**; **no
  money without permission** (give the estimate; the user approved specific budgets each time); **ask before
  installing anything**; `/mnt/storage/tv` is READ-ONLY (the user creates files there, e.g. `.showvenist-ignore`);
  no copyrighted media in tests/git; `data/` is gitignored.
- The user wants a partner: recommend, question, one topic at a time; a rejected prompt means "let me talk".
- Paid LLM runs: key is in `~/.config/showvenist/claude.api.key`; run with
  `ANTHROPIC_API_KEY="$(cat ~/.config/showvenist/claude.api.key)" .venv/bin/showvenist eval … --llm auto --yes`
  (never print the key; the shell's own export does not reach the tool). TMDB token: `~/.config/tmdb/api.token`
  → `SHOWVENIST_TMDB_API_KEY="$(cat …)"` for `ingest`. Both are free/approved only as stated by the user.
- Env overrides: `SHOWVENIST_<SECTION>_<FIELD>` (e.g. `SHOWVENIST_LLM_SCOPE=season`, `SHOWVENIST_BLOCKS_BLOCK_SLACK=1`).
  With `SHOWVENIST_LLM_SCOPE=season` and `--llm auto` and no `--yes`, eval uses the LLM only where every answer is
  cached (so a re-run is free); per-disc LLM windows change the candidate block and would miss the cache.
- Shell gotchas that bit this session: never `pgrep -f`/`pkill -f` a pattern that appears in your own command line
  (it matches itself and hangs or kills the shell); the safety check blocks `rm` inside loops (avoid it; use
  fresh file names); long evals go in the background with `setsid nohup … &` and a bounded `for` wait
  (a foreground command times out at 10 minutes).
- Subagent method (docs/plan.md §1): orchestrator writes the contract; parallel agents on disjoint files (Opus
  for subtle logic, Sonnet for features, Haiku for mechanical); one Opus review per milestone; reports ≤ 15 lines.

## 6. Data kept in `data/` (gitignored; recreate from the commands in §7)

`bundles/<slug>/show.json` for 13 shows (babylon-5-1994, burn-notice-2007, cowboy-bebop-1998, farscape-1999,
firefly-2002, samurai-jack-2001, supernatural-2005, the-expanse-2015, the-incredible-hulk-1977,
the-invisible-man-2000, the-west-wing-1999, the-white-shadow-1978, the-wire-2002); Samurai Jack has an
`overrides.yaml` installed from `data/samurai-jack-overrides-draft.yaml` (custom ordering `disk`; Birth of Evil
= S03E11/E12, Scotsman Saves Jack = S04E06/E07). Manifests: `manifest-named.csv` (410 hand-named files),
`manifest-all.csv` (538 rows incl. the labeled sample). Sheets: `extras-sheet.csv` (1,314 unnamed rips, 1,032
unique), `extras-sheet-sample.csv` (175 stratified), `extras-sheet-sample-labelled.csv` (the user's labels:
108), `extras-sample-priority.csv`. Eval outputs: `eval-*.txt`, `thr*`, `slack-*`, `final-*`, `cmp-*`.
Real-library facts for parser tests: `title-tags.tsv`, `library-tree.txt`. Beads backup: `beads-backup/`.

## 7. Commands that reproduce the current state

```
# manifest from the user's labels (sets order=disk for Samurai Jack, blu-ray-order for West Wing S03,
# intended-order for Firefly automatically)
.venv/bin/python -m showvenist.eval.mkmanifest /mnt/storage/tv --sheet data/extras-sheet.csv \
  --sample data/extras-sheet-sample-labelled.csv --bundles data/bundles -o data/manifest-all.csv
# free eval (LLM off), blocks on; add --compare-blocks, --show-errors, --fit, --ablate
.venv/bin/showvenist eval data/manifest-all.csv -o data/bundles --library /mnt/storage/tv --show-errors
```

## 8. Next steps (in this order; the user decides)

1. **Read `data/eval-all-off.txt` fully and report** (it finished): extras metrics, the 19 long rows, per-show
   results; investigate **Farscape (top-1 18%) and Samurai Jack (25%)**. Check `--show-errors`, the `disk`
   ordering, the flat season-tag folders and the Babylon 5 S05 label/season mismatch.
2. **Re-tune the unclaimed penalty u** (Beads `sv-o2p.8`): now possible with labeled extras. Zero near-length
   extras exist, so the u-vs-prior concern is mostly moot; confirm with the eval.
3. Optional cheap addition: fill the 35 skipped Supernatural near-length rows with `episode?` (more "must not
   become an extra" tests); the user said it is optional.
4. A paid eval with the LLM over the labeled raw rips (estimate first, ask for the budget) if the user wants the
   with-LLM numbers on them; per-disc LLM windows are not exercised by the cached runs.
5. **M5 design pass with the user** (state.db, history prior, `assign/history/forget/undo`, move plan, move log,
   crash recovery). Open design questions to bring: what the plan does with duplicates (exact and near; keep
   the best copy, move the rest to `extras/`, never delete?), `extra-multi` and `.showvenist-ignore` handling in
   the plan, safety rules for the first command that touches the user's files. Start with the contract.
6. Deferred in Beads: adjacent-episode calibration (`sv-60p.16`), sparse-subtitle read cost, loose-file + disc
   block model, closed captions, report `available`/flags (`sv-60p.14`), duration σ with `season=None`
   (`sv-31q`), near-duplicate scoring in the harness (`sv-60p.15`), cues shares evidence with synopsis
   (`sv-60p.13`), M7 adapters (OpenSubtitles stays parked).

## 9. Waiting on the user

- The extras labeling is **done** (108 rows). The user may still fill in more Supernatural rows (optional).
- Samurai Jack: the user confirmed Scotsman Saves Jack S04E06/E07 and BoE S03E11/E12; Season 2 has 13 episodes
  + 1 near-length extra + 2 short extras; Season 5 `t11` = s05e09.
- Farscape: `Peacekeeper_Wars/` needs `touch /mnt/storage/tv/Farscape/Peacekeeper_Wars/.showvenist-ignore` (the user
  does it); Roughnecks files are being moved to Movies; the empty Farscape `Season05` folder was removed.
- The user pushes branches; ask before assuming the remote state.
