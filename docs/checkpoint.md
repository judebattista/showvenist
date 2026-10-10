# showvenist: session checkpoint (2026-10-09)

Written so a fresh context can resume. The spec (`docs/spec.md`, decision log §19, decisions 38–55) stays the
source of truth; Beads (`bd ready`, `bd list`) holds the tasks; this file is the narrative state and the
operating notes. Read `CLAUDE.md`, then this, then `bd ready`.

## 1. Where the code is

- Branch **`m4-hierarchy`** (head `f44d7a6`). Earlier branches are linear ancestors: `m0-spike` → `m1-bundle` →
  `m3-scoring` → `m4-hierarchy`. **Nothing has been pushed; the user pushes.** Subagents never commit.
- 808 tests pass; `ruff check .` clean. Python is **`.venv/bin/python`**, `.venv/bin/showvenist`, `.venv/bin/ruff`
  (`uv` is no longer on the PATH; do not install anything).
- Milestones: **M0–M4 done** and reviewed (one Opus conformance review each, findings fixed). Since the last
  checkpoint, on the same branch: the **title-card scorer** (decisions 54–55, epic `sv-mxo`, closed), an **eval
  fix for file-order leakage** (`sv-cn7`), and report flags (`sv-60p.14`). **M5 (state and renames) is next and
  was ON HOLD** by the user; the user has now asked to start it with a design talk (§8). M6 (TUI) and M7 (more
  adapters) follow. Epics: M5 `sv-o4f`, M6 `sv-zrm`, M7 `sv-opu`.
- Beads data is local (Dolt DB is gitignored); backup lives in `data/beads-backup/` (`bd backup sync`).

## 2. What exists (package `showvenist/`)

bundle (schema, store, overrides incl. `new_episodes`, inherited orderings and `title_card`, quality, ingest) ·
adapters (tmdb, localsubs, opensubtitles [mocked only, parked], llm_cues) · analysis (probe, tracks, subtitles,
ocr, windows, pipeline, duplicates, inspect, cache, **cardframes**) · scoring (duration, title, **titlecard**,
refsubs, synopsis, characters, cues, llm, candidates) · fusion · assign (basic, confidence, blocks, ordering) ·
hierarchy (parse, marker, readiness) · session · report (+ **edgeflags**: advisory `play_all`,
`possible_two_parter`) · eval (harness, manifest, runner, fit, render, labelsheet, mkmanifest) · cli (`ingest`,
`bundle check`, `inspect [--card]`, `identify`, `eval`). Not built: state.db/history, rename plan/move log/undo
(M5), TUI (M6), TVmaze/Wikipedia/merge (M7).

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

- **LLM off, 538 files (409 hand-named + labeled raw rips), shuffled:** top-1 **50.2%**, coverage at high 8.0%,
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

## 4. Decisions in this stretch (spec §19, 38–55; read them there)

38 M0 outcome 3 · 39 `wordfreq` OCR word check · 40 LLM evidence = `logit(p)+log(n−1)` · 41 `episode?` labels ·
42 duration scorer · 43 OpenSubtitles live use parked · 44 M3 gate passed with a documented deviation · 45 gate
wording; `llm.mode` stays `off` · 46 ordering detection solves every ordering · 47 ordering threshold 15 · 48 block
params (slack 1, u=2, margin 4, threshold 15) · 49 `extra-multi` label · 50 two-part label `S01E01+S01E02` · 51
`.showvenist-ignore` folders · 52 near-duplicate detection by dialogue overlap · 53 overrides `new_episodes` and
inherited orderings · **54 title-card scorer (opt-in per show via `overrides.yaml`, closed-set match, for-only)** ·
**55 a run of exactly four I in a Roman reading is read as three**. Also in the spec now: §16.1 file-order rule
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

`bundles/<slug>/show.json` for 13 shows (babylon-5-1994, burn-notice-2007, cowboy-bebop-1998, farscape-1999,
firefly-2002, samurai-jack-2001, supernatural-2005, the-expanse-2015, the-incredible-hulk-1977,
the-invisible-man-2000, the-west-wing-1999, the-white-shadow-1978, the-wire-2002); Samurai Jack has an
`overrides.yaml` (custom ordering `disk`; Birth of Evil = S03E11/E12, Scotsman Saves Jack = S04E06/E07; and the
`title_card` block; backup of the earlier file: `overrides.yaml.bak-before-card`). Manifests: `manifest-named.csv`
(410 hand-named files), `manifest-all.csv` (538 rows incl. the labeled sample, regenerated from the corrected
labels). Sheets: `extras-sheet.csv` (1,314 unnamed rips), `extras-sheet-sample.csv` (175 stratified),
`extras-sheet-sample-labelled.csv` (the user's labels, 108; Babylon 5 S05 and two Samurai Jack rows corrected),
`extras-sample-priority.csv`. Eval outputs: `eval-*.txt` (latest: `eval-rebase-off.txt`, `eval-sj-card2-*.txt`,
`eval-weak-llm2.txt`), older `thr*`, `final-*`, `cmp-*`. Real-library facts for parser tests: `title-tags.tsv`,
`library-tree.txt`. Beads backup: `beads-backup/`.

## 7. Commands that reproduce the current state

```
# manifest from the user's labels (sets order=disk for Samurai Jack, blu-ray-order for West Wing S03,
# intended-order for Firefly automatically)
.venv/bin/python -m showvenist.eval.mkmanifest /mnt/storage/tv --sheet data/extras-sheet.csv \
  --sample data/extras-sheet-sample-labelled.csv --bundles data/bundles -o data/manifest-all.csv
# free eval (LLM off), blocks on, files shuffled; add --compare-blocks, --show-errors, --fit, --ablate
.venv/bin/showvenist eval data/manifest-all.csv -o data/bundles --library /mnt/storage/tv --show-errors
# cached-only LLM (free)
SHOWVENIST_LLM_SCOPE=season .venv/bin/showvenist eval data/manifest-named.csv -o data/bundles \
  --library /mnt/storage/tv --llm auto
```

## 8. Next steps (in this order; the user decides)

1. **M5 design pass with the user** (state.db, history prior, `assign/history/forget/undo`, move plan, move log,
   crash recovery). Open design questions to bring: what the plan does with duplicates (exact and near; keep the
   best copy, move the rest to `extras/`, never delete?), `extra-multi` and `.showvenist-ignore` handling in the
   plan, safety rules for the first command that touches the user's files. Start with the contract (`sv-o4f.1`).
2. Open small items: `sv-60p.13` (cues gate: unanswerable without paid cue generation; leave experimental),
   `sv-60p.15` (near-duplicate top-1 rule), `sv-31q` (duration σ with no season).
3. Deferred: sparse-subtitle read cost (`sv-o2p.7`), loose files + disc blocks (`sv-o2p.9`), M6 TUI (`sv-zrm`,
   after M5), M7 adapters (`sv-opu`; OpenSubtitles stays parked).
4. Optional: a paid eval with the LLM over the labeled raw rips (estimate first, ask for the budget); fill the 35
   skipped Supernatural near-length rows with `episode?`; generalize title cards to other shows (config only);
   make production tie-breaking independent of file order (e.g. canonicalize order by fingerprint) if the user
   wants it strictly.

## 9. Waiting on the user

- Farscape: `touch /mnt/storage/tv/Farscape/Peacekeeper_Wars/.showvenist-ignore` (the user does it); Roughnecks
  files are being moved to Movies.
- The Wire S03E05/E06 swap inside one disc is left as a manual fix (the user's decision; revisit if the error
  turns out to be general).
- The user pushes branches; ask before assuming the remote state.
