# showvenist: Implementation plan

**Status:** Draft · **Date:** 2026-10-04 · **Builds:** spec v0.2 (`docs/spec.md`)

The spec says **what** to build and when a milestone is done. This plan says **how** the work is run: who does each piece (the main session or a subagent, and at which model tier), in what order, and how the pieces fit together.

M0 is planned in full. M1–M7 are planned as **draft work packages**, because M0 ends in a decision that can change the spec (§17.1, decision 35). After that decision, the packages become Beads tasks and are deleted from §3 of this file, so tasks live in one place only. §1 stays as the working method.

---

## 1. How the work is run

### 1.1 Roles and model tiers

| Role | Runs as | Does |
|---|---|---|
| **Orchestrator** | Main session (Opus 5.5) | Writes the interface contracts; owns shared files; integrates; runs tests; analyzes results; brings every decision to you; commits each wave (never pushes) |
| **Opus agent** | `general-purpose`, `model: opus` | Code that is subtle or can lose data, where a bug is costly *and* tests may not catch it: fusion and confidence, block DP and ordering detection, the conditional-logit fit, rename/move log/undo, multi-source alignment, the PGS/VobSub fixture encoder. Also one spec-conformance review per milestone |
| **Sonnet agent** | `general-purpose`, `model: sonnet` | Feature modules written against a fixed contract with a clear spec: adapters, analysis, most scorers, eval metrics, state, the TUI |
| **Haiku agent** | `general-purpose`, `model: haiku` | Mechanical work: scaffolding, simple adapters, small formula scorers, report formatting, label-sheet and manifest scripts, snapshot tests, docs |
| **Fork** | `subagent_type: fork` (inherits the orchestrator's context) | Opus-level work that needs broad spec context: the fork already has the spec loaded, so it doesn't re-read 88 KB |

**The tier follows risk, not size.** A big but mechanical module goes to Haiku; a 150-line DP goes to Opus.

### 1.2 Efficiency rules

1. **No agent for small things.** Running tests, a one-line fix, or wiring a CLI command takes the orchestrator a couple of tool calls. A cold agent has to re-read context first.
2. **Batch small packages into one agent.** Don't split off a small, easy part of a package just to run it on a cheaper tier: a second cold start costs more than the tier saves.
3. **Follow-ups go to the same agent** (`SendMessage`), which still has its context. Fresh spawns are for new packages.
4. **Run agents in parallel only against a frozen contract and disjoint files.** Everything else runs one after another.
5. **Briefs point to spec sections** (§ numbers). Agents read those sections, not the whole spec.
6. **Short reports:** at most 15 lines (what was done, deviations from the contract, open issues). This keeps the orchestrator's context small.
7. **Decisions never happen inside an agent.** Agents don't see you, and you don't see their output. Anything that needs a choice comes back to the orchestrator and then to you.

### 1.3 Milestone wave pattern

1. **Contract** (orchestrator, no agents). Pydantic models, protocols and function signatures, with docstrings citing spec sections. Stub modules that raise `NotImplementedError`. Acceptance-test skeletons. Every dependency the milestone needs is added to `pyproject.toml` up front. One author writes all of this, because every agent builds on it.
2. **Fan-out** (background agents, in parallel). One agent per package, each owning a disjoint set of files in the same working tree. No worktrees: with disjoint files they aren't needed, and they'd add extra branches to merge.
3. **Integrate** (orchestrator). Run the full test suite and ruff, fix the seams, add the CLI commands, and send larger fixes back to the agent that wrote the code.
4. **Review** (one Opus agent). Checks the milestone's code against its spec sections. Findings go to the orchestrator, and fixes are routed by tier.
5. **Exit check** against the spec's exit criteria, including the steps that need real media or network (§1.5). Each wave is committed once its tests pass (§1.6).

**Shared files belong to the orchestrator:** `pyproject.toml`, `uv.lock`, `showvenist/cli.py`, `showvenist/config.py`, `tests/conftest.py`, and package `__init__.py` files. Agents expose plain functions; the orchestrator adds the typer commands. An agent that needs a new dependency reports it instead of adding it.

### 1.4 Agent brief template

```
Goal: <one sentence>. Implements spec §<n>, §<m> (read those sections in docs/spec.md).
You own: <files>. Don't edit anything else. Read-only: <files>.
Contract: <stub path>. Implement it as written; report any deviation you need.
Acceptance: `uv run pytest <tests>` passes, plus tests you add for <cases>.
Rules: don't touch pyproject.toml or uv.lock (report deps you need); no network
unless stated; no copyrighted media; no commits; no installs.
Report (≤ 15 lines): what you did, deviations, open issues.
```

### 1.5 Where things run

- **Everything runs on eweb.** Development, unit tests and real-media runs all happen there, with Claude Code running on the server. The library is under `/mnt/storage/tv`, and the TMDB token is on the server, so the M0 loop of "run, inspect, adjust" needs no copying.
- **Unit tests use synthetic fixtures only** (§16.4), even though real media is right there. Fixtures are generated with PyAV, which bundles ffmpeg's libraries, so tests don't need an ffmpeg binary.
- **`data/` is gitignored**, so `mkv-titles.tsv` doesn't arrive with a clone. It's copied over, or regenerated on eweb from the library.
- **Installs on eweb (each one asked first):**
  - `tesseract-ocr`, `tesseract-ocr-eng` and `uv`, from M0.
  - `bd`, after M0 (§2.5).
  - Python ≥ 3.11 is managed by uv. A dictionary for OCR quality and for name filtering comes from the `wordfreq` package, not a system word list.

### 1.6 Commits

- The orchestrator commits each integrated wave once tests are green, without asking first. **It never pushes**: you push.
- Agents never commit.
- Each milestone gets its own branch (`m0-spike`, `m1-bundle`, …), so the milestone can be reviewed as one diff before it reaches `main`.

---

## 2. M0: Feasibility spike

Throwaway code in `spike/`, not part of the package (§17.2). Each script is a PEP 723 single-file script (`uv run spike/x.py`) that declares its own dependencies, so agents never share a `pyproject.toml`. Output goes to `spike/out/`, which is gitignored.

### 2.1 Data

Four labeled seasons from four shows, two with PGS subtitles and two with VobSub:

| Season | Files | Track type |
|---|---|---|
| Supernatural S2 | 22 (the 8 `…a.mkv` duplicates are left out) | PGS, known |
| The Incredible Hulk S2 | 22 | VobSub, expected; S3 (23 files) is the fallback |
| The West Wing S1 | 22 | Unknown |
| A fourth, picked after probing to make two of each | about 22 | Candidates: Babylon 5 S1, The Invisible Man S1, Supernatural S1. A show not already in the set is preferred |

- **Step 0: probe the candidates' headers** (seconds per season). This lists their subtitle tracks, including any closed-caption text track MakeMKV extracted from a DVD.
- **Why four seasons.** With two seasons, track type and show would be the same variable: PGS on a show with good TMDB data, VobSub on a 1978 show. A weak result then couldn't be traced to OCR or to data. Four seasons also mean each leave-one-season-out fit uses three seasons.
- **Labels** come from the file names, in aired order for all four seasons.
- **Firefly is left out.** It has only 14 files, labeled in its "Intended Order", which would need the episode-group mapping from M1.
- **There are no extras**, so coverage will look better than it will on raw rips. That's a known limit of M0, not something to fix there.
- **OCR is iterated on about 3 files per season.** The full run happens once OCR is stable, because every OCR fix means extracting again.

### 2.2 Stage contracts (written by the orchestrator first)

| Stage | Output |
|---|---|
| `extract.py` | `out/dialogue/<show>-s<NN>/<basename>.json`: path, size, duration, chosen track (codec, language, events), windows (start, end, lines with time and text), OCR quality, and the lines of a closed-caption text track if there is one. `--probe-only` lists each file's tracks |
| `tmdb.py` | `out/tmdb/<show>-s<NN>.json`: episodes (id, S/E, title, overview, guest stars with character, writers/directors) and the series cast (for removing names) |
| `labels.py` | `out/labels.csv`: path, show, season, label |
| `score.py` | `out/results/<show>-s<NN>.json`, `out/results/pooled.json` and a printed summary |

### 2.3 Work packages

| # | Package | Runs as | Wave | Notes |
|---|---|---|---|---|
| 0.0 | README (how to run on eweb), stage contracts, `.gitignore` for `spike/out/` | Orchestrator | 1 | |
| 0.1 | `extract.py`: `--probe-only`, track selection (§8.2), PGS/VobSub decode via PyAV (`BitmapSubtitle` planes + palette), preprocessing, Tesseract, sampled windows at 20/50/80% (≥ 60 lines or 5 min), closed-caption text tracks, normalization (§8.3), OCR quality | **Opus** | 2 | The riskiest technical piece, and it can only be debugged against real media. The same agent is kept for the fixes the real runs show up |
| 0.2 | `tmdb.py` + `labels.py` for the candidate seasons | Haiku | 2 | Small and mechanical, so batched into one agent |
| 0.3 | `score.py`: synopsis BM25 with all cast/character names removed, robust z, chance-max threshold, 0.8/σ, clamp +4; guest names IDF-weighted, dictionary words filtered, clamp +6; prior with π_x = 0.15; F × (E+F) basic assignment; Δ by re-solve; conditional-logit fit of the two weights, leave-one-season-out and in-sample; captions scored alongside OCR where they exist; metrics pooled and per season | Sonnet | 2 | Tested on a synthetic matrix with a known answer. The orchestrator checks the math against §9–§11 and §16.3 |
| 0.4 | Run on the library: probe the candidates and pick the fourth season; iterate OCR on about 3 files per season; full run of all four; fetch; score | Orchestrator + 0.1's agent | 3 | Needs tesseract and uv installed on eweb |
| 0.5 | `llm.py`, only if outcome 1 (§2.5) isn't reached: the files below high, full-season candidates, structured output, quote check, raw (unclamped) log-odds, cost from usage data | Sonnet (loads the `claude-api` skill) | 4 | Built after wave 3. **Runs only with your OK and a cost estimate; ceiling $1 per season** |
| 0.6 | `spike/RESULTS.md`: tables from the results JSON (Haiku), analysis and recommendation (orchestrator) | Haiku + orchestrator | 5 | |

### 2.4 What gets measured

**The deciding number is coverage at Δ ≥ 4.6 with weights fitted on the other seasons** (leave-one-season-out, §16.3), pooled across the four seasons and per season.

- **Coverage under the default weights** is reported alongside. The defaults are placeholders (§9.1), so it can't decide anything.
- **Coverage with in-sample weights** is reported as an optimistic bound. If it's far above the cross-fitted figure, the weights don't carry over between shows, which is a finding in itself.

The spec's other metrics: top-1 accuracy, the Δ distribution, and errors at high.

**Diagnostics that don't depend on the weights:**

- the true episode's rank under each scorer on its own (rank-1 rate, MRR);
- the gap between the true episode's score and the best other score.

**OCR:**

- OCR quality for PGS vs. VobSub;
- samples of the raw OCR text, to read;
- where a closed-caption text track exists, the captions' scores vs. the OCR'd VobSub's for the same files. This shows directly what OCR costs.

**The LLM (if 0.5 runs):** what it rescues, its cost, and its raw log-odds.

- **With the spec's defaults, LLM evidence can't reach high.** The defaults are a ±3 clamp and weight 1. With E = 22, an episode needs about 1.4 nats to beat "extra" (§11.2), plus 4.6 for high: about 6 nats. The spike has no other scorer in the LLM regime, so it can reach at most 3.
- The spike therefore lifts the clamp and fits the LLM's weight leave-one-season-out on the files below high, like the base weights.

### 2.5 Decision (with you, before any Beads epic exists)

These rules were set on 2026-10-04, before the run. They're also in the spec's M0 row (§17.2).

**Checks first:**

1. **Any error at high**, with default or fitted weights: find the cause before deciding (evidence counted twice, or a clamp that's too loose).
2. **VobSub is clearly worse than PGS**, with garbled samples, or captions scoring far better than OCR: OCR is the bottleneck. Fix it and rerun before reading the VobSub seasons.
3. **A season well below the others** (under about 75% coverage): find its cause (OCR, thin data, or a real weakness) before deciding.

**Outcome, by pooled coverage at high with cross-fitted weights:**

| Outcome | Condition | Change |
|---|---|---|
| 1. Offline is enough | ≥ 90%, no errors at high | None |
| 2. Offline + LLM | Below 90%; the LLM on the files below high brings it to ≥ 90% at ≤ $1 per season | The LLM becomes the recommended setup (README, quality report). Its default mode stays `off`. §1.2 is reworded: offline works, with more review |
| 3. Signal too weak | Below 90% even with the LLM, or only above $1 per season | Move reference subtitles (OpenSubtitles) ahead of the M3 gate; test cues; revisit §1.2 |

**Then:**

1. Spec revision if needed: the orchestrator drafts it and you review it.
2. Beads: install `bd` (asked first); check for `.beads/` with embedded Dolt in git; `bd init`.
3. Create epics M1–M7 and the M3 gate task, with blocks dependencies.
4. Turn §3's packages into tasks and delete §3 from this file.

---

## 3. M1–M7: Draft work packages

*Seeds for Beads after M0. Each milestone starts with a contract wave by the orchestrator (§1.3), and packages in the same wave run in parallel. Exit criteria are the spec's (§17.2) and aren't repeated here.*

### M1: Bundle, TMDB, local SRT

| Package | Tier | Owns | Wave |
|---|---|---|---|
| Scaffold: `pyproject.toml` (uv, Python ≥ 3.11, M1–M2 dependencies), package tree from §15.1, ruff and pytest config, README with the TMDB notice, `--version` | Haiku | the files listed | 0 |
| Contract: `bundle/schema.py` (§6.3), `config.py` (§4.6) | Orchestrator | | 1 |
| Store, `overrides.yaml` merge, validation errors with file and line (§6.2, §6.6) | Sonnet | `bundle/store.py`, `bundle/overrides.py` | 2 |
| TMDB adapter: search, `append_to_response`, seasons, every episode group as an ordering with slug and `type_rank`, `/find`, v3 key or v4 token; recorded fixtures with the key stripped (§6.4) | Sonnet | `adapters/tmdb.py` + fixtures | 2 |
| Local SRT adapter (§6.4) | Haiku | `adapters/localsubs.py` | 2 |
| Quality report + `bundle check` (§6.7) | Sonnet | `bundle/quality.py` | 2 |

- Recording the fixtures needs your TMDB token once.
- The quality report's reference cross-check row uses the M3 scorers, so it's added in M3.
- Proposed shows for the exit criterion: Firefly (episode groups), The West Wing (large, guest-heavy), Samurai Jack (animation, likely sparse data).

### M2: Analysis

| Package | Tier | Owns | Wave |
|---|---|---|---|
| Contract: probe, track, window and dialogue types; cache API with extractor versions (§7.3) | Orchestrator | | 1 |
| Probe, fingerprint, content key (§7.1, §8.4 stage 1) | Sonnet | `analysis/probe.py`, `state/fingerprint.py` | 2 |
| Track selection, text subtitles, normalization (§8.2–§8.3) | Sonnet | `analysis/tracks.py`, `analysis/subtitles.py` | 2 |
| PGS/VobSub OCR + OCR quality, ported from the spike | Sonnet | `analysis/ocr.py` | 2 |
| PGS/VobSub fixture generator: PGS segments and VobSub SPU with RLE, muxed with PyAV (§16.4) | **Opus** | `tests/fixtures/bitmapsubs.py` | 2 |
| Windows, escalation, cache, `inspect` (§8.4) | Sonnet | `analysis/windows.py`, `analysis/cache.py` | 2 |
| Text-subtitle MKV fixtures (PyAV) | Haiku | `tests/fixtures/textsubs.py` | 2 |

- The OCR agent writes its tests against the generator's contract (`make_mkv(path, cues, kind)`) and runs them once the generator lands.
- If the spike's OCR needed heavy rework, the OCR port moves up to Opus.

**Suggestion:** M2 needs only two values from M1, the bundle's language and a vocabulary for OCR quality. With those in the contract, M1 and M2 can run in parallel. The spec chains M1 → M2, so that's your call.

### M3: Scoring, assignment, report, eval

| Package | Tier | Owns | Wave |
|---|---|---|---|
| Contract: `Scorer` protocol (LLR matrix + per-window scores, can-be-negative, clamp), the two regimes (§11.1), weight config, report JSON v1, manifest format (§16.1) | Orchestrator | | 1 |
| Duration + title scorers (§9.2) | Haiku | `scoring/duration.py`, `scoring/title.py` | 2 |
| Synopsis, guest-name and cues scorers, sharing tokenization and name removal; ported from the spike | Sonnet | `scoring/synopsis.py`, `scoring/characters.py`, `scoring/cues.py` | 2 |
| Reference-subtitle scorer + the quality report's cross-check row (§9.2, §6.7) | Sonnet | `scoring/refsubs.py`, `bundle/quality.py` | 2 |
| LLM scorer + cue generation via the Batches API (§10). Loads the `claude-api` skill | Sonnet | `scoring/llm.py`, `adapters/llm_cues.py` | 2 |
| Fusion, basic assignment with unmatched columns, Δ, runner-up (§11.1–§11.2, §11.5, §11.7) | **Opus** | `fusion.py`, `assign/basic.py`, `assign/confidence.py` | 2 |
| Report tables + JSON (§12.3) | Haiku | `report.py` | 2 |
| Eval harness: manifest, label resolution through orderings, duplicate sets, every §16.1 metric including Clopper-Pearson | Sonnet | `eval/harness.py`, `eval/manifest.py` | 2 |
| `eval --fit`: conditional logit, regularized toward defaults, extras weighted to their real share, two regimes, leave-one-season-out (§16.3) | **Opus** | `eval/fit.py` | 3 |
| `identify --report/--json` on flat lists, exit codes (§4.3, §4.5) | Orchestrator | `cli.py` | 3 |

- The LLM's disc-window scope (§10.2) needs disc folders, so it's built in M4. In M3, LLM candidates are the full season.

**Labels: a human task to start during M1, not at the gate.**

| Package | Tier | When |
|---|---|---|
| Manifest script, run on eweb: the 422 named files → `path,show,label,order,disc`. Order is per show (Firefly → `intended-order`, since you named files chronologically) | Haiku | After M1 |
| Extras labeling sheet for the 1,314 unnamed rips: duration, size, title tag, disc | Haiku | During M1 |

- The sheet is sorted with **files close to episode length first**, because those matter most (§16.2). Files aren't pre-labeled by duration: the fit and eval would then be grading the duration scorer against its own guesses.
- You label the extras; π_x is measured from them (§11.2).
- **Open item for the M3 contract:** a raw rip labeled "episode, number unknown" can count toward unmatched precision and recall, but not toward top-1.

**M3 gate** (orchestrator + you): eval on eweb in four setups, {base, LLM} × {cues off, cues on}. The orchestrator writes the analysis; you decide go or adjust.

### M4: Hierarchy and block model

| Package | Tier | Owns | Wave |
|---|---|---|---|
| Contract: tree and disc types, block solver interface | Orchestrator | | 1 |
| Folder, volume-label and title-tag parsing; creation-time disc order suggestion; tests from `data/mkv-titles.tsv` (§5.1) | Sonnet | `hierarchy/parse.py` | 2 |
| Files already named (§5.5), marker file, show disambiguation, Jellyfin ID hint (§5.4) | Haiku | `hierarchy/marker.py`, `hierarchy/named.py` | 2 |
| Readiness: size/mtime stability + Matroska duration and cues (§5.2) | Sonnet | `hierarchy/readiness.py` | 2 |
| Concurrent read/OCR pipeline (§8.5) | Sonnet | `analysis/pipeline.py` | 2 |
| Block DP with block-score caching, ordering detection, season hard filter + out-of-scope warning, `--episodes`, LLM disc window (§10.2, §11.3, §11.5–§11.6) | **Opus** | `assign/blocks.py`, `assign/ordering.py` | 2 |

- `data/mkv-titles.tsv` is gitignored. Its test cases are copied into a test table, as basenames and tags only.

### M5: State and renames

| Package | Tier | Owns | Wave |
|---|---|---|---|
| Contract: `state.db` DDL (§7.2), migrator, plan types | Orchestrator | | 1 |
| History, history prior + duplicate warnings, confirmed extras, `assign`, `history`, `forget`, `eval --from-history` | Sonnet | `state/db.py`, `state/migrations.py`, `state/history.py` | 2 |
| Move plan, sidecars, extras moves, collision/permission/cross-filesystem checks, move log + crash recovery, undo (§13); crash test kills a subprocess between `planned` and `done` | **Opus** | `rename/*` | 2 |

### M6: TUI

| Package | Tier | Owns | Wave |
|---|---|---|---|
| Contract: `Session` API with an event stream for live updates (§12.1) | Orchestrator | `session.py` (signatures) | 1 |
| `Session` implementation over the engine | Sonnet | `session.py` | 2 |
| Textual app: tree pane, evidence pane, keys, pin → re-solve → next least-confident file, plan screen, startup dialogs, Pilot flow tests (§12.2). One agent, because the panes share state | Sonnet | `tui/*` | 3 |
| Snapshot tests | Haiku | `tests/tui/test_snapshots.py` | 4 |

- Exit includes you testing over SSH from Windows Terminal.

### M7: More adapters

**M7 depends only on the gate, so it can run alongside M4–M6.**

| Package | Tier | Owns | Wave |
|---|---|---|---|
| TVmaze adapter | Haiku | `adapters/tvmaze.py` | 1 |
| Wikipedia adapter: MediaWiki API, `{{Episode list}}`, transclusion | Sonnet | `adapters/wikipedia.py` | 1 |
| OpenSubtitles adapter, resumable within the quota | Sonnet | `adapters/opensubtitles.py` | 1 |
| Multi-source merge with content alignment and recorded conflicts (§6.5) | **Opus** | `bundle/merge.py` | 1 |
| `bundle edit` + `--refresh` diff (§6.6, §6.8) | Sonnet | `bundle/edit.py`, `bundle/diff.py` | 1 |
| Eval with reference subtitles, n-gram survival, OCR-quality thresholds | Orchestrator + Sonnet | `eval/*` | 2 |

- The OpenSubtitles exit criterion needs your account.

### Rough size

About 50 agent runs across M0–M7: roughly 12 Haiku, 23 Sonnet and 14 Opus (7 of the Opus runs are milestone reviews).
