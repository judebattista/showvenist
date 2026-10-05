# showvenist

Identifies the season and episode of TV disc rips against a reference bundle for the show.

- `docs/spec.md` is the source of truth for what to build and why, and it owns the milestones and their exit criteria. Its decision log (§19) records settled questions: don't re-argue them without new evidence.
- `docs/plan.md` covers how the work is run: the orchestrator, subagent tiers, waves and agent briefs (§1). Follow it when building.
- Tasks live in Beads once it's set up (after M0). Never put task lists in the spec.

## Rules

- **Commit without asking:** each integrated wave once its tests pass, on the milestone's branch (`m0-spike`, `m1-bundle`, …). **Never push**; the user pushes. Subagents never commit.
- **No spending money without permission.** Ask before any paid API call, such as the LLM scorer, cue generation or eval with `--llm`, and give the estimated cost. The LLM scorer's default mode stays `off`.
- **Ask before installing anything:** apt packages, uv tools, `bd`, npm globals. Also ask before `bd setup claude`, which installs hooks.
- **The library under `/mnt/storage/tv` is read-only during development.** Read it for spikes, eval and inspection. Never point code that renames, moves or deletes files at it; use synthetic fixtures or copies.
- **No copyrighted media in tests or in git** (spec §16.4).
- `data/` is gitignored. `data/mkv-titles.tsv` is a dump of the library's MKV title tags (basenames only).

## Environment

- Development and real-media runs happen on eweb, where the library lives. The TMDB token is on the server (config file or `SHOWVENIST_TMDB_API_KEY`).
- Python ≥ 3.11, managed with uv.

## Working with the user

- The user wants a partner, not affirmation. Question decisions you disagree with, and give a recommendation with its reasoning rather than a survey of options.
- Design happens topic by topic. A rejected multiple-choice or plan-approval prompt means "let me talk": ask in plain text what they want to clarify.
