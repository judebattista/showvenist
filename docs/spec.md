# showvenist: Specification

**Status:** Draft v0.2 (reviewed) · **Date:** 2026-10-04 · **Scope:** v1 in detail, v2 and v3 outlined

showvenist identifies which season and episode of a known TV show a video file contains. It's built for rips of retail DVDs and Blu-rays, whose files have meaningless names (`title_t03.mkv`). It compares signals extracted from the media (subtitle dialogue, duration, container metadata) against a reference data bundle for the show. It solves a whole season at once, explains every match, and renames files only after you confirm.

This document is the source of truth for **what** to build and **why**. Task breakdown lives in Beads (see §17). Milestones and their exit criteria live here.

---

## Contents

1. [Overview](#1-overview)
2. [Prior art and differentiation](#2-prior-art-and-differentiation)
3. [Environment and workflow](#3-environment-and-workflow)
4. [Command-line interface](#4-command-line-interface)
5. [Directory hierarchy input](#5-directory-hierarchy-input)
6. [Reference bundle](#6-reference-bundle)
7. [Local state](#7-local-state)
8. [Media analysis](#8-media-analysis)
9. [Scoring](#9-scoring)
10. [LLM usage](#10-llm-usage)
11. [Fusion and assignment](#11-fusion-and-assignment)
12. [Review TUI and reports](#12-review-tui-and-reports)
13. [Renaming](#13-renaming)
14. [Edge cases](#14-edge-cases)
15. [Architecture, dependencies, packaging](#15-architecture-dependencies-packaging)
16. [Evaluation and testing](#16-evaluation-and-testing)
17. [Roadmap and milestones](#17-roadmap-and-milestones)
18. [Open questions and risks](#18-open-questions-and-risks)
19. [Decision log](#19-decision-log)

---

## 1. Overview

### 1.1 Problem

Retail disc rips (MakeMKV output) come with meaningless names, and the titles on a disc are often not in episode order. Identifying each file by hand means watching part of it. Existing tools depend on reference subtitles downloaded from OpenSubtitles. That's the strongest signal when available, but it needs an account, has a daily download quota, and sometimes no subtitles exist for a show.

### 1.2 Goals

- Correctly identify season and episode for disc rips and badly named files of a **known** show.
- Work **offline** once the show's reference bundle exists. Network-dependent features (LLM scorer, OpenSubtitles) are optional extras.
- **Explain** every match: which signals contributed, how much, and what the runner-up was.
- **Never change files destructively.** Nothing moves without explicit confirmation; every move is logged and can be undone.
- Be **measurably** accurate. An evaluation harness exists from the first scoring milestone, and confidence levels are calibrated against labeled data.

### 1.3 Non-goals (v1–v3)

- Identifying an unknown show (the user always names the show).
- Identifying movies.
- Ripping, transcoding or downloading media.
- Integration with the Sonarr or Jellyfin APIs.
- One episode split across multiple files.
- Unattended or daemon operation (cron jobs, a `watch` mode). The tool is run by hand.
- Platforms other than Linux (WSL counts as Linux).

### 1.4 Glossary

| Term | Meaning |
|---|---|
| **Bundle** | The reference data for one show: episodes, synopses, crew, orderings, optional reference subtitles. One directory per show. |
| **Candidate** | An episode that a file might be. The candidate set is the episodes in scope (a season, or a range). |
| **Episode ID** | A stable identifier for an episode that doesn't depend on any ordering, e.g. `tmdb:62091`. |
| **Ordering** | A numbering of episodes: TMDB's aired order, any TMDB episode group (DVD, absolute, production, …), or a custom ordering from `overrides.yaml`. |
| **Release order** | The ordering a particular disc release follows. Detected automatically (§11.6). |
| **Naming order** | The ordering used in filenames, set per show in the marker file (§13.1). |
| **Evidence (LLR)** | A log-likelihood ratio in nats: positive means evidence that a file *is* an episode, negative means evidence that it isn't. |
| **Block** | The consecutive range of episodes on one disc, in release order. |
| **Fingerprint** | A file's identity: size + hash of its first and last 64 KB. Never the path. |
| **Pin** | A match the user has fixed by hand during review. |
| **Extra** | A file that isn't an episode: featurette, trailer, deleted scenes. |
| **Play-all title** | A file containing several episodes concatenated. |
| **Δ (margin)** | How much worse, in nats, the best alternative assignment is if this file isn't given its chosen episode. Odds ≈ e^Δ. |

---

## 2. Prior art and differentiation

| Tool | Approach | Limitation addressed here |
|---|---|---|
| [mkv-episode-matcher](https://github.com/Jsakkos/mkv-episode-matcher) | TMDb + OpenSubtitles reference subtitles + Whisper speech-to-text | Depends on reference subtitles; identifies files independently, not jointly |
| [tv-rip-identifier](https://github.com/timcrob/tv-rip-identifier) | 6-word phrase matching of embedded/OCR'd subtitles against reference subtitles | Depends on reference subtitles |
| [RipWeaver](https://github.com/fajis1/ripweaver) | Web app wrapping ripping, identification and organization | Rip workflow focus; also subtitle-based matching |

What showvenist does differently:

1. **Combines several signals** and still works without reference subtitles: synopsis matching, guest character names, duration, container title tags, an optional LLM scorer, and (v2) on-screen credits. Reference subtitles are used as the strongest signal when available.
2. **Solves each season jointly.** Each episode can be used at most once, files that match nothing are flagged as extras, and **disc folders narrow the search** because each disc holds a consecutive block of episodes (§11.5).
3. **Calibrated confidence and explanations.** Evidence from each signal is additive, confidence is a margin against the best alternative, and the report shows how much each signal contributed.
4. **Remembers your library.** Confirmed matches persist and act as a soft prior for later discs, and they also serve as labeled test data.

---

## 3. Environment and workflow

### 3.1 Target environment

- **Media server:** Ubuntu server on the LAN. The rip tree lives on its local disks and is exported over SMB.
- **Ripping:** MakeMKV runs on a Windows machine on the LAN with the optical drive. It writes **straight onto the server's SMB share**.
- **Where showvenist runs:** on the Ubuntu server, over SSH. Files are local there (disk-speed reads, local atomic renames), and state and bundles live next to the media they describe. Running from another Linux/WSL machine over the network is supported but limited by network throughput. If you do that, mount the share inside Linux (NFS or CIFS), not through a Windows mapped drive under `/mnt/<letter>`.
- **Media server software:** Jellyfin, with no plugins. Series are linked by hand to their TMDB or IMDb IDs (§13.1 covers what that means for naming).
- **Hardware:** no GPU. Everything compute-heavy is planned for CPU.
- **Permissions:** files created through Samba belong to the Samba user. showvenist must run as a user with write access to the rip tree, and checks this before planning any move (§14).

### 3.2 Typical session

```
# On the Windows box: MakeMKV output → \\server\rips\Firefly\Season 1\
# (MakeMKV creates one folder per disc, named after the volume label)

$ ssh server
$ showvenist identify ~/rips/Firefly/
```

1. First run for this show: the TUI asks which show is meant (if the name is ambiguous) and confirms writing the marker file.
2. The bundle is ingested if it doesn't exist yet (TMDB by default), and a data-quality report is shown.
3. The parsed tree is shown: seasons, discs, file counts, files still being written.
4. Analysis runs (sampled subtitle windows, OCR), and results update live.
5. You review the files that aren't high confidence. Each choice is pinned and the season re-solved.
6. The plan screen shows every move. You confirm, and files are flattened into the season folder with Jellyfin-friendly names.

---

## 4. Command-line interface

The executable is `showvenist`. All commands that change files or the environment ask for confirmation.

### 4.1 Commands

| Command | Purpose |
|---|---|
| `ingest <show>` | Build or refresh a show's reference bundle |
| `bundle check <show>` | Print the bundle's data-quality report |
| `bundle edit <show>` | Open `overrides.yaml` in `$EDITOR`, validate on save |
| `identify PATHS...` | Identify files: review TUI on a TTY, report otherwise |
| `assign FILE SxxEyy` | Set a match by hand |
| `history` | Show what you have and what's missing for a show |
| `forget FILE\|FINGERPRINT` | Remove a file's records from history |
| `undo` | Reverse an applied rename run |
| `inspect FILE` | Dump extracted signals, for debugging |
| `eval` | Accuracy harness and weight fitting |

### 4.2 `ingest`

```
showvenist ingest <show> [--ref URL|ID ...] [--source tmdb,tvmaze,wikipedia]
                         [--subs opensubtitles|DIR] [--season N]
                         [--year YYYY] [--tmdb-id N] [--refresh] [-o DIR]
```

- Each `--ref` is routed to the adapter that handles it:
  - `themoviedb.org/tv/…` → TMDB
  - `tvmaze.com/shows/…` → TVmaze
  - `en.wikipedia.org/wiki/List_of_…` → Wikipedia
  - `imdb.com/title/tt…` → **looked up through TMDB's `/find` endpoint only**. IMDb is never scraped: that's against its terms of use and fragile.
- Ambiguous names ("The Office"): interactive pick, or `--year` / `--tmdb-id`. Non-interactive runs fail and list the candidates.
- `--refresh` re-fetches an existing bundle and prints a diff (§6.8).
- Every ingest ends with the data-quality report (§6.7).

### 4.3 `identify`

```
showvenist identify PATHS... [--show NAME | --ref DIR] [--season N]
                    [--episodes RANGE|LIST] [--order NAME]
                    [--no-blocks] [--include-specials] [--allow-duplicates]
                    [--signals LIST] [--llm off|auto|all] [--reanalyze]
                    [--report | --json]
```

- `PATHS` can be files, or directories walked as a Show/Season/Disc hierarchy (§5). Explicit flags override anything inferred from folders or the marker file.
- `--show NAME` resolves to the bundle at `$XDG_DATA_HOME/showvenist/shows/<slug>/`. If it doesn't exist, it's ingested with default sources. `--ref DIR` points to a bundle directory explicitly.
- `--episodes 5-8` or `--episodes S02E05,S02E09`: a manual scope override, e.g. typed from a case insert, or for a single disc ripped on its own without history.
- `--order`: the starting matching order: any ordering in the bundle. Defaults to the show's `naming_order`, because you name season folders by your own convention. Ordering detection (§11.6) may still switch to a better-fitting order and report it.
- `--no-blocks`: turn off the disc block model and solve each season without block constraints.
- `--include-specials`: make S00 episodes candidates. This is automatic for a `Specials` / `Season 0` folder.
- `--allow-duplicates`: relax the at-most-once constraint (for duplicate rips of the same episode).
- `--signals`: enable or disable individual scorers (for diagnosis and eval).
- `--llm`: overrides the configured LLM mode. `off`; `auto` sends files below high confidence after the first solve; `all` sends every file (used by eval and fitting).
- `--reanalyze`: ignore history and the analysis cache for these files.
- On a TTY, `identify` opens the **review TUI** (§12). With `--report` or `--json`, or when stdout isn't a TTY, it prints results and **never touches media files**. It still records proposed matches in history, updates the analysis cache, and with `--show` may ingest a missing bundle.
- Files already confirmed in history are reported straight from it, without re-analysis.

### 4.4 Other commands

- `assign FILE S02E05 [--show NAME] [--rename]`: records a confirmed match with source `manual`. `--rename` shows the planned move and asks y/N.
- `assign FILE --extra`: records the file as a confirmed extra.
- `history [--show NAME] [--season N]`: per season, which episodes are confirmed (with file paths), which are missing, and which files are confirmed extras.
- `forget FILE|FINGERPRINT`: removes all records for that file (asks for confirmation).
- `undo [--last | RUN_ID]`: reverses a run's moves, after showing them and asking for confirmation (§13.5).
- `inspect FILE`: probe data, readiness, chosen subtitle track, OCR'd dialogue per window, cache status.
- `eval (MANIFEST.csv | --from-history) [--show NAME] [--fit] [--llm …]`: see §16. `--show` limits a multi-show manifest to one show.

### 4.5 Exit codes

Report mode (`--report`, `--json`, non-TTY):

| Code | Meaning |
|---|---|
| 0 | Every file is high confidence or already confirmed |
| 2 | At least one file needs review (medium, low or not ready, whether its best option is an episode or "extra") |
| 1 | Error |

### 4.6 Configuration

`$XDG_CONFIG_HOME/showvenist/config.toml`. Environment variables (`SHOWVENIST_*`) override it, and command-line flags override both.

```toml
[tmdb]
api_key = "…"            # v3 API key or v4 read access token; or env SHOWVENIST_TMDB_API_KEY

[opensubtitles]
api_key = "…"
username = "…"
password_env = "SHOWVENIST_OPENSUBTITLES_PASSWORD"

[naming]
template = "{show_snake}_s{s:02}e{e:02}"   # → the_west_wing_s01e02.mkv
move_extras = true       # move identified and confirmed extras to <Season>/extras/

[analysis]
io_concurrency = 1       # concurrent reads per storage root
ocr_workers = 0          # 0 = number of CPU cores
window_min_lines = 60
window_max_minutes = 5
readiness_stable_seconds = 60

[llm]
mode = "off"             # off | auto | all
model = "claude-sonnet-5-5"
effort = "low"
send_both_orders = true  # correct for list-position bias (§10.2)
scope = "disc"           # disc | season: candidate list where disc folders exist (§10.2)

[weights]                # written by `eval --fit`; defaults built in
# duration = 1.0 …

[weights.llm]            # files the LLM has scored (§11.1)
# duration = 1.0 …
```

Anthropic credentials are resolved by the Anthropic SDK (`ANTHROPIC_API_KEY`, or a profile from `ant auth login`). showvenist never stores them.

---

## 5. Directory hierarchy input

### 5.1 Layout

```
<Show>/
  .showvenist.toml          ← marker file (§5.4)
  Season 1/
    FIREFLY_D1/             ← disc folder (MakeMKV volume label, or "Disc 1")
      title_t00.mkv
      title_t01.mkv
    FIREFLY_D2/
      …
  Specials/
    …
```

- Show, season and disc are parsed from folder names using **configurable patterns**. The defaults cover:
  - seasons: `Season 1`, `Season 01`, `S01`, `Specials`, `Season 0`;
  - discs: `Disc 1`, `D1`, `DISC_1`, and MakeMKV volume labels such as `FIREFLY_D1` or `FRIENDS_S2_D1`.
- **The MKV title tag is a second source of disc identity.** MakeMKV sets the segment title to the disc name (`THE WEST WING COMPLETE SERIES S3D2`, `The Expanse: Season One (Disc 2)`), and the tag survives flattening, so it identifies the disc of files already moved out of disc folders. The same patterns parse it. When both the folder and the tag parse, the folder wins and any disagreement is reported.
- The parsed tree (show / season / disc / number of files / files not ready) is **always shown before analysis**.
- If a level can't be parsed, the tool asks in interactive mode and fails in non-interactive mode. **It never guesses.**
- **Discs without numbers.** If volume labels carry no disc number, or are identical, the tool **suggests** an order based on folder creation time (discs are ripped in order) and asks you to confirm. Non-interactive runs fail.

### 5.2 Readiness

MakeMKV writes titles one after another into their final filenames over SMB, so a running rip leaves partially written files. A file is analyzed only when:

1. its size and mtime have been stable for `readiness_stable_seconds` (default 60), **and**
2. it's structurally complete: the Matroska segment has a duration and cues.

Files that aren't ready are skipped and reported ("Disc 3: 2 files still being written"). Their disc's block is marked **incomplete** (§11.5).

### 5.3 What the hierarchy contributes

- **The season folder is a hard filter.** Only that season's episodes can be assigned, with "season" read in the ordering being solved (§11.6). An episode can be a special in one ordering and a regular episode in another: TMDB's aired order puts Firefly's three unaired episodes in Specials, while its "Intended Order" group has them in Season 1. Out-of-scope episodes are still scored, only so the tool can warn when a file's evidence strongly points to another season. **A file is never silently moved into a different season.**
- **Disc folders narrow the candidates** through the block model (§11.5). Each disc is a consecutive block of episodes in release order, and blocks follow disc numbers. Block boundaries are **found from the content, not from file counts**. Disc folders contain extras, play-all titles and duplicate angles, so a window based on counts would be wrong, and the error would carry into every later disc.
- **The season is the unit of the solve**, so discs constrain each other: one confident match pins the neighbouring boundaries.

Premise (confirmed for retail media): the order of titles *within* a disc is random, but discs themselves are chronological. Nobody has to swap discs out of order to watch the show in sequence.

### 5.4 Marker file

`<Show>/.showvenist.toml`, created on the first run **after you confirm** which show was resolved. Later runs read it and skip the lookup and disambiguation. Command-line flags override its values.

```toml
show = "Firefly"
tmdb_id = 1437
bundle = "firefly-2002"
language = "en"
naming_order = "intended-order"   # any ordering in the bundle (§13.1)
# [defaults]
# include_specials = false
```

If the show folder's name doesn't contain a Jellyfin ID hint, the report **suggests** (never applies) renaming it to `Firefly (2002) [tmdbid-1437]`. That pins Jellyfin to the same database showvenist used.

### 5.5 Files already named

A file whose name already parses as an episode is **ground truth**. The patterns are configurable; the defaults cover `S01E02`, `show_s01e02` and `s01_e02`, read in the show's `naming_order`.

- It's recorded in history as a confirmed match with source `name`, and `identify` doesn't analyze it.
- It acts as a pin: it holds its episode in the season's solve, and if its disc is known (folder or title tag, §5.1), that disc is a fixed block (§11.5).
- It's **never renamed or moved automatically** (§13.2).
- A name that doesn't resolve to an episode in `naming_order` is reported: for example S01E14 when `naming_order` is `aired` and that season has 11 episodes.
- Names that differ only by a suffix (`s02e09` and `s02e09a`) form a duplicate set (§14).

Checking names against the media is eval's job (§16), not `identify`'s.

---

## 6. Reference bundle

### 6.1 Format decision

**One JSON file per show, covering all seasons.** Rejected alternatives:

- **CSV per season.** The data is nested (several synopses per episode, crew lists, actor/character pairs), and orderings cross season boundaries (DVD order can move episodes between seasons, and anime uses absolute numbering).
- **A database.** Matching loads the whole bundle into memory (a few hundred episodes at most), so a database adds nothing. JSON is diffable, inspectable, shareable and easy to use as a test fixture.

### 6.2 Layout

```
$XDG_DATA_HOME/showvenist/shows/<slug>/
  show.json          ← generated by ingest; never hand-edited
  overrides.yaml     ← optional; hand-written; merged over show.json on load
  subs/
    tmdb-62091.en.srt
```

Bundles live in the data directory, not the cache: they hold hand edits and are referenced by history.

### 6.3 Schema

Defined as pydantic models with a `schema_version` field. Illustrative `show.json`:

```json
{
  "schema_version": 1,
  "bundle_revision": "b3:9f2c…",
  "show": {
    "title": "Firefly",
    "year": 2002,
    "language": "en",
    "in_production": false,
    "ids": { "tmdb": 1437, "tvmaze": null, "imdb": "tt0303461", "wikipedia": "List_of_Firefly_episodes" }
  },
  "episodes": [
    {
      "id": "tmdb:62091",
      "title": "Serenity",
      "air_date": "2002-12-20",
      "runtime_min": 86,
      "production_code": "1AGE79",
      "synopses": [
        { "source": "tmdb", "text": "…" },
        { "source": "wikipedia", "text": "…" }
      ],
      "directors": ["…"],
      "writers": ["…"],
      "guest_cast": [ { "actor": "…", "character": "…" } ],
      "subtitles": [ { "lang": "en", "path": "subs/tmdb-62091.en.srt", "source": "localsubs" } ],
      "alt_titles": [],
      "excluded": false,
      "llm_cues": {
        "names": ["…"], "terms": ["…"], "lines": ["…"],
        "model": "claude-sonnet-5-5", "generated_at": "2026-10-02T12:00:00Z"
      }
    }
  ],
  "orderings": {
    "aired": {
      "source": "tmdb", "type": "aired",
      "entries": [ { "season": 1, "episode": 11, "episode_id": "tmdb:62091" } ]
    },
    "intended-order": {
      "source": "tmdb", "tmdb_group": "…", "name": "Intended Order", "type": "absolute", "type_rank": 1,
      "entries": [ { "season": 1, "episode": 1, "episode_id": "tmdb:62091" } ]
    },
    "dvd-order": {
      "source": "tmdb", "tmdb_group": "…", "name": "DVD Order", "type": "dvd", "type_rank": 1,
      "entries": [ … ]
    }
  },
  "provenance": [
    { "adapter": "tmdb", "url": "https://api.themoviedb.org/3/tv/1437", "fetched_at": "2026-10-02T12:00:00Z" },
    { "conflict": { "adapter": "wikipedia", "kind": "unaligned_episode", "detail": "…" } }
  ]
}
```

- `id` doesn't depend on any ordering. TMDB episode IDs are used when available. Bundles without TMDB use a fallback scheme (`<source>:<source id>`).
- **Every TMDB episode group is kept as its own ordering**, keyed by a slug of its name (`intended-order`, with `-2` etc. on collisions). TMDB groups are user-contributed, and a show can have several of one type. `type_rank` is the group's position among groups of the same type in TMDB's list. Jellyfin uses rank 1 (§13.1).
- `bundle_revision` is a content hash, recorded on every match in history.
- `llm_cues` is optional and experimental (§10.3).

### 6.4 Adapters

All adapters implement `fetch(show_ref) -> PartialBundle`.

| Adapter | Milestone | Data | Notes |
|---|---|---|---|
| **TMDB** (primary) | M1 | Episodes, overview, runtime, crew (writer/director), guest stars with character, episode groups (DVD/absolute orders), production status | `/search/tv`, `/tv/{id}` with `append_to_response` for seasons, `/tv/{id}/season/{n}`, `/tv/{id}/episode_groups`, `/find` for IMDb IDs. Free API key. |
| **Local SRT folder** | M1 | Reference subtitles | Maps files to episodes by `SxxEyy` in the filename (in aired order unless configured otherwise). |
| **TVmaze** | M7 | Episodes, summaries (HTML), runtimes | No key. `/singlesearch/shows`, `/shows/{id}/episodes?specials=1`. |
| **Wikipedia** | M7 | Title, writer, director, air date, short summary | MediaWiki API wikitext; `{{Episode list}}` templates parsed with `mwparserfromhell`; must resolve transcluded season pages. |
| **OpenSubtitles** | M7 | Reference subtitles | REST API; needs a key and a login; daily download quota, so ingest is **resumable** and picks up where it stopped. |

TMDB terms: the README and `showvenist --version` show "This product uses the TMDB API but is not endorsed or certified by TMDB."

### 6.5 Merging sources (M7)

**The primary source decides which episodes exist; secondary sources only add data.**

- The primary source (TMDB by default) defines the episodes, their IDs, their numbering and their alternative orderings.
- Secondary episode lists are **lined up by content, not by number**: a sequence alignment scored on fuzzy title similarity, air-date proximity and production code. This handles:
  - double-length pilots counted as one episode or two,
  - specials placed inside a season or in S00,
  - anime cours split into separate seasons or merged into one,
  - unaired episodes appended or slotted in.
- Lined-up episodes get the extra data. Synopses from every source are kept; credits and guest characters are combined and deduplicated.
- Episodes that can't be lined up, and numbering disagreements, are recorded as **conflicts** in `provenance` and listed in the quality report. They're **never added automatically**; you resolve them in overrides.

### 6.6 Overrides

`overrides.yaml` is hand-written, keyed by episode ID, and merged over `show.json` on load. It survives re-ingests. The tool **never writes it**. Comments are encouraged, to record *why* something was overridden.

```yaml
# Firefly overrides
episodes:
  tmdb:62091:
    synopses:
      - source: manual
        text: "Mal's crew takes on passengers…"   # TMDB overview was one line
    runtime_min: 86
  tmdb:62104:
    excluded: true          # unaired; not on my disc set
orderings:
  release_bluray_2008:      # custom ordering for an odd release
    - { season: 1, episode: 1, episode_id: tmdb:62091 }
```

- Validated on load with pydantic. Errors name the file and line.
- After a refresh, overrides pointing to episodes that no longer exist produce warnings.
- `showvenist bundle edit <show>` opens `$EDITOR` and validates on save, reopening the editor on errors.

### 6.7 Data-quality report

Printed at the end of every ingest and by `bundle check`. Per season:

| Metric | Why it matters |
|---|---|
| Share of episodes with a synopsis, median synopsis length | Synopsis and LLM scorers |
| Share with non-generic guest characters (filters roles like "Cop #2" or "Waitress") | Guest-character scorer |
| Share with writer/director | Credits scorer (v2) |
| Share with reference subtitles | Strongest dialogue signal |
| Whether runtimes vary per episode or are one placeholder value | Duration scorer σ |
| Unresolved merge conflicts | Data you should look at |
| Reference subtitles that look like another episode: each reference's clean text is scored against every episode with the guest-character and synopsis scorers, and one whose best match isn't its own episode is listed with the episode it resembles | A mislabeled or misordered reference (common on OpenSubtitles) gives up to +12 to the wrong episode, more than any other signal can outvote. Listed, not excluded, because the check relies on the weaker scorers; you fix it in `overrides.yaml` |

It ends with a plain-language prediction, e.g. *"Season 2: synopsis coverage 40%, no reference subtitles. Expect many files to need review. Consider `--subs` or enabling the LLM scorer."*

### 6.8 Refreshing

- Bundles **never update silently**, because history records the `bundle_revision` each match was made against.
- `ingest --refresh` re-fetches and prints a diff: episodes added or removed, renumbered, episode groups added or removed, a change in which group ranks first for its type (what Jellyfin uses for that display order), and synopses, credits or subtitles added.
- `identify` warns when a bundle is older than 30 days (configurable) **and** TMDB reports the show as still in production.
- Later: flag library files that were named with numbering that has since changed.

---

## 7. Local state

### 7.1 File identity

A file's identity is its **content fingerprint**, never its path: paths change, partly because showvenist renames them. Fingerprint = file size + BLAKE2b hash of its first 64 KB and last 64 KB. It's fast even on 30 GB remuxes. A remuxed file gets a new fingerprint, which is correct. Two paths with the same fingerprint are byte-identical copies: they're reported, and only one is processed.

**Content key.** Two rips of the same disc title get different fingerprints even though their audio and video are identical. MakeMKV writes random segment, track and chapter IDs and a mux date into each file's header; two such files in the library are the same size and differ in 105 bytes, all near the start. The content key, file size + BLAKE2b hash of 64 KB at 25%, 50% and 75% of the file, skips the header and finds these **exact duplicates** (§14). It's stored alongside the fingerprint but never used as identity.

### 7.2 `state.db` (kept; losing it loses data)

Location: `$XDG_DATA_HOME/showvenist/state.db`. SQLite through the standard library `sqlite3`, no ORM, WAL mode.

| Table | Columns | Purpose |
|---|---|---|
| `media_file` | `fingerprint` PK, `content_key`, `size`, `last_path`, `show_slug`, `season_label`, `disc_label`, `first_seen`, `last_seen` | Known files. `disc_label` survives flattening, so blocks from earlier discs can still constrain later ones. |
| `run` | `id`, `started_at`, `command`, `args` | One row per `identify` / `assign` / `undo` invocation |
| `match` | `id`, `fingerprint`, `show_slug`, `episode_id` (NULL = extra), `season`, `episode`, `title`, `status` (`proposed`/`confirmed`/`rejected`), `source` (`auto`/`manual`/`name`), `confidence`, `margin`, `bundle_revision`, `run_id`, `created_at` | Match history. S/E/title are stored as they were at the time, so records survive bundle changes. |
| `rename_op` | `id`, `run_id`, `fingerprint`, `old_path`, `new_path`, `status` (`planned`/`done`/`undone`), `applied_at`, `undone_at` | The move log, written before each move (§13.4); also drives `undo` |

Lifecycle:

- `identify` records `proposed` matches.
- Applying a rename, pinning in the TUI followed by applying, or `assign` makes a match `confirmed`.
- A file already named as an episode is recorded as `confirmed` with source `name` the first time it's seen (§5.5).
- When you override a proposed match, the old one is recorded as `rejected`. Rejected matches are negative labels for eval.
- **Confirmed extras** are `match` rows with `episode_id = NULL` and `status = confirmed`. Re-runs skip those fingerprints.

Schema migrations use a `PRAGMA user_version` migrator from the first release.

### 7.3 `analysis.db` (disposable)

Location: `$XDG_CACHE_HOME/showvenist/analysis.db`.

`analysis(fingerprint, extractor, extractor_version, params_hash, payload, created_at)`

- Caches probe results, OCR'd text **per subtitle window**, and LLM responses (§10.2).
- Bumping an extractor's version invalidates its old entries.
- Deleting this file costs only recomputation. **It never touches history.**

---

## 8. Media analysis

### 8.1 Media layer

**PyAV** for demuxing, seeking and decoding in-process (its wheels bundle ffmpeg's libraries). The `ffprobe`/`ffmpeg` command line is an optional fallback. Containers: MKV is the primary target; anything ffmpeg reads should work, and v1 tests MKV and MP4.

### 8.2 Subtitle track selection (no OCR needed)

1. Keep tracks whose language tag matches the bundle's language. MakeMKV takes tags from disc metadata, so they're reliable.
2. Drop tracks flagged *forced* and tracks with few events. Events are counted from packets, which is cheap.
3. Prefer the track with the most events (SDH tracks are fine; normalization strips their annotations).
4. If there's no matching track, the dialogue scorers report "not available" (0 evidence), not negative evidence.

### 8.3 Dialogue sources

- **Text subtitles** (SRT, ASS, closed-caption tracks): parsed directly.
- **Image subtitles: PGS (Blu-ray) and VobSub (DVD). Required in v1**, because retail rips almost never carry text subtitles.
  - Decoded to bitmaps via PyAV (`BitmapSubtitle.planes` + palette).
  - Preprocessing: palette → grayscale, binarize, invert to dark text on light, upscale low-resolution VobSub.
  - OCR with Tesseract in the bundle's language.
- **Normalization:** strip SDH annotations (`[door slams]`, `(laughing)`), speaker labels and markup; lowercase; remove punctuation; collapse whitespace.
- **OCR quality:** the share of the file's normalized words found in the show's vocabulary (its reference subtitles and bundle text) plus a dictionary for the bundle's language. Text subtitle tracks count as 1.0. It gates negative reference evidence (§9.2) and is shown by `inspect`, the evidence pane and eval.

### 8.4 Staged extraction

Subtitle packets are spread through the whole file, so a full track means a full read: a 24-episode Blu-ray season is about 130 GB. Extraction therefore escalates in stages:

| Stage | What | Cost |
|---|---|---|
| 1. Probe | Header (duration, streams, chapters, title tags) + fingerprint + content key | Instant |
| 2. Sampled windows | Seek to ~20%, 50% and 80% of the runtime; decode subtitles until each window has ≥ 60 lines or reaches 5 minutes | ~20% of the file |
| 3. Full track | Only for files still below high confidence after the joint solve; then re-solve | Full read |

Each window is cached separately, so escalating never re-reads.

**v2: credits frames.** Decode **keyframes only** over the first 12 and last 4 minutes (configurable). Blu-ray GOPs of about 1 second give roughly one frame per second, with sequential reads.

**v3: speech-to-text.** faster-whisper on CPU (int8, small model) over the same sampled windows. Used only when a file has no usable subtitle track.

### 8.5 Concurrency

- **I/O stage:** limited per storage root (default 1). Parallel reads on one HDD array or network share cause seek thrashing and are slower, not faster.
- **Compute stage:** OCR (and later speech-to-text) runs in a process pool sized to the CPU cores.
- The stages form a pipeline: reading the next file overlaps with OCR of the current one.

### 8.6 Performance target

To be verified with eval: a 24-episode Blu-ray season identified **without escalation in ≤ 5 minutes** on gigabit-class I/O, and faster on the server's local disks. The work is I/O bound. OCR of about 2,000 sampled subtitle images is roughly 3 CPU-minutes, overlapped with reading.

---

## 9. Scoring

### 9.1 Evidence framework

Each scorer *k* emits, for each file *i* and candidate episode *j*, a log-likelihood ratio in nats:

```
LLR_k(i,j) = log  P(observation_k | file i is episode j)
                  ───────────────────────────────────────
                  P(observation_k | file i is background)
```

"Background" means the file is not episode *j*: another episode, or an extra.

- Positive values are evidence for, negative values are evidence against, and **0 means no information**.
- Dialogue scorers keep a score **per sampled window** as well as per file. The solve uses the file's score; the per-window scores reveal play-all titles, where different windows point to different episodes (§14).
- A missing observation contributes exactly 0. That includes a reference subtitle missing for only *some* candidates. If a file clearly fails to match the 18 reference subtitles that exist, the 4 episodes without one correctly become more likely. No special cases are needed, with one condition: "clearly fails" has to be a real measurement, not an OCR failure. Negative reference evidence therefore depends on the file's OCR quality (§8.3).
- Each scorer declares whether it can produce **negative** evidence. Signals that can't (OCR may simply miss a name) never penalize.
- Each scorer has a **clamp range**, so one badly tuned signal can't swamp the others.
- **Scorers must not share evidence.** Adding scores assumes the scorers are independent given the episode. Where two would read the same evidence, either their inputs are made disjoint (names belong to the guest-character scorer only) or one replaces the other (the LLM, §11.1). Otherwise the shared evidence counts twice and confidence is overstated.
- All parameters below are **placeholders until `eval --fit`** replaces them (§16.3).

### 9.2 Scorers

| Scorer | Milestone | Direction | Feature | Default mapping |
|---|---|---|---|---|
| Duration | M3 | for and against | Gaussian on (file duration vs. episode runtime) against a broad background density; best of 1.0× and PAL 0.959× speed | σ = 2 min if the season's runtimes vary, 4 min if they're all one value (likely a placeholder); falls back to the season median; clamp [−8, +2] |
| Track/container title | M3 | for only | rapidfuzz ratio of the MKV title tag against episode titles (and `alt_titles`) | ratio ≥ 90 against a non-generic title → +6, else 0 |
| Reference-subtitle dialogue | M3 | for and against | Containment of IDF-weighted word 5-grams: weighted \|shingles(file) ∩ shingles(ref)\| / weighted \|shingles(file)\| | Piecewise linear: c ≤ 0.02 → −4, c ≥ 0.25 → +12; scaled down when the file has little dialogue; **negative values scaled by OCR quality** (none below ~0.7, full above ~0.9, linear between; thresholds set by eval in M7), because bad OCR lowers containment against every reference, its own included; same language required |
| Synopsis dialogue | M3 | for only | BM25: each episode's title + synopses are the documents, the file's dialogue is the query, all cast and character names removed from both; robust z-score within the file | Credited only above the expected chance maximum over E candidates (≈ √(2 ln E)); 0.8 per σ; clamp +4 |
| Guest character names | M3 | for only | IDF-weighted distinct guest character names heard in the dialogue; names that are dictionary words are filtered out | clamp +6 |
| LLM | M3 (optional) | for and against | §10; replaces synopsis, guest characters and cues for the files it scores (§11.1) | w · (log p_j − log p_none); clamp ±3; w fitted |
| Dialogue cues | M3 (experimental) | for only | BM25 of the dialogue against each episode's `llm_cues`, ignoring cues that are cast or character names | Fitted separately; removed if the gate shows no gain |
| Credits | v2 | for only | Fuzzy match of OCR'd credit text against the closed set of person names among the candidates (writers, directors, guest actors), IDF-weighted | clamp +8 |

Notes:

- **Track/container title** will rarely fire on raw rips. In 1,738 MakeMKV rips from the server, no title tag held an episode title: they hold disc names (§5.1) or nothing. It stays because it costs nothing and catches files tagged by other tools.
- **Duration** mostly *rejects* extras and play-all titles. Within a season of identical runtimes it can't tell episodes apart, and its positive clamp reflects that.
- **IDF weighting** (computed across the candidate set) automatically discounts what's common to every episode: main-cast names, subtitled theme songs, catchphrases.
- **Guest characters** are kept separate from the synopsis scorer because TMDB's character lists are structured and far more reliable than words picked out of a synopsis. Synopses name guest characters too, so the synopsis and cues scorers ignore all cast and character names; otherwise one name heard in the dialogue would count two or three times.

---

## 10. LLM usage

### 10.1 Principles

- **The LLM produces evidence; the solver decides.** LLMs are bad at satisfying a constraint like "each episode at most once" across a season; the assignment solver is exact.
- **Strictly additive.** Everything works offline without an API key. The LLM scorer is off unless configured.
- **Replaces, doesn't add.** The LLM reads the same dialogue, synopses and guest names as the synopsis, guest and cues scorers, so adding its evidence to theirs would count the same evidence twice and inflate confidence. For a file it has scored, its evidence replaces those three (§11.1). It never sees any scorer's results and isn't sent the file's duration.
- **Not used for** OCR (v1), identifying the show, or the final decision.
- **Data leaving the server:** when enabled, dialogue excerpts and episode metadata are sent to the Anthropic API. This is opt-in and documented.

### 10.2 Per-file scorer (M3, optional)

**Inputs:**
- show name;
- the **candidate list**, each with title, synopses and guest characters (see *Candidate scope* below);
- the file's OCR'd dialogue excerpt (about 3k tokens). **Not** its duration: the duration scorer already models it, and sending it would count it twice.

**Candidate scope:**
- **Season split into disc folders: one disc at a time.** Disc folders are taken as correct, and every episode on disc *d* comes before every episode on disc *d* + 1. So a file on disc *d* can only be one of the **next n_d episodes** after the last episode of disc *d* − 1. Here n_d is the number of files on disc *d* that aren't identified extras, exact duplicates or play-all titles. The last episode of the previous disc comes from confirmed history if there is one, otherwise from the current solve. The first disc numbered 1 starts at the season's first episode. After a missing disc number, the window runs to the end of the season.
- **Season in a single folder: the full season.**
- **No season given:** the top 30 by the other signals.

The disc window is structural: it comes from the folder layout and disc order, not from ranking the other signals. A top-K by those signals would risk dropping the true episode exactly when they're weak, which is when the LLM is needed.

**When the disc window fails**, the file is flagged "disc window failed". That happens when the LLM rates "none of these" highest for a file that isn't an identified extra, or when the file is still below high after the re-solve. The likely cause is a wrong boundary on the previous disc. The TUI offers to re-ask with the full season (`L`), and report mode lists the flag. `[llm] scope = "season"` makes the full season the default.

**Output:** structured output validated against a pydantic schema (`client.messages.parse()`):
- a probability for each candidate plus "none of these";
- 1–3 quoted dialogue lines supporting each candidate rated above a threshold.

Probabilities are validated and renormalized.

**Quote check.** Each quote is fuzzy-matched against the excerpt that was actually sent. If a quote isn't there, the model recited it from memory or hallucinated it, and that candidate's positive evidence is dropped. Verified quotes are shown in the TUI's evidence pane.

**Correcting list-position bias.** LLMs favour candidates near the top of a list. Each season uses two fixed candidate orders, forward and reversed, and the two results are averaged (`send_both_orders`, on by default). Eval decides whether that's worth double the per-file cost.

**Prompt layout for caching:** instructions → candidate block (explicit cache breakpoint, 5-minute TTL) → per-file excerpt. Every file in a season shares the cached prefix, and requests seconds apart keep the cache warm.

**Determinism.** Current models reject sampling parameters, so there's no "temperature 0". Instead, responses are **cached in `analysis.db`**, keyed by (model, prompt version, candidate-block hash, excerpt hash). Re-solves and re-runs are identical and cost nothing.

**Model:** default `claude-sonnet-5-5`, configurable. Adaptive thinking at `low` effort to start, tuned via eval. One model, no cheap-then-expensive cascade: caches don't carry over between models, and a single model at the right effort is usually simpler and as good.

**Refusals:** after a response with `stop_reason == "refusal"`, the file stays in the base regime (§11.1) and the refusal is logged. The API's server-side fallback (`fallbacks: "default"`) is enabled on the Claude API.

**Which files are sent:**
- `auto`: after the first solve, files below high confidence are sent, then the season is re-solved;
- `all`: every file (eval and fitting);
- the TUI's **`l` key** sends the selected file on demand, and **`L`** sends it with the full season (§10.2, *When the disc window fails*).

### 10.3 Ingest-time dialogue cues (M3, experimental)

The idea is to spend the LLM once per show, not once per file.

- At ingest, for each episode, the model generates:
  - character and place names,
  - distinctive terms and objects,
  - optionally up to 3 memorable lines,

  based on the synopses plus what it knows about the show. It's told to leave out anything it isn't confident about.
- Stored in the bundle as `llm_cues` with `llm:<model>` provenance; overridable.
- Generated through the Batches API (half price) when you can wait for it.
- Used by the offline **cues scorer**, which only ever adds evidence, so invented cues mostly just fail to match. The M3 gate decides whether it stays.

### 10.4 Cost

Estimates, to be replaced by real figures: eval reports actual cost per season from the API's usage data.

| Setting (Sonnet 5.5, both orders) | Per 24-episode season |
|---|---|
| Every file sent | ≈ $0.75 |
| Only files below high (~25%) | ≈ $0.20 |
| Cue generation | Cents per show |

---

## 11. Fusion and assignment

### 11.1 Score

```
S(i, j)         = log_prior(j) + Σ_k  w_k · clamp_k( LLR_k(i, j) )
S(i, unmatched) = 0          ← the background hypothesis is the reference point
```

**Two regimes.** The sum ranges over a different set of scorers depending on whether the LLM has scored file *i*:

| Regime | Scorers summed | Weights |
|---|---|---|
| Base (no LLM result) | duration, title, reference subtitles, synopsis, guest characters, cues | `[weights]` |
| LLM (the LLM has scored the file) | duration, title, reference subtitles, LLM | `[weights.llm]` |

In the LLM regime, the LLM's evidence **replaces** the synopsis, guest and cues scorers, because it reads the same dialogue, synopses and guest names they do (§10.1). A refusal or failed call leaves the file in the base regime.

### 11.2 Prior

```
log_prior(j) = log( (1 − π_x) / E ) − log( π_x )
```

π_x is the expected fraction of extras and E is the number of candidates. π_x defaults to 0.15 until it's measured on hand-labeled raw rips (§16.2). Final discs carrying dozens of titles suggest the real share is higher. "Unmatched" is a real competing hypothesis, and the prior is easy to reason about:

- with E = 22, an episode needs more than about 1.4 nats of net evidence to beat "extra";
- with E ≈ 200 (no season given), it needs about 3.5.

More candidates automatically demand more evidence.

### 11.3 Scope

The season folder and `--episodes` are **hard filters**: out-of-scope episodes aren't candidates. The season folder is read in the ordering being solved, so each ordering tried by ordering detection (§11.6) brings its own candidate set. An ordering without real seasons (one long absolute list) keeps the starting order's season and only changes the episode order. They're still scored, so the report can warn when a file's best out-of-scope score beats its assigned score by more than 4.6 nats ("evidence points to S03E07").

Specials (S00) are candidates only with `--include-specials` or under a `Specials` / `Season 0` folder.

### 11.4 History prior

An episode already confirmed for a *different* fingerprint gets −h added (default h = ln 10, configurable). It's a soft penalty, not an exclusion, so a re-rip can still match. If such an episode still wins, the report shows a **duplicate warning** naming the existing file's last known path. Eval turns this prior off (§16.1).

### 11.5 Assignment

**Basic assignment** (plain file lists, or `--no-blocks`). An F × (E + F) score matrix. The extra F columns are each file's private "unmatched" option with score 0; every other file gets −∞ in those columns. It's solved with `scipy.optimize.linear_sum_assignment(maximize=True)`, so each episode is used at most once and any file can stay unmatched.

**Disc block assignment** (the default whenever there are disc folders):

- Episodes are positions 1..E in the matching order within the season. Discs d = 1..D are sorted by their parsed number. Disc *d* gets the block [a_d, b_d); blocks are in order and don't overlap.
- Block score:

  ```
  block_score(d, a, b) = BA(files of disc d → positions [a, b), with unmatched options)
                         − u · (number of positions in [a, b) that no file claims)
  ```

  BA is the basic assignment above. Block length is bounded by the number of ready files on the disc. For a disc marked **incomplete** (§5.2), u = 0 and its upper boundary is left open.

- Dynamic programming over discs:

  ```
  best(d, b) = max over a ≤ b, b' ≤ a of:
               best(d−1, b') + block_score(d, a, b) − gap(d, b', a)
  ```

  `gap` charges g per skipped position, except:
  - it's free where disc numbers skip (folders 1, 2, 4 leave room for disc 3);
  - it's free before the first disc present if that disc's number is greater than 1;
  - it's always free after the last disc present (later discs may not be ripped yet).

- Discs already confirmed in history (via `media_file.disc_label`) enter the DP as **fixed blocks**. So a disc processed on its own still gets a tight window.
- Cost: about D·E²/2 small assignment solves. That's roughly 1,800 for 6 discs × 24 episodes, which takes milliseconds. Confidence (§11.7) re-runs the DP once per file, so block scores are cached and reused for every disc a change didn't touch.

### 11.6 Ordering detection

Disc blocks are consecutive in the *release's* order, which may not be the bundle's aired order. For example, some releases follow the creator's intended order.

1. Solve with blocks in the starting order (`--order`, default the show's `naming_order`).
2. Compare the block solution's total with the basic assignment's total. If the blocks fit badly (the block total falls short by more than a configurable threshold), solve with blocks in every other ordering in the bundle (each TMDB episode group, custom orderings), with the season filter recomputed in each (§11.3), and keep the best.
3. Report the result: "This release follows 'Intended Order'."
4. If no ordering fits, warn that the discs don't look sequential and fall back to the basic assignment.

The matching order affects the candidate set and the block constraint. Filenames always use `naming_order` (§13.1). When an episode's season in `naming_order` differs from its folder's season, the plan moves it to the naming season's folder. That move is listed explicitly and needs a pin: **a file never changes season automatically.**

### 11.7 Confidence

```
Δ_i = best_total − best_total_with( (i, σ(i)) forbidden )      odds ≈ e^Δ_i
```

This reruns whichever solver produced the result (basic assignment or block DP).

| Level | Rule | Meaning |
|---|---|---|
| High | Δ ≥ 4.6 | ≈ 99:1 or better; **included in the move plan without review** |
| Medium | 2.2 ≤ Δ < 4.6 | ≈ 9:1 to 99:1; needs a pin |
| Low | Δ < 2.2 | needs a pin |

The levels apply the same way whether a file is assigned an episode or its unmatched column. A file left unmatched with Δ ≥ 4.6 is an **identified extra**: it goes in the move plan without review, like a high-confidence episode. Below high, it's a candidate extra and needs a pin. Most extras are identified this way, because the duration scorer strongly rejects every episode for them. Extras close to episode length get a small Δ and go to review.

Why this measure:
- **Exact and cheap:** F small re-solves.
- **Respects the constraints.** If two files could be swapped (say, parts 1 and 2 of a two-parter), forbidding either pair produces the swap, so both get a small Δ and both go to review.
- The re-solve also yields the **runner-up** shown in the report.

### 11.8 Calibration

The thresholds only mean something if the evidence scores are calibrated. `eval` reports a **reliability check**: what fraction of each confidence level was correct, with high held to the error bound in §16.1. `eval --fit` fits the weights w_k (§16.3).

---

## 12. Review TUI and reports

### 12.1 Engine and UI are separate

The engine exposes a `Session` API. The TUI and report mode are both thin clients of it, and **no matching logic lives in the UI**:

| Operation | Effect |
|---|---|
| `analyze()` | Runs extraction; streams per-file progress and results |
| `solve()` | Fusion + assignment + confidence |
| `pin(file, episode \| extra)` / `unpin(file)` | Fix or release a match; triggers a re-solve |
| `ask_llm(file)` | Run the LLM scorer on one file; triggers a re-solve |
| `plan()` | Build the move plan (§13) |
| `apply(plan)` | Execute it, writing the move log first |

### 12.2 TUI (Textual)

Full-screen, keyboard-driven, works over SSH (including Windows Terminal).

```
┌ Firefly · Season 1 · DVD order ──────────────── analyzing 15/17 ┐
│ FIREFLY_D1              │ FIREFLY_D3 / title_t01.mkv   44m03s   │
│  ✓ t00 E01 Serenity     │ ● a) E09 Ariel          14:1          │
│  ✓ t03 E03 Bushwhacked  │   b) E10 War Stories     1:14         │
│  ✗ t04 extra (6m12s)    │ Evidence      a        b              │
│ FIREFLY_D3              │  refsubs    +9.1     −2.0             │
│  ? t01 E09 Ariel  14:1  │  guests     +2.3      0               │
│  … t02 (writing…)       │  duration   +0.4     +0.4             │
│                         │ Lines: "…" · "…" · "…"                │
│                         │ Synopsis (a): …                       │
├─────────────────────────┴───────────────────────────────────────┤
│ a/b accept · e enter S01E## · x extra · u unpin · l ask LLM ·   │
│ p plan · q quit                                                 │
└─────────────────────────────────────────────────────────────────┘
```

- **Left pane:** season → disc → file tree with status: ✓ high, ? needs review, ✗ extra, … not ready or still analyzing.
- **Right pane:** the selected file's candidates with odds; each signal's contribution for the winner vs. the runner-up; matched dialogue lines; guest names found; duration vs. runtime; the synopsis; verified LLM quotes when available.
- **Live updates:** results appear as each file finishes analysis.
- **Pin + re-solve:** accept a candidate, enter `S01E##`, mark as an extra, or unpin. Each action re-solves the season immediately (fast, because analysis is cached and block scores are reused for discs the pin didn't touch, §11.5) and moves the cursor to the **next least-confident file**. Files that become high confidence drop out of the queue, so you usually review fewer files than were initially flagged.
- **`l` ask LLM** (when configured): runs the LLM scorer on the selected file and re-solves. **`L`** does the same with the full season as candidates, offered when the disc window failed (§10.2).
- **`p` plan screen:** every planned move (before/after paths, sidecars, extras moves), collision check, then confirm → apply → summary.
- **Startup dialogs:** show disambiguation, marker-file confirmation, disc-order confirmation for unnumbered discs.

### 12.3 Report mode

`--report` (or non-TTY output) prints a table per season, grouped by disc, for each file:
- chosen S/E and title,
- odds (e.g. "120:1"),
- each signal's contribution compared with the runner-up,
- flags: duplicate, out of scope, possible two-parter, play-all, not ready.

Report mode **never touches media files**: no renames, no moves.

`--json` emits the same content in a stable, versioned schema (`report_version`), for scripts and eval.

---

## 13. Renaming

### 13.1 Naming

- Template (configurable): `{show_snake}_s{s:02}e{e:02}`, giving `the_west_wing_s01e02.mkv`: the convention already used in the library, which Jellyfin parses. `{show_snake}` is the show title lowercased, with each run of other characters replaced by `_`. `{show}` and `{title}` are also available. The extension is kept.
- Season and episode numbers come from the show's **`naming_order`** (marker file): any ordering in the bundle. Default: aired. This is deliberately separate from the matching order used by the block model.
- **Choosing `naming_order`** depends on where the media server gets episode titles:
  - **Fetched from TMDB by episode number:** `naming_order` must match the series' Display order in Jellyfin. Jellyfin maps a display order to a TMDB group type and uses the **first** group of that type in TMDB's list (verified in Jellyfin's TMDB provider). `naming_order` therefore also accepts Jellyfin's keywords (`dvd`, `absolute`, `production`, `storyArc`, `digital`, `tv`, `originalAirDate`) and resolves them the same way, to the group with `type_rank` 1.
  - **Entered by hand:** `naming_order` is your own convention. Where air order differs from series chronology, that's chronology: for Firefly, the "Intended Order" group.
- Titles are sanitized for filesystem-safe characters.

### 13.2 What goes in the plan

- High-confidence matches and identified extras (§11.7), and pinned matches of any confidence. Medium and low matches only once pinned.
- **Files already named as episodes are never renamed or moved automatically** (§5.5), not even to apply the naming template or to flatten a disc folder.
- **Hierarchy input:** matched files are **flattened** from `<Season>/<Disc>/` into `<Season>/`.
- **Plain file arguments:** renamed in place.
- **Identified and confirmed extras** move to `<Season>/extras/`, which Jellyfin treats as extras. This is on by default (`move_extras`) and each one can be unticked on the plan screen.
- Unresolved files stay where they are. **No folder is ever deleted**, even when it ends up empty.
- **Sidecars** sharing the video's basename (`title_t00.en.srt`, `.nfo`) move with it.

### 13.3 Safety checks before anything moves

- All target paths are computed first. **Any collision aborts the whole plan.** Existing files are never overwritten.
- Write permission is checked on every source and target directory (Samba ownership, §3.1).
- Moves are same-filesystem renames (atomic). A move that would cross filesystems is reported and needs explicit confirmation.

### 13.4 Move log

Each move is journaled in `rename_op`:

1. insert the row as `planned`;
2. `rename()`;
3. mark it `done` and confirm the match.

On startup, leftover `planned` rows are checked against which path actually exists, then marked `done` or rolled back.

### 13.5 Undo

`undo [--last | RUN_ID]` lists a run's `done` moves, asks for confirmation, and moves the files back in reverse order. The undone matches go back to `proposed`. Collisions during undo abort with a report rather than overwriting anything.

---

## 14. Edge cases

| Case | Handling |
|---|---|
| Extras, featurettes, trailers | Unmatched (duration and dialogue evidence lose to the extras prior). High-confidence extras are identified and moved to `extras/` without review, like high-confidence episodes (§11.7); the rest go to review |
| Play-all titles | Duration ≳ 1.8 × the median runtime **and** sampled windows pointing strongly to different episodes (per-window scores, §9.1) → flagged "play-all" and kept out of the solve |
| Two-part episode muxed as one file | Duration ≈ two consecutive episodes' runtimes and both score high → flagged in v1; resolved with two-position candidates in v2 |
| Duplicate rips (same episode twice, alternate angle) | **Exact duplicates** (same content key, §7.1) are found at probe time: one copy is analyzed and solved, and the others are flagged as its duplicates. **Near duplicates** (e.g. a playlist with an extra recap, so a different size) are flagged by high dialogue similarity; `--allow-duplicates` relaxes at-most-once for them. A content-key match against history is reported as a re-rip of a confirmed file. Only one copy of a duplicate set goes in the move plan (the first by path, switchable in review); the others stay where they are, flagged, for you to keep or delete |
| Files still being written over SMB | Readiness check (§5.2): skipped and reported; disc block marked incomplete |
| Samba ownership / no write permission | Checked before planning; clear error naming the path and the user showvenist runs as |
| Missing disc folders | Gaps allowed where disc numbers skip (§11.5) |
| Skipped titles (MakeMKV minimum length) | Unclaimed position inside a block costs a soft penalty, not a failure |
| Release order differs from aired order | Ordering detection (§11.6) |
| An episode is a special in one ordering and a regular episode in another (Firefly's three unaired episodes) | The season filter is read in the ordering being solved (§11.3) |
| An episode's naming season differs from its folder's season | Planned move to the naming season's folder, listed explicitly; needs a pin (§11.6) |
| Files already named as episodes | Ground truth: pinned, not analyzed, never renamed or moved automatically (§5.5) |
| A single disc ripped on its own | Fixed blocks from history, or `--episodes` |
| Unnumbered or identical disc labels | Order suggested by folder creation time, confirmed by you; non-interactive runs fail |
| PAL speedup (4% shorter) | Duration scorer tests both 1.0× and 0.959× |
| Subtitle language doesn't match the bundle | Dialogue scorers unavailable (0 evidence); flagged |
| SDH tracks / forced-only tracks | SDH annotations normalized away; forced tracks skipped |
| Missing per-episode runtimes | Falls back to the season median with a wider σ |
| Anime absolute numbering | `orderings.absolute` |
| Season not given (no hierarchy, no flag) | All seasons are candidates; the prior demands more evidence (§11.2) |
| No usable subtitle track at all | Dialogue scorers unavailable; v3 speech-to-text fills the gap; until then such files go to review |
| Dialogue-free sampled windows | Windows extend until a line count is reached; fully silent files escalate to a full read |

---

## 15. Architecture, dependencies, packaging

### 15.1 Package layout

```
showvenist/
  cli.py            typer entry point
  config.py
  session.py        engine API used by TUI and report mode (§12.1)
  bundle/           schema.py, store.py, merge.py, overrides.py, quality.py
  adapters/         tmdb.py, localsubs.py, tvmaze.py, wikipedia.py, opensubtitles.py, llm_cues.py
  state/            db.py, migrations.py, fingerprint.py, history.py
  hierarchy/        parse.py (patterns, volume labels), marker.py, readiness.py
  analysis/         probe.py, tracks.py, subtitles.py, ocr.py, windows.py, asr.py (v3), cache.py
  scoring/          duration.py, title.py, refsubs.py, synopsis.py, characters.py,
                    cues.py, llm.py, credits.py (v2)
  fusion.py
  assign/           basic.py, blocks.py (DP), ordering.py, confidence.py
  rename/           plan.py, movelog.py, sidecars.py, undo.py
  report.py         rich tables + JSON
  tui/              Textual app
  eval/             harness.py, fit.py, manifest.py
```

The library API mirrors the CLI (`ingest()`, `identify()` returning a `Session`).

### 15.2 Dependencies

| Kind | Packages |
|---|---|
| Core | httpx, pydantic, typer, rich, textual, rapidfuzz, scipy, numpy, Pillow, av (PyAV), Tesseract bindings (tesserocr or pytesseract), pysubs2, mwparserfromhell, PyYAML (read-only use) |
| Standard library | sqlite3, hashlib (BLAKE2b) |
| Optional extras | `[llm]` anthropic · `[asr]` faster-whisper (v3) |
| External binaries | tesseract + language data (**required in v1**); ffmpeg/ffprobe (optional fallback) |

- Python ≥ 3.11, managed with uv.
- **Platform: Linux only** (WSL counts). Paths follow XDG: `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `XDG_CONFIG_HOME` are honored.
- Designed to run headless over SSH.

### 15.3 Keys and quotas

| Service | Needs | Quota notes |
|---|---|---|
| TMDB | Free API key | Generous rate limit; ingest uses `append_to_response` to minimize calls |
| TVmaze | Nothing | Rate-limited; adapter backs off |
| Wikipedia | Nothing | Polite User-Agent required |
| OpenSubtitles | API key + account login | Daily download quota; ingest is resumable |
| Anthropic | API key or `ant auth login` profile | Paid per token; eval reports cost per season |

---

## 16. Evaluation and testing

### 16.1 Eval harness (from M3)

Without measurement, the weights and thresholds are guesswork. The harness is therefore part of the first scoring milestone, not an afterthought.

**Inputs:** either a manifest CSV or `--from-history`.

```csv
path,show,label,order,disc
/srv/media/tv/Firefly/Season 1/Firefly_s01e01.mkv,firefly-2002,S01E01,dvd,
/srv/rips/Firefly/Season 1/FIREFLY_D1/title_t04.mkv,firefly-2002,extra,,
```

- `show` is a bundle slug, so one manifest can cover many shows.
- `order` names the ordering the label is written in: any ordering in the bundle, including custom ones from `overrides.yaml`. Default `aired`.
- `disc` is optional. When it's empty, disc identity comes from the folder or the MKV title tag (§5.1).
- Every label is resolved to an episode ID before scoring. Labels that don't resolve to exactly one episode (e.g. a two-part pilot the bundle lists as one episode) are reported and left out, not counted as errors.
- Files that share a label form a **duplicate set**. They're scored on the duplicate flag (§14), not on assignment, and are kept out of precision at high.

`--from-history` uses confirmed matches (and confirmed extras) as positive labels and rejected matches as negative labels. Normal use keeps adding labels.

**Metrics:**

| Metric | Definition |
|---|---|
| Top-1 accuracy | Share of files whose assignment (episode or "extra") is correct |
| **Precision at high** | Share of high-confidence decisions that are correct, episodes and identified extras alike (an episode sent to `extras/` at high confidence is an error), with the one-sided 95% upper bound on the error rate (Clopper-Pearson). **Target: bound ≤ 1%** |
| **Coverage at high** | Share of files that reach high confidence, as an episode or as an identified extra (how much review is avoided). **Target ≥ 90%**, secondary to precision: a file flagged for review costs a minute, while a wrong match can push other files out of their places in the season's solve. Thresholds are never loosened to reach coverage |
| Per-season and per-show breakdown | All the above for each season and each show |
| Reliability check | Correct share per confidence level |
| Unmatched precision / recall | Detecting extras |
| Per-signal ablation | All the above with each scorer removed in turn |
| Blocks vs. no blocks | All the above with and without the block model (from M4) |
| Cost per season | Actual LLM token cost from API usage data |
| Time per season | Wall-clock time, split into I/O, OCR and solve |

Eval always runs with the **history prior turned off**, so it doesn't just reproduce its own labels.

**Why a bound, not a percentage.** 60 high-confidence files with no errors only show that the error rate is below about 5%. Showing ≤ 1% takes 299 files with no errors, or 474 with one. The bound treats files as independent, and they aren't: errors cluster by season (one badly OCR'd track, one thin bundle). That's why the gate also requires breadth across seasons and shows (§17.2), and why results are broken down per season and per show, where an aggregate would hide a failing season.

### 16.2 Labeled corpus

**The main corpus is the existing library** on the server (as of 2026-10-03): 422 files named by hand as `show_sXXeYY.mkv`, across 25 seasons of 8 shows, all MakeMKV rips with their subtitle tracks kept. The labels are treated as correct.

- **Label ordering.** Where air order differs from series chronology, files were named in chronological order. Two-part pilots count as two episodes. Manifests record the ordering per row (§16.1); a chronology that isn't among the bundle's orderings goes into `overrides.yaml` as a custom ordering.
- **Disc structure.** The files have been flattened, but about 290 of them carry a disc-level MKV title tag (§5.1). Blocks can be evaluated on The West Wing, The Wire, Supernatural, The Expanse and Firefly.
- **No extras.** The labeled discs have no leftover files, so extras are hand-labeled from the 1,314 unnamed rips (`*_tNN.mkv`). They're needed for the unmatched metrics, for fitting (§16.3), and to measure the real share of extras (π_x, §11.2). Most extras are obvious from their size. The few close to episode length matter most: they're what tests the balance between matching a file and leaving it unmatched.
- **Duplicates.** Supernatural S2 has 8 duplicate rips (`…a.mkv`) of episodes that are also present. The S02E09 pair is an exact duplicate: same size, 105 header bytes different. The other pairs still need checking. Exact pairs test the content key; any pairs of different sizes test near-duplicate detection. All of them are kept out of the precision gate.
- **Imbalance.** The West Wing is about a third of the corpus, so per-show results matter more than the aggregate.
- **No reference subtitles.** The rips carry only PGS and VobSub tracks, and there's no SRT collection. Reference-subtitle accuracy is therefore evaluated in M7, once OpenSubtitles is available.

To exercise every signal, the target variety is:
- a sitcom with cold opens (**missing**),
- a drama with guest-star credits (The West Wing, Supernatural),
- a show whose disc release order differs from aired order (Firefly),
- an anime (**missing**),
- at least one DVD set (VobSub) and one Blu-ray set (PGS) (Supernatural S2 is Blu-ray with PGS; **the rest still need checking**).

### 16.3 Weight fitting

`eval --fit`:
- fits the weights w_k with a **conditional logit**: for each labeled file, a softmax over its candidates plus "unmatched", using the §11.1 scores with the prior held fixed, maximizing the probability of the labeled option. That's the solver's own model without the at-most-once constraint, so fitted scores come out in the units the confidence thresholds use, and extras are simply files labeled "unmatched". Pairwise logistic regression on (file, episode) pairs was rejected: its intercept duplicates the prior, easy negatives dominate it, and "unmatched" never enters it;
- fits **weights only** at first: about 10 parameters across both regimes. Shape parameters (duration σ, BM25 and containment breakpoints) stay at their defaults and are released one at a time, only where eval shows a shape is clearly wrong. With a few hundred labeled files, fitting them all invites overfitting;
- **weights extras up to their real share** of raw rips. The named library contains no extras, and a fit that never sees a file that should stay unmatched learns to over-match;
- fits **two weight sets, each on the files it's used for** (§11.1): base weights on all files, without the LLM; LLM-regime weights only on the files `--llm auto` would send (eval runs a first solve and takes the files below high). Weights fitted on every file would be fitted mostly on easy files that `auto` never sends;
- is **regularized toward the built-in defaults**, because labeled data will be small at first;
- holds out **whole seasons** (leave-one-season-out), never single files: files in a season share a subtitle track type, a bundle and a disc set, so a file-level split leaks;
- writes the result to the `[weights]` section of the config;
- prints metrics before and after on the held-out seasons.

### 16.4 Automated tests

- **No copyrighted media in tests, ever.**
- **Synthetic media:** MKVs generated with ffmpeg (`testsrc`, controlled durations, muxed text subtitles). M2 includes a fixture generator that renders known text into **PGS and VobSub** bitmaps (Pillow) and muxes them, so OCR has known ground truth.
- **Adapters:** recorded HTTP fixtures (no live network in CI).
- **Engine:** scorers, fusion, basic assignment, block DP, ordering detection and confidence tested on synthetic score matrices with known answers.
- **State:** migration tests; move-log crash recovery (kill between `planned` and `done`); undo round-trip.
- **TUI:** Textual's headless driver (`App.run_test()` / Pilot) for pin → re-solve → plan → apply; snapshot tests for layout. Engine tests never touch the UI.
- **LLM:** recorded responses; quote-check and refusal paths tested offline.

---

## 17. Roadmap and milestones

### 17.1 How planning works

- **This spec owns the milestones:** goal, scope, exit criteria and order.
- **Beads (`bd`) owns the tasks.** One epic per milestone, with tasks that point to spec sections (e.g. "implements §11.5 block DP").
- Order is enforced with **"blocks" dependencies** between epics: **M0** → M1 → M2 → M3 → **M3 gate** → M4 → M5 → M6, and M7 is blocked by the gate. M0 ends in a decision that can revise this spec, so Beads epics for M1 onward are created only after M0. The gate is an explicit task that blocks everything after it.
- The spec contains no task lists, so the two can't drift apart.

### 17.2 v1 milestones

The order is chosen to **measure accuracy as early as possible**: the biggest unknown is whether matching works well enough on real rips. Without reference subtitles (§16.2), the main path is synopsis + guest names on OCR'd dialogue, so M0 measures that before anything is built. M1 and M2 contain only what the gate needs; everything else comes after it.

| Milestone | Scope | Exit criteria |
|---|---|---|
| **M0: Feasibility spike** | Throwaway code in `spike/`, not part of the package. On four labeled library seasons from four shows, two with PGS and two with VobSub (Supernatural S2, a Hulk season, The West Wing S1, and a fourth picked by probing track types): extract and OCR the subtitles, fetch the seasons from TMDB, score with synopsis BM25 (names removed) + guest names, fit their weights leave-one-season-out, solve with the basic assignment. If offline coverage falls short, run the LLM scorer on the files below high, with your OK and at most $1 per season | Measured: top-1 accuracy, the Δ distribution, coverage at Δ ≥ 4.6 with weights fitted on the other seasons (pooled and per season), OCR quality per track type, and (if run) what the LLM rescues and what it costs. **Decision, by rules set before the run** (`docs/plan.md` §2.5 has the checks that come first): pooled coverage at high ≥ 90% with no errors at high → the plan stands; ≥ 90% only with the LLM → the LLM becomes the recommended setup, and its default stays `off`; below 90% either way → reference subtitles move ahead of the M3 gate and the offline goal (§1.2) is revisited. Spec revisions happen before Beads epics for M1 onward are created |
| **M1: Bundle, TMDB, local SRT** | Schema and store; `overrides.yaml` merge; TMDB adapter including every episode group; local SRT adapter; data-quality report (`bundle check`) | Valid bundles for 3 shows of different types, each with a quality report; tests pass against recorded HTTP fixtures |
| **M2: Analysis** | Probe; fingerprint and content key; track selection; text subtitles; PGS/VobSub OCR; normalization; OCR quality; sampled windows and escalation; analysis cache; `inspect`; PGS/VobSub fixture generator | `inspect` shows clean OCR'd dialogue for a real PGS rip and a real VobSub rip; a second run reads nothing from disk |
| **M3: Scoring, assignment, report, eval** | Duration, title, reference-subtitle, synopsis and guest-character scorers; optional LLM scorer (quote check, two orders, response cache, refusal handling); experimental cue enrichment + cues scorer; fusion; basic assignment with unmatched options; confidence Δ; `identify --report/--json`; `eval` with all §16.1 metrics; `eval --fit` | Eval runs end to end on flat file lists, with and without the LLM |
| **M3 gate: accuracy** | Eval on the library corpus (§16.2), ≥ 10 seasons of ≥ 5 shows, in two setups: {with, without} LLM, plus cues on/off. No reference-subtitle setups: there are none until M7 (§16.2) | **Required, in every setup:** the one-sided 95% upper bound on the error rate at high is ≤ 1% (299 high-confidence files with no errors, or 474 with one). A setup with too few high-confidence files to reach that bound must have no errors at high, and its bound is reported. **Target:** coverage at high ≥ 90%; missing it means adjusting, never trading away precision. **Reviewed together:** coverage at high; per-signal ablation; OCR quality per track type (VobSub vs. PGS); what the LLM adds in coverage vs. its cost; whether cues earn their place; whether the two-order bias correction is worth it. **Outcome:** go, or adjust (weights, signals, LLM default mode and model) before anything else is built |
| **M4: Hierarchy and block model** | Path parsing incl. MakeMKV volume labels, title tags and the creation-time suggestion; files already named (§5.5); marker file and interactive show disambiguation; readiness check; concurrent read/OCR pipeline; season hard filter + out-of-scope warning; block DP; ordering detection; `--episodes` | Eval compares blocks vs. `--no-blocks` on labeled multi-disc seasons; Firefly's release order, which differs from its aired order, is detected; partially written files are skipped; the §8.6 performance target is measured |
| **M5: State and renames** | `state.db` + migrations; history prior + duplicate warnings; confirmed extras; `assign`, `history`, `forget`, `undo`; move plan with sidecars, extras moves, collision and permission checks; move log + crash recovery; `eval --from-history` | A kill between `planned` and `done` recovers correctly; undo round-trips; re-runs skip confirmed files |
| **M6: TUI** | `Session` API; Textual review UI (tree, evidence pane, pin + re-solve, ask LLM, plan screen, startup dialogs) | Pilot tests pass for pin → re-solve → plan → apply; you confirm it works over SSH from Windows Terminal |
| **M7: More adapters** | TVmaze; Wikipedia (MediaWiki API, `{{Episode list}}`, transclusion); OpenSubtitles (resumable within quota); multi-source merge with content alignment and recorded conflicts; `bundle edit`; `--refresh` with diff | A merged bundle with recorded conflicts; a refresh prints a diff; an OpenSubtitles ingest resumes after hitting its quota; eval runs with reference subtitles on the library corpus, measures OCR n-gram survival, and sets the OCR-quality thresholds (§9.2) |

### 17.3 Later

- **v2:** credits OCR and the credits scorer; two-position candidates that resolve two-part episodes; flagging library files named with outdated numbering.
- **Possible future:** match extras to TMDB's Specials (S00) entries, which for many shows include the disc extras (Firefly's list has its gag reel, a making-of documentary and an audition tape). Matched extras would be named as specials, so Jellyfin shows them with titles instead of filing them under `extras/`.
- **v3:** speech-to-text with faster-whisper on CPU (int8, small model) for files without usable subtitles.

---

## 18. Open questions and risks

| Area | Question / risk | Mitigation or plan |
|---|---|---|
| Accuracy | Without reference subtitles, synopsis + guest names may not separate episodes well enough, making the LLM the main path | M0 measures it on labeled seasons before anything is built |
| Accuracy | Initial weights before there's eval data | Conservative defaults; M3 gate; `eval --fit` |
| OCR | VobSub OCR quality (low resolution, coloured or italic text) may break exact 5-grams | Report OCR quality per track type at the gate; measure n-gram survival once reference subtitles exist (M7); fall back to shorter n-grams or fuzzy shingles for OCR'd tracks |
| Data | TMDB synopsis and guest-data coverage for older or obscure shows | Quality report predicts it; overrides; Wikipedia (M7); LLM scorer |
| Data | OpenSubtitles quota vs. full-season ingests | Resumable ingest |
| Data | How much Wikipedia template structure varies | Adapter tolerant of variants; conflicts surfaced, not guessed |
| Data | Episode ID stability for bundles without TMDB | Fallback ID scheme; S/E/title stored on every match |
| Naming | Jellyfin resolves a display order to the first TMDB group of that type, and anyone on TMDB can add groups, so the group Jellyfin uses can change under you | `naming_order` keywords resolve the same way (§13.1); `--refresh` flags a change in which group ranks first |
| Library | Unresolved files left in disc folders inside a Jellyfin library may be scanned as junk episodes | Finish review; identified and confirmed extras move to `extras/` |
| Block model | Size of the unclaimed-position and gap penalties, and the ordering-detection threshold | Tune with eval on real multi-disc seasons (M4) |
| Block model | Two-part episodes merged into one file break "one position per file" | Flag in v1; two-position candidates in v2 |
| Performance | Sampled windows landing on dialogue-free stretches | Line-count-driven windows; escalation |
| Performance | Server CPU core count bounds OCR throughput | Measure at M2; process pool |
| LLM | Invented dialogue cues; list-position bias; calibration of its probabilities; refusals on violent dialogue; price changes | Quote check; positive-only cues; two orders; fitted weight and clamp; refusal = 0 evidence; eval reports real cost |
| LLM | Haiku 4.5 as a cheaper alternative only caches prompts of ≥ 4,096 tokens | Default is Sonnet 5.5; revisit with eval data |
| Scope | Whether history needs to be shared or portable across machines | v1: local only |
| Containers | Official support beyond MKV | v1 tests MKV and MP4 |

---

## 19. Decision log

Decisions made while writing this spec (2026-10-02) and reviewing it (from 2026-10-03), with the reasoning, so later work doesn't re-argue them.

| # | Decision | Reasoning |
|---|---|---|
| 1 | The show is always known; identifying the show is a non-goal | A bounded search over ~20–200 candidates instead of an open-ended one |
| 2 | Reference data comes from API and Wikipedia adapters; **no IMDb scraping** | IMDb's terms prohibit scraping, and it's fragile; IMDb IDs are resolved through TMDB `/find` |
| 3 | Reference subtitles are optional and pluggable | Strongest signal, but needs an account and has quotas; showvenist must work without them |
| 4 | A batch is solved jointly with an "unmatched" option | The at-most-once constraint improves accuracy and flags extras |
| 5 | Python CLI + library | Best ecosystem for media, OCR, speech-to-text and the solvers |
| 6 | Evidence scores (LLRs) are added, with an explicit background hypothesis | Missing signals contribute 0; partial reference-subtitle coverage handled correctly; extras are a real hypothesis |
| 7 | Confidence = margin against the best alternative assignment; auto-include at ≥ 99:1 | Exact, respects constraints, and its meaning is clear |
| 8 | Disc *title* order is ignored; disc *sequence* is chronological | Confirmed for retail media |
| 9 | Show/Season/Disc hierarchy input; disc blocks found from content, not counts | Much smaller search spaces without miscounts cascading |
| 10 | No disc maps, no `--disc`; `--episodes` stays as a manual override | The block model infers disc contents from the content |
| 11 | Season folder is a hard filter + warning | Never silently move a file into a different season |
| 12 | Renames flatten into the season folder; extras to `extras/` | Matches the Jellyfin layout; leftover files stay where they were for you to handle |
| 13 | Marker file created on first run, after confirmation | Show identity travels with the folder |
| 14 | One JSON bundle per show; YAML overrides; SQLite state + a separate disposable cache | Nested data; hand edits with comments; history must survive clearing the cache |
| 15 | History is a soft prior + duplicate warnings; confirmed matches become eval labels | Cross-disc constraints and free labeled data, without wrong past matches causing repeated errors |
| 16 | Image-subtitle OCR in v1 | Retail rips have PGS/VobSub, not text subtitles |
| 17 | Linux only; run on the media server; CPU-only | User's environment; local I/O; no GPU |
| 18 | Readiness detection in v1 | MakeMKV writes straight to the share |
| 19 | Full-screen Textual TUI with pin + re-solve; interactive only | Chosen by the user; the engine stays UI-agnostic and report mode stays scriptable |
| 20 | The primary source decides episodes; secondaries only add data | Avoids numbering fights; conflicts are visible |
| 21 | `naming_order` is separate from the matching order; default TMDB aired (refined by 33) | Filenames must match Jellyfin; the block model needs the release's order |
| 22 | Optional LLM scorer in M3; Sonnet 5.5 default; sent for files below high + the TUI key; ingest-time cues experimental | Closes the dialogue-to-synopsis gap; measured at the gate before it's made a default |
| 23 | Milestones measure accuracy early, with an explicit accuracy gate; the spec owns milestones, Beads owns tasks | The riskiest assumption gets tested before UI work; the two can't drift apart |
| 24 | The M3 gate requires an upper bound on the high-confidence error rate (≤ 1% at 95%) across ≥ 10 seasons of ≥ 5 shows, not a point estimate on 3 seasons | 3 seasons can't show 99%: 60 error-free files only bound the error rate at about 5%. Errors cluster by season, so breadth matters as much as file count |
| 25 | The existing hand-named library is the main eval corpus; manifest labels can be written in any ordering the bundle knows | 422 trusted labels across 25 seasons at no cost; the labels follow series chronology where it differs from air order |
| 26 | The MKV title tag is a second source of disc identity | MakeMKV writes the disc name there, and it survives flattening |
| 27 | Exact duplicates are found by a content key that skips the header; only one copy of a duplicate set is ever moved | Two rips of one title differ only in random header IDs (verified: 105 bytes in 5.5 GB). Moving both would collide on the target name and abort the whole plan (§13.3) |
| 28 | Scorers must not share evidence: names belong to the guest scorer only; the LLM's evidence replaces the synopsis, guest and cues scorers for the files it scores, and it isn't sent the duration; two weight sets, each fitted on the files it's used for | Adding correlated evidence inflates Δ and puts files in "high" that shouldn't be there, which is what the M3 gate measures. One weight can't fit both the easy files and the hard ones `auto` sends |
| 29 | Weights are fitted with a conditional logit (a per-file choice among candidates and "unmatched"), weights only at first, with extras weighted to their real share | It's the solver's own model, so fitted scores match the confidence thresholds. Shape parameters would overfit a few hundred files. A fit without extras learns to over-match |
| 30 | Coverage at high targets ≥ 90%, but precision always comes first | Chosen by the user: a file flagged for review is cheap, while a wrong match can displace other files in the solve |
| 31 | A file left unmatched at high confidence is an identified extra: it moves to `extras/` without review and counts toward coverage | Chosen by the user. Requiring confirmation for every extra would by itself push review past 10% of a raw season. Precision at high includes extras, so the gate guards against episodes sent to `extras/` |
| 32 | Negative reference evidence is scaled by OCR quality; every reference is cross-checked against episode metadata at ingest; the M3 gate runs without reference subtitles, and their accuracy is evaluated in M7 | Bad OCR lowers containment against every reference, which would hand the win to episodes without one or to "extra". A mislabeled reference outvotes every other signal. There are no reference subtitles for the library before the OpenSubtitles adapter |
| 33 | `naming_order` is any ordering in the bundle, set per show; every TMDB episode group is kept with its rank; Jellyfin keywords resolve to the first group of that type, as Jellyfin does; the season folder is read in the ordering being solved; `--order` defaults to `naming_order`; changing season always needs a pin | Episode titles come from TMDB or are entered by hand, so filenames follow either Jellyfin's display order or your own convention. TMDB's aired order puts three Firefly episodes in Specials, which a fixed aired-order season filter would never consider |
| 34 | Files already named as episodes are ground truth: pinned, not analyzed, never renamed or moved automatically | Chosen by the user. The hand-named library is correct, and Jellyfin tracks items by path, so renaming a known file can lose its hand-entered metadata |
| 35 | An M0 feasibility spike comes before M1; what the gate doesn't need (`bundle edit`, `--refresh` diff, show disambiguation, readiness, the concurrent pipeline) moves after it | Without reference subtitles, the main path is untested, and the hand-named library makes it cheap to measure now. If it's weak, the LLM default, the offline goal and the milestone order change, and that should happen before building on them |
| 36 | Where disc folders split a season, the LLM's candidates are one disc's window: the next n_d episodes after the previous disc's last. A flat season folder gets the full season. A failed disc window offers a full-season retry | Chosen by the user. Disc folders and disc order are given by the directory structure, so elimination shrinks each request to a handful of candidates. The retry covers a wrong boundary on the previous disc |
| 37 | M0 runs on four labeled seasons from four shows, two PGS and two VobSub. Its outcome is decided by pooled coverage at high with weights fitted leave-one-season-out, against rules set before the run. The LLM's default mode stays `off` whatever M0 shows, and paid API calls need your OK (at most $1 per season in M0) | With two seasons, track type and show are the same variable, so a weak result couldn't be traced to OCR or to thin data. Default weights are placeholders, so coverage under them can't decide anything. Rules set in advance keep the result from being argued after the fact. Chosen by the user: no spending money without permission, so an API key that happens to be in the environment never turns the LLM on |
