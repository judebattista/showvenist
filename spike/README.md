# M0 feasibility spike

Throwaway code (spec §17.2, plan §2). Not part of the package. Each script is a
PEP 723 single-file script: `uv run spike/<script>.py`. Output goes to
`spike/out/` (gitignored). Runs on eweb; the library at `/mnt/storage/tv` is read-only.

## Seasons

| Key | Path | Files | Track |
|---|---|---|---|
| supernatural-s02 | /mnt/storage/tv/Supernatural/Season02 | 22 (skip `…a.mkv` duplicates) | PGS |
| the-incredible-hulk-s02 | /mnt/storage/tv/The_Incredible_Hulk/Season_02 | 22 | VobSub (expected) |
| the-west-wing-s01 | /mnt/storage/tv/The_West_Wing/Season_01 | 22 | unknown |
| fourth | picked after `extract.py --probe-only` | ~22 | to make two PGS + two VobSub |

Labels come from file names (aired order).

## Stage contracts

| Stage | Output |
|---|---|
| `extract.py` | `out/dialogue/<show>-s<NN>/<basename>.json`: path, size, duration, chosen track (codec, language, events), windows (start, end, lines of {t, text}), ocr_quality, caption_lines (closed-caption text track, if any). `--probe-only` lists each file's tracks. |
| `tmdb.py` | `out/tmdb/<show>-s<NN>.json`: episodes (id, season, episode, title, overview, guest_stars [{actor, character}], writers, directors) and series_cast (names to strip). Needs the TMDB token (`SHOWVENIST_TMDB_API_KEY` or config.toml). |
| `labels.py` | `out/labels.csv`: path, show, season, label (`SxxEyy`). |
| `score.py` | `out/results/<show>-s<NN>.json`, `out/results/pooled.json`, printed summary. |
