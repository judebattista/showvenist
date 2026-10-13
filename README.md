# showvenist

Identifies the season and episode of TV disc rips against a reference bundle for the show.

## Installation

Install with `uv`:

```bash
uv sync
```

## Requirements

The external binary `tesseract` is required for OCR functionality. Install it on your system:

- **Ubuntu/Debian:** `apt install tesseract-ocr`
- **Fedora:** `dnf install tesseract`
- **macOS:** `brew install tesseract`

## Usage

```bash
showvenist ingest "Show Name" -o data/bundles          # build the reference bundle (TMDB, then TVmaze)
showvenist ingest SLUG --refresh --diff -o data/bundles # preview a refresh; --refresh --yes applies it
showvenist identify DIR --plan                          # report and plan; writes nothing
showvenist identify DIR --apply --confirm TOKEN         # run the plan (or a typed yes on a terminal)
showvenist assign FILE S01E11+S01E12                    # confirm a file by hand (two-episode files too)
showvenist undo --last                                  # reverse the last run
showvenist eval MANIFEST --library DIR                  # accuracy over a labeled manifest
```

TMDB needs `SHOWVENIST_TMDB_API_KEY` (exported) or a token in the config file. The LLM scorer is off unless
you pass `--llm` and a `--budget`. See `docs/spec.md` for the design and `docs/checkpoint.md` for the current state.

## Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.
