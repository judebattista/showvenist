"""Cast name tokens for the synopsis and cues scorers (spec §9.2), mirroring the M0 spike's
`removable_names` (spike/score.py) so scorer results match the spike's.

Stripped from BM25 query and documents:
- every token of a SERIES (main) cast actor or character name, always, including dictionary words
  (the main cast is named in nearly every episode, so it carries no discriminating signal);
- tokens of guest actor and guest character names, but only the non-dictionary ones, so a guest
  character called "Doctor" or "Nurse" keeps its ordinary meaning in the synopsis text.
Directors and writers are not stripped (the spike did not).
"""

from __future__ import annotations

from showvenist.bundle.schema import Bundle
from showvenist.scoring.text import is_dictionary_word, name_tokens


def series_cast_tokens(bundle: Bundle) -> set[str]:
    """Every token of the series (main) cast actor and character names."""
    out: set[str] = set()
    for c in bundle.series_cast:
        out.update(name_tokens(c.actor))
        out.update(name_tokens(c.character))
    return out


def cast_names(bundle: Bundle, language: str = "en") -> set[str]:
    series = series_cast_tokens(bundle)
    allnames = set(series)
    for ep in bundle.episodes:
        for c in ep.guest_cast:
            allnames.update(name_tokens(c.actor))
            allnames.update(name_tokens(c.character))
    return series | {t for t in allnames if not is_dictionary_word(t, language)}
