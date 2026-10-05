"""Cast and crew name tokens, shared so names are scored by the guest scorer only (spec §9.2)."""

from __future__ import annotations

from showvenist.bundle.schema import Bundle
from showvenist.scoring.text import name_tokens


def cast_names(bundle: Bundle) -> set[str]:
    """Normalized tokens of every guest actor, guest character, director and writer name in the
    bundle. The synopsis and cues scorers strip these from query and documents."""
    out: set[str] = set()
    for ep in bundle.episodes:
        for c in ep.guest_cast:
            out.update(name_tokens(c.actor))
            out.update(name_tokens(c.character))
        for n in (*ep.directors, *ep.writers):
            out.update(name_tokens(n))
    return out
