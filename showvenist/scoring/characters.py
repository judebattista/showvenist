"""Guest character scorer (spec §9.2): IDF-weighted distinct guest character names heard."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from showvenist.scoring.text import (
    dialogue_text,
    has_dialogue,
    is_dictionary_word,
    name_parts,
    name_tokens,
    tokens,
    window_text,
)
from showvenist.scoring.types import FileInfo, ScoreContext, ScoreResult


class GuestScorer:
    name = "guests"
    can_be_negative = False

    def score(self, files: list[FileInfo], ctx: ScoreContext) -> ScoreResult:
        F, E = len(files), ctx.n_candidates
        res = ScoreResult.empty(F, E)
        lang = ctx.language
        per_ep: list[set[str]] = []
        for c in ctx.candidates:
            toks: set[str] = set()
            for g in c.episode.guest_cast:
                for part in name_parts(g.character):
                    toks.update(t for t in name_tokens(part) if not is_dictionary_word(t, lang))
            per_ep.append(toks)
        df = Counter(t for s in per_ep for t in s)
        idf = {t: math.log(E / n) for t, n in df.items()}

        def llr(heard: set[str]) -> np.ndarray:
            return np.array([sum(idf[t] for t in s if t in heard) for s in per_ep], float)

        n_win = max((len(f.dialogue.windows) for f in files if has_dialogue(f.dialogue)), default=0)
        if n_win:
            res.window_llr = np.zeros((F, n_win, E))
        for i, f in enumerate(files):
            if not has_dialogue(f.dialogue):
                continue
            heard = set(tokens(dialogue_text(f.dialogue)))
            if not heard:
                continue
            res.available[i] = True
            res.llr[i] = llr(heard)
            for w, win in enumerate(f.dialogue.windows):
                res.window_llr[i, w] = llr(set(tokens(window_text(win))))
            for j in range(E):
                if res.llr[i, j] > 0:
                    res.notes[(i, j)] = {"names": sorted(per_ep[j] & heard)}
        return res
