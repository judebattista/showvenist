"""Synopsis scorer (spec §9.2): BM25 of the file's dialogue against episode title + synopses."""

from __future__ import annotations

import numpy as np

from showvenist.scoring.names import cast_names
from showvenist.scoring.text import (
    Bm25,
    bm25_llr,
    content_tokens,
    dialogue_text,
    has_dialogue,
    window_text,
)
from showvenist.scoring.types import FileInfo, ScoreContext, ScoreResult


def score_bm25_files(
    files: list[FileInfo], ctx: ScoreContext, docs: list[list[str]], strip: set[str]
) -> ScoreResult:
    """Shared by the synopsis and cues scorers: BM25 + chance-max credit per file and window."""
    F, E = len(files), ctx.n_candidates
    res = ScoreResult.empty(F, E)
    bm = Bm25(docs)
    lang = ctx.language
    n_win = max((len(f.dialogue.windows) for f in files if has_dialogue(f.dialogue)), default=0)
    if n_win:
        res.window_llr = np.zeros((F, n_win, E))
    for i, f in enumerate(files):
        if not has_dialogue(f.dialogue):
            continue
        q = content_tokens(dialogue_text(f.dialogue), strip, lang)
        if not q:
            continue
        res.available[i] = True
        res.llr[i] = bm25_llr(bm, q)
        for w, win in enumerate(f.dialogue.windows):
            wq = content_tokens(window_text(win), strip, lang)
            if wq:
                res.window_llr[i, w] = bm25_llr(bm, wq)
        for j in range(E):
            if res.llr[i, j] > 0:
                res.notes[(i, j)] = {"terms": bm.top_terms(q, j)}
    return res


class SynopsisScorer:
    name = "synopsis"
    can_be_negative = False

    def score(self, files: list[FileInfo], ctx: ScoreContext) -> ScoreResult:
        strip = cast_names(ctx.bundle, ctx.language)
        docs = [
            content_tokens(
                " ".join([c.episode.title, *(s.text for s in c.episode.synopses)]),
                strip,
                ctx.language,
            )
            for c in ctx.candidates
        ]
        return score_bm25_files(files, ctx, docs, strip)
