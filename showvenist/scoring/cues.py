"""Dialogue cues scorer (spec §9.2, §10.3): BM25 of dialogue against each episode's llm_cues."""

from __future__ import annotations

from showvenist.scoring.names import cast_names
from showvenist.scoring.synopsis import score_bm25_files
from showvenist.scoring.text import content_tokens
from showvenist.scoring.types import FileInfo, ScoreContext, ScoreResult


class CuesScorer:
    name = "cues"
    can_be_negative = False

    def score(self, files: list[FileInfo], ctx: ScoreContext) -> ScoreResult:
        if not any(c.episode.llm_cues for c in ctx.candidates):
            return ScoreResult.empty(len(files), ctx.n_candidates)
        strip = cast_names(ctx.bundle, ctx.language)  # cast/character names dropped
        docs = []
        for c in ctx.candidates:
            cues = c.episode.llm_cues
            text = " ".join([*cues.names, *cues.terms, *cues.lines]) if cues else ""
            docs.append(content_tokens(text, strip, ctx.language))
        return score_bm25_files(files, ctx, docs, strip)
