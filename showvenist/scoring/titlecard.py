"""Title-card scorer (spec §9.2, decision 54). Contract; the body is task sv-2k9.

For each file with `card_reads` not None: apply `cfg.pattern` to each read's text, normalize the
`ref` group per `cfg.kind` (roman/number: look-alike characters l, |, 1 -> I where a numeral is
expected), and compare against the candidates' titles or numbers as a closed set (kind "title":
fuzzy ratio >= 90 like the title scorer). Frames combine by consensus: a reading counts when two
or more frames agree or it matches exactly one candidate; readings that disagree, or match
nothing, give 0. Raw LLR is `MATCH_LLR` on the single matched candidate, never negative.
"""

from __future__ import annotations

from showvenist.scoring.types import FileInfo, ScoreContext, ScoreResult

MATCH_LLR = 8.0


class TitleCardScorer:
    name = "titlecard"
    can_be_negative = False

    def score(self, files: list[FileInfo], ctx: ScoreContext) -> ScoreResult:
        # Contract stub: reports unavailable for every file until sv-2k9 lands.
        return ScoreResult.empty(len(files), ctx.n_candidates)
