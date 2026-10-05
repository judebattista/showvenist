# M0 results

Run 2026-10-05 on eweb. Four labeled seasons, 89 files: Supernatural S2 (22, PGS), The Incredible Hulk S2 (22, PGS), The West Wing S1 (22, PGS), The Invisible Man S1 (23, VobSub). The plan's two-and-two track split wasn't available: the library has one labeled VobSub season (Hulk S2 is PGS). Decision: run the four we have.

## OCR

Quality (share of words in the English vocabulary) is 0.99–1.00 on every file checked, PGS and VobSub alike; hand-read samples are nearly error-free (rare dropped apostrophes, merged words). About 130–175 lines and 800–1,200 words per file from three sampled windows; 4–15 s per file. The metric barely varies, so it can't yet tell good OCR from bad (needs a show-specific vocabulary). No text/caption tracks existed on these files, so the caption-vs-OCR comparison wasn't possible. OCR is not the bottleneck.

## Offline (synopsis BM25 + guest names), no LLM

Weights fitted leave-one-season-out (in-sample fit is identical to two decimals: [synopsis 2.54, guest 1.35]).

| Season | Top-1 | Coverage at Δ≥4.6 | Errors at high |
|---|---|---|---|
| Supernatural S2 | 9% | 0% | 0 |
| Hulk S2 | 32% | 5% | 0 |
| West Wing S1 | 27% | 9% | 0 |
| Invisible Man S1 | 30% | 9% | 0 |
| **Pooled** | **25%** | **6%** | **0** |

Default weights: 16% top-1, 0% coverage. Per-scorer rank-1 of the true episode: synopsis 31% (MRR 0.40); guest names about 1%.

Cause: thin reference data. TMDB gives about 0.5 usable guest-name tokens per episode that appear in the sampled dialogue (Supernatural S2: 11 tokens in total, 2 heard; West Wing S1: 13, 4 heard); synopses are one-liners. This is a data finding, not a scorer bug (checked directly).

## With the LLM (Sonnet 5.5, both candidate orders, full-season candidates)

The model picks the right episode in **89 of 89** files (no refusals). Cost: $0.27–0.53 per season, **$1.42 total** (ceiling was $1 per season).

LLM regime (LLM evidence only, no other scorer), weight fitted leave-one-season-out (w≈1.15–1.5):

| | Top-1 (after solve) | Coverage at Δ≥4.6 | Errors at high |
|---|---|---|---|
| Pooled, λ=1.0 | 89% | 3% | 3 |
| Pooled, λ=0.1 | 89% | 8% | 3 (of 7 high) |

Why it fails the bar although the ranking is perfect:

1. **Ceiling.** Log-odds is log(p_j / p_none) and the model's "none of these" floor is about 0.025, so the best per-file log-odds has a median of 3.7 and a maximum of 4.6. With the −1.4 prior, Δ≥4.6 needs about 5. No file reaches it at a sensible weight.
2. **"None of these" answers.** In 10 files the true episode got log-odds below zero (down to −3.6), and the solver leaves those unmatched. Three Invisible Man files do so at high confidence: these are the 3 errors at high (identified extras that are really episodes). Plan check 1.

## Decision per the rules set before the run (plan §2.5, spec §17.2)

- Check 1 (errors at high): present only in the LLM regime, cause above (the LLM's calibration of "none of these").
- Check 2 (VobSub vs PGS): not triggered; OCR is clean on both.
- Check 3 (a season well below the others): Supernatural S2 offline (0–9%), cause thin synopses.
- Outcome: pooled coverage is 6% offline and 3–8% with the LLM, both below 90%. By the rules this is **outcome 3: signal too weak** (reference subtitles ahead of the M3 gate; §1.2 revisited).

Caveats that bear on how the user reads it (not re-arguing the rule): the LLM's ranking is not weak, its confidence scale is; M0 had no duration scorer, no title scorer, no block model, and no extras in the data. All of these add or sharpen evidence in the real pipeline.
