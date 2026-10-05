"""LLM scorer (spec §10.2, decision 40).

Per file, one structured-output call per candidate order (forward, reversed). Evidence for
candidate j among n shown is logit(p_j) + log(n - 1) with the order-averaged p_j clipped to
[0.01, 0.99]; "none of these" is not evidence (it feeds `disc_window_failed` only). Columns not
shown get the floor (the evidence at p = 0.01), never 0. No network at import: `anthropic` is
imported only inside `default_client_factory`.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Collection
from typing import Any

from pydantic import BaseModel
from rapidfuzz import fuzz

from showvenist.analysis.types import EXTRACTOR_VERSIONS, AnalysisCache, params_hash
from showvenist.scoring.types import Candidate, FileInfo, ScoreContext, ScoreResult

log = logging.getLogger(__name__)

# AnalysisCache.get() indexes EXTRACTOR_VERSIONS; register our extractor if types.py hasn't.
EXTRACTOR_VERSIONS.setdefault("llm", 1)
EXTRACTOR = "llm"

PROMPT_VERSION = "v1"
MAX_TOKENS = 8000
EXCERPT_WORDS = 2200  # ~3k tokens
QUOTE_THRESHOLD = 0.15  # candidates above this must give quotes
QUOTE_MATCH = 85  # rapidfuzz partial_ratio
EPS = 1e-4
P_CLIP = (0.01, 0.99)
EST_OUTPUT_TOKENS = 1500  # dry-run allowance (JSON + low thinking)
CHARS_PER_TOKEN = 4.0

# USD per million tokens (claude-api skill, cached 2026-09-25). The one price table.
PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.40},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
}
DEFAULT_PRICE_MODEL = "claude-sonnet-5-5"  # used for models missing from the table

INSTRUCTIONS = f"""You identify which episode of a TV show a video file is, from an OCR'd dialogue excerpt.
You get the candidate episodes (id, title, synopsis, guest characters) and then the excerpt of the file's subtitles. The excerpt is sampled from several points in the episode and is imperfect OCR.
Rate every candidate with a probability that the file is that episode, plus a probability for "none of these" (the file is not any listed episode). Probabilities over all candidates plus none must sum to 1.
For every candidate whose probability is above {QUOTE_THRESHOLD}, give 1 to 3 short lines quoted EXACTLY (verbatim) from the excerpt that support it. Never quote from memory; a quote not in the excerpt voids the candidate. Leave quotes empty for other candidates.
Use the synopses, guest characters and anything you know of the show, but base the rating on the excerpt."""  # noqa: E501


class CandidateScore(BaseModel):
    episode: str
    probability: float
    quotes: list[str]


class LLMResult(BaseModel):
    scores: list[CandidateScore]
    none_probability: float


# ---------- pure helpers ----------


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def evidence(p: float, n: int) -> float:
    """Decision 40: logit(clip(p)) + log(n - 1)."""
    p = min(max(p, P_CLIP[0]), P_CLIP[1])
    return logit(p) + math.log(max(n - 1, 1))


def floor_evidence(n: int) -> float:
    return evidence(P_CLIP[0], n)


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def cost_usd(usage: dict, model: str) -> float:
    p = PRICES.get(model) or PRICES[DEFAULT_PRICE_MODEL]
    return (
        usage.get("input_tokens", 0) * p["input"]
        + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
        + usage.get("output_tokens", 0) * p["output"]
    ) / 1e6


def usage_dict(u: Any) -> dict:
    return {
        k: int(getattr(u, k, 0) or 0)
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    }


def estimate_call_cost(prefix_toks: int, excerpt_toks: int, warm: bool, model: str) -> float:
    p = PRICES.get(model) or PRICES[DEFAULT_PRICE_MODEL]
    pre = (prefix_toks + est_tokens(INSTRUCTIONS)) * (p["cache_read"] if warm else p["cache_write"])
    return (pre + excerpt_toks * p["input"] + EST_OUTPUT_TOKENS * p["output"]) / 1e6


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def quote_found(quote: str, excerpt_norm: str) -> bool:
    q = _norm(quote)
    return bool(q) and fuzz.partial_ratio(q, excerpt_norm) >= QUOTE_MATCH


def build_excerpt(file: FileInfo, max_words: int = EXCERPT_WORDS) -> str:
    """Equal word budget per window so every file gets a same-size excerpt."""
    d = file.dialogue
    if d is None or d.track is None:
        return ""
    wins = [
        " ".join(" ".join(c.text.split()) for c in w.cues if c.text.strip())
        for w in d.windows
        if w.cues
    ]
    wins = [w for w in wins if w]
    if not wins:
        return ""
    per = max(1, max_words // len(wins))
    return "\n[...]\n".join(" ".join(w.split()[:per]) for w in wins)


def candidate_block(ctx: ScoreContext, cands: list[Candidate], reverse: bool = False) -> str:
    cs = cands[::-1] if reverse else cands
    parts = [f"Show: {ctx.bundle.show.title}", "Candidate episodes:"]
    for c in cs:
        ep = c.episode
        syn = " ".join(s.text for s in ep.synopses if s.text) or "(none)"
        chars = [g.character for g in ep.guest_cast if g.character]
        parts.append(
            f"[{c.label}] {ep.title}\nSynopsis: {syn}\n"
            f"Guest characters: {', '.join(chars) if chars else '(none)'}"
        )
    return "\n\n".join(parts)


def build_request(model: str, effort: str, block: str, excerpt: str) -> dict:
    return dict(
        model=model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        output_format=LLMResult,
        system=INSTRUCTIONS,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": block, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "Dialogue excerpt from the file:\n\n" + excerpt},
                ],
            }
        ],
    )


def renormalize(parsed: dict, ids: list[str]) -> tuple[dict[str, float], float]:
    """Unknown ids dropped, missing ids get EPS, floors applied, sum to 1."""
    p = {i: EPS for i in ids}
    for s in parsed.get("scores", []):
        if s.get("episode") in p:
            p[s["episode"]] = max(EPS, float(s["probability"]))
    pn = max(EPS, float(parsed.get("none_probability", 0.0)))
    tot = sum(p.values()) + pn
    return {i: v / tot for i, v in p.items()}, pn / tot


def default_client_factory() -> Any:
    import anthropic  # optional extra; imported lazily so tests and offline runs never need it

    return anthropic.Anthropic()


# ---------- scorer ----------


class LlmScorer:
    name = "llm"
    can_be_negative = True

    def __init__(
        self,
        client: Any = None,
        *,
        client_factory: Callable[[], Any] | None = None,
        model: str = "claude-sonnet-5-5",
        effort: str = "low",
        send_both_orders: bool = True,
        cache: AnalysisCache | None = None,
        send: Collection[int] | None = None,
        shown: Collection[int] | None = None,
    ):
        self._client = client
        self._factory = client_factory or default_client_factory
        self.model, self.effort = model, effort
        self.send_both_orders = send_both_orders
        self.cache = cache
        self.send = None if send is None else set(send)
        self.shown = None if shown is None else set(shown)
        self.total_cost_usd = 0.0  # new (non-cached) API spend, accumulated over score() calls
        self.file_info: dict[int, dict] = {}  # file index -> per-order details

    # -- orders / candidates
    def _orders(self) -> list[tuple[str, bool]]:
        return [("forward", False), ("reversed", True)] if self.send_both_orders else [
            ("forward", False)
        ]

    def _shown(self, ctx: ScoreContext) -> list[Candidate]:
        if self.shown is None:
            return list(ctx.candidates)
        return [c for c in ctx.candidates if c.index in self.shown]

    def _client_obj(self) -> Any:
        if self._client is None:
            self._client = self._factory()
        return self._client

    # -- cache
    def _key(self, block: str, excerpt: str) -> tuple[str, str]:
        import hashlib

        def h(s: str) -> str:
            return hashlib.sha256(s.encode()).hexdigest()

        params = params_hash(
            model=self.model, prompt=PROMPT_VERSION, effort=self.effort, block=h(block)
        )
        return h(excerpt), params

    def _cache_get(self, block: str, excerpt: str) -> dict | None:
        if self.cache is None:
            return None
        fp, params = self._key(block, excerpt)
        raw = self.cache.get(fp, EXTRACTOR, params)
        return json.loads(raw) if raw else None

    def _cache_put(self, block: str, excerpt: str, rec: dict) -> None:
        if self.cache is not None:
            fp, params = self._key(block, excerpt)
            self.cache.put(fp, EXTRACTOR, params, json.dumps(rec))

    def _call(self, block: str, excerpt: str) -> dict | None:
        """Cached record {parsed, stop_reason, usage, cached}; None on a failed call."""
        hit = self._cache_get(block, excerpt)
        if hit is not None:
            hit["cached"] = True
            return hit
        try:
            resp = self._client_obj().messages.parse(
                **build_request(self.model, self.effort, block, excerpt)
            )
        except Exception as e:  # network, rate limit, API error: file stays in the base regime
            log.warning("llm call failed: %s", e)
            return None
        parsed = getattr(resp, "parsed_output", None)
        if isinstance(parsed, BaseModel):
            parsed = parsed.model_dump()
        rec = {
            "parsed": parsed,
            "stop_reason": getattr(resp, "stop_reason", None),
            "usage": usage_dict(resp.usage),
            "cached": False,
        }
        self._cache_put(block, excerpt, rec)
        return rec

    # -- scoring
    def _score_order(
        self, parsed: dict, cands: list[Candidate], excerpt: str
    ) -> dict:
        ids = [c.label for c in cands]
        n = len(ids)
        probs, pn = renormalize(parsed, ids)
        exn = _norm(excerpt)
        quotes_by = {s.get("episode"): s.get("quotes") or [] for s in parsed.get("scores", [])}
        verified, failed, p_eff = {}, [], {}
        for i in ids:
            v = [q for q in quotes_by.get(i, []) if quote_found(q, exn)]
            verified[i] = v
            p = probs[i]
            if p > QUOTE_THRESHOLD and not v:
                failed.append(i)
                p = min(p, 1.0 / n)  # evidence exactly 0: positive evidence dropped
            p_eff[i] = p
        return {"p": p_eff, "p_raw": probs, "p_none": pn, "quotes": verified, "failed": failed}

    def _score_file(self, fi: int, f: FileInfo, ctx: ScoreContext, cands: list[Candidate]):
        excerpt = build_excerpt(f)
        if not excerpt or not cands:
            return None
        results: dict[str, dict] = {}
        cost_new = 0.0
        usage_total: dict[str, int] = {}
        for oname, rev in self._orders():
            block = candidate_block(ctx, cands, rev)
            rec = self._call(block, excerpt)
            if rec is None:
                continue
            c = cost_usd(rec["usage"], self.model)
            if not rec["cached"]:
                cost_new += c
            for k, v in rec["usage"].items():
                usage_total[k] = usage_total.get(k, 0) + v
            if rec["stop_reason"] == "refusal" or not rec["parsed"]:
                log.warning("llm refusal/empty for %s (%s order)", f.path, oname)
                continue
            results[oname] = self._score_order(rec["parsed"], cands, excerpt)
        self.total_cost_usd += cost_new
        self.file_info[fi] = {
            "orders": sorted(results),
            "expected_orders": [o for o, _ in self._orders()],
            "p_none": {o: r["p_none"] for o, r in results.items()},
            "none_top": {
                o: r["p_none"] > max(r["p_raw"].values()) for o, r in results.items()
            },
            "cost_usd": cost_new,
            "usage": usage_total,
        }
        return results

    def score(self, files: list[FileInfo], ctx: ScoreContext) -> ScoreResult:
        E = len(ctx.candidates)
        out = ScoreResult.empty(len(files), E)
        cands = self._shown(ctx)
        n = len(cands)
        if n == 0:
            return out
        pos = {c.index: k for k, c in enumerate(ctx.candidates)}
        floor = floor_evidence(n)
        for i, f in enumerate(files):
            if self.send is not None and i not in self.send:
                continue
            results = self._score_file(i, f, ctx, cands)
            if not results:
                continue
            out.available[i] = True
            out.llr[i, :] = floor  # columns not shown
            for c in cands:
                label = c.label
                by_order = {o: r["p"][label] for o, r in results.items()}
                p = sum(by_order.values()) / len(by_order)
                col = pos[c.index]
                out.llr[i, col] = evidence(p, n)
                out.notes[(i, col)] = {
                    "p": p,
                    "p_by_order": by_order,
                    "p_raw_by_order": {o: r["p_raw"][label] for o, r in results.items()},
                    "p_none_by_order": {o: r["p_none"] for o, r in results.items()},
                    "quotes": [q for r in results.values() for q in r["quotes"][label]],
                    "quote_failed": any(label in r["failed"] for r in results.values()),
                }
        return out

    def disc_window_failed(self, i: int) -> bool:
        """'None of these' rated highest in EVERY expected order (both when send_both_orders)."""
        info = self.file_info.get(i)
        if not info or set(info["orders"]) != set(info["expected_orders"]):
            return False
        return all(info["none_top"][o] for o in info["orders"])

    # -- dry run
    def estimate_cost(self, files: list[FileInfo], ctx: ScoreContext) -> float:
        """Rough USD for the calls `score()` would make (cache hits are free). No API."""
        cands = self._shown(ctx)
        if not cands:
            return 0.0
        total, warm = 0.0, {}
        blocks = {o: candidate_block(ctx, cands, rev) for o, rev in self._orders()}
        for i, f in enumerate(files):
            if self.send is not None and i not in self.send:
                continue
            ex = build_excerpt(f)
            if not ex:
                continue
            for o, b in blocks.items():
                if self._cache_get(b, ex) is not None:
                    continue
                total += estimate_call_cost(
                    est_tokens(b), est_tokens(ex), warm.get(o, False), self.model
                )
                warm[o] = True
        return total

