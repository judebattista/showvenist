#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic", "pydantic", "rapidfuzz"]
# ///
"""M0 spike LLM scorer (optional). Spec 10.1, 10.2; plan 2.3 row 0.5, 2.4.

Per file: one structured-output call per candidate order (forward, reversed),
candidate scope = FULL SEASON. Output: raw (unclamped) log-odds
log p_j - log p_none per episode, verified quotes, refusal flag, usage, cost.

Default is --dry-run (no API call, projected cost only). Real calls need --run.
Usage: uv run spike/llm.py [--run] [--ceiling 1.0] [--only KEY] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from pydantic import BaseModel
from rapidfuzz import fuzz

MODEL = "claude-sonnet-5-5"
PROMPT_VERSION = "v1"
EFFORT = "low"
MAX_TOKENS = 8000

# Prices, USD per million tokens (claude-api skill, cached 2026-09-25). One place.
PRICES = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
}

EXCERPT_WORDS = 2200          # ~3k tokens
QUOTE_THRESHOLD = 0.15        # candidates above this must give quotes
QUOTE_MATCH = 85              # rapidfuzz partial_ratio
EPS = 1e-4
EST_OUTPUT_TOKENS = 1500      # dry-run / ceiling allowance (JSON + low thinking)
CHARS_PER_TOKEN = 4.0

INSTRUCTIONS = f"""You identify which episode of a TV show a video file is, from an OCR'd dialogue excerpt.
You get the candidate episodes (id, title, synopsis, guest characters) and then the excerpt of the file's subtitles. The excerpt is sampled from several points in the episode and is imperfect OCR.
Rate every candidate with a probability that the file is that episode, plus a probability for "none of these" (the file is not any listed episode). Probabilities over all candidates plus none must sum to 1.
For every candidate whose probability is above {QUOTE_THRESHOLD}, give 1 to 3 short lines quoted EXACTLY (verbatim) from the excerpt that support it. Never quote from memory; a quote not in the excerpt voids the candidate. Leave quotes empty for other candidates.
Use the synopses, guest characters and anything you know of the show, but base the rating on the excerpt."""


class CandidateScore(BaseModel):
    episode: str
    probability: float
    quotes: list[str]


class LLMResult(BaseModel):
    scores: list[CandidateScore]
    none_probability: float


class CeilingExceeded(RuntimeError):
    pass


# ---------- prompt building ----------

def candidate_ids(tmdb: dict) -> list[str]:
    return [f"S{e['season']:02d}E{e['episode']:02d}" for e in tmdb["episodes"]]


def candidate_block(tmdb: dict, reverse: bool = False) -> str:
    eps = list(tmdb["episodes"])
    if reverse:
        eps = eps[::-1]
    parts = [f"Show: {tmdb.get('show_title', '')}", "Candidate episodes:"]
    for e in eps:
        chars = [g.get("character") for g in e.get("guest_stars", []) if g.get("character")]
        parts.append(
            f"[S{e['season']:02d}E{e['episode']:02d}] {e.get('title', '')}\n"
            f"Synopsis: {e.get('overview') or '(none)'}\n"
            f"Guest characters: {', '.join(chars) if chars else '(none)'}"
        )
    return "\n\n".join(parts)


def build_excerpt(dialogue: dict, max_words: int = EXCERPT_WORDS) -> str:
    """Same-size excerpt for every file: equal word budget per window, truncated by words."""
    wins = []
    for w in dialogue.get("windows", []):
        lines = [" ".join(l["text"].split()) for l in w.get("lines", []) if l.get("norm", l["text"]).strip()]
        if lines:
            wins.append(lines)
    if not wins:
        return ""
    per = max(1, max_words // len(wins))
    out = []
    for lines in wins:
        words = " ".join(lines).split()[:per]
        out.append(" ".join(words))
    return "\n[...]\n".join(out)


def build_request(model: str, block: str, excerpt: str) -> dict:
    return dict(
        model=model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": EFFORT},
        output_format=LLMResult,
        system=INSTRUCTIONS,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": block, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Dialogue excerpt from the file:\n\n" + excerpt},
            ],
        }],
    )


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


# ---------- cost ----------

def cost_usd(usage: dict, model: str = MODEL) -> float:
    p = PRICES[model]
    return (
        usage.get("input_tokens", 0) * p["input"]
        + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]
        + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
        + usage.get("output_tokens", 0) * p["output"]
    ) / 1e6


def usage_dict(u) -> dict:
    return {k: int(getattr(u, k, 0) or 0) for k in
            ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}


def add_usage(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def estimate_call_cost(block_toks: int, excerpt_toks: int, warm: bool, model: str = MODEL) -> float:
    p = PRICES[model]
    prefix = block_toks + est_tokens(INSTRUCTIONS)
    pre = prefix * (p["cache_read"] if warm else p["cache_write"])
    return (pre + excerpt_toks * p["input"] + EST_OUTPUT_TOKENS * p["output"]) / 1e6


# ---------- cache ----------

def cache_key(model: str, block: str, excerpt: str, effort: str = EFFORT) -> str:
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    return hashlib.sha256("|".join([model, PROMPT_VERSION, effort, h(block), h(excerpt)]).encode()).hexdigest()


def cache_get(cache_dir: Path, key: str):
    p = cache_dir / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cache_put(cache_dir: Path, key: str, rec: dict):
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(rec))


# ---------- scoring ----------

def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def quote_found(quote: str, excerpt_norm: str) -> bool:
    q = _norm(quote)
    if not q:
        return False
    return fuzz.partial_ratio(q, excerpt_norm) >= QUOTE_MATCH


def renormalize(parsed: dict, ids: list[str]) -> tuple[dict, float, bool]:
    """Return ({id: p}, p_none, valid). Unknown ids dropped, missing ids get EPS, floors, sum to 1."""
    p = {i: EPS for i in ids}
    for s in parsed["scores"]:
        if s["episode"] in p:
            p[s["episode"]] = max(EPS, float(s["probability"]))
    pn = max(EPS, float(parsed["none_probability"]))
    tot = sum(p.values()) + pn
    return {i: v / tot for i, v in p.items()}, pn / tot, True


def score_order(parsed: dict, ids: list[str], excerpt: str) -> dict:
    """Per-order result: log-odds, verified quotes, quote_failed ids."""
    probs, pn, _ = renormalize(parsed, ids)
    exn = _norm(excerpt)
    quotes_by = {s["episode"]: s.get("quotes") or [] for s in parsed["scores"]}
    lo, verified, failed = {}, {}, []
    for i in ids:
        v = [q for q in quotes_by.get(i, []) if quote_found(q, exn)]
        verified[i] = v
        x = math.log(probs[i]) - math.log(pn)
        if probs[i] > QUOTE_THRESHOLD and not v:
            failed.append(i)
            x = min(x, 0.0)  # drop positive evidence
        lo[i] = x
    return {"logodds": lo, "quotes": verified, "quote_failed": failed, "p": probs, "p_none": pn}


def call_one(client, cache_dir: Path, block: str, excerpt: str, model: str = MODEL) -> dict:
    """Return {'parsed','stop_reason','usage','cached'}; caches every response."""
    key = cache_key(model, block, excerpt)
    hit = cache_get(cache_dir, key)
    if hit:
        hit["cached"] = True
        return hit
    resp = client.messages.parse(**build_request(model, block, excerpt))
    parsed = getattr(resp, "parsed_output", None)
    rec = {
        "parsed": parsed.model_dump() if parsed is not None else None,
        "stop_reason": resp.stop_reason,
        "usage": usage_dict(resp.usage),
        "cached": False,
    }
    cache_put(cache_dir, key, rec)
    return rec


class Budget:
    def __init__(self, ceiling: float):
        self.ceiling, self.spent = ceiling, 0.0

    def check(self, est: float):
        if self.spent + est > self.ceiling:
            raise CeilingExceeded(f"spent ${self.spent:.3f} + next ~${est:.3f} > ceiling ${self.ceiling:.2f}")


def score_file(client, cache_dir, tmdb, dialogue, budget: Budget, warm: dict, model=MODEL) -> dict:
    ids = candidate_ids(tmdb)
    excerpt = build_excerpt(dialogue)
    orders, usage_total, cost_new = [], {}, 0.0
    for name, rev in (("forward", False), ("reversed", True)):
        block = candidate_block(tmdb, rev)
        key = cache_key(model, block, excerpt)
        if client is not None and cache_get(cache_dir, key) is None:
            budget.check(estimate_call_cost(est_tokens(block), est_tokens(excerpt), warm.get(name, False), model))
        rec = call_one(client, cache_dir, block, excerpt, model)
        warm[name] = True
        c = cost_usd(rec["usage"], model)
        if not rec["cached"]:
            budget.spent += c
            cost_new += c
        usage_total = add_usage(usage_total, rec["usage"])
        refused = rec["stop_reason"] == "refusal" or rec["parsed"] is None
        o = {"order": name, "stop_reason": rec["stop_reason"], "refusal": refused,
             "cached": rec["cached"], "usage": rec["usage"], "cost_usd": c}
        if not refused:
            o["result"] = score_order(rec["parsed"], ids, excerpt)
        orders.append(o)
    good = [o["result"] for o in orders if "result" in o]
    out = {
        "refusal": any(o["refusal"] for o in orders),
        "no_evidence": not good,
        "orders": [{k: v for k, v in o.items() if k != "result"} | ({"p_none": o["result"]["p_none"]} if "result" in o else {}) for o in orders],
        "episodes": {},
        "usage": usage_total,
        "cost_usd": sum(o["cost_usd"] for o in orders),
        "cost_new_usd": cost_new,
    }
    if good:
        for i in ids:
            qs = []
            for r in good:
                qs += [q for q in r["quotes"][i] if q not in qs]
            out["episodes"][i] = {
                "logodds": sum(r["logodds"][i] for r in good) / len(good),
                "p": sum(r["p"][i] for r in good) / len(good),
                "quotes": qs,
                "quote_failed": any(i in r["quote_failed"] for r in good),
            }
        out["p_none"] = sum(r["p_none"] for r in good) / len(good)
    return out


# ---------- driver ----------

def load_truth(labels_csv: Path) -> dict:
    t = {}
    if labels_csv.exists():
        for r in csv.DictReader(labels_csv.open()):
            t[r["path"]] = r["label"]
    return t


def run(out_dir: Path, client, ceiling: float, only=None, limit=None, dry=True, model=MODEL) -> dict:
    truth = load_truth(out_dir / "labels.csv")
    cache_dir = out_dir / "llm_cache"
    totals = {}
    for sdir in sorted((out_dir / "dialogue").iterdir()):
        key = sdir.name
        if only and key != only:
            continue
        tmdb_p = out_dir / "tmdb" / f"{key}.json"
        if not tmdb_p.exists():
            print(f"{key}: no tmdb json, skipped", file=sys.stderr)
            continue
        tmdb = json.loads(tmdb_p.read_text())
        files = sorted(sdir.glob("*.json"))[:limit]
        if dry:
            totals[key] = dry_season(tmdb, files, cache_dir, model)
            continue
        budget, warm, results = Budget(ceiling), {}, {}
        aborted = None
        for f in files:
            d = json.loads(f.read_text())
            try:
                r = score_file(client, cache_dir, tmdb, d, budget, warm, model)
            except CeilingExceeded as e:
                aborted = str(e)
                print(f"{key}: ABORT {e}", file=sys.stderr)
                break
            r["path"], r["basename"], r["truth"] = d.get("path"), d.get("basename", f.stem), truth.get(d.get("path"))
            results[f.stem] = r
        outp = out_dir / "llm" / f"{key}.json"
        outp.parent.mkdir(parents=True, exist_ok=True)
        tot_all = sum(r["cost_usd"] for r in results.values())
        outp.write_text(json.dumps({
            "key": key, "model": model, "prompt_version": PROMPT_VERSION, "effort": EFFORT,
            "scope": "season", "files": results,
            "cost_new_usd": budget.spent, "cost_total_usd": tot_all, "aborted": aborted,
        }, indent=1))
        print(f"{key}: files={len(results)} new_cost=${budget.spent:.3f} total_incl_cached=${tot_all:.3f}"
              f"{' ABORTED' if aborted else ''}")
        totals[key] = budget.spent
    return totals


def dry_season(tmdb, files, cache_dir, model=MODEL) -> float:
    total, warm, n_cached = 0.0, {}, 0
    blocks = {n: candidate_block(tmdb, r) for n, r in (("forward", False), ("reversed", True))}
    for f in files:
        ex = build_excerpt(json.loads(f.read_text()))
        for n, b in blocks.items():
            if cache_get(cache_dir, cache_key(model, b, ex)):
                n_cached += 1
                continue
            total += estimate_call_cost(est_tokens(b), est_tokens(ex), warm.get(n, False), model)
            warm[n] = True
    bt = est_tokens(blocks["forward"]) + est_tokens(INSTRUCTIONS)
    print(f"{tmdb.get('show_title')} s{tmdb['episodes'][0]['season']:02d}: files={len(files)} "
          f"prefix~{bt} tok, excerpt~{est_tokens(build_excerpt(json.loads(files[0].read_text()))) if files else 0} tok, "
          f"cached calls={n_cached}, projected ${total:.3f} (estimate, chars/{CHARS_PER_TOKEN:g}; DRY RUN, no API call)")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--run", action="store_true", help="make real API calls (spends money)")
    ap.add_argument("--dry-run", action="store_true", help="default; no API call")
    ap.add_argument("--ceiling", type=float, default=1.0, help="USD per season")
    ap.add_argument("--only")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    dry = not a.run
    client = None
    if not dry:
        import anthropic
        client = anthropic.Anthropic()
    totals = run(Path(a.out), client, a.ceiling, a.only, a.limit, dry, a.model)
    print(("projected total $" if dry else "new cost total $") + f"{sum(totals.values()):.3f}")


if __name__ == "__main__":
    main()
