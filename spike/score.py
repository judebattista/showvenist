#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "wordfreq"]
# ///
"""M0 spike scoring stage: synopsis BM25 + guest-name scorers, prior, basic
assignment, Delta confidence, conditional-logit weight fit, metrics.

Spec: 9.1, 9.2 (synopsis, guest names), 11.1, 11.2, 11.5, 11.7, 16.3.
Usage: uv run spike/score.py [--out spike/out] [--lam 1.0] [--show KEY ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

PI_X = 0.15
HIGH, MEDIUM = 4.6, 2.2
CLAMP_SYN, CLAMP_GUEST = 4.0, 6.0
SYN_PER_SIGMA = 0.8
DICT_ZIPF = 3.0       # name tokens at/above this are "dictionary words" (guest filter)
STOP_ZIPF = 5.5       # query/doc tokens at/above this are stopwords for BM25
NEG = -1e9
SCORERS = ("synopsis", "guest")

_zipf_cache: dict[str, float] = {}


def zipf(tok: str) -> float:
    if tok not in _zipf_cache:
        from wordfreq import zipf_frequency
        _zipf_cache[tok] = zipf_frequency(tok, "en")
    return _zipf_cache[tok]


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()) if len(t) >= 3 or t in ("i",)]


def name_tokens(name: str) -> list[str]:
    name = re.sub(r"\(.*?\)", " ", name or "")
    return [t.replace("'s", "") for t in tokens(name)]


# ---------------------------------------------------------------- scorers

def cast_token_sets(tmdb: dict):
    """Returns (series_tokens, all_name_tokens) over series cast + guest stars
    (actors and characters). Series_cast entries may be str or dict."""
    series: set[str] = set()
    for c in tmdb.get("series_cast", []) or []:
        if isinstance(c, str):
            series.update(name_tokens(c))
        else:
            for k in ("name", "actor", "character"):
                series.update(name_tokens(c.get(k) or ""))
    allnames = set(series)
    for ep in tmdb["episodes"]:
        for g in ep.get("guest_stars", []) or []:
            allnames.update(name_tokens(g.get("actor") or ""))
            allnames.update(name_tokens(g.get("character") or ""))
    return series, allnames


def removable_names(series: set[str], allnames: set[str]) -> set[str]:
    """Name tokens stripped before BM25: series-cast tokens always; other name
    tokens only if not dictionary words (so guest 'Doctor' keeps its meaning)."""
    return series | {t for t in allnames if zipf(t) < DICT_ZIPF}


def content_tokens(text: str, strip: set[str]) -> list[str]:
    return [t for t in tokens(text) if t not in strip and zipf(t) < STOP_ZIPF]


def bm25_scores(docs: list[list[str]], query: list[str], k1=1.2, b=0.75) -> np.ndarray:
    N = len(docs)
    avgdl = max(np.mean([len(d) for d in docs]), 1e-9)
    df = Counter(t for d in docs for t in set(d))
    tfs = [Counter(d) for d in docs]
    out = np.zeros(N)
    for t in set(query):  # unique query terms: repeated words in dialogue don't dominate
        if t not in df:
            continue
        idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
        for j in range(N):
            f = tfs[j].get(t, 0)
            if f:
                out[j] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(docs[j]) / avgdl))
    return out


def synopsis_llr(doc_tokens: list[list[str]], query: list[str]) -> np.ndarray:
    E = len(doc_tokens)
    s = bm25_scores(doc_tokens, query)
    if E < 3 or not np.any(s):
        return np.zeros(E)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    if mad < 1e-9:
        mad = max(s.std(), 1e-9)
    z = (s - med) / mad
    zc = math.sqrt(2 * math.log(E))
    # 0.8 nats per sigma *above the chance maximum* (spec ambiguity: could be
    # 0.8*z once z>zc; we credit only the excess, which is continuous at zc).
    llr = np.where(z > zc, SYN_PER_SIGMA * (z - zc), 0.0)
    return np.minimum(llr, CLAMP_SYN)


def guest_name_table(tmdb: dict, series: set[str]):
    """Per episode: set of distinct guest-name tokens (non-dictionary, non-series-cast);
    and IDF over episodes."""
    per_ep: list[set[str]] = []
    for ep in tmdb["episodes"]:
        toks: set[str] = set()
        for g in ep.get("guest_stars", []) or []:
            for part in re.split(r"[/;]", g.get("character") or ""):
                for t in name_tokens(part):
                    if t not in series and zipf(t) < DICT_ZIPF:
                        toks.add(t)
        per_ep.append(toks)
    E = len(per_ep)
    df = Counter(t for s in per_ep for t in s)
    idf = {t: math.log(E / df[t]) for t in df}
    return per_ep, idf


def guest_llr(per_ep, idf, heard: set[str]) -> np.ndarray:
    out = np.array([sum(idf[t] for t in toks if t in heard) for toks in per_ep], float)
    return np.minimum(out, CLAMP_GUEST)


class Season:
    """Scorer state for one season's TMDB data."""

    def __init__(self, tmdb: dict):
        self.tmdb = tmdb
        self.eps = tmdb["episodes"]
        self.E = len(self.eps)
        self.series, allnames = cast_token_sets(tmdb)
        self.strip = removable_names(self.series, allnames)
        self.docs = [content_tokens(f"{e.get('title','')} {e.get('overview','')}", self.strip)
                     for e in self.eps]
        self.guest_per_ep, self.guest_idf = guest_name_table(tmdb, self.series)

    def features(self, lines: list[str]) -> np.ndarray:
        """E x 2 clamped LLRs [synopsis, guest] for one file's dialogue."""
        text = " ".join(lines)
        q = content_tokens(text, self.strip)
        heard = set(tokens(text))
        return np.stack([synopsis_llr(self.docs, q), guest_llr(self.guest_per_ep, self.guest_idf, heard)], axis=1)


def log_prior(E: int, pi_x: float = PI_X) -> float:
    return math.log((1 - pi_x) / E) - math.log(pi_x)


def score_matrix(feat: np.ndarray, w, E: int) -> np.ndarray:
    """feat: F x E x K -> S: F x E (episode columns only; unmatched is 0)."""
    return log_prior(E) + feat @ np.asarray(w, float)


# ---------------------------------------------------------------- assignment

def _solve(S: np.ndarray, forbid: tuple[int, int] | None = None):
    F, E = S.shape
    M = np.full((F, E + F), NEG)
    M[:, :E] = S
    M[np.arange(F), E + np.arange(F)] = 0.0
    if forbid is not None:
        M[forbid] = NEG
    r, c = linear_sum_assignment(M, maximize=True)
    sigma = np.empty(F, int)
    sigma[r] = c
    total = M[r, c].sum()
    return sigma, float(total)


def assign(S: np.ndarray):
    """Basic assignment (11.5) + Delta by re-solve (11.7).
    Returns dict with sigma (episode idx or -1 = unmatched), delta, runner_up."""
    F, E = S.shape
    sigma, total = _solve(S)
    delta = np.zeros(F)
    runner = np.full(F, -2)
    for i in range(F):
        s2, t2 = _solve(S, (i, sigma[i]))
        delta[i] = total - t2
        runner[i] = s2[i] if s2[i] < E else -1
    out = np.where(sigma < E, sigma, -1)
    return {"sigma": out, "delta": delta, "runner_up": runner, "total": total}


def level(d: float) -> str:
    return "high" if d >= HIGH else "medium" if d >= MEDIUM else "low"


# ---------------------------------------------------------------- fitting

def nll(w, feats: list[np.ndarray], truths: list[int], E_list: list[int], lam: float, w0) -> float:
    """Conditional logit: softmax over E candidates + unmatched(0); prior fixed."""
    tot = 0.0
    for f, y, E in zip(feats, truths, E_list):
        # f: F x E x K
        logits = log_prior(E) + f @ w  # F x E
        full = np.concatenate([logits, np.zeros((len(logits), 1))], axis=1)
        lse = np.logaddexp.reduce(full, axis=1)
        idx = np.where(np.asarray(y) >= 0, y, E)
        tot -= (full[np.arange(len(full)), idx] - lse).sum()
    # L2 pull toward defaults, in total-NLL units (== a N(w0, 1/(2*lam)) prior)
    return tot + lam * float(np.sum((np.asarray(w) - w0) ** 2))


def fit_weights(feats, truths, E_list, lam=1.0, w0=None) -> np.ndarray:
    K = feats[0].shape[2]
    w0 = np.ones(K) if w0 is None else np.asarray(w0, float)
    res = minimize(nll, w0, args=(feats, truths, E_list, lam, w0), method="L-BFGS-B",
                   bounds=[(0.0, 5.0)] * K)
    return res.x


# ---------------------------------------------------------------- metrics

def rank_of_true(vals: np.ndarray, t: int) -> tuple[float, bool]:
    """(average-tie rank, strictly-first?)."""
    greater = int((vals > vals[t]).sum())
    ties = int((vals == vals[t]).sum()) - 1
    return 1 + greater + ties / 2, greater == 0 and ties == 0


def season_metrics(feat: np.ndarray, truth: list[int], w) -> dict:
    """feat F x E x K; truth: episode idx per file (-1 = extra)."""
    F, E, K = feat.shape
    S = score_matrix(feat, w, E)
    a = assign(S)
    truth = np.asarray(truth)
    sig, d = a["sigma"], a["delta"]
    correct = sig == truth
    lv = [level(x) for x in d]
    high = np.array([l == "high" for l in lv])
    med = np.array([l == "medium" for l in lv])
    argmax = np.where(S.max(1) > 0, S.argmax(1), -1)
    gaps, per = [], {}
    for i, t in enumerate(truth):
        if t >= 0:
            others = np.delete(S[i], t)
            gaps.append(float(S[i, t] - others.max()) if len(others) else 0.0)
    for k, name in enumerate(SCORERS):
        rr, first = [], 0
        n = 0
        for i, t in enumerate(truth):
            if t < 0:
                continue
            r, f = rank_of_true(feat[i, :, k], t)
            rr.append(1 / r)
            first += f
            n += 1
        per[name] = {"rank1_rate": first / n if n else None, "mrr": float(np.mean(rr)) if rr else None}
    # combined (current weights), no assignment constraint
    rr, first, n = [], 0, 0
    for i, t in enumerate(truth):
        if t >= 0:
            r, f = rank_of_true(S[i], t)
            rr.append(1 / r)
            first += f
            n += 1
    per["combined"] = {"rank1_rate": first / n if n else None, "mrr": float(np.mean(rr)) if rr else None}
    return {
        "n_files": int(F), "n_episodes": int(E),
        "top1_assigned": float(correct.mean()) if F else None,
        "top1_argmax": float((argmax == truth).mean()) if F else None,
        "coverage_high": float(high.mean()) if F else None,
        "correct_at_high": int((high & correct).sum()),
        "errors_at_high": int((high & ~correct).sum()),
        "n_high": int(high.sum()), "n_medium": int(med.sum()), "n_low": int((~high & ~med).sum()),
        "delta_quantiles": {str(q): float(np.quantile(d, q)) for q in (0, .1, .25, .5, .75, .9, 1)} if F else {},
        "per_scorer": per,
        "gap_true_vs_best_other": {
            "mean": float(np.mean(gaps)) if gaps else None,
            "median": float(np.median(gaps)) if gaps else None,
            "frac_positive": float(np.mean([g > 0 for g in gaps])) if gaps else None},
        "weights": [float(x) for x in w],
        "_files": {"delta": [float(x) for x in d], "assigned": [int(x) for x in sig],
                   "truth": [int(x) for x in truth], "level": lv,
                   "runner_up": [int(x) for x in a["runner_up"]]},
    }


def pool(per_season: dict[str, dict]) -> dict:
    """Pool file-level outcomes across seasons (re-derived from stored per-file data)."""
    ms = list(per_season.values())
    d = np.concatenate([m["_files"]["delta"] for m in ms])
    sig = np.concatenate([m["_files"]["assigned"] for m in ms])
    tr = np.concatenate([m["_files"]["truth"] for m in ms])
    correct = sig == tr
    high = d >= HIGH
    med = (d >= MEDIUM) & ~high
    out = {
        "n_files": int(len(d)),
        "top1_assigned": float(correct.mean()),
        "coverage_high": float(high.mean()),
        "correct_at_high": int((high & correct).sum()),
        "errors_at_high": int((high & ~correct).sum()),
        "n_high": int(high.sum()), "n_medium": int(med.sum()), "n_low": int((~high & ~med).sum()),
        "delta_quantiles": {str(q): float(np.quantile(d, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
    }
    # pooled per-scorer / gap: weight by files
    n = np.array([m["n_files"] for m in ms], float)
    def wavg(f):
        vals = [f(m) for m in ms]
        ok = [(v, x) for v, x in zip(vals, n) if v is not None]
        return float(sum(v * x for v, x in ok) / sum(x for _, x in ok)) if ok else None
    out["top1_argmax"] = wavg(lambda m: m["top1_argmax"])
    out["per_scorer"] = {k: {"rank1_rate": wavg(lambda m, k=k: m["per_scorer"][k]["rank1_rate"]),
                             "mrr": wavg(lambda m, k=k: m["per_scorer"][k]["mrr"])}
                         for k in ms[0]["per_scorer"]}
    out["gap_true_vs_best_other"] = {
        "mean": wavg(lambda m: m["gap_true_vs_best_other"]["mean"]),
        "frac_positive": wavg(lambda m: m["gap_true_vs_best_other"]["frac_positive"])}
    return out


def run_modes(data: dict[str, dict], lam: float, w_default=(1.0, 1.0)) -> dict:
    """data[key] = {"feat": F x E x K, "truth": [...]}.  Returns metrics for
    default / loso / insample weights."""
    keys = list(data)
    def fit_on(ks):
        return fit_weights([data[k]["feat"] for k in ks], [data[k]["truth"] for k in ks],
                           [data[k]["feat"].shape[1] for k in ks], lam)
    res: dict[str, dict] = {"default": {}, "loso": {}, "insample": {}}
    w_all = fit_on(keys)
    for k in keys:
        f, t = data[k]["feat"], data[k]["truth"]
        res["default"][k] = season_metrics(f, t, w_default)
        res["insample"][k] = season_metrics(f, t, w_all)
        others = [o for o in keys if o != k]
        res["loso"][k] = season_metrics(f, t, fit_on(others) if others else np.asarray(w_default))
    return {"modes": res, "weights_insample": [float(x) for x in w_all]}


# ---------------------------------------------------------------- I/O

def _lines_of(window_lines) -> list[str]:
    return [l["text"] if isinstance(l, dict) else str(l) for l in window_lines]


def file_lines(d: dict) -> tuple[list[str], list[str]]:
    ocr = [t for w in d.get("windows", []) for t in _lines_of(w.get("lines", []))]
    cap = _lines_of(d.get("caption_lines") or [])
    return ocr, cap


def load_season(out: Path, key: str, rows: list[dict]):
    tmdb = json.loads((out / "tmdb" / f"{key}.json").read_text())
    season = Season(tmdb)
    idx = {(int(e["season"]), int(e["episode"])): i for i, e in enumerate(season.eps)}
    by_name = {Path(r["path"]).name: r for r in rows}
    feats_o, feats_c, truth, names, oq = [], [], [], [], []
    has_cap = True
    for p in sorted((out / "dialogue" / key).glob("*.json")):
        d = json.loads(p.read_text())
        r = by_name.get(Path(d.get("path", p.stem)).name) or by_name.get(p.stem) or by_name.get(p.stem + ".mkv")
        if r is None:
            print(f"  warn: no label for {p.name}", file=sys.stderr)
            continue
        m = re.fullmatch(r"S(\d+)E(\d+)", r["label"].strip(), re.I)
        t = idx.get((int(m[1]), int(m[2])), -1) if m else -1
        ocr, cap = file_lines(d)
        feats_o.append(season.features(ocr))
        if cap:
            feats_c.append(season.features(cap))
        else:
            has_cap = False
            feats_c.append(None)
        truth.append(t)
        names.append(p.stem)
        oq.append(d.get("ocr_quality"))
    return {"feat": np.stack(feats_o), "truth": truth, "names": names, "ocr_quality": oq,
            "cap_feats": feats_c, "has_cap": has_cap and any(f is not None for f in feats_c),
            "season": season}


def summarize(tag: str, m: dict) -> str:
    return (f"{tag:<9} top1={m['top1_assigned']:.2f} cov@4.6={m['coverage_high']:.2f} "
            f"err@high={m['errors_at_high']} (high={m['n_high']} med={m['n_medium']} low={m['n_low']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--lam", type=float, default=1.0, help="L2 pull toward w=1")
    ap.add_argument("--show", nargs="*", help="restrict to season keys")
    a = ap.parse_args()
    out = Path(a.out)
    groups: dict[str, list[dict]] = {}
    with open(out / "labels.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            groups.setdefault(r['show'], []).append(r)
    data = {}
    for key, rows in groups.items():
        if a.show and key not in a.show:
            continue
        if not (out / "dialogue" / key).is_dir() or not (out / "tmdb" / f"{key}.json").exists():
            print(f"skip {key}: missing dialogue or tmdb", file=sys.stderr)
            continue
        data[key] = load_season(out, key, rows)
    if not data:
        sys.exit("no seasons loaded")
    run = run_modes(data, a.lam)
    resdir = out / "results"
    resdir.mkdir(parents=True, exist_ok=True)
    pooled = {"weights_insample": run["weights_insample"], "lam": a.lam, "modes": {}}
    for mode, per in run["modes"].items():
        pooled["modes"][mode] = {"pooled": pool(per),
                                 "per_season": {k: {kk: vv for kk, vv in v.items() if kk != "_files"}
                                                for k, v in per.items()}}
    # captions: score seasons with caption lines using the same weights (default/in-sample)
    cap_report = {}
    for k, d in data.items():
        if d["has_cap"] and all(f is not None for f in d["cap_feats"]):
            cf = np.stack(d["cap_feats"])
            w = run["weights_insample"]
            cap_report[k] = {"captions": {kk: vv for kk, vv in season_metrics(cf, d["truth"], w).items() if kk != "_files"},
                             "ocr": {kk: vv for kk, vv in run["modes"]["insample"][k].items() if kk != "_files"}}
    pooled["captions_vs_ocr_insample"] = cap_report
    (resdir / "pooled.json").write_text(json.dumps(pooled, indent=1))
    for k, d in data.items():
        names = d["names"]
        season = d["season"]
        files = []
        for mode in run["modes"]:
            pass
        rec = {"key": k, "files": names, "ocr_quality": d["ocr_quality"],
               "episodes": [f"S{int(e['season']):02d}E{int(e['episode']):02d}" for e in season.eps],
               "modes": {}}
        for mode, per in run["modes"].items():
            m = per[k]
            rec["modes"][mode] = {kk: vv for kk, vv in m.items() if kk != "_files"}
            rec["modes"][mode]["per_file"] = [
                {"file": names[i], "truth": m["_files"]["truth"][i], "assigned": m["_files"]["assigned"][i],
                 "runner_up": m["_files"]["runner_up"][i], "delta": m["_files"]["delta"][i],
                 "level": m["_files"]["level"][i]} for i in range(len(names))]
        if k in cap_report:
            rec["captions_vs_ocr_insample"] = cap_report[k]
        (resdir / f"{k}.json").write_text(json.dumps(rec, indent=1))
    print(f"in-sample weights [synopsis, guest]: {np.round(run['weights_insample'], 3).tolist()}")
    for mode, per in run["modes"].items():
        print(f"\n== {mode} ==")
        for k, m in per.items():
            print(summarize(k[:9], m))
        p = pooled["modes"][mode]["pooled"]
        print(summarize("POOLED", p))
        print("  per-scorer:", {s: (round(v['rank1_rate'] or 0, 2), round(v['mrr'] or 0, 2)) for s, v in p["per_scorer"].items()})
    for k, c in cap_report.items():
        print(f"captions {k}: top1={c['captions']['top1_assigned']:.2f} cov={c['captions']['coverage_high']:.2f} "
              f"vs ocr top1={c['ocr']['top1_assigned']:.2f} cov={c['ocr']['coverage_high']:.2f}")


if __name__ == "__main__":
    main()
