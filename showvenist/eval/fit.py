"""Weight fitting for `eval --fit` (spec §16.3; decisions 40, 41).

Conditional logit, the solver's own model without the at-most-once constraint: for each
labeled file a softmax over its E candidate scores

    S_j = log_prior(E, π_x) + Σ_k w_k · clamp_k(LLR_k(j))      and "unmatched" with score 0,

maximizing the log-probability of the labeled option (for an `episode?` file, decision 41,
the total probability of all episodes), minus a ridge penalty λ/2 · Σ_k (w_k − w0_k)² that
pulls toward the built-in defaults. Weights are bounded below by 0. Only weights are fitted;
clamps, the prior and shape parameters stay fixed.

This module is purely numerical: the pipeline that analyzes and scores labeled files builds
the `FitItem`s. Base-regime weights are fitted on all files; LLM-regime weights only on the
files `--llm auto` would send. The caller chooses those subsets.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from showvenist.config import LlmWeights, WeightsConfig
from showvenist.fusion import _range, log_prior
from showvenist.scoring.types import (
    DEFAULT_CLAMPS,
    DEFAULT_WEIGHTS,
    PI_X_DEFAULT,
    REGIME_SCORERS,
    Regime,
)

Target = Literal["episode", "extra", "any_episode"]
_KIND = {"episode": 0, "extra": 1, "any_episode": 2}


@dataclass(frozen=True)
class FitItem:
    """One labeled file as the fit sees it.

    `llr` maps scorer name -> RAW (unclamped, unweighted) (E,) evidence for this file; a scorer
    that is missing or unavailable for the file is simply absent (it contributes 0).
    `target == "any_episode"` is the decision-41 set label for `episode?` files: the target
    is the softmax mass of the columns in `target_set` (the episodes no other labeled file
    holds), or of every episode when `target_set` is None.
    """

    season_key: str
    regime: Regime
    llr: dict[str, np.ndarray]
    n_candidates: int
    target: Target
    target_index: int | None = None
    target_set: tuple[int, ...] | None = None


# ---- the objective ---------------------------------------------------------------------------


@dataclass
class _Data:
    names: tuple[str, ...]
    X: np.ndarray  # (N, Emax, K) clamped features, 0 in padding
    mask: np.ndarray  # (N, Emax) True for real candidates
    prior: np.ndarray  # (N,)
    kind: np.ndarray  # (N,) 0 episode, 1 extra, 2 any_episode
    tidx: np.ndarray  # (N,) target column for kind 0, else 0
    tset: np.ndarray  # (N, Emax) True for the set-label columns of kind 2
    c: np.ndarray  # (N,) item weights, sum == N

    @property
    def n(self) -> int:
        return len(self.kind)


def _item_weights(kinds: np.ndarray, extras_share: float | None) -> np.ndarray:
    n = len(kinds)
    c = np.ones(n)
    if extras_share is None or n == 0:
        return c
    if not 0.0 < extras_share < 1.0:
        raise ValueError("extras_share must be in (0, 1)")
    is_x = kinds == _KIND["extra"]
    n_x = int(is_x.sum())
    if n_x == 0:
        raise ValueError("extras_share given but no item is labeled extra")
    if n_x == n:
        raise ValueError("extras_share given but every item is labeled extra")
    # extras carry `extras_share` of the total weight; the total stays n (normalized objective)
    c[is_x] = extras_share * n / n_x
    c[~is_x] = (1.0 - extras_share) * n / (n - n_x)
    return c


def _build(
    items: Sequence[FitItem],
    regime: Regime,
    clamps: dict[str, tuple[float, float]],
    pi_x: float,
    extras_share: float | None,
) -> _Data:
    names = REGIME_SCORERS[regime]
    items = [it for it in items if it.regime == regime]
    if not items:
        raise ValueError(f"no fit items in the {regime!r} regime")
    n, k = len(items), len(names)
    emax = max(it.n_candidates for it in items)
    X = np.zeros((n, emax, k))
    mask = np.zeros((n, emax), dtype=bool)
    prior = np.zeros(n)
    kind = np.zeros(n, dtype=int)
    tidx = np.zeros(n, dtype=int)
    tset = np.zeros((n, emax), dtype=bool)
    for i, it in enumerate(items):
        e = it.n_candidates
        if e < 1:
            raise ValueError(f"item {i}: n_candidates must be >= 1")
        if it.target not in _KIND:
            raise ValueError(f"item {i}: unknown target {it.target!r}")
        mask[i, :e] = True
        prior[i] = log_prior(e, pi_x)
        kind[i] = _KIND[it.target]
        if it.target == "episode":
            if it.target_index is None or not 0 <= it.target_index < e:
                raise ValueError(f"item {i}: episode target_index out of range")
            tidx[i] = it.target_index
        elif it.target == "any_episode":
            if it.target_set is None:
                tset[i, :e] = True
            else:
                cols = list(it.target_set)
                if not cols or any(not 0 <= c < e for c in cols):
                    raise ValueError(f"item {i}: target_set empty or out of range")
                tset[i, cols] = True
        for j, name in enumerate(names):
            raw = it.llr.get(name)
            if raw is None:
                continue
            raw = np.nan_to_num(np.asarray(raw, dtype=float))
            if raw.shape != (e,):
                raise ValueError(f"item {i}: llr[{name!r}] has shape {raw.shape}, want ({e},)")
            lo, hi = _range(name, clamps)
            X[i, :e, j] = np.clip(raw, lo, hi)
    return _Data(names, X, mask, prior, kind, tidx, tset, _item_weights(kind, extras_share))


def _per_item(w: np.ndarray, d: _Data) -> tuple[np.ndarray, np.ndarray]:
    """Per-item negative log-likelihood (N,) and its gradient wrt w (N, K)."""
    z = d.prior[:, None] + d.X @ w
    z = np.where(d.mask, z, -np.inf)
    lse_ep = logsumexp(z, axis=1)  # log Σ_j exp(S_j)
    lse = np.logaddexp(lse_ep, 0.0)  # ... plus unmatched (score 0)
    p = np.exp(z - lse[:, None])  # softmax mass of each episode (0 in padding)
    expected = np.einsum("ne,nek->nk", p, d.X)

    rows = np.arange(d.n)
    tgt = np.zeros(d.n)
    tfeat = np.zeros_like(expected)
    ep = d.kind == 0
    tgt[ep] = z[rows[ep], d.tidx[ep]]
    tfeat[ep] = d.X[rows[ep], d.tidx[ep]]
    anyep = d.kind == 2
    if anyep.any():
        zs = np.where(d.tset[anyep], z[anyep], -np.inf)  # the set-label columns only
        lse_set = logsumexp(zs, axis=1)
        q = np.exp(zs - lse_set[:, None])  # softmax over the target set
        tgt[anyep] = lse_set
        tfeat[anyep] = np.einsum("ne,nek->nk", q, d.X[anyep])
    return lse - tgt, expected - tfeat


def _objective(w: np.ndarray, d: _Data, lam: float, w0: np.ndarray) -> tuple[float, np.ndarray]:
    nll, grad = _per_item(w, d)
    diff = w - w0
    f = float(d.c @ nll) + 0.5 * lam * float(diff @ diff)
    g = d.c @ grad + lam * diff
    return f, g


def _defaults(regime: Regime, defaults: dict[str, float] | None) -> np.ndarray:
    src = DEFAULT_WEIGHTS if defaults is None else {**DEFAULT_WEIGHTS, **defaults}
    return np.array([float(src[n]) for n in REGIME_SCORERS[regime]])


def fit_weights(
    items: Sequence[FitItem],
    regime: Regime,
    defaults: dict[str, float] | None = None,
    lam: float = 1.0,
    pi_x: float = PI_X_DEFAULT,
    extras_share: float | None = None,
    clamps: dict[str, tuple[float, float]] = DEFAULT_CLAMPS,
) -> dict[str, float]:
    """Fit the weights of `REGIME_SCORERS[regime]` on the items of that regime (§16.3).

    `defaults` (built-in DEFAULT_WEIGHTS where omitted) is both the start point and the ridge
    target. `extras_share` re-weights items so extras carry that fraction of the total weight
    (the named library has no extras); the total weight stays the number of items, so λ means
    the same thing with or without re-weighting.
    """
    if lam < 0:
        raise ValueError("lam must be >= 0")
    d = _build(items, regime, clamps, pi_x, extras_share)
    w0 = _defaults(regime, defaults)
    res = minimize(
        _objective,
        w0.copy(),
        args=(d, lam, w0),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * len(w0),
    )
    if not np.all(np.isfinite(res.x)):  # L-BFGS-B may stop "abnormally" at the optimum
        raise RuntimeError(f"weight fit failed: {res.message}")
    return {n: float(v) for n, v in zip(d.names, res.x, strict=True)}


def mean_log_loss(
    items: Sequence[FitItem],
    regime: Regime,
    weights: dict[str, float],
    pi_x: float = PI_X_DEFAULT,
    clamps: dict[str, tuple[float, float]] = DEFAULT_CLAMPS,
) -> float:
    """Mean per-file negative log-probability of the labeled option (unweighted)."""
    d = _build(items, regime, clamps, pi_x, None)
    w = np.array([float(weights.get(n, DEFAULT_WEIGHTS[n])) for n in d.names])
    nll, _ = _per_item(w, d)
    return float(nll.mean())


# ---- leave-one-season-out ---------------------------------------------------------------------


@dataclass
class FoldResult:
    season_key: str
    n_items: int
    weights: dict[str, float]  # fitted without this season
    log_loss_default: float
    log_loss_fitted: float


@dataclass
class LosoResult:
    regime: Regime
    lam: float
    defaults: dict[str, float]
    folds: list[FoldResult] = field(default_factory=list)
    pooled_default: float = math.nan  # file-weighted mean held-out log-loss
    pooled_fitted: float = math.nan
    insample_weights: dict[str, float] = field(default_factory=dict)
    insample_log_loss: float = math.nan


def fit_loso(
    items: Sequence[FitItem],
    regime: Regime,
    defaults: dict[str, float] | None = None,
    lam: float = 1.0,
    pi_x: float = PI_X_DEFAULT,
    extras_share: float | None = None,
    clamps: dict[str, tuple[float, float]] = DEFAULT_CLAMPS,
) -> LosoResult:
    """Hold out whole seasons (never single files, §16.3): fit on the rest, score the held-out
    season with fitted and default weights. Also fits on everything (the weights to write)."""
    items = [it for it in items if it.regime == regime]
    seasons = sorted({it.season_key for it in items})
    if len(seasons) < 2:
        raise ValueError("leave-one-season-out needs items from at least two seasons")
    w0 = dict(zip(REGIME_SCORERS[regime], _defaults(regime, defaults).tolist(), strict=True))
    kw = dict(defaults=w0, lam=lam, pi_x=pi_x, clamps=clamps)
    out = LosoResult(regime=regime, lam=lam, defaults=w0)
    tot_n, tot_def, tot_fit = 0, 0.0, 0.0
    for key in seasons:
        held = [it for it in items if it.season_key == key]
        train = [it for it in items if it.season_key != key]
        w = fit_weights(train, regime, extras_share=_share_for(train, extras_share), **kw)
        ll_def = mean_log_loss(held, regime, w0, pi_x, clamps)
        ll_fit = mean_log_loss(held, regime, w, pi_x, clamps)
        out.folds.append(FoldResult(key, len(held), w, ll_def, ll_fit))
        tot_n += len(held)
        tot_def += ll_def * len(held)
        tot_fit += ll_fit * len(held)
    out.pooled_default = tot_def / tot_n
    out.pooled_fitted = tot_fit / tot_n
    out.insample_weights = fit_weights(
        items, regime, extras_share=_share_for(items, extras_share), **kw
    )
    out.insample_log_loss = mean_log_loss(items, regime, out.insample_weights, pi_x, clamps)
    return out


def _share_for(items: Iterable[FitItem], extras_share: float | None) -> float | None:
    """Re-weighting needs both extras and non-extras in the training set; a fold whose
    training data lacks one of them is fitted unweighted rather than failing."""
    if extras_share is None:
        return None
    kinds = {it.target == "extra" for it in items}
    return extras_share if kinds == {True, False} else None


def format_fit_report(result: LosoResult) -> str:
    """Before/after on held-out seasons, then the weights."""
    names = list(result.defaults)
    lines = [
        f"Weight fit, {result.regime} regime (λ={result.lam:g}): held-out log-loss per file",
        f"{'season':<28} {'files':>5} {'default':>8} {'fitted':>8} {'change':>8}",
    ]
    for f in result.folds:
        lines.append(
            f"{f.season_key:<28} {f.n_items:>5} {f.log_loss_default:>8.3f} "
            f"{f.log_loss_fitted:>8.3f} {f.log_loss_fitted - f.log_loss_default:>+8.3f}"
        )
    n = sum(f.n_items for f in result.folds)
    lines.append(
        f"{'pooled':<28} {n:>5} {result.pooled_default:>8.3f} {result.pooled_fitted:>8.3f} "
        f"{result.pooled_fitted - result.pooled_default:>+8.3f}"
    )
    lines.append("")
    lines.append(f"{'weight':<10} {'default':>8} {'fitted':>8} {'fold min':>9} {'fold max':>9}")
    for name in names:
        per = [f.weights[name] for f in result.folds]
        lines.append(
            f"{name:<10} {result.defaults[name]:>8.3f} {result.insample_weights[name]:>8.3f} "
            f"{min(per):>9.3f} {max(per):>9.3f}"
        )
    lines.append(f"in-sample log-loss with fitted weights: {result.insample_log_loss:.3f}")
    return "\n".join(lines)


# ---- writing the config -----------------------------------------------------------------------


def to_weights_config(base: dict[str, float], llm: dict[str, float]) -> WeightsConfig:
    """Fitted weights -> WeightsConfig (missing keys keep their built-in default)."""
    return WeightsConfig(**{k: float(v) for k, v in base.items()}, llm=LlmWeights(**llm))


_HEADER = re.compile(r"^\s*\[")
_BASE_HDR = re.compile(r"^\s*\[\s*weights\s*\]\s*(#.*)?$")
_LLM_HDR = re.compile(r"^\s*\[\s*weights\s*\.\s*llm\s*\]\s*(#.*)?$")


def _fmt(v: float) -> str:
    return repr(float(round(v, 4)))


def _rewrite_table(
    lines: list[str],
    hdr: re.Pattern[str],
    header_text: str,
    values: dict[str, float],
    drop: re.Pattern[str] | None = None,
    after: re.Pattern[str] | None = None,
) -> list[str]:
    start = next((i for i, ln in enumerate(lines) if hdr.match(ln)), None)
    if start is None:
        block = [header_text] + [f"{k} = {_fmt(v)}" for k, v in values.items()]
        pos = len(lines)
        if after is not None:  # place right after the given table, if present
            a = next((i for i, ln in enumerate(lines) if after.match(ln)), None)
            if a is not None:
                pos = next(
                    (i for i in range(a + 1, len(lines)) if _HEADER.match(lines[i])), len(lines)
                )
        prefix = lines[:pos]
        while prefix and not prefix[-1].strip():
            prefix.pop()
        sep = [""] if prefix else []
        tail = lines[pos:]
        return prefix + sep + block + ([""] if tail else []) + tail
    end = next((i for i in range(start + 1, len(lines)) if _HEADER.match(lines[i])), len(lines))
    body: list[str] = []
    seen: set[str] = set()
    for ln in lines[start + 1 : end]:
        if drop is not None and drop.match(ln):
            continue
        m = re.match(r"^(\s*)([A-Za-z0-9_-]+)(\s*=\s*)([^#]*?)(\s*#.*)?$", ln)
        if m and m.group(2) in values:
            key = m.group(2)
            if key in seen:
                continue
            seen.add(key)
            ln = f"{m.group(1)}{key}{m.group(3)}{_fmt(values[key])}{m.group(5) or ''}"
        body.append(ln)
    missing = [f"{k} = {_fmt(v)}" for k, v in values.items() if k not in seen]
    # insert missing keys after the last non-blank line of the table body
    cut = len(body)
    while cut and not body[cut - 1].strip():
        cut -= 1
    body = body[:cut] + missing + body[cut:]
    return lines[: start + 1] + body + lines[end:]


def render_weights(text: str, base: dict[str, float], llm: dict[str, float] | None) -> str:
    """Return `text` with only the [weights] and [weights.llm] tables rewritten.

    An empty or None `llm` (no LLM-regime fit) leaves every existing LLM weight untouched.
    """
    cfg = to_weights_config(base, llm or {})  # validates the keys
    b = {k: float(getattr(cfg, k)) for k in REGIME_SCORERS["base"]}
    lw = {k: float(getattr(cfg.llm, k)) for k in REGIME_SCORERS["llm"]}
    old_llm = tomllib.loads(text).get("weights", {}).get("llm") if text.strip() else None
    lines = text.splitlines()
    if llm:
        # inside [weights], `llm = …` / `llm.x = …` would clash with the [weights.llm] table
        lines = _rewrite_table(
            lines, _BASE_HDR, "[weights]", b, drop=re.compile(r"^\s*llm\s*[.=]")
        )
        lines = _rewrite_table(lines, _LLM_HDR, "[weights.llm]", lw, after=_BASE_HDR)
    else:
        lines = _rewrite_table(lines, _BASE_HDR, "[weights]", b)
    new = "\n".join(lines) + "\n"
    parsed = tomllib.loads(new).get("weights", {})
    got_b = {k: parsed.get(k) for k in b}
    ok = got_b == {k: float(_fmt(v)) for k, v in b.items()}
    if llm:
        got_l = {k: parsed.get("llm", {}).get(k) for k in lw}
        ok = ok and got_l == {k: float(_fmt(v)) for k, v in lw.items()}
    else:
        ok = ok and parsed.get("llm") == old_llm
    if not ok:
        raise ValueError("could not rewrite the [weights] tables of this config safely")
    return new


def write_weights(path: Path, base: dict[str, float], llm: dict[str, float] | None) -> None:
    """Rewrite ONLY [weights] and [weights.llm] in config.toml, creating file/tables if needed.

    Every other line and comment is kept. With `llm` empty or None the LLM-regime weights are
    left exactly as they are. The result is checked with tomllib before the file is replaced
    (atomically).
    """
    path = Path(path)
    text = path.read_text() if path.exists() else ""
    new = render_weights(text, base, llm)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        if path.exists():
            os.chmod(tmp, path.stat().st_mode & 0o777)
        with os.fdopen(fd, "w") as fh:
            fh.write(new)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
