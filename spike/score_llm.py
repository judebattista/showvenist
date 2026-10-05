#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "wordfreq"]
# ///
"""M0: LLM regime. One feature (raw, unclamped log-odds from spike/out/llm/),
same prior, assignment, Delta and leave-one-season-out fit as score.py.
Spec 11.1 (LLM regime), 16.3. Usage: uv run spike/score_llm.py [--lam 1.0 0.1 0.01]
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("score", here / "score.py")
sc = importlib.util.module_from_spec(spec); sys.modules["score"] = sc; spec.loader.exec_module(sc)
sc.SCORERS = ("llm",)

def load(out: Path):
    data = {}
    for f in sorted((out / "llm").glob("*.json")):
        d = json.loads(f.read_text())
        tm = json.loads((out / "tmdb" / f"{d['key']}.json").read_text())
        names = [f"S{int(e['season']):02d}E{int(e['episode']):02d}" for e in tm["episodes"]]
        feat, truth, cost = [], [], 0.0
        for b, x in sorted(d["files"].items()):
            row = np.zeros(len(names))
            for i, n in enumerate(names):
                e = x["episodes"].get(n)
                if e and not x.get("refusal"):
                    row[i] = e["logodds"]
            feat.append(row[:, None]); truth.append(names.index(x["truth"]) if x["truth"] in names else -1)
            cost += x.get("cost_usd", 0)
        data[d["key"]] = {"feat": np.stack([f_.T[0] for f_ in feat])[:, :, None], "truth": truth, "cost": cost}
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, nargs="*", default=[1.0, 0.1, 0.01])
    a = ap.parse_args()
    out = here / "out"; data = load(out)
    res = {}
    for lam in a.lam:
        run = sc.run_modes(data, lam, w_default=(1.0,))
        res[str(lam)] = {"weights_insample": run["weights_insample"], "modes": {}}
        print(f"\n##### lam={lam}  in-sample w_llm={np.round(run['weights_insample'],3).tolist()}")
        for mode, per in run["modes"].items():
            p = sc.pool(per)
            res[str(lam)]["modes"][mode] = {"pooled": p, "per_season": {k: {kk: vv for kk, vv in v.items() if kk != "_files"} for k, v in per.items()}}
            print(f"== {mode} ==")
            for k, m in per.items():
                print(sc.summarize(k[:9], m), "w=", round(m["weights"][0], 2))
            print(sc.summarize("POOLED", p))
    res["cost_usd"] = {k: v["cost"] for k, v in data.items()}
    (out / "results" / "llm_regime.json").write_text(json.dumps(res, indent=1))
    print("\ncost per season:", {k: round(v, 3) for k, v in res["cost_usd"].items()})

main()
