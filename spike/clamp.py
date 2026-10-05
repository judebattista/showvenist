# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "wordfreq"]
# ///
"""M0 follow-up (post hoc, 2026-10-05): other ways to turn the LLM's probabilities into evidence.
Reads the cached responses in spike/out/llm/ (no API calls, no cost). Usage: uv run spike/clamp.py
The M0 outcome (RESULTS.md, decision 38) is unchanged; this feeds the M3 contract (plan section 4, item 2)."""
import importlib.util, json, sys, math
from pathlib import Path
import numpy as np
here = Path('/home/jude/src/showvenist/spike')
spec = importlib.util.spec_from_file_location("score", here / "score.py")
sc = importlib.util.module_from_spec(spec); sys.modules["score"] = sc; spec.loader.exec_module(sc)
sc.SCORERS = ("llm",)
out = here / "out"
raw = {}
for f in sorted((out/"llm").glob("*.json")):
    d = json.loads(f.read_text()); raw[d["key"]] = d
# --- stats
tot=ok=0; tops=[]; nones=[]
for k,d in raw.items():
    for b,x in d["files"].items():
        eps=x["episodes"]; ps=sorted(((v["p"],n) for n,v in eps.items()),reverse=True)
        tot+=1; ok+= ps[0][1]==x["truth"]; tops.append(ps[0][0]); nones.append(x.get("p_none",0))
import statistics as st
print("files",tot,"top1",ok)
print("p_top deciles",[round(v,3) for v in st.quantiles(tops,n=10)])
print("p_none deciles",[round(v,3) for v in st.quantiles(nones,n=10)])
print("p_none>0.2:",sum(n>0.2 for n in nones),"p_top<0.5:",sum(t<0.5 for t in tops))
def build(transform):
    data={}
    for k,d in raw.items():
        tm=json.loads((out/"tmdb"/f"{k}.json").read_text())
        names=[f"S{int(e['season']):02d}E{int(e['episode']):02d}" for e in tm["episodes"]]
        E=len(names); feat=[]; truth=[]
        for b,x in sorted(d["files"].items()):
            row=np.zeros(E)
            for i,n in enumerate(names):
                e=x["episodes"].get(n)
                if e and not x.get("refusal"): row[i]=transform(e,x,E)
            feat.append(row); truth.append(names.index(x["truth"]) if x["truth"] in names else -1)
        data[k]={"feat":np.stack(feat)[:,:,None],"truth":truth,"cost":0}
    return data
def clip(p,eps=1e-4): return min(max(p,eps),1-eps)
def logit(p): p=clip(p); return math.log(p/(1-p))
def B(e,x,E): return logit(e["p"])+math.log(E-1)
def H(e,x,E): return B(e,x,E)-(logit(x.get("p_none",0.025))+math.log(E))   # extras evidence from p_none
def Hc(e,x,E): return max(-8,min(8,H(e,x,E)))
def Bs(eps):
    return lambda e,x,E: math.log(clip(e["p"],eps)/(1-clip(e["p"],eps)))+math.log(E-1)
T={
 "A raw log(p/p_none)": lambda e,x,E: e["logodds"],
 "B logit(p)+log(E-1)": B,
 "H B minus p_none shift": H,
 "Hc H clamp +-8": Hc,
 "B eps=0.01 (p<=.99)": Bs(0.01),
 "B eps=0.03 (p<=.97)": Bs(0.03),
}
for name,tf in T.items():
    data=build(tf)
    run=sc.run_modes(data,0.1,w_default=(1.0,))
    print("\n####",name,"in-sample w",np.round(run["weights_insample"],2).tolist())
    for mode,per in run["modes"].items():
        if mode not in ("loso",) and "loso" not in mode and "default" not in mode: continue
        p=sc.pool(per); print(" ",mode, sc.summarize("POOLED",p))
print("\n--- the p_none>0.2 files: top candidate correct?  B and H evidence (w=1)")
for k,d in raw.items():
    tm=json.loads((out/"tmdb"/f"{k}.json").read_text()); E=len(tm["episodes"])
    for b,x in sorted(d["files"].items()):
        if x.get("p_none",0)>0.2:
            top=max(x["episodes"].items(),key=lambda kv:kv[1]["p"])
            e=top[1]; print(f"{k[:12]} {b[-14:]} truth={x['truth']} top={top[0]} p={e['p']:.2f} p_none={x['p_none']:.2f}  B={B(e,x,E):5.1f}  H={H(e,x,E):5.1f}  old={e['logodds']:.1f}")
