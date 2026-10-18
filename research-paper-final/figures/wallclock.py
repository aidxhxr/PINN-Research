"""Collect measured wall-clock time per regime for each estimator into
experiments/17-sensitivity-pinn/runs/20261017_paper/wallclock.json (used by Fig 6E and the
text). Every number is read from a run artifact; nothing is estimated.
  * PINN (integral): training_seconds of start 0 (seed 42 and 7), CPU 16 thr;
  * SGN-PINN: seconds of both schedules summed (both seeds), one GPU, on top
    of the PINN it starts from;
  * MLE: seconds of the single-start relative-weighted fit, CPU 14 processes;
  * CRLB: Fisher computation, 360 sensitivity solves per regime, 24 CPU
    processes (timed once for all four regimes; per-regime = total/4).
"""
import glob
import json
import os

import style as st
import paperdata as P

out = {}
pinn = []
for seed in (42, 7):
    for r in st.REGIMES:
        s = P._json(os.path.join(P.SENS, "runs", f"20261002_integral_seed{seed}", f"{P.SAFE[r]}_starts.json"))
        if s:
            pinn.append(next(x for x in s["starts"] if x["start"] == 0)["timings"]["training_seconds"])
sgn = []
for seed in (42, 7):
    for r in st.REGIMES:
        t = [P._json(os.path.join(P.RUN, f"sgn_seed{seed}", f"{P.SAFE[r]}_pinn{tag}.json")) for tag in ("_S1", "_S2")]
        if all(t):
            sgn.append(sum(x["seconds"] for x in t))
mle = [json.load(open(f))["seconds"] for f in glob.glob(os.path.join(P.RUN, "ms_tight", "*_init_s*_rel.json"))]
fs = json.load(open(os.path.join(P.RUN, "fim2_base", "fim_summary.json")))
fim = [v["seconds"] for v in fs.values()]
out["CRLB (Fisher)"] = dict(range_s=[min(fim), max(fim)], device="24 CPU", color=st.INK)
out["MLE"] = dict(range_s=[min(mle), max(mle)], device="14 CPU", color=P.EST_COLOR["MLE"])
out["SGN-PINN"] = dict(range_s=[min(sgn), max(sgn)], device="1 GPU, + PINN", color=P.EST_COLOR["SGN-PINN"])
out["PINN (integral)"] = dict(range_s=[min(pinn), max(pinn)], device="16 CPU threads", color=P.EST_COLOR["PINN (integral)"])
for k, v in out.items():
    v["n"] = {"MLE": len(mle), "CRLB (Fisher)": len(fim), "SGN-PINN": len(sgn), "PINN (integral)": len(pinn)}.get(k, 1)
json.dump(out, open(os.path.join(P.RUN, "wallclock.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
