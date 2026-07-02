"""Standalone: per-parameter identifiability verdict for a finished (or
in-progress) run. For EACH unknown it reports the analytic RHS sensitivity
(a-priori, data-free) alongside the empirical mean recovery error over the
regimes, and emits a YES / MARGINAL / NO verdict + identifiability.csv/.json.

Recovery errors are read from each regime's *_recovered.json (written when a
regime finishes), so partial runs report only the regimes done so far.

Usage:
    python3 identifiability.py                 # newest runs/<ts>/ dir
    python3 identifiability.py runs/2026..../   # a specific run dir
"""
import os
import sys
import glob
import json

from config import BASELINE, REGIMES, UNKNOWN
from reference import generate_references
from plotting import print_sensitivity, report_identifiability


def _safe(name):
    return name.replace(" ", "_").replace("/", "_")


def main(run_dir, T=150.0):
    recovered, true_params = {}, {}
    for name in REGIMES:
        path = os.path.join(run_dir, f"{_safe(name)}_recovered.json")
        if not os.path.exists(path):
            print(f"  (skip {name}: not finished — no *_recovered.json)")
            continue
        with open(path) as fh:
            d = json.load(fh)
        recovered[name] = d["recovered"]
        true_params[name] = d["true"]

    if not recovered:
        print("No finished regimes in", run_dir, "- nothing to report yet.")
        return

    # restrict to the regimes that actually finished
    finished = {n: REGIMES[n] for n in recovered}
    print(f"regimes reported: {list(finished)}")

    print("\nComputing analytic sensitivity (scipy references) ...")
    refs = generate_references(T=T, n_pts=2000)
    sens = print_sensitivity(refs)

    # ensure verdict uses only finished regimes
    import plotting
    plotting.REGIMES = finished  # report_identifiability iterates list(REGIMES)
    report_identifiability(recovered, true_params, sens, out_dir=run_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rd = sys.argv[1]
    else:
        dirs = sorted(glob.glob("runs/*/"))
        rd = dirs[-1] if dirs else "."
    print("run dir:", rd)
    main(rd)
