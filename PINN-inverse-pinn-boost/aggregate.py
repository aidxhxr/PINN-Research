"""Print the per-regime under-10% recovery count for a run dir, vs the
multicond PINN baseline (10/4/5/7)."""
import glob
import json
import os
import sys

BASELINE = {"Normal": 10, "Early Adenoma": 4,
            "Advanced Adenoma": 5, "Severe APC Loss": 7}
ORDER = ["Normal", "Early Adenoma", "Advanced Adenoma", "Severe APC Loss"]


def count_under(rec, tru, tol=0.10):
    return sum(1 for k in tru if tru[k] != 0
               and abs(rec[k] - tru[k]) / abs(tru[k]) < tol)


def main(run_dir):
    safe2name = {r.replace(" ", "_").replace("/", "_"): r for r in ORDER}
    got = {}
    for f in glob.glob(os.path.join(run_dir, "*_recovered.json")):
        d = json.load(open(f))
        safe = os.path.basename(f).replace("_recovered.json", "")
        name = safe2name.get(safe, safe)
        got[name] = count_under(d["recovered"], d["true"])
    print(f"\nrun: {run_dir}")
    print(f"{'regime':<20s} {'under10%':>9s} {'baseline':>9s} {'delta':>6s}")
    tot = base_tot = 0
    for r in ORDER:
        if r in got:
            b = BASELINE[r]
            print(f"{r:<20s} {got[r]:>9d} {b:>9d} {got[r]-b:>+6d}")
            tot += got[r]; base_tot += b
    print(f"{'TOTAL':<20s} {tot:>9d} {base_tot:>9d} {tot-base_tot:>+6d}")


if __name__ == "__main__":
    main(sys.argv[1])
