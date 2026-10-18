"""Main comparison table: inverse PINN vs PINN + SGN vs classical MLE vs CRLB.

Usage: python3 aggregate.py <sgn_dir> [--pinn-dir D] [--mle-dir D] [--tag T]
"""
import argparse, glob, json, os
import numpy as np
import bench
from wnt_pinn.model.parameters import UNKNOWN

REG = ["Normal", "Early Adenoma", "Advanced Adenoma", "Severe APC Loss"]


def n10(errs):
    return int(sum(float(e) < 0.10 for e in errs.values()))


def pinn_count(d, r):
    f = os.path.join(d, r.replace(" ", "_") + "_recovered.json")
    if not os.path.exists(f):
        return None
    rec = json.load(open(f))["recovered"]
    return n10(bench.rel_errors(rec, r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sgn_dir")
    ap.add_argument("--pinn-dir", default="../10-inverse-integral-residual/runs/20260711_203325_integral")
    ap.add_argument("--mle-dir", default="runs/20261002_mle")
    ap.add_argument("--tag", default="pinn")
    a = ap.parse_args()
    rows = []
    hdr = f"{'regime':18s} {'PINN':>5s} {'+SGN':>5s} {'SGN med%':>8s} {'MLE(init)':>9s} {'MLE(truth)':>10s} {'CRLB sd<5%':>10s} {'SGN sd<5%':>9s}"
    print(hdr)
    tot = np.zeros(6)
    for r in REG:
        s = r.replace(" ", "_")
        sg = json.load(open(os.path.join(a.sgn_dir, f"{s}_{a.tag}.json")))
        mi = json.load(open(os.path.join(a.mle_dir, f"{s}_init.json")))
        mt = json.load(open(os.path.join(a.mle_dir, f"{s}_truth.json")))
        F = np.load(f"../11-fisher-information/runs/20260711_203325_fisher/{s}_FIM.npy") * 150 / 200
        ev, V = np.linalg.eigh(F); ev = np.maximum(ev, 1e-14 * ev.max())
        crlb = np.sqrt((V ** 2 / ev).sum(1))
        sgsd = np.array([sg["sd_logtheta"][k] for k in UNKNOWN])
        pc = pinn_count(a.pinn_dir, r)
        med = 100 * np.median(list(sg["errors"].values()))
        vals = [pc or 0, sg["under10"], mi["under10"], mt["under10"], int((crlb < .05).sum()), int((sgsd < .05).sum())]
        tot += vals
        print(f"{r:18s} {str(pc):>5s} {sg['under10']:5d} {med:8.2f} {mi['under10']:9d} {mt['under10']:10d} {vals[4]:10d} {vals[5]:9d}")
    print(f"{'TOTAL':18s} {int(tot[0]):5d} {int(tot[1]):5d} {'':8s} {int(tot[2]):9d} {int(tot[3]):10d} {int(tot[4]):10d} {int(tot[5]):9d}")


if __name__ == "__main__":
    main()
