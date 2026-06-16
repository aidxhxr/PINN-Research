"""
Plot PINN results directly from saved checkpoint files.
Works mid-training or after training completes.
Usage: python3 plot_from_checkpoints.py
PNGs are saved in the current directory.
"""
import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import wnt_pinn_trainer as tr
import wnt_model_physics as phys
import wnt_reference_solver as ref

CKPT_DIR = "pinn_checkpoints"


def load_from_checkpoints(subdir, model):
    pattern = os.path.join(CKPT_DIR, subdir, "ckpt_win_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  no checkpoints found in {subdir}/")
        return None

    ts, zs, losses = [], [], []
    for f in files:
        d = np.load(f)
        ts.append(d['t'])
        zs.append(d['z'])
        losses.append(float(d['loss']))

    t = np.concatenate(ts)
    Z = np.concatenate(zs)
    Pp = phys.build_params(model)
    Ba = Z[:, tr.iBa]
    Ci = Z[:, tr.iCi]
    BcatTcf = tr.bctf(torch.tensor(Ba), torch.tensor(Ci), Pp).numpy()

    print(f"  {subdir}: {len(files)} windows, t=[{t[0]:.0f},{t[-1]:.0f}], "
          f"last loss={losses[-1]:.3e}")

    return dict(t=t, win_losses=losses,
                H5=Z[:, tr.iH5], H13=Z[:, tr.iH13],
                M=Z[:, tr.iM], Mi=Z[:, tr.iMi],
                Ca=Z[:, tr.iCa], Ci=Ci, Ba=Ba,
                P=Z[:, tr.iP], Bp=Z[:, tr.iBp],
                BcatTcf=BcatTcf, V=Z[:, tr.iV],
                Di=Z[:, tr.iDi], Db=Z[:, tr.iDb],
                Da=Z[:, tr.iDa], X=Z[:, tr.iX],
                Nr=Z[:, tr.iNr], R=Z[:, tr.iR])


print("Loading checkpoints...")
sol_const = load_from_checkpoints("const", "const")
sol_dyn = load_from_checkpoints("dyn", "dyn")

print("Computing reference solutions...")
def reference_dict(model):
    t, U, P = ref.solve_reference(model)
    BcatTcf = ref.bcat_tcf(U, P)
    i = phys
    return dict(t=t,
                H5=U[:, i.iH5], H13=U[:, i.iH13], M=U[:, i.iM], Mi=U[:, i.iMi],
                Ca=U[:, i.iCa], Ci=U[:, i.iCi], Ba=U[:, i.iBa], P=U[:, i.iP],
                Bp=U[:, i.iBp], BcatTcf=BcatTcf, V=U[:, i.iV], Di=U[:, i.iDi],
                Db=U[:, i.iDb], Da=U[:, i.iDa], X=U[:, i.iX], Nr=U[:, i.iNr],
                R=U[:, i.iR], U=U)

ref_const = reference_dict('const')
ref_dyn = reference_dict('dyn')

print("Plotting...")
figs = tr.plot_all(sol_const, sol_dyn, ref_const=ref_const, ref_dyn=ref_dyn, show=False)

names = ['hox_comparison', 'wnt_comparison', 'hox_components', 'wnt_dynamics']
for fig, name in zip(figs, names):
    path = f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")

print("Done.")
