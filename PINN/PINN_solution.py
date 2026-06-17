
import sys
sys.path.insert(0, '/home/jovyan/.local/lib/python3.12/site-packages')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # no display; must come before pyplot import
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

import wnt_model_physics as phys
import wnt_reference_solver as ref
import wnt_pinn_trainer as tr

print('device:', tr.DEVICE)
print('trainer using stub physics:', tr.USING_STUB, '(should be False)')

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

print('\nComputing reference solutions...')
ref_const = reference_dict('const')
print('const reference:', ref_const['U'].shape, 'finite:', np.isfinite(ref_const['U']).all())
ref_dyn = reference_dict('dyn')
print('dyn   reference:', ref_dyn['U'].shape,   'finite:', np.isfinite(ref_dyn['U']).all())

# ---------------------------------------------------------------------------
# 2. Cross-check: residual at SciPy solution
# ---------------------------------------------------------------------------
def residual_at_reference(model, refd):
    P = phys.build_params(model)
    t = refd['t']; U = refd['U']
    dU = np.gradient(U, t, axis=0)
    sl = slice(2, len(t) - 2)
    tt = torch.tensor(t[sl]).reshape(-1, 1)
    zz = torch.tensor(U[sl])
    dz = torch.tensor(dU[sl])
    re, ri = phys.residuals(tt, zz, dz, P)
    res = torch.cat([re, ri], 1).abs()
    print(f'[{model}] max|res|={res.max().item():.2e}  '
          f'median|res|={res.median().item():.2e}')
    per_eq = res.mean(0)
    worst = torch.argsort(per_eq, descending=True)[:5]
    print('   worst eqs:', [(int(k), f'{per_eq[k].item():.1e}') for k in worst])

print('\nCross-checking physics residuals...')
residual_at_reference('const', ref_const)

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
FULL = dict(win_len=500.0, iters=4000, lr=2e-3, n_col=4096,
            n_out=300, lbfgs_iters=80, max_windows=None)

cfg = FULL
CKPT_DIR = "pinn_checkpoints"

print('\nTraining const model...')
pinn_const = tr.solve_pinn('const', gamma1=1.0, checkpoint_dir=f"{CKPT_DIR}/const", **cfg)
print('const window losses:', [f'{l:.2e}' for l in pinn_const['win_losses']])

print('\nTraining dyn model...')
pinn_dyn = tr.solve_pinn('dyn', gamma1=1.0, checkpoint_dir=f"{CKPT_DIR}/dyn", **cfg)
print('dyn window losses:', [f'{l:.2e}' for l in pinn_dyn['win_losses']])

# Save results so they survive even if plotting fails
np.savez('pinn_results.npz',
         t_const=pinn_const['t'], t_dyn=pinn_dyn['t'],
         win_losses_const=pinn_const['win_losses'],
         win_losses_dyn=pinn_dyn['win_losses'])
print('\nResults saved to pinn_results.npz')

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
print('Plotting...')
figs = tr.plot_all(pinn_const, pinn_dyn,
                   ref_const=ref_const, ref_dyn=ref_dyn,
                   show=False)

names = ['hox_comparison', 'wnt_comparison', 'hox_components', 'wnt_dynamics']
for fig, name in zip(figs, names):
    path = f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  saved {path}')

print('\nDone.')
