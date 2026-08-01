# Physics-Informed Neural Networks for a WNT–RA–HOX Cancer-Signaling Model

We use **Physics-Informed Neural Networks (PINNs)** to solve the forward and
inverse problems of a reduced gene-regulatory network that governs the
WNT / retinoic-acid (RA) / HOX axis in colorectal-cancer signaling — and we
push on the harder, more honest question underneath it: *given noisy trajectory
data, which of the model's biological parameters can we actually recover, and
which are fundamentally non-identifiable?*

![WNT–RA–HOX regulatory network](network-diagram/preview.png)

---

## The problem we're studying

The biology reduces to a **7-state ODE system** — β-catenin, APC, HOXA5,
HOXA13, MYC, retinoic acid, and CYP26A1 — driven by circadian RA forcing and a
smooth ATRA pulse, with WNT drive `W` and APC functionality `θP` as the two
knobs that separate four clinical regimes:

| Regime | Meaning |
|--------|---------|
| **Normal** | healthy tissue, low WNT |
| **Early Adenoma** | early lesion |
| **Advanced Adenoma** | elevated WNT |
| **Severe APC Loss** | saturated WNT, loss of APC control |

The system has **36 kinetic parameters**. Recovering them from a handful of
trajectories is a classically ill-posed inverse problem (condition number
≈ 10⁶), so most of our work is about *measuring* and *beating* that ceiling
rather than pretending it isn't there.

---

## What we built

**Forward PINN.** A Fourier-feature MLP (width 256 × depth 4, GELU) that solves
the ODE system directly. The fast circadian + pulse modes cause spectral bias
in a plain MLP ("everything collapses to lines"), so we use a multi-scale
Fourier time embedding to represent them. We validate against a stiff
`solve_ivp` Radau reference (`rtol=1e-10`, `atol=1e-12`).

**Inverse PINN.** We recover the 36 parameters from trajectory data, scaling
from a 2-parameter proof of concept up to the full set, across all four regimes
and multiple experimental conditions.

**Identifiability analysis.** We don't just report point estimates — we quantify
how trustworthy each recovered parameter is, using three independent lenses
(sensitivity, Fisher information, and profile likelihood) that agree with one
another.

---

## Headline results

**1 — More information beats a better network.** We tested two levers for
raising the recovery count. Shrinking and regularizing the network did nothing
(the architecture is *not* the bottleneck). But adding **parameter-exciting
conditions** — exogenous pulses that directly perturb the WNT and MYC nodes and
light up the "dark" half of the network — lifted the classical ODE-fit recovery
from a **18/17/13/13** baseline to **24/23/21/14** parameters under 10 % error
(Normal / Early Adenoma / Advanced Adenoma / Severe APC Loss).

**2 — The PINN's autodiff-derivative ceiling is breakable.** The inverse PINN
had a lower ceiling (~8/36) than the classical ODE-fit, which we traced to a
biased autodiff `dz/dt` residual. Replacing it with a **derivative-free integral
(trapezoidal) residual** lifted the *PINN itself* from a **10/4/5/7** baseline
to **17/16/10/7** — roughly doubling the recovered count (+24 total) without
falling back to the ODE-fit.

**3 — A structural wall we can see from three directions.** In high-WNT regimes,
β-catenin saturates and `θP` (APC functionality) becomes non-identifiable. We
confirmed this is *practical / conditioning* non-identifiability — not a bug —
independently via:

- a **Fisher Information Matrix** (one hard-null direction per regime; the count
  of near-perfectly-correlated parameter pairs climbs 0 → 2 → 7 → 10 with WNT
  drive; cond(FIM) ≈ 10⁷–10¹⁷), and
- **Raue-style profile-likelihood** identifiability (95 % CIs → IDENT / WEAK /
  NON-IDENT).

As a clean contrast, restricting the problem to the **8 most-identifiable
parameters** gives a well-posed system — cond(FIM) ≈ 10²–10³, zero null
directions, all eight IDENT — showing precisely where the recoverable
information lives.

---

## Repository map

| Folder | What it is |
|--------|-----------|
| `PINN-smaller/forward_pinn_train/` | Forward PINN — supervised interpolation baseline |
| `PINN-smaller/forward-pinn-train-hybrid/` | Forward PINN — honest sparse-data solve (40 obs + IC + physics) |
| `PINN-inverse-solve/` | First inverse PINN, 2 → 36 parameters, single condition |
| `PINN-inverse-better/` | Multi-condition + log-parameterization + adaptive weighting |
| `PINN-inverse-multicond/` | 6 conditions + the classical ODE-fit recovery (`recover_odefit.py`) |
| `PINN-inverse-multicond-excite/` | The **information lever** — WNT/MYC-exciting conditions |
| `PINN-inverse-pinn-boost/` | The **integral-residual** PINN that breaks the derivative-bias ceiling |
| `PINN-fisher-matrix/` | Full 36-parameter Fisher-information identifiability analysis |
| `PINN-fisher-matrix-top8/` | Reduced 8-parameter well-posed contrast |
| `network-diagram/` | The regulatory-network schematic (TikZ source + PDF) |

---

## Reproducing our runs

Each experiment folder ships a `run.sh` that timestamps a `runs/<...>/`
directory, sets `PYTHONPATH`, and streams a training log:

```bash
cd PINN-inverse-pinn-boost
bash run.sh            # writes runs/<timestamp>/train.log
```

The Fisher-matrix analyses run on CPU (`solve_ivp` sensitivities); the PINN
training uses CUDA if available.

---

## Tech stack

**PyTorch** (PINN models, autodiff residuals, CUDA training) · **SciPy /
NumPy** (stiff ODE reference, classical ODE-fit) · **lmfit** &
**`identifiability`** (profile-likelihood CIs) · **SALib** (sensitivity) ·
**Matplotlib** (diagnostics) · **TikZ** (the network schematic).

A pinned environment is in [`requirements-lock.txt`](requirements-lock.txt).

---

*This repository documents an active research effort: forward and inverse PINN
solvers, a systematic study of what makes biological parameters recoverable, and
a rigorous, multi-method identifiability verdict on a real cancer-signaling
model.*
