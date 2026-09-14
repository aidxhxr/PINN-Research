# Physics-informed neural networks for a WNT–RA–HOX model

This repository studies state reconstruction, parameter recovery and learned
regulatory terms in a model of colorectal cancer stemness. It contains forward
and inverse physics-informed neural networks (PINNs), Fisher and Bayesian
analyses, and neural–mechanistic hybrids. The experiments use synthetic data
generated from the model.

## Model

Seven coupled ODEs describe β-catenin, APC, HOXA5, HOXA13, MYC, retinoic acid
and CYP26A1. Retinoic-acid input includes circadian forcing and an ATRA pulse.
The inverse problem has 36 unknown parameters. WNT drive `W` and APC
functionality `θP` define four model regimes:

| Regime | W | θP |
|---|---:|---:|
| Normal | 0.8 | 1.00 |
| Early Adenoma | 1.0 | 0.75 |
| Advanced Adenoma | 1.5 | 0.50 |
| Severe APC Loss | 2.0 | 0.25 |

![WNT–RA–HOX regulatory network](docs/figures/network.png)

The state order is `[b, p, h5, h13, m, r, c]`. Reference trajectories use
SciPy's Radau solver over dimensionless time `[0, 150]`, with the default ATRA
pulse on `[40, 88]`. Setting `DR=0` removes the pulse while retaining the
circadian input.

## Forward and inverse PINNs

The forward PINN maps Fourier features of time to seven state trajectories.
Training combines observation, initial-condition and ODE-residual losses.
With 40 observation times, its mean relative L2 error is 2.41% across seven
states and four regimes. A separate supervised network trained on 100 labels
has 1.06% error; the two runs use different training objectives.

![Forward PINN and Radau reference trajectories](docs/figures/forward_pinn.png)

Solid curves show the reference solution; dashed curves show the PINN.
The shaded interval marks ATRA treatment.

The inverse PINN fits one state network per experimental condition and shares
the biological parameters across conditions. The following runs each use ten
conditions, including WNT and MYC perturbations. Counts are parameters within
10% of their true values.

| Inverse PINN | Normal | Early | Advanced | Severe | Total / 144 |
|---|---:|---:|---:|---:|---:|
| Autodiff residual | 16 | 9 | 6 | 6 | 37 |
| Integral residual | 17 | 16 | 10 | 7 | 50 |

The integral implementation uses a trapezoidal residual, relative state
weighting, fixed collocation and multiple starts. These changes were made
together, so the comparison does not isolate the effect of the residual.
Accurate trajectories can still accompany large parameter errors.

![Eight best-recovered parameters in an inverse-PINN run](docs/figures/inverse_recovery_best8.png)

The figure shows a selected subset of eight parameters; the table reports
recovery over all 36.

## Identifiability and uncertainty

Fisher information, profile likelihood and Bayesian sampling examine which
parameter combinations the observations constrain. High WNT weakens
sensitivity to APC functionality. Each full Fisher matrix has one near-null
direction under the analysis threshold. An eight-parameter subset has no
near-null directions and condition numbers of about 100–665 when the other
28 parameters are fixed at truth. This is a local sensitivity result;
PINN recovery of that subset has not been tested.

![Fisher-information eigenvalue spectra across regimes](docs/figures/fim_spectra.png)

The forward Bayesian model samples network weights with Hamiltonian Monte
Carlo (HMC). The inverse model freezes the state networks and samples the
36 biological parameters. The inverse HMC runs fail effective-sample-size and
coverage checks, so their interval widths are unreliable.

![Bayesian forward predictions for β-catenin](docs/figures/bayesian_forward_bands.png)

Forward posterior bands for β-catenin. Coverage and sampling diagnostics are
saved with each run.

## Neural–mechanistic hybrids

A small neural network replaces one regulatory term inside the ODE system.
For β-catenin activation of MYC, the learned term has two hidden layers of
five tanh units and satisfies `f(0)=0`. The remaining biological parameters
and state networks are fitted jointly with it.

MYC activation error is 6.2–9.5% NRMSE over the observed regulator ranges.
On the 34 parameters shared by the hybrid and mechanistic control, recovery
within 10% of truth falls from 48 to 43 out of 136 parameter–regime pairs.
Function accuracy and parameter recovery therefore need separate evaluation.

![Learned regulatory term compared with the true mechanism](docs/figures/hybrid_learned_term.png)

An anchor at `f(0)=0` can leave basal production ambiguous when the data never
approach zero. On the observed range, an offset in the learned function can
compensate for an error in the basal term. In an equation-level MYC screen,
adding HOXA13 feedback suppression to WNT depletion reduces Normal-regime
basal error from 2.170% to 0.021%, with 11 conditions in each arm.

![Basal-parameter error across depletion doses](docs/figures/hybrid_dose_response.png)

These intervention screens fit one equation to exact states, with parameters
outside that equation fixed at truth. Basal recovery can improve while
function recovery worsens. Validation with noisy, sparse state estimates
remains to be done.

## Posters

Both posters are 48 × 36 inches and include editable LuaLaTeX sources.

| Poster | Authors | Files |
|---|---|---|
| PINN dynamics and parameter recovery | Nathaniel Kim · Pascal Kataboh | [PDF](research-poster-latex/poster.pdf) · [Sources and build instructions](research-poster-latex/) |
| Neural–mechanistic hybrids | Amirkhan Aidarkhan · Pascal Kataboh | [PDF](research-poster-latex-hybrid/poster.pdf) · [Sources and build instructions](research-poster-latex-hybrid/) |

The [first poster's source notes](research-poster-latex/SOURCES.md) and
[second poster's source notes](research-poster-latex-hybrid/SOURCES.md)
document the runs, metrics and limitations behind the reported results.

## Repository layout

| Directory | Contents |
|---|---|
| `PINN-smaller/forward-pinn-train-hybrid/` | Sparse-data forward PINN |
| `PINN-smaller/forward_pinn_train/` | Supervised forward baseline |
| `PINN-inverse-solve/` | Historical inverse baseline; preserved unchanged |
| `PINN-inverse-multicond/` | Multi-condition inverse PINN and classical ODE fitting |
| `PINN-inverse-multicond-excite/` | WNT and MYC perturbation experiments |
| `PINN-inverse-pinn-boost/` | Integral-residual inverse PINN |
| `PINN-fisher-matrix/`, `PINN-fisher-matrix-top8/` | Full and reduced Fisher analyses |
| `PINN-bayesian/`, `PINN-forward-bayesian/` | Inverse and forward HMC experiments |
| `PINN-hybrid-ude/` | Learned regulatory terms and intervention screens |
| `PINN/` | Original notebooks and sensitivity-analysis tables |
| `network-diagram/` | TikZ regulatory schematic |
| `research-paper/` | LaTeX manuscript |
| `docs/figures/` | Figures used in this README |

## Running experiments

The Python environment is recorded in
[`requirements-lock.txt`](requirements-lock.txt). Training uses PyTorch and
CUDA when available. Run experiments in tmux and record each session in
[`AGENTS.md`](AGENTS.md) immediately after launch.

From the repository root, with the Python environment active, launch the
integral inverse PINN with two starts per regime:

```bash
tmux new-session -d -s pinn_inverse -c "$PWD/PINN-inverse-pinn-boost"
tmux send-keys -t pinn_inverse 'bash run_boost.sh integral 2' C-m
tmux attach -t pinn_inverse
```

The runner creates `PINN-inverse-pinn-boost/runs/<timestamp>_integral/` with
per-regime logs, checkpoints and a summary. It launches all four regimes in
parallel; CPU thread allocation is set in `run_boost.sh`.

For a MYC hybrid run with three starts, also from the repository root:

```bash
tmux new-session -d -s pinn_hybrid -c "$PWD/PINN-hybrid-ude"
tmux send-keys -t pinn_hybrid 'bash run_hybrid.sh bm_myc 3' C-m
tmux attach -t pinn_hybrid
```

In `PINN-hybrid-ude/`, `anchor_report.py` checks the regulator ranges covered
by the reference data, and `screen_terms.py` fits individual equations.
Its `--help` lists the available screening options.

## Author

Amirkhan Aidarkhan, Swarthmore College.
Contact: <aaidark1@swarthmore.edu> · <amirkhanaidarkhan06@gmail.com>.
