# 15 · Neural–mechanistic hybrids: learned regulatory terms and the experiments that identify them

A hybrid model (a universal differential equation) keeps the mechanistic ODE
but replaces one closed-form regulation, for example the β-catenin→MYC Hill
term, with a small network `f(u)` beside the equation's basal rate `a`. What
does that cost the mechanistic parameters, and what restores them? On the
observed support `a + f(u) = (a + c) + [f(u) - c]` for any constant `c`, so the
anchor `f(0) = 0` separates `a` from `f` only if some condition drives the
regulator `u` to zero. No condition of the ten-condition base design does. The
network learns the function well and loses the basal rate under every
parameterisation tried. A knockdown that visits the anchor, in a condition that
still contains the basal term, removes most of the loss; information-matched
conditions of the same size do not. A training-free Fisher bound of the hybrid
equation and a prospective test of a design table agree with this.

| | |
|---|---|
| **Status** | complete |
| **Period** | 2026-07 to 2026-10 |
| **Builds on** | [10 · inverse-integral-residual](../10-inverse-integral-residual/) (training reused unchanged), [11 · fisher-information](../11-fisher-information/) |
| **Paper** | [paper](../../paper/main.pdf): Sections 2.8–2.10, 4.7–4.8; Figs 8–11; S2–S4 Tables; S1 Text |

## Result

Synthetic data with known truth. "Basal error" is the relative error of the
hosting equation's basal parameter; recovered means error below 10%.

- **The learned term loses its host's basal rate** (full pipeline, Fig 8A).
  The β-catenin→MYC network learns the function to 6–10% NRMSE, but `a_M` is
  32.3% and 52.1% off in the two low-WNT regimes, where the mechanistic model
  recovers it to 0.4% and 4.6%. In Early Adenoma the learned function sits a
  near-constant 0.094 above the true Hill term over the support (`b ≥ 0.11`),
  and `a_M` is low by the same 0.094. Shared-parameter counts (registry
  `hybrid_comparison`): MYC 43 vs control 48 of 136 regime–parameter pairs;
  RA→HOXA5 53 vs 51 of 136; the calibrated, frozen APC-loss law 53 vs 52 of
  140. Removing the anchor (`_nc`) changed the totals by at most one pair.
- **No network shape closes the gap** (equation-local screen, Fig 8C,D). Over
  the seven production edges the gated network leaves the basal parameter
  16–50% off and the monotone exact-anchor network (`sc`) 14–74%, against
  7–25% for the mechanistic control. In the low-WNT regimes the control
  recovers all 14 basal parameters to below 7.5%; the two networks recover 4 of
  28. On the MYC edge, where the control recovers `a_M` to 0.0–0.1%, `gated`,
  `sc` and `lin` leave it 6.5–38% off, `lin_mono` 200–400%, and the
  oracle-bounded `sc_bounded` 10.5–17.9%. Modulator edges (`f(u)·y`) have no
  constant to absorb: CYP26A1⊣RA and HOXA5⊣β-catenin are learned to 0.0–1.7%
  functional error with every surviving equation parameter recovered in the
  low-WNT regimes. The other two modulators degrade to 36–95% functional error
  in the high-WNT regimes (`runs/20260921_aggregations/atlas_4regime.txt`).
- **Visiting the anchor, not adding data** (Fig 10). A WNT knockdown
  `k_W = 0.02` plus a retinoid-free medium (RA → 0 exactly) takes the two
  RA-driven edges from 10.9–33.5% to 0.09–1.1% basal error. MYC, whose
  regulator only approaches zero, improves up to 30-fold but stops at 2.5–2.9%;
  β-catenin→CYP26A1 does not improve. Two information-matched conditions
  change the basal error by a factor of 0.76–1.54 (equation parameters 16 → 15
  of 28); the depletion pair recovers 26 of 28.
- **At fixed effort** (11 conditions at every dose, Fig 11A,B). At `k = 0` the
  single WNT knockdown leaves `a_M` at 1.45% (Normal) and 2.25% (Severe APC
  Loss); knocking down both β-catenin sources (`k_W = k_13 = 0`), one extra
  siRNA, gives 0.003% and 0.002%. Means over three disjoint seed sets of five
  restarts; seed spread 0.2 and 0.8 points.
- **Full pipeline** (S3 Table). With that double knockdown and the
  retinoid-free medium, the MYC basal error falls from 10.3/11.7/17.4/16.4%
  (single knockdown) to 1.4/0.6/0.6/0.7%, against a control at 0.2–0.3%; the
  hybrid recovers 58 of 136 shared pairs, the control 62.
- **Prospective test of the design table** (Fig 11C, S4 Table). For
  MYC→HOXA13 the prescribed `mycKO` took the basal error from 12.8% to 0.1%
  (Normal) and 95.6% to 0.05% (Severe); the near-miss `bcatKO` left 51.5% and
  59.8%. For HOXA13→β-catenin the prediction failed (16.2% in all three Normal
  arms): the protocol that reaches its anchor sets `k_W = 0` and deletes `W`,
  the edge's basal parameter. Hence the two-clause rule: *reach the anchor,
  with a condition that still contains the basal term.*
- **Training-free Fisher bound** (Fig 9). Under the base design the network
  loses a factor of 56 to 4×10⁵ in basal-rate precision relative to the
  mechanistic control. Information-matched conditions change the bound by
  0.90–1.00; a depletion pair reaching the RA anchors shrinks it by
  1.8–6.1×10⁻⁴, to 1.0–1.4 times the control. Over 116 distinct screen fits its
  rank correlation with the measured basal error is +0.79 (anchor ratio alone:
  +0.55).

## Method

- **Learned term.** A shared 2×5 tanh network (46 weights, Glorot-uniform,
  final bias zero), non-negative with `f(0) = 0`; the replaced `(eta, kappa)`
  leave the unknowns (36 → 34). L2 penalty `1e-8` on the mean squared weight
  (`HYBRID_WD`), calibrated to this loss scale. The network is built after the
  state networks so control and hybrid share state-network initialisations.
- **Registry.** `config.HYBRID_TERMS` and `hybrid.py` define 13 learnable edges
  (S2 Table: 7 production, 4 modulator, 1 ratio, the APC-loss law) and five
  constraint forms (`HYBRID_PARAM`): `gated`, `sc`, `sc_bounded` (oracle
  asymptote), `lin`, `lin_mono`.
- **Full pipeline** (`run_hybrid.py`). The integral-residual inverse PINN of
  [10](../10-inverse-integral-residual/) (relative weights, fixed collocation,
  three starts), scored against a mechanistic control on the shared unknowns.
  The APC-loss law is first calibrated at known severities
  (`calibrate_apc_mutation.py`) and then frozen, so it has extra information.
- **Equation-local screen** (`screen_terms.py`). Only the hosting equation is
  fitted, from noise-free reference states (stride-8 grid), all other
  parameters at truth, the equation's own from 1.5× truth; network and closed
  form use the same seeds and budget (3000 Adam, 300 L-BFGS; lowest-residual
  restart kept).
- **Designs.** `HYBRID_DEPLETION=1` adds `wntKO` (`kW = 0.02`) and `raKO`;
  `HYBRID_DEPLETION=bcat` adds `bcatKO` (`kW = k13 = 0`) and `raKO`;
  `HYBRID_INFOCTL=1` adds the information-matched `midATRA` and `wntPulse2`.
  The dose-response appends one graded depletion
  `k ∈ {1, 0.3, 0.1, 0.03, 0.01, 0}` in three arms (`ra`, `wnt`, `bcat`).
- **Anchor ratio and design table.** `rho_u = min u / max u` needs only
  reference trajectories. `anchor_reach.py` ranks knockdown protocols by
  laboratory cost, names the cheapest that reaches each anchor, and rejects
  protocols that zero the basal parameter's own knob.
- **Hybrid Fisher bound** (`hybrid_fim.py`). Fisher information of the screen's
  residual model with the learned term (the screen's own networks, fitted to
  the true term) profiled out; reports `Gamma = sd(log a)/sigma_r`, checked by
  finite differences and 200 Monte Carlo refits (`--verify`).

## Experiments in this folder

"In repo" marks run directories whose result files are tracked; the others
hold checkpoints and stay local, with tracked summaries where the paper uses
them.

| run directories (`runs/`) | question | in repo | paper |
|---|---|---|---|
| `20260726_195830_control`, `20260726_214316_ra_h5`, `*_ra_h5_nc`, `20260727_012448_bm_myc`, `*_bm_myc_nc` | Round 1, full pipeline: what does a learned term cost; does the anchor matter? | `bm_myc` only | Fig 8A |
| `20260728_232743_apc_calibration`, `20260728_233450_apc_mutation_frozen` | APC-loss law calibrated across severities, then frozen | no | registry |
| `20260801_204046_screen_atlas` (Normal), `20260802_screen_atlas_rest` | Edge atlas: 12 edges × 4 regimes × {`gated`, `sc`}, 2 restarts | yes | Fig 8C |
| `20260921_screen_5form` | Five constraint forms, MYC edge, 3 restarts | yes | Fig 8D |
| `20260801_depletion_ab`, `20260801_infoctl` | Depletion pair vs base design; vs information-matched pair (1 restart) | yes | Fig 10 |
| `20260802_anchor_reach` | Training-free design table | yes | S4 Table |
| `20260802_dose_response`, `20260802_dose_starts5`, `20260921_dose_seeds`, `20260921_dose_sc` | Dose-response; Severe at 5 restarts; k = 0 over three seed sets; `sc` form | yes (`dose_sc`: summary) | Fig 11A,B |
| `20260802_protocol_test` | Prospective test of the design table | yes | Fig 11C |
| `*_control_dep`, `*_bm_myc_dep`, `*_ra_h5_dep`, `*_control_depb`, `*_bm_myc_depb` | Full pipeline with depletion (12 conditions, 3 starts) | summaries | S3 Table |
| `20261017_hybrid_fim` | Fisher bound of a learned term's basal parameter | yes | Fig 9 |
| `20260921_aggregations` | Text tables of the runs above | yes | S3 Table |
| `20260921_odefit_thetaP`, `20260801_negctl_*` | Superseded `thetaP` fit; inconclusive first negative control | summary / no | — |

## Contents

| path | purpose |
|---|---|
| `hybrid.py`, `config.py` | Network constructions and `true_term`; model, conditions and `HYBRID_*` switches (read at import) |
| `run_hybrid.py`, `run_hybrid.sh`, `training.py` | Full pipeline (one process per regime) |
| `calibrate_apc_mutation.py`, `run_apc_pipeline.sh` | Two-stage APC-loss hybrid |
| `screen_terms.py`, `dose_response.py`, `protocol_test.py` | Equation-local screen, dose-response, prospective test |
| `anchor_report.py`, `anchor_doses.py`, `anchor_reach.py` | Training-free anchor ratios, dose grids, design table |
| `hybrid_fim.py`, `hybrid_fim_analysis.py` | Hybrid Fisher bound and its analysis |
| `aggregate_hybrid.py`, `aggregate_screen.py`, `compare_screens.py`, `aggregate_dose.py`, `aggregate_protocol.py`, `plot_learned_term.py` | Scoring (the dose and protocol scripts score the registered predictions) and figures |
| `test_hybrid_terms.py` | Learned-term tests, run from `tests/test_legacy_hybrid.py` |
| `run_*.sh`, `chain_bm_myc.sh` | Launchers; each header states what it ran (`run_queue_shape.sh` has no recorded run) |
| `run_boost.py`, `main.py`, `recover_odefit.py`, `*identifiability.py`, plotting | Inherited from [10](../10-inverse-integral-residual/) |
| `odes.py`, `model.py`, `residual.py` | Re-exports from [`wnt_pinn`](../../src/wnt_pinn/) |

## Reproduce

Install the package first (`python -m pip install -e '.[ml]'`,
[installation](../../docs/installation.md)). `HYBRID_*` variables must be set
before Python starts. Screens and the Fisher bound need `HYBRID_TERM=none`
(`dose_response.py` refuses to start without it). Run long jobs in a
persistent terminal session; outputs go to `runs/`.

```bash
cd experiments/15-hybrid-ude

# Full pipeline (GPU, about 2 GPU-hours per variant; PINN_CONCURRENCY limits parallel regimes)
bash run_hybrid.sh control 3            # mechanistic control
bash run_hybrid.sh bm_myc 3             # also ra_h5, *_nc, <edge>__<form>
bash run_queue_depletion.sh 3           # control, bm_myc, ra_h5 + wntKO, raKO
bash run_queue_depletion_bcat.sh 3      # control, bm_myc + bcatKO, raKO
bash run_apc_pipeline.sh 3              # calibrate, then freeze the APC-loss law (about 2 h)
python3 aggregate_hybrid.py runs/<hybrid> runs/<control>

# Equation-local screens (exact states; GPU if available)
export HYBRID_TERM=none
python3 screen_terms.py --out runs/<ts>_atlas --terms all --params gated,sc \
    --starts 2 --stride 8 --adam 3000 --lbfgs 300      # 3 regimes took about 7 h
OUT=runs/<ts>_5form bash run_5form_screen.sh           # 2.7 h
python3 dose_response.py --arm bcat --edge bm_myc --out runs/<ts>_dose   # arms ra (ra_h5), wnt, bcat
python3 protocol_test.py --edge m_h13 --protocols none,bcatKO,mycKO --out runs/<ts>_protocol
python3 aggregate_screen.py runs/20260801_204046_screen_atlas runs/20260802_screen_atlas_rest
python3 aggregate_dose.py runs/20260802_dose_response

# Training-free (CPU)
python3 anchor_reach.py
python3 hybrid_fim.py --out runs/<ts>_hybrid_fim       # 8 workers, about 13 min; add --verify for FD + MC
python3 hybrid_fim_analysis.py runs/<ts>_hybrid_fim
```

Some launchers pass a cached `refs.pkl` from a local run; `screen_terms.py`
regenerates references when it is missing. `run_protocol_test.sh` and
`run_atlas_rest.sh` first wait for another log (`WAIT_LOG`), so call the Python
drivers directly. Hybrid variants also run through the versioned runner
(`configs/hybrid_myc.json`, [running experiments](../../docs/experiments.md)).
Figure-to-run mapping: [REPRODUCE.md](../../paper/REPRODUCE.md).

## Notes

- **Exact states.** The screen has no measurement noise or state-estimation
  error, so its absolute errors bound the full pipeline from below; the
  full-pipeline confirmations agree in direction, with larger errors.
- **Restart noise.** The same MYC cell (Advanced, gated) reads 68.3% in the
  two-restart atlas and 18.9% in the three-restart five-form run; lowest
  residual is not always lowest parameter error. Compare within an experiment.
  The depletion and information-matched comparisons used one restart; the
  replicated contrast is the k = 0 dose pair.
- **Registered predictions, as scored** (S1 Text). Dose-response: two of six
  passed as registered, three after Severe was rerun with five restarts; the
  falsifier failed in Normal (single knockdown reached 1.3%, predicted floor
  2%). Prospective test: two of five passed. The second clause of the rule came
  after the HOXA13→β-catenin failure; the Fisher bound was computed after all
  fits. `mycKO` fixed the MYC→HOXA13 parameter while its functional error rose
  (57% → 68% Normal, 12% → 81% Severe).
- **Fisher bound.** Where the anchor is not visited, its absolute value depends
  on the SVD rank tolerance by orders of magnitude, and it overstates the
  Monte Carlo scatter of the MYC edge 23-fold; where the anchor is visited it is
  calibrated within 3%. Only orderings, design ratios and rank correlations
  (stable to 0.04) are interpreted.
- **Stage-3 frozen refinement is effectively a no-op** here; see
  [decision 0003](../../docs/decisions/0003-hybrid-protocols.md).
- **Provenance.** The Normal atlas cells exist only in
  `runs/20260801_204046_screen_atlas/screen.log` (parsed by
  `aggregate_screen.py`). Registry record `constraint_screen` preserves an
  earlier one-start Normal five-form table (13.8–202.8%) without a recovered
  run mapping; `runs/20260921_screen_5form` is the complete rerun.
- **`thetaP` by fit.** `run_odefit_thetaP.sh` and
  `runs/20260921_aggregations/odefit_thetaP.txt` are a default-tolerance fit,
  superseded by the converged fits of [17 · sensitivity-pinn](../17-sensitivity-pinn/)
  used in Fig 7B.
- **Scope.** Knockdowns are model interventions with exactly known
  efficiencies, not validated laboratory protocols; see the
  [registry](../../results/registry.json) (`hybrid_comparison`,
  `anchor_interventions`).
