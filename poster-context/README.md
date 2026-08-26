# poster-context — source material for the two research posters

Context pack for building **two independent posters** from the PINN-Research
work, per the professor's ask: one basic, one higher-level. Each subfolder is
self-contained — every number in these files was checked against the notes in
`notes/` and the paper (`research-paper/paper.tex`) as of 2026-08-26.

## Hard constraint

**Maximum poster size: 48" × 36", either orientation** (landscape 48w×36h or
portrait 36w×48h). See `00-design-guidelines.md`.

## The split

| | Poster A — basic | Poster B — advanced |
|---|---|---|
| folder | `poster-A-basic/` | `poster-B-advanced/` |
| story | Can a PINN solve and invert a stiff 7-ODE cancer model? | When parameters (or whole terms) are *not* recoverable — why, how to prove it, and how to design experiments that fix it |
| content | model + significance → forward PINN (spectral bias, sparse-data hybrid) → inverse PINN (the ~8/36 ceiling, its two causes, the integral-residual fix, the ODE-fit) | model + significance (own, shorter version) → identifiability three ways (FIM, profile likelihood, Bayesian posterior) → neural-mechanistic hybrid (UDE) and the anchor-visitation result |
| headline | Fourier-feature PINN: 1.06% forward error dense / 2.41% from 40 points; inverse recovery 37 → 50 of 144 params via a derivative-free residual | An `f(0)=0` constraint is inert where data never visit; one extra siRNA (2.2% → 0.0% error) and a training-free design table that predicts prospectively |

**Independence:** both posters carry their own model-introduction panel and
their own references. Neither cites the other. Poster B does *not* assume the
audience saw the forward/inverse chain — its intro panel states the recovery
ceiling as one sentence of motivation and moves on.

Rationale for this split (vs. "Bayesian + mechanistic" only): the Bayesian
work's honest result is an *identifiability verdict*, and it lands much harder
next to the FIM/profile-likelihood evidence and the hybrid/UDE line — together
they are one story: "which parts of a mechanistic model can data actually pin
down, and how do you buy back the parts it can't." The forward + inverse PINN
chain is the methods story and stands alone cleanly.

## Files

```
poster-context/
├── README.md                     ← this file
├── 00-design-guidelines.md       ← size, layout, palette, fonts, figure rules
├── poster-A-basic/
│   ├── 00-outline.md             ← panel-by-panel layout plan
│   ├── 01-model-and-significance.md
│   ├── 02-forward-pinn.md
│   ├── 03-inverse-pinn.md
│   └── 04-figures.md             ← figure manifest with repo paths
└── poster-B-advanced/
    ├── 00-outline.md
    ├── 01-model-and-significance.md
    ├── 02-identifiability-and-bayesian.md
    ├── 03-hybrid-ude-anchor.md
    └── 04-figures.md
```

## Ground rules carried over from the repo

- Numbers come from files on disk, not memory: SA tables in
  `PINN/sa_results/`, forward errors in `runs/*/forward_error_table.{txt,json}`,
  hybrid results in `PINN-hybrid-ude/runs/`.
- **Never quote** `bayes_summary.md`'s "36/36 IDENT" (miscalibration artifact)
  or the original-box SA "thetaP dominates globally" (divergent-sample
  artifact). The honest replacements are in the poster-B files.
- Single seed (42) everywhere → no error bars on recovery counts; the only
  measured noise figure is the hybrid screen's ±0.5–0.9 pp restart spread.
- Figures copied out of `runs/` are snapshots — re-copy after any regen wave.
