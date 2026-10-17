# Scripts

| script | purpose |
|---|---|
| `run.py` | Launch a validated experiment profile: `python scripts/run.py configs/<profile>.json` (same as `wnt-pinn run`). Writes a new timestamped directory under the git-ignored root `runs/`; `--resume runs/<run-id>` continues an interrupted run. |
| `benchmark_kernels.py` | Compare the eager, Triton and compiled integral-residual kernels on a profile. |
| `benchmark_reduction.py` | Time the fused final reduction against the eager reference. |

Run long jobs inside tmux. Read [the experiment protocol](../docs/experiments.md)
for configuration, manifests, resource settings and resume behavior.

The runner calls the maintained integral and hybrid trainers in their experiment
folders. `configs/integral.json` and `configs/hybrid_myc.json` are full training
profiles; `configs/smoke_cpu.json` is a small execution check and provides no
scientific result.
