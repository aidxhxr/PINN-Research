# Experiments

Use `python experiments/run.py configs/smoke_cpu.json` from a registered tmux
session after installing the project. Read [the experiment protocol](../docs/experiments.md)
for configuration, manifests, resource settings and resume behavior.

The runner calls the maintained integral and hybrid trainers in their existing
directories. Saved historical runs keep their original paths and seed protocol.
`configs/integral.json` and `configs/hybrid_myc.json` are full training profiles;
the smoke profile is a small execution check and provides no scientific result.
