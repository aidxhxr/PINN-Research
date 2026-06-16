## Plotting results

### After training finishes

`PINN_solution.py` auto-saves four PNGs into `PINN/` when training completes:

- `hox_comparison.png`
- `wnt_comparison.png`
- `hox_components.png`
- `wnt_dynamics.png`

Nothing extra needed — open them in VS Code's file explorer or `scp` them to your local machine.

### Mid-training (from checkpoints)

Run this from a second tmux pane at any point during training:

```bash
cd ~/PINN-Research/PINN
python3 plot_from_checkpoints.py
```

It reads all `.npz` checkpoint files saved so far, reconstructs the solution up to the current window, and saves the same 4 PNGs. Re-run it whenever you want updated plots.

### Viewing PNGs on a headless VM

- **VS Code**: click the file in the explorer — it renders inline.
- **scp**: `scp user@vm:~/PINN-Research/PINN/*.png .`
