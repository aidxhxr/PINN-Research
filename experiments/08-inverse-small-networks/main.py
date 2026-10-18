import torch

from config import DEVICE, REGIMES, CONDITIONS, UNKNOWN
from reference import generate_references
from training import train_inverse
from plotting import (plot_states, plot_param_convergence,
                      plot_recovery_bars, plot_param_error, plot_losses,
                      print_summary, print_sensitivity,
                      report_identifiability)


def main(T=150.0):
    print("=" * 60)
    print(f"Inverse PINN (multicond-better) — recover {len(UNKNOWN)} parameters")
    print("  smaller weight-decayed nets + alternating two-timescale opt")
    print(f"  {len(CONDITIONS)} experimental conditions: "
          f"{[c['name'] for c in CONDITIONS]}")
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name()}")
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {gb:.1f} GB")
    print(f"Threads: {torch.get_num_threads()}")
    print("=" * 60)

    print("\n[1/3] Reference trajectories (scipy Radau, per condition) ...")
    refs = generate_references(T=T, n_pts=5000)

    print("\n[2/3] Inverse training (shared params over conditions) ...")
    solutions, all_hist, observations = {}, {}, {}
    recovered, true_params = {}, {}
    for name in REGIMES:
        print(f"\n── {name} ──")
        sol, params, hist, obs, true_vals = train_inverse(
            name, refs[name],
            T=T,
            width=128, depth=3,        # lever #1: ~207k -> ~38k net params
            n_fourier=16, fourier_sigma=4.0,
            n_colloc=10_000,
            n_data=150,            # denser -> pins the state net to the true traj
            noise_std=0.002,       # lower noise -> cleaner net derivative
            adam_epochs=2500,
            lbfgs_steps=600,
            param_refine_steps=500,    # Stage 3: gradient-match on frozen nets
            param_refine_colloc=10_000,
            lr=1e-3,
            lr_param=5e-3,
            net_weight_decay=1e-4,     # lever #1: regularise the net (anti-overfit)
            net_steps=5, par_steps=5,  # lever #3: alternating two-timescale opt
            lam_data=1.0,
            lam_phys=1.0,
            lam_ic=20.0,
            adaptive_weights=True,
        )
        solutions[name]    = sol
        all_hist[name]     = hist
        observations[name] = obs
        recovered[name]    = params.values()
        true_params[name]  = true_vals

    print("\n[3/3] Plotting + diagnostics ...")
    plot_states(solutions, refs, observations)
    plot_param_convergence(all_hist, true_params)
    plot_recovery_bars(recovered, true_params)
    plot_param_error(all_hist, true_params)
    plot_losses(all_hist)
    print_summary(recovered, true_params)
    sens = print_sensitivity(refs)
    report_identifiability(recovered, true_params, sens)

    return solutions, refs, all_hist, recovered, true_params


if __name__ == "__main__":
    main()
