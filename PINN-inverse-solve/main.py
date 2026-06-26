import torch

from config import DEVICE, REGIMES, UNKNOWN
from reference import generate_references
from training import train_inverse
from plotting import (plot_states, plot_param_convergence,
                      plot_recovery_bars, plot_losses, print_summary)


def main(T=150.0):
    print("=" * 60)
    print(f"Inverse PINN - recover {len(UNKNOWN)} parameters "
          f"for the 7-ODE WNT-RA-HOX model")
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name()}")
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {gb:.1f} GB")
    print(f"Threads: {torch.get_num_threads()}")
    print("=" * 60)

    print("\n[1/3] Reference / ground-truth trajectories (scipy Radau) ...")
    refs = generate_references(T=T, n_pts=5000)

    print("\n[2/3] Inverse training (joint state net + unknown params) ...")
    solutions, all_hist, observations = {}, {}, {}
    recovered, true_params = {}, {}
    for name in REGIMES:
        print(f"\n── {name} ──")
        tr, yr = refs[name]
        sol, net, params, hist, obs, true_vals = train_inverse(
            name, tr, yr,
            T=T,
            width=256, depth=4,
            n_fourier=16, fourier_sigma=4.0,
            n_colloc=20_000,
            n_data=80,
            noise_std=0.01,
            adam_epochs=2000,
            lbfgs_steps=400,
            lr=1e-3,
            lr_param=5e-3,
            lam_data=1.0,
            lam_phys=1.0,
            lam_ic=20.0,
        )
        solutions[name]    = sol
        all_hist[name]     = hist
        observations[name] = obs
        recovered[name]    = params.values()
        true_params[name]  = true_vals

    print("\n[3/3] Plotting ...")
    plot_states(solutions, refs, observations)
    plot_param_convergence(all_hist, true_params)
    plot_recovery_bars(recovered, true_params)
    plot_losses(all_hist)
    print_summary(recovered, true_params)

    return solutions, refs, all_hist, recovered, true_params


if __name__ == "__main__":
    main()
