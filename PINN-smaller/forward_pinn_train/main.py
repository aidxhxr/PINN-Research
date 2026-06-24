import torch

from config import DEVICE, REGIMES
from reference import generate_references
from training import train_regime
from plotting import plot_all, plot_losses, print_summary


def main(T=150.0):
    print("="*60)
    print("Forward PINN - 7-ODE Reduced WNT-RA-HOX Model")
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name()}")
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM   : {gb:.1f} GB")
    print(f"Threads: {torch.get_num_threads()}")
    print("="*60)

    print("\n[1/3] Reference solutions (scipy Radau) ...")
    refs = generate_references(T=T, n_pts=5000)

    print("\n[2/3] Training PINNs ...")
    solutions, all_hist = {}, {}
    for name in REGIMES:
        print(f"\n── {name} ──")
        tr, yr = refs[name]
        sol, net, hist = train_regime(
            name, tr, yr,
            T=T,
            width=256,
            depth=4,
            n_fourier=16,
            fourier_sigma=4.0,
            n_colloc=50_000,
            n_data=3000,
            adam_epochs=5000,
            lbfgs_steps=500,
            lr=1e-3,
            lam_data=0.7,
            lam_phys=0.3,
        )
        solutions[name] = sol
        all_hist[name]  = hist

    print("\n[3/3] Plotting ...")
    plot_all(solutions, refs)
    plot_losses(all_hist)
    print_summary(solutions)

    return solutions, refs, all_hist


if __name__ == "__main__":
    solutions, refs, histories = main()
