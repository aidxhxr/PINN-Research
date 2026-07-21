#!/usr/bin/env bash
# Copy the repository-native figures used by the deck into assets/figures/.
#
# Every figure in the presentation comes from a real run directory. This script
# is the single place that records WHICH run each slide's figure came from, so
# the deck can be refreshed after a retrain by editing one line here.
#
# Usage:  bash sync_figures.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/presentation/assets/figures"
mkdir -p "$DEST"

# dest-name <- source path (relative to repo root)
copy() {
  local dst="$1" src="$2"
  if [[ ! -f "$REPO/$src" ]]; then
    echo "MISSING  $src" >&2
    return 1
  fi
  cp -f "$REPO/$src" "$DEST/$dst"
  printf '  %-34s <- %s\n' "$dst" "$src"
}

echo "Syncing deck figures into assets/figures/"

# -- slide 3: biological network schematic (vector) -------------------------
copy schematic.pdf \
     network-diagram/schematic-better.pdf

# -- slide 7: reference dynamics from the stiff ODE solver ------------------
copy scipy_core_dynamics.png \
     PINN-smaller/forward_pinn_train/scipy_core_dynamics.png

# -- slides 8, 11: architecture diagrams (vector) ---------------------------
copy forward_architecture.pdf \
     PINN-smaller/forward-pinn-train-hybrid/architecture.pdf
copy inverse_architecture.pdf \
     PINN-inverse-better/architecture.pdf

# -- slide 9: forward PINN training losses ---------------------------------
#    run: config-sync regen wave, 2026-07-11
copy forward_losses.png \
     PINN-smaller/forward_pinn_train/runs/20260711_203325/pinn7_losses.png

# -- slide 10: forward PINN vs ODE solver ----------------------------------
copy pinn_forward_core_dynamics.png \
     PINN-smaller/forward_pinn_train/pinn_forward_core_dynamics.png

# -- slides 12, 14: inverse PINN losses + parameter recovery ---------------
#    run: pinn-boost integral residual, replot with fixed legend
copy inverse_losses.png \
     PINN-inverse-pinn-boost/runs/20260711_170848_replot_legendfix/inv_losses.png
copy inverse_recovery_bars_best8.png \
     PINN-inverse-pinn-boost/runs/20260711_170848_replot_legendfix/inv_recovery_bars_best8.png

# -- slide 13: inverse PINN vs ODE solver ----------------------------------
copy pinn_inverse_core_dynamics.png \
     PINN-inverse-pinn-boost/pinn_inverse_core_dynamics.png

# -- slides 15, 16: Bayesian posterior marginals ---------------------------
#    run: 20260712_204532_bayes. The top8/worst8 panels were regenerated
#    2026-07-13 (relabelled "most-sensitive", fixed Normal coverage) and are
#    the newest versions of these two figures in the repository.
copy normal_top8_marginals.png \
     PINN-bayesian/runs/20260712_204532_bayes/Normal_top8_marginals.png
copy severe_worst8_marginals.png \
     PINN-bayesian/runs/20260712_204532_bayes/Severe_APC_Loss_worst8_marginals.png

echo "Done."
