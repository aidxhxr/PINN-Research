#!/usr/bin/env bash
# Regenerate the STANDARD inverse-PINN plot set (the same figures pinn-boost
# produces) for every hybrid run dir. HYBRID_TERM must match the dir, because
# config.UNKNOWN (and therefore every param figure's axis set) depends on it.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE" MPLBACKEND=Agg
for d in "$HERE"/runs/*/; do
  d="${d%/}"; b="$(basename "$d")"
  case "$b" in
    *_control)   T=none ;;
    *_ra_h5|*_ra_h5_nc)   T=ra_h5 ;;
    *_bm_myc|*_bm_myc_nc) T=bm_myc ;;
    *_apc_mutation_frozen) T=apc_mutation ;;
    *) continue ;;
  esac
  echo "=== $b  (HYBRID_TERM=$T) ==="
  HYBRID_TERM=$T python3 -u "$HERE/regen_plots.py" "$d" 2>&1 | grep -viE "warning|warnings.warn"
  # plus the hybrid-only functional-identifiability figure
  if [ "$T" != "none" ]; then
    HYBRID_TERM=$T python3 -u "$HERE/plot_learned_term.py" "$d" 2>&1 | grep -viE "warning|warnings.warn"
  fi
done
