#!/usr/bin/env python3
"""Standalone entry point; the same study also runs through wnt-pinn run."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    from wnt_pinn.population.study import run_study
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'configs/population_extension.json'
    print(run_study(config, root=ROOT))
