"""Run the existing hybrid science checks without polluting other imports."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.legacy
def test_existing_hybrid_term_checks():
    pytest.importorskip("torch", reason="Install the ml extra to check hybrid networks")
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        WNT_PINN_DEVICE="cpu",
        WNT_PINN_THREADS="2",
    )
    # Legacy config imports set Torch's default dtype and thread count. Keep
    # these process-wide changes inside the child, then cap CPU use explicitly.
    script = """
import sys
import unittest
import torch
import test_hybrid_terms

torch.set_num_threads(2)
suite = unittest.defaultTestLoader.loadTestsFromModule(test_hybrid_terms)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project / "PINN-hybrid-ude",
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
