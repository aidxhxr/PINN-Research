"""Configuration, launcher failure and real PINN interruption checks."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

import pytest

from wnt_pinn.runs import load_config, run_experiment, validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_config_resolves_documented_protocol():
    cfg = load_config(ROOT / "configs" / "smoke_cpu.json")
    assert cfg["protocol"] == "independent-seeds-v1"
    assert cfg["conditions"] == ["ctrl", "noATRA"]
    assert cfg["hybrid"]["term"] == "none"
    assert cfg["training"]["lam_ic"] == 20.0


@pytest.mark.parametrize("override", [
    {"network": {"widht": 32}},
    {"resources": {"threads": 0}},
    {"resources": {"device": "gpu"}},
    {"training": {"n_colloc": 3}},
    {"training": {"adam_epochs": 0}},
    {"data": {"noise_std": float("nan")}},
    {"data": {"n_data": 5001}},
    {"seeds": {"initialization": 1042}},
    {"regimes": ["normal"]},
    {"conditions": ["ctrl", "ctrl"]},
    {"hybrid": {"term": "bm_myc"}},
    {"pipeline": "hybrid", "hybrid": {"freeze": True}},
])
def test_config_rejects_invalid_or_ambiguous_inputs(override):
    with pytest.raises(ValueError):
        validate_config(override)


@pytest.mark.parametrize("folder,launcher,variant", [
    ("PINN-inverse-pinn-boost", "run_boost.sh", "integral"),
    ("PINN-hybrid-ude", "run_hybrid.sh", "control"),
])
def test_shell_launcher_propagates_worker_failure(tmp_path, folder, launcher, variant):
    # A failed child must prevent aggregation from presenting a partial run as
    # complete. The fake interpreter isolates the shell supervisor contract.
    target = tmp_path / "experiment"
    target.mkdir()
    shutil.copy2(ROOT / folder / launcher, target / launcher)
    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "python3"
    fake.write_text("#!/bin/sh\ncase \"$2\" in\n *prep_refs.py) exit 0;;\n"
                    " *aggregate*) touch \"$FAIL_TEST_AGGREGATE\"; exit 0;;\n"
                    " *) exit 7;;\nesac\n")
    fake.chmod(0o755)
    marker = tmp_path / "aggregate_was_called"
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
               FAIL_TEST_AGGREGATE=str(marker), PINN_CONCURRENCY="1", PINN_THREADS="1")
    result = subprocess.run(["bash", str(target / launcher), variant, "1"],
                            capture_output=True, text=True, env=env, timeout=15)
    assert result.returncode == 1, result.stdout + result.stderr
    assert not marker.exists()


def _command(config, resume=None):
    command = [sys.executable, str(ROOT / "experiments" / "run.py"), str(config), "--root", str(ROOT)]
    if resume:
        command += ["--resume", str(resume)]
    return command


def _env():
    return dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))


def _new_run(name, previous):
    new = set((ROOT / "runs").glob(f"*_{name}_*")) - previous
    return next(iter(new)) if len(new) == 1 else None


def _run_complete(config):
    proc = subprocess.run(_command(config), capture_output=True, text=True, env=_env(), timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return Path(proc.stdout.strip().splitlines()[-1])


@pytest.mark.integration
@pytest.mark.parametrize("pipeline", ["integral", "hybrid"])
def test_real_pinn_interrupt_resume_matches_uninterrupted(tmp_path, pipeline):
    """Interrupt actual ODE/PINN training and compare every final model tensor."""
    torch = pytest.importorskip("torch")
    cfg = load_config(ROOT / "configs" / "smoke_cpu.json")
    cfg["name"] = f"test-resume-{pipeline}-{uuid.uuid4().hex[:8]}"
    cfg["pipeline"] = pipeline
    cfg["training"]["adam_epochs"] = 40
    if pipeline == "hybrid":
        cfg["hybrid"]["term"] = "bm_myc"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    full = _run_complete(config)
    previous = set((ROOT / "runs").glob(f"*_{cfg['name']}_*"))
    log = tmp_path / "interrupted.log"
    directory = None
    checkpoint_epoch = None
    with log.open("w") as stream:
        proc = subprocess.Popen(_command(config), stdout=stream, stderr=subprocess.STDOUT, env=_env())
        try:
            deadline = time.monotonic() + 90
            while proc.poll() is None and time.monotonic() < deadline:
                directory = _new_run(cfg["name"], previous)
                checkpoint = directory / "checkpoints/Normal/start-0000.json" if directory else None
                if checkpoint and checkpoint.exists():
                    state = json.loads(checkpoint.read_text())
                    if state["stage"] == "adam" and state["epoch"] < cfg["training"]["adam_epochs"]:
                        checkpoint_epoch = state["epoch"]
                        proc.send_signal(signal.SIGTERM)
                        break
                time.sleep(0.01)
            assert checkpoint_epoch is not None, "Did not observe an Adam checkpoint before completion"
            assert proc.wait(timeout=60) != 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    assert directory is not None
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "interrupted", log.read_text()
    assert manifest["outputs"]
    result = subprocess.run(_command(config, directory), capture_output=True, text=True, env=_env(), timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["attempts"]) == 2
    assert "resume adam at Adam epoch" in (directory / "Normal.log").read_text()
    # Matching parameters alone would miss corrupted Fourier buffers or state
    # networks. Check every tensor in the saved inference models and terms.
    models = sorted(full.glob("*.pt"))
    assert len(models) >= 3
    for model in models:
        expected = torch.load(model, weights_only=True, map_location="cpu")
        actual = torch.load(directory / model.name, weights_only=True, map_location="cpu")
        assert expected.keys() == actual.keys()
        for key in expected:
            assert torch.equal(expected[key], actual[key]), f"{model.name}:{key} changed after resume"
    assert json.loads((full / "Normal_history.json").read_text()) == json.loads((directory / "Normal_history.json").read_text())
    starts = json.loads((directory / "Normal_starts.json").read_text())
    assert len(starts["starts"]) == 2
    first, second = starts["starts"]
    assert first["observation_hashes"] == second["observation_hashes"]
    assert first["initialization_seed"] != second["initialization_seed"]
    assert starts["selected_start"] == min(starts["starts"], key=lambda s: s["final_phys"])["start"]
    for entry in starts["starts"]:
        assert entry["n_unknown"] == (34 if pipeline == "hybrid" else 36)
    altered = copy.deepcopy(cfg)
    altered["training"]["adam_epochs"] += 1
    config.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="configuration differs"):
        run_experiment(config, root=ROOT, resume=directory)
