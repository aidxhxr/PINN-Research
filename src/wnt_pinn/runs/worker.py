"""One regime per process, isolated from legacy module names and hybrid env."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from .checkpoint import RunState, check_stop, install_stop_handlers
from .config import load_config
from .integrity import atomic_json, commit_pair, recover_pair, sha256


def run_worker(config_path, directory, root, regime, *, resume=False):
    cfg = load_config(config_path)
    directory, root = Path(directory), Path(root)
    safe = regime.replace(" ", "_")
    pipeline = root / ("PINN-inverse-pinn-boost" if cfg["pipeline"] == "integral" else "PINN-hybrid-ude")
    install_stop_handlers()
    sys.path.insert(0, str(pipeline))
    # Every effective hybrid setting is explicit. Ambient HYBRID_* variables
    # cannot silently change the condition count or learned mechanism.
    for key in list(os.environ):
        if key.startswith("HYBRID_"):
            del os.environ[key]
    hybrid = cfg["hybrid"]
    os.environ.update({
        "HYBRID_TERM": hybrid["term"], "HYBRID_CONSTRAINT": hybrid["constraint"],
        "HYBRID_PARAM": hybrid["param"], "HYBRID_WIDTH": str(hybrid["width"]),
        "HYBRID_DEPTH": str(hybrid["depth"]), "HYBRID_WD": str(hybrid["weight_decay"]),
        "HYBRID_FREEZE": "1" if hybrid["freeze"] else "0",
        "HYBRID_STATE": str(root / hybrid["state"]) if hybrid["state"] else "",
        "HYBRID_DEPLETION": "0", "HYBRID_ACT": "tanh",
    })
    legacy_config = importlib.import_module("config")
    resource = cfg["resources"]
    device = resource["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    legacy_config.DEVICE = torch.device(device)
    torch.set_num_threads(resource["threads"])
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    choices = {c["name"]: c for c in legacy_config.CONDITIONS}
    legacy_config.CONDITIONS = [choices[name] for name in cfg["conditions"]]
    train = importlib.import_module("training")
    ref_module = importlib.import_module("reference")
    cache = directory / f"{safe}_references.npz"
    if resume:
        recover_pair(cache)
    external = cfg["data"]["reference_cache"]
    refs = {}
    if external:
        with np.load(root / external, allow_pickle=False) as data:
            for condition in cfg["conditions"]:
                stem = f"{safe}__{condition}"
                refs[condition] = (data[f"{stem}__t"].copy(), data[f"{stem}__y"].copy())
    elif resume and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            refs = {c: (data[f"{c}__t"].copy(), data[f"{c}__y"].copy()) for c in cfg["conditions"]}
    else:
        for condition in legacy_config.CONDITIONS:
            check_stop()
            _, name, t, y = ref_module._solve_one((regime, condition["name"], condition["forcing"],
                                                cfg["data"]["T"], cfg["data"]["reference_points"]))
            refs[name] = t, y
    for name, (t, y) in refs.items():
        if (t.shape != (cfg["data"]["reference_points"],) or y.shape != (len(t), 7)
                or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y))
                or not np.all(np.diff(t) > 0) or t[0] != 0 or t[-1] != cfg["data"]["T"]):
            raise ValueError(f"Invalid reference arrays for {regime}/{name}")
    if not cache.exists():
        tmp = cache.with_name(cache.name + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **{f"{c}__{key}": arr for c, pair in refs.items()
                                  for key, arr in zip(("t", "y"), pair)})
        commit_pair(tmp, cache, {})
    effective = {
        "regime": regime, "device": str(legacy_config.DEVICE), "dtype": str(torch.get_default_dtype()),
        "torch_version": torch.__version__, "threads": torch.get_num_threads(),
        "baseline": legacy_config.BASELINE, "regime_parameters": legacy_config.REGIMES[regime],
        "initial_state": legacy_config.Y0.tolist(), "conditions": legacy_config.CONDITIONS,
        "unknown_parameters": list(legacy_config.UNKNOWN),
        "reference_solver": {"method": "Radau", "rtol": 1e-10, "atol": 1e-12},
        "reference_source": external or "generated and cached",
        "reference_sha256": sha256(cache),
        "cuda_device_name": torch.cuda.get_device_name(legacy_config.DEVICE) if device.startswith("cuda") else None,
    }
    effective_path = directory / f"{safe}_effective.json"
    if resume and effective_path.exists():
        previous = json.loads(effective_path.read_text())
        for key in ("device", "dtype", "torch_version", "cuda_device_name"):
            if previous[key] != effective[key]:
                raise ValueError(f"Resume effective {key} differs from the saved run")
    atomic_json(effective_path, effective)
    seeds, data, settings = cfg["seeds"], cfg["data"], dict(cfg["training"])
    checkpoint_every = settings.pop("checkpoint_every")
    train.train_inverse(
        regime, refs, T=data["T"], n_data=data["n_data"], noise_std=data["noise_std"],
        **cfg["network"], **settings,
        rel_weight=True, colloc_mode="fixed", residual_mode="integral", activation="gelu",
        data_seed=seeds["data"], init_seed=seeds["initialization"],
        collocation_seed=seeds["collocation"], out_dir=str(directory),
        run_state=RunState(directory / "checkpoints", every=checkpoint_every, resume=resume),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    status = Path(args.directory) / f"{args.regime.replace(' ', '_')}_status.json"
    atomic_json(status, {"status": "running"})
    try:
        run_worker(args.config, args.directory, args.root, args.regime, resume=args.resume)
    except InterruptedError as error:
        atomic_json(status, {"status": "interrupted", "error": str(error)})
        print(str(error), flush=True)
        return 130
    except BaseException as error:
        atomic_json(status, {"status": "failed", "error": str(error), "type": type(error).__name__})
        traceback.print_exc()
        return 1
    atomic_json(status, {"status": "complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
