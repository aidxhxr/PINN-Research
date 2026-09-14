"""Validated configurations for the versioned experiment runner."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re

REGIMES = ["Normal", "Early Adenoma", "Advanced Adenoma", "Severe APC Loss"]
CONDITIONS = ["ctrl", "noATRA", "earlyATRA", "lateATRA", "lowATRA", "strongCirc",
              "wntPulse", "wntInhib", "mycPulse", "wntRAcombo"]
DEFAULTS = {
    "schema_version": 1,
    "protocol": "independent-seeds-v1",
    "name": "integral",
    "pipeline": "integral",
    "regimes": REGIMES,
    "conditions": CONDITIONS,
    "parent_run": None,
    "network": {"width": 256, "depth": 4, "n_fourier": 16, "fourier_sigma": 4.0},
    "data": {"T": 150.0, "reference_points": 5000, "n_data": 150,
             "noise_std": 0.002, "reference_cache": None},
    "training": {"adam_epochs": 2000, "lbfgs_steps": 150,
                 "param_refine_steps": 600, "n_colloc": 8000,
                 "param_refine_colloc": 8000, "lr": 0.001, "lr_param": 0.005,
                 "lam_data": 1.0, "lam_phys": 1.0, "lam_ic": 20.0,
                 "adaptive_weights": True, "weight_every": 200,
                 "weight_beta": 0.1, "n_starts": 2, "init_jitter": 0.15,
                 "log_every": 200, "checkpoint_every": 100},
    "seeds": {"data": 1042, "initialization": 2042, "collocation": 3042},
    "resources": {"device": "cpu", "threads": 1, "concurrency": 1},
    "hybrid": {"term": "none", "constraint": "anchored", "param": "gated",
               "width": 5, "depth": 2, "weight_decay": 1e-8,
               "freeze": False, "state": None},
}


def _merge(default: dict, value: dict, prefix: str = "") -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{prefix or 'configuration'} must be an object")
    unknown = set(value) - set(default)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {prefix}{', '.join(sorted(unknown))}")
    result = copy.deepcopy(default)
    for key, item in value.items():
        result[key] = (_merge(default[key], item, f"{prefix}{key}.")
                       if isinstance(default[key], dict) else item)
    return result


def _number(value, name, *, minimum=0, integer=False, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not math.isfinite(value) or (value <= minimum if strict else value < minimum):
        raise ValueError(f"{name} is outside its allowed range")


def validate_config(value: dict) -> dict:
    """Return defaults plus validated values; reject unknown fields and typos."""
    cfg = _merge(DEFAULTS, value)
    if cfg["schema_version"] != 1 or type(cfg["schema_version"]) is not int:
        raise ValueError("Only configuration schema_version 1 is supported")
    if cfg["protocol"] != "independent-seeds-v1":
        raise ValueError("Use independent-seeds-v1; historical scripts retain the old protocol")
    if cfg["pipeline"] not in {"integral", "hybrid"}:
        raise ValueError("pipeline must be integral or hybrid")
    if not isinstance(cfg["name"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", cfg["name"]):
        raise ValueError("name must contain lowercase letters, numbers, hyphens or underscores")
    for key, choices in (("regimes", REGIMES), ("conditions", CONDITIONS)):
        vals = cfg[key]
        if (not isinstance(vals, list) or not vals or any(not isinstance(v, str) for v in vals)
                or len(set(vals)) != len(vals) or not set(vals) <= set(choices)):
            raise ValueError(f"{key} must be a nonempty, unique list drawn from {choices}")
    if cfg["parent_run"] is not None and not isinstance(cfg["parent_run"], str):
        raise ValueError("parent_run must be a run ID or null")
    for key in ("width", "depth"):
        _number(cfg["network"][key], f"network.{key}", minimum=1, integer=True)
    _number(cfg["network"]["n_fourier"], "network.n_fourier", integer=True)
    _number(cfg["network"]["fourier_sigma"], "network.fourier_sigma", strict=True)
    for key in ("reference_points", "n_data"):
        _number(cfg["data"][key], f"data.{key}", minimum=2, integer=True)
    if cfg["data"]["n_data"] > cfg["data"]["reference_points"]:
        raise ValueError("n_data cannot exceed reference_points")
    _number(cfg["data"]["T"], "data.T", strict=True)
    _number(cfg["data"]["noise_std"], "data.noise_std")
    cache = cfg["data"]["reference_cache"]
    if cache is not None and not isinstance(cache, str):
        raise ValueError("data.reference_cache must be a path to an NPZ file or null")
    tr = cfg["training"]
    for key in ("adam_epochs", "weight_every", "n_starts", "log_every", "checkpoint_every"):
        _number(tr[key], f"training.{key}", minimum=1, integer=True)
    for key in ("n_colloc", "param_refine_colloc"):
        _number(tr[key], f"training.{key}", minimum=10, integer=True)
    for key in ("lbfgs_steps", "param_refine_steps"):
        _number(tr[key], f"training.{key}", integer=True)
    for key in ("lr", "lr_param"):
        _number(tr[key], f"training.{key}", strict=True)
    for key in ("lam_data", "lam_phys", "lam_ic", "init_jitter", "weight_beta"):
        _number(tr[key], f"training.{key}")
    if tr["weight_beta"] > 1 or type(tr["adaptive_weights"]) is not bool:
        raise ValueError("weight_beta must be at most 1 and adaptive_weights must be boolean")
    for key, value in cfg["seeds"].items():
        _number(value, f"seeds.{key}", integer=True)
        if value + 1000 * tr["n_starts"] >= 2**32:
            raise ValueError("Seeds plus start offsets must fit in 32 bits")
    if len(set(cfg["seeds"].values())) != 3:
        raise ValueError("Use distinct data, initialization and collocation seeds")
    res = cfg["resources"]
    for key in ("threads", "concurrency"):
        _number(res[key], f"resources.{key}", minimum=1, integer=True)
    if not isinstance(res["device"], str) or not re.fullmatch(r"cpu|auto|cuda(?::[0-9]+)?", res["device"]):
        raise ValueError("resources.device must be cpu, auto, cuda or cuda:N")
    hy = cfg["hybrid"]
    if hy["term"] not in {"none", "ra_h5", "bm_myc", "apc_mutation"}:
        raise ValueError("The runner supports none, ra_h5, bm_myc and apc_mutation terms")
    if hy["constraint"] not in {"anchored", "none", "anchored_monotone"} or hy["param"] != "gated":
        raise ValueError("Runner hybrid profiles use gated terms with a supported constraint")
    for key in ("width", "depth"):
        _number(hy[key], f"hybrid.{key}", minimum=1, integer=True)
    _number(hy["weight_decay"], "hybrid.weight_decay")
    if type(hy["freeze"]) is not bool:
        raise ValueError("hybrid.freeze must be boolean")
    if hy["state"] is not None and not isinstance(hy["state"], str):
        raise ValueError("hybrid.state must be a checkpoint path or null")
    if hy["freeze"] and not hy["state"]:
        raise ValueError("Frozen hybrid terms require a calibration checkpoint")
    if cfg["pipeline"] == "integral" and hy != DEFAULTS["hybrid"]:
        raise ValueError("Hybrid settings require pipeline=hybrid")
    if hy["term"] == "apc_mutation" and hy["constraint"] != "anchored_monotone":
        raise ValueError("APC mutation uses constraint=anchored_monotone")
    if hy["term"] == "none" and (hy["state"] or hy["freeze"]):
        raise ValueError("A hybrid checkpoint requires a learned term")
    return cfg


def load_config(path: str | Path) -> dict:
    """Load a UTF-8 JSON experiment configuration and validate it."""
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(json.load(stream))
