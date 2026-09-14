"""Epoch checkpoints for the active PINN trainers.

These files contain optimizer and RNG state and must only be loaded from a
trusted local run. An unfinished L-BFGS or refinement stage restarts from the
last Adam checkpoint, including its saved RNG state.
"""
from __future__ import annotations

import os
from pathlib import Path
import random
import signal
import threading

import numpy as np
import torch

from .integrity import commit_pair, recover_pair

_STOP = threading.Event()


def install_stop_handlers():
    _STOP.clear()
    def request_stop(signum, frame):
        _STOP.set()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def check_stop():
    if _STOP.is_set():
        raise InterruptedError("Stop requested; resume from the saved Adam boundary")


class StartCheckpoint:
    def __init__(self, path, *, every=100, resume=False):
        self.path = Path(path)
        self.every = every
        self.resume = resume
        self.stage = "adam"

    def bind(self, *, conds, params, optimizer, scheduler, generator=None, terms=None):
        self.conds = conds
        self.params = params
        self.uses_cuda = any(p.is_cuda for p in params.parameters())
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.generator = generator
        self.terms = terms or {}
        if not self.resume or recover_pair(self.path) is None:
            return None
        state = torch.load(self.path, map_location="cpu", weights_only=False)
        if state["schema_version"] != 1:
            raise ValueError("Unsupported training checkpoint schema")
        if list(state["nets"]) != [c["name"] for c in conds]:
            raise ValueError("Checkpoint condition order differs")
        if set(state["terms"]) != set(self.terms):
            raise ValueError("Checkpoint learned terms differ")
        for cond in conds:
            cond["net"].load_state_dict(state["nets"][cond["name"]])
        params.load_state_dict(state["params"])
        for name, term in self.terms.items():
            term.load_state_dict(state["terms"][name])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if generator is not None:
            generator.set_state(state["collocation_rng"])
        self.stage = state["stage"]
        return state

    def save(self, epoch, hist, lam_phys, *, stage="adam"):
        state = {
            "schema_version": 1, "stage": stage, "epoch": epoch,
            "hist": hist, "lam_phys": lam_phys,
            "nets": {c["name"]: c["net"].state_dict() for c in self.conds},
            "params": self.params.state_dict(),
            "terms": {name: term.state_dict() for name, term in self.terms.items()},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.uses_cuda else [],
            "collocation_rng": self.generator.get_state() if self.generator is not None else None,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        torch.save(state, tmp)
        commit_pair(tmp, self.path, {
            "stage": stage, "epoch": epoch, "checkpoint": self.path.name,
        })
        self.stage = stage

    def check_interrupt(self):
        check_stop()

    def after_epoch(self, epoch, last_epoch, hist, lam_phys):
        if epoch % self.every == 0 or epoch == last_epoch or _STOP.is_set():
            self.save(epoch, hist, lam_phys)
        check_stop()


class RunState:
    """A worker-local factory; the runner validates sources/config before resume."""
    def __init__(self, directory, *, every=100, resume=False):
        self.directory = Path(directory)
        self.every = every
        self.resume = resume

    def start(self, regime, index):
        safe = regime.replace(" ", "_").replace("/", "_")
        return StartCheckpoint(self.directory / safe / f"start-{index:04d}.pt",
                               every=self.every, resume=self.resume)
