"""Synchronized stage boundaries, without synchronization on every epoch."""

import time

import torch


class TrainingTimer:
    def __init__(self, device):
        self.device = device
        self.marks = {}
        self.adam_epochs = 0
        self.warmup_epochs = 0
        self.closures = {"lbfgs": 0, "refine": 0}

    def mark(self, name):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.marks[name] = time.perf_counter()

    def after_adam(self):
        self.adam_epochs += 1
        if self.adam_epochs in (1, 10):
            self.mark("adam_warmup_end")
            self.warmup_epochs = self.adam_epochs
            if self.adam_epochs == 1:
                self.marks["adam_first_end"] = self.marks["adam_warmup_end"]

    def result(self):
        m = self.marks
        result = {"adam_epochs": self.adam_epochs, "warmup_epochs": self.warmup_epochs,
                  "lbfgs_closures": self.closures["lbfgs"],
                  "refine_closures": self.closures["refine"],
                  "adam_seconds": m["adam_end"] - m["adam_start"],
                  "lbfgs_seconds": m["lbfgs_end"] - m["adam_end"],
                  "refine_seconds": m["refine_end"] - m["lbfgs_end"],
                  "training_seconds": m["refine_end"] - m["adam_start"]}
        if self.adam_epochs:
            result["first_adam_seconds"] = m["adam_first_end"] - m["adam_start"]
            remaining = self.adam_epochs - self.warmup_epochs
            result["steady_adam_epochs"] = remaining
            result["steady_adam_seconds"] = m["adam_end"] - m["adam_warmup_end"]
            if remaining:
                result["steady_adam_ms_per_epoch"] = result["steady_adam_seconds"] * 1000 / remaining
        return result
