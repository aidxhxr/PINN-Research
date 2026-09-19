"""Matched GPU smoke training through the versioned runner; launch in tmux.

Example: PYTHONPATH=src python3 experiments/benchmark_kernels.py --out runs/<new-dir>
The comparison keeps the scientific configuration and all three seeds fixed.
Backends are interleaved in reversed order on alternate repeats. This is a
short execution/performance study, not evidence of converged inverse recovery.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from wnt_pinn.runs import load_config, run_experiment

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def microbenchmark(out):
    import torch
    import triton

    from wnt_pinn.kernels.integral import make_integral_loss, trapezoidal_loss
    from wnt_pinn.kernels.trapezoid_triton import trapezoidal_loss as fused_loss
    from wnt_pinn.model.hybrid_rhs import physics_rhs
    from wnt_pinn.model.parameters import UNKNOWN, parameters

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    torch.manual_seed(2042)
    torch.use_deterministic_algorithms(True)
    records = []
    for n in (1024, 8000):
        t = torch.linspace(0, 150, n, device="cuda").reshape(-1, 1)
        z = (torch.rand(n, 7, device="cuda") + 0.1).requires_grad_()
        f = torch.rand_like(z, requires_grad=True)
        dt, w = t[1:] - t[:-1], torch.ones((1, 7), device="cuda")
        p = parameters()
        for key in UNKNOWN:
            p[key] = torch.tensor(p[key], device="cuda", requires_grad=True)
        functions = {
            "residual_eager": (lambda: trapezoidal_loss(z, f, dt, w), [z, f]),
            "residual_triton": (lambda: fused_loss(z, f, dt, w), [z, f]),
        }
        for backend in ("eager", "triton", "compiled"):
            loss = make_integral_loss(physics_rhs, backend)
            functions[f"physics_{backend}"] = (
                lambda loss=loss: loss(t, z, p, w), [z] + [p[k] for k in UNKNOWN])
        for name, (function, inputs) in functions.items():
            for _ in range(5):
                torch.autograd.grad(function(), inputs)
            torch.cuda.synchronize()
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(30):
                    torch.autograd.grad(function(), inputs)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000 / 30)
            records.append({"n_colloc": n, "name": name, "forward_backward_ms": samples,
                            "median_ms": statistics.median(samples)})
            print(records[-1], flush=True)
        if n == 1024:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
                for _ in range(3):
                    function, inputs = functions["physics_eager"]
                    torch.autograd.grad(function(), inputs)
            profile.export_chrome_trace(str(out / "physics-trace.json"))
            (out / "physics-profile.txt").write_text(
                profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
    write_json(out / "microbenchmark.json", {
        "torch": torch.__version__, "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(), "dtype": "float64", "records": records})


def compare_runs(records):
    reference = next(record for record in records if record["backend"] == "eager")
    baseline = reference["start"]
    results = {}
    for backend in dict.fromkeys(record["backend"] for record in records):
        group = [record for record in records if record["backend"] == backend]
        results[backend] = {
            key: statistics.median(record["start"]["timings"][key] for record in group)
            for key in ("steady_adam_ms_per_epoch", "adam_seconds", "lbfgs_seconds",
                        "refine_seconds", "training_seconds", "first_adam_seconds")}
        results[backend]["process_wall_seconds"] = statistics.median(
            record["process_wall_seconds"] for record in group)
        results[backend]["first_run_training_seconds"] = group[0]["start"]["timings"]["training_seconds"]
        results[backend]["repeat_training_seconds"] = [
            r["start"]["timings"]["training_seconds"] for r in group[1:]]
        results[backend]["final_physics_losses"] = [r["start"]["final_phys"] for r in group]
        results[backend]["max_parameter_relative_difference"] = max(
            abs(r["start"]["recovered"][key] - value) / max(abs(value), 1e-15)
            for r in group for key, value in baseline["recovered"].items())
        results[backend]["observations_identical"] = all(
            r["start"]["observation_hashes"] == baseline["observation_hashes"] for r in group)
        results[backend]["closure_counts"] = [
            {key: r["start"]["timings"][key] for key in ("lbfgs_closures", "refine_closures")}
            for r in group]
    for backend, result in results.items():
        result["steady_adam_speedup"] = (
            results["eager"]["steady_adam_ms_per_epoch"] / result["steady_adam_ms_per_epoch"])
        result["training_speedup_including_first_call"] = (
            results["eager"]["training_seconds"] / result["training_seconds"])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/kernel_smoke.json")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--backends", nargs="+", choices=("eager", "triton", "compiled"),
                        default=["eager", "triton", "compiled"])
    parser.add_argument("--skip-micro", action="store_true")
    parser.add_argument("--micro-only", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("TMUX"):
        parser.error("Launch this benchmark in tmux and record its session in AGENTS.md")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if "eager" not in args.backends or len(set(args.backends)) != len(args.backends):
        parser.error("Include eager exactly once and do not repeat backends")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.micro_only:
        microbenchmark(out)
        return
    if (out / "comparison.json").exists():
        parser.error("Use a new output directory to preserve earlier comparisons")
    cfg = load_config(args.config)
    if len(cfg["regimes"]) != 1 or cfg["training"]["n_starts"] != 1:
        parser.error("The smoke comparison requires one regime and one start")
    if cfg["training"]["adam_epochs"] <= 10:
        parser.error("Use more than ten Adam epochs to measure time after warmup")
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "2"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(out / "training-compiler-cache")
    os.environ["TRITON_CACHE_DIR"] = str(out / "training-triton-cache")
    safe = cfg["regimes"][0].replace(" ", "_")
    records = []
    for repeat in range(args.repeats):
        order = list(args.backends)
        if repeat % 2:
            order.reverse()
        for backend in order:
            config = copy.deepcopy(cfg)
            config["name"] = f"kernel-{backend}-r{repeat}"
            config["training"]["physics_backend"] = backend
            path = out / f"config-{backend}-{repeat}.json"
            write_json(path, config)
            print(f"Starting {backend}, repeat {repeat}", flush=True)
            start = time.perf_counter()
            directory = run_experiment(path, root=ROOT)
            elapsed = time.perf_counter() - start
            result = json.loads((directory / f"{safe}_starts.json").read_text())["starts"][0]
            records.append({"backend": backend, "repeat": repeat, "directory": str(directory),
                            "process_wall_seconds": elapsed, "start": result})
            write_json(out / "training-runs.json", records)
            print(f"Finished {backend}: {result['timings']}", flush=True)
    comparison = compare_runs(records)
    write_json(out / "comparison.json", {"config": cfg, "repeats": args.repeats,
                                         "results": comparison, "runs": records})
    lines = ["# Matched float64 kernel smoke comparison", "",
             f"{len(cfg['conditions'])} conditions, {cfg['training']['n_colloc']} collocation points; "
             "fixed observations and initialization; short training only.", "",
             "| Backend | Adam ms/epoch after warmup | Speedup | Training seconds including first call |",
             "|---|---:|---:|---:|"]
    for backend, result in comparison.items():
        lines.append(f"| {backend} | {result['steady_adam_ms_per_epoch']:.2f} | "
                     f"{result['steady_adam_speedup']:.3f}x | {result['training_seconds']:.2f} |")
    lines += ["", "Medians across interleaved repeats. Compilation/cache loading is included in",
              "first-call and total times, excluded from steady Adam timing (first 10 epochs).",
              "Reference generation, checkpoint files and process startup are in process wall time.",
              "The GPU is shared; measurements describe this session, not an isolated machine.", ""]
    (out / "report.md").write_text("\n".join(lines))
    environment = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(out / "micro-compiler-cache"),
                       TRITON_CACHE_DIR=str(out / "micro-triton-cache"),
                       CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if not args.skip_micro:
        with (out / "microbenchmark.log").open("w") as stream:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--out", str(out),
                            "--micro-only"], cwd=ROOT, env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
