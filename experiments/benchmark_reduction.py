"""Compare the previous and fused final reduction; run in a recorded tmux session.

Host timings include dispatch and allocation. CUDA graph timings isolate device
work and are not a prediction for the trainers, which do not use CUDA graphs.
Both paths share the residual and backward kernels; only the final sum differs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import triton
from torch.library import triton_op, wrap_triton

from wnt_pinn.kernels.trapezoid_triton import (
    _autograd_backward,
    _forward,
    _loss_op,
    _reduce_partials,
    _setup_context,
)

ROOT = Path(__file__).resolve().parents[1]


@triton_op("wnt_pinn_benchmark::previous_trapezoid_loss", mutates_args={})
def previous_loss(z: torch.Tensor, f: torch.Tensor, dt: torch.Tensor,
                  weight: torch.Tensor) -> torch.Tensor:
    """The pre-fusion implementation, retained only for this comparison."""
    count = (z.shape[0] - 1) * 7
    blocks = triton.cdiv(count, 256)
    parts = torch.empty((blocks,), dtype=z.dtype, device=z.device)
    wrap_triton(_forward)[(blocks,)](
        z, f, dt, weight, parts, z.shape[0], 256, enable_fp_fusion=False)
    return parts.sum() / count


previous_loss.register_autograd(_autograd_backward, setup_context=_setup_context)


def fused_reduction(parts, count, loss):
    tile = min(1024, triton.next_power_of_2(parts.numel()))
    _reduce_partials[(1,)](
        parts, loss, parts.numel(), count, tile, enable_fp_fusion=False)
    return loss


def previous_reduction(parts, count, loss):
    torch.sum(parts, dim=0, out=loss)
    return torch.div(loss, count, out=loss)


def host_us(function, iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def capture(function, calls):
    # Warm on the capture stream before recording allocations and autograd.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [function() for _ in range(calls)]
    torch.cuda.synchronize()
    return graph, outputs  # Retain captured outputs for the graph's lifetime.


def graph_us(graph, calls, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (calls * iterations)


def compare(functions, repeats, iterations, graph_calls):
    for function in functions.values():
        for _ in range(10):
            function()
    torch.cuda.synchronize()
    graphs = {name: capture(function, graph_calls) for name, function in functions.items()}
    samples = {name: {"host_us": [], "cuda_graph_us": []} for name in functions}
    for repeat in range(repeats):
        order = list(functions)
        if repeat % 2:
            order.reverse()
        for name in order:
            samples[name]["host_us"].append(host_us(functions[name], iterations))
            samples[name]["cuda_graph_us"].append(
                graph_us(graphs[name][0], graph_calls, iterations))
    for values in samples.values():
        for key in ("host_us", "cuda_graph_us"):
            values[f"median_{key}"] = statistics.median(values[key])
    return {"samples": samples, "speedup": {
        key: samples["previous"][f"median_{key}"] / samples["fused"][f"median_{key}"]
        for key in ("host_us", "cuda_graph_us")}}


def kernel_names(function):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities) as profile:
        function()
        torch.cuda.synchronize()
    return [event.name for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
            and not event.name.startswith(("Memcpy", "Memset"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", default=6, type=int)
    parser.add_argument("--iterations", default=64, type=int)
    parser.add_argument("--graph-calls", default=32, type=int)
    args = parser.parse_args()
    if not os.environ.get("TMUX"):
        parser.error("Launch in tmux and record the session in AGENTS.md")
    if min(args.repeats, args.iterations, args.graph_calls) < 1:
        parser.error("repeats, iterations and graph-calls must be positive")
    if args.out.exists():
        parser.error("Use a new result path to preserve earlier measurements")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device="cuda").manual_seed(2042)
    records, profiles = [], {}
    for n in (1024, 8000, 40001):
        z = torch.randn((n, 7), device="cuda", generator=generator, requires_grad=True)
        f = torch.randn((n, 7), device="cuda", generator=generator, requires_grad=True)
        dt = torch.rand((n - 1, 1), device="cuda", generator=generator) + 0.01
        weight = torch.tensor([[0.05, 1, 20, 3, 0.4, 7, 2]], device="cuda")
        count = (n - 1) * 7
        parts = torch.rand((triton.cdiv(count, 256),), device="cuda", generator=generator)
        # Isolate reduction launches with preallocated outputs on both sides.
        # The complete loss measurements below include the real allocations.
        old_scalar, new_scalar = torch.empty((), device="cuda"), torch.empty((), device="cuda")
        functions = {
            "reduction_preallocated": {
                "previous": lambda: previous_reduction(parts, count, old_scalar),
                "fused": lambda: fused_reduction(parts, count, new_scalar)},
            "loss_forward": {
                "previous": lambda: previous_loss(z, f, dt, weight),
                "fused": lambda: _loss_op(z, f, dt, weight)},
            "loss_forward_backward": {
                "previous": lambda: torch.autograd.grad(previous_loss(z, f, dt, weight), (z, f)),
                "fused": lambda: torch.autograd.grad(_loss_op(z, f, dt, weight), (z, f))},
        }
        for mode, implementations in functions.items():
            old, new = implementations["previous"](), implementations["fused"]()
            torch.testing.assert_close(new, old, rtol=2e-13, atol=2e-13)
            differences = ([abs(new.item() - old.item())] if isinstance(old, torch.Tensor)
                           else [(a - b).abs().max().item() for a, b in zip(old, new)])
            record = {"n_colloc": n, "mode": mode, "max_absolute_difference": max(differences),
                      **compare(implementations, args.repeats, args.iterations, args.graph_calls)}
            records.append(record)
            print(json.dumps(record), flush=True)
        if n == 8000:
            profiles = {name: kernel_names(function)
                        for name, function in functions["loss_forward"].items()}
    sources = [Path(__file__).resolve(), ROOT / "src/wnt_pinn/kernels/trapezoid_triton.py"]
    result = {
        "torch": torch.__version__, "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(), "dtype": "float64", "seed": 2042,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        "repeats": args.repeats, "iterations": args.iterations, "graph_calls": args.graph_calls,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "forward_cuda_kernels_at_8000": profiles, "records": records,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
