"""Run manifests, input verification and bounded worker scheduling."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

from .config import load_config


class RunFailed(RuntimeError):
    """A worker failed; its run directory and logs are preserved."""


class RunInterrupted(RunFailed):
    """A signal requested an interruption; an existing run can be resumed."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _git(root, *args):
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip())
    return proc.stdout


def _source_hashes(root, pipeline):
    folder = root / ("PINN-inverse-pinn-boost" if pipeline == "integral" else "PINN-hybrid-ude")
    package = root / "src" / "wnt_pinn"
    sources = list(folder.glob("*.py"))
    for component in ("runs", "model", "networks"):
        sources.extend((package / component).rglob("*.py"))
    sources.extend([package / "__init__.py", root / "experiments" / "run.py"])
    sources = sorted(set(path for path in sources if path.is_file()))
    return {str(path.relative_to(root)): sha256(path) for path in sources}


def _input_hashes(root, config):
    paths = [config["data"]["reference_cache"], config["hybrid"]["state"]]
    result = {}
    for value in paths:
        if value:
            path = (root / value).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing configured input: {path}")
            result[value] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    return result


def _numerical_environment():
    return {"python": platform.python_version(),
            "packages": {name: metadata.version(name) for name in ("numpy", "scipy", "torch")}}


def _outputs(directory):
    return {str(p.relative_to(directory)): {"sha256": sha256(p), "size_bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file()
            and p.suffix in {".npz", ".pt", ".json"} and p.name != "manifest.json"
            and "source" not in p.relative_to(directory).parts}


def _verify_outputs(directory, manifest):
    # Refuse to resume from changed reference arrays or checkpoints.
    for name, item in manifest.get("outputs", {}).items():
        path = directory / name
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Saved run output changed or is missing: {name}")


def _spawn_worker(command, *, root, env, stream, lock_fd):
    # Keep the same open-file-description lock in each worker. If the
    # supervisor is killed, its surviving workers still exclude a second
    # writer until they finish and close their inherited descriptors.
    return subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                            stderr=subprocess.STDOUT, pass_fds=(lock_fd,))


def _run_workers(config, directory, root, manifest, resume, lock_fd):
    concurrency = config["resources"]["concurrency"]
    queue = []
    codes = {}
    for regime in config["regimes"]:
        status = directory / f"{regime.replace(' ', '_')}_status.json"
        if resume and status.exists() and json.loads(status.read_text())["status"] == "complete":
            codes[regime] = 0
        else:
            queue.append(regime)
    active = {}
    interrupted = False
    failed = False
    old_handlers = {}

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        for proc, stream in active.values():
            proc.send_signal(signal.SIGTERM)

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, stop)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(config["resources"]["threads"])
    try:
        while active or (queue and not interrupted and not failed):
            while queue and len(active) < concurrency and not interrupted and not failed:
                regime = queue.pop(0)
                log = directory / f"{regime.replace(' ', '_')}.log"
                stream = log.open("a", encoding="utf-8")
                command = [sys.executable, "-u", "-m", "wnt_pinn.runs.worker",
                           "--config", str(directory / "config.json"),
                           "--directory", str(directory), "--root", str(root), "--regime", regime]
                if resume:
                    command.append("--resume")
                try:
                    proc = _spawn_worker(command, root=root, env=env, stream=stream, lock_fd=lock_fd)
                except BaseException:
                    stream.close()
                    raise
                active[regime] = proc, stream
                manifest["attempts"][-1]["workers"].append({"regime": regime, "pid": proc.pid,
                    "command": command, "log": log.name})
                _write(directory / "manifest.json", manifest)
            for regime, (proc, stream) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                stream.close()
                codes[regime] = code
                del active[regime]
                if code:
                    failed = True
                    for other, _ in active.values():
                        other.send_signal(signal.SIGTERM)
            if active:
                time.sleep(0.1)
    finally:
        for proc, stream in active.values():
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stream.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return codes, interrupted, queue


def run_experiment(config_path, *, root=None, resume=None):
    """Run a validated integral/hybrid config; return its timestamped directory.

    ``root`` is the research repository checkout. ``resume`` is an existing run
    directory. Calls belong in a tmux session and the thread's main thread.
    Resume rejects changed configuration, source code, inputs or saved outputs.
    """
    root = Path(root or Path.cwd()).resolve()
    if not (root / "PINN-inverse-pinn-boost" / "training.py").is_file():
        raise ValueError("root must be the PINN-Research repository checkout")
    config = load_config(config_path)
    sources = _source_hashes(root, config["pipeline"])
    inputs = _input_hashes(root, config)
    config_hash = _json_hash(config)
    if resume:
        directory = Path(resume).resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["config_sha256"] != config_hash or _json_hash(load_config(directory / "config.json")) != config_hash:
            raise ValueError("Resume configuration differs from the saved resolved configuration")
        if manifest["source_hashes"] != sources or manifest["inputs"] != inputs:
            raise ValueError("Resume source code or input hashes differ from the saved run")
        if manifest["numerical_environment"] != _numerical_environment():
            raise ValueError("Resume Python or numerical package versions differ from the saved run")
        _verify_outputs(directory, manifest)
        if manifest["status"] == "complete":
            return directory
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{config['name']}_{uuid.uuid4().hex[:8]}"
        directory = root / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=False)
        _write(directory / "config.json", config)
        patch = _git(root, "diff", "HEAD", "--binary")
        (directory / "dirty.diff").write_bytes(patch)
        for relative in sources:
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / relative).read_bytes())
        versions = {dist.metadata["Name"]: dist.version for dist in metadata.distributions() if dist.metadata["Name"]}
        manifest = {
            "schema_version": 1, "id": run_id, "parent_run": config["parent_run"],
            "protocol": config["protocol"], "created_at": _now(), "status": "created", "attempts": [],
            "config_sha256": config_hash, "source_hashes": sources, "inputs": inputs,
            "git": {"revision": _git(root, "rev-parse", "HEAD").decode().strip(),
                    "status_porcelain": _git(root, "status", "--porcelain").decode().splitlines(),
                    "dirty_diff_sha256": hashlib.sha256(patch).hexdigest()},
            "environment": {"python": sys.version, "executable": sys.executable,
                            "platform": platform.platform(), "machine": platform.machine(),
                            "processor": platform.processor(), "cpu_count": os.cpu_count(), "packages": versions},
            "resources": config["resources"],
            "numerical_environment": _numerical_environment(),
            "seed_protocol": {"observations": "data seed + condition index, fixed across starts",
                              "initialization": "initialization seed + 1000 * start index",
                              "collocation": "collocation seed + 1000 * start index; fixed integral grid uses no draws"},
            "resume_policy": "Adam epochs restore models, optimizers, scheduler and RNG; interrupted L-BFGS/refinement replay from Adam boundary",
        }
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Run already has an active writer: {directory}") from error
        manifest["status"] = "running"
        # Previous attempts were verified above. While this attempt is live,
        # checkpoints legitimately advance; a hard-killed supervisor must not
        # leave those previous hashes posing as a snapshot of the newer files.
        manifest["outputs"] = {}
        manifest["attempts"].append({"started_at": _now(), "resume": bool(resume), "workers": []})
        _write(directory / "manifest.json", manifest)
        try:
            codes, interrupted, remaining = _run_workers(config, directory, root, manifest, bool(resume), lock.fileno())
            manifest["status"] = "interrupted" if interrupted else ("failed" if any(codes.values()) or remaining else "complete")
            manifest["attempts"][-1].update({"exit_codes": codes, "not_started": remaining})
        except BaseException as error:
            manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            manifest["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            manifest["attempts"][-1]["finished_at"] = _now()
            manifest["outputs"] = _outputs(directory)
            manifest["updated_at"] = _now()
            _write(directory / "manifest.json", manifest)
        if manifest["status"] == "interrupted":
            raise RunInterrupted(f"Interrupted run preserved at {directory}")
        if manifest["status"] != "complete":
            raise RunFailed(f"Worker failure; inspect {directory}/manifest.json and regime logs")
    return directory
