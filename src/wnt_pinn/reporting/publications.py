"""Build publications in isolated directories from explicit project inputs."""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .metrics import CALCULATORS
from .registry import contained_path, load_registry, sha256, _check_expected, _verify_records


def load_catalog(root):
    catalog = json.loads((Path(root) / "publications/catalog.json").read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported publication catalog schema")
    return catalog


def build_publication(root, publication_id, output_dir, figures_only=False):
    """Build a fresh copy, preserving the reviewed PDFs and their preview URLs.

    Run this function inside the recorded tmux validation/build session.
    A publication build never launches training or inference.
    """
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    catalog = load_catalog(root)
    publications = {record["id"]: record for record in catalog["publications"]}
    if publication_id not in publications:
        raise ValueError(f"Unknown publication: {publication_id}")
    publication = publications[publication_id]
    if figures_only and not publication.get("figure_command"):
        raise ValueError(f"{publication_id} has no saved-array figure generator")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Preserve previous publication build: {output_dir}")
    registry = load_registry(root)
    by_id = {record["id"]: record for record in registry["results"]}
    records = [by_id[key] for key in publication["result_ids"]]
    verification = _verify_records(root, records)
    if not verification["ok"]:
        raise ValueError("Publication inputs failed verification: " + "; ".join(verification["errors"]))
    for record in records:
        if record["kind"] != "figure_snapshot":
            inputs = {source["role"]: contained_path(root, source["path"]) for source in record["inputs"]}
            values = CALCULATORS[record["calculator"]](inputs)
            _check_expected(values["metrics"], record.get("expected", {}))
    project = contained_path(root, publication["source_directory"])
    destination = output_dir / "source"
    destination.mkdir(parents=True)
    source_hashes = {}
    # Explicit catalog globs also work from release archives without .git.
    for pattern in publication["input_globs"]:
        for source in sorted(project.glob(pattern)):
            if not source.is_file() or any(part in ("builds", "archive", "build", "__pycache__") for part in source.relative_to(project).parts):
                continue
            relative = source.relative_to(project)
            if not source.resolve().is_relative_to(project):
                raise ValueError(f"Publication input escapes its project: {source}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[str(source.relative_to(root))] = sha256(source)
    commands = [publication["figure_command"]] if figures_only else publication["build_commands"]
    expected = publication["figure_outputs"] if figures_only else publication["outputs"]
    for relative in expected:
        target = destination / relative
        if target.is_file():
            target.unlink()
    log_path = output_dir / "build.log"
    report = {"publication_id": publication_id, "source_directory": publication["source_directory"],
              "started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hashes,
              "result_ids": publication["result_ids"], "figures_only": figures_only,
              "validation_scope": ("Figure generation and registered metric checks; PDF page layout is not checked. "
                                   "Historical transcription limits remain in the result registry."
                                   if figures_only else publication["validation_scope"]),
              "warnings": verification["warnings"],
              "registry_sha256": sha256(root / "results/registry.json"),
              "catalog_sha256": sha256(root / "publications/catalog.json"),
              "adapter_sha256": sha256(Path(__file__)),
              "commands": commands, "status": "running"}
    report_path = output_dir / "build-manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        with log_path.open("w") as log:
            for command in commands:
                command = [sys.executable if value == "{python}" else value for value in command]
                log.write("Command: " + " ".join(command) + "\n")
                log.flush()
                env = dict(os.environ)
                env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
                subprocess.run(command, cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        outputs = {}
        for relative in expected:
            path = destination / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Build omitted expected output: {relative}")
            outputs[relative] = {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        report.update(status="complete", ok=True, outputs=outputs)
    except Exception as exc:
        report.update(status="failed", ok=False, error=str(exc))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
