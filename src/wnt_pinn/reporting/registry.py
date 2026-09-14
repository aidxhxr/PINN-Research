"""Verify result provenance and write fresh, independently derived reports."""

import csv
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .metrics import CALCULATORS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root, relative):
    """Resolve registry paths without permitting reads outside the checkout."""
    root = Path(root).resolve()
    if Path(relative).is_absolute():
        raise ValueError(f"Registry path must be relative: {relative}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Registry path leaves the repository: {relative}")
    return resolved


def load_registry(root):
    registry = json.loads((Path(root) / "results/registry.json").read_text())
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported result registry schema")
    ids = [record["id"] for record in registry["results"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate result identifiers")
    for record in registry["results"]:
        for key in ("id", "title", "kind", "inputs", "metric_definition", "denominator", "limitations", "used_by", "reproduction"):
            if key not in record:
                raise ValueError(f"Result {record.get('id')} is missing {key}")
        if record["kind"] != "figure_snapshot" and record.get("calculator") not in CALCULATORS:
            raise ValueError(f"Unknown metric calculator: {record.get('calculator')}")
        roles = [source["role"] for source in record["inputs"]]
        if len(set(roles)) != len(roles):
            raise ValueError(f"Duplicate input roles in {record['id']}")
    return registry


def _verify_records(root, records):
    errors, warnings, checked = [], [], []
    cache = {}
    for record in records:
        for source in [*record["inputs"], *record.get("upstream", [])]:
            try:
                path = contained_path(root, source["path"])
                required = source in record["inputs"] or source.get("required", False)
                if not path.is_file():
                    message = f"{record['id']}: missing {'input' if required else 'upstream artifact'} {source['path']}"
                    (errors if required else warnings).append(message)
                    continue
                if source["path"] not in cache:
                    cache[source["path"]] = sha256(path)
                if cache[source["path"]] != source["sha256"]:
                    errors.append(f"{record['id']}: SHA-256 mismatch for {source['path']}")
            except (KeyError, OSError, ValueError) as exc:
                errors.append(f"{record['id']}: {exc}")
        checked.append(record["id"])
    return {"ok": not errors, "results": checked, "files_checked": len(cache), "errors": errors, "warnings": warnings}


def verify_registry(root):
    """Validate schema and input hashes; numerical checks run in reproduce()."""
    try:
        registry = load_registry(root)
        return _verify_records(root, registry["results"])
    except (KeyError, OSError, ValueError) as exc:
        return {"ok": False, "results": [], "files_checked": 0, "errors": [str(exc)], "warnings": []}


def _check_expected(actual, expected, location="metrics"):
    """Check registered claims as a subset of the recomputed metrics."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            _check_expected(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Claim shape changed at {location}")
        for index, (value, target) in enumerate(zip(actual, expected)):
            _check_expected(value, target, f"{location}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(float(actual), expected, rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError(f"Claim changed at {location}: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"Claim changed at {location}: {actual} != {expected}")


def _write_table(output, columns, rows):
    with (output / "table.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    def cell(value):
        return f"{value:.6g}" if isinstance(value, float) else str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    (output / "table.md").write_text("\n".join(lines) + "\n")


def _plot(output, record, values):
    """Regenerate compact figures that can be inspected without LaTeX."""
    if record["id"] not in ("inverse_recovery", "fisher_information", "normal_treatment"):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    figure, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    if record["id"] == "inverse_recovery":
        positions = np.arange(4)
        for index, (method, color) in enumerate((("autodiff", "#777777"), ("integral", "#2376a3"))):
            ax.bar(positions + (index - .5) * .36, values["metrics"][method]["counts"], .36, label=method, color=color)
        ax.set(xticks=positions, xticklabels=["Normal", "Early", "Advanced", "Severe"],
               ylim=(0, 36), ylabel="Parameters with relative error < 10%")
    elif record["id"] == "fisher_information":
        for regime, result in values["metrics"]["full"].items():
            ax.semilogy(np.arange(1, 37), np.maximum(result["eigenvalues"][::-1], 1e-4), label=regime)
        ax.axhline(1, color="black", lw=.7, ls="--")
        ax.set(xlabel="Eigenvalue rank", ylabel="Fisher eigenvalue (display floor: 0.0001)")
    else:
        conditions = list(values["metrics"])
        positions = np.arange(len(conditions))
        for index, (state, name) in enumerate((("m", "MYC"), ("p", "APC"))):
            ax.bar(positions + (index - .5) * .36,
                   [values["metrics"][condition][state] for condition in conditions], .36, label=name)
        ax.set(xticks=positions, xticklabels=conditions, ylabel="PINN relative L2 error (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"figure.{suffix}", dpi=160)
    plt.close(figure)
    return ["figure.png", "figure.pdf"]


def reproduce(root, result_ids=None, output_dir=None):
    """Recompute selected records and save tables, figures, and provenance.

    None or ["all"] selects every record. Each result gets a new directory.
    Existing result directories are rejected so prior evidence is preserved.
    """
    root = Path(root).resolve()
    registry = load_registry(root)
    if isinstance(result_ids, str):
        result_ids = [result_ids]
    by_id = {record["id"]: record for record in registry["results"]}
    selected = list(by_id) if result_ids is None or result_ids == ["all"] else result_ids
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select one or more distinct result identifiers")
    unknown = set(selected) - set(by_id)
    if unknown:
        raise ValueError(f"Unknown result identifiers: {sorted(unknown)}")
    records = [by_id[key] for key in selected]
    verification = _verify_records(root, records)
    if not verification["ok"]:
        raise ValueError("Result inputs failed verification: " + "; ".join(verification["errors"]))
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_dir = root / "runs" / f"{stamp}_reproduction"
    output_dir = Path(output_dir).resolve()
    for record in records:
        if (output_dir / record["id"]).exists():
            raise FileExistsError(f"Preserve previous results: {output_dir / record['id']} already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    reports = {}
    for record in records:
        destination = output_dir / record["id"]
        destination.mkdir()
        paths = {source["role"]: contained_path(root, source["path"]) for source in record["inputs"]}
        if record["kind"] == "figure_snapshot":
            source = paths["figure"]
            shutil.copy2(source, destination / source.name)
            values = {"metrics": {"snapshot_sha256": sha256(source)}, "outputs": [source.name]}
        else:
            values = CALCULATORS[record["calculator"]](paths)
            _check_expected(values["metrics"], record.get("expected", {}))
            _write_table(destination, values["columns"], values["rows"])
            values["outputs"] = ["table.csv", "table.md", *_plot(destination, record, values)]
        values.update({"result_id": record["id"], "metric_definition": record["metric_definition"],
                       "denominator": record["denominator"], "limitations": record["limitations"],
                       "source_inputs": record["inputs"], "registry_sha256": sha256(root / "results/registry.json"),
                       "reproduction_git_revision": git.stdout.strip() or None,
                       "reproduction_code_sha256": {str(path.relative_to(root)): sha256(path)
                                                    for path in Path(__file__).parent.glob("*.py")
                                                    if path.is_relative_to(root)},
                       "output_sha256": {name: sha256(destination / name) for name in values["outputs"]}})
        (destination / "result.json").write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
        reports[record["id"]] = str(destination / "result.json")
    report = {"ok": True, "output_dir": str(output_dir), "results": reports, "warnings": verification["warnings"]}
    (output_dir / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
