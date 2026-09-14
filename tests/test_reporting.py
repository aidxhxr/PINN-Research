"""Scientific scoring, provenance, and saved-result regression checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from wnt_pinn.reporting import reproduce, verify_registry
from wnt_pinn.reporting.metrics import common_recovery, fisher_classification, recovery_count
from wnt_pinn.reporting.publications import load_catalog
from wnt_pinn.reporting.registry import contained_path, load_registry, sha256

ROOT = Path(__file__).resolve().parents[1]


def test_recovery_uses_strict_threshold_and_actual_denominator():
    truth = {"below": 100, "boundary": 100, "above": 100}
    recovered = {"below": 109, "boundary": 110, "above": 111}
    assert recovery_count(recovered, truth) == (1, 3)
    with pytest.raises(ValueError, match="Invalid relative error"):
        recovery_count({"zero": 1}, {"zero": 0})


def test_hybrid_comparison_removes_replaced_parameters_from_both_scores():
    control = {"true": {"basal": 1, "eta": 1}, "recovered": {"basal": 2, "eta": 1}}
    hybrid = {"true": {"basal": 1}, "recovered": {"basal": 1}}
    result = common_recovery(control, hybrid)
    assert (result["control"], result["hybrid"], result["denominator"]) == (0, 1, 1)
    hybrid["true"]["basal"] = 2
    with pytest.raises(ValueError, match="disagree"):
        common_recovery(control, hybrid)


def test_fisher_parameter_counts_differ_from_eigen_direction_counts():
    # One unconstrained combination spans all three parameter axes.
    direction = np.ones(3) / np.sqrt(3)
    matrix = 1e6 * (np.eye(3) - np.outer(direction, direction)) + .01 * np.outer(direction, direction)
    result = fisher_classification(matrix)
    assert result["near_null_directions"] == 1
    assert result["nonidentifiable"] == 3
    with pytest.raises(ValueError, match="negative eigenvalue"):
        fisher_classification(np.diag([-10, 1]))


def test_publication_catalog_links_valid_registered_claims():
    registry = load_registry(ROOT)
    identifiers = {record["id"] for record in registry["results"]}
    for publication in load_catalog(ROOT)["publications"]:
        assert set(publication["result_ids"]) <= identifiers
        assert contained_path(ROOT, publication["source_directory"]).is_dir()
        assert publication["validation_scope"]


def test_registry_rejects_tampered_input_and_external_path(tmp_path):
    (tmp_path / "results").mkdir()
    source = tmp_path / "figure.png"
    source.write_bytes(b"a preserved figure")
    record = dict(id="figure", title="figure", kind="figure_snapshot", inputs=[
        dict(role="figure", path="figure.png", sha256=sha256(source))], metric_definition="byte equality",
        denominator="none", limitations=[], used_by=["README.md"], reproduction={})
    (tmp_path / "results/registry.json").write_text(json.dumps(dict(schema_version=1, results=[record])))
    assert verify_registry(tmp_path)["ok"]
    source.write_bytes(b"a changed figure")
    report = verify_registry(tmp_path)
    assert not report["ok"]
    assert "SHA-256 mismatch" in report["errors"][0]
    with pytest.raises(ValueError, match="leaves the repository"):
        contained_path(tmp_path, "../outside")


def test_saved_inverse_counts_and_report_provenance(tmp_path):
    pytest.importorskip("matplotlib")
    result = reproduce(ROOT, ["inverse_recovery"], tmp_path / "report")
    report = json.loads(Path(result["results"]["inverse_recovery"]).read_text())
    assert report["metrics"]["autodiff"]["total"] == 37
    assert report["metrics"]["integral"]["total"] == 50
    assert report["metrics"]["integral"]["denominator"] == 144
    assert report["source_inputs"]
    assert report["output_sha256"]["table.csv"]
    assert report["output_sha256"]["figure.pdf"]
    with pytest.raises(FileExistsError, match="Preserve previous results"):
        reproduce(ROOT, ["inverse_recovery"], tmp_path / "report")
