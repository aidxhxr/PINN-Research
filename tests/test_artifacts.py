"""Artifact restoration must recover exact bytes without overwriting other files."""

import hashlib
import io
import json
import shutil
import subprocess
import tarfile

import pytest

from wnt_pinn import artifacts


def _digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name, data in {
        "experiment/runs/001/model.pt": b"model weights\x00\x01",
        "experiment/runs/001/metrics.json": b'{"error": 0.1}\n',
        "other/runs/002/reference.npz": b"reference values",
        "README.md": b"not a research artifact\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run([
        "git", "-c", "user.name=Research Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "fixture",
    ], cwd=root, check=True)
    receipt = artifacts.inventory(root)
    assert receipt["files"] == 3
    return root


def test_inventory_records_git_objects_and_ignores_source_files(repository):
    catalog = json.loads((repository / artifacts.CATALOG_PATH).read_text())
    paths = {entry["path"] for entry in catalog["files"]}
    assert "README.md" not in paths
    assert all(len(entry["git_blob"]) == 40 for entry in catalog["files"])
    assert artifacts.verify(repository)["status"] == "ok"


def test_local_roundtrip_preserves_bytes_and_existing_inputs(repository, tmp_path):
    original = (repository / "experiment/runs/001/model.pt").read_bytes()
    store = tmp_path / "store"
    receipt = artifacts.archive(repository, store)
    assert receipt["files"] == 3
    assert receipt["shards"] == 2
    assert artifacts.verify(repository)["status"] == "ok"
    (repository / "experiment/runs/001/model.pt").unlink()
    restored = artifacts.restore(repository, store, ["experiment/runs/001"])
    assert restored["restored"] == 1
    assert restored["already_present"] == 1
    assert (repository / "experiment/runs/001/model.pt").read_bytes() == original
    assert artifacts.verify(repository)["status"] == "ok"


def test_selected_archive_contains_only_requested_prefix(repository, tmp_path):
    store = tmp_path / "selected"
    artifacts.archive(repository, store, ["other/runs"])
    catalog = json.loads((store / "catalog.json").read_text())
    assert [entry["path"] for entry in catalog["files"]] == ["other/runs/002/reference.npz"]
    with pytest.raises(ValueError, match="No catalog artifacts match"):
        artifacts.archive(repository, tmp_path / "unused", ["missing"])
    with pytest.raises(ValueError, match="outside the repository"):
        artifacts.archive(repository, repository / "store")


def test_verify_reports_missing_and_corrupt_inputs(repository):
    (repository / "experiment/runs/001/model.pt").unlink()
    (repository / "other/runs/002/reference.npz").write_bytes(b"wrong data")
    receipt = artifacts.verify(repository)
    assert receipt["status"] == "error"
    assert len(receipt["failures"]) == 2


def test_restore_refuses_different_existing_file(repository, tmp_path):
    store = tmp_path / "store"
    artifacts.archive(repository, store)
    target = repository / "experiment/runs/001/model.pt"
    target.unlink()
    target.write_bytes(b"keep these different bytes")
    with pytest.raises(ValueError, match="corrupt artifact"):
        artifacts.restore(repository, store)
    assert target.read_bytes() == b"keep these different bytes"


def test_restore_refuses_corrupt_store_object(repository, tmp_path):
    store = tmp_path / "store"
    artifacts.archive(repository, store)
    target = repository / "experiment/runs/001/model.pt"
    target.unlink()
    exported = json.loads((store / "catalog.json").read_text())
    entry = next(item for item in exported["files"] if item["path"].endswith("model.pt"))
    (store / entry["object"]).write_bytes(b"corruption")
    with pytest.raises(ValueError, match="corrupt artifact"):
        artifacts.restore(repository, store, [entry["path"]])
    assert not target.exists()


def test_shards_restore_without_the_local_object_tree(repository, tmp_path):
    store = tmp_path / "store"
    artifacts.archive(repository, store)
    shutil.rmtree(store / "objects")
    shutil.rmtree(repository / "experiment/runs")
    receipt = artifacts.restore(repository, store, ["experiment"])
    assert receipt["restored"] == 2
    assert artifacts.verify(repository)["status"] == "ok"


def test_https_restores_only_the_needed_shard(repository, tmp_path, monkeypatch):
    store = tmp_path / "store"
    artifacts.archive(repository, store)
    target = repository / "experiment/runs/001/model.pt"
    target.unlink()
    requested = []

    def download(url, timeout):
        assert timeout == 60
        assert url.startswith("https://archive.example.invalid/release/")
        relative = url.removeprefix("https://archive.example.invalid/release/")
        requested.append(relative)
        return io.BytesIO((store / relative).read_bytes())

    monkeypatch.setattr(artifacts, "urlopen", download)
    receipt = artifacts.restore(repository, "https://archive.example.invalid/release", [
        "experiment/runs/001/model.pt"
    ])
    assert receipt["restored"] == 1
    assert len(requested) == 2
    assert requested[0] == "catalog.json"
    assert requested[1].startswith("shards/experiment-")
    assert artifacts.verify(repository)["status"] == "ok"


@pytest.mark.parametrize("invalid", ["../escape", "/absolute/path", "safe/../escape", "safe\\escape"])
def test_restore_rejects_catalog_path_traversal(repository, tmp_path, invalid):
    path = repository / artifacts.CATALOG_PATH
    catalog = json.loads(path.read_text())
    catalog["files"][0]["path"] = invalid
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="path"):
        artifacts.restore(repository, tmp_path / "unused")


def test_restore_rejects_symlinked_parent(repository, tmp_path):
    store = tmp_path / "store"
    artifacts.archive(repository, store)
    shutil.rmtree(repository / "experiment/runs")
    outside = tmp_path / "outside"
    outside.mkdir()
    (repository / "experiment/runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        artifacts.restore(repository, store, ["experiment"])
    assert list(outside.iterdir()) == []


def test_restore_rejects_unsafe_tar_member_before_extracting(repository, tmp_path):
    store = tmp_path / "store"
    artifacts.archive(repository, store, ["experiment/runs/001/model.pt"])
    shutil.rmtree(store / "objects")
    target = repository / "experiment/runs/001/model.pt"
    contents = target.read_bytes()
    target.unlink()
    catalog = json.loads((store / "catalog.json").read_text())
    shard = catalog["shards"][0]
    shard_path = store / shard["path"]
    with tarfile.open(shard_path, "w:gz") as output:
        for name, data in [("experiment/runs/001/model.pt", contents), ("../escape", b"bad")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            output.addfile(info, io.BytesIO(data))
    shard["sha256"] = _digest(shard_path.read_bytes())
    shard["size"] = shard_path.stat().st_size
    shard["files"].append("../escape")
    (store / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="path"):
        artifacts.restore(repository, store, ["experiment/runs/001/model.pt"])
    assert not target.exists()
    assert not (repository.parent / "escape").exists()


def test_restore_detects_missing_file_in_shard_membership(repository, tmp_path):
    store = tmp_path / "store"
    artifacts.archive(repository, store)
    shutil.rmtree(store / "objects")
    target = repository / "experiment/runs/001/model.pt"
    target.unlink()
    catalog = json.loads((store / "catalog.json").read_text())
    for shard in catalog["shards"]:
        if target.relative_to(repository).as_posix() in shard["files"]:
            shard["files"].remove(target.relative_to(repository).as_posix())
    (store / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="Unexpected|omits"):
        artifacts.restore(repository, store, ["experiment/runs/001/model.pt"])
    assert not target.exists()


def test_fetch_result_inputs_restores_and_verifies_only_registered_dependencies(repository, tmp_path):
    store = tmp_path / "store"
    artifacts.archive(repository, store)
    target = repository / "experiment/runs/001/model.pt"
    registry = {"results": [
        {"id": "model_result", "inputs": [{
            "path": "experiment/runs/001/model.pt", "sha256": _digest(target.read_bytes())
        }]},
        {"id": "other_result", "inputs": [{
            "path": "other/runs/002/reference.npz",
            "sha256": _digest((repository / "other/runs/002/reference.npz").read_bytes()),
        }]},
    ]}
    (repository / "results").mkdir()
    (repository / "results/registry.json").write_text(json.dumps(registry))
    target.unlink()
    (repository / "other/runs/002/reference.npz").unlink()
    receipt = artifacts.restore_required_inputs(repository, store, ["model_result"])
    assert receipt["restored"] == 1
    assert receipt["inputs_verified"] == 1
    assert not (repository / "other/runs/002/reference.npz").exists()
    with pytest.raises(ValueError, match="Unknown result"):
        artifacts.restore_required_inputs(repository, store, ["unknown"])


def test_github_history_uses_immutable_catalog_commit_and_exact_selection(repository, monkeypatch):
    target = repository / "experiment/runs/001/model.pt"
    contents = target.read_bytes()
    target.unlink()
    untouched = repository / "other/runs/002/reference.npz"
    untouched.unlink()
    catalog = json.loads((repository / artifacts.CATALOG_PATH).read_text())
    revision = catalog["history_source"]["revision"]
    requested = []

    def download(url, timeout):
        assert timeout == 60
        requested.append(url)
        return io.BytesIO(contents)

    monkeypatch.setattr(artifacts, "urlopen", download)
    receipt = artifacts.restore(repository, "github-history", ["experiment/runs/001/model.pt"])
    assert requested == [
        f"https://raw.githubusercontent.com/aidxhxr/PINN-Research/{revision}/"
        "experiment/runs/001/model.pt"
    ]
    assert receipt["backend"] == "github-history"
    assert receipt["source_revision"] == revision
    assert receipt["restored"] == 1
    assert target.read_bytes() == contents
    assert not untouched.exists()


def test_github_history_checks_download_hash_before_publishing(repository, monkeypatch):
    target = repository / "experiment/runs/001/model.pt"
    size = target.stat().st_size
    target.unlink()
    monkeypatch.setattr(artifacts, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b"x" * size))
    with pytest.raises(ValueError, match="checksum failed"):
        artifacts.restore(repository, "github-history", ["experiment/runs/001/model.pt"])
    assert not target.exists()
    assert list(target.parent.iterdir()) == [target.parent / "metrics.json"]


@pytest.mark.parametrize("field,value", [
    ("revision", "main"),
    ("revision", "../unsafe"),
    ("repository", "somewhere/else"),
    ("backend", "shell-command"),
])
def test_github_history_rejects_untrusted_source_metadata(repository, monkeypatch, field, value):
    path = repository / artifacts.CATALOG_PATH
    catalog = json.loads(path.read_text())
    catalog["history_source"][field] = value
    path.write_text(json.dumps(catalog))

    def download(*_args, **_kwargs):
        pytest.fail("Invalid history metadata must be rejected before making a request")

    monkeypatch.setattr(artifacts, "urlopen", download)
    with pytest.raises(ValueError, match="history source"):
        artifacts.restore(repository, "github-history")


def test_inventory_retains_untracked_and_missing_history_at_original_revision(repository):
    path = repository / artifacts.CATALOG_PATH
    before = json.loads(path.read_text())
    original_revision = before["git_revision"]
    target = "experiment/runs/001/model.pt"
    subprocess.run(["git", "rm", "--cached", "--quiet", target], cwd=repository, check=True)
    subprocess.run([
        "git", "-c", "user.name=Research Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "Keep artifact in history",
    ], cwd=repository, check=True)
    (repository / target).unlink()
    receipt = artifacts.inventory(repository)
    refreshed = json.loads(path.read_text())
    assert receipt["files"] == len(before["files"])
    assert refreshed["git_revision"] != original_revision
    assert refreshed["history_source"]["revision"] == original_revision
    assert next(entry for entry in refreshed["files"] if entry["path"] == target)["git_revision"] == original_revision


def test_inventory_does_not_replace_changed_historical_truth(repository):
    catalog_path = repository / artifacts.CATALOG_PATH
    preserved = catalog_path.read_bytes()
    (repository / "experiment/runs/001/model.pt").write_bytes(b"new and different bytes")
    with pytest.raises(ValueError, match="Preserved artifact changed"):
        artifacts.inventory(repository)
    assert catalog_path.read_bytes() == preserved


def _new_run(repository, status="complete"):
    run = repository / "runs/new-run"
    run.mkdir(parents=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    (run / "manifest.json").write_text(json.dumps({"status": status, "git": {"revision": revision}}))
    (run / "model.pt").write_bytes(b"new run weights")
    (run / "run.log").write_text("Recorded run output\n")
    (run / "source").mkdir()
    (run / "source/config.py").write_text("SEED = 10\n")
    return run, revision


def test_explicit_run_inventory_and_archive_roundtrip_includes_manifest_and_logs(repository, tmp_path):
    run, revision = _new_run(repository)
    assert artifacts.inventory(repository)["files"] == 3
    receipt = artifacts.inventory(repository, include_runs=["runs/new-run"])
    assert receipt["registered_run_files"] == 4
    catalog = json.loads((repository / artifacts.CATALOG_PATH).read_text())
    entries = [entry for entry in catalog["files"] if entry["path"].startswith("runs/new-run/")]
    assert len(entries) == 4
    assert all(entry["git_blob"] is None and entry["git_revision"] is None for entry in entries)
    assert all(entry["run"]["source_revision"] == revision for entry in entries)
    assert all(entry["run"]["status"] == "complete" for entry in entries)
    store = tmp_path / "new-run-store"
    assert artifacts.archive(repository, store, ["runs/new-run"])["files"] == 4
    shutil.rmtree(run)
    assert artifacts.restore(repository, store, ["runs/new-run"])["restored"] == 4
    assert (run / "run.log").read_text() == "Recorded run output\n"
    assert (run / "manifest.json").is_file()
    assert artifacts.verify(repository, ["runs/new-run"])["status"] == "ok"


def test_github_history_rejects_unpublished_runs_before_download(repository, monkeypatch):
    run, _ = _new_run(repository)
    artifacts.inventory(repository, include_runs=[run])
    (run / "model.pt").unlink()

    def download(*_args, **_kwargs):
        pytest.fail("Unpublished runs must not trigger GitHub requests")

    monkeypatch.setattr(artifacts, "urlopen", download)
    with pytest.raises(ValueError, match="not published in Git history"):
        artifacts.restore(repository, "github-history", ["runs/new-run"])


@pytest.mark.parametrize("status", ["created", "running", "unknown"])
def test_run_inventory_requires_a_stopped_run(repository, status):
    run, _ = _new_run(repository, status=status)
    with pytest.raises(ValueError, match="Stop the run"):
        artifacts.inventory(repository, include_runs=[run])


def test_run_inventory_rejects_missing_manifest_and_paths_outside_runs(repository, tmp_path):
    run = repository / "runs/no-manifest"
    run.mkdir(parents=True)
    with pytest.raises(ValueError, match="manifest.json"):
        artifacts.inventory(repository, include_runs=[run])
    with pytest.raises(ValueError, match="explicit run directory"):
        artifacts.inventory(repository, include_runs=["experiment/runs/001"])
    with pytest.raises(ValueError, match="inside the repository"):
        artifacts.inventory(repository, include_runs=[tmp_path / "elsewhere"])


def test_inventory_does_not_treat_run_infrastructure_source_as_an_artifact(repository):
    source = repository / "src/wnt_pinn/runs/runner.py"
    source.parent.mkdir(parents=True)
    source.write_text("RUN_INFRASTRUCTURE = True\n")
    subprocess.run(["git", "add", str(source)], cwd=repository, check=True)
    assert artifacts.inventory(repository)["files"] == 3
