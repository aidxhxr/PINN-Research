"""Simulate abrupt stops in checksum-pair commits without running training."""
from pathlib import Path

import pytest

from wnt_pinn.runs import integrity


class SimulatedCrash(BaseException):
    pass


def _commit(target, payload, epoch):
    temporary = target.with_name(target.name + ".test.tmp")
    temporary.write_bytes(payload)
    return integrity.commit_pair(temporary, target, {"epoch": epoch})


def _crash_on_replace(monkeypatch, target, after):
    original = integrity.os.replace
    def interrupted(source, destination):
        if Path(destination) == target:
            if after:
                original(source, destination)
            raise SimulatedCrash()
        return original(source, destination)
    monkeypatch.setattr(integrity.os, "replace", interrupted)


@pytest.mark.parametrize("suffix", [".pt", ".npz"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("after_replace", [False, True])
def test_recover_interrupted_payload_record_pair(tmp_path, monkeypatch, suffix, existing, after_replace):
    target = tmp_path / ("state" + suffix)
    if existing:
        _commit(target, b"old verified payload", 1)
    with monkeypatch.context() as context:
        _crash_on_replace(context, target, after_replace)
        with pytest.raises(SimulatedCrash):
            _commit(target, b"new verified payload", 2)
    metadata = integrity.recover_pair(target)
    assert target.read_bytes() == b"new verified payload"
    assert metadata["epoch"] == 2
    assert integrity.recover_pair(target) == metadata
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.json", target.name]


def test_pending_incomplete_write_retains_verified_previous_pair(tmp_path, monkeypatch):
    target = tmp_path / "state.pt"
    original = _commit(target, b"old verified payload", 1)
    with monkeypatch.context() as context:
        _crash_on_replace(context, target, after=False)
        with pytest.raises(SimulatedCrash):
            _commit(target, b"new verified payload", 2)
    target.with_name(target.name + ".test.tmp").write_bytes(b"incomplete")
    assert integrity.recover_pair(target) == original
    assert target.read_bytes() == b"old verified payload"
    assert not target.with_name(target.name + ".pending.json").exists()


@pytest.mark.parametrize("after_replace", [False, True])
def test_journal_never_accepts_changed_committed_bytes(tmp_path, monkeypatch, after_replace):
    target = tmp_path / "state.pt"
    _commit(target, b"old verified payload", 1)
    with monkeypatch.context() as context:
        _crash_on_replace(context, target, after_replace)
        with pytest.raises(SimulatedCrash):
            _commit(target, b"new verified payload", 2)
    target.write_bytes(b"unexpected corruption")
    with pytest.raises(ValueError, match="Committed payload differs"):
        integrity.recover_pair(target)


def test_checksum_failure_without_journal_is_refused(tmp_path):
    target = tmp_path / "state.pt"
    _commit(target, b"verified payload", 1)
    target.write_bytes(b"unexpected corruption")
    with pytest.raises(ValueError, match="Payload differs"):
        integrity.recover_pair(target)


def test_missing_payload_without_journal_is_refused(tmp_path):
    target = tmp_path / "state.pt"
    _commit(target, b"verified payload", 1)
    target.unlink()
    with pytest.raises(ValueError, match="Payload is missing"):
        integrity.recover_pair(target)


def test_corrupt_pending_first_creation_is_refused(tmp_path, monkeypatch):
    target = tmp_path / "state.npz"
    with monkeypatch.context() as context:
        _crash_on_replace(context, target, after=False)
        with pytest.raises(SimulatedCrash):
            _commit(target, b"new verified payload", 1)
    target.with_name(target.name + ".test.tmp").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="Pending first payload"):
        integrity.recover_pair(target)


def test_worker_keeps_run_lock_after_supervisor_descriptor_closes(tmp_path):
    import fcntl
    import os
    import sys

    from wnt_pinn.runs.runner import _spawn_worker

    path = tmp_path / ".run.lock"
    with path.open("a") as supervisor, (tmp_path / "worker.log").open("w") as log:
        fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = _spawn_worker([sys.executable, "-c", "import time; time.sleep(60)"],
                              root=tmp_path, env=os.environ.copy(), stream=log,
                              lock_fd=supervisor.fileno())
        try:
            supervisor.close()
            with path.open("a") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                child.terminate()
                child.wait(timeout=5)
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
