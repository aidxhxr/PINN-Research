"""Durable replacement of a payload and its checksum record.

A write-ahead journal makes the two-file update recoverable. At most one
pending payload is kept per target; completed checkpoints are not accumulated.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _record(path):
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or not isinstance(record.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
        raise ValueError(f"Invalid integrity record: {path}")
    return record


def _journal_path(target):
    return target.with_name(target.name + ".pending.json")


def _finish(target, temporary, record, journal):
    if temporary is not None:
        os.replace(temporary, target)
        _sync_directory(target.parent)
    atomic_json(target.with_suffix(".json"), record)
    journal.unlink()
    _sync_directory(target.parent)


def commit_pair(temporary, target, metadata):
    """Commit a finished temporary payload and its metadata as a recoverable pair."""
    target, temporary = Path(target), Path(temporary)
    if temporary.parent != target.parent or not temporary.name.startswith(target.name + "."):
        raise ValueError("Temporary payload must use the target filename prefix in the same directory")
    record = dict(metadata, sha256=sha256(temporary))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    journal = _journal_path(target)
    atomic_json(journal, {"schema_version": 1, "temporary": temporary.name, "metadata": record})
    _finish(target, temporary, record, journal)
    return record


def recover_pair(target):
    """Finish a pending update and verify its bytes, or retain a valid older pair.

    None means the first payload has not been committed yet. Unexpected bytes
    in a committed payload are rejected, even if a pending temporary file exists.
    """
    target = Path(target)
    sidecar, journal = target.with_suffix(".json"), _journal_path(target)
    if journal.exists():
        pending = json.loads(journal.read_text())
        if not isinstance(pending, dict) or pending.get("schema_version") != 1:
            raise ValueError(f"Invalid integrity journal: {journal}")
        temporary_name = pending.get("temporary")
        if (not isinstance(temporary_name, str) or Path(temporary_name).name != temporary_name
                or not temporary_name.startswith(target.name + ".")
                or temporary_name in {journal.name, sidecar.name}):
            raise ValueError(f"Invalid temporary payload in journal: {journal}")
        temporary = target.parent / temporary_name
        record = pending.get("metadata")
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError(f"Invalid pending checksum: {journal}")
        current_hash = sha256(target) if target.is_file() else None
        if current_hash == record["sha256"]:
            # The payload replacement completed; only its record or journal
            # cleanup was interrupted. Existing bytes must match the journal.
            _finish(target, None, record, journal)
            if temporary.exists():
                temporary.unlink()
                _sync_directory(target.parent)
            return record
        previous = _record(sidecar) if sidecar.exists() else None
        previous_valid = previous is not None and current_hash == previous["sha256"]
        first_creation = not target.exists() and previous is None
        if not (previous_valid or first_creation):
            raise ValueError(f"Committed payload differs from its integrity record: {target}")
        if temporary.is_file() and sha256(temporary) == record["sha256"]:
            _finish(target, temporary, record, journal)
            return record
        if previous_valid:
            # The old checkpoint remains usable if the pending write was lost
            # or incomplete. Never adopt those unverified temporary bytes.
            journal.unlink()
            if temporary.exists():
                temporary.unlink()
            _sync_directory(target.parent)
            return previous
        raise ValueError(f"Pending first payload is missing or corrupt: {target}")
    if not target.exists():
        if sidecar.exists():
            raise ValueError(f"Payload is missing: {target}")
        return None
    if not sidecar.exists():
        raise ValueError(f"Integrity record is missing: {sidecar}")
    record = _record(sidecar)
    if sha256(target) != record["sha256"]:
        raise ValueError(f"Payload differs from its integrity record: {target}")
    return record
