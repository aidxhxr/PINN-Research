"""Checksummed research artifacts in local stores, HTTPS shards or Git history."""

from collections import defaultdict
from datetime import datetime, timezone
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from urllib.parse import quote, urlsplit
from urllib.request import urlopen


CATALOG_PATH = "artifacts/catalog.json"
SHARD_TARGET_BYTES = 1_000_000_000
MAX_SHARD_BYTES = 2_000_000_000
MAX_CATALOG_BYTES = 32 * 1024 * 1024
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
HISTORY_REPOSITORY = "aidxhxr/PINN-Research"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path):
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError(f"Invalid artifact path: {path!r}")
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError(f"Artifact path must stay inside its root: {path!r}")
    return candidate


def _safe_path(root, relative):
    parts = _relative(relative).parts
    current = Path(root)
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Artifact path contains a symlink: {current}")
    return current


def _validate_catalog(catalog):
    if catalog.get("schema_version") != 1 or not isinstance(catalog.get("files"), list):
        raise ValueError("Unsupported artifact catalog")
    seen = set()
    for entry in catalog["files"]:
        path = str(_relative(entry["path"]))
        if path in seen:
            raise ValueError(f"Duplicate catalog path: {path}")
        seen.add(path)
        if not isinstance(entry.get("sha256"), str) or not _HASH.fullmatch(entry["sha256"]):
            raise ValueError(f"Invalid SHA256 for {path}")
        if type(entry.get("size")) is not int or entry["size"] < 0:
            raise ValueError(f"Invalid size for {path}")
        revision = entry.get("git_revision")
        if revision is not None and (not isinstance(revision, str) or not _GIT_COMMIT.fullmatch(revision)):
            raise ValueError(f"Invalid immutable source revision for {path}")
        if "object" in entry:
            _relative(entry["object"])
        if "shard" in entry:
            _relative(entry["shard"])
    for shard in catalog.get("shards", []):
        _relative(shard["path"])
        if not _HASH.fullmatch(shard["sha256"]):
            raise ValueError("Invalid shard checksum")
        if type(shard.get("size")) is not int or not 0 < shard["size"] < MAX_SHARD_BYTES:
            raise ValueError("Invalid shard size")
    if "history_source" in catalog:
        _history_base_url(catalog)
    return catalog


def _history_base_url(catalog, entry=None):
    source = catalog.get("history_source")
    if not isinstance(source, dict) or source.get("repository") != HISTORY_REPOSITORY:
        raise ValueError("GitHub history source must identify aidxhxr/PINN-Research")
    revision = source.get("revision")
    if not isinstance(revision, str) or not _GIT_COMMIT.fullmatch(revision):
        raise ValueError("GitHub history source needs a complete immutable Git commit")
    if source.get("backend") != "github-history":
        raise ValueError("Unsupported history source backend")
    if entry is not None:
        revision = entry.get("git_revision", revision)
        if revision is None:
            raise ValueError(f"Artifact is not published in Git history; use a local or HTTPS store: {entry['path']}")
        if not isinstance(revision, str) or not _GIT_COMMIT.fullmatch(revision):
            raise ValueError("GitHub history source needs a complete immutable Git commit")
    return f"https://raw.githubusercontent.com/{HISTORY_REPOSITORY}/{revision}"


def _load_catalog(root):
    path = _safe_path(root, CATALOG_PATH)
    return _validate_catalog(json.loads(path.read_text()))


def _selected_entries(catalog, selected):
    if selected is None:
        return list(catalog["files"])
    selectors = [selected] if isinstance(selected, str) else list(selected)
    if not selectors:
        raise ValueError("The artifact selection is empty")
    selectors = [str(_relative(item.rstrip("/"))) for item in selectors]
    found = {selector: False for selector in selectors}
    entries = []
    for entry in catalog["files"]:
        matches = [s for s in selectors if entry["path"] == s or entry["path"].startswith(s + "/")]
        if matches:
            entries.append(entry)
            found.update({selector: True for selector in matches})
    missing = [selector for selector, matched in found.items() if not matched]
    if missing:
        raise ValueError(f"No catalog artifacts match: {', '.join(missing)}")
    return entries


def _matches(path, entry):
    return path.is_file() and path.stat().st_size == entry["size"] and _sha256(path) == entry["sha256"]


def _require_match(path, entry):
    if not _matches(path, entry):
        raise ValueError(f"Missing or corrupt artifact: {path}")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cached_hashes(root):
    audits = sorted(root.glob("notes/organization_*/preserved_hashes.json"))
    if not audits:
        return {}, 0
    latest = audits[-1]
    return json.loads(latest.read_text()), latest.stat().st_mtime_ns


def _registered_run_entries(root, requested):
    if requested is None:
        return []
    runs = [requested] if isinstance(requested, (str, Path)) else list(requested)
    entries = []
    for run in runs:
        supplied = Path(run)
        if supplied.is_absolute():
            try:
                supplied = supplied.relative_to(root)
            except ValueError as error:
                raise ValueError("Registered runs must be inside the repository's runs/ directory") from error
        relative = _relative(supplied.as_posix())
        if relative.parts[0] != "runs" or len(relative.parts) < 2:
            raise ValueError("Register an explicit run directory under runs/, not the runs/ root")
        directory = _safe_path(root, str(relative))
        if not directory.is_dir():
            raise ValueError(f"Run directory does not exist: {relative}")
        manifest_path = _safe_path(root, f"{relative}/manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"Run directory needs manifest.json: {relative}")
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        status = manifest.get("status")
        if status not in {"complete", "failed", "interrupted"}:
            raise ValueError(f"Stop the run before archiving it; manifest status is {status!r}: {relative}")
        source_revision = manifest.get("git", {}).get("revision")
        if source_revision is not None and (
            not isinstance(source_revision, str) or not _GIT_COMMIT.fullmatch(source_revision)
        ):
            raise ValueError(f"Run manifest has an invalid source revision: {relative}")
        run_metadata = {"path": str(relative), "status": status,
                        "source_revision": source_revision,
                        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
        for candidate in sorted(directory.rglob("*")):
            name = candidate.relative_to(root).as_posix()
            path = _safe_path(root, name)
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"Run artifact must be a regular file: {name}")
            info = path.stat()
            entries.append({"path": name, "size": info.st_size, "sha256": _sha256(path),
                            "git_blob": None, "git_revision": None,
                            "git_mode": "100755" if info.st_mode & 0o111 else "100644",
                            "group": "runs", "run": dict(run_metadata)})
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError(f"Run manifest changed during inventory: {relative}")
    return entries


def inventory(root, output_path=None, include_runs=None):
    """Preserve catalogued inputs and register tracked history or explicit finished runs."""
    root = Path(root).resolve()
    tracked = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    changed = set(subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "-z"], cwd=root
    ).decode().split("\0"))
    cache, audited_at = _cached_hashes(root)
    previous_path = _safe_path(root, CATALOG_PATH)
    previous = _load_catalog(root) if previous_path.exists() else None
    entries = {}
    if previous is not None:
        old_revision = previous.get("history_source", {}).get("revision", previous.get("git_revision"))
        for original in previous["files"]:
            entry = dict(original)
            entry.setdefault("git_revision", old_revision if entry.get("git_blob") else None)
            path = _safe_path(root, entry["path"])
            if path.exists() and not _matches(path, entry):
                raise ValueError(f"Preserved artifact changed; keep its original catalog record: {entry['path']}")
            entries[entry["path"]] = entry
    reused = 0
    for record in tracked.decode().split("\0"):
        if not record:
            continue
        metadata, name = record.split("\t", 1)
        mode, blob, stage = metadata.split()
        relative = _relative(name)
        if name in entries:
            continue
        if relative.parts[0] in {"src", "tests", "docs", "configs", "results", "publications",
                                ".github", "artifacts", "runs"}:
            continue
        is_artifact = (
            "runs" in relative.parts
            or relative.suffix in {".pt", ".pth", ".ckpt"}
            or any(marker in name.lower() for marker in ("ref_cache", "reference_cache", "refcache"))
        )
        if not is_artifact:
            continue
        if stage != "0" or mode not in {"100644", "100755"}:
            raise ValueError(f"Artifact must be an ordinary, unmerged file: {name}")
        path = _safe_path(root, name)
        info = path.stat()
        use_cache = name in cache and name not in changed and info.st_mtime_ns <= audited_at
        digest = cache[name] if use_cache else _sha256(path)
        reused += int(use_cache)
        entries[name] = {
            "path": name,
            "size": info.st_size,
            "sha256": digest,
            "git_blob": blob if name not in changed else None,
            "git_revision": revision if name not in changed else None,
            "git_mode": mode,
            "group": relative.parts[0],
        }
    included = _registered_run_entries(root, include_runs)
    for entry in included:
        name = entry["path"]
        if name in entries:
            if any(entries[name][key] != entry[key] for key in ("size", "sha256")):
                raise ValueError(f"Preserved artifact changed; use a new run directory: {name}")
        else:
            entries[name] = entry
    now = datetime.now(timezone.utc).isoformat()
    history_source = previous.get("history_source") if previous is not None else None
    if history_source is None:
        history_source = {"backend": "github-history", "repository": HISTORY_REPOSITORY,
                          "revision": revision}
    catalog = _validate_catalog({
        "schema_version": 1,
        "created_at": previous.get("created_at", now) if previous is not None else now,
        "updated_at": now,
        "git_revision": revision,
        "git_blob_source": "index",
        "history_source": history_source,
        "selection": "preserved artifacts, tracked history, and explicitly registered runs",
        "files": sorted(entries.values(), key=lambda entry: entry["path"]),
    })
    output = Path(output_path) if output_path is not None else root / CATALOG_PATH
    if not output.is_absolute():
        output = root / output
    if output.is_symlink():
        raise ValueError("Catalog output cannot be a symlink")
    _write_json(output, catalog)
    return {"status": "ok", "catalog": str(output), "files": len(entries),
            "bytes": sum(entry["size"] for entry in entries.values()), "reused_audit_hashes": reused,
            "preserved_entries": len(previous["files"]) if previous is not None else 0,
            "registered_run_files": len(included)}


def verify(root, selected=None):
    """Check the repository copies against the catalog; report every failed input."""
    root = Path(root).resolve()
    entries = _selected_entries(_load_catalog(root), selected)
    failures = []
    for entry in entries:
        try:
            path = _safe_path(root, entry["path"])
            if not _matches(path, entry):
                failures.append({"path": entry["path"], "reason": "missing or SHA256/size mismatch"})
        except (OSError, ValueError) as error:
            failures.append({"path": entry["path"], "reason": str(error)})
    return {"status": "error" if failures else "ok", "checked": len(entries), "failures": failures}


def _store_object(source, target, entry):
    if target.exists():
        _require_match(target, entry)
        return "reused"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlinked"
    except FileExistsError:
        _require_match(target, entry)
        return "reused"
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
            raise
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        shutil.copyfile(source, temporary)
        _require_match(temporary, entry)
        try:
            os.link(temporary, target)
        except FileExistsError:
            _require_match(target, entry)
    finally:
        temporary.unlink(missing_ok=True)
    return "copied"


def _shard_batches(entries):
    group = []
    size = 0
    for entry in entries:
        padded = ((entry["size"] + 511) // 512 + 4) * 512
        if padded >= SHARD_TARGET_BYTES:
            raise ValueError(f"Artifact exceeds the individual shard budget: {entry['path']}")
        if group and size + padded > SHARD_TARGET_BYTES:
            yield group
            group, size = [], 0
        group.append(entry)
        size += padded
    if group:
        yield group


def _make_shard(store, name, part, entries):
    shards = _safe_path(store, "shards")
    shards.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=shards, delete=False) as output:
        temporary = Path(output.name)
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive_file:
                for entry in entries:
                    item = tarfile.TarInfo(entry["path"])
                    item.size = entry["size"]
                    item.mode = 0o755 if entry.get("git_mode") == "100755" else 0o644
                    item.mtime = 0
                    with _safe_path(store, entry["object"]).open("rb") as source:
                        archive_file.addfile(item, source)
    try:
        size = temporary.stat().st_size
        if size >= MAX_SHARD_BYTES:
            raise ValueError("Compressed shard exceeds the 2 GB publication limit")
        digest = _sha256(temporary)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        relative = f"shards/{safe_name}-{part:03d}-{digest[:16]}.tar.gz"
        target = _safe_path(store, relative)
        _store_object(temporary, target, {"size": size, "sha256": digest})
        return {"path": relative, "size": size, "sha256": digest,
                "files": [entry["path"] for entry in entries]}
    finally:
        temporary.unlink(missing_ok=True)


def archive(root, destination, selected=None):
    """Verify and copy selected inputs to an external content-addressed store."""
    root = Path(root).resolve()
    destination_path = Path(destination).expanduser()
    if destination_path.is_symlink():
        raise ValueError("Artifact store cannot be a symlink")
    store = destination_path.resolve()
    if store == root or root in store.parents:
        raise ValueError("Place the artifact store outside the repository")
    catalog = _load_catalog(root)
    entries = [dict(entry) for entry in _selected_entries(catalog, selected)]
    store.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    for entry in entries:
        source = _safe_path(root, entry["path"])
        _require_match(source, entry)
        entry["object"] = f"objects/{entry['sha256'][:2]}/{entry['sha256'][2:]}"
        target = _safe_path(store, entry["object"])
        counts[_store_object(source, target, entry)] += 1
    groups = defaultdict(list)
    for entry in entries:
        groups[PurePosixPath(entry["path"]).parts[0]].append(entry)
    shards = []
    for group, grouped_entries in sorted(groups.items()):
        for part, batch in enumerate(_shard_batches(grouped_entries), 1):
            shard = _make_shard(store, group, part, batch)
            shards.append(shard)
            for entry in batch:
                entry["shard"] = shard["path"]
    exported = {**catalog, "files": entries, "shards": shards,
                "storage": {"layout": "sha256-objects-and-tar-gzip-shards",
                            "hardlinks_are_independent_backups": False}}
    _validate_catalog(exported)
    manifest_digest = hashlib.sha256(json.dumps(exported, sort_keys=True).encode()).hexdigest()
    _write_json(_safe_path(store, f"catalogs/{manifest_digest}.json"), exported)
    _write_json(_safe_path(store, "catalog.json"), exported)
    return {"status": "ok", "store": str(store), "catalog": str(store / "catalog.json"),
            "files": len(entries), "bytes": sum(entry["size"] for entry in entries),
            "objects": dict(counts), "shards": len(shards),
            "compressed_bytes": sum(shard["size"] for shard in shards)}


def _remote_url(base, relative):
    parts = urlsplit(base)
    if parts.scheme != "https" or not parts.netloc or parts.query or parts.fragment:
        raise ValueError("Remote artifact store must be an HTTPS base URL without query or fragment")
    return base.rstrip("/") + "/" + quote(str(_relative(relative)), safe="/")


def _download(base, relative, target, maximum):
    with urlopen(_remote_url(base, relative), timeout=60) as response, target.open("wb") as stream:
        final_url = response.geturl() if hasattr(response, "geturl") else _remote_url(base, relative)
        if urlsplit(final_url).scheme != "https":
            raise ValueError("Artifact download redirected away from HTTPS")
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > maximum:
                raise ValueError(f"Downloaded artifact exceeds its declared size: {relative}")
            stream.write(chunk)


def _restore_stream(root, entry, source):
    target = _safe_path(root, entry["path"])
    if target.exists():
        _require_match(target, entry)
        return "already_present"
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _safe_path(root, entry["path"])
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            digest = hashlib.sha256()
            total = 0
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > entry["size"]:
                    raise ValueError(f"Restored artifact exceeds its declared size: {entry['path']}")
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
            if total != entry["size"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Restored artifact checksum failed: {entry['path']}")
            os.chmod(temporary, 0o755 if entry.get("git_mode") == "100755" else 0o644)
            _safe_path(root, entry["path"])
            # An atomic hard-link publication refuses to replace an existing
            # file, including one created after the initial existence check.
            try:
                os.link(temporary, target)
            except FileExistsError:
                _require_match(target, entry)
        finally:
            temporary.unlink(missing_ok=True)
    return "restored"


def _restore_shard(root, path, shard, wanted, counts):
    _require_match(path, shard)
    with tarfile.open(path, "r:gz") as archive_file:
        members = archive_file.getmembers()
        seen = set()
        for member in members:
            name = str(_relative(member.name))
            if not member.isfile() or name in seen or name not in shard["files"]:
                raise ValueError(f"Unexpected or unsafe shard member: {name}")
            seen.add(name)
            if name in wanted and member.size != wanted[name]["size"]:
                raise ValueError(f"Wrong member size in shard: {name}")
        required = set(wanted)
        if not required.issubset(shard["files"]):
            raise ValueError("Shard catalog omits a requested artifact")
        if not required.issubset(seen):
            raise ValueError("Shard is missing requested artifacts")
        for member in members:
            if member.name in required:
                with archive_file.extractfile(member) as source:
                    counts[_restore_stream(root, wanted[member.name], source)] += 1


def restore(root, destination, selected=None):
    """Retrieve selected inputs from a store, HTTPS base URL, or github-history."""
    root = Path(root).resolve()
    catalog = _load_catalog(root)
    expected = _selected_entries(catalog, selected)
    wanted = {}
    counts = defaultdict(int)
    for entry in expected:
        path = _safe_path(root, entry["path"])
        if path.exists():
            _require_match(path, entry)
            counts["already_present"] += 1
        else:
            wanted[entry["path"]] = entry
    if not wanted:
        return {"status": "ok", "files": len(expected), **dict(counts)}
    location = str(destination)
    if location == "github-history":
        urls = {name: _remote_url(_history_base_url(catalog, entry), name)
                for name, entry in wanted.items()}
        for entry in wanted.values():
            url = urls[entry["path"]]
            with urlopen(url, timeout=60) as response:
                final_url = response.geturl() if hasattr(response, "geturl") else url
                if urlsplit(final_url).scheme != "https":
                    raise ValueError("Artifact download redirected away from HTTPS")
                counts[_restore_stream(root, entry, response)] += 1
        revisions = sorted({entry.get("git_revision", catalog["history_source"]["revision"])
                            for entry in wanted.values()})
        return {"status": "ok", "backend": "github-history", "files": len(expected),
                "source_revision": revisions[0] if len(revisions) == 1 else None,
                "source_revisions": revisions, **dict(counts)}
    remote = "://" in location
    store = None if remote else Path(destination).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="wnt-pinn-artifacts-") as temporary_dir:
        temporary = Path(temporary_dir)
        if remote:
            manifest_path = temporary / "catalog.json"
            _download(location, "catalog.json", manifest_path, MAX_CATALOG_BYTES)
        else:
            manifest_path = _safe_path(store, "catalog.json")
        available = _validate_catalog(json.loads(manifest_path.read_text()))
        by_path = {entry["path"]: entry for entry in available["files"]}
        for name, entry in wanted.items():
            if name not in by_path:
                raise ValueError(f"Artifact store lacks requested input: {name}")
            stored = by_path[name]
            if any(stored[key] != entry[key] for key in ("size", "sha256")):
                raise ValueError(f"Store and repository catalogs disagree: {name}")
        pending = dict(wanted)
        if not remote:
            for name, entry in wanted.items():
                object_name = by_path[name].get("object")
                if object_name is None:
                    continue
                source_path = _safe_path(store, object_name)
                if source_path.exists():
                    _require_match(source_path, entry)
                    with source_path.open("rb") as source:
                        counts[_restore_stream(root, entry, source)] += 1
                    del pending[name]
        shard_map = {shard["path"]: shard for shard in available.get("shards", [])}
        required_shards = set()
        for name in pending:
            shard_name = by_path[name].get("shard")
            if shard_name not in shard_map:
                raise ValueError(f"No retrievable object or shard for {name}")
            required_shards.add(shard_name)
        for index, shard_name in enumerate(sorted(required_shards)):
            shard = shard_map[shard_name]
            if remote:
                shard_path = temporary / f"shard-{index}.tar.gz"
                _download(location, shard_name, shard_path, shard["size"])
            else:
                shard_path = _safe_path(store, shard_name)
            shard_inputs = {name: entry for name, entry in pending.items()
                            if by_path[name]["shard"] == shard_name}
            _restore_shard(root, shard_path, shard, shard_inputs, counts)
        for entry in wanted.values():
            _require_match(_safe_path(root, entry["path"]), entry)
    return {"status": "ok", "files": len(expected), **dict(counts)}


def restore_required_inputs(root, destination, result_ids=None, include_upstream=False):
    """Retrieve missing catalogued inputs for selected registered results and verify them."""
    root = Path(root).resolve()
    registry = json.loads(_safe_path(root, "results/registry.json").read_text())
    records = registry["results"]
    requested = set(result_ids or (record["id"] for record in records))
    known = {record["id"] for record in records}
    if requested - known:
        raise ValueError(f"Unknown result identifiers: {sorted(requested - known)}")
    sources = {}
    for record in records:
        if record["id"] not in requested:
            continue
        inputs = list(record["inputs"])
        inputs.extend(source for source in record.get("upstream", [])
                      if include_upstream or source.get("required", False))
        for source in inputs:
            name = str(_relative(source["path"]))
            if name in sources and sources[name]["sha256"] != source["sha256"]:
                raise ValueError(f"Result records disagree about the input hash: {name}")
            sources[name] = source
    catalog = {entry["path"]: entry for entry in _load_catalog(root)["files"]}
    missing = []
    for name, source in sources.items():
        path = _safe_path(root, name)
        if path.exists():
            if not path.is_file() or _sha256(path) != source["sha256"]:
                raise ValueError(f"Result input checksum failed: {name}")
        elif name not in catalog:
            raise ValueError(f"Missing result input is not archived; restore it from Git: {name}")
        elif source["sha256"] != catalog[name]["sha256"]:
            raise ValueError(f"Result and artifact catalogs disagree: {name}")
        else:
            missing.append(name)
    receipt = restore(root, destination, selected=missing) if missing else {
        "status": "ok", "restored": 0,
    }
    for name, source in sources.items():
        if _sha256(_safe_path(root, name)) != source["sha256"]:
            raise ValueError(f"Result input checksum failed after restore: {name}")
    return {**receipt, "results": sorted(requested), "inputs_verified": len(sources)}
