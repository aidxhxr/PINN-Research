# Research artifacts

`artifacts/catalog.json` records the archived files under `runs/`, historical
checkpoints outside those directories, and reference caches. Each entry has
its repository path, byte size, SHA256, Git index blob identity and file mode.
The catalog connects an external copy to the original repository file.

The migration preserves every original file and checkpoint in the local
archive, original run directories and existing Git history. Large artifacts
can be omitted from the current checkout after their archives and retrieval
paths are verified. Git history is not rewritten. New runs should use
timestamped directories and save their manifests before their artifacts are
archived. Their external storage destination is still an owner decision;
new run archives are local until that destination is configured.

## Inventory and verification

From the installed repository checkout:

```bash
wnt-pinn artifacts inventory
wnt-pinn artifacts verify
```

Inventory preserves every existing catalog entry, including artifacts removed
from the current Git tree or absent from a shallow checkout. It adds newly
tracked historical artifacts and explicitly selected new runs. Maintained
package source and test fixtures are excluded. Local drafts and unselected
runs are not added accidentally. When the local preservation audit is
available, the first inventory can reuse its hashes for unchanged files whose
modification times predate that audit. Refreshing an existing catalog checks
the bytes of every preserved file that is present.

The Git blob field refers to the index when a historical artifact was first
recorded. Each historical entry retains its original immutable Git revision
through later inventory refreshes. SHA256 identifies the file bytes
independently of Git. The catalog also records its creation time, latest
inventory time and the revision used for that inventory. A changed preserved
file is rejected before the catalog is written. Restore the original bytes
or record the changed result at a new path; inventory never replaces the old
expected hash silently.

Every command returns a JSON receipt. Verification reports missing or corrupt
files and fails through the CLI. A selector may be an exact catalog path or a
directory prefix; an unknown selector is an error.

## Register a new run

New runs stay outside Git. Register a finished run explicitly before archiving:

```bash
wnt-pinn artifacts inventory --run runs/<run-id>
wnt-pinn artifacts archive /path/to/external/wnt-pinn-artifacts \
  --select runs/<run-id>
wnt-pinn artifacts verify --select runs/<run-id>
```

The selected directory must be inside the repository's `runs/` directory and
contain `manifest.json`. Inventory includes all regular files in that run,
including the manifest, logs, checkpoints and saved source snapshot. It records
the manifest's status and source-code revision. Created or running jobs are
rejected; complete, failed and interrupted runs can be recorded once their
files have stopped changing. Symbolic links and paths outside `runs/` are
rejected.

These new entries have null Git blob and artifact revision fields because
their output bytes have not been published in Git. The source-code revision
inside their run metadata is a separate fact. Their bytes are available from
the local archive, or from an HTTPS archive after its destination is configured.
`github-history` rejects an unpublished input rather than attempting to fetch
it from the code revision.

Registration freezes the recorded output identities. If an interrupted run
will resume, preserve its archived snapshot and use a new run directory for
the continuation before registering changed outputs. Training never updates
the global artifact catalog automatically.

## Create a local archive

Choose a storage directory outside the repository:

```bash
wnt-pinn artifacts archive /path/to/external/wnt-pinn-artifacts
wnt-pinn artifacts archive /path/to/external/selected-run \
  --select PINN-inverse-pinn-boost/runs/20260711_203325_integral
```

The store contains:

```text
catalog.json              Current export manifest
catalogs/<hash>.json       Preserved export manifests
objects/<prefix>/<hash>   Files addressed by SHA256
shards/*.tar.gz           Compressed files grouped by experiment
```

Object creation uses hard links when the source and destination support
them, otherwise it copies the bytes. A hard link is not an independent
backup: changing either linked file changes the same underlying bytes.
Treat archived objects and historical runs as immutable. Verification
detects later changes. The compressed shards contain independent copies.

Shards are grouped by top-level experiment directory and split before their
uncompressed payload reaches 1 GB. Each compressed shard is checked to stay
below 2 GB and has its own SHA256 in the export manifest. A single file too
large for that budget is rejected with an explicit error. File metadata and
gzip timestamps are fixed so the same inputs produce the same shard bytes.

Export manifests are preserved by hash, while `catalog.json` points to the
latest export in that store. Archiving a selected subset creates a manifest
for that subset. Use separate store directories when publishing independent
collections.

## Restore selected inputs

Historical artifacts already published in this repository can be retrieved
directly from their immutable Git revision:

```bash
wnt-pinn artifacts restore github-history \
  --select PINN-inverse-pinn-boost/runs/20260711_203325_integral
```

The `github-history` backend reads the checked-in catalog's `history_source`.
The initial source is commit
`587842f8981bab4b20692d43dd6da3fc059e04e8` of `aidxhxr/PINN-Research`.
The implementation accepts that repository and a complete 40-character
commit identity for each historical input. It downloads the selected paths from GitHub's HTTPS raw
file service, then verifies the catalog's expected size and SHA256 before
publishing each file. It does not use a branch name, execute Git commands,
publish new files, or infer output paths from response headers.

This makes the existing historical inputs accessible from a shallow checkout
without first fetching all old Git objects. Removing large files from the
current tree does not reduce a full clone's history size. A full Git clone
still downloads the retained historical objects. New run artifacts that were
never published in that history require their local or configured HTTPS store.

For a local store:

```bash
wnt-pinn artifacts restore /path/to/external/wnt-pinn-artifacts \
  --select PINN-inverse-pinn-boost/runs/20260711_203325_integral
wnt-pinn artifacts verify \
  --select PINN-inverse-pinn-boost/runs/20260711_203325_integral
```

For an HTTPS store, publish the export's `catalog.json` and `shards/` directory
under a stable base URL, preserving their relative paths:

```bash
wnt-pinn artifacts restore https://archive.example.org/wnt-pinn/v1 \
  --select PINN-inverse-pinn-boost/runs/20260711_203325_integral
```

This is a URL example, not a configured project archive. Storage ownership,
retention and the project's publication destination must be chosen before
uploading. No credentials or remote uploads are part of this implementation.
A static institutional HTTPS archive can serve this layout directly. A
release host that flattens asset names needs a layout adapter or a static
index with matching paths.

The restore command uses objects when they are available locally and otherwise
reads the required compressed shards. HTTPS retrieval downloads only the
shards needed by the selected files. It verifies the repository's expected
hashes, the store manifest and downloaded shard checksums before publishing
restored files. Existing files are accepted only if their bytes match.
Different existing files are never overwritten. Path traversal, symbolic
links and unexpected archive members are rejected. New files are written to
temporary files, verified, then published atomically without replacement.

## Retrieve the inputs of a registered result

The Python API resolves dependencies from `results/registry.json`:

```bash
wnt-pinn artifacts fetch github-history --result inverse_recovery --include-upstream
```

The equivalent Python interface also accepts a local store:

```python
from pathlib import Path
from wnt_pinn.artifacts import restore_required_inputs

receipt = restore_required_inputs(
    Path.cwd(),
    "/path/to/external/wnt-pinn-artifacts",
    result_ids=["inverse_recovery"],
)
print(receipt)
```

The call restores missing catalogued inputs and checks every required input
against its registered hash. Files that belong in Git, such as small poster
caches, must still be present in the checkout. Optional upstream run artifacts
are included only with `include_upstream=True`; upstream files marked
`required` are always checked. After retrieval, run `make reproduce` or the
selected result's reproduction command.

The core Python interfaces are
`inventory(root, output_path=None, include_runs=None)`,
`archive(root, destination, selected=None)`,
`restore(root, destination, selected=None)` and `verify(root, selected=None)`.
An archive destination is a local path. A restore destination is a local
store path, an HTTPS base URL, or the literal `github-history`.
