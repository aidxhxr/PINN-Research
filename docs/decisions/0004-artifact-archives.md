# Artifact archives and the public research history

The migration preserves historical artifacts in a SHA256 catalog, external
local archives and independently restored compressed shards. The local object
cache can use hard links; compressed shards are separate file contents.
All 2,924 archived files were restored from shards and verified before their
ordinary Git tracking changed. The historical inverse baseline remains tracked.

Small inputs needed to reproduce current reported results are cached under
`results/inputs/`. A shallow checkout can run the result and publication checks
without downloading all model checkpoints. Historical checkpoints are also
available through the immutable public Git revision recorded in the artifact
catalog. Full Git history is preserved, so an ordinary full clone still
includes the old artifact history.

The first backend uses explicit JSON catalogs and compressed archives, with
local and HTTPS retrieval. This allows restoration without an experiment
tracking service. A durable publishing destination for new runs remains an
owner choice. The local archive is not described as an off-site backup.
The API and checksums can support a later institutional storage or DVC remote
without changing the scientific result records.
