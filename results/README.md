# Registered research results

[`registry.json`](registry.json) records the current README's numerical claims,
all seven README figures, both current posters, and overlapping manuscript and
presentation results. It contains 17 records: ten numerical groups and seven
figure snapshots.

Each record includes input paths and SHA-256 hashes, the calculation, its
denominator, limitations, source code references, and publications that use it.
Unknown original training revisions remain `null`. The audited code revision
is recorded separately and must not be described as a known training revision.

```bash
wnt-pinn verify
wnt-pinn reproduce --result inverse_recovery --output runs/reproduce-inverse
wnt-pinn reproduce --all --output runs/reproduce-all
```

Run reproduction and publication builds in a recorded tmux session, following
[`AGENTS.md`](../AGENTS.md). Choose a fresh output directory each time. These
commands read saved results and do not train models or rerun HMC.

`verify` checks the schema and hashes. `reproduce` also recomputes metrics,
checks registered expected values, and writes JSON, CSV, Markdown tables and
selected figures. Each output includes input and output hashes plus the
reproduction code hashes. Existing result directories are never overwritten.

## Evidence levels

| Status | Meaning |
|---|---|
| `recomputed_saved_results` | Recompute metrics or aggregates from saved numerical inputs, at the level specified by the record |
| `verified_asset_snapshot` | The README figure matches an identified original asset byte for byte; reproduction copies it |
| `historical_transcription` | Preserve a reported result whose complete original run provenance is unavailable |

Forward accuracy averages saved per-state errors. It does not rerun checkpoint
predictions. Anchor screens audit saved error dictionaries. Hybrid function
NRMSE, inverse counts, Fisher classifications, Normal state errors, and inverse
HMC diagnostics are recomputed from their saved arrays or parameter values.

The second poster's five-constraint range has a documented gap. Its full
original five-form run mapping has not been recovered. The tracked
[`constraint_screen_transcription.json`](inputs/constraint_screen_transcription.json)
preserves the dated note's table and source hash. Recomputing the range of that
table does not verify the underlying fits.

Input files are necessary for reproduction. `upstream` files document the
original exports or run artifacts. Available upstream files are hash checked;
missing optional upstream files produce explicit warnings. Retrieve missing
artifacts before claiming the full provenance chain was inspected.

Add a result by defining its immutable inputs, implementing its metric in
`src/wnt_pinn/reporting/metrics.py`, registering expected claims, and adding a
scientific check. Review any changed input hash against the original run; do
not update hashes merely to silence a verification failure.
