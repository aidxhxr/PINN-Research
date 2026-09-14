# Publication builds

[`catalog.json`](catalog.json) registers current posters, the historical
manuscript and presentation, and the regulatory network. It records original
source directories, result dependencies, input globs, build commands, outputs
and validation scope.

```bash
wnt-pinn publications build --id poster_first --output build/poster-review
wnt-pinn publications build --id poster_second --output build/poster-review
```

Run builds in a recorded tmux session. Each adapter creates an isolated source
copy, verifies registered result inputs, compiles it and saves a build manifest.
This leaves reviewed PDFs and the two localhost preview links intact. The
poster projects remain in their original directories, so existing links and
editable source archives stay valid.

`--figures-only` regenerates poster figures and native tables from saved data.
It needs the Python publication dependencies, while full builds also need TeX.
No adapter trains a model, solves a new reference trajectory or reruns HMC.

Both current posters use saved numerical arrays and existing layout checks.
The manuscript and presentation adapters compile historical material; their
catalog entries explicitly state that the full historical narratives have
not been re-audited. Compiling a publication does not certify its claims.

See [the reproduction guide](../docs/reproduction.md) for output paths,
evidence levels and intentional updates to the live poster files.
