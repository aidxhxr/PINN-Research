# Contributing

Start with the [installation guide](docs/installation.md) and the
[repository layout](docs/layout.md). The code studies synthetic data from a
seven-state model. A successful fit alone is not evidence of accurate
biological parameters or calibrated uncertainty.

## Change code without changing the research by accident

Use a branch and keep commits focused on one change. Describe what changed,
why it was needed, and how it was checked. Keep existing authorship intact.

Maintain the state order `[b, p, h5, h13, m, r, c]` and the documented runtime
parameter values. Changes to an equation, parameter order, forcing protocol,
objective, data generator or seed policy need an explicit protocol revision
and a corresponding check. Match old reference calculations before migrating
an experiment to shared code.

`experiments/05-inverse-single-condition/` is the preserved historical
baseline and must remain unchanged. Preserve existing runs and checkpoints
elsewhere too. Changes to an active pipeline must not silently reinterpret its
previous outputs.

## Record new experiments

Launch long training jobs in a persistent terminal session (for example
`tmux`) and write each run to a new timestamped directory. Maintained
experiments run through `wnt-pinn run configs/<profile>.json` and write to the
git-ignored root `runs/`; folder-local launchers write to that experiment's
git-ignored `runs/`.

Save the resolved configuration, code revision and local changes, dependency
versions, hardware, input hashes, seed assignments, selection rule and outcome.
Use separate seeds for observations, initialization and collocation. Keep
observations fixed when comparing network starts unless varying the data is
part of the stated protocol. Retain metrics for every start, including failures.

For learned terms, evaluate function error only over regulator ranges covered
by the data. Compare parameter recovery on the intersection of the control
and hybrid parameter sets. Record the denominator and report functional and
parametric recovery separately. Basal recovery in an equation-level screen
does not establish full inverse-PINN recovery.

Add new findings to the tracked result registry with their source inputs,
calculation, interpretation limits and publication uses. Record durable model
and protocol decisions in [`docs/decisions/`](docs/decisions/), and describe a
new experiment in its folder's README.

## Check and review

From the development environment:

```bash
make check
make reproduce
```

Add tests for scientific invariants and meaningful failure cases. Avoid tests
that repeat the implementation. The maintained suite belongs in `tests/`;
experiment scripts with names such as `*_test.py` are not collected implicitly.
CI checks the shared code and saved interfaces on CPU. Full training belongs
in a recorded experiment, not in pull-request CI.

If a change affects a poster, also run `make posters` with LuaLaTeX installed
and check the generated PDF and its verification report. Changes to the paper
should use `paper/build.sh` and keep every figure's audit file current.

A pull request should link affected result records, list validation commands,
and explain any checkpoint compatibility change. Disclose checks that remain
unrun. Keep large new outputs in the configured artifact store, referenced by
their manifest and checksums, and verify restoration before changing how old
artifacts are tracked.

## Citation, releases and licensing

Use [CITATION.cff](CITATION.cff) for the repository's software citation and
include the specific Git commit and result identifier. Publication author
lists are recorded with each publication and may differ from the software
author list.

A research release should identify its source revision, include checksums and
reproduction instructions, and state which reported results were verified.
The tag `paper-v1` marks the state from which the manuscript was built.

By contributing you agree that code is released under the [MIT License](LICENSE)
and text, figures and other documentation under
[CC BY 4.0](LICENSE-CC-BY-4.0.txt).
