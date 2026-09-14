# Ecology project separation

The ecology project was extracted into
[ecosystem-engineer-model](https://github.com/aidxhxr/ecosystem-engineer-model)
on 14 September 2026. Its local checkout is
`/home/29/aidahxr/ecosystem-engineer-model`. The split import commit is
`795517aa452c912a71adf1612021a6d795ced191`; standalone documentation is in
`9b2d718`. All 23 file hashes and the original subtree identity were verified
before the remote push. PINN-Research retains a pointer and its original history.

## Preservation record

The inspected source revision is
`587842f8981bab4b20692d43dd6da3fc059e04e8`. Its `new-theme/` tree is
`d6d794201bac67be6f869b54f536cb862db68e66`.

The subtree contains 23 tracked files totaling 3,622,196 bytes:

| Contents | Files |
|---|---:|
| README and bibliography | 2 |
| Original report PDF | 1 |
| Literature and assessment notes in `prior-art/` | 10 |
| Verification scripts and saved numerical outputs | 10 |

There were no tracked modifications or untracked files within `new-theme/`
at inspection time. Sixteen commits record its history, from the initial
report import through the September 7 publication-context update. Retain
that history through `git subtree split`, including the existing author
and committer metadata.

`combined_report.pdf` has 25 pages. Its printed date is 10 August 2026 and its
credit is `Prepared for Kings`; that credit also appears in the PDF metadata.
Preserve the document as received. Git commit authorship does not establish
authorship of the supplied report, so do not invent a new report author list.
Its SHA256 is:

```text
59aa01d507c6a492788d08922fba1628277d7710e8550429759ed345bbe9db13
```

The original LaTeX source and a complete reproduction pipeline for the report
are not present in the subtree. The verification scripts check selected
calculations; they do not regenerate every report figure. Preserve this
distinction in the independent repository's documentation.

## Dependency audit

The Python scripts import NumPy, SciPy and the standard library only. They
do not import the WNT package or read files elsewhere in PINN-Research.
Relative links in `prior-art/` point to files that remain inside the extracted
tree. The README mentions an earlier local `notes/` copy, but the corresponding
published synthesis is already tracked under `prior-art/`.

Several scripts depend on neighboring files or the working directory:

| Script | Dependency or output behavior |
|---|---|
| `verification/verify.py` | Standalone calculations; prints results |
| `verification/verify2.py` | Standalone calculations; prints results |
| `verification/minors.py` | Opens `verify2.py` from the current directory and executes its initial portion |
| `verification/mechanism_test.py` | Runs a parameter search at import/execution time; prints results |
| `verification/mechanism_test2.py` | Loads helpers from the neighboring `mechanism_test.py`; writes `mechanism_hits.json` to the current directory |
| `verification/piskovsky_check.py` | Loads helpers and the saved `mechanism_hits.json` relative to its own file |

Extraction itself needs no Python path fixes. Historical commands that use
`minors.py` must still run from `verification/`. Do not rerun
`mechanism_test2.py` there as part of migration validation: it would replace
the saved, tracked search output. Imports also execute calculations, so
these files must not become implicit pytest collection targets.

No scientific calculation was rerun during this audit.

## Migration sequence

1. Record the source revision and the subtree tree identity above. Confirm
   that `new-theme/` still has no local modifications.
2. Split the subtree into its own history, then create the independent local
   checkout at the proposed destination. The initial split commit's root
   tree must equal the original `new-theme/` tree.
3. Verify all 23 file hashes and the report PDF hash before making any
   independent-repository edits. Record the original and split commit IDs
   in a tracked migration note in the new repository.
4. Apply the standalone documentation and environment changes below as
   separate commits after the preserved extraction.
5. Create or use the chosen remote, push the extracted history, and verify
   its remote commit and downloadable contents. Do not remove the source
   files from PINN-Research before that verification.
6. Replace `new-theme/` here with a short README linking to the independent
   repository and its preserved import commit. Update the research index.

The extraction does not require rewriting PINN-Research's existing history.
Its old commits continue to retain the original subtree and document
provenance. Any later history-size reduction is a separate operation.

## Standalone repository edits

The initial migration should keep scientific sources and saved outputs
unchanged. Add the following project setup around them:

- Rename the README heading and displayed directory tree to the independent
  project name. Replace references to the surrounding PINN repository with
  the migration provenance link. Keep the September 7 corrections and the
  distinction between current findings and superseded proposals.
- Add a minimal environment declaration for Python 3.11 or 3.12, NumPy and
  SciPy. Do not carry over the WNT project's CUDA or Torch requirements.
- Add a local `.gitignore` for virtual environments, Python caches and new
  `runs/` directories. Keep the original PDF, bibliography, assessments and
  saved verification outputs tracked.
- Add an `AGENTS.md` with the independent repository root and its own tmux
  session table. Retain the rules that new experiments use tmux, save new
  timestamped outputs and preserve historical files.
- Document the working-directory constraints above. A later code change can
  extract shared helpers from `exec(...)`, add explicit script entry points,
  and direct fresh search outputs to timestamped run directories. Check
  numerical agreement before replacing those historical implementations.
- Do not add a reuse license or new publication authorship without an explicit
  owner decision. Preserve the existing bibliography's verification tags,
  source links, article credits and stated limitations.

Suggested README and `.gitignore` additions are saved as a local patch in
`notes/ecology-standalone-recommendations.patch`. The patch is a proposal;
it has not been applied to `new-theme/` or to an independent checkout.

## References in PINN-Research

The tracked files outside `new-theme/` had no references to it at the audited
source revision. The new organization work adds a `new-theme/` row to
`docs/research-index.md`; change that row to the verified independent remote.
The new `new-theme/README.md` pointer and this migration note should identify
the source subtree and extracted import commit. Update any root README or
repository-map link added during the same organization work to use the final
remote URL.
