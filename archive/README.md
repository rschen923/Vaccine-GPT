# VaccineGPT code archive

The pre-rebuild project snapshot is stored in
`VaccineGPT-code-archive-2026-10-07.zip`. The active implementation has since
been restored from that snapshot and rebuilt under `VaccineGPT\`.

The ZIP preserves the repository-relative paths for the project source,
configuration, dependency manifests, lineage contracts, and supporting
documentation. It excludes Git internals and all data directories.

Data was intentionally left in place:

- The data inventory and audit metadata remain in
  `VaccineGPT\data\metadata\`.
- The separate dataset tree remains at `..\data\` relative to the repository
  root. At archiving time it contained 113,730 files totaling 305,784,270,251
  bytes, across `metadata`, `processed`, `raw`, and `refseq_bacteria`.

No dataset files were moved, renamed, or included in the ZIP. Git history
remains available in the repository.
