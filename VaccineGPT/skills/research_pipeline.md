# Integrated research pipeline

`topic → query manifest → retrieval cache → screening → evidence table →
critical review → experiment design → NSFC proposal → Markdown/PPT/figures →
QA → continuation record`

Each stage consumes a versioned manifest and emits a report plus machine-
readable metadata. The pipeline is resumable: a failed source or parser does
not invalidate prior cached work. Research claims are linked to citation IDs;
dataset records are linked to source checksums and lineage tuples.
