# Literature retrieval and download

## Workflow

1. Translate the topic into biological concepts, methods, organisms, outcomes,
   and synonyms. Generate narrow, medium, and broad queries.
2. Search OpenAlex for discovery and citation counts; query PubMed/Europe PMC
   for biomedical metadata and abstracts; use Crossref/DOI as an identity
   resolver.
3. Deduplicate by DOI, PMID, normalized title, then author-year-title.
4. Rank by direct relevance, evidence quality, method transparency, citation
   influence, and recency. Keep the score components in the record.
5. Prefer open-access full text from PMC, institutional repositories, or
   publisher links. Download only after checking content type and file size.
6. Store a JSONL record with query, source, title, authors, year, DOI/PMID,
   abstract/full-text URL, access status, checksum, and retrieval timestamp.

## Reproducibility

Use a fixed query manifest, UTC timestamps, retry limits, rate limiting, and a
cache. A failed source is reported as failed; it is never replaced by an
uncited generated summary.

## Validation

Check DOI uniqueness, year range, abstract presence, URL reachability, PDF
magic bytes, and whether the document is a correction/editorial rather than
primary evidence. Preserve all exclusion reasons.
