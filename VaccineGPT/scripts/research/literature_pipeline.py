"""Resumable, provenance-preserving literature evidence pipeline.

The implementation intentionally uses only Python's standard library for HTTP
and serialization.  Network errors are records, never silently converted into
papers.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

DATE_FROM = "2024-09-19"
DATE_TO = "2026-09-19"
SCHEMA_VERSION = "literature-query-manifest/v1"
USER_AGENT = "VaccineGPT-literature-pipeline/1.0"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").casefold()).strip()


def canonical_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    return value.removesuffix(".")


def identifiers(record: dict[str, Any]) -> set[str]:
    keys = ("doi", "pmid", "pmcid", "normalized_title")
    result = set()
    for key in keys:
        value = record.get(key)
        if value:
            result.add(key + ":" + str(value).casefold())
    return result


def in_date_window(date: str | None) -> bool:
    return bool(date and DATE_FROM <= date[:10] <= DATE_TO)


def normalize_date(value: Any) -> str:
    """Return an ISO date only when the source supplies a usable publication date."""
    text = str(value or "").strip()
    match = re.match(r"^(\d{4})(?:[- /](\d{1,2}|[A-Za-z]{3,9}))?(?:[- /](\d{1,2}))?", text)
    if not match:
        return ""
    year, month, day = match.groups()
    if not month:
        return year
    if not month.isdigit():
        try:
            month = str(list(_dt.datetime.strptime(m, "%b").month for m in [month[:3]])[0])
        except ValueError:
            return year
    return f"{year}-{int(month):02d}-{int(day or 1):02d}"


def _http_json(url: str, timeout: int = 30) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    try:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8")), None
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _http_bytes(url: str, timeout: int = 45) -> tuple[bytes | None, str | None]:
    try:
        request = Request(url, headers={"Accept": "*/*", "User-Agent": USER_AGENT})
        with urlopen(request, timeout=timeout) as response:
            return response.read(), None
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _paper(source: str, item: dict[str, Any], section: dict[str, Any]) -> dict[str, Any] | None:
    title = item.get("title") or item.get("title_display") or ""
    if isinstance(title, list):
        title = " ".join(title)
    title = str(title).strip()
    date = item.get("publication_date") or item.get("published") or item.get("published-print")
    if isinstance(date, dict):
        date = "-".join(str(date.get(k, "01")).zfill(2) for k in ("year", "month", "day"))
    date = normalize_date(date)
    if not title or not in_date_window(date):
        return None
    doi = canonical_doi(item.get("doi") or item.get("DOI"))
    pmid = str(item.get("pmid") or "").strip() or None
    pmcid = str(item.get("pmcid") or "").strip() or None
    authors = item.get("authors") or item.get("author") or []
    if isinstance(authors, list):
        authors = [a.get("name") if isinstance(a, dict) else str(a) for a in authors]
    return {
        "record_type": "paper", "schema_version": "literature-record/v1",
        "source": source, "section": section["id"], "query_id": section.get("_query_id"),
        "title": title, "normalized_title": normalize_title(title), "publication_date": date,
        "doi": doi, "pmid": pmid, "pmcid": pmcid, "authors": authors,
        "journal": item.get("journal") or item.get("container-title") or item.get("venue"),
        "publication_type": item.get("publication_type") or item.get("type"),
        "abstract": item.get("abstract"), "landing_url": item.get("landing_url") or item.get("url"),
        "oa_url": item.get("oa_url"), "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {"provider": source, "raw_id": item.get("id") or doi or pmid},
    }


def query_openalex(query: str, section: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    params = urlencode({"search": query, "filter": f"from_publication_date:{DATE_FROM},to_publication_date:{DATE_TO}", "per-page": 100})
    data, error = _http_json("https://api.openalex.org/works?" + params)
    if error or not isinstance(data, dict):
        return [], error or "invalid OpenAlex response"
    papers = []
    for item in data.get("results", []):
        authors = [{"name": (a.get("author") or {}).get("display_name")} for a in item.get("authorships", [])]
        primary_location = item.get("primary_location") or {}
        primary_source = primary_location.get("source") or {}
        papers.append(_paper("openalex", {
            "id": item.get("id"), "title": item.get("title"), "publication_date": item.get("publication_date"),
            "doi": item.get("doi"), "authors": authors, "journal": primary_source.get("display_name"),
            "type": item.get("type"), "oa_url": (item.get("open_access") or {}).get("oa_url"),
            "url": primary_location.get("landing_page_url"),
        }, section))
    return [p for p in papers if p], None


def query_europe_pmc(query: str, section: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    params = urlencode({"query": f"({query}) AND FIRST_PDATE:[{DATE_FROM} TO {DATE_TO}]", "format": "json", "pageSize": 100})
    data, error = _http_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + params)
    if error or not isinstance(data, dict):
        return [], error or "invalid Europe PMC response"
    papers = []
    for item in data.get("resultList", {}).get("result", []):
        papers.append(_paper("europepmc", {
            "id": item.get("id"), "title": item.get("title"), "publication_date": item.get("firstPublicationDate"),
            "doi": item.get("doi"), "pmid": item.get("pmid"), "pmcid": item.get("pmcid"),
            "authors": [item.get("authorString")] if item.get("authorString") else [],
            "journal": item.get("journalTitle"), "type": item.get("pubType"),
            "oa_url": ("https://europepmc.org/articles/" + item["pmcid"]) if item.get("pmcid") else None,
            "url": "https://europepmc.org/article/MED/" + str(item.get("pmid")) if item.get("pmid") else None,
        }, section))
    return [p for p in papers if p], None


def query_pubmed(query: str, section: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    """Retrieve PubMed metadata through NCBI E-utilities (JSON only)."""
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(
        {"db": "pubmed", "term": f"({query}) AND ({DATE_FROM}[PDAT] : {DATE_TO}[PDAT])",
         "retmode": "json", "retmax": 100})
    search, error = _http_json(search_url)
    if error or not isinstance(search, dict):
        return [], error or "invalid PubMed esearch response"
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return [], None
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urlencode(
        {"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    data, error = _http_json(summary_url)
    if error or not isinstance(data, dict):
        return [], error or "invalid PubMed esummary response"
    papers = []
    for pmid in ids:
        item = data.get("result", {}).get(str(pmid), {})
        papers.append(_paper("pubmed", {
            "id": pmid, "title": item.get("title"), "publication_date": item.get("pubdate"),
            "pmid": pmid, "authors": [a.get("name") for a in item.get("authors", [])],
            "journal": item.get("fulljournalname"), "type": item.get("pubtype"),
            "url": "https://pubmed.ncbi.nlm.nih.gov/" + str(pmid) + "/",
        }, section))
    return [p for p in papers if p], None


def query_crossref(query: str, section: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    params = urlencode({"query.bibliographic": query, "filter": f"from-pub-date:{DATE_FROM},until-pub-date:{DATE_TO}", "rows": 100})
    data, error = _http_json("https://api.crossref.org/works?" + params)
    if error or not isinstance(data, dict):
        return [], error or "invalid Crossref response"
    papers = []
    for item in data.get("message", {}).get("items", []):
        date_parts = item.get("published", {}).get("date-parts", [[]])[0]
        date = "-".join(str(x).zfill(2) for x in date_parts) if date_parts else ""
        papers.append(_paper("crossref", {
            "id": item.get("DOI"), "title": (item.get("title") or [""])[0], "publication_date": date,
            "doi": item.get("DOI"), "authors": [a.get("given", "") + " " + a.get("family", "") for a in item.get("author", [])],
            "journal": (item.get("container-title") or [""])[0], "type": item.get("type"),
            "url": item.get("URL"),
        }, section))
    return [p for p in papers if p], None


def deduplicate(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for record in records:
        keys = identifiers(record)
        prior = next((seen[k] for k in keys if k in seen), None)
        if prior:
            # Preserve the most useful identifiers and provenance from both providers.
            for key in ("doi", "pmid", "pmcid", "abstract", "oa_url", "landing_url"):
                if not prior.get(key) and record.get(key):
                    prior[key] = record[key]
            prior.setdefault("provenance", {}).setdefault("providers", []).append(record.get("source"))
        else:
            seen[next(iter(sorted(keys)))] = record
            for key in keys:
                seen[key] = record
    unique: list[dict[str, Any]] = []
    seen_objects: set[int] = set()
    for value in seen.values():
        if id(value) not in seen_objects:
            unique.append(value)
            seen_objects.add(id(value))
    return unique


def compact_papers(path: Path) -> dict[str, int]:
    """Rewrite an existing paper cache through the canonical deduplicator."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique = deduplicate(rows)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in unique),
        encoding="utf-8",
    )
    return {"before": len(rows), "after": len(unique), "removed": len(rows) - len(unique)}


def resolve_fulltext(paper: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    url = paper.get("oa_url")
    if not url and paper.get("pmcid"):
        url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/" + paper["pmcid"] + "/unicode"
    result = {"paper_key": paper.get("doi") or paper.get("pmid") or paper["normalized_title"], "url": url, "status": "unresolved"}
    if not url:
        result["reason"] = "no_open_access_url"
        return result
    content, error = _http_bytes(url)
    if error:
        result.update(status="failed", reason=error)
        return result
    if not content:
        result.update(status="failed", reason="empty_fulltext")
        return result
    lowered = content[:512].lstrip().lower()
    content_type_ok = (
        content.startswith(b"%PDF-")
        or lowered.startswith(b"<?xml")
        or lowered.startswith(b"{")
        or b"<html" not in lowered[:256]
    )
    if not content_type_ok:
        result.update(status="rejected", reason="response_is_not_recognized_fulltext")
        return result
    digest = hashlib.sha256(content).hexdigest()
    expected = paper.get("fulltext_sha256")
    if expected and digest != expected:
        result.update(status="checksum_mismatch", sha256=digest, expected_sha256=expected)
        return result
    path = output_dir / "fulltext" / (hashlib.sha256(result["paper_key"].encode()).hexdigest() + ".bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content or b"")
    result.update(status="downloaded", sha256=digest, path=str(path))
    return result


def screen(paper: dict[str, Any], decision: str, reason: str, screener: str = "manual") -> dict[str, Any]:
    if decision not in {"include", "exclude", "uncertain"} or not reason.strip():
        raise ValueError("screening decision must be include/exclude/uncertain with an explicit reason")
    return {"record_type": "screening", "schema_version": "literature-record/v1",
            "paper_key": paper.get("doi") or paper.get("pmid") or paper["normalized_title"],
            "decision": decision, "reason": reason, "screener": screener}


def evidence_record(paper: dict[str, Any], finding: str, *, locator: str = "",
                    strength: str = "ungraded") -> dict[str, Any]:
    """Create a traceable evidence JSONL record; ``finding`` is never inferred."""
    if not finding.strip():
        raise ValueError("evidence finding is required")
    return {"record_type": "evidence", "schema_version": "literature-record/v1",
            "paper_key": paper.get("doi") or paper.get("pmid") or paper["normalized_title"],
            "finding": finding, "locator": locator, "strength": strength,
            "provenance": paper.get("provenance", {})}


def claim_record(claim: str, evidence_keys: list[str], *, status: str = "supported",
                 provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a claim record requiring explicit links to evidence records."""
    if not claim.strip() or not evidence_keys:
        raise ValueError("claim and at least one evidence key are required")
    return {"record_type": "claim", "schema_version": "literature-record/v1",
            "claim": claim, "evidence_keys": list(evidence_keys), "status": status,
            "provenance": provenance or {}}


def validate_quotas(papers: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    included = [p for p in papers if p.get("screening", {}).get("decision") == "include"]
    sections = {}
    failures = []
    for sec in manifest["sections"]:
        rows = [p for p in included if p["section"] == sec["id"]]
        reviews = sum(1 for p in rows if "review" in str(p.get("publication_type", "")).lower())
        research = len(rows) - reviews
        minimum = int(sec.get("minimum", 0))
        composition = sec.get("composition", {})
        ok = len(rows) >= minimum and reviews >= int(composition.get("minimum_reviews", 0)) and research >= int(composition.get("minimum_research", 0))
        sections[sec["id"]] = {"included": len(rows), "reviews": reviews, "research": research, "minimum": minimum, "ok": ok}
        if not ok:
            failures.append(sec["id"])
    return {"ok": not failures, "sections": sections, "failed_sections": failures, "date_window": [DATE_FROM, DATE_TO]}


def citation_gbt(paper: dict[str, Any]) -> str:
    authors = paper.get("authors") or ["[作者缺失]"]
    names = ", ".join(str(a).strip() for a in authors if a) or "[作者缺失]"
    title = paper.get("title", "[题名缺失]")
    journal = paper.get("journal") or "[期刊缺失]"
    doi = (" DOI:" + paper["doi"]) if paper.get("doi") else ""
    return f"{names}. {title}[J]. {journal}, {paper.get('publication_date', '')}{doi}."


def screen_pending(output_dir: Path) -> dict[str, int]:
    """Record metadata-only candidates as uncertain, never as included."""
    papers_path = output_dir / "papers.jsonl"
    screening_path = output_dir / "screening.jsonl"
    rows = [json.loads(line) for line in papers_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [
        screen(
            paper,
            "uncertain",
            "metadata candidate only; full text, study design, code/data availability, and eligibility not yet screened",
            screener="automated-metadata",
        )
        for paper in rows
    ]
    screening_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    return {"candidates": len(rows), "uncertain": len(records), "included": 0}


def screen_fulltext(output_dir: Path, *, per_section: int = 20, workers: int = 6) -> dict[str, int]:
    """Verify a bounded OA sample and include only downloaded full text.

    This is intentionally conservative: a metadata hit or landing page is not
    an included study. Existing screening records are replaced for candidates
    touched by this command and remain auditable in the JSONL ledger.
    """
    papers_path = output_dir / "papers.jsonl"
    rows = [json.loads(line) for line in papers_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    existing_path = output_dir / "screening.jsonl"
    existing = {}
    if existing_path.exists():
        for line in existing_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                existing[item["paper_key"]] = item
    counts = {"checked": 0, "included": 0, "uncertain": 0, "failed": 0}
    by_section: dict[str, int] = {}
    buckets: dict[str, list[dict[str, Any]]] = {}
    for paper in rows:
        buckets.setdefault(str(paper.get("section", "unknown")), []).append(paper)
    selected = []
    for section, section_rows in buckets.items():
        section_rows.sort(
            key=lambda paper: "review" not in (
                str(paper.get("publication_type", "")) + " " + str(paper.get("title", ""))
            ).lower()
        )
        chosen = section_rows[:per_section]
        by_section[section] = len(chosen)
        selected.extend(chosen)
    counts["checked"] = len(selected)
    results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(resolve_fulltext, paper, output_dir): index for index, paper in enumerate(selected)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    for index, paper in enumerate(selected):
        result = results[index]
        paper["fulltext"] = result
        key = paper.get("doi") or paper.get("pmid") or paper["normalized_title"]
        if result["status"] == "downloaded":
            kind = str(paper.get("publication_type") or "").lower()
            is_review = "review" in kind
            paper["screening"] = {
                "decision": "include",
                "reason": "open full text downloaded and metadata date/identity checks passed",
                "study_type": "review" if is_review else "research",
                "screener": "automated-fulltext",
            }
            counts["included"] += 1
        else:
            paper["screening"] = {
                "decision": "uncertain",
                "reason": f"full text verification {result['status']}: {result.get('reason', '')}",
                "study_type": "unknown",
                "screener": "automated-fulltext",
            }
            counts["uncertain"] += 1
            if result["status"] in {"failed", "rejected", "checksum_mismatch"}:
                counts["failed"] += 1
        existing[key] = screen(
            paper,
            paper["screening"]["decision"],
            paper["screening"]["reason"],
            screener=paper["screening"]["screener"],
        ) | {"study_type": paper["screening"]["study_type"], "fulltext": result}
    papers_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    existing_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in existing.values()),
        encoding="utf-8",
    )
    _json(output_dir / "fulltext_screening_summary.json", counts | {"sections_checked": by_section})
    return counts


def export_citations(output_dir: Path, destination: Path) -> int:
    papers_path = output_dir / "papers.jsonl"
    rows = [json.loads(line) for line in papers_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(f"[{index}] {citation_gbt(paper)}" for index, paper in enumerate(rows, 1)) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def qa(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    papers_path = output_dir / "papers.jsonl"
    papers = []
    if papers_path.exists():
        for line in papers_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                papers.append(json.loads(line))
    result = {"record_count": len(papers), "duplicate_keys": [], "invalid_dates": [], "missing_provenance": []}
    keys = set()
    for paper in papers:
        keyset = identifiers(paper)
        if keys.intersection(keyset):
            result["duplicate_keys"].append(paper.get("title"))
        keys.update(keyset)
        if not in_date_window(paper.get("publication_date")):
            result["invalid_dates"].append(paper.get("title"))
        if not paper.get("provenance"):
            result["missing_provenance"].append(paper.get("title"))
    screening_path = output_dir / "screening.jsonl"
    screening = {}
    if screening_path.exists():
        for line in screening_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                screening[item["paper_key"]] = item
    for paper in papers:
        key = paper.get("doi") or paper.get("pmid") or paper["normalized_title"]
        if key in screening:
            paper["screening"] = screening[key]
    result["screening"] = {
        "records": len(screening),
        "included": sum(item.get("decision") == "include" for item in screening.values()),
        "uncertain": sum(item.get("decision") == "uncertain" for item in screening.values()),
        "excluded": sum(item.get("decision") == "exclude" for item in screening.values()),
    }
    result["quotas"] = validate_quotas(papers, manifest)
    result["ok"] = not any(result[k] for k in ("duplicate_keys", "invalid_dates", "missing_provenance")) and result["quotas"]["ok"]
    _json(output_dir / "qa_report.json", result)
    return result


def retrieve(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    papers_path = output_dir / "papers.jsonl"
    if papers_path.exists():
        existing = [json.loads(x) for x in papers_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    failures = []
    new = []
    completed = set(_read_json(output_dir / "completed_queries.json") if (output_dir / "completed_queries.json").exists() else [])
    for section in manifest["sections"]:
        for query in section.get("queries", []):
            qid = section["id"] + ":" + query["id"]
            if qid in completed:
                continue
            sec = dict(section, _query_id=query["id"])
            for provider in query.get("providers", ["openalex", "europepmc", "crossref"]):
                fn = {"openalex": query_openalex, "pubmed": query_pubmed,
                      "europepmc": query_europe_pmc, "crossref": query_crossref}.get(provider)
                if not fn:
                    failures.append({"query_id": qid, "provider": provider, "reason": "unsupported_provider"})
                    continue
                try:
                    rows, error = fn(query["q"], sec)
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    rows, error = [], f"{type(exc).__name__}: {exc}"
                if error:
                    failures.append({"query_id": qid, "provider": provider, "reason": error})
                new.extend(rows)
            completed.add(qid)
            _json(output_dir / "completed_queries.json", sorted(completed))
    merged = deduplicate(existing + new)
    papers_path.write_text("".join(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n" for p in merged), encoding="utf-8")
    append_jsonl(output_dir / "failures.jsonl", failures)
    summary = {"retrieved_this_run": len(new), "unique_papers": len(merged), "network_failures": len(failures), "manifest": str(manifest_path)}
    _json(output_dir / "retrieval_summary.json", summary)
    return summary


def make_manifest(path: Path) -> None:
    _json(path, {"schema_version": SCHEMA_VERSION, "date_window": {"from": DATE_FROM, "to": DATE_TO},
                 "resumability": {"completed_queries_file": "completed_queries.json", "query_identity": "section.id:query.id"},
                 "sections": [{"id": "safety", "minimum": 10, "composition": {"minimum_reviews": 3, "minimum_research": 5},
                               "queries": [{"id": "safety-vaccine", "q": "vaccine safety adverse events", "providers": ["openalex", "pubmed", "europepmc", "crossref"]}]}]})


def self_test() -> None:
    assert normalize_title("A  Vaccine: Study!") == "a vaccine study"
    assert normalize_date("2025 Jan 03") == "2025-01-03"
    a = {"title": "A Vaccine Study", "normalized_title": normalize_title("A Vaccine Study"), "doi": "10.1/x"}
    b = dict(a, source="crossref", doi="https://doi.org/10.1/x")
    assert len(deduplicate([a, b])) == 1
    assert in_date_window(DATE_FROM) and in_date_window(DATE_TO) and not in_date_window("2024-09-18")
    try:
        screen(a, "include", "")
    except ValueError:
        pass
    else:
        raise AssertionError("empty screening reason accepted")
    print("literature_pipeline self-test: OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init"); p.add_argument("manifest", type=Path)
    p = sub.add_parser("retrieve"); p.add_argument("manifest", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("qa"); p.add_argument("manifest", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("compact"); p.add_argument("papers", type=Path)
    p = sub.add_parser("screen-pending"); p.add_argument("output", type=Path)
    p = sub.add_parser("screen-fulltext"); p.add_argument("output", type=Path); p.add_argument("--per-section", type=int, default=20); p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("export-citations"); p.add_argument("output", type=Path); p.add_argument("destination", type=Path)
    sub.add_parser("self-test")
    args = parser.parse_args(argv)
    if args.command == "self-test":
        self_test(); return 0
    if args.command == "init":
        make_manifest(args.manifest); return 0
    if args.command == "retrieve":
        print(json.dumps(retrieve(args.manifest, args.output), indent=2)); return 0
    if args.command == "compact":
        print(json.dumps(compact_papers(args.papers), indent=2)); return 0
    if args.command == "screen-pending":
        print(json.dumps(screen_pending(args.output), indent=2)); return 0
    if args.command == "screen-fulltext":
        print(json.dumps(screen_fulltext(args.output, per_section=args.per_section, workers=args.workers), indent=2)); return 0
    if args.command == "export-citations":
        print(json.dumps({"citations": export_citations(args.output, args.destination)}, indent=2)); return 0
    report = qa(args.output, _read_json(args.manifest))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
