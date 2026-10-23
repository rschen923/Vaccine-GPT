from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


API_URL = "https://query-api.iedb.org/api/v1/antigen_search"
SELECT = (
    "parent_source_antigen_id,parent_source_antigen_names,"
    "parent_source_antigen_source_org_iri,parent_source_antigen_source_org_name,"
    "qualitative_measures,reference_titles,pubmed_ids,reference_ids,iedb_assay_ids,"
    "linear_sequences,linear_sequence_lengths"
)
FILTER = "cs.{taxon_protein:2}"
PAGE_SIZE = 10_000
USER_AGENT = "VaccineGPT-public-data-audit/1.0"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_outputs(output_dir: Path, iedb_filename: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_files = []
    checked_records = 0
    for path in sorted(output_dir.glob("*.jsonl")):
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
                label = record.get("label")
                if label is not None and (type(label) is not int or label not in {0, 1}):
                    raise ValueError(f"{path}:{line_number}: label must be null, 0, or 1")
                if path.name == iedb_filename and (
                    label is not None
                    or record.get("protection_status")
                    != "not_established_by_this_aggregate_record"
                ):
                    raise ValueError(
                        f"{path}:{line_number}: IEDB aggregate cannot assert protection"
                    )
                count += 1
        checked_records += count
        normalized_files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "records": count,
            }
        )
    validation = {
        "jsonl_files_checked": len(normalized_files),
        "jsonl_records_checked": checked_records,
        "all_jsonl_parse": True,
        "label_shapes_checked": True,
        "no_negative_labels_created": True,
        "validated_at": datetime.now(timezone.utc).date().isoformat(),
    }
    return normalized_files, validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download bacterial-source antigen evidence from the public IEDB IQ-API."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2].parent / "data",
        help="Root data directory containing raw/ and processed/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Existing public-source processing batch; defaults to the latest batch.",
    )
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Re-validate an existing output batch without making network requests.",
    )
    args = parser.parse_args()
    if not 1 <= args.page_size <= PAGE_SIZE:
        parser.error(f"--page-size must be between 1 and {PAGE_SIZE}")
    if args.delay_seconds < 0:
        parser.error("--delay-seconds cannot be negative")

    raw_dir = args.data_root / "raw" / "public_training_sources" / "reference_databases" / "IEDB"
    if args.output_dir is None:
        output_dirs = sorted(
            (args.data_root / "processed").glob("public_training_sources_*"),
            key=lambda path: path.name,
        )
        if not output_dirs:
            raise FileNotFoundError("No public_training_sources output batch exists")
        output_dir = output_dirs[-1]
    else:
        output_dir = args.output_dir
    audit_path = output_dir / "source_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Expected existing generated audit: {audit_path}")
    if args.validate_existing:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        normalized_files, validation = validate_outputs(
            output_dir, "iedb_bacterial_antigen_search.jsonl"
        )
        audit["normalized_files"] = normalized_files
        audit["validation"] = validation
        audit["updated_at"] = datetime.now(timezone.utc).isoformat()
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_dir / "query_manifest.json"
    normalized_path = output_dir / "iedb_bacterial_antigen_search.jsonl"
    if manifest_path.exists() or normalized_path.exists():
        raise FileExistsError(
            "IEDB query artifacts already exist; refusing to overwrite: "
            f"{manifest_path}, {normalized_path}"
        )

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "in_progress",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_url": API_URL,
        "query": {
            "select": SELECT,
            "parent_source_antigen_iri_search": FILTER,
            "filter_semantics": (
                "PostgREST array containment for taxon_protein:2; includes bacterial "
                "taxonomy descendants in the IEDB antigen search index."
            ),
            "ordering": "parent_source_antigen_id ascending",
            "page_size": args.page_size,
        },
        "license_status": (
            "Public API; exact data license terms were not independently confirmed "
            "from the retrieved IEDB citation page. Preserve attribution and verify "
            "before redistribution."
        ),
        "response_pages": [],
    }
    write_json(manifest_path, manifest)

    source_records: list[dict[str, Any]] = []
    total_count: int | None = None
    offset = 0
    page_number = 0
    while total_count is None or offset < total_count:
        query = urlencode(
            {
                "select": SELECT,
                "parent_source_antigen_iri_search": FILTER,
                "order": "parent_source_antigen_id.asc",
            }
        )
        url = f"{API_URL}?{query}"
        start = offset
        end = offset + args.page_size - 1
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Prefer": "count=exact",
                "Range-Unit": "items",
                "Range": f"{start}-{end}",
                "User-Agent": USER_AGENT,
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            status = response.status
            content_range = response.headers.get("Content-Range", "")
            payload = response.read()
            retrieved_at = datetime.now(timezone.utc).isoformat()
        match = re.fullmatch(r"(\d+)-(\d+)/(\d+|\*)", content_range)
        if status not in {200, 206} or not match:
            raise RuntimeError(
                f"Unexpected IEDB API response: HTTP {status}, Content-Range={content_range!r}"
            )
        response_start, response_end = int(match.group(1)), int(match.group(2))
        response_total = None if match.group(3) == "*" else int(match.group(3))
        if response_start != start or response_end < response_start:
            raise RuntimeError(
                f"Unexpected IEDB range: requested {start}-{end}, received {content_range}"
            )
        if total_count is None:
            if response_total is None:
                raise RuntimeError("IEDB did not provide an exact result count")
            total_count = response_total
        elif response_total is not None and response_total != total_count:
            raise RuntimeError(
                f"IEDB result count changed during pagination: {total_count} -> {response_total}"
            )
        page_path = raw_dir / f"antigen_search_bacteria_page_{page_number:03d}.json"
        if page_path.exists():
            raise FileExistsError(f"Refusing to overwrite raw response: {page_path}")
        page_path.write_bytes(payload)
        page_manifest = {
            "request_url": url,
            "request_headers": {
                "Prefer": "count=exact",
                "Range-Unit": "items",
                "Range": f"{start}-{end}",
            },
            "retrieved_at": retrieved_at,
            "http_status": status,
            "content_range": content_range,
            "path": str(page_path),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        manifest["response_pages"].append(page_manifest)
        write_json(manifest_path, manifest)
        page_records = json.loads(payload)
        if not isinstance(page_records, list):
            raise ValueError("IEDB API page must contain a JSON array")
        expected_count = min(args.page_size, total_count - offset)
        if len(page_records) != expected_count:
            raise RuntimeError(
                f"IEDB page {page_number} has {len(page_records)} rows; expected {expected_count}"
            )
        page_manifest["records"] = len(page_records)
        write_json(manifest_path, manifest)
        for page_row, row in enumerate(page_records):
            if not isinstance(row, dict):
                raise ValueError(f"IEDB page {page_number} row {page_row} is not an object")
            source_records.append(
                {
                    "record_id": (
                        f"IEDB:antigen_search:{row.get('parent_source_antigen_id')}:"
                        f"{offset + page_row + 1}"
                    ),
                    "source": "Immune Epitope Database public IQ-API",
                    "source_page": page_number,
                    "source_request_url": url,
                    "source_retrieved_at": retrieved_at,
                    "source_row_number": offset + page_row + 1,
                    "task": "H2_related_antigen_evidence",
                    "endpoint": (
                        "IEDB aggregated antigen/epitope assay annotations; not a "
                        "vaccine-protection endpoint"
                    ),
                    "label": None,
                    "protection_status": "not_established_by_this_aggregate_record",
                    "parent_source_antigen_id": row.get("parent_source_antigen_id"),
                    "parent_source_antigen_names": row.get("parent_source_antigen_names"),
                    "source_organism_taxonomy_id": row.get(
                        "parent_source_antigen_source_org_iri"
                    ),
                    "source_organism": row.get("parent_source_antigen_source_org_name"),
                    "qualitative_measures_across_assays": row.get("qualitative_measures"),
                    "reference_titles": row.get("reference_titles"),
                    "pubmed_ids": row.get("pubmed_ids"),
                    "iedb_reference_ids": row.get("reference_ids"),
                    "iedb_assay_ids": row.get("iedb_assay_ids"),
                    "reported_epitope_sequences": row.get("linear_sequences"),
                    "reported_epitope_lengths": row.get("linear_sequence_lengths"),
                    "source_fields": row,
                }
            )
        offset += len(page_records)
        page_number += 1
        if offset < total_count and args.delay_seconds:
            time.sleep(args.delay_seconds)

    identifiers = [
        row.get("parent_source_antigen_id") for row in source_records
    ]
    nonempty_identifiers = [value for value in identifiers if value]
    unique_identifiers = len(set(nonempty_identifiers))
    with normalized_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in source_records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["total_records"] = len(source_records)
    manifest["unique_nonempty_antigen_ids"] = unique_identifiers
    manifest["repeated_antigen_id_rows"] = len(nonempty_identifiers) - unique_identifiers
    manifest["normalized_file"] = {
        "path": str(normalized_path),
        "bytes": normalized_path.stat().st_size,
        "sha256": sha256_file(normalized_path),
        "records": len(source_records),
    }
    write_json(manifest_path, manifest)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["source_files"]["iedb_bacterial_antigen_search"] = {
        "source_url": API_URL,
        "query_manifest": str(manifest_path),
        "response_pages": manifest["response_pages"],
        "license_status": manifest["license_status"],
        "status": "retrieved_via_official_public_API",
    }
    audit["datasets"]["iedb_bacterial_antigen_search"] = {
        "input_query_manifest": str(manifest_path),
        "output": str(normalized_path),
        "records": len(source_records),
        "unique_nonempty_antigen_ids": unique_identifiers,
        "repeated_antigen_id_rows": len(nonempty_identifiers) - unique_identifiers,
        "source_taxonomy_filter": FILTER,
        "negative_or_protection_labels_created": 0,
        "license_status": manifest["license_status"],
    }
    normalized_files, validation = validate_outputs(
        output_dir, normalized_path.name
    )
    audit["normalized_files"] = normalized_files
    audit["validation"] = validation
    audit["updated_at"] = datetime.now(timezone.utc).isoformat()
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "total_records": len(source_records),
                "unique_antigen_ids": unique_identifiers,
                "repeated_antigen_id_rows": len(nonempty_identifiers) - unique_identifiers,
                "response_pages": len(manifest["response_pages"]),
                "normalized_file": manifest["normalized_file"],
                "query_manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
