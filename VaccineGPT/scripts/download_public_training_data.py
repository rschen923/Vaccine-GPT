from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


USER_AGENT = "VaccineGPT-public-data-acquisition/1.0 (research provenance)"
ENA_FIELDS = (
    "run_accession,fastq_ftp,fastq_md5,fastq_bytes,library_strategy,"
    "experiment_title,study_title,sample_accession"
)
STUDIES = {
    "PRJNA480066": {"expected_tnseq_runs": {"SRR8208794", "SRR8208795"}},
    "PRJNA1082880": {"expected_tnseq_runs": {"SRR28236933", "SRR28236989"}},
    # ENA currently reports these CRISPRi-seq libraries as OTHER rather than
    # Tn-seq/CRISPRi-seq. Keep the complete study metadata and do not filter it
    # by library_strategy; the processing step resolves assay/condition roles.
    "PRJNA1113837": {"expected_run_count": 70, "select_all_runs": True},
}
STATIC_FILES = (
    {
        "relative_path": "reference_databases/DEG/deg_bacteria.csv.zip",
        "url": "https://tubic.org/deg/public/download/deg_bacteria.csv.zip",
        "source": "DEG 15.2 official bacteria organism/study catalogue (not gene-level labels)",
        "species_scope": "Multi-species catalogue; no Edwardsiella piscicida EIB202 entry found",
        "license_status": "No explicit license verified at retrieval; retain as source data and do not redistribute",
    },
    {
        "relative_path": "reference_databases/DEG/deg_annotation_p.csv.zip",
        "url": "https://tubic.org/deg/public/download/deg_annotation_p.csv.zip",
        "source": "DEG official bacteria essential-gene annotation download",
        "species_scope": "Multi-species gene-level reference; not EIB202-specific labels",
        "license_status": "No explicit license verified at retrieval; retain as source data and do not redistribute",
    },
    {
        "relative_path": "reference_databases/DEG/DEG10.aa.gz",
        "url": "https://tubic.org/deg/public/download/DEG10.aa.gz",
        "source": "DEG 10 protein-coding essential-gene amino-acid sequences",
        "species_scope": "Multi-species essential-gene sequence reference; not EIB202-specific labels",
        "license_status": "No explicit license verified at retrieval; retain as source data and do not redistribute",
    },
    {
        "relative_path": "reference_databases/DEG/DEG10.nt.gz",
        "url": "https://tubic.org/deg/public/download/DEG10.nt.gz",
        "source": "DEG 10 protein-coding essential-gene nucleotide sequences",
        "species_scope": "Multi-species essential-gene sequence reference; not EIB202-specific labels",
        "license_status": "No explicit license verified at retrieval; retain as source data and do not redistribute",
    },
    {
        "relative_path": "reference_databases/PA_gold_standard/pcbi.1013945.s011.xlsx",
        "url": "https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1013945.s011&type=supplementary",
        "source": "Van Maele et al. 2026, DOI 10.1371/journal.pcbi.1013945, S2 Table",
        "species_scope": "Pseudomonas aeruginosa; GOLD_84 and GOLD_115, not EIB202 labels",
        "license_status": "Article is CC BY; confirm item-specific reuse requirements before redistribution",
    },
    {
        "relative_path": "genomes/EIB202/CP001135.1.gb",
        "url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=CP001135.1&rettype=gbwithparts&retmode=text",
        "source": "NCBI Nucleotide RefSeq/GenBank accession CP001135.1",
        "species_scope": "Edwardsiella piscicida EIB202 chromosome",
        "license_status": "NCBI sequence record; retain accession and retrieval provenance",
    },
)
UNAVAILABLE_SUPPLEMENTS = (
    {
        "source": "Wei et al. 2019, PMC6498171, Supplemental files 1 and 2",
        "urls": [
            "https://pmc.ncbi.nlm.nih.gov/articles/instance/6498171/bin/AEM.00233-19-s0001.pdf",
            "https://pmc.ncbi.nlm.nih.gov/articles/instance/6498171/bin/AEM.00233-19-sd002.xlsx",
        ],
        "status": "not_acquired",
        "reason": "PMC returned an interactive proof-of-work challenge instead of the supplement; no challenge workaround was attempted.",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hashes(path: Path) -> dict[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            sha256.update(chunk)
            md5.update(chunk)
    return {"sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def validate_artifact(path: Path) -> None:
    suffix = path.suffix.casefold()
    with path.open("rb") as stream:
        prefix = stream.read(8)
    if suffix in {".xlsx", ".zip"}:
        if not zipfile.is_zipfile(path):
            raise ValueError(f"expected ZIP/OOXML payload, received a non-archive: {path}")
        with zipfile.ZipFile(path) as archive:
            corrupt_member = archive.testzip()
            if corrupt_member:
                raise ValueError(f"corrupt archive member {corrupt_member!r}: {path}")
            if suffix == ".xlsx" and "xl/workbook.xml" not in archive.namelist():
                raise ValueError(f"OOXML workbook is missing xl/workbook.xml: {path}")
    elif suffix == ".pdf" and not prefix.startswith(b"%PDF-"):
        raise ValueError(f"expected PDF signature, received another payload: {path}")
    elif suffix == ".gb":
        with path.open("rb") as stream:
            header = stream.read(256).decode("ascii", errors="ignore")
        if not header.startswith("LOCUS"):
            raise ValueError(f"expected GenBank LOCUS record: {path}")
    elif suffix == ".gz" and prefix[:2] != b"\x1f\x8b":
        raise ValueError(f"expected gzip-compressed FASTQ: {path}")


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} while fetching {url}")
        return response.read()


def save_immutable_response(path: Path, url: str, content: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        existing = hashlib.sha256(path.read_bytes()).hexdigest()
        if existing != digest:
            raise FileExistsError(f"refusing to replace changed source response: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return {
        "url": url,
        "retrieved_at_utc": utc_now(),
        "bytes": len(content),
        "sha256": digest,
        "stored_at": str(path),
    }


def download_file(
    root: Path,
    relative_path: str,
    url: str,
    *,
    expected_bytes: int | None = None,
    expected_md5: str | None = None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validate_artifact(target)
        result_hashes = hashes(target)
        if expected_bytes is not None and target.stat().st_size != expected_bytes:
            raise FileExistsError(f"existing file has unexpected size; not overwriting: {target}")
        if expected_md5 and result_hashes["md5"].casefold() != expected_md5.casefold():
            raise FileExistsError(f"existing file has unexpected MD5; not overwriting: {target}")
        return {
            **provenance,
            "url": url,
            "stored_at": str(target),
            "bytes": target.stat().st_size,
            **result_hashes,
            "verification": "existing_file_verified",
        }

    temporary = target.with_name(target.name + ".partial")
    if temporary.exists() and (expected_bytes is None or not expected_md5):
        raise FileExistsError(f"partial download requires expected size and MD5 to resume: {temporary}")
    byte_count = temporary.stat().st_size if temporary.exists() else 0
    if expected_bytes is not None and byte_count > expected_bytes:
        raise ValueError(f"partial download exceeds expected size: {temporary}")
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    if byte_count:
        with temporary.open("rb") as existing:
            while chunk := existing.read(1024 * 1024):
                sha256.update(chunk)
                md5.update(chunk)
    if expected_bytes is None or byte_count < expected_bytes:
        headers = {"User-Agent": USER_AGENT}
        if byte_count:
            headers["Range"] = f"bytes={byte_count}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            if byte_count:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if (
                    response.status != 206
                    or match is None
                    or int(match.group(1)) != byte_count
                    or int(match.group(3)) != expected_bytes
                ):
                    raise RuntimeError(
                        f"server did not honor safe resume at byte {byte_count} for {url}: "
                        f"HTTP {response.status}, Content-Range={content_range!r}"
                    )
            elif response.status != 200:
                raise RuntimeError(f"HTTP {response.status} while downloading {url}")
            mode = "ab" if byte_count else "xb"
            with temporary.open(mode) as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    sha256.update(chunk)
                    md5.update(chunk)
                    byte_count += len(chunk)
    if expected_bytes is not None and byte_count != expected_bytes:
        raise ValueError(
            f"download size mismatch for {url}: expected {expected_bytes}, got {byte_count}"
        )
    if expected_md5 and md5.hexdigest().casefold() != expected_md5.casefold():
        raise ValueError(
            f"download MD5 mismatch for {url}: expected {expected_md5}, got {md5.hexdigest()}"
        )
    validate_artifact(temporary)
    temporary.replace(target)
    return {
        **provenance,
        "url": url,
        "stored_at": str(target),
        "retrieved_at_utc": utc_now(),
        "bytes": byte_count,
        "sha256": sha256.hexdigest(),
        "md5": md5.hexdigest(),
        "verification": "ena_md5_and_size" if expected_md5 and expected_bytes else "sha256_recorded",
    }


def fetch_ena_metadata(root: Path, accession: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "accession": accession,
            "result": "read_run",
            "fields": ENA_FIELDS,
            "format": "tsv",
        }
    )
    url = f"https://www.ebi.ac.uk/ena/portal/api/filereport?{query}"
    response = request_bytes(url)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    response_digest = hashlib.sha256(response).hexdigest()
    raw_path = root / "source_responses" / f"ena_{accession}_{date}_{response_digest[:12]}.tsv"
    raw_record = save_immutable_response(raw_path, url, response)
    rows = list(csv.DictReader(io.StringIO(response.decode("utf-8")), delimiter="\t"))
    if not rows:
        raise ValueError(f"ENA returned no read-run rows for {accession}")
    return rows, raw_record


def append_jsonl_immutable(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str], str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (
                str(record.get("stored_at") or record.get("attempted_path") or record.get("url") or ""),
                str(record.get("sha256") or record.get("response_sha256") or ""),
            )
            existing[key] = line
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for record in records:
            key = (
                str(record.get("stored_at") or record.get("attempted_path") or record.get("url") or ""),
                str(record.get("sha256") or record.get("response_sha256") or ""),
            )
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            if key not in existing:
                stream.write(line + "\n")
                existing[key] = line
            elif existing[key] != line:
                comparable_existing = json.loads(existing[key])
                immutable_fields = ("url", "bytes", "sha256", "md5")
                if any(
                    comparable_existing.get(field) != record.get(field)
                    for field in immutable_fields
                    if field in comparable_existing or field in record
                ):
                    raise ValueError(f"manifest conflict for {key}")


def acquire(root: Path, include_raw_fastq: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    acquired: list[dict[str, Any]] = []
    for item in STATIC_FILES:
        acquired.append(
            download_file(
                root,
                item["relative_path"],
                item["url"],
                provenance={
                    "source": item["source"],
                    "species_scope": item["species_scope"],
                    "license_status": item["license_status"],
                    "data_kind": "reference_or_processed_supplement",
                },
            )
        )

    study_summaries = []
    for accession, spec in STUDIES.items():
        rows, raw_response = fetch_ena_metadata(root, accession)
        selected = (
            rows
            if spec.get("select_all_runs")
            else [row for row in rows if row.get("library_strategy", "").casefold() == "tn-seq"]
        )
        observed_runs = {row["run_accession"] for row in selected}
        expected_runs = spec.get("expected_tnseq_runs")
        if expected_runs is not None and observed_runs != expected_runs:
            raise ValueError(
                f"{accession}: expected Tn-seq runs {sorted(expected_runs)}, "
                f"observed {sorted(observed_runs)}; inspect raw ENA metadata"
            )
        expected_count = spec.get("expected_run_count")
        if expected_count is not None and len(observed_runs) != expected_count:
            raise ValueError(
                f"{accession}: expected {expected_count} metadata runs, "
                f"observed {len(observed_runs)}; inspect raw ENA metadata"
            )
        run_downloads = []
        for row in selected:
            files = row["fastq_ftp"].split(";")
            md5s = row["fastq_md5"].split(";")
            byte_sizes = row["fastq_bytes"].split(";")
            if not (len(files) == len(md5s) == len(byte_sizes)):
                raise ValueError(f"{row['run_accession']}: ENA file/checksum/size columns differ")
            for index, (file_url, md5, size) in enumerate(zip(files, md5s, byte_sizes), 1):
                filename = file_url.rsplit("/", 1)[-1]
                url = "https://" + file_url
                relative_path = f"studies/{accession}/fastq/{filename}"
                provenance = {
                    "source": accession,
                    "study_title": row.get("study_title"),
                    "experiment_title": row.get("experiment_title"),
                    "sample_accession": row.get("sample_accession"),
                    "run_accession": row["run_accession"],
                    "library_strategy": row.get("library_strategy"),
                    "species_scope": "Edwardsiella piscicida EIB202",
                    "read_file_index": index,
                    "data_kind": "raw_fastq",
                    "license_status": "public sequence archive; reuse terms not independently adjudicated",
                }
                item = (relative_path, url, int(size), md5, provenance)
                run_downloads.append(item)
        study_summaries.append(
            {
                "accession": accession,
                "all_library_rows": len(rows),
                "tnseq_runs": sorted(observed_runs),
                "selected_runs": sorted(observed_runs),
                "selection_policy": (
                    "all study runs; ENA library_strategy is currently OTHER"
                    if spec.get("select_all_runs")
                    else "library_strategy == Tn-Seq"
                ),
                "non_tnseq_rows_excluded": len(rows) - len(selected),
                "metadata_response": raw_response,
                "fastq_files": len(run_downloads),
                "fastq_expected_bytes": sum(item[2] for item in run_downloads),
            }
        )
        if include_raw_fastq:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        download_file,
                        root,
                        relative_path,
                        url,
                        expected_bytes=size,
                        expected_md5=md5,
                        provenance=provenance,
                    ): relative_path
                    for relative_path, url, size, md5, provenance in run_downloads
                }
                for future in as_completed(futures):
                    record = future.result()
                    acquired.append(record)
                    print(
                        f"verified {record['stored_at']} ({record['bytes']} bytes; "
                        f"{record['verification']})",
                        flush=True,
                    )

    summary = {
        "schema_version": "vaccinegpt-public-training-data-1.0",
        "retrieved_at_utc": utc_now(),
        "data_root": str(root),
        "studies": study_summaries,
        "raw_fastq_downloaded": include_raw_fastq,
        "unavailable_sources": list(UNAVAILABLE_SUPPLEMENTS),
        "files_acquired": len(acquired),
        "total_bytes": sum(int(item["bytes"]) for item in acquired),
        "files": acquired,
        "limitations": [
            "The 2019 target-organism TIS experiment is seawater fitness evidence, not host protection or universal essentiality.",
            "The 2025 EIB202 Tn-seq study has only one listed run per condition and is exploratory without biological replication.",
            "The DEG 15.2 organism catalogue is not gene-level data; its annotation and DEG10 sequence files are cross-species references requiring organism/accession matching.",
            "Pseudomonas and other DEG records are cross-species references, never EIB202 gold labels.",
            "Downloading raw FASTQ does not by itself produce validated insertion counts or fitness calls.",
        ],
    }
    manifest_path = root / "dataset_manifest.json"
    failed_attempts: list[dict[str, Any]] = []
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        blocked_names = {
            url.rsplit("/", 1)[-1]
            for unavailable in UNAVAILABLE_SUPPLEMENTS
            for url in unavailable["urls"]
        }
        failed_attempts = [
            {
                "source": item.get("source"),
                "url": item.get("url"),
                "attempted_path": item.get("stored_at"),
                "response_sha256": item.get("sha256"),
                "status": "rejected_not_a_data_file",
                "reason": "Response was an interactive proof-of-work HTML challenge, not the advertised supplement.",
                "recorded_at_utc": utc_now(),
            }
            for item in previous.get("files", [])
            if Path(item.get("stored_at", "")).name in blocked_names
            and not Path(item.get("stored_at", "")).exists()
        ]
        attempt_log = root / "dataset_manifest.jsonl"
        if attempt_log.exists():
            known_failures = {
                (item.get("url"), item.get("response_sha256"))
                for item in failed_attempts
            }
            for line in attempt_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if (
                    Path(item.get("stored_at", "")).name in blocked_names
                    and not Path(item.get("stored_at", "")).exists()
                    and (item.get("url"), item.get("sha256")) not in known_failures
                ):
                    failed_attempts.append(
                        {
                            "source": item.get("source"),
                            "url": item.get("url"),
                            "attempted_path": item.get("stored_at"),
                            "response_sha256": item.get("sha256"),
                            "status": "rejected_not_a_data_file",
                            "reason": (
                                "Response was an interactive proof-of-work HTML challenge, "
                                "not the advertised supplement."
                            ),
                            "recorded_at_utc": utc_now(),
                        }
                    )
        previous_by_path = {
            item["stored_at"]: item
            for item in previous.get("files", [])
            if Path(item["stored_at"]).is_file()
        }
        current_by_path = {item["stored_at"]: item for item in acquired}
        for path, item in previous_by_path.items():
            if path in current_by_path and item.get("sha256") != current_by_path[path].get("sha256"):
                raise ValueError(f"downloaded artifact changed across runs: {path}")
        merged = dict(summary)
        merged["files"] = list({**previous_by_path, **current_by_path}.values())
        merged["files_acquired"] = len(merged["files"])
        merged["total_bytes"] = sum(int(item["bytes"]) for item in merged["files"])
        merged["raw_fastq_downloaded"] = any(
            item.get("data_kind") == "raw_fastq" for item in merged["files"]
        )
        summary = merged
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_jsonl_immutable(root / "dataset_manifest.jsonl", acquired)
    if failed_attempts:
        append_jsonl_immutable(root / "download_failures.jsonl", failed_attempts)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire public target-organism Tn-seq and essential-gene reference data with checksums."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".." / "data" / "raw" / "public_training_sources",
    )
    parser.add_argument(
        "--skip-raw-fastq",
        action="store_true",
        help="Fetch supplements, reference databases, genome, and ENA metadata only.",
    )
    args = parser.parse_args()
    result = acquire(args.data_root.resolve(), include_raw_fastq=not args.skip_raw_fastq)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
