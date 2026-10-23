from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def download(url: str, destination: Path, retries: int = 3) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "LAV-Net/0.1 research data downloader"})
            with urlopen(request, timeout=120) as response, destination.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            return {"status": "downloaded", "bytes": destination.stat().st_size, "sha256": digest}
        except Exception as exc:
            errors.append(str(exc))
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    return {"status": "failed", "errors": errors}


def download_raw_reads(metadata_path: Path, destination_root: Path) -> list[dict]:
    """Download every ENA FASTQ listed in a run table and verify provider MD5."""
    jobs = []
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            run = row["run_accession"]
            urls = [item for item in (row.get("fastq_ftp") or "").split(";") if item]
            expected_md5 = [item for item in (row.get("fastq_md5") or "").split(";") if item]
            expected_bytes = [item for item in (row.get("fastq_bytes") or "").split(";") if item]
            if not urls:
                jobs.append({"run_accession": run, "status": "missing_fastq_url"})
                continue
            for index, ftp_path in enumerate(urls):
                url = "https://" + ftp_path if ftp_path.startswith("ftp.") else ftp_path
                name = Path(ftp_path).name
                destination = destination_root / run / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                expected_size = int(expected_bytes[index]) if index < len(expected_bytes) and expected_bytes[index].isdigit() else None
                expected = expected_md5[index] if index < len(expected_md5) else None
                jobs.append({
                    "run_accession": run, "url": url, "destination": destination,
                    "expected": expected, "expected_size": expected_size,
                })

    def fetch(job: dict) -> dict:
        if "status" in job:
            return job
        destination = job["destination"]
        if destination.exists() and (job["expected_size"] is None or destination.stat().st_size == job["expected_size"]):
            digest = hashlib.md5(destination.read_bytes()).hexdigest()
            if job["expected"] is None or digest == job["expected"]:
                return {"run_accession": job["run_accession"], "path": str(destination), "status": "cached", "md5": digest}
            destination.unlink()
        command = [
            "curl.exe", "-L", "--fail", "--retry", "5", "--retry-delay", "3",
            "--continue-at", "-", "--silent", "--show-error",
            "--output", str(destination), job["url"],
        ]
        try:
            subprocess.run(command, check=True, timeout=24 * 60 * 60)
            digest = hashlib.md5(destination.read_bytes()).hexdigest()
            if job["expected"] and digest != job["expected"]:
                ftp_url = job["url"].replace("https://", "ftp://", 1)
                fallback = destination.with_suffix(destination.suffix + ".ftp")
                fallback_command = [
                    "curl.exe", "--ftp-pasv", "--fail", "--retry", "5",
                    "--retry-delay", "3", "--silent", "--show-error",
                    "--output", str(fallback), ftp_url,
                ]
                subprocess.run(fallback_command, check=True, timeout=24 * 60 * 60)
                fallback_digest = hashlib.md5(fallback.read_bytes()).hexdigest()
                if fallback_digest == job["expected"]:
                    destination.replace(destination.with_suffix(destination.suffix + ".https"))
                    fallback.replace(destination)
                    digest = fallback_digest
                else:
                    return {
                        "run_accession": job["run_accession"], "path": str(destination),
                        "status": "failed",
                        "error": (
                            f"MD5 mismatch: HTTPS {digest}, FTP {fallback_digest}; "
                            f"expected {job['expected']}"
                        ),
                    }
            if job["expected_size"] is not None and destination.stat().st_size != job["expected_size"]:
                return {"run_accession": job["run_accession"], "path": str(destination), "status": "failed",
                        "error": f"size mismatch: expected {job['expected_size']}, got {destination.stat().st_size}"}
            return {"run_accession": job["run_accession"], "path": str(destination), "status": "downloaded",
                    "bytes": destination.stat().st_size, "md5": digest}
        except Exception as exc:
            return {"run_accession": job["run_accession"], "path": str(destination), "status": "failed", "error": str(exc)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(fetch, jobs))


def download_uniprot(url: str, destination: Path) -> dict:
    """Follow UniProt cursor pagination so the proteome is not silently capped."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    current = url
    total = 0
    try:
        with destination.open("wb") as handle:
            first_page = True
            while current:
                request = Request(current, headers={"User-Agent": "LAV-Net/0.1"})
                with urlopen(request, timeout=120) as response:
                    payload = response.read()
                    if not first_page:
                        payload = payload.split(b"\n", 1)[1] if b"\n" in payload else b""
                    handle.write(payload)
                    total += len(payload)
                    first_page = False
                    next_url = None
                    for link in response.headers.get_all("Link", []) or []:
                        match = re.search(r"<([^>]+)>\s*;\s*rel=\"next\"", link)
                        if match:
                            next_url = match.group(1)
                            break
                current = next_url
        return {
            "status": "downloaded",
            "bytes": total,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    except Exception as exc:
        return {"status": "failed", "errors": [str(exc)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/public_sources.json")
    parser.add_argument("--include-raw-reads", action="store_true")
    parser.add_argument("--raw-output", default="data/raw/fastq")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / args.manifest).read_text(encoding="utf-8"))
    results = []
    for source in manifest["sources"]:
        destination = root / "data" / "raw" / source["output"]
        fetch = download_uniprot if source.get("kind") == "uniprot_search" else download
        result = {"id": source["id"], "url": source["url"], **fetch(source["url"], destination)}
        result["path"] = str(destination.relative_to(root))
        results.append(result)
        if source.get("raw_reads") and args.include_raw_reads:
            raw_results = download_raw_reads(
                destination,
                root / args.raw_output / source["accession"],
            )
            results.append({
                "id": source["id"] + "_raw_reads",
                "status": "completed" if all(item["status"] in {"downloaded", "cached"} for item in raw_results) else "failed",
                "runs": len(raw_results),
                "files": raw_results,
            })
    output = root / "data" / "manifests" / "download_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": args.manifest,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
