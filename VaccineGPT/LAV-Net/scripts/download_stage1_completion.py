from __future__ import annotations

import csv
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen


STUDIES = [
    "PRJNA1451832",
    "PRJNA872296",
    "PRJNA1051295",
    "PRJNA1079515",
    "PRJNA1082880",
    "PRJNA480066",
]

ENA_URL = (
    "https://www.ebi.ac.uk/ena/portal/api/search"
    "?result=read_run&query=study_accession%3D{study}"
    "&fields=run_accession%2Csample_accession%2Cexperiment_accession%2C"
    "fastq_ftp%2Cfastq_md5%2Cfastq_bytes%2Clibrary_strategy%2C"
    "experiment_title%2Cstudy_accession&format=tsv&limit=2000"
)
UNCLUST_URL = (
    "https://wwwuser.gwdg.de/~compbiol/uniclust/2022_02/"
    "UniRef30_2022_02_hhsuite.tar.gz"
)

def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "LAV-Net/0.2"})
    with urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def fetch_metadata(root: Path) -> list[Path]:
    output = root / "data" / "raw" / "stage1_manifests"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for study in STUDIES:
        path = output / f"{study}_tnseq_runs.tsv"
        if not path.exists():
            path.write_text(fetch_text(ENA_URL.format(study=study)), encoding="utf-8")
        paths.append(path)
    return paths


def build_jobs(manifests: list[Path], root: Path) -> list[dict]:
    jobs = []
    for manifest in manifests:
        with manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if (row.get("library_strategy") or "").lower() != "tn-seq":
                    continue
                urls = [item for item in (row.get("fastq_ftp") or "").split(";") if item]
                md5s = [item for item in (row.get("fastq_md5") or "").split(";") if item]
                sizes = [item for item in (row.get("fastq_bytes") or "").split(";") if item]
                for index, source in enumerate(urls):
                    filename = Path(source).name
                    destination = (
                        root / "data" / "raw" / "stage1_fastq"
                        / row["study_accession"] / row["run_accession"] / filename
                    )
                    jobs.append(
                        {
                            "study": row["study_accession"],
                            "run": row["run_accession"],
                            "url": "https://" + source,
                            "destination": destination,
                            "md5": md5s[index] if index < len(md5s) else "",
                            "bytes": int(sizes[index]) if index < len(sizes) and sizes[index].isdigit() else None,
                        }
                    )
    return jobs


def download_job(job: dict) -> dict:
    destination: Path = job["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        size_ok = job["bytes"] is None or destination.stat().st_size == job["bytes"]
        digest = md5_file(destination)
        if size_ok and (not job["md5"] or digest == job["md5"]):
            return {"study": job["study"], "run": job["run"], "status": "cached", "path": str(destination)}
        if size_ok and job["md5"] and digest != job["md5"]:
            destination.replace(destination.with_suffix(destination.suffix + ".invalid"))
    command = [
        "curl.exe", "--noproxy", "*", "-L", "--fail", "--retry", "20", "--retry-all-errors",
        "--retry-delay", "10", "--connect-timeout", "60", "--http1.1",
        "--continue-at", "-", "--silent", "--show-error",
        "--output", str(destination), job["url"],
    ]
    try:
        subprocess.run(command, check=True, timeout=24 * 60 * 60)
        digest = md5_file(destination)
        size = destination.stat().st_size
        if job["bytes"] is not None and size != job["bytes"]:
            return {"study": job["study"], "run": job["run"], "status": "failed", "error": f"size {size} != {job['bytes']}"}
        if job["md5"] and digest != job["md5"]:
            fallback = destination.with_suffix(destination.suffix + ".ftp")
            fallback_command = [
                "curl.exe", "--noproxy", "*", "--ftp-pasv", "--fail",
                "--retry", "20", "--retry-all-errors", "--retry-delay", "10",
                "--connect-timeout", "60", "--continue-at", "-", "--silent", "--show-error",
                "--output", str(fallback), job["url"].replace("https://", "ftp://", 1),
            ]
            subprocess.run(fallback_command, check=True, timeout=24 * 60 * 60)
            fallback_digest = md5_file(fallback)
            if fallback_digest != job["md5"]:
                return {
                    "study": job["study"], "run": job["run"], "status": "failed",
                    "error": f"md5 HTTPS={digest} FTP={fallback_digest} != {job['md5']}",
                }
            destination.replace(destination.with_suffix(destination.suffix + ".https"))
            fallback.replace(destination)
        return {"study": job["study"], "run": job["run"], "status": "downloaded", "path": str(destination)}
    except Exception as exc:
        return {"study": job["study"], "run": job["run"], "status": "failed", "error": str(exc)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--evolution-db", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifests = fetch_metadata(root)
    if args.metadata_only:
        print(json.dumps({"manifests": [str(path) for path in manifests]}, indent=2))
        return
    jobs = build_jobs(manifests, root)
    manifest_root = root / "data" / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "stage1_download_jobs.json").write_text(
        json.dumps(
            [
                {
                    **job,
                    "destination": str(job["destination"]),
                }
                for job in jobs
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.evolution_db:
        db_path = root / "data" / "raw" / "msa_db" / Path(UNCLUST_URL).name
        db_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "curl.exe", "--noproxy", "*", "-L", "--fail", "--retry", "20", "--retry-all-errors",
            "--retry-delay", "10", "--connect-timeout", "60", "--http1.1",
            "--continue-at", "-", "--silent", "--show-error",
            "--output", str(db_path), UNCLUST_URL,
        ]
        subprocess.run(command, check=True, timeout=7 * 24 * 60 * 60)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download_job, jobs))
    (manifest_root / "stage1_download_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps({"jobs": len(jobs), "failed": sum(result["status"] == "failed" for result in results)}, indent=2))


if __name__ == "__main__":
    main()
