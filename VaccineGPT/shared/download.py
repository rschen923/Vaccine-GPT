from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
import urllib.error
import urllib.parse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_http(url: str, destination: Path, user_agent: str, retries: int = 3, timeout: int = 60) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(retries):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": user_agent}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with temporary.open(mode) as output:
                    shutil.copyfileobj(response, output)
            break
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    temporary.replace(destination)


def download_http_with_fallbacks(source: Dict, destination: Path, user_agent: str, retries: int, timeout: int) -> str:
    errors = []
    for url in [source["url"], *source.get("fallback_urls", [])]:
        try:
            download_http(url, destination, user_agent, retries=retries, timeout=timeout)
            return url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if destination.with_suffix(destination.suffix + ".partial").exists():
                destination.with_suffix(destination.suffix + ".partial").unlink()
    raise RuntimeError("all download URLs failed: " + " | ".join(errors))


def validate_download(source: Dict, destination: Path) -> None:
    if source["kind"] == "sra":
        if not destination.exists() or not destination.is_dir():
            raise ValueError(f"SRA output directory was not created: {destination}")
        return
    if not destination.exists() or destination.stat().st_size < int(source.get("min_bytes", 1)):
        raise ValueError(f"download is empty or smaller than min_bytes: {destination}")
    expected = source.get("sha256")
    if expected and sha256(destination) != expected:
        raise ValueError(f"SHA256 mismatch for {destination}")
    expected_type = source.get("content_type")
    if expected_type:
        with destination.open("rb") as handle:
            prefix = handle.read(512).lower()
        if expected_type == "json" and not prefix.lstrip().startswith((b"{", b"[")):
            raise ValueError(f"expected JSON content, got non-JSON: {destination}")
        if expected_type == "tsv" and b"<html" in prefix:
            raise ValueError(f"expected TSV content, got HTML: {destination}")


def download_ncbi_datasets(source: Dict, destination: Path, project_root: Path) -> None:
    executable_name = source.get("required_external", "datasets")
    executable = shutil.which(executable_name)
    local_executable = project_root / executable_name
    if executable is None and local_executable.exists():
        executable = str(local_executable)
    if executable is None:
        raise RuntimeError("NCBI source requires the official `datasets` CLI in PATH")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable, "download", "genome", "taxon", source["query"],
        "--include", ",".join(source.get("include", ["genome"])),
        "--filename", str(destination),
    ]
    if source.get("dehydrated", False):
        command.append("--dehydrated")
    subprocess.run(command, check=True)


def download_paged_json(source: Dict, destination: Path, user_agent: str, timeout: int = 60) -> None:
    """Download a PostgREST-style endpoint without silently truncating results."""
    base_url = source["url"]
    page_size = int(source.get("page_size", 1000))
    max_pages = int(source.get("max_pages", 100000))
    keyset_field = source.get("keyset_field")
    last_value = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as output:
        output.write("[")
        first = True
        for page in range(max_pages):
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(base_url).query))
            query["limit"] = str(page_size)
            if keyset_field and last_value is not None:
                query[keyset_field] = f"gt.{last_value}"
                query["order"] = keyset_field
            elif not keyset_field:
                query["offset"] = str(page * page_size)
            parts = urllib.parse.urlsplit(base_url)
            url = urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))
            request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"paged JSON endpoint returned {type(payload).__name__}, expected list")
            for item in payload:
                if not first:
                    output.write(",")
                output.write(json.dumps(item, ensure_ascii=True, separators=(",", ":")))
                first = False
            if keyset_field and payload:
                last_value = payload[-1].get(keyset_field)
                if last_value is None:
                    raise ValueError(f"keyset field missing from endpoint response: {keyset_field}")
            if len(payload) < page_size:
                break
        else:
            raise RuntimeError("paged JSON endpoint exceeded max_pages; narrow the query")
        output.write("]")
    temporary.replace(destination)


def download_uniprot_search(source: Dict, destination: Path, user_agent: str, timeout: int = 60) -> None:
    """Follow UniProt cursor pagination and write one TSV with a bounded retry loop."""
    url = source["url"]
    max_pages = int(source.get("max_pages", 10000))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as output:
        for page in range(max_pages):
            request = urllib.request.Request(
                url,
                headers={"User-Agent": user_agent, "Accept": "text/tab-separated-values"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                output.write(response.read())
                next_url = None
                for value in response.headers.get_all("Link", []):
                    for item in value.split(","):
                        if 'rel="next"' in item:
                            next_url = item.split("<", 1)[1].split(">", 1)[0]
                            break
                    if next_url:
                        break
            if not next_url:
                break
            url = next_url
        else:
            raise RuntimeError("UniProt pagination exceeded max_pages")
    temporary.replace(destination)


def run_download_manifest(config_path: str | Path, only: list[str] | None = None) -> Dict:
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    project_root = config_file.parent.parent
    root = Path(config["download_root"])
    if not root.is_absolute():
        root = project_root / root
    metadata_root = Path(config.get("metadata_root", "data/metadata"))
    if not metadata_root.is_absolute():
        metadata_root = project_root / metadata_root
    metadata_root.mkdir(parents=True, exist_ok=True)
    results = []
    for source in config["sources"]:
        if only and source["name"] not in only:
            continue
        if not source.get("enabled", False):
            results.append({"name": source["name"], "status": "disabled"})
            continue
        destination = root / source["output"]
        try:
            if destination.exists() and (
                destination.is_dir() or destination.stat().st_size >= int(source.get("min_bytes", 1))
            ):
                try:
                    validate_download(source, destination)
                    results.append({
                        "name": source["name"], "status": "cached",
                        "url": source.get("url", "ncbi datasets CLI"),
                        "path": str(destination.relative_to(project_root)), "bytes": (
                            destination.stat().st_size if destination.is_file() else None
                        ), "sha256": sha256(destination) if destination.is_file() else None,
                    })
                    continue
                except Exception:
                    pass
            if source["kind"] == "http":
                resolved_url = download_http_with_fallbacks(
                    source,
                    destination,
                    config["user_agent"],
                    retries=int(config.get("retries", 3)),
                    timeout=int(config.get("timeout", 60)),
                )
            elif source["kind"] == "paged_json":
                download_paged_json(source, destination, config["user_agent"], int(config.get("timeout", 60)))
                resolved_url = source["url"]
            elif source["kind"] == "uniprot_search":
                download_uniprot_search(source, destination, config["user_agent"], int(config.get("timeout", 60)))
                resolved_url = source["url"]
            elif source["kind"] == "ncbi_datasets":
                download_ncbi_datasets(source, destination, project_root)
                resolved_url = "ncbi datasets CLI"
            elif source["kind"] == "sra":
                executable_name = source.get("required_external", "prefetch")
                executable = shutil.which(executable_name)
                local_executable = project_root / executable_name
                if executable is None and local_executable.exists():
                    executable = str(local_executable)
                if executable is None:
                    raise RuntimeError("SRA source requires NCBI SRA Toolkit (`prefetch`) in PATH")
                destination.mkdir(parents=True, exist_ok=True)
                accession_path = Path(source["accessions"])
                if not accession_path.is_absolute():
                    accession_path = project_root / accession_path
                accessions = accession_path.read_text(encoding="utf-8").splitlines()
                accessions = [item.strip() for item in accessions if item.strip() and not item.startswith("#")]
                if not accessions:
                    raise RuntimeError("SRA accession manifest is empty")
                for accession in accessions:
                    subprocess.run([executable, accession, "--output-directory", str(destination)], check=True)
                resolved_url = "SRA Toolkit"
            elif source["kind"] == "manual":
                results.append({
                    "name": source["name"], "status": "pending",
                    "url": source.get("url"), "note": source.get("note"),
                })
                continue
            else:
                raise ValueError(f"unsupported source kind: {source['kind']}")
            validate_download(source, destination)
            results.append({
                "name": source["name"], "status": "downloaded",
                "url": resolved_url,
                "path": str(destination.relative_to(project_root)),
                "bytes": destination.stat().st_size if destination.is_file() else None,
                "sha256": sha256(destination) if destination.is_file() else None,
            })
        except Exception as exc:
            results.append({"name": source["name"], "status": "failed", "error": str(exc)})
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "results": results,
    }
    output = metadata_root / "download_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
