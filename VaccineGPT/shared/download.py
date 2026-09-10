from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_http(url: str, destination: Path, user_agent: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def download_ncbi_datasets(source: Dict, destination: Path) -> None:
    executable = shutil.which(source.get("required_external", "datasets"))
    if executable is None:
        raise RuntimeError("NCBI source requires the official `datasets` CLI in PATH")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable, "download", "genome", "taxon", source["query"],
        "--include", ",".join(source.get("include", ["genome"])),
        "--filename", str(destination),
    ]
    subprocess.run(command, check=True)


def run_download_manifest(config_path: str | Path) -> Dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    root = Path(config["download_root"])
    metadata_root = Path(config.get("metadata_root", "data/metadata"))
    metadata_root.mkdir(parents=True, exist_ok=True)
    results = []
    for source in config["sources"]:
        if not source.get("enabled", False):
            results.append({"name": source["name"], "status": "disabled"})
            continue
        destination = root / source["output"]
        try:
            if source["kind"] == "http":
                download_http(source["url"], destination, config["user_agent"])
            elif source["kind"] == "ncbi_datasets":
                download_ncbi_datasets(source, destination)
            elif source["kind"] == "manual":
                results.append({
                    "name": source["name"], "status": "manual",
                    "url": source.get("url"), "note": source.get("note"),
                })
                continue
            else:
                raise ValueError(f"unsupported source kind: {source['kind']}")
            results.append({
                "name": source["name"], "status": "downloaded",
                "url": source.get("url", "ncbi datasets CLI"),
                "path": str(destination), "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
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
