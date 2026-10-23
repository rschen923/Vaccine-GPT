from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify downloaded VaccineGPT artifacts and checksums")
    parser.add_argument("--manifest", default="data/metadata/download_manifest.json")
    parser.add_argument("--output", default="data/metadata/file_inventory.json")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_root = manifest_path.resolve().parents[2]
    checked = []
    for item in manifest.get("results", []):
        path_value = item.get("path")
        path = Path(path_value) if path_value else None
        if path and (not path.is_absolute() or not path.exists()):
            if path.is_absolute() and not path.exists():
                lowered = str(path).replace("/", "\\").lower()
                marker = "\\data\\"
                if marker in lowered:
                    suffix = str(path)[lowered.index(marker) + 1 :]
                    path = project_root / suffix
            path = project_root / path
        exists = bool(path and path.exists())
        actual_sha256 = None
        if exists and path.is_file():
            import hashlib
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
        checked.append({
            "name": item.get("name"),
            "status": item.get("status"),
            "path": str(path) if path else None,
            "exists": exists,
            "bytes": path.stat().st_size if exists and path.is_file() else None,
            "sha256": actual_sha256,
            "checksum_matches": (
                actual_sha256 == item.get("sha256")
                if actual_sha256 and item.get("sha256") else None
            ),
        })
    report = {
        "manifest": str(manifest_path),
        "checked": checked,
        "missing": [item["name"] for item in checked if not item["exists"] and item["status"] in {"downloaded", "cached"}],
        "checksum_failures": [item["name"] for item in checked if item["checksum_matches"] is False],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["missing"] or report["checksum_failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
