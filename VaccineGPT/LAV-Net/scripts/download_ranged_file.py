from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
from pathlib import Path


def download_part(job: dict) -> dict:
    """Fetch the missing tail of one fixed-size segment without discarding bytes.

    The provider publishes no checksum, so each segment is validated against its
    exact byte span and the caller validates the merged archive separately.
    """
    index: int = job["index"]
    url: str = job["url"]
    path: Path = job["path"]
    start: int = job["start"]
    end: int = job["end"]
    rate: str | None = job["rate"]
    expected = end - start + 1

    if path.exists() and path.stat().st_size == expected:
        return {"index": index, "status": "cached"}
    if path.exists() and path.stat().st_size > expected:
        path.unlink()

    have = path.stat().st_size if path.exists() else 0
    offset = start + have
    target = path if have == 0 else path.with_name(path.name + ".tail")

    command = [
        "curl.exe", "--noproxy", "*", "-L", "--fail", "--retry", "20",
        "--retry-all-errors", "--retry-delay", "10", "--connect-timeout", "60",
        "--http1.1", "--range", f"{offset}-{end}", "--silent", "--show-error",
    ]
    if rate:
        command += ["--limit-rate", rate]
    command += ["--output", str(target), url]
    subprocess.run(command, check=True, timeout=7 * 24 * 60 * 60)

    if target is not path:
        with path.open("ab") as handle, target.open("rb") as tail:
            while block := tail.read(8 * 1024 * 1024):
                handle.write(block)
        target.unlink()

    if path.stat().st_size != expected:
        raise RuntimeError(
            f"{path.name}: {path.stat().st_size} bytes, expected {expected}"
        )
    return {"index": index, "status": "downloaded"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    parser.add_argument("size", type=int)
    parser.add_argument("--chunk-mib", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rate", default=None, help="per-stream curl --limit-rate value")
    parser.add_argument(
        "--parts-suffix", default=".parts",
        help="parts directory suffix; point at an existing set to resume it",
    )
    args = parser.parse_args()

    parts_dir = args.destination.with_name(args.destination.name + args.parts_suffix)
    parts_dir.mkdir(parents=True, exist_ok=True)
    width = args.chunk_mib * 1024 * 1024
    jobs = []
    for index in range((args.size + width - 1) // width):
        start = index * width
        end = min(args.size, (index + 1) * width) - 1
        if start > end:
            continue
        jobs.append({
            "index": index,
            "url": args.url,
            "path": parts_dir / f"part-{index:03d}",
            "start": start,
            "end": end,
            "rate": args.rate,
        })
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = 0
        for result in pool.map(download_part, jobs):
            completed += 1
            if result["status"] == "downloaded":
                print(f"[{completed}/{len(jobs)}] segment {result['index']:04d} downloaded", flush=True)
    print(f"all {len(jobs)} segments present", flush=True)

    temporary = args.destination.with_name(args.destination.name + ".merged")
    with temporary.open("wb") as output:
        for job in jobs:
            with job["path"].open("rb") as part:
                while chunk := part.read(8 * 1024 * 1024):
                    output.write(chunk)
    if temporary.stat().st_size != args.size:
        raise RuntimeError(f"merged file has {temporary.stat().st_size} bytes, expected {args.size}")
    if args.destination.exists():
        # Keep any earlier partial transfer instead of discarding downloaded bytes.
        args.destination.replace(
            args.destination.with_name(args.destination.name + ".single-stream.bak")
        )
    temporary.replace(args.destination)
    print(f"merged -> {args.destination}", flush=True)


if __name__ == "__main__":
    main()
