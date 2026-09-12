#!/usr/bin/env python
"""Safely extract BraTS archives, validate subjects, and create metadata."""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.discovery import build_metadata, discover_patient_directories  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    if destination != root and root not in destination.parents:
        raise ValueError(f"Archive member escapes extraction directory: {member_name}")
    return destination


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract zip/tar archives after path-traversal and link validation."""
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                _safe_destination(resolved, member.filename)
            handle.extractall(resolved)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as handle:
            for member in handle.getmembers():
                _safe_destination(resolved, member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive links are not allowed: {member.name}")
            handle.extractall(resolved, filter="data")
        return
    raise ValueError(f"Unsupported archive format: {archive}")


def find_archives(raw_dir: Path) -> list[Path]:
    suffixes = (".zip", ".tar", ".tar.gz", ".tgz")
    return sorted(path for path in raw_dir.rglob("*") if path.is_file() and path.name.lower().endswith(suffixes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--archive", type=Path, action="append", help="Archive to extract; may be repeated.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--extract-dir", type=Path, default=Path("data/extracted"))
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--allow-missing-optional-modalities", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_root = Path(config["data"]["root_dir"])
    existing = dataset_root.is_dir() and bool(discover_patient_directories(dataset_root))
    if existing and not args.force_extract:
        print(f"Valid-looking extracted data already exists at {dataset_root}; skipping extraction.")
    else:
        archives = args.archive or find_archives(args.raw_dir)
        if not archives:
            raise SystemExit(f"No supported archives found in {args.raw_dir}. Run the download script first.")
        queue = [archive.resolve() for archive in archives]
        extracted: set[Path] = set()
        while queue:
            archive = queue.pop(0)
            if archive in extracted:
                continue
            archive_name = archive.name.lower()
            is_patient_archive = archive_name.startswith("brats2021_") and "training_data" not in archive_name
            destination = args.extract_dir / archive.name.split(".tar")[0] if is_patient_archive else args.extract_dir
            print(f"Extracting {archive} -> {destination}")
            safe_extract(archive, destination.resolve())
            extracted.add(archive)
            # Kaggle's outer download may contain the large BraTS training tar.
            nested_archives = find_archives(args.extract_dir)
            nested_archives.sort(key=lambda path: ("training_data" not in path.name.lower(), path.name.lower()))
            for nested in nested_archives:
                resolved_nested = nested.resolve()
                if resolved_nested not in extracted and resolved_nested not in queue:
                    queue.append(resolved_nested)
    frame = build_metadata(
        dataset_root,
        config["data"]["metadata_csv"],
        require_all_modalities=not args.allow_missing_optional_modalities,
        skip_invalid=args.skip_invalid,
    )
    print(f"Wrote metadata for {len(frame)} patients to {config['data']['metadata_csv']}")


if __name__ == "__main__":
    main()
