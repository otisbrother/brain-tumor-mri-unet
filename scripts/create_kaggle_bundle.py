#!/usr/bin/env python
"""Create a Kaggle-compatible source ZIP with POSIX archive paths."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ENTRIES = (
    "app",
    "configs",
    "debug",
    "notebooks",
    "scripts",
    "src",
    "tests",
    "README.md",
    "requirements.txt",
    "pytest.ini",
    ".gitignore",
    "LICENSE",
)
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def should_include(path: Path) -> bool:
    """Exclude caches and generated Python bytecode from the source bundle."""
    relative = path.relative_to(PROJECT_ROOT)
    return not (
        any(part in IGNORED_PARTS for part in relative.parts)
        or path.suffix.lower() in IGNORED_SUFFIXES
        or path.name == "03_staged_kaggle.ipynb"
    )


def create_bundle(output_path: Path) -> tuple[Path, int]:
    """Write project source with forward-slash member names required by Kaggle."""
    destination = output_path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry_name in SOURCE_ENTRIES:
            entry = PROJECT_ROOT / entry_name
            if not entry.exists():
                continue
            files = [entry] if entry.is_file() else sorted(entry.rglob("*"))
            for file_path in files:
                if file_path.is_file() and should_include(file_path):
                    archive.write(file_path, arcname=file_path.relative_to(PROJECT_ROOT).as_posix())
                    file_count += 1
    return destination, file_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "brain-tumor-source.zip",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bundle, count = create_bundle(args.output)
    print(f"Created {bundle} with {count} source files.")
