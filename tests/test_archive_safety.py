import io
import tarfile
from pathlib import Path

import pytest

from scripts.prepare_dataset import safe_extract


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        payload = b"unsafe"
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="escapes extraction directory"):
        safe_extract(archive, tmp_path / "destination")
    assert not (tmp_path / "outside.txt").exists()

