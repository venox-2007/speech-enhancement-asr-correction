"""Verify that fresh-clone assets restore safely and dataset ZIPs stay contained."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.download_dataset import extract_archive
from scripts.setup_demo import restore_assets


def test_restore_assets_checks_hash_and_preserves_local_files(tmp_path):
    source = tmp_path / "artifacts/sample.txt"
    source.parent.mkdir()
    source.write_bytes(b"bundled sample")
    manifest = {"restore_files": [{"source": "artifacts/sample.txt", "destination": "data/sample.txt",
                                  "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}
    (tmp_path / "artifacts/manifest.json").write_text(json.dumps(manifest))
    assert restore_assets(tmp_path) == 1
    target = tmp_path / "data/sample.txt"
    assert target.read_bytes() == b"bundled sample"
    target.write_bytes(b"user recording")
    assert restore_assets(tmp_path) == 0
    assert target.read_bytes() == b"user recording"
    source.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore_assets(tmp_path)


def test_extract_archive_supports_flat_and_nested_dataset_members(tmp_path):
    archive = tmp_path / "samples.zip"
    with ZipFile(archive, "w") as source:
        source.writestr("clean_testset_wav/p232_001.wav", b"audio fixture")
        source.writestr("p232_001.txt", b"Please call Stella.")
        source.writestr("__MACOSX/._p232_001.wav", b"metadata")
    output = tmp_path / "output"
    assert extract_archive(archive, output) == 2
    assert (output / "p232_001.txt").read_text() == "Please call Stella."
    assert not (output / "clean_testset_wav").exists()


def test_extract_archive_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as source:
        source.writestr("../outside.txt", b"bad")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        extract_archive(archive, tmp_path / "output")
    assert not (tmp_path / "outside.txt").exists()


def test_bundled_runtime_assets_have_valid_checksums():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "artifacts/manifest.json").read_text())
    assert len(manifest["restore_files"]) == 13
    for item in manifest["restore_files"]:
        assert hashlib.sha256((root / item["source"]).read_bytes()).hexdigest() == item["sha256"]
