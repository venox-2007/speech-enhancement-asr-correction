"""Download official VoiceBank+DEMAND archives and optionally resample to 16 kHz.

python scripts/download_dataset.py --split all --resample
Dataset attribution, license and checksums: data/README.md and data/sources.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path, destination: Path) -> int:
    """Extract dataset WAV/text files to one named folder without traversing paths."""
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            name = PurePosixPath(entry.filename.replace("\\", "/"))
            if entry.is_dir() or "__MACOSX" in name.parts or name.name.startswith("._"):
                continue
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe ZIP member: {entry.filename}")
            if name.suffix.lower() not in {".wav", ".txt"}:
                continue
            if name.name in seen:
                raise ValueError(f"Duplicate dataset filename: {name.name}")
            seen.add(name.name)
            target = destination / name.name
            if target.exists() and target.stat().st_size == entry.file_size:
                continue
            with source.open(entry) as inp, target.open("wb") as out:
                shutil.copyfileobj(inp, out)
    return len(seen)


def download(entry: dict, root: Path) -> None:
    archive = root / entry["archive"]
    expected = entry.get("sha256")
    if not archive.exists() or (expected and sha256(archive) != expected):
        temporary = archive.with_suffix(".zip.part")
        req = urllib.request.Request(entry["url"], headers={"User-Agent": "SpeechEnhancement-ASR/1.0"})
        print(f"Downloading {entry['archive']} ...", flush=True)
        with urllib.request.urlopen(req, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        if expected and sha256(temporary) != expected:
            raise ValueError(f"Archive checksum mismatch: {temporary}")
        with zipfile.ZipFile(temporary) as source:
            if source.testzip() is not None:
                raise ValueError(f"Corrupt archive: {temporary}")
        temporary.replace(archive)
    count = extract_archive(archive, root / entry["directory"])
    if count != entry["expected_files"]:
        raise ValueError(f"{entry['archive']}: expected {entry['expected_files']} files, got {count}")
    print(f"Verified and extracted {count} files: {entry['directory']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["test", "all"], default="test")
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/voicebank_demand")
    args = parser.parse_args()
    raw = args.raw_dir.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))
    selected = [x for x in manifest["archives"] if args.split == "all" or x["split"] == "test"]
    for entry in selected:
        download(entry, raw)
    if args.resample:
        output = ROOT / "data/processed/voicebank_demand_16k"
        subprocess.run([sys.executable, "-m", "src.preprocessing.resample_dataset",
                        "--input-dir", str(raw), "--output-dir", str(output), "--target-sr", "16000"],
                       cwd=ROOT, check=True)
        for entry in selected:
            if entry["directory"].endswith("_txt"):
                shutil.copytree(raw / entry["directory"], output / entry["directory"], dirs_exist_ok=True)
    print("Dataset restoration complete. See data/README.md for validation and training.")


if __name__ == "__main__":
    main()
