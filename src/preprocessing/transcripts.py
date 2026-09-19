"""VoiceBank+DEMAND transcript utilities.

Handles acquiring and loading clean reference text transcripts for VoiceBank+DEMAND
from the official Edinburgh DataShare repository (handle 10283/2791).
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

EDINBURGH_TESTSET_TXT_URL = (
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/testset_txt.zip"
)


def download_testset_transcripts(
    dest_dir: str | Path,
    url: str = EDINBURGH_TESTSET_TXT_URL,
    timeout: int = 30,
) -> Path:
    """Download and extract testset_txt.zip from Edinburgh DataShare.

    Parameters
    ----------
    dest_dir : Path or str
        Directory to extract text files into.
    url : str
        URL to testset_txt.zip.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    Path
        Path to the extracted directory containing .txt files.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded and populated
    existing_txts = list(dest_dir.glob("*.txt"))
    if len(existing_txts) >= 824:
        logger.info(
            "Transcripts already present in %s (%d files). Skipping download.",
            dest_dir,
            len(existing_txts),
        )
        return dest_dir

    logger.info("Downloading reference transcripts from %s ...", url)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SpeechEnhancement-ASR-Research/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()

    logger.info("Downloaded %d bytes. Unpacking testset_txt.zip ...", len(data))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if member.endswith(".txt") and not member.startswith("__MACOSX"):
                filename = Path(member).name
                content = zf.read(member).decode("utf-8", errors="replace").strip()
                out_file = dest_dir / filename
                out_file.write_text(content, encoding="utf-8")

    extracted = list(dest_dir.glob("*.txt"))
    logger.info("Successfully extracted %d reference transcripts to %s", len(extracted), dest_dir)
    return dest_dir


def load_transcripts(
    source_path: str | Path,
) -> dict[str, str]:
    """Load reference transcripts from a directory of .txt files or a zip archive.

    Parameters
    ----------
    source_path : Path or str
        Path to directory containing .txt files (or a .zip file).

    Returns
    -------
    dict[str, str]
        Mapping from utterance ID (e.g. 'p232_001') to reference transcript text.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Transcript source not found: {source_path}")

    transcripts: dict[str, str] = {}

    if source_path.is_file() and source_path.suffix == ".zip":
        with zipfile.ZipFile(source_path) as zf:
            for member in zf.namelist():
                if member.endswith(".txt") and not member.startswith("__MACOSX"):
                    stem = Path(member).stem
                    content = zf.read(member).decode("utf-8", errors="replace").strip()
                    transcripts[stem] = content
    elif source_path.is_dir():
        for txt_file in sorted(source_path.glob("*.txt")):
            stem = txt_file.stem
            transcripts[stem] = txt_file.read_text(encoding="utf-8").strip()
    else:
        raise ValueError(f"Invalid transcript source: {source_path}")

    if not transcripts:
        raise ValueError(f"No transcripts found in {source_path}")

    logger.info("Loaded %d reference transcripts from %s", len(transcripts), source_path)
    return transcripts
