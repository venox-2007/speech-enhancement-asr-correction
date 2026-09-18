"""Dataset validation script for VoiceBank+DEMAND.

Validates both the raw 48 kHz dataset and the processed 16 kHz dataset.

Usage (from project root):
    python -m src.preprocessing.validate_dataset --raw-dir data/raw/voicebank_demand
    python -m src.preprocessing.validate_dataset --processed-dir data/processed/voicebank_demand_16k
    python -m src.preprocessing.validate_dataset --all
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────── expected counts ──────────────────────────────────

EXPECTED_TRAIN_UTTERANCES = 11572
EXPECTED_TEST_UTTERANCES = 824
EXPECTED_TRAIN_SPEAKERS = 28
EXPECTED_TEST_SPEAKERS = 2
EXPECTED_TEST_SPEAKER_IDS = {"p232", "p257"}
EXPECTED_VAL_SPEAKER_IDS = {"p286", "p287"}  # standard literature convention

RAW_EXPECTED_SR = 48000
PROCESSED_EXPECTED_SR = 16000

SUBDIRS = {
    "clean_train": "clean_trainset_28spk_wav",
    "noisy_train": "noisy_trainset_28spk_wav",
    "clean_test":  "clean_testset_wav",
    "noisy_test":  "noisy_testset_wav",
}

# ─────────────────────────── helpers ──────────────────────────────────────────

import re
_SPEAKER_RE = re.compile(r"^(p\d+)_")


def _extract_speaker(name: str) -> str | None:
    m = _SPEAKER_RE.match(name)
    return m.group(1) if m else None


def _wav_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.wav"))


def _check_sample_rate(path: Path, expected_sr: int) -> tuple[bool, int]:
    info = sf.info(str(path))
    return info.samplerate == expected_sr, info.samplerate


def _check_clipping(path: Path, threshold: float = 1.0) -> tuple[bool, float]:
    """Return (clipped, peak_abs)."""
    audio, _ = sf.read(str(path), dtype="float32")
    peak = float(np.abs(audio).max())
    return peak >= threshold, peak

# ─────────────────────────── sub-validators ───────────────────────────────────


def validate_raw(raw_dir: Path, verbose: bool = False) -> bool:
    """Validate raw 48 kHz dataset."""
    ok = True
    logger.info("=" * 60)
    logger.info("Validating RAW dataset: %s", raw_dir)

    for key, subdir_name in SUBDIRS.items():
        subdir = raw_dir / subdir_name
        if not subdir.is_dir():
            logger.error("MISSING directory: %s", subdir)
            ok = False
            continue

        files = _wav_files(subdir)
        logger.info("%s: %d WAV files", subdir_name, len(files))

        if "train" in key:
            if len(files) != EXPECTED_TRAIN_UTTERANCES:
                logger.warning(
                    "Expected %d train files, got %d",
                    EXPECTED_TRAIN_UTTERANCES, len(files)
                )
        else:
            if len(files) != EXPECTED_TEST_UTTERANCES:
                logger.warning(
                    "Expected %d test files, got %d",
                    EXPECTED_TEST_UTTERANCES, len(files)
                )

    # ── Speaker checks ──
    for split_key, clean_key, noisy_key in [
        ("train", "clean_train", "noisy_train"),
        ("test", "clean_test", "noisy_test"),
    ]:
        clean_dir = raw_dir / SUBDIRS[clean_key]
        noisy_dir = raw_dir / SUBDIRS[noisy_key]
        if not clean_dir.is_dir() or not noisy_dir.is_dir():
            continue

        clean_names = {p.name for p in _wav_files(clean_dir)}
        noisy_names = {p.name for p in _wav_files(noisy_dir)}

        missing_noisy = clean_names - noisy_names
        missing_clean = noisy_names - clean_names
        if missing_noisy:
            logger.error("Clean files missing noisy pair: %d files, e.g. %s",
                         len(missing_noisy), next(iter(missing_noisy)))
            ok = False
        if missing_clean:
            logger.error("Noisy files missing clean pair: %d files, e.g. %s",
                         len(missing_clean), next(iter(missing_clean)))
            ok = False

        if not missing_noisy and not missing_clean:
            logger.info("%s split: %d perfectly matched clean/noisy pairs ✓",
                        split_key, len(clean_names))

        speakers = {_extract_speaker(n) for n in clean_names} - {None}
        logger.info("%s speakers: %s", split_key, sorted(speakers))

        if split_key == "test":
            if speakers != EXPECTED_TEST_SPEAKER_IDS:
                logger.warning(
                    "Test speakers %s differ from expected %s",
                    sorted(speakers), sorted(EXPECTED_TEST_SPEAKER_IDS)
                )
                ok = False
            else:
                logger.info("Test speaker IDs verified: %s ✓", sorted(speakers))
        else:
            if len(speakers) != EXPECTED_TRAIN_SPEAKERS:
                logger.warning(
                    "Expected %d train speakers, found %d: %s",
                    EXPECTED_TRAIN_SPEAKERS, len(speakers), sorted(speakers)
                )

    # ── Sample rate spot-check (check first 5 files per subdir) ──
    for key, subdir_name in SUBDIRS.items():
        subdir = raw_dir / subdir_name
        if not subdir.is_dir():
            continue
        files = _wav_files(subdir)[:5]
        wrong_sr = []
        for f in files:
            correct, sr = _check_sample_rate(f, RAW_EXPECTED_SR)
            if not correct:
                wrong_sr.append((f.name, sr))
        if wrong_sr:
            logger.error("%s: unexpected sample rates: %s", subdir_name, wrong_sr)
            ok = False
        else:
            logger.info("%s: sample rate spot-check OK (48 kHz on 5 files) ✓", subdir_name)

    return ok


def validate_processed(proc_dir: Path, check_clipping: bool = True, verbose: bool = False) -> bool:
    """Validate processed 16 kHz dataset."""
    ok = True
    logger.info("=" * 60)
    logger.info("Validating PROCESSED dataset: %s", proc_dir)

    for key, subdir_name in SUBDIRS.items():
        subdir = proc_dir / subdir_name
        if not subdir.is_dir():
            logger.error("MISSING processed directory: %s", subdir)
            ok = False
            continue

        files = _wav_files(subdir)
        logger.info("%s: %d WAV files", subdir_name, len(files))

        if "train" in key:
            expected = EXPECTED_TRAIN_UTTERANCES
        else:
            expected = EXPECTED_TEST_UTTERANCES

        if len(files) != expected:
            logger.warning("Expected %d files, got %d in %s",
                           expected, len(files), subdir_name)

    # ── Pairing checks ──
    for clean_key, noisy_key in [("clean_train", "noisy_train"), ("clean_test", "noisy_test")]:
        clean_dir = proc_dir / SUBDIRS[clean_key]
        noisy_dir = proc_dir / SUBDIRS[noisy_key]
        if not clean_dir.is_dir() or not noisy_dir.is_dir():
            continue
        clean_names = {p.name for p in _wav_files(clean_dir)}
        noisy_names = {p.name for p in _wav_files(noisy_dir)}
        if clean_names == noisy_names:
            logger.info("Pairing: %s ↔ %s — %d pairs match ✓",
                        SUBDIRS[clean_key], SUBDIRS[noisy_key], len(clean_names))
        else:
            logger.error("Pairing mismatch: %d extra clean, %d extra noisy",
                         len(clean_names - noisy_names), len(noisy_names - clean_names))
            ok = False

    # ── Sample rate, duration, clipping spot-check ──
    stats: dict[str, list] = {"durations": [], "peaks": []}

    for key, subdir_name in SUBDIRS.items():
        subdir = proc_dir / subdir_name
        if not subdir.is_dir():
            continue
        files = _wav_files(subdir)
        check_files = files[:10]   # spot check first 10

        wrong_sr = []
        for f in check_files:
            correct, sr = _check_sample_rate(f, PROCESSED_EXPECTED_SR)
            if not correct:
                wrong_sr.append((f.name, sr))
            else:
                info = sf.info(str(f))
                stats["durations"].append(info.duration)

        if wrong_sr:
            logger.error("%s: wrong sample rate (expected 16000 Hz): %s", subdir_name, wrong_sr)
            ok = False
        else:
            logger.info("%s: 16 kHz sample rate OK on %d files ✓", subdir_name, len(check_files))

    # Duration stats
    if stats["durations"]:
        durations = stats["durations"]
        logger.info(
            "Duration stats (spot sample of %d files): min=%.2fs mean=%.2fs max=%.2fs",
            len(durations), min(durations), sum(durations) / len(durations), max(durations)
        )

    # Clipping check (noisy_train, spot 20 files)
    if check_clipping:
        noisy_train_dir = proc_dir / SUBDIRS["noisy_train"]
        if noisy_train_dir.is_dir():
            files = _wav_files(noisy_train_dir)[:20]
            clipped = []
            peaks = []
            for f in files:
                is_clipped, peak = _check_clipping(f)
                peaks.append(peak)
                if is_clipped:
                    clipped.append((f.name, peak))
            if clipped:
                logger.warning(
                    "%d/%d spot-checked files have peak ≥ 1.0 (potential clipping): %s",
                    len(clipped), len(files), clipped[:3]
                )
            else:
                logger.info(
                    "Clipping check: no clipping found in %d noisy_train spot files. "
                    "Peak range: [%.4f, %.4f] ✓",
                    len(files), min(peaks), max(peaks)
                )

    # ── Speaker IDs on processed test ──
    clean_test_proc = proc_dir / SUBDIRS["clean_test"]
    if clean_test_proc.is_dir():
        test_spk = {_extract_speaker(p.name) for p in _wav_files(clean_test_proc)} - {None}
        logger.info("Processed test speakers: %s", sorted(test_spk))
        if test_spk == EXPECTED_TEST_SPEAKER_IDS:
            logger.info("Test speaker IDs verified in processed set ✓")
        else:
            logger.warning("Processed test speakers %s differ from expected %s",
                           sorted(test_spk), sorted(EXPECTED_TEST_SPEAKER_IDS))

    return ok


def disk_usage(directory: Path) -> str:
    total = sum(f.stat().st_size for f in directory.rglob("*.wav"))
    gb = total / (1024 ** 3)
    return f"{gb:.2f} GB"


# ─────────────────────────── main ─────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate VoiceBank+DEMAND raw and processed datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/voicebank_demand"),
                   help="Path to the raw (48 kHz) dataset.")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed/voicebank_demand_16k"),
                   help="Path to the processed (16 kHz) dataset.")
    p.add_argument("--all", action="store_true",
                   help="Validate both raw and processed datasets.")
    p.add_argument("--no-clipping-check", action="store_true",
                   help="Skip clipping check on processed dataset.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    validate_raw_flag = args.all or not (args.all) and args.raw_dir.is_dir()
    validate_proc_flag = args.all or not (args.all) and args.processed_dir.is_dir()

    results = []

    if args.raw_dir.is_dir():
        logger.info("Raw directory found: %s", args.raw_dir)
        results.append(validate_raw(args.raw_dir, verbose=args.verbose))
        if any((args.raw_dir / SUBDIRS[k]).is_dir() for k in SUBDIRS):
            logger.info("Raw disk usage: %s", disk_usage(args.raw_dir))
    else:
        logger.warning("Raw directory not found: %s (skipping raw validation)", args.raw_dir)

    if args.processed_dir.is_dir():
        has_content = any((args.processed_dir / SUBDIRS[k]).is_dir() for k in SUBDIRS)
        if has_content:
            results.append(validate_processed(
                args.processed_dir,
                check_clipping=not args.no_clipping_check,
                verbose=args.verbose,
            ))
            logger.info("Processed disk usage: %s", disk_usage(args.processed_dir))
        else:
            logger.warning("Processed directory found but empty: %s", args.processed_dir)
    else:
        logger.warning("Processed directory not found: %s (skipping)", args.processed_dir)

    if not results:
        logger.error("No valid dataset directories found. Check --raw-dir and --processed-dir.")
        return 1

    if all(results):
        logger.info("=" * 60)
        logger.info("ALL VALIDATION CHECKS PASSED ✓")
        return 0
    else:
        logger.error("=" * 60)
        logger.error("VALIDATION FAILED — check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
