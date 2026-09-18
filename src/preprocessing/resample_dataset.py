"""Offline preprocessing script to resample VoiceBank+DEMAND to 16 kHz.

This script converts all audio in VoiceBank+DEMAND subdirectories to
16 kHz mono once on disk, avoiding expensive on-the-fly polyphase
resampling during GPU training.

Expected directory structure is preserved::

    <output_dir>/
      clean_trainset_28spk_wav/
      noisy_trainset_28spk_wav/
      clean_testset_wav/
      noisy_testset_wav/

Usage::

    python -m src.preprocessing.resample_dataset \
        --input-dir data/raw/voicebank_demand \
        --output-dir data/processed/voicebank_demand_16k \
        --target-sr 16000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import soundfile as sf

from src.preprocessing.audio_io import WHISPER_SAMPLE_RATE, load_audio
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_SUBDIRS = [
    "clean_trainset_28spk_wav",
    "noisy_trainset_28spk_wav",
    "clean_testset_wav",
    "noisy_testset_wav",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Resample VoiceBank+DEMAND dataset to 16 kHz mono offline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Root directory of the original VoiceBank+DEMAND dataset.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Target directory to write the resampled 16 kHz dataset.",
    )
    p.add_argument(
        "--target-sr",
        type=int,
        default=WHISPER_SAMPLE_RATE,
        help="Target sample rate in Hz (16000 for Whisper and enhancement).",
    )
    p.add_argument(
        "--subdirs",
        nargs="+",
        default=DEFAULT_SUBDIRS,
        help="Subdirectories to process.",
    )
    p.add_argument(
        "--subtype",
        default="PCM_16",
        help="WAV subtype for output files (e.g. PCM_16 or FLOAT).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


def resample_subdirectory(
    in_dir: Path,
    out_dir: Path,
    target_sr: int,
    subtype: str = "PCM_16",
) -> int:
    """Resample all WAV files in `in_dir` to `out_dir`."""
    if not in_dir.is_dir():
        logger.warning("Subdirectory %s does not exist; skipping.", in_dir)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(in_dir.glob("*.wav"))
    if not wav_files:
        logger.warning("No WAV files found in %s.", in_dir)
        return 0

    count = 0
    for file_path in wav_files:
        out_path = out_dir / file_path.name
        audio = load_audio(file_path, target_sr=target_sr)
        sf.write(str(out_path), audio, target_sr, subtype=subtype)
        count += 1
        if count % 1000 == 0:
            logger.info("Processed %d / %d in %s", count, len(wav_files), in_dir.name)

    logger.info("Resampled %d files in %s -> %s", count, in_dir.name, out_dir.name)
    return count


def verify_pairing(output_dir: Path, subdirs: Sequence[str]) -> None:
    """Verify that clean and noisy file sets match in count and filenames."""
    dir_map = {name: output_dir / name for name in subdirs if (output_dir / name).is_dir()}

    pairs_to_check = [
        ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav"),
        ("clean_testset_wav", "noisy_testset_wav"),
    ]

    for clean_name, noisy_name in pairs_to_check:
        if clean_name in dir_map and noisy_name in dir_map:
            clean_files = {p.name for p in dir_map[clean_name].glob("*.wav")}
            noisy_files = {p.name for p in dir_map[noisy_name].glob("*.wav")}
            if clean_files == noisy_files:
                logger.info("Pairing check passed for %s & %s: %d pairs match exactly.",
                            clean_name, noisy_name, len(clean_files))
            else:
                missing_in_noisy = clean_files - noisy_files
                missing_in_clean = noisy_files - clean_files
                if missing_in_noisy:
                    logger.warning("%d clean files missing matching noisy file in %s",
                                   len(missing_in_noisy), noisy_name)
                if missing_in_clean:
                    logger.warning("%d noisy files missing matching clean file in %s",
                                   len(missing_in_clean), clean_name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    logger.info("Starting offline resampling: %s -> %s (target %d Hz)",
                input_dir, output_dir, args.target_sr)

    total_files = 0
    for subdir_name in args.subdirs:
        in_subdir = input_dir / subdir_name
        out_subdir = output_dir / subdir_name
        total_files += resample_subdirectory(
            in_subdir,
            out_subdir,
            target_sr=args.target_sr,
            subtype=args.subtype,
        )

    if total_files == 0:
        logger.error("No audio files were resampled. Check input directory.")
        return 1

    verify_pairing(output_dir, args.subdirs)
    logger.info("Resampling complete! Total files processed: %d", total_files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
