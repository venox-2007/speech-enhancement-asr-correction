"""ASR baseline CLI: audio file -> Whisper -> transcript -> WER.

Examples::

    python -m src.asr.baseline --audio path/to/utt.wav --reference "the reference text"
    python -m src.asr.baseline --audio utt.wav --reference-file utt.txt --model small.en --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.asr.whisper_asr import DEFAULT_MODEL, ASRConfig, ASRError, WhisperASR
from src.evaluation.wer import alignment_report, compute_wer
from src.preprocessing.audio_io import AudioLoadError
from src.utils.device import VALID_DEVICES
from src.utils.logging_utils import setup_logging

logger = logging.getLogger("src.asr.baseline")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.asr.baseline",
        description="Transcribe an audio file with Whisper and score it against a reference (WER).",
    )
    p.add_argument("--audio", required=True, type=Path, help="Path to the audio file (WAV/FLAC/OGG).")
    ref = p.add_mutually_exclusive_group()
    ref.add_argument("--reference", help="Reference transcript text.")
    ref.add_argument("--reference-file", type=Path, help="Text file containing the reference transcript.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model name (default: {DEFAULT_MODEL}).")
    p.add_argument("--device", default="auto", choices=VALID_DEVICES, help="Compute device (default: auto).")
    p.add_argument("--language", default="en", help="Language code, or 'auto' to let Whisper detect it.")
    p.add_argument("--no-normalize", action="store_true", help="Score raw text (keep case and punctuation).")
    p.add_argument("--show-alignment", action="store_true", help="Print the word-level alignment.")
    p.add_argument("--json-out", type=Path, help="Write transcription + WER to this JSON file.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p


def read_reference(args: argparse.Namespace) -> str | None:
    if args.reference is not None:
        return args.reference
    if args.reference_file is not None:
        if not args.reference_file.is_file():
            raise FileNotFoundError(f"Reference file not found: {args.reference_file}")
        return args.reference_file.read_text(encoding="utf-8").strip()
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference = read_reference(args)
    config = ASRConfig(
        model_name=args.model,
        device=args.device,
        language=None if args.language == "auto" else args.language,
    )
    asr = WhisperASR(config)
    transcription = asr.transcribe(args.audio)

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audio": str(args.audio),
        "config": vars(config) | {"resolved_device": asr.device},
        "transcription": transcription.to_dict(),
        "reference": reference,
        "wer": None,
    }

    print(f"\nTranscript : {transcription.text}")
    if reference is None:
        logger.info("No reference supplied; skipping WER.")
    else:
        wer = compute_wer(reference, transcription.text, normalize=not args.no_normalize)
        record["wer"] = wer.to_dict()
        print(f"Reference  : {reference}")
        print(
            f"WER        : {wer.wer:.4f}  "
            f"(S={wer.substitutions} D={wer.deletions} I={wer.insertions} "
            f"H={wer.hits} N={wer.reference_words})"
        )
        if args.show_alignment:
            print("\n" + alignment_report(reference, transcription.text, not args.no_normalize))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info("Saved results to %s", args.json_out)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        run(args)
    except (FileNotFoundError, AudioLoadError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except (ASRError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
