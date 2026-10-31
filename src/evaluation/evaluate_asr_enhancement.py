"""Downstream Whisper ASR evaluation pipeline for speech enhancement.

Evaluates OpenAI Whisper (e.g. base.en) on the held-out VoiceBank+DEMAND test set (824 utterances)
comparing original noisy audio against U-Net enhanced audio.

Features:
- Normalized Word Error Rate (WER) using jiwer
- Detailed edit counts: Substitutions (S), Deletions (D), Insertions (I), Correct (C / hits), Total (N)
- Corpus-level aggregate WER and per-utterance WER distribution
- Breakdown by speaker (p232, p257)
- Breakdown by input SNR condition (<5 dB, 5-10 dB, 10-15 dB, >=15 dB)
- Persistent transcription caching and resumption for fast iterative execution
- Zero test-speaker leakage validation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from src.asr.whisper_asr import ASRConfig, WhisperASR
from src.enhancement.dataset import (
    DatasetConfig,
    VoiceBankDEMANDDataset,
    _extract_speaker,
)
from src.enhancement.model import load_enhancer_checkpoint
from src.evaluation.evaluate_enhancement import verify_zero_leakage
from src.evaluation.speech_metrics import _assign_snr_bin, calculate_snr
from src.evaluation.wer import compute_wer
from src.preprocessing.transcripts import download_testset_transcripts, load_transcripts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def aggregate_asr_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute corpus-level aggregate WER and error counts overall and by condition."""
    if not records:
        return {"count": 0, "overall": {}}

    def _calc_stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
        total_words = sum(r["ref_words"] for r in recs)
        if total_words == 0:
            return {}

        noisy_s = sum(r["noisy_sub"] for r in recs)
        noisy_d = sum(r["noisy_del"] for r in recs)
        noisy_i = sum(r["noisy_ins"] for r in recs)
        noisy_c = sum(r["noisy_hits"] for r in recs)

        enh_s = sum(r["enhanced_sub"] for r in recs)
        enh_d = sum(r["enhanced_del"] for r in recs)
        enh_i = sum(r["enhanced_ins"] for r in recs)
        enh_c = sum(r["enhanced_hits"] for r in recs)

        noisy_corpus_wer = (noisy_s + noisy_d + noisy_i) / total_words
        enh_corpus_wer = (enh_s + enh_d + enh_i) / total_words
        delta_corpus_wer = enh_corpus_wer - noisy_corpus_wer

        # Per-utterance WER stats
        noisy_wers = [r["noisy_wer"] for r in recs]
        enh_wers = [r["enhanced_wer"] for r in recs]
        delta_wers = [r["delta_wer"] for r in recs]

        return {
            "count": len(recs),
            "total_words": total_words,
            "noisy": {
                "corpus_wer": float(noisy_corpus_wer),
                "substitutions": int(noisy_s),
                "deletions": int(noisy_d),
                "insertions": int(noisy_i),
                "correct_words": int(noisy_c),
                "utt_wer_mean": float(np.mean(noisy_wers)),
                "utt_wer_std": float(np.std(noisy_wers)),
                "utt_wer_median": float(np.median(noisy_wers)),
            },
            "enhanced": {
                "corpus_wer": float(enh_corpus_wer),
                "substitutions": int(enh_s),
                "deletions": int(enh_d),
                "insertions": int(enh_i),
                "correct_words": int(enh_c),
                "utt_wer_mean": float(np.mean(enh_wers)),
                "utt_wer_std": float(np.std(enh_wers)),
                "utt_wer_median": float(np.median(enh_wers)),
            },
            "delta": {
                "delta_corpus_wer": float(delta_corpus_wer),
                "delta_substitutions": int(enh_s - noisy_s),
                "delta_deletions": int(enh_d - noisy_d),
                "delta_insertions": int(enh_i - noisy_i),
                "delta_correct_words": int(enh_c - noisy_c),
                "delta_utt_wer_mean": float(np.mean(delta_wers)),
                "delta_utt_wer_std": float(np.std(delta_wers)),
                "delta_utt_wer_median": float(np.median(delta_wers)),
            },
        }

    overall = _calc_stats(records)

    # Group by speaker
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        spk = r.get("speaker", "unknown")
        by_speaker.setdefault(spk, []).append(r)

    speaker_summary: dict[str, Any] = {}
    for spk, spk_recs in sorted(by_speaker.items()):
        speaker_summary[spk] = _calc_stats(spk_recs)

    # Group by SNR bracket
    by_snr: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        tier = r.get("snr_bracket", "unknown")
        by_snr.setdefault(tier, []).append(r)

    snr_order = [
        "< 5 dB (Low SNR)",
        "5 - 10 dB (Medium-Low SNR)",
        "10 - 15 dB (Medium-High SNR)",
        ">= 15 dB (High SNR)",
        "unknown",
    ]
    snr_summary: dict[str, Any] = {}
    for tier in snr_order:
        if tier in by_snr:
            snr_summary[tier] = _calc_stats(by_snr[tier])

    return {
        "count": len(records),
        "overall": overall,
        "by_speaker": speaker_summary,
        "by_snr_bracket": snr_summary,
    }


def evaluate_asr(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    transcripts_dir: str | Path | None = None,
    enhanced_audio_dir: str | Path | None = None,
    model_name: str = "base.en",
    device: str = "auto",
    resume: bool = True,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run downstream Whisper ASR evaluation on noisy and enhanced speech."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / "asr_cache.json"
    cache: dict[str, dict[str, Any]] = {}
    if resume and cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info("Resumed transcriptions cache with %d entries from %s", len(cache), cache_file)
        except Exception as exc:
            logger.warning("Failed to load existing cache (%s); starting fresh.", exc)

    # Validate only the requested utterances, including small offline datasets.
    test_config = DatasetConfig(root=str(data_dir))
    dataset = VoiceBankDEMANDDataset(test_config, split="test")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    total_items = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    selected_pairs = dataset.pairs[:total_items]
    required_stems = {clean.stem for clean, _ in selected_pairs}

    # 1. Load reference transcripts
    if transcripts_dir is None:
        transcripts_dir = data_dir / "testset_txt"
    transcripts_dir = Path(transcripts_dir)
    existing_stems = {p.stem for p in transcripts_dir.glob("*.txt")}
    if required_stems - existing_stems:
        download_testset_transcripts(transcripts_dir)
    transcripts = load_transcripts(transcripts_dir)
    missing = required_stems - transcripts.keys()
    if missing:
        raise ValueError(f"Missing reference transcripts: {sorted(missing)[:5]}")

    # 2. Load dataset
    logger.info("Found %d test utterances across speakers %s", len(dataset), dataset.speakers)

    # 3. Leakage verification
    try:
        verify_zero_leakage(dataset, data_dir)
    except FileNotFoundError as exc:
        logger.warning("Could not run full dataset-leakage check (%s); proceeding with caution.", exc)

    # 4. Check enhanced audio availability or load enhancement model
    enh_model = None
    if enhanced_audio_dir is not None:
        enh_dir = Path(enhanced_audio_dir)
    else:
        enh_dir = Path("results/enhancement_eval/audio")

    needs_enhancer = any(
        cache.get(clean.stem, {}).get("enh_text") is None and not (enh_dir / clean.name).is_file()
        for clean, _ in selected_pairs
    )
    if needs_enhancer:
        logger.info("Pre-enhanced audio directory %s not fully populated. Loading U-Net enhancer for on-the-fly inference...", enh_dir)
        enh_model = load_enhancer_checkpoint(checkpoint_path, device=device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        enh_model.eval()

    # 5. Load Whisper ASR
    asr_config = ASRConfig(model_name=model_name, device=device, language="en", temperature=0.0)
    asr = WhisperASR(asr_config)

    # 6. Evaluation loop
    records: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    logger.info("Evaluating Whisper ASR on %d utterances...", total_items)
    for idx in range(total_items):
        clean_path, noisy_path = dataset.pairs[idx]
        stem = clean_path.stem
        utt_name = clean_path.name
        speaker = _extract_speaker(utt_name) or "unknown"

        ref_text = transcripts.get(stem)
        if not ref_text:
            raise ValueError(f"Missing reference transcript for utterance '{stem}'")

        # Check cache
        item_cache = cache.get(stem, {})
        noisy_text = item_cache.get("noisy_text")
        enh_text = item_cache.get("enh_text")
        snr_val = item_cache.get("noisy_snr")

        # Compute or load input SNR
        if snr_val is None:
            noisy_audio, clean_audio = dataset[idx]
            snr_val = calculate_snr(clean_audio, noisy_audio)
            item_cache["noisy_snr"] = round(snr_val, 2)

        # Transcribe noisy
        if noisy_text is None:
            res_noisy = asr.transcribe(noisy_path)
            noisy_text = res_noisy.text
            item_cache["noisy_text"] = noisy_text
            item_cache["noisy_time_s"] = res_noisy.processing_time_s

        # Transcribe enhanced
        if enh_text is None:
            pre_enh_path = enh_dir / utt_name
            if pre_enh_path.is_file():
                res_enh = asr.transcribe(pre_enh_path)
                enh_text = res_enh.text
                item_cache["enh_text"] = enh_text
                item_cache["enh_time_s"] = res_enh.processing_time_s
            elif enh_model is not None:
                # Generate on the fly
                noisy_tensor, clean_tensor = dataset[idx]
                dev = next(enh_model.parameters()).device
                with torch.no_grad():
                    enh_tensor = enh_model(noisy_tensor.unsqueeze(0).to(dev)).squeeze(0).cpu()
                enh_np = torch.clamp(enh_tensor, -1.0, 1.0).numpy()
                res_enh = asr.transcribe(enh_np)
                enh_text = res_enh.text
                item_cache["enh_text"] = enh_text
                item_cache["enh_time_s"] = res_enh.processing_time_s
            else:
                raise RuntimeError(f"Enhanced audio not found at {pre_enh_path} and no model loaded.")

        cache[stem] = item_cache

        # Compute WERs
        w_noisy = compute_wer(ref_text, noisy_text, normalize=True)
        w_enh = compute_wer(ref_text, enh_text, normalize=True)

        snr_bracket = _assign_snr_bin(snr_val)

        rec: dict[str, Any] = {
            "clean_file": utt_name,
            "speaker": speaker,
            "reference": w_noisy.reference,
            "noisy_hypothesis": w_noisy.hypothesis,
            "enhanced_hypothesis": w_enh.hypothesis,
            "ref_words": w_noisy.reference_words,
            "noisy_wer": w_noisy.wer,
            "noisy_sub": w_noisy.substitutions,
            "noisy_del": w_noisy.deletions,
            "noisy_ins": w_noisy.insertions,
            "noisy_hits": w_noisy.hits,
            "enhanced_wer": w_enh.wer,
            "enhanced_sub": w_enh.substitutions,
            "enhanced_del": w_enh.deletions,
            "enhanced_ins": w_enh.insertions,
            "enhanced_hits": w_enh.hits,
            "delta_wer": w_enh.wer - w_noisy.wer,
            "delta_hits": w_enh.hits - w_noisy.hits,
            "noisy_snr": snr_val,
            "snr_bracket": snr_bracket,
        }
        records.append(rec)

        # Periodic checkpoint
        if (idx + 1) % 50 == 0 or (idx + 1) == total_items:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
            elapsed = time.perf_counter() - start_time
            rate = (idx + 1) / elapsed
            curr_noisy_wer = sum(r["noisy_sub"] + r["noisy_del"] + r["noisy_ins"] for r in records) / sum(r["ref_words"] for r in records)
            curr_enh_wer = sum(r["enhanced_sub"] + r["enhanced_del"] + r["enhanced_ins"] for r in records) / sum(r["ref_words"] for r in records)
            logger.info(
                "Processed %d/%d utterances (%.1f utt/s) | Current Corpus WER: Noisy %.2f%% vs Enhanced %.2f%% (\u0394 %+.2f%%)",
                idx + 1,
                total_items,
                rate,
                curr_noisy_wer * 100,
                curr_enh_wer * 100,
                (curr_enh_wer - curr_noisy_wer) * 100,
            )

    total_time = time.perf_counter() - start_time
    logger.info("Completed ASR evaluation on %d utterances in %.2f seconds.", len(records), total_time)

    # Save cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    # Save per-utterance CSV
    csv_path = output_dir / "results_per_utterance.csv"
    if records:
        fieldnames = list(records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        logger.info("Saved per-utterance results CSV: %s", csv_path)

    # Aggregate summary
    summary = aggregate_asr_metrics(records)
    summary["metadata"] = {
        "asr_model": model_name,
        "enhancement_checkpoint": str(checkpoint_path),
        "total_utterances": len(records),
        "total_words": summary["overall"]["total_words"],
        "runtime_seconds": round(total_time, 2),
        "device": asr.device,
    }

    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved ASR summary JSON: %s", json_path)

    # Print summary table
    ov = summary["overall"]
    logger.info("=" * 75)
    logger.info("WHISPER DOWNSTREAM ASR EVALUATION SUMMARY (%s, %d Utterances)", model_name, len(records))
    logger.info("=" * 75)
    logger.info("%-18s %-12s %-12s %-12s", "Metric", "Noisy", "Enhanced", "Delta")
    logger.info("-" * 75)
    logger.info(
        "%-18s %10.2f%%   %10.2f%%   %+10.2f%%",
        "Corpus WER",
        ov["noisy"]["corpus_wer"] * 100,
        ov["enhanced"]["corpus_wer"] * 100,
        ov["delta"]["delta_corpus_wer"] * 100,
    )
    logger.info(
        "%-18s %10.2f%%   %10.2f%%   %+10.2f%%",
        "Mean Utt WER",
        ov["noisy"]["utt_wer_mean"] * 100,
        ov["enhanced"]["utt_wer_mean"] * 100,
        ov["delta"]["delta_utt_wer_mean"] * 100,
    )
    logger.info(
        "%-18s %10d   %10d   %+10d",
        "Substitutions (S)",
        ov["noisy"]["substitutions"],
        ov["enhanced"]["substitutions"],
        ov["delta"]["delta_substitutions"],
    )
    logger.info(
        "%-18s %10d   %10d   %+10d",
        "Deletions (D)",
        ov["noisy"]["deletions"],
        ov["enhanced"]["deletions"],
        ov["delta"]["delta_deletions"],
    )
    logger.info(
        "%-18s %10d   %10d   %+10d",
        "Insertions (I)",
        ov["noisy"]["insertions"],
        ov["enhanced"]["insertions"],
        ov["delta"]["delta_insertions"],
    )
    logger.info(
        "%-18s %10d   %10d   %+10d",
        "Correct Words (C)",
        ov["noisy"]["correct_words"],
        ov["enhanced"]["correct_words"],
        ov["delta"]["delta_correct_words"],
    )
    logger.info(
        "%-18s %10d   %10d   %10s",
        "Total Words (N)",
        ov["total_words"],
        ov["total_words"],
        "-",
    )
    logger.info("=" * 75)

    # Per-SNR breakdown
    logger.info("BREAKDOWN BY INPUT SNR TIER:")
    logger.info("%-28s %-12s %-12s %-12s", "SNR Tier (Count)", "Noisy WER", "Enhanced WER", "Delta WER")
    logger.info("-" * 75)
    for tier, s_data in summary["by_snr_bracket"].items():
        cnt = s_data["count"]
        n_wer = s_data["noisy"]["corpus_wer"] * 100
        e_wer = s_data["enhanced"]["corpus_wer"] * 100
        d_wer = s_data["delta"]["delta_corpus_wer"] * 100
        logger.info("%-28s %10.2f%%   %10.2f%%   %+10.2f%%", f"{tier} ({cnt})", n_wer, e_wer, d_wer)
    logger.info("=" * 75)

    return summary


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate downstream Whisper ASR on noisy vs enhanced speech."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/enhancement_train/checkpoint_best.pt",
        help="Path to trained enhancement model checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/voicebank_demand_16k",
        help="Root path to processed 16 kHz VoiceBank+DEMAND dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/asr_enhancement_eval",
        help="Output directory for results, CSV, and summary JSON.",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default=None,
        help="Path to directory containing ground truth reference text files.",
    )
    parser.add_argument(
        "--enhanced-audio-dir",
        type=str,
        default="results/enhancement_eval/audio",
        help="Directory containing pre-enhanced audio WAVs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="base.en",
        help="Whisper model name (default: 'base.en').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Do not resume from cached transcriptions.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum utterances to evaluate (useful for smoke tests).",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parsed = parse_args(args)
    evaluate_asr(
        checkpoint_path=parsed.checkpoint,
        data_dir=parsed.data_dir,
        output_dir=parsed.output_dir,
        transcripts_dir=parsed.transcripts_dir,
        enhanced_audio_dir=parsed.enhanced_audio_dir,
        model_name=parsed.model_name,
        device=parsed.device,
        resume=parsed.resume,
        max_samples=parsed.max_samples,
    )


if __name__ == "__main__":
    main()
