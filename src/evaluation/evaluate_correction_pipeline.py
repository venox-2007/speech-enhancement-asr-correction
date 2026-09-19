"""Evaluation of context-aware ASR correction on VoiceBank+DEMAND test set.

Evaluates the phonetic candidate generator and GPT-2 context-aware language model
correction pipeline on both Noisy and Enhanced Whisper transcriptions across all
824 held-out test utterances.

Key metrics computed:
- Raw WER vs Corrected WER (corpus-level and mean per-utterance)
- Substitutions, deletions, insertions, correct words, total reference words
- Successful corrections (errors repaired to match ground truth)
- Overcorrections (correct words mistakenly altered into errors)
- Failed corrections (errors changed to another incorrect word)
- Unchanged correct words and unchanged errors
- Correction precision: Successful / (Successful + Overcorrections + Failed)
- Per-speaker breakdown (p232, p257)
- Per-input-SNR breakdown (< 5 dB, 5-10 dB, 10-15 dB, >= 15 dB)
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

from src.correction.context_scorer import ContextAwareScorer, ContextScorerConfig
from src.correction.lm_scorer import CausalLMScorer
from src.correction.pipeline import CorrectionPipeline, CorrectionResult, WordCorrection
from src.enhancement.dataset import _extract_speaker
from src.evaluation.correction_eval import (
    FAILED_CORRECTION,
    OVERCORRECTION,
    SUCCESSFUL_CORRECTION,
    UNCHANGED_CORRECT,
    UNCHANGED_ERROR,
    evaluate_correction,
)
from src.evaluation.speech_metrics import _assign_snr_bin
from src.evaluation.wer import compute_wer
from src.preprocessing.transcripts import load_transcripts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def aggregate_condition_stats(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    """Compute corpus-level correction metrics for a single condition prefix ('noisy' or 'enhanced')."""
    total_words = sum(r[f"{prefix}_raw_ref_words"] for r in records)
    if total_words == 0:
        return {}

    raw_s = sum(r[f"{prefix}_raw_sub"] for r in records)
    raw_d = sum(r[f"{prefix}_raw_del"] for r in records)
    raw_i = sum(r[f"{prefix}_raw_ins"] for r in records)
    raw_c = sum(r[f"{prefix}_raw_hits"] for r in records)

    corr_s = sum(r[f"{prefix}_corr_sub"] for r in records)
    corr_d = sum(r[f"{prefix}_corr_del"] for r in records)
    corr_i = sum(r[f"{prefix}_corr_ins"] for r in records)
    corr_c = sum(r[f"{prefix}_corr_hits"] for r in records)

    raw_corpus_wer = (raw_s + raw_d + raw_i) / total_words
    corr_corpus_wer = (corr_s + corr_d + corr_i) / total_words
    delta_corpus_wer = corr_corpus_wer - raw_corpus_wer

    successful = sum(r[f"{prefix}_successful"] for r in records)
    overcorrections = sum(r[f"{prefix}_overcorrections"] for r in records)
    failed = sum(r[f"{prefix}_failed"] for r in records)
    unchanged_correct = sum(r[f"{prefix}_unchanged_correct"] for r in records)
    unchanged_errors = sum(r[f"{prefix}_unchanged_errors"] for r in records)
    total_changes = successful + overcorrections + failed

    precision = (successful / total_changes) if total_changes > 0 else None

    raw_wers = [r[f"{prefix}_raw_wer"] for r in records]
    corr_wers = [r[f"{prefix}_corr_wer"] for r in records]
    delta_wers = [r[f"{prefix}_delta_wer"] for r in records]

    return {
        "count": len(records),
        "total_words": total_words,
        "raw_corpus_wer": float(raw_corpus_wer),
        "corrected_corpus_wer": float(corr_corpus_wer),
        "delta_corpus_wer": float(delta_corpus_wer),
        "raw_substitutions": int(raw_s),
        "raw_deletions": int(raw_d),
        "raw_insertions": int(raw_i),
        "raw_correct_words": int(raw_c),
        "corrected_substitutions": int(corr_s),
        "corrected_deletions": int(corr_d),
        "corrected_insertions": int(corr_i),
        "corrected_correct_words": int(corr_c),
        "successful_corrections": int(successful),
        "overcorrections": int(overcorrections),
        "failed_corrections": int(failed),
        "total_changes": int(total_changes),
        "unchanged_correct": int(unchanged_correct),
        "unchanged_errors": int(unchanged_errors),
        "correction_precision": float(precision) if precision is not None else None,
        "raw_utt_wer_mean": float(np.mean(raw_wers)),
        "raw_utt_wer_std": float(np.std(raw_wers)),
        "corrected_utt_wer_mean": float(np.mean(corr_wers)),
        "corrected_utt_wer_std": float(np.std(corr_wers)),
        "delta_utt_wer_mean": float(np.mean(delta_wers)),
        "delta_utt_wer_std": float(np.std(delta_wers)),
    }


def aggregate_all_correction_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate correction results overall, by speaker, and by input-SNR condition."""
    if not records:
        return {"count": 0, "overall": {}}

    def _calc_both(recs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "noisy": aggregate_condition_stats(recs, "noisy"),
            "enhanced": aggregate_condition_stats(recs, "enhanced"),
        }

    overall = _calc_both(records)

    # By speaker
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        spk = r.get("speaker", "unknown")
        by_speaker.setdefault(spk, []).append(r)

    speaker_summary: dict[str, Any] = {}
    for spk, spk_recs in sorted(by_speaker.items()):
        speaker_summary[spk] = _calc_both(spk_recs)

    # By SNR tier
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
            snr_summary[tier] = _calc_both(by_snr[tier])

    return {
        "count": len(records),
        "overall": overall,
        "by_speaker": speaker_summary,
        "by_snr_bracket": snr_summary,
    }


def evaluate_correction_stage(
    asr_cache_path: str | Path,
    transcripts_dir: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    threshold: float = 0.7,
    phonetic_weight: float = 1.0,
    max_candidates: int = 20,
    resume: bool = True,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run comparative evaluation of the correction pipeline on noisy and enhanced ASR transcripts."""
    asr_cache_path = Path(asr_cache_path)
    if not asr_cache_path.is_file():
        raise FileNotFoundError(f"ASR cache not found at: {asr_cache_path}")

    with open(asr_cache_path, "r", encoding="utf-8") as f:
        asr_cache: dict[str, dict[str, Any]] = json.load(f)
    logger.info("Loaded %d ASR transcription entries from %s", len(asr_cache), asr_cache_path)

    transcripts_dir = Path(transcripts_dir)
    transcripts = load_transcripts(transcripts_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / "correction_cache.json"
    cache: dict[str, dict[str, Any]] = {}
    if resume and cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info("Resumed correction cache with %d entries from %s", len(cache), cache_file)
        except Exception as exc:
            logger.warning("Failed to load existing correction cache (%s); starting fresh.", exc)

    # Initialize correction pipeline
    lm_scorer = CausalLMScorer(device=device, batch_size=32)
    context_cfg = ContextScorerConfig(
        threshold=threshold,
        phonetic_weight=phonetic_weight,
        max_candidates=max_candidates,
    )
    scorer = ContextAwareScorer(lm=lm_scorer, config=context_cfg)
    pipeline = CorrectionPipeline(scorer=scorer)

    stems = sorted(asr_cache.keys())
    if max_samples is not None:
        stems = stems[:max_samples]

    records: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    logger.info("Evaluating context-aware correction on %d utterances...", len(stems))
    for idx, stem in enumerate(stems):
        asr_entry = asr_cache[stem]
        noisy_hyp = asr_entry.get("noisy_text", "")
        enh_hyp = asr_entry.get("enh_text", "")
        snr_val = asr_entry.get("noisy_snr")
        snr_bracket = _assign_snr_bin(snr_val)

        ref_text = transcripts.get(stem)
        if not ref_text:
            raise ValueError(f"Missing reference transcript for utterance '{stem}'")

        utt_name = f"{stem}.wav"
        speaker = _extract_speaker(stem) or "unknown"

        item_cache = cache.get(stem, {})

        # Correct Noisy Hypothesis
        noisy_cache = item_cache.get("noisy")
        if noisy_cache is None:
            noisy_corr_res = pipeline.correct(noisy_hyp)
            noisy_cache = {
                "corrected": noisy_corr_res.corrected,
                "corrections": [wc.to_dict() for wc in noisy_corr_res.word_corrections],
            }
            item_cache["noisy"] = noisy_cache
        else:
            # Reconstruct CorrectionResult from cache
            noisy_corr_res = CorrectionResult(
                original=noisy_hyp,
                corrected=noisy_cache["corrected"],
                word_corrections=[
                    WordCorrection(
                        position=wc["position"],
                        original_token=wc["original_token"],
                        corrected_token=wc["corrected_token"],
                        was_changed=wc["was_changed"],
                        decision=None,
                    )
                    for wc in noisy_cache.get("corrections", [])
                ],
            )

        # Correct Enhanced Hypothesis
        enh_cache = item_cache.get("enhanced")
        if enh_cache is None:
            enh_corr_res = pipeline.correct(enh_hyp)
            enh_cache = {
                "corrected": enh_corr_res.corrected,
                "corrections": [wc.to_dict() for wc in enh_corr_res.word_corrections],
            }
            item_cache["enhanced"] = enh_cache
        else:
            enh_corr_res = CorrectionResult(
                original=enh_hyp,
                corrected=enh_cache["corrected"],
                word_corrections=[
                    WordCorrection(
                        position=wc["position"],
                        original_token=wc["original_token"],
                        corrected_token=wc["corrected_token"],
                        was_changed=wc["was_changed"],
                        decision=None,
                    )
                    for wc in enh_cache.get("corrections", [])
                ],
            )

        cache[stem] = item_cache

        # Evaluate word-level outcomes against ground-truth
        eval_noisy = evaluate_correction(ref_text, noisy_hyp, noisy_corr_res)
        eval_enh = evaluate_correction(ref_text, enh_hyp, enh_corr_res)

        rec: dict[str, Any] = {
            "clean_file": utt_name,
            "speaker": speaker,
            "reference": eval_noisy.reference,
            "noisy_snr": snr_val,
            "snr_bracket": snr_bracket,
            # Noisy condition metrics
            "noisy_raw_hyp": eval_noisy.wer_before.hypothesis,
            "noisy_corr_hyp": eval_noisy.wer_after.hypothesis,
            "noisy_raw_ref_words": eval_noisy.wer_before.reference_words,
            "noisy_raw_wer": eval_noisy.wer_before.wer,
            "noisy_raw_sub": eval_noisy.wer_before.substitutions,
            "noisy_raw_del": eval_noisy.wer_before.deletions,
            "noisy_raw_ins": eval_noisy.wer_before.insertions,
            "noisy_raw_hits": eval_noisy.wer_before.hits,
            "noisy_corr_wer": eval_noisy.wer_after.wer,
            "noisy_corr_sub": eval_noisy.wer_after.substitutions,
            "noisy_corr_del": eval_noisy.wer_after.deletions,
            "noisy_corr_ins": eval_noisy.wer_after.insertions,
            "noisy_corr_hits": eval_noisy.wer_after.hits,
            "noisy_delta_wer": eval_noisy.wer_delta,
            "noisy_successful": eval_noisy.successful_corrections,
            "noisy_overcorrections": eval_noisy.overcorrections,
            "noisy_failed": eval_noisy.failed_corrections,
            "noisy_unchanged_correct": eval_noisy.unchanged_correct,
            "noisy_unchanged_errors": eval_noisy.unchanged_errors,
            "noisy_precision": eval_noisy.correction_precision,
            # Enhanced condition metrics
            "enhanced_raw_hyp": eval_enh.wer_before.hypothesis,
            "enhanced_corr_hyp": eval_enh.wer_after.hypothesis,
            "enhanced_raw_ref_words": eval_enh.wer_before.reference_words,
            "enhanced_raw_wer": eval_enh.wer_before.wer,
            "enhanced_raw_sub": eval_enh.wer_before.substitutions,
            "enhanced_raw_del": eval_enh.wer_before.deletions,
            "enhanced_raw_ins": eval_enh.wer_before.insertions,
            "enhanced_raw_hits": eval_enh.wer_before.hits,
            "enhanced_corr_wer": eval_enh.wer_after.wer,
            "enhanced_corr_sub": eval_enh.wer_after.substitutions,
            "enhanced_corr_del": eval_enh.wer_after.deletions,
            "enhanced_corr_ins": eval_enh.wer_after.insertions,
            "enhanced_corr_hits": eval_enh.wer_after.hits,
            "enhanced_delta_wer": eval_enh.wer_delta,
            "enhanced_successful": eval_enh.successful_corrections,
            "enhanced_overcorrections": eval_enh.overcorrections,
            "enhanced_failed": eval_enh.failed_corrections,
            "enhanced_unchanged_correct": eval_enh.unchanged_correct,
            "enhanced_unchanged_errors": eval_enh.unchanged_errors,
            "enhanced_precision": eval_enh.correction_precision,
        }
        records.append(rec)

        # Periodic checkpoint
        if (idx + 1) % 25 == 0 or (idx + 1) == len(stems):
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
            elapsed = time.perf_counter() - start_time
            rate = (idx + 1) / elapsed
            logger.info(
                "Processed %d/%d utterances (%.1f utt/s)",
                idx + 1,
                len(stems),
                rate,
            )

    total_time = time.perf_counter() - start_time
    logger.info("Completed correction evaluation on %d utterances in %.2f seconds.", len(records), total_time)

    # Save final cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    # Save CSV
    csv_path = output_dir / "results_per_utterance.csv"
    if records:
        fieldnames = list(records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        logger.info("Saved per-utterance results CSV: %s", csv_path)

    # Aggregate summary
    summary = aggregate_all_correction_metrics(records)
    summary["metadata"] = {
        "threshold": threshold,
        "phonetic_weight": phonetic_weight,
        "max_candidates": max_candidates,
        "lm_model": "gpt2",
        "total_utterances": len(records),
        "runtime_seconds": round(total_time, 2),
        "device": device,
    }

    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved correction summary JSON: %s", json_path)

    # Print summary tables
    logger.info("=" * 80)
    logger.info("CONTEXT-AWARE ASR CORRECTION EVALUATION SUMMARY (%d Utterances)", len(records))
    logger.info("=" * 80)

    for cond_name, prefix in [("Noisy Speech \u2192 Whisper \u2192 Correction", "noisy"),
                              ("Enhanced Speech \u2192 Whisper \u2192 Correction", "enhanced")]:
        c_stats = summary["overall"][prefix]
        logger.info("Condition: %s", cond_name)
        logger.info("-" * 80)
        logger.info(
            "  Corpus WER:       %6.2f%% \u2192 %6.2f%% (\u0394 %+.2f%%)",
            c_stats["raw_corpus_wer"] * 100,
            c_stats["corrected_corpus_wer"] * 100,
            c_stats["delta_corpus_wer"] * 100,
        )
        logger.info(
            "  Mean Utt WER:     %6.2f%% \u2192 %6.2f%% (\u0394 %+.2f%%)",
            c_stats["raw_utt_wer_mean"] * 100,
            c_stats["corrected_utt_wer_mean"] * 100,
            c_stats["delta_utt_wer_mean"] * 100,
        )
        logger.info(
            "  Substitutions (S): %5d  \u2192 %5d  (\u0394 %+d)",
            c_stats["raw_substitutions"],
            c_stats["corrected_substitutions"],
            c_stats["corrected_substitutions"] - c_stats["raw_substitutions"],
        )
        logger.info(
            "  Deletions (D):     %5d  \u2192 %5d  (\u0394 %+d)",
            c_stats["raw_deletions"],
            c_stats["corrected_deletions"],
            c_stats["corrected_deletions"] - c_stats["raw_deletions"],
        )
        logger.info(
            "  Insertions (I):    %5d  \u2192 %5d  (\u0394 %+d)",
            c_stats["raw_insertions"],
            c_stats["corrected_insertions"],
            c_stats["corrected_insertions"] - c_stats["raw_insertions"],
        )
        prec_str = f"{c_stats['correction_precision']*100:.1f}%" if c_stats["correction_precision"] is not None else "N/A"
        logger.info(
            "  Decisions:        %d successful, %d overcorrections, %d failed | Precision: %s",
            c_stats["successful_corrections"],
            c_stats["overcorrections"],
            c_stats["failed_corrections"],
            prec_str,
        )
        logger.info(
            "  Preserved Words:  %d unchanged correct, %d unchanged errors",
            c_stats["unchanged_correct"],
            c_stats["unchanged_errors"],
        )
        logger.info("-" * 80)

    # Per-SNR breakdown table
    logger.info("BREAKDOWN BY INPUT SNR TIER (Corpus WER):")
    logger.info("%-28s %-20s %-20s", "SNR Tier (Count)", "Noisy (Raw \u2192 Corr)", "Enhanced (Raw \u2192 Corr)")
    logger.info("-" * 80)
    for tier, t_data in summary["by_snr_bracket"].items():
        n_raw = t_data["noisy"]["raw_corpus_wer"] * 100
        n_cor = t_data["noisy"]["corrected_corpus_wer"] * 100
        n_d = t_data["noisy"]["delta_corpus_wer"] * 100

        e_raw = t_data["enhanced"]["raw_corpus_wer"] * 100
        e_cor = t_data["enhanced"]["corrected_corpus_wer"] * 100
        e_d = t_data["enhanced"]["delta_corpus_wer"] * 100

        n_str = f"{n_raw:.2f}% \u2192 {n_cor:.2f}% ({n_d:+.2f}%)"
        e_str = f"{e_raw:.2f}% \u2192 {e_cor:.2f}% ({e_d:+.2f}%)"
        logger.info("%-28s %-20s %-20s", f"{tier} ({t_data['noisy']['count']})", n_str, e_str)
    logger.info("=" * 80)

    return summary


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate context-aware ASR correction on VoiceBank+DEMAND test set."
    )
    parser.add_argument(
        "--asr-cache",
        type=str,
        default="results/asr_enhancement_eval/asr_cache.json",
        help="Path to cached ASR transcriptions JSON.",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default="data/processed/voicebank_demand_16k/testset_txt",
        help="Path to reference transcripts directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/correction_eval",
        help="Directory to save correction evaluation results, CSV, and summary JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for applying a correction (default: 0.7).",
    )
    parser.add_argument(
        "--phonetic-weight",
        type=float,
        default=1.0,
        help="Weight for phonetic similarity in candidate scoring (default: 1.0).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="Max phonetic candidates per target word (default: 20).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Do not resume from cached correction results.",
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
    evaluate_correction_stage(
        asr_cache_path=parsed.asr_cache,
        transcripts_dir=parsed.transcripts_dir,
        output_dir=parsed.output_dir,
        device=parsed.device,
        threshold=parsed.threshold,
        phonetic_weight=parsed.phonetic_weight,
        max_candidates=parsed.max_candidates,
        resume=parsed.resume,
        max_samples=parsed.max_samples,
    )


if __name__ == "__main__":
    main()
