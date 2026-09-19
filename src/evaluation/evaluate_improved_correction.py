"""Comparative evaluation of Raw Whisper vs Baseline Correction vs Improved Correction.

Evaluates the improved post-ASR correction pipeline with principled safeguards
(min_score_margin=3.5, min_phonetic_similarity=0.80, retention_bias=1.0) on the
frozen 824 held-out test utterances of VoiceBank+DEMAND.
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

from src.correction.candidates import PhoneticCandidateGenerator
from src.correction.context_scorer import ContextAwareScorer, ContextScorerConfig
from src.correction.lm_scorer import CausalLMScorer
from src.correction.pipeline import CorrectionPipeline, CorrectionResult, WordCorrection
from src.enhancement.dataset import _extract_speaker
from src.evaluation.correction_eval import evaluate_correction
from src.evaluation.speech_metrics import _assign_snr_bin
from src.evaluation.wer import compute_wer
from src.preprocessing.transcripts import load_transcripts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def aggregate_pipeline_stats(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    """Compute corpus-level correction metrics for a single condition prefix."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate improved post-ASR correction pipeline.")
    parser.add_argument(
        "--asr-cache",
        type=Path,
        default=Path("results/asr_enhancement_eval/asr_cache.json"),
        help="Path to cached Whisper ASR outputs",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("data/processed/voicebank_demand_16k/testset_txt"),
        help="Path to ground-truth test transcripts directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/improved_correction_eval"),
        help="Directory to save evaluation results and cache",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for GPT-2 evaluation (cuda or cpu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Probability threshold",
    )
    parser.add_argument(
        "--min-score-margin",
        type=float,
        default=3.5,
        help="Minimum score margin required before replacing",
    )
    parser.add_argument(
        "--min-phonetic-similarity",
        type=float,
        default=0.80,
        help="Minimum phonetic similarity required for candidates",
    )
    parser.add_argument(
        "--retention-bias",
        type=float,
        default=1.0,
        help="Penalty applied to non-original candidates",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "improved_correction_cache.json"

    if not args.asr_cache.exists():
        logger.error("Whisper cache not found at %s. Aborting.", args.asr_cache)
        sys.exit(1)

    with open(args.asr_cache, "r", encoding="utf-8") as f:
        asr_cache = json.load(f)
    logger.info("Loaded %d cached Whisper ASR transcriptions from %s", len(asr_cache), args.asr_cache)

    transcripts = load_transcripts(args.transcripts_dir)
    logger.info("Loaded %d ground-truth transcripts from %s", len(transcripts), args.transcripts_dir)

    # Load existing improved cache if present
    improved_cache: dict[str, Any] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                improved_cache = json.load(f)
            logger.info("Loaded %d existing improved correction cache entries from %s", len(improved_cache), cache_path)
        except Exception as e:
            logger.warning("Could not read improved correction cache: %s. Starting fresh.", e)
            improved_cache = {}

    config = ContextScorerConfig(
        threshold=args.threshold,
        phonetic_weight=1.0,
        max_candidates=20,
        min_score_margin=args.min_score_margin,
        min_phonetic_similarity=args.min_phonetic_similarity,
        retention_bias=args.retention_bias,
    )
    logger.info("Initializing Improved Pipeline with config: %s", config)
    generator = PhoneticCandidateGenerator(max_candidates=20)
    lm = CausalLMScorer(device=args.device, batch_size=32)
    scorer = ContextAwareScorer(generator=generator, lm=lm, config=config)
    pipeline = CorrectionPipeline(scorer=scorer)

    sorted_keys = sorted(asr_cache.keys())
    start_time = time.time()
    num_processed = 0

    for idx, key in enumerate(sorted_keys, start=1):
        if key in improved_cache and "noisy" in improved_cache[key] and "enhanced" in improved_cache[key]:
            continue

        item = asr_cache[key]
        noisy_raw = item.get("noisy_text", "")
        enh_raw = item.get("enh_text", "")

        noisy_corr_res = pipeline.correct(noisy_raw)
        enh_corr_res = pipeline.correct(enh_raw)

        improved_cache[key] = {
            "noisy": {
                "raw": noisy_raw,
                "corrected": noisy_corr_res.corrected,
                "corrections": [wc.to_dict() for wc in noisy_corr_res.word_corrections],
            },
            "enhanced": {
                "raw": enh_raw,
                "corrected": enh_corr_res.corrected,
                "corrections": [wc.to_dict() for wc in enh_corr_res.word_corrections],
            },
        }
        num_processed += 1
        if idx % 50 == 0 or idx == len(sorted_keys):
            logger.info("Processed %d/%d utterances (%.1fs elapsed)", idx, len(sorted_keys), time.time() - start_time)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(improved_cache, f)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(improved_cache, f)
    elapsed = time.time() - start_time
    logger.info("Inference complete in %.2f seconds (processed %d new utterances)", elapsed, num_processed)

    # Compile comparative per-utterance records
    records: list[dict[str, Any]] = []
    for key in sorted_keys:
        asr_item = asr_cache[key]
        corr_item = improved_cache[key]
        speaker = _extract_speaker(key)

        ref = transcripts.get(key, "")
        noisy_snr = asr_item.get("noisy_snr", 0.0)
        snr_bin = _assign_snr_bin(noisy_snr)

        noisy_raw_hyp = asr_item.get("noisy_text", "")
        noisy_corr_hyp = corr_item["noisy"]["corrected"]
        enh_raw_hyp = asr_item.get("enh_text", "")
        enh_corr_hyp = corr_item["enhanced"]["corrected"]

        noisy_raw_wer = compute_wer(ref, noisy_raw_hyp)
        noisy_corr_wer = compute_wer(ref, noisy_corr_hyp)
        enh_raw_wer = compute_wer(ref, enh_raw_hyp)
        enh_corr_wer = compute_wer(ref, enh_corr_hyp)

        # Reconstruct CorrectionResult for evaluate_correction
        noisy_word_corrs = [
            WordCorrection(
                position=wc["position"],
                original_token=wc["original_token"],
                corrected_token=wc["corrected_token"],
                was_changed=wc["was_changed"],
                decision=None,
            )
            for wc in corr_item["noisy"]["corrections"]
        ]
        noisy_corr_res = CorrectionResult(
            original=noisy_raw_hyp,
            corrected=noisy_corr_hyp,
            word_corrections=noisy_word_corrs,
        )
        noisy_eval = evaluate_correction(ref, noisy_raw_hyp, noisy_corr_res)

        enh_word_corrs = [
            WordCorrection(
                position=wc["position"],
                original_token=wc["original_token"],
                corrected_token=wc["corrected_token"],
                was_changed=wc["was_changed"],
                decision=None,
            )
            for wc in corr_item["enhanced"]["corrections"]
        ]
        enh_corr_res = CorrectionResult(
            original=enh_raw_hyp,
            corrected=enh_corr_hyp,
            word_corrections=enh_word_corrs,
        )
        enh_eval = evaluate_correction(ref, enh_raw_hyp, enh_corr_res)

        records.append({
            "clean_file": f"{key}.wav",
            "speaker": speaker,
            "reference": ref,
            "noisy_snr": noisy_snr,
            "snr_bracket": snr_bin,
            # Noisy
            "noisy_raw_hyp": noisy_raw_hyp,
            "noisy_corr_hyp": noisy_corr_hyp,
            "noisy_raw_ref_words": noisy_raw_wer.reference_words,
            "noisy_raw_wer": noisy_raw_wer.wer,
            "noisy_raw_sub": noisy_raw_wer.substitutions,
            "noisy_raw_del": noisy_raw_wer.deletions,
            "noisy_raw_ins": noisy_raw_wer.insertions,
            "noisy_raw_hits": noisy_raw_wer.hits,
            "noisy_corr_wer": noisy_corr_wer.wer,
            "noisy_corr_sub": noisy_corr_wer.substitutions,
            "noisy_corr_del": noisy_corr_wer.deletions,
            "noisy_corr_ins": noisy_corr_wer.insertions,
            "noisy_corr_hits": noisy_corr_wer.hits,
            "noisy_delta_wer": noisy_corr_wer.wer - noisy_raw_wer.wer,
            "noisy_successful": noisy_eval.successful_corrections,
            "noisy_overcorrections": noisy_eval.overcorrections,
            "noisy_failed": noisy_eval.failed_corrections,
            "noisy_unchanged_correct": noisy_eval.unchanged_correct,
            "noisy_unchanged_errors": noisy_eval.unchanged_errors,
            "noisy_precision": (noisy_eval.successful_corrections / (noisy_eval.successful_corrections + noisy_eval.overcorrections + noisy_eval.failed_corrections)) if (noisy_eval.successful_corrections + noisy_eval.overcorrections + noisy_eval.failed_corrections) > 0 else None,
            # Enhanced
            "enhanced_raw_hyp": enh_raw_hyp,
            "enhanced_corr_hyp": enh_corr_hyp,
            "enhanced_raw_ref_words": enh_raw_wer.reference_words,
            "enhanced_raw_wer": enh_raw_wer.wer,
            "enhanced_raw_sub": enh_raw_wer.substitutions,
            "enhanced_raw_del": enh_raw_wer.deletions,
            "enhanced_raw_ins": enh_raw_wer.insertions,
            "enhanced_raw_hits": enh_raw_wer.hits,
            "enhanced_corr_wer": enh_corr_wer.wer,
            "enhanced_corr_sub": enh_corr_wer.substitutions,
            "enhanced_corr_del": enh_corr_wer.deletions,
            "enhanced_corr_ins": enh_corr_wer.insertions,
            "enhanced_corr_hits": enh_corr_wer.hits,
            "enhanced_delta_wer": enh_corr_wer.wer - enh_raw_wer.wer,
            "enhanced_successful": enh_eval.successful_corrections,
            "enhanced_overcorrections": enh_eval.overcorrections,
            "enhanced_failed": enh_eval.failed_corrections,
            "enhanced_unchanged_correct": enh_eval.unchanged_correct,
            "enhanced_unchanged_errors": enh_eval.unchanged_errors,
            "enhanced_precision": (enh_eval.successful_corrections / (enh_eval.successful_corrections + enh_eval.overcorrections + enh_eval.failed_corrections)) if (enh_eval.successful_corrections + enh_eval.overcorrections + enh_eval.failed_corrections) > 0 else None,
        })

    # Write CSV
    csv_path = args.out_dir / "results_per_utterance.csv"
    if records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        logger.info("Saved per-utterance results to %s", csv_path)

    # Compute overall and segmented summaries
    summary = {
        "count": len(records),
        "overall": {
            "noisy": aggregate_pipeline_stats(records, "noisy"),
            "enhanced": aggregate_pipeline_stats(records, "enhanced"),
        },
        "by_speaker": {},
        "by_snr_bracket": {},
        "metadata": {
            "threshold": args.threshold,
            "min_score_margin": args.min_score_margin,
            "min_phonetic_similarity": args.min_phonetic_similarity,
            "retention_bias": args.retention_bias,
            "phonetic_weight": 1.0,
            "max_candidates": 20,
            "lm_model": "gpt2",
            "total_utterances": len(records),
            "runtime_seconds": round(elapsed, 2),
            "device": args.device,
        },
    }

    speakers = sorted(list({r["speaker"] for r in records}))
    for spk in speakers:
        spk_recs = [r for r in records if r["speaker"] == spk]
        summary["by_speaker"][spk] = {
            "noisy": aggregate_pipeline_stats(spk_recs, "noisy"),
            "enhanced": aggregate_pipeline_stats(spk_recs, "enhanced"),
        }

    snr_brackets = ["< 5 dB (Low SNR)", "5 - 10 dB (Medium-Low SNR)", "10 - 15 dB (Medium-High SNR)", ">= 15 dB (High SNR)"]
    for bracket in snr_brackets:
        bracket_recs = [r for r in records if r["snr_bracket"] == bracket]
        summary["by_snr_bracket"][bracket] = {
            "noisy": aggregate_pipeline_stats(bracket_recs, "noisy"),
            "enhanced": aggregate_pipeline_stats(bracket_recs, "enhanced"),
        }

    summary_path = args.out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    print("\n" + "=" * 80)
    print("IMPROVED ASR CORRECTION EVALUATION RESULTS (824 Held-Out Test Utterances)")
    print("=" * 80)
    for cond in ["noisy", "enhanced"]:
        st = summary["overall"][cond]
        print(f"\nCondition: {cond.upper()}")
        print(f"  Raw WER:       {st['raw_corpus_wer']*100:.2f}% (S={st['raw_substitutions']}, D={st['raw_deletions']}, I={st['raw_insertions']}, C={st['raw_correct_words']})")
        print(f"  Corrected WER: {st['corrected_corpus_wer']*100:.2f}% (S={st['corrected_substitutions']}, D={st['corrected_deletions']}, I={st['corrected_insertions']}, C={st['corrected_correct_words']})")
        print(f"  Delta WER:     {st['delta_corpus_wer']*100:+.2f}%")
        print(f"  Decisions:     Successful={st['successful_corrections']}, Overcorrections={st['overcorrections']}, Failed={st['failed_corrections']}")
        print(f"  Precision:     {st['correction_precision']*100 if st['correction_precision'] is not None else 0.0:.2f}% (Total Changes={st['total_changes']})")
        print(f"  Preserved:     Unchanged Correct={st['unchanged_correct']}, Unchanged Errors={st['unchanged_errors']}")


if __name__ == "__main__":
    main()
