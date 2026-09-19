"""Unit tests for the context-aware ASR correction evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.evaluation.evaluate_correction_pipeline import (
    aggregate_all_correction_metrics,
    aggregate_condition_stats,
    evaluate_correction_stage,
)


def test_aggregate_condition_stats():
    """Verify exact calculation of corpus-level correction stats, precision, and deltas."""
    sample_records = [
        {
            "noisy_raw_ref_words": 10,
            "noisy_raw_sub": 2,
            "noisy_raw_del": 0,
            "noisy_raw_ins": 0,
            "noisy_raw_hits": 8,
            "noisy_corr_sub": 1,
            "noisy_corr_del": 0,
            "noisy_corr_ins": 0,
            "noisy_corr_hits": 9,
            "noisy_raw_wer": 0.20,
            "noisy_corr_wer": 0.10,
            "noisy_delta_wer": -0.10,
            "noisy_successful": 1,
            "noisy_overcorrections": 0,
            "noisy_failed": 0,
            "noisy_unchanged_correct": 8,
            "noisy_unchanged_errors": 1,
        },
        {
            "noisy_raw_ref_words": 10,
            "noisy_raw_sub": 0,
            "noisy_raw_del": 0,
            "noisy_raw_ins": 0,
            "noisy_raw_hits": 10,
            "noisy_corr_sub": 1,
            "noisy_corr_del": 0,
            "noisy_corr_ins": 0,
            "noisy_corr_hits": 9,
            "noisy_raw_wer": 0.00,
            "noisy_corr_wer": 0.10,
            "noisy_delta_wer": 0.10,
            "noisy_successful": 0,
            "noisy_overcorrections": 1,
            "noisy_failed": 0,
            "noisy_unchanged_correct": 9,
            "noisy_unchanged_errors": 0,
        },
    ]

    stats = aggregate_condition_stats(sample_records, "noisy")

    assert stats["count"] == 2
    assert stats["total_words"] == 20
    # Raw WER: 2 errors / 20 words = 0.10 (10.0%)
    assert pytest.approx(stats["raw_corpus_wer"]) == 0.10
    # Corrected WER: 2 errors / 20 words = 0.10 (10.0%)
    assert pytest.approx(stats["corrected_corpus_wer"]) == 0.10
    assert pytest.approx(stats["delta_corpus_wer"]) == 0.0

    assert stats["successful_corrections"] == 1
    assert stats["overcorrections"] == 1
    assert stats["failed_corrections"] == 0
    assert stats["total_changes"] == 2
    # Precision: 1 / 2 = 0.50 (50.0%)
    assert pytest.approx(stats["correction_precision"]) == 0.50

    assert stats["unchanged_correct"] == 17
    assert stats["unchanged_errors"] == 1


def test_aggregate_condition_stats_empty():
    """Verify empty input handling."""
    stats = aggregate_condition_stats([], "noisy")
    assert stats == {}


def test_aggregate_all_correction_metrics():
    """Verify hierarchical aggregation by speaker and SNR bracket."""
    sample_records = [
        {
            "speaker": "p232",
            "snr_bracket": "< 5 dB (Low SNR)",
            "noisy_raw_ref_words": 10,
            "noisy_raw_sub": 1,
            "noisy_raw_del": 0,
            "noisy_raw_ins": 0,
            "noisy_raw_hits": 9,
            "noisy_corr_sub": 0,
            "noisy_corr_del": 0,
            "noisy_corr_ins": 0,
            "noisy_corr_hits": 10,
            "noisy_raw_wer": 0.10,
            "noisy_corr_wer": 0.00,
            "noisy_delta_wer": -0.10,
            "noisy_successful": 1,
            "noisy_overcorrections": 0,
            "noisy_failed": 0,
            "noisy_unchanged_correct": 9,
            "noisy_unchanged_errors": 0,
            "enhanced_raw_ref_words": 10,
            "enhanced_raw_sub": 0,
            "enhanced_raw_del": 0,
            "enhanced_raw_ins": 0,
            "enhanced_raw_hits": 10,
            "enhanced_corr_sub": 0,
            "enhanced_corr_del": 0,
            "enhanced_corr_ins": 0,
            "enhanced_corr_hits": 10,
            "enhanced_raw_wer": 0.00,
            "enhanced_corr_wer": 0.00,
            "enhanced_delta_wer": 0.00,
            "enhanced_successful": 0,
            "enhanced_overcorrections": 0,
            "enhanced_failed": 0,
            "enhanced_unchanged_correct": 10,
            "enhanced_unchanged_errors": 0,
        }
    ]

    summary = aggregate_all_correction_metrics(sample_records)
    assert summary["count"] == 1
    assert "p232" in summary["by_speaker"]
    assert "< 5 dB (Low SNR)" in summary["by_snr_bracket"]
    assert summary["overall"]["noisy"]["successful_corrections"] == 1
    assert summary["overall"]["enhanced"]["successful_corrections"] == 0


def test_evaluate_correction_with_cache(tmp_path: Path):
    """End-to-end test verifying that cached correction results are loaded and scored."""
    # 1. Mock ASR cache
    asr_cache_file = tmp_path / "asr_cache.json"
    asr_cache_data = {
        "p232_001": {
            "noisy_text": "Please call Stella.",
            "enh_text": "Please call Stella.",
            "noisy_snr": 12.0,
        }
    }
    asr_cache_file.write_text(json.dumps(asr_cache_data), encoding="utf-8")

    # 2. Reference transcripts
    txt_dir = tmp_path / "transcripts"
    txt_dir.mkdir()
    (txt_dir / "p232_001.txt").write_text("Please call Stella.", encoding="utf-8")

    # 3. Pre-populated correction cache
    out_dir = tmp_path / "correction_out"
    out_dir.mkdir()
    correction_cache_file = out_dir / "correction_cache.json"
    correction_cache_data = {
        "p232_001": {
            "noisy": {
                "corrected": "Please call Stella.",
                "corrections": [
                    {
                        "position": 0,
                        "original_token": "Please",
                        "corrected_token": "Please",
                        "was_changed": False,
                    },
                    {
                        "position": 1,
                        "original_token": "call",
                        "corrected_token": "call",
                        "was_changed": False,
                    },
                    {
                        "position": 2,
                        "original_token": "Stella.",
                        "corrected_token": "Stella.",
                        "was_changed": False,
                    },
                ],
            },
            "enhanced": {
                "corrected": "Please call Stella.",
                "corrections": [
                    {
                        "position": 0,
                        "original_token": "Please",
                        "corrected_token": "Please",
                        "was_changed": False,
                    },
                    {
                        "position": 1,
                        "original_token": "call",
                        "corrected_token": "call",
                        "was_changed": False,
                    },
                    {
                        "position": 2,
                        "original_token": "Stella.",
                        "corrected_token": "Stella.",
                        "was_changed": False,
                    },
                ],
            },
        }
    }
    correction_cache_file.write_text(json.dumps(correction_cache_data), encoding="utf-8")

    summary = evaluate_correction_stage(
        asr_cache_path=asr_cache_file,
        transcripts_dir=txt_dir,
        output_dir=out_dir,
        device="cpu",
        resume=True,
    )

    assert summary["count"] == 1
    assert summary["overall"]["noisy"]["raw_corpus_wer"] == 0.0
    assert summary["overall"]["noisy"]["corrected_corpus_wer"] == 0.0
    assert summary["overall"]["noisy"]["unchanged_correct"] == 3
    assert summary["overall"]["noisy"]["successful_corrections"] == 0
    assert (out_dir / "results_per_utterance.csv").is_file()
    assert (out_dir / "summary.json").is_file()
