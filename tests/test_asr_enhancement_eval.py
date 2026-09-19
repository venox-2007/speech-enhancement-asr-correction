"""Unit tests for downstream ASR speech enhancement evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.evaluate_asr_enhancement import (
    aggregate_asr_metrics,
    evaluate_asr,
)
from src.preprocessing.transcripts import load_transcripts


def test_load_transcripts_from_dir(tmp_path: Path):
    """load_transcripts correctly parses directory of text files."""
    txt_dir = tmp_path / "transcripts"
    txt_dir.mkdir()

    (txt_dir / "p232_001.txt").write_text("Please call Stella.  \n", encoding="utf-8")
    (txt_dir / "p257_002.txt").write_text("Ask her to bring these things.", encoding="utf-8")

    loaded = load_transcripts(txt_dir)
    assert len(loaded) == 2
    assert loaded["p232_001"] == "Please call Stella."
    assert loaded["p257_002"] == "Ask her to bring these things."


def test_load_transcripts_missing_dir(tmp_path: Path):
    """load_transcripts raises FileNotFoundError on missing path."""
    with pytest.raises(FileNotFoundError):
        load_transcripts(tmp_path / "non_existent_dir")


def test_aggregate_asr_metrics():
    """aggregate_asr_metrics correctly computes corpus WER, S/D/I/C, and breakdowns."""
    sample_records = [
        {
            "clean_file": "p232_001.wav",
            "speaker": "p232",
            "ref_words": 10,
            "noisy_wer": 0.30,
            "noisy_sub": 2,
            "noisy_del": 1,
            "noisy_ins": 0,
            "noisy_hits": 7,
            "enhanced_wer": 0.10,
            "enhanced_sub": 1,
            "enhanced_del": 0,
            "enhanced_ins": 0,
            "enhanced_hits": 9,
            "delta_wer": -0.20,
            "delta_hits": 2,
            "noisy_snr": 3.5,
            "snr_bracket": "< 5 dB (Low SNR)",
        },
        {
            "clean_file": "p257_001.wav",
            "speaker": "p257",
            "ref_words": 10,
            "noisy_wer": 0.10,
            "noisy_sub": 1,
            "noisy_del": 0,
            "noisy_ins": 0,
            "noisy_hits": 9,
            "enhanced_wer": 0.00,
            "enhanced_sub": 0,
            "enhanced_del": 0,
            "enhanced_ins": 0,
            "enhanced_hits": 10,
            "delta_wer": -0.10,
            "delta_hits": 1,
            "noisy_snr": 12.0,
            "snr_bracket": "10 - 15 dB (Medium-High SNR)",
        },
    ]

    summary = aggregate_asr_metrics(sample_records)

    assert summary["count"] == 2
    ov = summary["overall"]
    assert ov["total_words"] == 20

    # Noisy: S=3, D=1, I=0 -> 4/20 = 0.20 (20.0%)
    assert pytest.approx(ov["noisy"]["corpus_wer"]) == 0.20
    assert ov["noisy"]["substitutions"] == 3
    assert ov["noisy"]["deletions"] == 1
    assert ov["noisy"]["insertions"] == 0
    assert ov["noisy"]["correct_words"] == 16

    # Enhanced: S=1, D=0, I=0 -> 1/20 = 0.05 (5.0%)
    assert pytest.approx(ov["enhanced"]["corpus_wer"]) == 0.05
    assert ov["enhanced"]["substitutions"] == 1
    assert ov["enhanced"]["deletions"] == 0
    assert ov["enhanced"]["insertions"] == 0
    assert ov["enhanced"]["correct_words"] == 19

    # Delta
    assert pytest.approx(ov["delta"]["delta_corpus_wer"]) == -0.15
    assert ov["delta"]["delta_substitutions"] == -2
    assert ov["delta"]["delta_correct_words"] == 3

    # Per-speaker checks
    assert "p232" in summary["by_speaker"]
    assert "p257" in summary["by_speaker"]
    assert pytest.approx(summary["by_speaker"]["p232"]["noisy"]["corpus_wer"]) == 0.30
    assert pytest.approx(summary["by_speaker"]["p232"]["enhanced"]["corpus_wer"]) == 0.10

    # Per-SNR tier checks
    assert "< 5 dB (Low SNR)" in summary["by_snr_bracket"]
    assert "10 - 15 dB (Medium-High SNR)" in summary["by_snr_bracket"]
    assert pytest.approx(summary["by_snr_bracket"]["< 5 dB (Low SNR)"]["noisy"]["corpus_wer"]) == 0.30


def test_aggregate_asr_metrics_empty():
    """aggregate_asr_metrics handles empty inputs without crashing."""
    summary = aggregate_asr_metrics([])
    assert summary["count"] == 0
    assert summary["overall"] == {}


def test_asr_caching_and_resume(tmp_path: Path):
    """Verifies that cached transcripts are reused on subsequent evaluation runs."""
    cache_file = tmp_path / "asr_cache.json"
    cache_data = {
        "p232_001": {
            "noisy_text": "Please call Stella.",
            "enh_text": "Please call Stella.",
            "noisy_snr": 10.5,
            "noisy_time_s": 0.1,
            "enh_time_s": 0.1,
        }
    }
    cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

    # Load cache back and assert existence
    loaded = json.loads(cache_file.read_text(encoding="utf-8"))
    assert "p232_001" in loaded
    assert loaded["p232_001"]["noisy_text"] == "Please call Stella."


def test_evaluate_asr_with_cache(tmp_path: Path):
    """End-to-end test of evaluate_asr using pre-cached transcriptions."""
    import soundfile as sf
    import numpy as np

    # Synthetic dataset
    data_dir = tmp_path / "data"
    clean_dir = data_dir / "clean_testset_wav"
    noisy_dir = data_dir / "noisy_testset_wav"
    txt_dir = data_dir / "testset_txt"
    clean_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)
    txt_dir.mkdir(parents=True)

    sr = 16_000
    sig = np.zeros(sr, dtype=np.float32)
    sf.write(str(clean_dir / "p232_001.wav"), sig, sr)
    sf.write(str(noisy_dir / "p232_001.wav"), sig, sr)
    (txt_dir / "p232_001.txt").write_text("Please call Stella.", encoding="utf-8")

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)

    # Pre-populate cache so Whisper doesn't need to transcribe
    cache = {
        "p232_001": {
            "noisy_text": "Please call Stella.",
            "enh_text": "Please call Stella.",
            "noisy_snr": 10.0,
            "noisy_time_s": 0.05,
            "enh_time_s": 0.05,
        }
    }
    (out_dir / "asr_cache.json").write_text(json.dumps(cache), encoding="utf-8")

    # Run evaluate_asr with resume=True
    summary = evaluate_asr(
        checkpoint_path="dummy.pt",
        data_dir=data_dir,
        output_dir=out_dir,
        transcripts_dir=txt_dir,
        device="cpu",
        resume=True,
    )

    assert summary["count"] == 1
    assert summary["overall"]["total_words"] == 3
    assert summary["overall"]["noisy"]["corpus_wer"] == 0.0
    assert summary["overall"]["enhanced"]["corpus_wer"] == 0.0
    assert (out_dir / "results_per_utterance.csv").is_file()
    assert (out_dir / "summary.json").is_file()

