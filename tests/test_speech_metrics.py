"""Unit tests for objective speech enhancement metrics and evaluation flow."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import numpy as np
import pytest
import soundfile as sf
import torch

from src.enhancement.model import EnhancementModelConfig, WaveformUNetEnhancer
from src.enhancement.unet import UNetConfig
from src.enhancement.viz import plot_waveform_spectrogram_grid
from src.evaluation.speech_metrics import (
    aggregate_metrics,
    calculate_pesq,
    calculate_sisdr,
    calculate_snr,
    calculate_stoi,
    compute_metrics_pair,
)
from src.evaluation.evaluate_enhancement import evaluate


@pytest.fixture
def clean_signal() -> np.ndarray:
    """Deterministic synthetic speech-like signal with amplitude modulation."""
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # Temporal envelope (4 Hz syllable-like rate)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    # Fundamental + harmonics
    carrier = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
    )
    sig = (env * carrier).astype(np.float32)
    return sig


def test_stoi_identical_signals(clean_signal: np.ndarray):
    """STOI of identical signals should be close to 1.0."""
    score = calculate_stoi(clean_signal, clean_signal, sr=16_000)
    assert 0.98 <= score <= 1.01


def test_stoi_noisy_signal(clean_signal: np.ndarray):
    """STOI score should monotonically decrease with increasing noise."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, size=clean_signal.shape).astype(np.float32)

    noisy_mild = clean_signal + 0.1 * noise
    noisy_severe = clean_signal + 0.8 * noise

    score_mild = calculate_stoi(clean_signal, noisy_mild, sr=16_000)
    score_severe = calculate_stoi(clean_signal, noisy_severe, sr=16_000)

    assert score_mild > score_severe
    assert score_severe < 0.95


def test_sisdr_calculation(clean_signal: np.ndarray):
    """SI-SDR should be high for clean, scale-invariant, and lower for noise."""
    # Scale invariance test: 2.5 * clean_signal should have very high SI-SDR
    scaled_clean = 2.5 * clean_signal
    sisdr_clean = calculate_sisdr(clean_signal, scaled_clean)
    assert sisdr_clean > 50.0  # Essentially infinite/capped by epsilon

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1.0, size=clean_signal.shape).astype(np.float32)
    # Power ratio: clean power vs noise power
    noisy_10db = clean_signal + 0.1 * noise
    sisdr_noisy = calculate_sisdr(clean_signal, noisy_10db)
    assert 0.0 < sisdr_noisy < sisdr_clean


def test_snr_calculation(clean_signal: np.ndarray):
    """SNR should correctly reflect relative noise level."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1.0, size=clean_signal.shape).astype(np.float32)

    clean_power = np.mean(clean_signal**2)
    desired_snr_db = 10.0
    desired_noise_power = clean_power / (10 ** (desired_snr_db / 10.0))
    actual_noise = noise * np.sqrt(desired_noise_power / np.mean(noise**2))

    noisy = clean_signal + actual_noise
    snr = calculate_snr(clean_signal, noisy)
    assert pytest.approx(desired_snr_db, abs=0.5) == snr


def test_pesq_graceful_handling(clean_signal: np.ndarray):
    """PESQ should return either a float in valid range or None if not installed."""
    res = calculate_pesq(clean_signal, clean_signal, sr=16_000)
    if res is not None:
        assert isinstance(res, float)
        assert -0.5 <= res <= 5.0
    else:
        # None is expected when C compiled wheel is absent on Windows
        assert res is None


def test_compute_metrics_pair(clean_signal: np.ndarray):
    """compute_metrics_pair returns all fields with exact delta relations."""
    rng = np.random.default_rng(42)
    noisy = clean_signal + 0.2 * rng.normal(0, 1.0, size=clean_signal.shape).astype(np.float32)
    enhanced = clean_signal + 0.05 * rng.normal(0, 1.0, size=clean_signal.shape).astype(np.float32)

    res = compute_metrics_pair(clean_signal, noisy, enhanced, sr=16_000)

    expected_keys = [
        "noisy_stoi",
        "enhanced_stoi",
        "delta_stoi",
        "noisy_sisdr",
        "enhanced_sisdr",
        "delta_sisdr",
        "noisy_snr",
        "enhanced_snr",
        "delta_snr",
        "noisy_pesq",
        "enhanced_pesq",
        "delta_pesq",
    ]
    for k in expected_keys:
        assert k in res

    assert pytest.approx(res["delta_stoi"]) == res["enhanced_stoi"] - res["noisy_stoi"]
    assert pytest.approx(res["delta_sisdr"]) == res["enhanced_sisdr"] - res["noisy_sisdr"]
    assert pytest.approx(res["delta_snr"]) == res["enhanced_snr"] - res["noisy_snr"]
    # Enhanced should outperform noisy here
    assert res["delta_stoi"] > 0
    assert res["delta_sisdr"] > 0
    assert res["delta_snr"] > 0


def test_aggregate_metrics():
    """aggregate_metrics produces correct overall and stratified statistics."""
    sample_records = [
        {
            "clean_file": "p232_001.wav",
            "speaker": "p232",
            "noisy_snr": 3.0,
            "enhanced_snr": 8.0,
            "delta_snr": 5.0,
            "noisy_stoi": 0.70,
            "enhanced_stoi": 0.85,
            "delta_stoi": 0.15,
            "noisy_sisdr": 2.5,
            "enhanced_sisdr": 7.5,
            "delta_sisdr": 5.0,
            "noisy_pesq": None,
            "enhanced_pesq": None,
            "delta_pesq": None,
        },
        {
            "clean_file": "p257_001.wav",
            "speaker": "p257",
            "noisy_snr": 12.0,
            "enhanced_snr": 15.0,
            "delta_snr": 3.0,
            "noisy_stoi": 0.90,
            "enhanced_stoi": 0.95,
            "delta_stoi": 0.05,
            "noisy_sisdr": 11.5,
            "enhanced_sisdr": 14.5,
            "delta_sisdr": 3.0,
            "noisy_pesq": None,
            "enhanced_pesq": None,
            "delta_pesq": None,
        },
    ]

    agg = aggregate_metrics(sample_records)
    assert agg["count"] == 2
    assert pytest.approx(agg["overall"]["delta_snr"]["mean"]) == 4.0
    assert pytest.approx(agg["overall"]["delta_stoi"]["mean"]) == 0.10

    # Per-speaker check
    assert "p232" in agg["by_speaker"]
    assert "p257" in agg["by_speaker"]
    assert agg["by_speaker"]["p232"]["count"] == 1
    assert pytest.approx(agg["by_speaker"]["p232"]["metrics"]["delta_snr"]["mean"]) == 5.0

    # Per-SNR bracket check
    assert "< 5 dB (Low SNR)" in agg["by_snr_bracket"]
    assert "10 - 15 dB (Medium-High SNR)" in agg["by_snr_bracket"]
    assert agg["by_snr_bracket"]["< 5 dB (Low SNR)"]["count"] == 1


def test_aggregate_metrics_empty():
    """aggregate_metrics handles empty records list without crashing."""
    agg = aggregate_metrics([])
    assert agg["count"] == 0
    assert agg["overall"] == {}


def test_plot_waveform_spectrogram_grid(clean_signal: np.ndarray):
    """Verify plot_waveform_spectrogram_grid generates 2x3 comparison figure."""
    t_clean = torch.from_numpy(clean_signal)
    t_noisy = t_clean + 0.1 * torch.randn_like(t_clean)
    t_enh = t_clean + 0.02 * torch.randn_like(t_clean)

    fig = plot_waveform_spectrogram_grid(
        clean=t_clean,
        noisy=t_noisy,
        enhanced=t_enh,
        sample_rate=16_000,
        title="Test Comparison",
        metrics={"noisy_stoi": 0.85, "enhanced_stoi": 0.95, "delta_stoi": 0.10},
    )
    import matplotlib.pyplot as plt

    assert fig is not None
    # 2 rows, 3 columns = 6 axes
    assert len(fig.axes) >= 6
    plt.close(fig)


def test_evaluate_smoke_test(tmp_path: Path):
    """End-to-end test of evaluate() on synthetic audio pairs and model checkpoint."""
    # Create synthetic test dataset directory layout
    data_dir = tmp_path / "data"
    clean_dir = data_dir / "clean_testset_wav"
    noisy_dir = data_dir / "noisy_testset_wav"
    clean_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)

    sr = 16_000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False, dtype=np.float32)
    clean_sig = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noisy_sig = clean_sig + 0.1 * np.random.randn(*clean_sig.shape).astype(np.float32)

    # Save 2 sample pairs for speaker p232 and p257
    sf.write(str(clean_dir / "p232_001.wav"), clean_sig, sr)
    sf.write(str(noisy_dir / "p232_001.wav"), noisy_sig, sr)
    sf.write(str(clean_dir / "p257_001.wav"), clean_sig, sr)
    sf.write(str(noisy_dir / "p257_001.wav"), noisy_sig, sr)

    # Create and save a lightweight model checkpoint
    unet_cfg = UNetConfig(in_channels=1, out_channels=1, channels=(4, 8))
    model_cfg = EnhancementModelConfig(unet=unet_cfg)
    enhancer = WaveformUNetEnhancer(model_cfg)

    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": enhancer.state_dict(),
            "config": {
                "unet": {"channels": (4, 8)},
                "sample_rate": 16000,
            },
            "epoch": 1,
            "best_val_loss": 0.05,
        },
        ckpt_path,
    )

    out_dir = tmp_path / "eval_out"
    summary = evaluate(
        checkpoint_path=ckpt_path,
        data_dir=data_dir,
        output_dir=out_dir,
        device="cpu",
        save_audio=True,
        num_plots=2,
    )

    assert summary["count"] == 2
    assert (out_dir / "metrics_per_utterance.csv").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "audio" / "p232_001.wav").is_file()
    assert (out_dir / "audio" / "p257_001.wav").is_file()

    # Verify enhanced audio has exact length
    enh_audio, enh_sr = sf.read(str(out_dir / "audio" / "p232_001.wav"))
    assert enh_sr == sr
    assert len(enh_audio) == len(clean_sig)
