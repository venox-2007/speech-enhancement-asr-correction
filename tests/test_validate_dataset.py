"""Tests for src.preprocessing.validate_dataset.

Uses synthetic 48 kHz and 16 kHz WAV files in temporary directories;
does NOT require the real VoiceBank+DEMAND dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.preprocessing.validate_dataset import (
    EXPECTED_TEST_SPEAKER_IDS,
    EXPECTED_TRAIN_SPEAKERS,
    EXPECTED_TEST_SPEAKERS,
    PROCESSED_EXPECTED_SR,
    RAW_EXPECTED_SR,
    SUBDIRS,
    _extract_speaker,
    _check_sample_rate,
    _check_clipping,
    _wav_files,
    validate_raw,
    validate_processed,
    main,
)


# ─────────────────────────── helpers ──────────────────────────────────────────

def _make_wav(path: Path, sr: int = 48000, duration: float = 0.5,
              peak: float = 0.5) -> None:
    """Write a synthetic WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * duration)
    audio = np.sin(2 * np.pi * 440 * np.arange(n) / sr).astype(np.float32) * peak
    sf.write(str(path), audio, sr, subtype="PCM_16")


def _make_clipped_wav(path: Path, sr: int = 16000) -> None:
    """Write a WAV file with peak ≥ 1.0 (clipping).

    Uses FLOAT subtype so soundfile preserves the exact 1.0 peak;
    PCM_16 would quantize 1.0 → ~0.9999... which falls below the threshold.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * 0.5)
    audio = np.ones(n, dtype=np.float32)  # peak exactly 1.0
    sf.write(str(path), audio, sr, subtype="FLOAT")


def _populate_split(
    base: Path,
    speaker_ids: list[str],
    n_per_speaker: int = 5,
    sr: int = 48000,
) -> None:
    """Create synthetic clean/noisy train or test files for given speakers."""
    clean_dir = base / SUBDIRS["clean_train"]
    noisy_dir = base / SUBDIRS["noisy_train"]
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    for spk in speaker_ids:
        for i in range(n_per_speaker):
            fname = f"{spk}_001_{i:04d}.wav"
            _make_wav(clean_dir / fname, sr=sr)
            _make_wav(noisy_dir / fname, sr=sr)


def _populate_test_split(
    base: Path,
    speaker_ids: list[str] | None = None,
    n_per_speaker: int = 3,
    sr: int = 48000,
) -> None:
    if speaker_ids is None:
        speaker_ids = sorted(EXPECTED_TEST_SPEAKER_IDS)
    clean_dir = base / SUBDIRS["clean_test"]
    noisy_dir = base / SUBDIRS["noisy_test"]
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    for spk in speaker_ids:
        for i in range(n_per_speaker):
            fname = f"{spk}_ch01_{i:04d}.wav"
            _make_wav(clean_dir / fname, sr=sr)
            _make_wav(noisy_dir / fname, sr=sr)


# ─────────────────────────── unit tests ───────────────────────────────────────


class TestHelpers:
    def test_extract_speaker_standard(self):
        assert _extract_speaker("p232_001_0001.wav") == "p232"

    def test_extract_speaker_none(self):
        assert _extract_speaker("unknown_file.wav") is None

    def test_check_sample_rate_correct(self, tmp_path):
        p = tmp_path / "a.wav"
        _make_wav(p, sr=48000)
        ok, sr = _check_sample_rate(p, 48000)
        assert ok
        assert sr == 48000

    def test_check_sample_rate_wrong(self, tmp_path):
        p = tmp_path / "a.wav"
        _make_wav(p, sr=16000)
        ok, sr = _check_sample_rate(p, 48000)
        assert not ok
        assert sr == 16000

    def test_check_clipping_no_clip(self, tmp_path):
        p = tmp_path / "a.wav"
        _make_wav(p, sr=16000, peak=0.5)
        is_clipped, peak = _check_clipping(p)
        assert not is_clipped
        assert peak < 1.0

    def test_check_clipping_at_unity(self, tmp_path):
        p = tmp_path / "a.wav"
        _make_clipped_wav(p, sr=16000)
        is_clipped, peak = _check_clipping(p)
        assert is_clipped
        assert peak >= 1.0

    def test_wav_files_sorted(self, tmp_path):
        for name in ["c.wav", "a.wav", "b.wav"]:
            _make_wav(tmp_path / name)
        result = _wav_files(tmp_path)
        names = [f.name for f in result]
        assert names == sorted(names)


# ─────────────────────────── validate_raw ─────────────────────────────────────


class TestValidateRaw:
    def test_missing_all_subdirs(self, tmp_path):
        """Empty directory should fail validation."""
        assert not validate_raw(tmp_path)

    def test_valid_minimal_raw(self, tmp_path):
        """Synthetic files at 48 kHz should pass validation."""
        speakers = [f"p{i:03d}" for i in range(200, 228)]  # 28 speakers
        _populate_split(tmp_path, speakers, n_per_speaker=2, sr=48000)
        _populate_test_split(tmp_path, sr=48000)
        # validate_raw checks speakers, pairing, sample rate
        result = validate_raw(tmp_path)
        assert result

    def test_mismatched_pairs(self, tmp_path):
        """Extra clean file with no noisy pair should fail."""
        clean_dir = tmp_path / SUBDIRS["clean_train"]
        noisy_dir = tmp_path / SUBDIRS["noisy_train"]
        clean_dir.mkdir(parents=True)
        noisy_dir.mkdir(parents=True)
        _make_wav(clean_dir / "p200_001_0000.wav", sr=48000)
        _make_wav(clean_dir / "p200_001_0001.wav", sr=48000)  # extra
        _make_wav(noisy_dir / "p200_001_0000.wav", sr=48000)
        # Add test dirs too so those checks pass
        _populate_test_split(tmp_path, sr=48000)
        assert not validate_raw(tmp_path)

    def test_wrong_sample_rate(self, tmp_path):
        """Files at 16 kHz in a 48 kHz raw dir should fail."""
        speakers = [f"p{i:03d}" for i in range(200, 228)]
        _populate_split(tmp_path, speakers, n_per_speaker=2, sr=16000)  # wrong SR
        _populate_test_split(tmp_path, sr=16000)
        assert not validate_raw(tmp_path)

    def test_wrong_test_speakers(self, tmp_path):
        """Non-standard test speaker IDs should trigger a warning and fail."""
        speakers = [f"p{i:03d}" for i in range(200, 228)]
        _populate_split(tmp_path, speakers, n_per_speaker=2, sr=48000)
        _populate_test_split(tmp_path, speaker_ids=["p100", "p101"], sr=48000)
        assert not validate_raw(tmp_path)

    def test_correct_test_speaker_ids(self, tmp_path):
        """Expected test speakers p232 and p257 should pass."""
        speakers = [f"p{i:03d}" for i in range(200, 228)]
        _populate_split(tmp_path, speakers, n_per_speaker=2, sr=48000)
        _populate_test_split(tmp_path, speaker_ids=["p232", "p257"], sr=48000)
        assert validate_raw(tmp_path)


# ─────────────────────────── validate_processed ───────────────────────────────


class TestValidateProcessed:
    def _make_processed(self, tmp_path: Path, sr: int = 16000) -> None:
        speakers = [f"p{i:03d}" for i in range(200, 228)]
        _populate_split(tmp_path, speakers, n_per_speaker=2, sr=sr)
        _populate_test_split(tmp_path, speaker_ids=["p232", "p257"], n_per_speaker=2, sr=sr)

    def test_valid_processed(self, tmp_path):
        self._make_processed(tmp_path, sr=16000)
        assert validate_processed(tmp_path, check_clipping=True)

    def test_wrong_sample_rate_processed(self, tmp_path):
        """48 kHz files in processed dir should fail."""
        self._make_processed(tmp_path, sr=48000)
        assert not validate_processed(tmp_path, check_clipping=False)

    def test_missing_processed_subdir(self, tmp_path):
        """Missing subdirectory should fail."""
        assert not validate_processed(tmp_path, check_clipping=False)

    def test_no_clipping_in_synthetic(self, tmp_path):
        """Synthetic 0.5-peak files should not be flagged as clipped."""
        self._make_processed(tmp_path, sr=16000)
        # Should pass without clipping warnings
        assert validate_processed(tmp_path, check_clipping=True)

    def test_pairing_mismatch_processed(self, tmp_path):
        """Unequal clean/noisy files in processed set should fail."""
        clean_dir = tmp_path / SUBDIRS["clean_train"]
        noisy_dir = tmp_path / SUBDIRS["noisy_train"]
        clean_dir.mkdir(parents=True)
        noisy_dir.mkdir(parents=True)
        _make_wav(clean_dir / "p200_001_0000.wav", sr=16000)
        _make_wav(clean_dir / "p200_001_0001.wav", sr=16000)  # extra clean
        _make_wav(noisy_dir / "p200_001_0000.wav", sr=16000)
        _populate_test_split(tmp_path, speaker_ids=["p232", "p257"], sr=16000)
        assert not validate_processed(tmp_path, check_clipping=False)


# ─────────────────────────── CLI (main) ───────────────────────────────────────


class TestValidateDatasetCLI:
    def test_main_missing_dirs(self, tmp_path):
        """Non-existent directories should give exit code 1."""
        result = main([
            "--raw-dir", str(tmp_path / "no_raw"),
            "--processed-dir", str(tmp_path / "no_proc"),
        ])
        assert result == 1

    def test_main_valid_raw_only(self, tmp_path):
        raw_dir = tmp_path / "raw"
        speakers = [f"p{i:03d}" for i in range(200, 228)]
        _populate_split(raw_dir, speakers, n_per_speaker=2, sr=48000)
        _populate_test_split(raw_dir, speaker_ids=["p232", "p257"], sr=48000)
        result = main([
            "--raw-dir", str(raw_dir),
            "--processed-dir", str(tmp_path / "proc_missing"),
        ])
        assert result == 0

    def test_main_both_valid(self, tmp_path):
        raw_dir = tmp_path / "raw"
        proc_dir = tmp_path / "proc"
        speakers = [f"p{i:03d}" for i in range(200, 228)]
        _populate_split(raw_dir, speakers, n_per_speaker=2, sr=48000)
        _populate_test_split(raw_dir, speaker_ids=["p232", "p257"], sr=48000)
        _populate_split(proc_dir, speakers, n_per_speaker=2, sr=16000)
        _populate_test_split(proc_dir, speaker_ids=["p232", "p257"], sr=16000)
        result = main([
            "--raw-dir", str(raw_dir),
            "--processed-dir", str(proc_dir),
        ])
        assert result == 0
