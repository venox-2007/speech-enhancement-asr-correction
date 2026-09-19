"""Unit tests for the demo audio pipeline runner and helper functions."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from src.demo.pipeline_runner import (
    PipelineResult,
    PipelineRunner,
    StageTiming,
    get_default_device,
    load_audio_from_source,
    tensor_to_wav_bytes,
)


def _create_synthetic_wav_bytes(duration_s: float = 1.0, sr: int = 16_000) -> bytes:
    """Helper to generate in-memory synthetic sine wave WAV bytes."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_load_audio_from_source_bytes() -> None:
    wav_bytes = _create_synthetic_wav_bytes(duration_s=1.5, sr=16_000)
    tensor, sr = load_audio_from_source(wav_bytes, target_sample_rate=16_000)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 1
    assert sr == 16_000
    assert len(tensor) == 24_000
    assert tensor.dtype == torch.float32
    assert -1.0 <= tensor.min().item() <= 1.0
    assert -1.0 <= tensor.max().item() <= 1.0


def test_load_audio_from_source_resampling() -> None:
    # 8 kHz source audio
    wav_bytes = _create_synthetic_wav_bytes(duration_s=1.0, sr=8_000)
    tensor, sr = load_audio_from_source(wav_bytes, target_sample_rate=16_000)

    assert sr == 16_000
    assert len(tensor) == 16_000


def test_load_audio_from_source_empty_buffer() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_audio_from_source(b"")


def test_load_audio_from_source_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_audio_from_source(Path("non_existent_audio_path_12345.wav"))


def test_tensor_to_wav_bytes() -> None:
    tensor = torch.sin(torch.linspace(0, 2 * np.pi * 440, 16000))
    wav_bytes = tensor_to_wav_bytes(tensor, sample_rate=16_000)

    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0

    # Verify readable by soundfile
    data, sr = sf.read(io.BytesIO(wav_bytes))
    assert sr == 16_000
    assert len(data) == 16_000


def test_stage_timing_and_result_dataclass() -> None:
    timing = StageTiming(
        preprocessing_s=0.01,
        enhancement_s=0.05,
        asr_raw_s=0.20,
        asr_enhanced_s=0.21,
        correction_raw_s=0.10,
        correction_enhanced_s=0.11,
        total_s=0.68,
    )
    d = timing.to_dict()
    assert d["total_s"] == pytest.approx(0.68)
    assert d["asr_raw_s"] == pytest.approx(0.20)

    dummy_tensor = torch.zeros(16000)
    res = PipelineResult(
        input_audio=dummy_tensor,
        enhanced_audio=dummy_tensor,
        sample_rate=16000,
        duration_s=1.0,
        raw_transcript="hello world",
        enhanced_transcript="hello word",
        corrected_raw_transcript="hello world",
        corrected_enhanced_transcript="hello world",
        raw_corrections=[],
        enhanced_corrections=[],
        timing=timing,
        device="cpu",
        enhancer_loaded=True,
    )
    rd = res.to_dict()
    assert rd["duration_s"] == 1.0
    assert rd["raw_transcript"] == "hello world"
    assert rd["enhancer_loaded"] is True


def test_pipeline_runner_mocked_execution() -> None:
    # Test runner execution flow with lightweight mocks
    runner = PipelineRunner(checkpoint_path=None, device="cpu")

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = "this is a test sentence"
    runner._asr = mock_asr

    mock_corrector = MagicMock()
    mock_corr_res = MagicMock()
    mock_corr_res.corrected = "this is a test sentence"
    mock_corr_res.word_corrections = []
    mock_corrector.correct.return_value = mock_corr_res
    runner._corrector = mock_corrector

    wav_bytes = _create_synthetic_wav_bytes(duration_s=1.0, sr=16_000)
    res = runner.process(wav_bytes, run_enhancement=False, run_correction=True)

    assert res.raw_transcript == "this is a test sentence"
    assert res.corrected_raw_transcript == "this is a test sentence"
    assert res.enhanced_audio is None
    assert res.timing.total_s > 0
    assert res.timing.asr_raw_s > 0
    assert res.timing.correction_raw_s > 0
