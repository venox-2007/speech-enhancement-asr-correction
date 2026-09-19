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


def test_pipeline_runner_passes_numpy_to_whisper_and_extracts_transcription_result() -> None:
    """Regression test: WhisperASR must receive a numpy array, not torch.Tensor, and return .text."""
    from src.asr.whisper_asr import TranscriptionResult

    runner = PipelineRunner(checkpoint_path=None, device="cpu")

    captured_inputs = []

    def fake_transcribe(audio: np.ndarray) -> TranscriptionResult:
        # Enforce that audio is np.ndarray and not torch.Tensor
        assert isinstance(audio, np.ndarray), f"Expected np.ndarray, got {type(audio)}"
        assert audio.ndim == 1
        assert audio.dtype == np.float32
        captured_inputs.append(audio)
        return TranscriptionResult(
            text="hello from whisper transcription result",
            language="en",
            model_name="base.en",
            device="cpu",
            audio_duration_s=1.0,
            processing_time_s=0.1,
            segments=[],
        )

    mock_asr = MagicMock()
    mock_asr.transcribe.side_effect = fake_transcribe
    runner._asr = mock_asr

    wav_bytes = _create_synthetic_wav_bytes(duration_s=1.0, sr=16_000)
    res = runner.process(wav_bytes, run_enhancement=False, run_correction=False)

    assert len(captured_inputs) == 1
    assert isinstance(captured_inputs[0], np.ndarray)
    assert res.raw_transcript == "hello from whisper transcription result"


def test_app_module_defines_logger() -> None:
    """Regression test: app.py must define module logger and not raise NameError on logging."""
    import logging
    import app

    assert hasattr(app, "logger")
    assert isinstance(app.logger, logging.Logger)


def test_load_audio_from_source_with_getvalue() -> None:
    """Test loading audio from an object implementing getvalue() (e.g. st.audio_input UploadedFile)."""
    wav_bytes = _create_synthetic_wav_bytes(duration_s=1.0, sr=16_000)

    class MockStreamlitAudioFile:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def getvalue(self) -> bytes:
            return self._data

    mock_file = MockStreamlitAudioFile(wav_bytes)
    tensor, sr = load_audio_from_source(mock_file, target_sample_rate=16_000)

    assert isinstance(tensor, torch.Tensor)
    assert sr == 16_000
    assert len(tensor) == 16_000


def test_load_audio_from_source_file_like_read() -> None:
    """Test loading audio from a generic file-like object implementing read() and seek()."""
    wav_bytes = _create_synthetic_wav_bytes(duration_s=0.5, sr=16_000)
    buf = io.BytesIO(wav_bytes)

    tensor, sr = load_audio_from_source(buf, target_sample_rate=16_000)
    assert isinstance(tensor, torch.Tensor)
    assert sr == 16_000
    assert len(tensor) == 8_000


def test_load_audio_from_source_multichannel() -> None:
    """Test loading stereo 2-channel audio and verifying mono averaging."""
    t = np.linspace(0, 1.0, 16_000, endpoint=False, dtype=np.float32)
    stereo_data = np.stack([0.4 * np.sin(2 * np.pi * 440 * t), 0.6 * np.sin(2 * np.pi * 440 * t)], axis=1)

    buf = io.BytesIO()
    sf.write(buf, stereo_data, 16_000, format="WAV", subtype="PCM_16")
    buf.seek(0)

    tensor, sr = load_audio_from_source(buf, target_sample_rate=16_000)
    assert tensor.ndim == 1
    assert len(tensor) == 16_000
    assert sr == 16_000


def test_wer_result_attribute_access_for_dashboard_rendering() -> None:
    """Regression test: WERResult must be accessed via attributes, not dictionary indexing."""
    from src.evaluation.wer import compute_wer, WERResult
    import pandas as pd

    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quick brown fox jumped over a lazy dog"

    wer_res = compute_wer(ref, hyp)
    assert isinstance(wer_res, WERResult)

    # Dictionary indexing should raise TypeError
    with pytest.raises(TypeError, match="'WERResult' object is not subscriptable"):
        _ = wer_res["wer"]  # type: ignore

    # Attribute access must succeed and construct a valid pandas DataFrame
    wer_rows = [
        {
            "Condition": "Raw Input Speech",
            "WER": f"{wer_res.wer * 100:.1f}%",
            "Errors (S/D/I)": f"{wer_res.substitutions}/{wer_res.deletions}/{wer_res.insertions}",
        }
    ]
    df = pd.DataFrame(wer_rows)
    assert len(df) == 1
    assert df["Condition"].iloc[0] == "Raw Input Speech"
    assert df["WER"].iloc[0] == "22.2%"
    assert df["Errors (S/D/I)"].iloc[0] == "2/0/0"
