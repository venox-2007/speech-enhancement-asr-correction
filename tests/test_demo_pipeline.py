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


@pytest.mark.parametrize("duration, message", [(0.01, "too short"), (60.1, "60 seconds")])
def test_demo_rejects_unsupported_lengths_before_loading_models(duration, message):
    runner = PipelineRunner(checkpoint_path=None, device="cpu")
    with pytest.raises(ValueError, match=message):
        runner.process(_create_synthetic_wav_bytes(duration_s=duration))
    assert runner._asr is None


def test_demo_rejects_silence_before_asr():
    runner = PipelineRunner(checkpoint_path=None, device="cpu")
    silent = tensor_to_wav_bytes(torch.zeros(16000))
    with pytest.raises(ValueError, match="silent"):
        runner.process(silent, run_enhancement=False)
    assert runner._asr is None


def test_requested_enhancement_fails_clearly_when_checkpoint_is_missing():
    runner = PipelineRunner(checkpoint_path=None, device="cpu")
    with pytest.raises(RuntimeError, match="checkpoint"):
        runner.process(_create_synthetic_wav_bytes())
    assert runner._asr is None


def test_tensor_to_wav_bytes() -> None:
    tensor = torch.sin(torch.linspace(0, 2 * np.pi * 440, 16000))
    wav_bytes = tensor_to_wav_bytes(tensor, sample_rate=16_000)

    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0

    # Verify readable by soundfile
    data, sr = sf.read(io.BytesIO(wav_bytes))
    assert sr == 16_000
    assert len(data) == 16_000
    # A very short upload still needs a valid WAV preview before duration validation.
    single = tensor_to_wav_bytes(torch.tensor([0.2]))
    assert len(sf.read(io.BytesIO(single))[0]) == 1


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


def test_plot_audio_comparison_matplotlib_colors() -> None:
    """Regression test: plot_audio_comparison must use valid matplotlib colors without CSS rgba errors."""
    import matplotlib.pyplot as plt
    from app import plot_audio_comparison

    # Test with both raw and enhanced audio (2 columns)
    raw = torch.randn(1, 16_000)
    enh = torch.randn(1, 16_000)
    fig = plot_audio_comparison(raw, enh, sr=16_000)
    assert fig is not None
    assert len(fig.axes) == 4  # 2x2 grid (raw waveform/spectrogram + enh waveform/spectrogram)
    plt.close(fig)

    # Test with only raw audio (1 column)
    fig_single = plot_audio_comparison(raw, None, sr=16_000)
    assert fig_single is not None
    assert len(fig_single.axes) == 2  # 2x1 grid (raw waveform/spectrogram)
    plt.close(fig_single)
    short = plot_audio_comparison(torch.tensor([0.2]), None)
    assert len(short.axes) == 2
    plt.close(short)


def test_streamlit_config_watcher_settings() -> None:
    """Verify that .streamlit/config.toml exists and configures fileWatcherType to none."""
    import toml
    config_path = Path(".streamlit/config.toml")
    assert config_path.is_file(), ".streamlit/config.toml must exist"

    content = config_path.read_text(encoding="utf-8")
    assert "fileWatcherType" in content
    assert '"none"' in content
    assert "folderWatchBlacklist" in content


def test_pipeline_runner_branch_separation_and_correction_filtering() -> None:
    """Verify that PipelineRunner produces distinct Branch A & B outputs and filters genuine corrections only."""
    runner = PipelineRunner(checkpoint_path=None)

    # Mock components to run fast without heavy models
    mock_asr = MagicMock()
    mock_asr.transcribe.side_effect = [
        MagicMock(text="the quick brown fox jumps"),   # raw audio transcription
        MagicMock(text="the quick brown box jumps"),   # enhanced audio transcription
    ]
    runner._asr = mock_asr

    mock_enhancer = MagicMock(return_value=torch.zeros(16_000))
    runner._enhancer = mock_enhancer

    mock_corrector = MagicMock()
    # Mock WordCorrection: one unchanged, one changed
    mock_word_keep = MagicMock(was_changed=False, original_token="quick", corrected_token="quick")
    mock_word_change = MagicMock(
        was_changed=True,
        original_token="box",
        corrected_token="fox",
        to_dict=lambda: {"original_token": "box", "corrected_token": "fox", "decision": {"reason": "corrected"}},
    )

    mock_res_raw = MagicMock(corrected="the quick brown fox jumps", word_corrections=[mock_word_keep])
    mock_res_enh = MagicMock(corrected="the quick brown fox jumps", word_corrections=[mock_word_change])
    mock_corrector.correct.side_effect = [mock_res_raw, mock_res_enh]
    runner._corrector = mock_corrector

    dummy_audio = _create_synthetic_wav_bytes(duration_s=1.0)
    result = runner.process(dummy_audio, run_enhancement=True, run_correction=True)

    # Branch A (Raw) checks
    assert result.raw_transcript == "the quick brown fox jumps"
    assert result.corrected_raw_transcript == "the quick brown fox jumps"
    assert len(result.raw_corrections) == 0  # No changes -> empty list -> UI displays "No correction applied"

    # Branch B (Enhanced) checks
    assert result.enhanced_transcript == "the quick brown box jumps"
    assert result.corrected_enhanced_transcript == "the quick brown fox jumps"
    assert len(result.enhanced_corrections) == 1  # Genuine correction only
    assert result.enhanced_corrections[0]["original_token"] == "box"
    assert result.enhanced_corrections[0]["corrected_token"] == "fox"


