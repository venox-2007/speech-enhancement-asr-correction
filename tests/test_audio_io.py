import numpy as np
import pytest
import soundfile as sf

from src.preprocessing.audio_io import AudioLoadError, load_audio


def _write(path, sr, channels=1, seconds=0.5):
    t = np.arange(int(sr * seconds)) / sr
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    sf.write(path, data, sr)


def test_resamples_and_downmixes(tmp_path):
    path = tmp_path / "stereo_48k.wav"
    _write(path, 48_000, channels=2)
    audio = load_audio(path)
    assert audio.ndim == 1 and audio.dtype == np.float32
    assert audio.size == 8_000  # 0.5 s at 16 kHz


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "nope.wav")


def test_undecodable_file(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(AudioLoadError):
        load_audio(bad)
