"""Audio loading utilities.

Audio is decoded with ``soundfile`` (libsndfile), so WAV/FLAC/OGG files can be
read without an external ffmpeg installation.
"""

from __future__ import annotations

import logging
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16_000


class AudioLoadError(RuntimeError):
    """Raised when an audio file cannot be read or is unusable."""


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Average channels of a ``(frames, channels)`` array into a 1-D signal."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=1)
    raise AudioLoadError(f"Unsupported audio array shape: {audio.shape}")


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D signal using polyphase filtering."""
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g)


def load_audio(path: str | Path, target_sr: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as a mono float32 array at ``target_sr``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioLoadError(f"Could not decode audio file '{path}': {exc}") from exc

    audio = to_mono(audio)
    if audio.size == 0:
        raise AudioLoadError(f"Audio file contains no samples: {path}")

    logger.debug("Loaded %s: %d samples @ %d Hz", path.name, audio.size, sr)
    audio = resample(audio, sr, target_sr)
    return audio.astype(np.float32, copy=False)
