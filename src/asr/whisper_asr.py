"""Whisper-based ASR wrapper.

This module is UI-agnostic: the CLI, the future Streamlit app and the full
enhancement -> ASR -> correction pipeline should all call :class:`WhisperASR`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing.audio_io import WHISPER_SAMPLE_RATE, load_audio
from src.utils.device import resolve_device

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "base.en"


class ASRError(RuntimeError):
    """Raised when model loading or transcription fails."""


@dataclass(frozen=True)
class ASRConfig:
    """Settings that affect transcription output (saved with results for reproducibility)."""

    model_name: str = DEFAULT_MODEL
    device: str = "auto"
    language: str | None = "en"
    # 0.0 = greedy decoding, deterministic. Whisper's own default is a
    # temperature-fallback schedule, which can vary between runs.
    temperature: float = 0.0
    download_root: str | None = None  # None -> Whisper's default cache (~/.cache/whisper)


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    model_name: str
    device: str
    audio_duration_s: float
    processing_time_s: float
    segments: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WhisperASR:
    """Lazily-loaded Whisper model. The model is loaded on first use and reused."""

    def __init__(self, config: ASRConfig | None = None) -> None:
        self.config = config or ASRConfig()
        self.device = resolve_device(self.config.device)
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import whisper  # deferred: heavy import

        name = self.config.model_name
        if name not in whisper.available_models():
            raise ASRError(
                f"Unknown Whisper model '{name}'. Available: {', '.join(whisper.available_models())}"
            )
        logger.info("Loading Whisper model '%s' on %s (downloads on first use)...", name, self.device)
        start = time.perf_counter()
        try:
            self._model = whisper.load_model(
                name, device=self.device, download_root=self.config.download_root
            )
        except Exception as exc:  # network, disk, CUDA OOM, corrupted checkpoint...
            raise ASRError(f"Failed to load Whisper model '{name}': {exc}") from exc
        logger.info("Model loaded in %.1fs", time.perf_counter() - start)

    def transcribe(self, audio: str | Path | np.ndarray) -> TranscriptionResult:
        """Transcribe a file path or a mono float32 array sampled at 16 kHz."""
        if isinstance(audio, (str, Path)):
            logger.info("Loading audio: %s", audio)
            audio = load_audio(audio, WHISPER_SAMPLE_RATE)
        audio = self._validate_array(audio)

        self.load()
        duration = audio.size / WHISPER_SAMPLE_RATE
        logger.info("Transcribing %.2fs of audio...", duration)
        start = time.perf_counter()
        try:
            raw = self._model.transcribe(
                audio,
                language=self.config.language,
                temperature=self.config.temperature,
                fp16=self.device == "cuda",
                verbose=None,
            )
        except Exception as exc:
            raise ASRError(f"Transcription failed: {exc}") from exc
        elapsed = time.perf_counter() - start

        return TranscriptionResult(
            text=str(raw.get("text", "")).strip(),
            language=raw.get("language"),
            model_name=self.config.model_name,
            device=self.device,
            audio_duration_s=round(duration, 3),
            processing_time_s=round(elapsed, 3),
            segments=[_slim_segment(s) for s in raw.get("segments", [])],
        )

    @staticmethod
    def _validate_array(audio: np.ndarray) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Expected a path or numpy array, got {type(audio).__name__}")
        if audio.ndim != 1:
            raise ValueError(f"Expected mono 1-D audio, got shape {audio.shape}")
        if audio.size == 0:
            raise ValueError("Audio array is empty")
        return audio.astype(np.float32, copy=False)


def _slim_segment(seg: dict[str, Any]) -> dict[str, Any]:
    """Keep only the segment fields useful downstream (timings, text, confidence proxies)."""
    keys = ("start", "end", "text", "avg_logprob", "no_speech_prob")
    return {k: seg[k] for k in keys if k in seg}
