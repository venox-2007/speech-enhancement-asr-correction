"""Pipeline runner and audio processing service for the demo presentation layer.

Coordinates end-to-end execution across:
1. Audio loading, validation, and resampling to 16 kHz mono.
2. Optional speech enhancement via trained U-Net.
3. Whisper ASR transcription.
4. Context-aware phonetic correction with principled Bayesian safeguards.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.asr.whisper_asr import ASRConfig, WhisperASR
from src.correction.candidates import PhoneticCandidateGenerator
from src.correction.context_scorer import ContextAwareScorer, ContextScorerConfig
from src.correction.lm_scorer import CausalLMScorer
from src.correction.pipeline import CorrectionPipeline, CorrectionResult
from src.enhancement.model import WaveformUNetEnhancer, load_enhancer_checkpoint
from src.preprocessing.audio_io import resample

logger = logging.getLogger(__name__)


@dataclass
class StageTiming:
    """Execution latency breakdown in seconds."""

    preprocessing_s: float = 0.0
    enhancement_s: float = 0.0
    asr_raw_s: float = 0.0
    asr_enhanced_s: float = 0.0
    correction_raw_s: float = 0.0
    correction_enhanced_s: float = 0.0
    total_s: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Structured end-to-end execution output."""

    input_audio: torch.Tensor
    enhanced_audio: torch.Tensor | None
    sample_rate: int
    duration_s: float
    raw_transcript: str
    enhanced_transcript: str | None
    corrected_raw_transcript: str | None
    corrected_enhanced_transcript: str | None
    raw_corrections: list[dict[str, Any]] = field(default_factory=list)
    enhanced_corrections: list[dict[str, Any]] = field(default_factory=list)
    timing: StageTiming = field(default_factory=StageTiming)
    device: str = "cpu"
    enhancer_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "raw_transcript": self.raw_transcript,
            "enhanced_transcript": self.enhanced_transcript,
            "corrected_raw_transcript": self.corrected_raw_transcript,
            "corrected_enhanced_transcript": self.corrected_enhanced_transcript,
            "raw_corrections": self.raw_corrections,
            "enhanced_corrections": self.enhanced_corrections,
            "timing": self.timing.to_dict(),
            "device": self.device,
            "enhancer_loaded": self.enhancer_loaded,
        }


def get_default_device() -> str:
    """Detect default hardware accelerator."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_audio_from_source(
    audio_source: str | Path | bytes | io.BytesIO,
    target_sample_rate: int = 16_000,
) -> tuple[torch.Tensor, int]:
    """Load and validate audio from a file path or memory buffer.

    Parameters
    ----------
    audio_source : str, Path, bytes, or io.BytesIO
        Audio input.
    target_sample_rate : int
        Target sample rate in Hz (default: 16,000).

    Returns
    -------
    tuple[torch.Tensor, int]
        1D float32 audio tensor of shape (T,) and sample rate.
    """
    if isinstance(audio_source, (str, Path)):
        path = Path(audio_source)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        wav_data, sr = sf.read(str(path), dtype="float32")
    elif isinstance(audio_source, (bytes, bytearray)):
        if len(audio_source) == 0:
            raise ValueError("Provided audio byte buffer is empty.")
        buffer = io.BytesIO(audio_source)
        wav_data, sr = sf.read(buffer, dtype="float32")
    elif isinstance(audio_source, io.BytesIO):
        audio_source.seek(0)
        wav_data, sr = sf.read(audio_source, dtype="float32")
    else:
        raise TypeError(f"Unsupported audio source type: {type(audio_source)}")

    if wav_data is None or len(wav_data) == 0:
        raise ValueError("Decoded audio contains zero samples.")

    # Convert to torch tensor
    tensor = torch.from_numpy(wav_data)

    # Convert multi-channel to mono (average across channels)
    if tensor.ndim > 1:
        if tensor.shape[1] > tensor.shape[0]:  # (C, T)
            tensor = tensor.mean(dim=0)
        else:  # (T, C)
            tensor = tensor.mean(dim=1)
    else:
        tensor = tensor.squeeze()

    # Resample if needed
    if sr != target_sample_rate:
        resampled_np = resample(tensor.numpy(), orig_sr=sr, target_sr=target_sample_rate)
        tensor = torch.from_numpy(resampled_np)
        sr = target_sample_rate

    # Guard against NaN / Inf
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    # Normalize amplitude if clipping
    max_amp = torch.max(torch.abs(tensor)).item()
    if max_amp > 1.0:
        tensor = tensor / max_amp

    return tensor.float(), sr


def tensor_to_wav_bytes(waveform: torch.Tensor, sample_rate: int = 16_000) -> bytes:
    """Encode audio tensor as WAV in-memory bytes for playback."""
    audio_np = waveform.detach().cpu().squeeze().numpy()
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class PipelineRunner:
    """Manages model loading and pipeline execution."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = "results/enhancement_train/checkpoint_best.pt",
        whisper_model_name: str = "base.en",
        device: str | None = None,
        load_models_immediately: bool = False,
    ) -> None:
        self.device = device or get_default_device()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.whisper_model_name = whisper_model_name

        self._enhancer: WaveformUNetEnhancer | None = None
        self._asr: WhisperASR | None = None
        self._corrector: CorrectionPipeline | None = None

        if load_models_immediately:
            self.get_enhancer()
            self.get_asr()
            self.get_corrector()

    def get_enhancer(self) -> WaveformUNetEnhancer | None:
        """Lazily load trained U-Net speech enhancer."""
        if self._enhancer is None and self.checkpoint_path and self.checkpoint_path.is_file():
            try:
                logger.info("Loading speech enhancer from %s onto %s", self.checkpoint_path, self.device)
                self._enhancer = load_enhancer_checkpoint(self.checkpoint_path, device=self.device)
            except Exception as e:
                logger.warning("Could not load speech enhancer checkpoint: %s", e)
                self._enhancer = None
        return self._enhancer

    def get_asr(self) -> WhisperASR:
        """Lazily load Whisper ASR model."""
        if self._asr is None:
            logger.info("Loading Whisper model '%s' on %s", self.whisper_model_name, self.device)
            cfg = ASRConfig(model_name=self.whisper_model_name, device=self.device)
            self._asr = WhisperASR(cfg)
        return self._asr

    def get_corrector(self) -> CorrectionPipeline:
        """Lazily load context-aware phonetic corrector with improved safeguards."""
        if self._corrector is None:
            logger.info("Loading context-aware corrector with improved safeguards on %s", self.device)
            config = ContextScorerConfig(
                threshold=0.70,
                phonetic_weight=1.0,
                max_candidates=20,
                min_score_margin=3.5,
                min_phonetic_similarity=0.80,
                retention_bias=1.0,
            )
            generator = PhoneticCandidateGenerator(max_candidates=20)
            lm = CausalLMScorer(device=self.device, batch_size=16)
            scorer = ContextAwareScorer(generator=generator, lm=lm, config=config)
            self._corrector = CorrectionPipeline(scorer=scorer)
        return self._corrector

    def process(
        self,
        audio_source: str | Path | bytes | io.BytesIO,
        run_enhancement: bool = True,
        run_correction: bool = True,
    ) -> PipelineResult:
        """Run complete speech enhancement and ASR correction pipeline.

        Parameters
        ----------
        audio_source : str, Path, bytes, or BytesIO
            Input audio.
        run_enhancement : bool
            Whether to execute U-Net enhancement (if model is available).
        run_correction : bool
            Whether to execute context-aware post-ASR correction.

        Returns
        -------
        PipelineResult
            Complete results and per-stage timing.
        """
        timing = StageTiming()
        total_t0 = time.perf_counter()

        # 1. Ingestion & Preprocessing
        t0 = time.perf_counter()
        audio_tensor, sr = load_audio_from_source(audio_source, target_sample_rate=16_000)
        duration_s = float(audio_tensor.shape[-1] / sr)
        timing.preprocessing_s = time.perf_counter() - t0

        # 2. Speech Enhancement
        enhanced_audio: torch.Tensor | None = None
        enhancer = self.get_enhancer() if run_enhancement else None
        enhancer_loaded = enhancer is not None

        if run_enhancement and enhancer_loaded:
            t0 = time.perf_counter()
            with torch.no_grad():
                inp = audio_tensor.to(self.device)
                out = enhancer(inp)
                enhanced_audio = out.detach().cpu().squeeze()
            timing.enhancement_s = time.perf_counter() - t0

        # 3. Whisper ASR Transcription
        asr = self.get_asr()

        # Raw audio transcription
        t0 = time.perf_counter()
        raw_transcript = asr.transcribe(audio_tensor)
        timing.asr_raw_s = time.perf_counter() - t0

        # Enhanced audio transcription
        enhanced_transcript: str | None = None
        if enhanced_audio is not None:
            t0 = time.perf_counter()
            enhanced_transcript = asr.transcribe(enhanced_audio)
            timing.asr_enhanced_s = time.perf_counter() - t0

        # 4. Context-Aware Post-ASR Correction
        corrected_raw: str | None = None
        corrected_enh: str | None = None
        raw_corrections: list[dict[str, Any]] = []
        enh_corrections: list[dict[str, Any]] = []

        if run_correction:
            corrector = self.get_corrector()

            # Correct raw transcript
            t0 = time.perf_counter()
            raw_res = corrector.correct(raw_transcript)
            corrected_raw = raw_res.corrected
            raw_corrections = [wc.to_dict() for wc in raw_res.word_corrections if wc.was_changed]
            timing.correction_raw_s = time.perf_counter() - t0

            # Correct enhanced transcript if available
            if enhanced_transcript is not None:
                t0 = time.perf_counter()
                enh_res = corrector.correct(enhanced_transcript)
                corrected_enh = enh_res.corrected
                enh_corrections = [wc.to_dict() for wc in enh_res.word_corrections if wc.was_changed]
                timing.correction_enhanced_s = time.perf_counter() - t0

        timing.total_s = time.perf_counter() - total_t0

        return PipelineResult(
            input_audio=audio_tensor,
            enhanced_audio=enhanced_audio,
            sample_rate=sr,
            duration_s=duration_s,
            raw_transcript=raw_transcript,
            enhanced_transcript=enhanced_transcript,
            corrected_raw_transcript=corrected_raw,
            corrected_enhanced_transcript=corrected_enh,
            raw_corrections=raw_corrections,
            enhanced_corrections=enh_corrections,
            timing=timing,
            device=self.device,
            enhancer_loaded=enhancer_loaded,
        )
