"""STFT and inverse-STFT for speech enhancement.

Wraps ``torch.stft`` / ``torch.istft`` with a configuration dataclass so
that the same parameters are used consistently across training,
evaluation, and inference.

Default parameters at 16 kHz:
    n_fft=512     →  32 ms window
    hop_length=256 →  16 ms hop (50 % overlap)
    win_length=512
    window=hann
    → 257 frequency bins (n_fft // 2 + 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class STFTConfig:
    """STFT hyper-parameters.

    All defaults are tuned for 16 kHz speech enhancement.
    """

    n_fft: int = 512
    hop_length: int = 256
    win_length: int = 512
    window: str = "hann"

    @property
    def n_freq(self) -> int:
        """Number of frequency bins (n_fft // 2 + 1)."""
        return self.n_fft // 2 + 1

    def __post_init__(self) -> None:
        if self.n_fft < 1:
            raise ValueError("n_fft must be >= 1")
        if self.hop_length < 1:
            raise ValueError("hop_length must be >= 1")
        if self.win_length < 1 or self.win_length > self.n_fft:
            raise ValueError("win_length must be in [1, n_fft]")
        if self.window not in ("hann", "hamming"):
            raise ValueError(f"Unsupported window type: {self.window!r}")


def _get_window(config: STFTConfig, device: torch.device | None = None) -> torch.Tensor:
    """Create the analysis/synthesis window tensor."""
    if config.window == "hann":
        return torch.hann_window(config.win_length, device=device)
    if config.window == "hamming":
        return torch.hamming_window(config.win_length, device=device)
    raise ValueError(f"Unsupported window: {config.window!r}")


# ------------------------------------------------------------------ forward

def stft(
    audio: torch.Tensor,
    config: STFTConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the magnitude and phase of the Short-Time Fourier Transform.

    Parameters
    ----------
    audio : Tensor
        1-D ``(T,)`` or batched ``(B, T)`` float waveform.
    config : STFTConfig or None
        STFT parameters.  Defaults to ``STFTConfig()``.

    Returns
    -------
    magnitude : Tensor
        ``(B, F, N)`` or ``(F, N)`` magnitude spectrogram.
    phase : Tensor
        ``(B, F, N)`` or ``(F, N)`` phase in radians.
    """
    config = config or STFTConfig()
    window = _get_window(config, device=audio.device)

    squeezed = False
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        squeezed = True

    spec = torch.stft(
        audio,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        return_complex=True,
    )

    magnitude = spec.abs()
    phase = spec.angle()

    if squeezed:
        magnitude = magnitude.squeeze(0)
        phase = phase.squeeze(0)

    return magnitude, phase


# ----------------------------------------------------------------- inverse

def istft(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    config: STFTConfig | None = None,
    length: int | None = None,
) -> torch.Tensor:
    """Reconstruct a waveform from magnitude and phase spectrograms.

    Parameters
    ----------
    magnitude : Tensor
        ``(B, F, N)`` or ``(F, N)`` magnitude spectrogram.
    phase : Tensor
        ``(B, F, N)`` or ``(F, N)`` phase in radians.
    config : STFTConfig or None
        Must match the config used in :func:`stft`.
    length : int or None
        Desired output length.  If *None*, length is inferred.

    Returns
    -------
    audio : Tensor
        ``(B, T)`` or ``(T,)`` reconstructed waveform.
    """
    config = config or STFTConfig()
    window = _get_window(config, device=magnitude.device)

    squeezed = False
    if magnitude.dim() == 2:
        magnitude = magnitude.unsqueeze(0)
        phase = phase.unsqueeze(0)
        squeezed = True

    spec = magnitude * torch.exp(1j * phase)
    audio = torch.istft(
        spec,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        length=length,
    )

    if squeezed:
        audio = audio.squeeze(0)

    return audio


# ------------------------------------------------------------ dB conversion

_LOG10 = math.log(10.0)
_FLOOR = 1e-10


def mag_to_db(
    magnitude: torch.Tensor,
    ref: float = 1.0,
    floor: float = _FLOOR,
) -> torch.Tensor:
    """Convert magnitude spectrogram to decibels.

    ``dB = 20 * log10(magnitude / ref)``, clamped above *floor*.
    """
    return 20.0 * torch.log10(magnitude.clamp(min=floor) / ref)


def db_to_mag(
    db: torch.Tensor,
    ref: float = 1.0,
) -> torch.Tensor:
    """Convert decibel spectrogram back to linear magnitude."""
    return ref * torch.pow(10.0, db / 20.0)
