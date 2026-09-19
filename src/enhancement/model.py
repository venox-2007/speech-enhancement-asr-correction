"""End-to-end speech enhancement model wrapper.

Connects the STFT/iSTFT pipeline with the 2D U-Net magnitude enhancer:
noisy waveform -> STFT -> magnitude -> U-Net -> enhanced magnitude -> noisy phase -> iSTFT.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from src.enhancement.unet import UNet2D, UNetConfig
from src.preprocessing.stft import STFTConfig, istft, stft

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnhancementModelConfig:
    """Configuration bundling STFT and U-Net parameters.

    Parameters
    ----------
    stft : STFTConfig
        Parameters for STFT and iSTFT analysis/synthesis.
    unet : UNetConfig
        Architecture configuration for the 2D U-Net.
    sample_rate : int
        Audio sample rate in Hz (default: 16,000).
    """

    stft: STFTConfig = field(default_factory=STFTConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    sample_rate: int = 16_000


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable and total parameters in a PyTorch module.

    Returns
    -------
    dict
        {'total': int, 'trainable': int, 'non_trainable': int}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def model_summary(model: nn.Module) -> str:
    """Return a human-readable summary of the model and its parameters."""
    counts = count_parameters(model)
    total_m = counts["total"] / 1e6
    train_m = counts["trainable"] / 1e6
    return (
        f"{model.__class__.__name__} Summary:\n"
        f"  Total parameters:     {counts['total']:,} ({total_m:.3f} M)\n"
        f"  Trainable parameters: {counts['trainable']:,} ({train_m:.3f} M)\n"
        f"  Non-trainable:        {counts['non_trainable']:,}"
    )


class WaveformUNetEnhancer(nn.Module):
    """End-to-end waveform speech enhancement wrapper around 2D U-Net.

    Data flow:
    1. Input waveform `y(t)` -> STFT -> magnitude `|Y(f, t)|` and phase `phi_Y(f, t)`.
    2. Magnitude -> 2D U-Net -> enhanced magnitude `|S_hat(f, t)|`.
    3. Enhanced magnitude + original noisy phase -> iSTFT -> enhanced waveform `s_hat(t)`.
    """

    def __init__(self, config: EnhancementModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnhancementModelConfig()
        self.stft_config = self.config.stft
        self.unet = UNet2D(self.config.unet)

    def enhance_spectrogram(self, noisy_mag: torch.Tensor) -> torch.Tensor:
        """Enhance magnitude spectrogram directly (suitable for spectrogram-level loss).

        Parameters
        ----------
        noisy_mag : Tensor
            ``(B, F, N)`` or ``(F, N)`` noisy magnitude spectrogram.

        Returns
        -------
        Tensor
            ``(B, F, N)`` or ``(F, N)`` enhanced magnitude spectrogram.
        """
        squeezed = False
        if noisy_mag.dim() == 2:
            noisy_mag = noisy_mag.unsqueeze(0)
            squeezed = True

        # Add channel dimension: (B, 1, F, N)
        mag_4d = noisy_mag.unsqueeze(1)
        enhanced_4d = self.unet(mag_4d)
        enhanced_mag = enhanced_4d.squeeze(1)

        if squeezed:
            enhanced_mag = enhanced_mag.squeeze(0)

        return enhanced_mag

    def forward(
        self,
        noisy_waveform: torch.Tensor,
        length: int | None = None,
    ) -> torch.Tensor:
        """Perform end-to-end speech enhancement on audio waveform.

        Parameters
        ----------
        noisy_waveform : Tensor
            1-D ``(T,)`` or 2-D ``(B, T)`` noisy audio waveform.
        length : int or None
            Exact target length for reconstructed audio.  If None, inferred
            from ``noisy_waveform.shape[-1]``.

        Returns
        -------
        Tensor
            Enhanced waveform of matching shape ``(T,)`` or ``(B, T)``.
        """
        if length is None:
            length = noisy_waveform.shape[-1]

        squeezed = False
        if noisy_waveform.dim() == 1:
            noisy_waveform = noisy_waveform.unsqueeze(0)
            squeezed = True

        # STFT
        noisy_mag, noisy_phase = stft(noisy_waveform, self.stft_config)

        # Enhance magnitude: (B, F, N) -> (B, 1, F, N) -> (B, 1, F, N) -> (B, F, N)
        enhanced_mag = self.enhance_spectrogram(noisy_mag)

        # iSTFT with original noisy phase
        enhanced_wav = istft(
            enhanced_mag,
            noisy_phase,
            self.stft_config,
            length=length,
        )

        if squeezed:
            enhanced_wav = enhanced_wav.squeeze(0)

        return enhanced_wav


def load_enhancer_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> WaveformUNetEnhancer:
    """Load a trained WaveformUNetEnhancer directly from a saved checkpoint.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to checkpoint file (e.g. ``checkpoint_best.pt``).
    device : str
        Target device for model execution ('cpu', 'cuda', etc.).

    Returns
    -------
    WaveformUNetEnhancer
        Instantiated enhancer with loaded weights set in eval mode.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    target_device = torch.device(device)
    ckpt = torch.load(str(path), map_location=target_device)

    cfg_dict = ckpt.get("config", {})
    stft_dict = cfg_dict.get("stft", {})
    unet_dict = cfg_dict.get("unet", {})
    sr = cfg_dict.get("sample_rate", 16000)

    # Reconstruct configs safely filtering valid fields
    stft_cfg = STFTConfig(**{k: v for k, v in stft_dict.items() if hasattr(STFTConfig, k)})
    unet_cfg = UNetConfig(**{k: v for k, v in unet_dict.items() if hasattr(UNetConfig, k)})
    model_cfg = EnhancementModelConfig(stft=stft_cfg, unet=unet_cfg, sample_rate=sr)

    model = WaveformUNetEnhancer(model_cfg).to(target_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
