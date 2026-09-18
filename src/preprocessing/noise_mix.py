"""Noise mixing utilities.

Add noise to clean speech at a specified signal-to-noise ratio (SNR).
Useful for data augmentation and creating controlled test conditions.
"""

from __future__ import annotations

import torch


def _rms(signal: torch.Tensor) -> torch.Tensor:
    """Root mean square energy of a 1-D signal."""
    return signal.pow(2).mean().sqrt()


def mix_at_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """Mix clean speech with noise at a target SNR.

    The *noise* signal is scaled so that the mixture has approximately
    the requested SNR.  The clean signal is not modified.

    Parameters
    ----------
    clean : Tensor
        1-D ``(T,)`` clean speech waveform.
    noise : Tensor
        1-D ``(T,)`` noise waveform (same length as *clean*).
    snr_db : float
        Desired signal-to-noise ratio in decibels.

    Returns
    -------
    Tensor
        ``(T,)`` noisy mixture ``clean + scaled_noise``.

    Raises
    ------
    ValueError
        If inputs have different lengths or are not 1-D.
    """
    if clean.dim() != 1 or noise.dim() != 1:
        raise ValueError("clean and noise must be 1-D tensors")
    if clean.shape[0] != noise.shape[0]:
        raise ValueError(
            f"clean and noise must have the same length, "
            f"got {clean.shape[0]} and {noise.shape[0]}"
        )

    clean_rms = _rms(clean)
    noise_rms = _rms(noise)

    if noise_rms < 1e-10:
        # Silent noise — return clean unchanged
        return clean.clone()

    # SNR = 20 * log10(clean_rms / noise_rms)
    # → noise_rms_target = clean_rms / 10^(snr_db / 20)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / noise_rms
    return clean + noise * scale


def estimate_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """Estimate SNR in dB from a clean/noisy pair.

    Assumes ``noisy = clean + noise``.

    Parameters
    ----------
    clean : Tensor
        1-D ``(T,)`` clean speech waveform.
    noisy : Tensor
        1-D ``(T,)`` noisy mixture.

    Returns
    -------
    float
        Estimated SNR in decibels.
    """
    if clean.dim() != 1 or noisy.dim() != 1:
        raise ValueError("clean and noisy must be 1-D tensors")
    if clean.shape[0] != noisy.shape[0]:
        raise ValueError("clean and noisy must have the same length")

    noise = noisy - clean
    clean_rms = _rms(clean)
    noise_rms = _rms(noise)

    if noise_rms < 1e-10:
        return float("inf")
    return 20.0 * torch.log10(clean_rms / noise_rms).item()
