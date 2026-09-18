"""Noise mixing utilities.

Add noise to clean speech at a specified signal-to-noise ratio (SNR).
Useful for data augmentation and creating controlled test conditions.
"""

from __future__ import annotations

import logging

import torch


logger = logging.getLogger(__name__)


def _rms(signal: torch.Tensor) -> torch.Tensor:
    """Root mean square energy of a 1-D signal."""
    return signal.pow(2).mean().sqrt()


def mix_at_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    clipping_policy: str = "warn",
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
    clipping_policy : {"warn", "error", "ignore", "normalize"}
        Action to take if the mixture exceeds dynamic range ``[-1.0, 1.0]``:
        - ``"warn"`` (default): log a warning, leave mixture unscaled so SNR is preserved.
        - ``"error"``: raise a ValueError.
        - ``"ignore"``: silently leave mixture unscaled.
        - ``"normalize"``: scale both clean and noise by ``1.0 / peak`` so the relative
          SNR is preserved, but overall gain is reduced.

    Returns
    -------
    Tensor
        ``(T,)`` noisy mixture ``clean + scaled_noise``.

    Raises
    ------
    ValueError
        If inputs have different lengths, are not 1-D, or if clipping occurs
        under ``clipping_policy="error"``.
    """
    valid_policies = ("warn", "error", "ignore", "normalize")
    if clipping_policy not in valid_policies:
        raise ValueError(
            f"Invalid clipping_policy: {clipping_policy!r}. "
            f"Must be one of {valid_policies}"
        )

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
    mixture = clean + noise * scale

    peak = mixture.abs().max().item()
    if peak > 1.0:
        if clipping_policy == "warn":
            logger.warning(
                "Audio mixture peak amplitude %.3f exceeds [-1.0, 1.0] range at SNR %.1f dB. "
                "Preserving intended SNR without scaling.",
                peak,
                snr_db,
            )
        elif clipping_policy == "error":
            raise ValueError(
                f"Audio mixture clipped with peak amplitude {peak:.3f} > 1.0 at SNR {snr_db:.1f} dB"
            )
        elif clipping_policy == "normalize":
            logger.info(
                "Peak amplitude %.3f exceeds 1.0; scaling mixture to prevent clipping while preserving SNR.",
                peak,
            )
            mixture = mixture / peak

    return mixture


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
