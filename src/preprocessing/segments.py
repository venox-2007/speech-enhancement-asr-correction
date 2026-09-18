"""Fixed-length audio segment extraction for training.

Speech enhancement models typically require fixed-length input during
training.  These helpers extract segments from longer utterances while
ensuring that *paired* clean/noisy segments use the same offset so the
alignment is preserved.
"""

from __future__ import annotations

import torch


def pad_or_crop(audio: torch.Tensor, length: int) -> torch.Tensor:
    """Return an audio tensor of exactly *length* samples.

    - If ``audio`` is shorter than *length*, zero-pad on the right.
    - If ``audio`` is longer, crop from the start.

    Parameters
    ----------
    audio : Tensor
        1-D ``(T,)`` waveform.
    length : int
        Desired number of samples.

    Returns
    -------
    Tensor
        ``(length,)`` waveform.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if audio.dim() != 1:
        raise ValueError(f"Expected 1-D tensor, got shape {audio.shape}")

    t = audio.shape[0]
    if t >= length:
        return audio[:length]
    padded = torch.zeros(length, dtype=audio.dtype, device=audio.device)
    padded[:t] = audio
    return padded


def random_crop(
    audio: torch.Tensor,
    length: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extract a random fixed-length segment.

    If *audio* is shorter than *length*, it is zero-padded first.

    Parameters
    ----------
    audio : Tensor
        1-D ``(T,)`` waveform.
    length : int
        Desired number of samples.
    generator : torch.Generator or None
        RNG for reproducibility.

    Returns
    -------
    Tensor
        ``(length,)`` waveform segment.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if audio.dim() != 1:
        raise ValueError(f"Expected 1-D tensor, got shape {audio.shape}")

    t = audio.shape[0]
    if t <= length:
        return pad_or_crop(audio, length)

    max_offset = t - length
    offset = torch.randint(0, max_offset + 1, (1,), generator=generator).item()
    return audio[offset : offset + length]


def paired_random_crop(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    length: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract aligned random segments from a clean/noisy pair.

    Both tensors are cropped at the **same** random offset so that the
    alignment between clean and noisy audio is preserved.

    Parameters
    ----------
    clean : Tensor
        1-D ``(T,)`` clean waveform.
    noisy : Tensor
        1-D ``(T,)`` noisy waveform (same length as *clean*).
    length : int
        Desired number of samples.
    generator : torch.Generator or None
        RNG for reproducibility.

    Returns
    -------
    clean_seg, noisy_seg : tuple[Tensor, Tensor]
        ``(length,)`` aligned segments.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if clean.dim() != 1 or noisy.dim() != 1:
        raise ValueError("Expected 1-D tensors")
    if clean.shape[0] != noisy.shape[0]:
        raise ValueError(
            f"clean and noisy must have the same length, "
            f"got {clean.shape[0]} and {noisy.shape[0]}"
        )

    t = clean.shape[0]
    if t <= length:
        return pad_or_crop(clean, length), pad_or_crop(noisy, length)

    max_offset = t - length
    offset = torch.randint(0, max_offset + 1, (1,), generator=generator).item()
    return clean[offset : offset + length], noisy[offset : offset + length]
