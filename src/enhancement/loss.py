"""Loss functions for speech enhancement.

Provides magnitude spectrogram losses (L1, MSE, Smooth L1) with optional
power-law compression and sequence-length masking for variable-length batches.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

LossType = Literal["l1", "mse", "smooth_l1"]


class SpectralLoss(nn.Module):
    """Loss function computed on magnitude spectrograms.

    Parameters
    ----------
    loss_type : {'l1', 'mse', 'smooth_l1'}
        Base loss function to apply between predicted and target magnitudes.
        Default: 'l1' (linear penalty, robust across formants and quiet consonants).
    mag_power : float
        Power-law compression factor applied before loss computation:
        ``mag ** mag_power``.  Default: 1.0 (linear magnitude).
        Values like 0.5 (amplitude compression) or 0.3 are common in literature
        to balance quiet and loud frequency bins.
    eps : float
        Epsilon for numerical stability during power compression.
    """

    def __init__(
        self,
        loss_type: LossType = "l1",
        mag_power: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if loss_type not in ("l1", "mse", "smooth_l1"):
            raise ValueError(f"Unsupported loss_type: {loss_type!r}. Choose from ('l1', 'mse', 'smooth_l1').")
        if mag_power <= 0.0:
            raise ValueError(f"mag_power must be positive, got {mag_power}")

        self.loss_type = loss_type
        self.mag_power = mag_power
        self.eps = eps

    def forward(
        self,
        pred_mag: torch.Tensor,
        target_mag: torch.Tensor,
        frame_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute spectral loss between predicted and clean target magnitudes.

        Parameters
        ----------
        pred_mag : Tensor
            Predicted magnitude spectrogram of shape ``(B, F, N)`` or ``(B, 1, F, N)``.
        target_mag : Tensor
            Ground-truth clean magnitude spectrogram of matching shape.
        frame_lengths : Tensor or None
            Optional 1-D tensor ``(B,)`` specifying the number of valid time frames
            for each utterance in a padded batch.  If None, all frames are included.

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        if pred_mag.shape != target_mag.shape:
            raise ValueError(
                f"Shape mismatch: pred_mag {pred_mag.shape} vs target_mag {target_mag.shape}"
            )

        p = pred_mag
        t = target_mag

        # Optional power compression
        if self.mag_power != 1.0:
            p = torch.pow(torch.clamp(p, min=self.eps), self.mag_power)
            t = torch.pow(torch.clamp(t, min=self.eps), self.mag_power)

        if self.loss_type == "l1":
            diff = torch.abs(p - t)
        elif self.loss_type == "mse":
            diff = torch.square(p - t)
        elif self.loss_type == "smooth_l1":
            diff = F.smooth_l1_loss(p, t, reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        if frame_lengths is not None:
            # Create a boolean mask of shape (B, 1, 1, N) or (B, 1, N) matching diff
            # frame_lengths is (B,)
            num_frames = diff.shape[-1]
            idx = torch.arange(num_frames, device=diff.device).unsqueeze(0)  # (1, N)
            mask = idx < frame_lengths.unsqueeze(1)  # (B, N)

            # Broadcast mask to diff dimensions
            while mask.dim() < diff.dim():
                mask = mask.unsqueeze(1)  # (B, 1, 1, N) or (B, 1, N)

            mask = mask.expand_as(diff).float()
            valid_elements = mask.sum().clamp(min=1.0)
            return (diff * mask).sum() / valid_elements

        return diff.mean()
