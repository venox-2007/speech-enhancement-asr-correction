"""Compute-device selection shared by ASR and (later) enhancement models."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

VALID_DEVICES = ("auto", "cpu", "cuda")


def resolve_device(requested: str = "auto") -> str:
    """Return ``"cuda"`` or ``"cpu"`` for the requested device.

    ``"auto"`` picks CUDA when available. Explicitly requesting ``"cuda"``
    on a machine without a usable GPU raises instead of silently falling back,
    so experiment metadata never misreports the device.
    """
    requested = requested.lower().strip()
    if requested not in VALID_DEVICES:
        raise ValueError(f"Unknown device '{requested}'. Choose from {VALID_DEVICES}.")

    cuda_ok = torch.cuda.is_available()
    if requested == "auto":
        device = "cuda" if cuda_ok else "cpu"
        logger.info("Auto-selected device: %s", device)
        return device
    if requested == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA was requested but no CUDA-capable GPU is available to PyTorch.")
    return requested
