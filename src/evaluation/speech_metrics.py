"""Objective speech enhancement evaluation metrics.

Implements standard speech quality and intelligibility metrics:
- **STOI**: Short-Time Objective Intelligibility (via ``pystoi``)
- **SI-SDR**: Scale-Invariant Signal-to-Distortion Ratio
- **SNR**: Signal-to-Noise Ratio and SNR improvement
- **PESQ**: Perceptual Evaluation of Speech Quality (wideband ITU-T P.862.2, with graceful degradation)

Also provides pairwise evaluation (noisy vs. enhanced) and result aggregation
across speakers and SNR condition bins.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

_PESQ_AVAILABLE: bool | None = None


def _check_pesq_available() -> bool:
    """Check whether the pesq package is importable."""
    global _PESQ_AVAILABLE
    if _PESQ_AVAILABLE is None:
        try:
            import pesq  # noqa: F401

            _PESQ_AVAILABLE = True
        except (ImportError, Exception):
            _PESQ_AVAILABLE = False
            logger.warning(
                "pesq library is not installed or failed to import. "
                "PESQ metrics will be reported as None. "
                "STOI, SI-SDR, and SNR will continue normally."
            )
    return _PESQ_AVAILABLE


def _to_numpy_1d(audio: torch.Tensor | np.ndarray) -> np.ndarray:
    """Ensure audio is a 1D float32 numpy array."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        if audio.shape[0] == 1:
            audio = audio[0]
        elif audio.shape[1] == 1:
            audio = audio[:, 0]
        else:
            raise ValueError(f"Expected mono audio, got 2D shape {audio.shape}")
    elif audio.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {audio.shape}")
    return audio


def calculate_stoi(
    clean: torch.Tensor | np.ndarray,
    deg: torch.Tensor | np.ndarray,
    sr: int = 16_000,
    extended: bool = False,
) -> float:
    """Compute Short-Time Objective Intelligibility (STOI).

    Parameters
    ----------
    clean : Tensor or ndarray
        Reference clean speech signal.
    deg : Tensor or ndarray
        Degraded (noisy or enhanced) speech signal.
    sr : int
        Sample rate (typically 16000 or 10000).
    extended : bool
        If True, compute extended STOI (ESTOI). Default False.

    Returns
    -------
    float
        STOI score in [0.0, 1.0]. Higher is better.
    """
    try:
        from pystoi import stoi
    except ImportError as exc:
        raise ImportError(
            "pystoi is required for STOI calculation. Install with: pip install pystoi"
        ) from exc

    c = _to_numpy_1d(clean)
    d = _to_numpy_1d(deg)
    min_len = min(len(c), len(d))
    if min_len == 0:
        return 0.0
    c = c[:min_len]
    d = d[:min_len]

    score = stoi(c, d, sr, extended=extended)
    return float(score)


def calculate_sisdr(
    clean: torch.Tensor | np.ndarray,
    deg: torch.Tensor | np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Formula:
        s_target = ( <s, d> / ||s||^2 ) * s
        e_noise = d - s_target
        SI-SDR = 10 * log10( ||s_target||^2 / ||e_noise||^2 )

    Parameters
    ----------
    clean : Tensor or ndarray
        Reference clean speech signal.
    deg : Tensor or ndarray
        Degraded (noisy or enhanced) speech signal.
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    float
        SI-SDR in decibels (dB). Higher is better.
    """
    c = _to_numpy_1d(clean)
    d = _to_numpy_1d(deg)
    min_len = min(len(c), len(d))
    if min_len == 0:
        return float("-inf")
    c = c[:min_len].astype(np.float64)
    d = d[:min_len].astype(np.float64)

    # Zero-mean normalization
    c = c - np.mean(c)
    d = d - np.mean(d)

    dot = np.dot(d, c)
    c_energy = np.dot(c, c) + eps
    alpha = dot / c_energy

    s_target = alpha * c
    e_noise = d - s_target

    target_pow = np.sum(s_target**2) + eps
    noise_pow = np.sum(e_noise**2) + eps

    sisdr = 10.0 * np.log10(target_pow / noise_pow)
    return float(sisdr)


def calculate_snr(
    clean: torch.Tensor | np.ndarray,
    deg: torch.Tensor | np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in dB.

    Formula:
        SNR = 10 * log10( ||s||^2 / ||d - s||^2 )

    Parameters
    ----------
    clean : Tensor or ndarray
        Reference clean speech signal.
    deg : Tensor or ndarray
        Degraded (noisy or enhanced) speech signal.
    eps : float
        Numerical stability epsilon.

    Returns
    -------
    float
        SNR in decibels (dB). Higher is better.
    """
    c = _to_numpy_1d(clean)
    d = _to_numpy_1d(deg)
    min_len = min(len(c), len(d))
    if min_len == 0:
        return float("-inf")
    c = c[:min_len].astype(np.float64)
    d = d[:min_len].astype(np.float64)

    clean_power = np.sum(c**2) + eps
    noise_power = np.sum((d - c) ** 2) + eps

    snr = 10.0 * np.log10(clean_power / noise_power)
    return float(snr)


def calculate_pesq(
    clean: torch.Tensor | np.ndarray,
    deg: torch.Tensor | np.ndarray,
    sr: int = 16_000,
) -> float | None:
    """Compute wideband Perceptual Evaluation of Speech Quality (PESQ).

    Supports wideband (16 kHz, ITU-T P.862.2) and narrowband (8 kHz, ITU-T P.862).
    If the ``pesq`` package is not installed on this system, gracefully returns
    ``None`` without halting evaluation.

    Parameters
    ----------
    clean : Tensor or ndarray
        Reference clean speech signal.
    deg : Tensor or ndarray
        Degraded (noisy or enhanced) speech signal.
    sr : int
        Sample rate (must be 8000 or 16000). Default: 16000.

    Returns
    -------
    float or None
        PESQ MOS score (typically between -0.5 and 4.5), or None if unavailable.
    """
    if not _check_pesq_available():
        return None

    try:
        from pesq import pesq

        mode = "wb" if sr == 16_000 else "nb"
        c = _to_numpy_1d(clean)
        d = _to_numpy_1d(deg)
        min_len = min(len(c), len(d))
        if min_len == 0:
            return None
        c = c[:min_len]
        d = d[:min_len]

        score = pesq(sr, c, d, mode)
        return float(score)
    except Exception as exc:
        logger.debug("PESQ calculation failed: %s", exc)
        return None


def compute_metrics_pair(
    clean: torch.Tensor | np.ndarray,
    noisy: torch.Tensor | np.ndarray,
    enhanced: torch.Tensor | np.ndarray,
    sr: int = 16_000,
) -> dict[str, float | None]:
    """Compute clean-vs-noisy and clean-vs-enhanced metrics plus deltas.

    Parameters
    ----------
    clean : Tensor or ndarray
        Clean speech ground truth.
    noisy : Tensor or ndarray
        Unprocessed noisy input speech.
    enhanced : Tensor or ndarray
        Model-enhanced output speech.
    sr : int
        Audio sample rate.

    Returns
    -------
    dict
        Dictionary of computed metrics:
        - ``noisy_stoi``, ``enhanced_stoi``, ``delta_stoi``
        - ``noisy_sisdr``, ``enhanced_sisdr``, ``delta_sisdr``
        - ``noisy_snr``, ``enhanced_snr``, ``delta_snr``
        - ``noisy_pesq``, ``enhanced_pesq``, ``delta_pesq``
    """
    n_stoi = calculate_stoi(clean, noisy, sr=sr)
    e_stoi = calculate_stoi(clean, enhanced, sr=sr)
    d_stoi = e_stoi - n_stoi

    n_sisdr = calculate_sisdr(clean, noisy)
    e_sisdr = calculate_sisdr(clean, enhanced)
    d_sisdr = e_sisdr - n_sisdr

    n_snr = calculate_snr(clean, noisy)
    e_snr = calculate_snr(clean, enhanced)
    d_snr = e_snr - n_snr

    n_pesq = calculate_pesq(clean, noisy, sr=sr)
    e_pesq = calculate_pesq(clean, enhanced, sr=sr)
    d_pesq = (
        (e_pesq - n_pesq)
        if (n_pesq is not None and e_pesq is not None)
        else None
    )

    return {
        "noisy_stoi": n_stoi,
        "enhanced_stoi": e_stoi,
        "delta_stoi": d_stoi,
        "noisy_sisdr": n_sisdr,
        "enhanced_sisdr": e_sisdr,
        "delta_sisdr": d_sisdr,
        "noisy_snr": n_snr,
        "enhanced_snr": e_snr,
        "delta_snr": d_snr,
        "noisy_pesq": n_pesq,
        "enhanced_pesq": e_pesq,
        "delta_pesq": d_pesq,
    }


def _stats_for_key(
    items: list[dict[str, Any]], key: str
) -> dict[str, float | None]:
    """Compute mean, std, min, max, median for a numeric metric key."""
    vals = [item[key] for item in items if item.get(key) is not None]
    if not vals:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(vals),
    }


def _assign_snr_bin(snr: float | None) -> str:
    """Classify input SNR into benchmark brackets."""
    if snr is None:
        return "unknown"
    if snr < 5.0:
        return "< 5 dB (Low SNR)"
    elif snr < 10.0:
        return "5 - 10 dB (Medium-Low SNR)"
    elif snr < 15.0:
        return "10 - 15 dB (Medium-High SNR)"
    else:
        return ">= 15 dB (High SNR)"


def aggregate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-utterance evaluation results into summary statistics.

    Provides:
    - Overall summary across all utterances.
    - Breakdown by speaker (e.g. p232, p257).
    - Breakdown by input SNR bracket.

    Parameters
    ----------
    results : list of dict
        List of per-utterance metric records, each containing metric keys
        and optional metadata (``speaker``, ``noisy_snr``, etc.).

    Returns
    -------
    dict
        Aggregated summary dictionary.
    """
    if not results:
        return {"count": 0, "overall": {}}

    metric_keys = [
        "noisy_stoi",
        "enhanced_stoi",
        "delta_stoi",
        "noisy_sisdr",
        "enhanced_sisdr",
        "delta_sisdr",
        "noisy_snr",
        "enhanced_snr",
        "delta_snr",
        "noisy_pesq",
        "enhanced_pesq",
        "delta_pesq",
    ]

    overall: dict[str, Any] = {}
    for k in metric_keys:
        overall[k] = _stats_for_key(results, k)

    # Per-speaker breakdown
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        spk = r.get("speaker", "unknown")
        by_speaker.setdefault(spk, []).append(r)

    speaker_summary: dict[str, dict[str, Any]] = {}
    for spk, spk_items in sorted(by_speaker.items()):
        speaker_summary[spk] = {
            "count": len(spk_items),
            "metrics": {k: _stats_for_key(spk_items, k) for k in metric_keys},
        }

    # Per-SNR bracket breakdown
    by_snr: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        snr_val = r.get("noisy_snr")
        bin_label = _assign_snr_bin(snr_val)
        by_snr.setdefault(bin_label, []).append(r)

    snr_summary: dict[str, dict[str, Any]] = {}
    snr_order = [
        "< 5 dB (Low SNR)",
        "5 - 10 dB (Medium-Low SNR)",
        "10 - 15 dB (Medium-High SNR)",
        ">= 15 dB (High SNR)",
        "unknown",
    ]
    for bin_label in snr_order:
        if bin_label in by_snr:
            snr_items = by_snr[bin_label]
            snr_summary[bin_label] = {
                "count": len(snr_items),
                "metrics": {k: _stats_for_key(snr_items, k) for k in metric_keys},
            }

    return {
        "count": len(results),
        "overall": overall,
        "by_speaker": speaker_summary,
        "by_snr_bracket": snr_summary,
    }
