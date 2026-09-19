"""Spectrogram visualization and dataset statistics.

Plotting requires ``matplotlib``.  The functions return ``Figure`` /
``Axes`` objects so callers can save or display them as needed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from src.preprocessing.stft import STFTConfig, mag_to_db, stft

logger = logging.getLogger(__name__)


def _ensure_matplotlib():
    """Import matplotlib, raising a clear error if missing."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        ) from exc


def plot_spectrogram(
    magnitude: torch.Tensor,
    config: STFTConfig | None = None,
    sample_rate: int = 16_000,
    title: str = "Spectrogram",
    ax: Any = None,
    db: bool = True,
) -> Any:
    """Plot a single magnitude spectrogram as a heatmap.

    Parameters
    ----------
    magnitude : Tensor
        ``(F, N)`` magnitude spectrogram.
    config : STFTConfig or None
        Used to compute axis labels.
    sample_rate : int
        Audio sample rate for the time axis.
    title : str
        Plot title.
    ax : matplotlib Axes or None
        Axes to plot on.  If *None*, a new figure is created.
    db : bool
        Convert magnitude to decibels before plotting.

    Returns
    -------
    ax : matplotlib Axes
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    config = config or STFTConfig()

    data = magnitude.detach().cpu()
    if data.dim() == 3 and data.shape[0] == 1:
        data = data.squeeze(0)
    elif data.dim() != 2:
        raise ValueError(
            f"Expected 2D magnitude spectrogram (F, N) or (1, F, N), got shape {magnitude.shape}"
        )

    if db:
        data = mag_to_db(data)
    data = data.numpy()

    if ax is None:
        _fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    n_frames = data.shape[1]
    duration = n_frames * config.hop_length / sample_rate
    freq_max = sample_rate / 2

    im = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[0, duration, 0, freq_max / 1000],
        cmap="magma",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, label="dB" if db else "Magnitude")

    return ax


def plot_comparison(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    enhanced: torch.Tensor | None = None,
    config: STFTConfig | None = None,
    sample_rate: int = 16_000,
    title: str = "Spectrogram Comparison",
) -> Any:
    """Side-by-side spectrograms of clean, noisy, and optionally enhanced audio.

    Parameters
    ----------
    clean : Tensor
        1-D ``(T,)`` or ``(1, T)`` clean waveform.
    noisy : Tensor
        1-D ``(T,)`` or ``(1, T)`` noisy waveform.
    enhanced : Tensor or None
        1-D ``(T,)`` or ``(1, T)`` enhanced waveform (optional).
    config : STFTConfig or None
        STFT parameters.
    sample_rate : int
        Audio sample rate.
    title : str
        Super-title for the figure.

    Returns
    -------
    fig : matplotlib Figure
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    config = config or STFTConfig()

    # Squeeze if (1, T)
    if clean.dim() == 2 and clean.shape[0] == 1:
        clean = clean.squeeze(0)
    if noisy.dim() == 2 and noisy.shape[0] == 1:
        noisy = noisy.squeeze(0)
    if enhanced is not None and enhanced.dim() == 2 and enhanced.shape[0] == 1:
        enhanced = enhanced.squeeze(0)

    n_cols = 3 if enhanced is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))

    mag_clean, _ = stft(clean, config)
    mag_noisy, _ = stft(noisy, config)

    plot_spectrogram(mag_clean, config, sample_rate, "Clean", axes[0])
    plot_spectrogram(mag_noisy, config, sample_rate, "Noisy", axes[1])

    if enhanced is not None:
        mag_enh, _ = stft(enhanced, config)
        plot_spectrogram(mag_enh, config, sample_rate, "Enhanced", axes[2])

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_waveform_spectrogram_grid(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    enhanced: torch.Tensor,
    config: STFTConfig | None = None,
    sample_rate: int = 16_000,
    title: str = "Clean vs. Noisy vs. Enhanced Comparison",
    metrics: dict[str, Any] | None = None,
) -> Any:
    """Publication-quality 2x3 comparison grid: waveforms (top) and spectrograms (bottom).

    Parameters
    ----------
    clean : Tensor
        Clean speech waveform.
    noisy : Tensor
        Noisy speech waveform.
    enhanced : Tensor
        Enhanced speech waveform.
    config : STFTConfig or None
        STFT configuration parameters.
    sample_rate : int
        Audio sample rate in Hz.
    title : str
        Main title for the figure.
    metrics : dict or None
        Optional dictionary containing metrics (e.g. STOI, SI-SDR, SNR) to display.

    Returns
    -------
    fig : matplotlib Figure
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    config = config or STFTConfig()

    def _prep_1d(t: torch.Tensor) -> np.ndarray:
        d = t.detach().cpu()
        if d.dim() == 2 and d.shape[0] == 1:
            d = d.squeeze(0)
        return d.numpy()

    c_np = _prep_1d(clean)
    n_np = _prep_1d(noisy)
    e_np = _prep_1d(enhanced)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=False)

    signals = [
        ("Clean Speech", c_np, "green"),
        ("Noisy Speech", n_np, "crimson"),
        ("U-Net Enhanced", e_np, "royalblue"),
    ]

    # Row 0: Waveforms
    for col_idx, (col_name, sig, color) in enumerate(signals):
        ax = axes[0, col_idx]
        t = np.arange(len(sig)) / sample_rate
        ax.plot(t, sig, color=color, linewidth=0.6, alpha=0.85)
        ax.set_title(col_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylim(-1.05, 1.05)

    # Row 1: Spectrograms
    mag_clean, _ = stft(torch.from_numpy(c_np), config)
    mag_noisy, _ = stft(torch.from_numpy(n_np), config)
    mag_enh, _ = stft(torch.from_numpy(e_np), config)

    plot_spectrogram(mag_clean, config, sample_rate, "Clean Spectrogram", axes[1, 0])
    plot_spectrogram(mag_noisy, config, sample_rate, "Noisy Spectrogram", axes[1, 1])
    plot_spectrogram(mag_enh, config, sample_rate, "Enhanced Spectrogram", axes[1, 2])

    if metrics:
        metric_parts = []
        if "noisy_stoi" in metrics and "enhanced_stoi" in metrics:
            metric_parts.append(
                f"STOI: {metrics['noisy_stoi']:.3f} \u2192 {metrics['enhanced_stoi']:.3f} "
                f"(\u0394 {metrics.get('delta_stoi', 0.0):+.3f})"
            )
        if "noisy_sisdr" in metrics and "enhanced_sisdr" in metrics:
            metric_parts.append(
                f"SI-SDR: {metrics['noisy_sisdr']:.1f} \u2192 {metrics['enhanced_sisdr']:.1f} dB "
                f"(\u0394 {metrics.get('delta_sisdr', 0.0):+.1f} dB)"
            )
        if "noisy_snr" in metrics and "enhanced_snr" in metrics:
            metric_parts.append(
                f"SNR: {metrics['noisy_snr']:.1f} \u2192 {metrics['enhanced_snr']:.1f} dB "
                f"(\u0394 {metrics.get('delta_snr', 0.0):+.1f} dB)"
            )
        subtitle = " | ".join(metric_parts)
        fig.suptitle(f"{title}\n{subtitle}", fontsize=13, y=0.99)
    else:
        fig.suptitle(title, fontsize=13, y=0.99)

    fig.tight_layout()
    return fig


def dataset_summary(dataset: Any) -> dict[str, Any]:
    """Compute summary statistics for a VoiceBankDEMANDDataset.

    Iterates over the dataset's file list (does **not** load audio) to
    gather per-speaker utterance counts.  Optionally loads a small sample
    to estimate duration statistics.

    Parameters
    ----------
    dataset : VoiceBankDEMANDDataset
        The dataset to summarize.

    Returns
    -------
    dict
        Summary with keys: ``split``, ``num_utterances``,
        ``num_speakers``, ``speakers``, ``per_speaker_counts``.
    """
    from collections import Counter

    per_speaker: Counter[str] = Counter()
    for clean_path, _noisy_path in dataset.pairs:
        from src.enhancement.dataset import _extract_speaker

        spk = _extract_speaker(clean_path.name)
        if spk:
            per_speaker[spk] += 1

    return {
        "split": dataset.split,
        "num_utterances": len(dataset),
        "num_speakers": len(dataset.speakers),
        "speakers": dataset.speakers,
        "per_speaker_counts": dict(sorted(per_speaker.items())),
    }
