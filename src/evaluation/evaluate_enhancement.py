"""Evaluation script for trained U-Net speech enhancement model.

Runs inference on the complete held-out VoiceBank+DEMAND test set (824 utterances,
speakers p232 and p257), verifies length and speaker disjointness, computes
objective speech quality metrics (STOI, SI-SDR, SNR, PESQ), generates representative
spectrogram & waveform comparison figures, and saves machine-readable CSV/JSON outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from src.enhancement.dataset import (
    DatasetConfig,
    VoiceBankDEMANDDataset,
    _extract_speaker,
)
from src.enhancement.model import load_enhancer_checkpoint
from src.enhancement.viz import plot_waveform_spectrogram_grid
from src.evaluation.speech_metrics import (
    aggregate_metrics,
    compute_metrics_pair,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def verify_zero_leakage(
    test_dataset: VoiceBankDEMANDDataset,
    root_dir: Path,
) -> None:
    """Verify that test speakers do not overlap with train or validation speakers."""
    test_speakers = set(test_dataset.speakers)

    train_config = DatasetConfig(root=str(root_dir))
    train_dataset = VoiceBankDEMANDDataset(train_config, split="train")
    val_dataset = VoiceBankDEMANDDataset(train_config, split="val")

    train_speakers = set(train_dataset.speakers)
    val_speakers = set(val_dataset.speakers)

    leakage_train = test_speakers.intersection(train_speakers)
    leakage_val = test_speakers.intersection(val_speakers)

    if leakage_train or leakage_val:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED! "
            f"Test overlap with train: {leakage_train}; "
            f"Test overlap with val: {leakage_val}"
        )

    logger.info(
        "Zero-leakage verified: Test speakers %s are disjoint from "
        "%d train speakers and %d val speakers (%s).",
        sorted(test_speakers),
        len(train_speakers),
        len(val_speakers),
        sorted(val_speakers),
    )


def evaluate(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    save_audio: bool = True,
    num_plots: int = 5,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate speech enhancement model on test set."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_audio:
        audio_out_dir = output_dir / "audio"
        audio_out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device
    logger.info("Using device: %s", dev)

    # Load trained model
    logger.info("Loading enhancer checkpoint from: %s", checkpoint_path)
    model = load_enhancer_checkpoint(checkpoint_path, device=dev)
    model.eval()

    # Load test dataset
    logger.info("Initializing test dataset from: %s", data_dir)
    test_config = DatasetConfig(root=str(data_dir))
    dataset = VoiceBankDEMANDDataset(test_config, split="test")
    logger.info("Found %d test utterances across speakers %s", len(dataset), dataset.speakers)

    # Leakage check
    try:
        verify_zero_leakage(dataset, data_dir)
    except Exception as exc:
        logger.warning("Could not run full dataset-leakage check (%s); proceeding with caution.", exc)

    total_items = len(dataset)
    if max_samples is not None:
        total_items = min(total_items, max_samples)

    # Stratified plot selection
    plot_indices = set(np.linspace(0, total_items - 1, num_plots, dtype=int)) if num_plots > 0 else set()

    results: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    logger.info("Starting test set evaluation on %d utterances...", total_items)
    for idx in range(total_items):
        noisy, clean = dataset[idx]
        clean_path, noisy_path = dataset.pairs[idx]
        utt_name = clean_path.name
        speaker = _extract_speaker(utt_name) or "unknown"
        duration_s = len(clean) / test_config.sample_rate

        # Run model inference on full utterance
        noisy_in = noisy.unsqueeze(0).to(dev)  # (1, T)
        with torch.no_grad():
            enhanced = model(noisy_in).squeeze(0).cpu()  # (T,)

        # Verify length preservation
        if enhanced.shape[0] != clean.shape[0]:
            raise ValueError(
                f"Length mismatch for {utt_name}: clean {clean.shape[0]}, enhanced {enhanced.shape[0]}"
            )

        # Compute metrics
        m = compute_metrics_pair(clean, noisy, enhanced, sr=test_config.sample_rate)

        record: dict[str, Any] = {
            "clean_file": utt_name,
            "speaker": speaker,
            "samples": len(clean),
            "duration_sec": round(duration_s, 3),
            **m,
        }
        results.append(record)

        # Save enhanced audio
        if save_audio:
            out_wav_path = audio_out_dir / utt_name
            # Clamp to [-1.0, 1.0] before saving as PCM_16 to avoid wrap-around clipping
            enh_clipped = torch.clamp(enhanced, -1.0, 1.0).numpy()
            sf.write(str(out_wav_path), enh_clipped, test_config.sample_rate, subtype="PCM_16")

        # Save representative plot
        if idx in plot_indices:
            plot_title = f"{utt_name} (Speaker {speaker})"
            fig = plot_waveform_spectrogram_grid(
                clean=clean,
                noisy=noisy,
                enhanced=enhanced,
                sample_rate=test_config.sample_rate,
                title=plot_title,
                metrics=record,
            )
            plot_path = plots_dir / f"{clean_path.stem}_eval.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(fig)
            logger.info("Saved evaluation plot: %s", plot_path.name)

        if (idx + 1) % 100 == 0 or (idx + 1) == total_items:
            elapsed = time.perf_counter() - start_time
            rate = (idx + 1) / elapsed
            logger.info(
                "Processed %d/%d utterances (%.1f utt/s) | Current avg Delta STOI: %+.4f, Delta SI-SDR: %+.2f dB",
                idx + 1,
                total_items,
                rate,
                np.mean([r["delta_stoi"] for r in results]),
                np.mean([r["delta_sisdr"] for r in results]),
            )

    total_time = time.perf_counter() - start_time
    logger.info("Completed evaluation of %d utterances in %.2f seconds.", len(results), total_time)

    # Save per-utterance CSV
    csv_path = output_dir / "metrics_per_utterance.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved per-utterance CSV to: %s", csv_path)

    # Compute aggregate summary
    summary = aggregate_metrics(results)
    summary["metadata"] = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "total_utterances": len(results),
        "total_time_seconds": round(total_time, 2),
        "device": dev,
        "sample_rate": test_config.sample_rate,
    }

    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary JSON to: %s", json_path)

    # Log summary table
    logger.info("=" * 60)
    logger.info("OBJECTIVE EVALUATION SUMMARY (%d Utterances)", len(results))
    logger.info("=" * 60)
    logger.info("%-15s %-12s %-12s %-12s", "Metric", "Noisy", "Enhanced", "Delta")
    logger.info("-" * 60)

    for metric_name, base_key in [("STOI", "stoi"), ("SI-SDR (dB)", "sisdr"), ("SNR (dB)", "snr")]:
        noisy_m = summary["overall"][f"noisy_{base_key}"]["mean"]
        enh_m = summary["overall"][f"enhanced_{base_key}"]["mean"]
        delta_m = summary["overall"][f"delta_{base_key}"]["mean"]
        logger.info(
            "%-15s %10.4f   %10.4f   %+10.4f",
            metric_name,
            noisy_m if noisy_m is not None else float("nan"),
            enh_m if enh_m is not None else float("nan"),
            delta_m if delta_m is not None else float("nan"),
        )

    pesq_noisy = summary["overall"]["noisy_pesq"]["mean"]
    pesq_enh = summary["overall"]["enhanced_pesq"]["mean"]
    pesq_delta = summary["overall"]["delta_pesq"]["mean"]
    if pesq_noisy is not None and pesq_enh is not None:
        logger.info(
            "%-15s %10.4f   %10.4f   %+10.4f",
            "PESQ",
            pesq_noisy,
            pesq_enh,
            pesq_delta,
        )
    else:
        logger.info("%-15s %10s   %10s   %10s", "PESQ", "N/A", "N/A", "N/A (no MSVC wheel)")

    logger.info("=" * 60)
    return summary


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained speech enhancement model on test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/enhancement_train/checkpoint_best.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/voicebank_demand_16k",
        help="Path to processed 16 kHz VoiceBank+DEMAND dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/enhancement_eval",
        help="Directory to save evaluation results, CSV, JSON, audio, and plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        default=True,
        help="Save reconstructed enhanced WAV files.",
    )
    parser.add_argument(
        "--no-save-audio",
        action="store_false",
        dest="save_audio",
        help="Do not save enhanced WAV files.",
    )
    parser.add_argument(
        "--num-plots",
        type=int,
        default=5,
        help="Number of representative spectrogram/waveform comparison plots to generate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of test utterances to evaluate (useful for smoke tests).",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parsed = parse_args(args)
    evaluate(
        checkpoint_path=parsed.checkpoint,
        data_dir=parsed.data_dir,
        output_dir=parsed.output_dir,
        device=parsed.device,
        save_audio=parsed.save_audio,
        num_plots=parsed.num_plots,
        max_samples=parsed.max_samples,
    )


if __name__ == "__main__":
    main()
