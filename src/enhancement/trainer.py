"""Training pipeline for U-Net speech enhancement model.

Includes reproducible train/val loops, AdamW, ReduceLROnPlateau, gradient clipping,
mixed precision (AMP), checkpointing (latest & best), early stopping, and history logging.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.enhancement.dataset import (
    DEFAULT_VAL_SPEAKERS,
    DatasetConfig,
    VoiceBankDEMANDDataset,
    create_dataloader,
)
from src.enhancement.loss import LossType, SpectralLoss
from src.enhancement.model import EnhancementModelConfig, WaveformUNetEnhancer
from src.enhancement.unet import UNetConfig
from src.preprocessing.stft import STFTConfig, stft
from src.utils.device import resolve_device

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for speech enhancement training."""

    # Data
    data_dir: str = "data/processed/voicebank_demand_16k"
    val_speakers: list[str] = field(default_factory=lambda: list(DEFAULT_VAL_SPEAKERS))
    batch_size: int = 16
    val_batch_size: int = 1
    segment_length: int = 16384
    sample_rate: int = 16000
    num_workers: int = 0

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 50
    grad_clip: float = 5.0
    use_amp: bool = True
    early_stopping_patience: int = 10
    lr_scheduler: str = "reduce_on_plateau"  # 'reduce_on_plateau', 'cosine', 'none'
    lr_patience: int = 5
    lr_factor: float = 0.5
    loss_type: LossType = "l1"
    mag_power: float = 1.0

    # Environment & Output
    seed: int = 42
    device: str = "auto"
    output_dir: str = "results/enhancement_train"
    save_every: int = 1
    max_batches_per_epoch: int | None = None
    max_val_batches: int | None = None

    # Model
    unet: UNetConfig = field(default_factory=UNetConfig)
    stft: STFTConfig = field(default_factory=STFTConfig)


class Trainer:
    """Orchestrates training, validation, checkpointing, and logging for U-Net."""

    def __init__(
        self,
        config: TrainingConfig,
        model: WaveformUNetEnhancer | None = None,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
    ) -> None:
        self.config = config
        set_seed(self.config.seed)

        # Device setup
        resolved_dev = resolve_device(self.config.device)
        self.device = torch.device(resolved_dev)
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        logger.info("Using device: %s (AMP: %s)", self.device, self.use_amp)

        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model setup
        if model is None:
            model_cfg = EnhancementModelConfig(
                stft=self.config.stft,
                unet=self.config.unet,
                sample_rate=self.config.sample_rate,
            )
            self.model = WaveformUNetEnhancer(model_cfg).to(self.device)
        else:
            self.model = model.to(self.device)

        self.stft_config = self.model.stft_config

        # Loss function
        self.criterion = SpectralLoss(
            loss_type=self.config.loss_type,
            mag_power=self.config.mag_power,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # LR Scheduler
        if self.config.lr_scheduler == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.config.lr_patience,
                factor=self.config.lr_factor,
            )
        elif self.config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
            )
        else:
            self.scheduler = None

        # Mixed precision scaler (using new PyTorch 2.x API torch.amp)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # DataLoaders
        if train_loader is not None and val_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader
        else:
            self.train_loader, self.val_loader = self._create_dataloaders()

        # State tracking
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history: list[dict[str, Any]] = []

    def _create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Create train and validation DataLoaders using existing pipeline."""
        train_ds_cfg = DatasetConfig(
            root=self.config.data_dir,
            sample_rate=self.config.sample_rate,
            segment_length=self.config.segment_length,
            val_speakers=self.config.val_speakers,
            seed=self.config.seed,
        )
        val_ds_cfg = DatasetConfig(
            root=self.config.data_dir,
            sample_rate=self.config.sample_rate,
            segment_length=None,  # full utterance evaluation
            val_speakers=self.config.val_speakers,
            seed=self.config.seed,
        )

        train_ds = VoiceBankDEMANDDataset(train_ds_cfg, split="train")
        val_ds = VoiceBankDEMANDDataset(val_ds_cfg, split="val")

        train_loader = create_dataloader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        val_loader = create_dataloader(
            val_ds,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        return train_loader, val_loader

    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if (
                self.config.max_batches_per_epoch is not None
                and batch_idx >= self.config.max_batches_per_epoch
            ):
                break

            noisy = batch.noisy.to(self.device)
            clean = batch.clean.to(self.device)

            self.optimizer.zero_grad()

            with torch.no_grad():
                noisy_mag, _ = stft(noisy, self.stft_config)
                clean_mag, _ = stft(clean, self.stft_config)
                noisy_mag_4d = noisy_mag.unsqueeze(1)
                clean_mag_4d = clean_mag.unsqueeze(1)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred_mag_4d = self.model.unet(noisy_mag_4d)
                loss = self.criterion(pred_mag_4d, clean_mag_4d)

            self.scaler.scale(loss).backward()

            if self.config.grad_clip > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation loop and compute spectral metrics."""
        self.model.eval()
        total_loss = 0.0
        total_l1 = 0.0
        total_mse = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.val_loader):
            if (
                self.config.max_val_batches is not None
                and batch_idx >= self.config.max_val_batches
            ):
                break

            noisy = batch.noisy.to(self.device)
            clean = batch.clean.to(self.device)

            noisy_mag, _ = stft(noisy, self.stft_config)
            clean_mag, _ = stft(clean, self.stft_config)
            noisy_mag_4d = noisy_mag.unsqueeze(1)
            clean_mag_4d = clean_mag.unsqueeze(1)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred_mag_4d = self.model.unet(noisy_mag_4d)
                loss = self.criterion(pred_mag_4d, clean_mag_4d)

            l1_err = torch.abs(pred_mag_4d - clean_mag_4d).mean().item()
            mse_err = torch.square(pred_mag_4d - clean_mag_4d).mean().item()

            total_loss += loss.item()
            total_l1 += l1_err
            total_mse += mse_err
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "val_loss": total_loss / n,
            "val_l1": total_l1 / n,
            "val_mse": total_mse / n,
        }

    def fit(self) -> dict[str, Any]:
        """Execute full training loop with early stopping and checkpointing."""
        logger.info(
            "Starting training for up to %d epochs (patience=%d)",
            self.config.max_epochs,
            self.config.early_stopping_patience,
        )

        start_time = time.time()

        for epoch in range(self.current_epoch + 1, self.config.max_epochs + 1):
            self.current_epoch = epoch
            t0 = time.time()

            train_loss = self.train_epoch()
            val_metrics = self.validate()
            val_loss = val_metrics["val_loss"]

            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - t0

            # Log progress
            logger.info(
                "Epoch %3d/%3d | Train Loss: %.4f | Val Loss: %.4f (L1: %.4f, MSE: %.4f) | LR: %.2e | Time: %.1fs",
                epoch,
                self.config.max_epochs,
                train_loss,
                val_loss,
                val_metrics["val_l1"],
                val_metrics["val_mse"],
                current_lr,
                epoch_time,
            )

            # Record history
            entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_l1": val_metrics["val_l1"],
                "val_mse": val_metrics["val_mse"],
                "lr": current_lr,
                "epoch_time_sec": round(epoch_time, 2),
            }
            self.history.append(entry)
            self._save_history()

            # Checkpoint: best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(self.output_dir / "checkpoint_best.pt", is_best=True)
                logger.info("New best checkpoint saved (Val Loss: %.4f)", val_loss)
            else:
                self.epochs_without_improvement += 1

            # Checkpoint: latest
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(self.output_dir / "checkpoint_latest.pt")

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.config.early_stopping_patience,
                )
                break

        total_time = time.time() - start_time
        logger.info("Training completed in %.1f seconds.", total_time)
        return {
            "total_epochs": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "total_time_sec": total_time,
            "history": self.history,
        }

    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save model checkpoint with full optimizer and scheduler states."""
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "config": asdict(self.config),
            "history": self.history,
            "is_best": is_best,
        }
        torch.save(checkpoint, str(path))
        logger.debug("Saved checkpoint to %s", path)

    def resume_checkpoint(self, path: Path) -> None:
        """Resume model, optimizer, scheduler, and training progress from checkpoint."""
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        self.history = checkpoint.get("history", [])

        logger.info(
            "Resumed from %s at epoch %d (best val loss: %.4f)",
            path,
            self.current_epoch,
            self.best_val_loss,
        )

    def _save_history(self) -> None:
        """Write training history to history.json and history.csv."""
        json_path = self.output_dir / "history.json"
        csv_path = self.output_dir / "history.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        if self.history:
            fieldnames = list(self.history[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.history)
