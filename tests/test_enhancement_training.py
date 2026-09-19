"""Synthetic tests for speech enhancement training pipeline.

Tests loss functions, training step execution, checkpoint round-trip,
reproducibility/determinism, CLI argument parsing, and early stopping.
No real dataset required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.enhancement.dataset import EnhancementBatch
from src.enhancement.loss import SpectralLoss
from src.enhancement.model import EnhancementModelConfig, WaveformUNetEnhancer
from src.enhancement.train import build_parser
from src.enhancement.trainer import Trainer, TrainingConfig
from src.enhancement.unet import UNetConfig
from src.preprocessing.stft import STFTConfig


# ==============================================================================
# Synthetic Dataset for Testing
# ==============================================================================


class SyntheticEnhancementDataset(Dataset):
    """Generates synthetic paired waveforms for unit testing."""

    def __init__(self, count: int = 8, length: int = 4096, seed: int = 42) -> None:
        g = torch.Generator().manual_seed(seed)
        self.clean = torch.randn(count, length, generator=g)
        self.noisy = self.clean + 0.3 * torch.randn(count, length, generator=g)
        self.lengths = torch.full((count,), length, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.noisy[idx], self.clean[idx]


def synthetic_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> EnhancementBatch:
    noisy_list, clean_list = zip(*batch)
    lengths = torch.tensor([n.shape[0] for n in noisy_list], dtype=torch.long)
    return EnhancementBatch(
        noisy=torch.stack(noisy_list),
        clean=torch.stack(clean_list),
        lengths=lengths,
    )


def make_synthetic_loaders(
    batch_size: int = 4,
    length: int = 4096,
) -> tuple[DataLoader, DataLoader]:
    train_ds = SyntheticEnhancementDataset(count=8, length=length, seed=42)
    val_ds = SyntheticEnhancementDataset(count=4, length=length, seed=99)
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=synthetic_collate)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=synthetic_collate)
    return train_loader, val_loader


# ==============================================================================
# SpectralLoss Tests
# ==============================================================================


class TestSpectralLoss:
    def test_loss_l1(self):
        loss_fn = SpectralLoss(loss_type="l1")
        pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        target = torch.tensor([[[[1.5, 2.0], [2.0, 5.0]]]])
        # abs diff: 0.5, 0.0, 1.0, 1.0 -> mean = 2.5 / 4 = 0.625
        loss = loss_fn(pred, target)
        assert torch.isclose(loss, torch.tensor(0.625))

    def test_loss_mse(self):
        loss_fn = SpectralLoss(loss_type="mse")
        pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        target = torch.tensor([[[[2.0, 2.0], [1.0, 6.0]]]])
        # sq diff: 1.0, 0.0, 4.0, 4.0 -> mean = 9.0 / 4 = 2.25
        loss = loss_fn(pred, target)
        assert torch.isclose(loss, torch.tensor(2.25))

    def test_loss_smooth_l1(self):
        loss_fn = SpectralLoss(loss_type="smooth_l1")
        pred = torch.randn(2, 1, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert loss.item() > 0.0

    def test_loss_power_compression(self):
        loss_fn = SpectralLoss(loss_type="l1", mag_power=0.5)
        pred = torch.tensor([[[[4.0, 9.0]]]])
        target = torch.tensor([[[[1.0, 4.0]]]])
        # sqrt: [2.0, 3.0] vs [1.0, 2.0] -> diffs: [1.0, 1.0] -> mean = 1.0
        loss = loss_fn(pred, target)
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)

    def test_frame_masking(self):
        """Padded frames outside frame_lengths must not affect loss."""
        loss_fn = SpectralLoss(loss_type="l1")
        # 2 utterances, 3 frames. Utterance 1 valid for 2 frames, utterance 2 valid for 1 frame.
        pred = torch.zeros(2, 1, 4, 3)
        target = torch.ones(2, 1, 4, 3)
        # Put giant error in padded region (frame index 2 for utt 0, frame indices 1, 2 for utt 1)
        pred[0, :, :, 2] = 1000.0
        pred[1, :, :, 1:] = 1000.0

        lengths = torch.tensor([2, 1])
        loss = loss_fn(pred, target, frame_lengths=lengths)
        # In valid region, diff is |0 - 1| = 1.0
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="Unsupported loss_type"):
            SpectralLoss(loss_type="cosine")
        with pytest.raises(ValueError, match="mag_power must be positive"):
            SpectralLoss(mag_power=0.0)

    def test_shape_mismatch(self):
        loss_fn = SpectralLoss()
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(torch.zeros(2, 16), torch.zeros(2, 32))


# ==============================================================================
# Trainer & Execution Tests
# ==============================================================================


class TestTrainerExecution:
    def _create_mini_trainer(self, tmp_path: Path, seed: int = 42) -> Trainer:
        train_loader, val_loader = make_synthetic_loaders(batch_size=4, length=4096)
        stft_cfg = STFTConfig(n_fft=256, hop_length=128, win_length=256)
        unet_cfg = UNetConfig(channels=(8, 16, 32))
        config = TrainingConfig(
            output_dir=str(tmp_path),
            max_epochs=2,
            batch_size=4,
            val_batch_size=1,
            segment_length=4096,
            device="cpu",
            use_amp=False,
            seed=seed,
            lr=1e-3,
            stft=stft_cfg,
            unet=unet_cfg,
        )
        return Trainer(config, model=None, train_loader=train_loader, val_loader=val_loader)

    def test_one_training_step_updates_weights(self, tmp_path):
        trainer = self._create_mini_trainer(tmp_path)
        # Record initial weights of first conv
        initial_weight = trainer.model.unet.inc.block[0].weight.clone()

        loss = trainer.train_epoch()
        assert loss > 0.0

        # Verify weights changed after training step
        updated_weight = trainer.model.unet.inc.block[0].weight
        assert not torch.equal(initial_weight, updated_weight)

    def test_validation_step(self, tmp_path):
        trainer = self._create_mini_trainer(tmp_path)
        metrics = trainer.validate()
        assert "val_loss" in metrics
        assert "val_l1" in metrics
        assert "val_mse" in metrics
        assert metrics["val_loss"] > 0.0

    def test_checkpoint_roundtrip(self, tmp_path):
        trainer = self._create_mini_trainer(tmp_path)
        trainer.current_epoch = 1
        trainer.best_val_loss = 0.42
        ckpt_path = tmp_path / "test_ckpt.pt"
        trainer.save_checkpoint(ckpt_path)

        # Fresh trainer
        new_trainer = self._create_mini_trainer(tmp_path, seed=12345)
        assert new_trainer.current_epoch == 0

        new_trainer.resume_checkpoint(ckpt_path)
        assert new_trainer.current_epoch == 1
        assert new_trainer.best_val_loss == 0.42

        # Verify model weights match exactly
        for (n1, p1), (n2, p2) in zip(
            trainer.model.named_parameters(), new_trainer.model.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_determinism(self, tmp_path):
        """Two trainers with identical seed and data must produce identical losses."""
        t1 = self._create_mini_trainer(tmp_path / "t1", seed=42)
        t2 = self._create_mini_trainer(tmp_path / "t2", seed=42)

        loss1 = t1.train_epoch()
        loss2 = t2.train_epoch()
        assert abs(loss1 - loss2) < 1e-6

        # Check weights are identical
        for (n1, p1), (n2, p2) in zip(
            t1.model.named_parameters(), t2.model.named_parameters()
        ):
            assert torch.allclose(p1, p2, atol=1e-6), f"Mismatch in {n1}"

    def test_early_stopping(self, tmp_path):
        """Trainer stops before max_epochs if validation loss does not improve."""
        trainer = self._create_mini_trainer(tmp_path)
        trainer.config.max_epochs = 10
        trainer.config.early_stopping_patience = 2

        # Mock validate to always return a constant worse loss
        trainer.best_val_loss = 0.01
        trainer.validate = lambda: {"val_loss": 1.0, "val_l1": 1.0, "val_mse": 1.0}

        result = trainer.fit()
        # Should stop after 2 epochs
        assert result["total_epochs"] == 2


# ==============================================================================
# CLI Parser Tests
# ==============================================================================


class TestCLIParser:
    def test_default_parser_arguments(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.epochs == 50
        assert args.batch_size == 16
        assert args.val_batch_size == 1
        assert args.lr == 1e-3
        assert args.loss == "l1"
        assert args.scheduler == "reduce_on_plateau"
        assert not args.no_amp
        assert not args.smoke_test

    def test_parser_custom_overrides(self):
        parser = build_parser()
        args = parser.parse_args([
            "--epochs", "25",
            "--batch-size", "8",
            "--lr", "5e-4",
            "--loss", "mse",
            "--no-amp",
            "--smoke-test",
            "--device", "cpu",
        ])
        assert args.epochs == 25
        assert args.batch_size == 8
        assert args.lr == 5e-4
        assert args.loss == "mse"
        assert args.no_amp
        assert args.smoke_test
        assert args.device == "cpu"
