"""Command-line interface for speech enhancement model training.

Usage:
    # Standard training:
    python -m src.enhancement.train --data-dir data/processed/voicebank_demand_16k --epochs 50

    # Short GPU smoke test (verifies 2 train batches, 2 val batches, checkpointing, and resume):
    python -m src.enhancement.train --smoke-test --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from src.enhancement.trainer import Trainer, TrainingConfig
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Train U-Net speech enhancement model on VoiceBank+DEMAND.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/voicebank_demand_16k",
        help="Path to 16 kHz processed dataset.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/enhancement_train",
        help="Directory to save checkpoints and metrics history.",
    )
    p.add_argument("--epochs", type=int, default=50, help="Maximum training epochs.")
    p.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    p.add_argument(
        "--val-batch-size",
        type=int,
        default=1,
        help="Validation batch size (1 safely handles variable-length utterances without padding).",
    )
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for AdamW.")
    p.add_argument(
        "--segment-length",
        type=int,
        default=16384,
        help="Training audio segment length in samples (~1.02 s at 16 kHz).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device.",
    )
    p.add_argument("--workers", type=int, default=0, help="DataLoader worker count.")
    p.add_argument(
        "--loss",
        type=str,
        default="l1",
        choices=["l1", "mse", "smooth_l1"],
        help="Spectral loss function.",
    )
    p.add_argument("--grad-clip", type=float, default=5.0, help="Max gradient norm clipping.")
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision on CUDA.",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default="reduce_on_plateau",
        choices=["reduce_on_plateau", "cosine", "none"],
        help="Learning rate scheduler.",
    )
    p.add_argument(
        "--lr-patience",
        type=int,
        default=2,
        help="Patience in epochs for ReduceLROnPlateau scheduler.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run short 1-epoch / 2-batch smoke test to verify pipeline, GPU, and checkpointing.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for speech enhancement training CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.smoke_test:
        logger.info("=== Running Speech Enhancement Training Smoke Test ===")
        config = TrainingConfig(
            data_dir=args.data_dir,
            output_dir=str(Path(args.output_dir) / "smoke_test"),
            max_epochs=2,
            batch_size=min(args.batch_size, 4),
            val_batch_size=1,
            segment_length=args.segment_length,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            num_workers=args.workers,
            loss_type=args.loss,
            use_amp=not args.no_amp,
            max_batches_per_epoch=2,
            max_val_batches=2,
        )
    else:
        config = TrainingConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            segment_length=args.segment_length,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            num_workers=args.workers,
            loss_type=args.loss,
            grad_clip=args.grad_clip,
            use_amp=not args.no_amp,
            lr_scheduler=args.scheduler,
            lr_patience=args.lr_patience,
            early_stopping_patience=args.patience,
        )

    trainer = Trainer(config)

    if args.resume:
        trainer.resume_checkpoint(Path(args.resume))

    result = trainer.fit()

    if args.smoke_test:
        # Verify checkpoint saving and reload capability
        ckpt_path = trainer.output_dir / "checkpoint_latest.pt"
        if not ckpt_path.is_file():
            logger.error("Smoke test failed: %s was not created.", ckpt_path)
            return 1
        logger.info("Verifying checkpoint resume from %s...", ckpt_path)
        trainer_resume = Trainer(config)
        trainer_resume.resume_checkpoint(ckpt_path)
        assert trainer_resume.current_epoch == 2
        logger.info("Smoke test passed successfully! GPU training, AMP, and resume verified.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
