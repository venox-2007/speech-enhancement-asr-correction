"""VoiceBank+DEMAND dataset for speech enhancement.

Provides a PyTorch ``Dataset`` that loads paired clean/noisy utterances
from the standard VoiceBank+DEMAND directory layout.  The train/val split
is **speaker-based** so there is no data leakage between partitions.

Expected directory structure (Edinburgh DataShare layout)::

    <root>/
      clean_trainset_28spk_wav/
      noisy_trainset_28spk_wav/
      clean_testset_wav/
      noisy_testset_wav/

All audio is loaded at 16 kHz mono.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.audio_io import load_audio
from src.preprocessing.segments import paired_random_crop

logger = logging.getLogger(__name__)

# Regex to extract speaker ID from VoiceBank filenames: p226_001.wav → p226
_SPEAKER_RE = re.compile(r"^(p\d+)_")

SAMPLE_RATE = 16_000
DEFAULT_SEGMENT_LENGTH = 16384  # ~1.02 s at 16 kHz


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for VoiceBank+DEMAND dataset.

    Parameters
    ----------
    root : str
        Path to the dataset root directory.
    clean_train_dir : str
        Name of the clean training utterances subdirectory.
    noisy_train_dir : str
        Name of the noisy training utterances subdirectory.
    clean_test_dir : str
        Name of the clean test utterances subdirectory.
    noisy_test_dir : str
        Name of the noisy test utterances subdirectory.
    sample_rate : int
        Target sample rate for all audio.
    segment_length : int or None
        Fixed segment length for training.  If *None*, full utterances are
        returned (suitable for evaluation).
    val_speakers : int or list[str]
        Number of training speakers to hold out for validation (int), or
        an explicit list of speaker IDs.  Default: 2 (last two in sorted
        order).
    seed : int
        Seed is currently unused but reserved for future stochastic
        augmentation.
    """

    root: str = ""
    clean_train_dir: str = "clean_trainset_28spk_wav"
    noisy_train_dir: str = "noisy_trainset_28spk_wav"
    clean_test_dir: str = "clean_testset_wav"
    noisy_test_dir: str = "noisy_testset_wav"
    sample_rate: int = SAMPLE_RATE
    segment_length: int | None = DEFAULT_SEGMENT_LENGTH
    val_speakers: int | list[str] = 2
    seed: int = 42


def _extract_speaker(filename: str) -> str | None:
    """Extract the speaker ID from a VoiceBank filename."""
    m = _SPEAKER_RE.match(filename)
    return m.group(1) if m else None


def _discover_pairs(
    clean_dir: Path,
    noisy_dir: Path,
) -> list[tuple[Path, Path]]:
    """Find matching clean/noisy WAV pairs by filename."""
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Clean directory not found: {clean_dir}")
    if not noisy_dir.is_dir():
        raise FileNotFoundError(f"Noisy directory not found: {noisy_dir}")

    clean_files = {p.name: p for p in sorted(clean_dir.glob("*.wav"))}
    pairs = []
    for name, clean_path in sorted(clean_files.items()):
        noisy_path = noisy_dir / name
        if noisy_path.is_file():
            pairs.append((clean_path, noisy_path))
        else:
            logger.warning("No noisy match for %s", name)
    return pairs


def _split_speakers(
    pairs: list[tuple[Path, Path]],
    val_speakers: int | list[str],
) -> tuple[list[str], list[str]]:
    """Determine train and validation speaker sets.

    Returns
    -------
    train_speakers, val_speakers_list
    """
    all_speakers = sorted(
        {_extract_speaker(p[0].name) for p in pairs} - {None}
    )

    if isinstance(val_speakers, list):
        val_set = set(val_speakers)
        unknown = val_set - set(all_speakers)
        if unknown:
            raise ValueError(f"Unknown validation speaker(s): {unknown}")
        train_spk = [s for s in all_speakers if s not in val_set]
        val_spk = [s for s in all_speakers if s in val_set]
    else:
        n = val_speakers
        if n < 0 or n >= len(all_speakers):
            raise ValueError(
                f"val_speakers={n} is out of range for {len(all_speakers)} speakers"
            )
        train_spk = all_speakers[: len(all_speakers) - n]
        val_spk = all_speakers[len(all_speakers) - n :] if n > 0 else []

    return train_spk, val_spk


def _filter_by_speakers(
    pairs: list[tuple[Path, Path]],
    speakers: set[str],
) -> list[tuple[Path, Path]]:
    """Keep only pairs whose speaker ID is in *speakers*."""
    return [
        (c, n)
        for c, n in pairs
        if _extract_speaker(c.name) in speakers
    ]


class VoiceBankDEMANDDataset(Dataset):
    """PyTorch Dataset for VoiceBank+DEMAND clean/noisy pairs.

    Parameters
    ----------
    config : DatasetConfig
        Dataset configuration.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.

    The dataset returns ``(noisy, clean)`` tensors.  During training with
    a ``segment_length``, a random crop is applied (same offset for both
    clean and noisy).  During evaluation the full utterance is returned.
    """

    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        self.config = config
        self.split = split
        self.root = Path(config.root)
        self._pairs: list[tuple[Path, Path]] = []
        self._speakers: list[str] = []

        if split == "test":
            self._pairs = _discover_pairs(
                self.root / config.clean_test_dir,
                self.root / config.noisy_test_dir,
            )
            self._speakers = sorted(
                {_extract_speaker(p[0].name) for p in self._pairs} - {None}
            )
        else:
            all_train_pairs = _discover_pairs(
                self.root / config.clean_train_dir,
                self.root / config.noisy_train_dir,
            )
            train_spk, val_spk = _split_speakers(
                all_train_pairs, config.val_speakers
            )
            if split == "train":
                self._speakers = train_spk
                self._pairs = _filter_by_speakers(
                    all_train_pairs, set(train_spk)
                )
            else:  # val
                self._speakers = val_spk
                self._pairs = _filter_by_speakers(
                    all_train_pairs, set(val_spk)
                )

        logger.info(
            "%s split: %d utterances from %d speakers",
            split,
            len(self._pairs),
            len(self._speakers),
        )

    @property
    def pairs(self) -> list[tuple[Path, Path]]:
        """List of (clean_path, noisy_path) pairs."""
        return list(self._pairs)

    @property
    def speakers(self) -> list[str]:
        """Sorted list of speaker IDs in this split."""
        return list(self._speakers)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(noisy, clean)`` float32 tensors.

        During training (with ``segment_length`` set), a random crop is
        applied.  During evaluation the full utterance is returned.
        """
        clean_path, noisy_path = self._pairs[idx]

        clean = torch.from_numpy(
            load_audio(clean_path, target_sr=self.config.sample_rate)
        )
        noisy = torch.from_numpy(
            load_audio(noisy_path, target_sr=self.config.sample_rate)
        )

        # Ensure same length (some datasets have tiny length mismatches)
        min_len = min(clean.shape[0], noisy.shape[0])
        clean = clean[:min_len]
        noisy = noisy[:min_len]

        # Apply segment cropping for training
        seg_len = self.config.segment_length
        if seg_len is not None and self.split == "train":
            clean, noisy = paired_random_crop(clean, noisy, seg_len)

        return noisy, clean


def _collate_variable_length(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length utterances by zero-padding to the longest.

    Returns ``(noisy_batch, clean_batch, lengths)`` where *lengths* stores
    the original sample count of each utterance.
    """
    noisy_list, clean_list = zip(*batch)
    lengths = torch.tensor([n.shape[0] for n in noisy_list])
    max_len = lengths.max().item()

    noisy_batch = torch.zeros(len(batch), max_len)
    clean_batch = torch.zeros(len(batch), max_len)
    for i, (n, c) in enumerate(zip(noisy_list, clean_list)):
        noisy_batch[i, : n.shape[0]] = n
        clean_batch[i, : c.shape[0]] = c

    return noisy_batch, clean_batch, lengths


def create_dataloader(
    dataset: VoiceBankDEMANDDataset,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool | None = None,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with appropriate collation.

    For training (fixed segment length), standard batching is used.
    For evaluation (variable length), ``_collate_variable_length`` is
    used so that utterances are padded to the longest in the batch.

    Parameters
    ----------
    dataset : VoiceBankDEMANDDataset
        The dataset to wrap.
    batch_size : int
        Batch size.
    num_workers : int
        Number of data-loading worker processes.
    shuffle : bool or None
        Shuffle the data.  Defaults to *True* for training, *False*
        otherwise.
    pin_memory : bool
        Pin memory for GPU transfer.
    """
    if shuffle is None:
        shuffle = dataset.split == "train"

    is_variable = (
        dataset.config.segment_length is None or dataset.split != "train"
    )
    collate_fn = _collate_variable_length if is_variable else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
