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
from typing import Any, Iterable, NamedTuple, Sequence

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

# Default held-out validation speakers (1 female: p286, 1 male: p287)
DEFAULT_VAL_SPEAKERS = ["p286", "p287"]


class EnhancementBatch(NamedTuple):
    """Standardized batch output for speech enhancement DataLoaders.

    Attributes
    ----------
    noisy : Tensor
        Batch of noisy speech waveforms ``(B, T)``.
    clean : Tensor
        Batch of clean speech waveforms ``(B, T)``.
    lengths : Tensor
        1-D int64 tensor ``(B,)`` with valid audio lengths for each utterance
        (equal to ``T`` if fixed segment length or after zero-padding).
    """

    noisy: torch.Tensor
    clean: torch.Tensor
    lengths: torch.Tensor


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
        an explicit list of speaker IDs (e.g. ``["p286", "p287"]``).
        Default: ``["p286", "p287"]`` (1 female, 1 male).
    seed : int
        Seed for reproducible random crop offsets and data shuffling.
    """

    root: str = ""
    clean_train_dir: str = "clean_trainset_28spk_wav"
    noisy_train_dir: str = "noisy_trainset_28spk_wav"
    clean_test_dir: str = "clean_testset_wav"
    noisy_test_dir: str = "noisy_testset_wav"
    sample_rate: int = SAMPLE_RATE
    segment_length: int | None = DEFAULT_SEGMENT_LENGTH
    val_speakers: int | list[str] = field(
        default_factory=lambda: list(DEFAULT_VAL_SPEAKERS)
    )
    seed: int = 42


def _extract_speaker(filename: str) -> str | None:
    """Extract the speaker ID from a VoiceBank filename."""
    m = _SPEAKER_RE.match(filename)
    return m.group(1) if m else None


def _speaker_sort_key(spk: str) -> tuple[int, int | str]:
    """Sort speakers by numeric ID if formatted as p<digits>, else fallback."""
    m = re.match(r"^p(\d+)$", spk)
    if m:
        return (0, int(m.group(1)))
    return (1, spk)


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
        {_extract_speaker(p[0].name) for p in pairs} - {None},
        key=_speaker_sort_key,
    )

    if isinstance(val_speakers, (list, tuple, set)):
        val_set = set(val_speakers)
        unknown = val_set - set(all_speakers)
        if unknown:
            raise ValueError(f"Unknown validation speaker(s): {sorted(unknown)}")
        train_spk = [s for s in all_speakers if s not in val_set]
        val_spk = [s for s in all_speakers if s in val_set]
    else:
        n = int(val_speakers)
        if n < 0 or n > len(all_speakers):
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
            if not self._pairs:
                raise RuntimeError(
                    f"No valid clean/noisy audio pairs found in '{self.root}' for split '{split}'."
                )
            self._speakers = sorted(
                {_extract_speaker(p[0].name) for p in self._pairs} - {None},
                key=_speaker_sort_key,
            )
        else:
            all_train_pairs = _discover_pairs(
                self.root / config.clean_train_dir,
                self.root / config.noisy_train_dir,
            )
            if not all_train_pairs:
                raise RuntimeError(
                    f"No valid clean/noisy audio pairs found in '{self.root}' for split '{split}'."
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

        if not self._pairs:
            if split == "val" and (
                (isinstance(config.val_speakers, int) and config.val_speakers == 0)
                or (isinstance(config.val_speakers, (list, tuple, set)) and len(config.val_speakers) == 0)
            ):
                raise RuntimeError(
                    "No validation pairs found because val_speakers is 0 or empty. "
                    "Set val_speakers > 0 or pass explicit speaker IDs."
                )
            raise RuntimeError(
                f"No valid clean/noisy audio pairs found in '{self.root}' for split '{split}'."
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

    def _get_generator(self) -> torch.Generator:
        """Worker-safe seeded generator for reproducible crops."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            wid = worker_info.id
            if not hasattr(self, "_worker_generators"):
                self._worker_generators = {}
            if wid not in self._worker_generators:
                g = torch.Generator()
                g.manual_seed(self.config.seed + (wid + 1) * 1000)
                self._worker_generators[wid] = g
            return self._worker_generators[wid]

        if not hasattr(self, "_generator") or self._generator is None:
            self._generator = torch.Generator()
            self._generator.manual_seed(self.config.seed)
        return self._generator

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(noisy, clean)`` float32 tensors.

        During training (with ``segment_length`` set), a reproducible random
        crop is applied.  During evaluation the full utterance is returned.
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
            clean, noisy = paired_random_crop(
                clean, noisy, seg_len, generator=self._get_generator()
            )

        return noisy, clean


def collate_enhancement_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> EnhancementBatch:
    """Collate utterances into a standardized EnhancementBatch.

    Pads variable-length utterances to the maximum length in the batch.
    For fixed-length batches, stacks directly.

    Returns
    -------
    EnhancementBatch
        NamedTuple with ``(noisy, clean, lengths)``.
    """
    noisy_list, clean_list = zip(*batch)
    lengths = torch.tensor([n.shape[0] for n in noisy_list], dtype=torch.long)
    max_len = lengths.max().item()

    # If all items already have identical length (fixed segment training), stack directly
    if all(n.shape[0] == max_len for n in noisy_list):
        noisy_batch = torch.stack(noisy_list)
        clean_batch = torch.stack(clean_list)
    else:
        noisy_batch = torch.zeros(len(batch), max_len, dtype=noisy_list[0].dtype)
        clean_batch = torch.zeros(len(batch), max_len, dtype=clean_list[0].dtype)
        for i, (n, c) in enumerate(zip(noisy_list, clean_list)):
            noisy_batch[i, : n.shape[0]] = n
            clean_batch[i, : c.shape[0]] = c

    return EnhancementBatch(noisy=noisy_batch, clean=clean_batch, lengths=lengths)


def seed_worker(worker_id: int) -> None:
    """Seed worker process for DataLoader multiprocessing reproducibility."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def create_dataloader(
    dataset: VoiceBankDEMANDDataset,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool | None = None,
    pin_memory: bool = True,
    worker_init_fn: Any = seed_worker,
) -> DataLoader:
    """Create a DataLoader with standardized EnhancementBatch collation.

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
    worker_init_fn : callable
        Worker initialization function for reproducibility.

    Returns
    -------
    DataLoader
        Yields ``EnhancementBatch(noisy, clean, lengths)`` across all splits.
    """
    if shuffle is None:
        shuffle = dataset.split == "train"

    generator = torch.Generator()
    generator.manual_seed(dataset.config.seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_enhancement_batch,
        worker_init_fn=worker_init_fn,
        generator=generator if shuffle else None,
    )
