"""Tests for the speech enhancement data pipeline.

All tests use **synthetic audio** (sine waves, white noise) written to
temporary directories.  The real VoiceBank+DEMAND dataset is NOT required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import soundfile as sf
import torch

from src.preprocessing.noise_mix import estimate_snr, mix_at_snr
from src.preprocessing.segments import pad_or_crop, paired_random_crop, random_crop
from src.preprocessing.stft import STFTConfig, db_to_mag, istft, mag_to_db, stft

# ============================================================ STFT tests


class TestSTFT:
    """STFT / iSTFT correctness."""

    def test_roundtrip_1d(self):
        """istft(stft(x)) ≈ x for a 1-D signal."""
        x = torch.randn(16000)
        cfg = STFTConfig()
        mag, phase = stft(x, cfg)
        y = istft(mag, phase, cfg, length=x.shape[0])
        assert y.shape == x.shape
        assert torch.allclose(x, y, atol=1e-5)

    def test_roundtrip_batched(self):
        """Batched STFT round-trip."""
        x = torch.randn(3, 8000)
        cfg = STFTConfig()
        mag, phase = stft(x, cfg)
        y = istft(mag, phase, cfg, length=x.shape[1])
        assert y.shape == x.shape
        assert torch.allclose(x, y, atol=1e-5)

    def test_output_shapes(self):
        """Magnitude/phase have expected (F, N) dimensions."""
        cfg = STFTConfig(n_fft=512, hop_length=256)
        x = torch.randn(16000)
        mag, phase = stft(x, cfg)

        assert mag.shape[0] == cfg.n_freq  # 257
        expected_frames = x.shape[0] // cfg.hop_length + 1
        # Allow ±1 frame due to padding
        assert abs(mag.shape[1] - expected_frames) <= 1
        assert mag.shape == phase.shape

    def test_batched_output_shapes(self):
        """Batched STFT has (B, F, N) shape."""
        cfg = STFTConfig()
        x = torch.randn(4, 8000)
        mag, phase = stft(x, cfg)
        assert mag.dim() == 3
        assert mag.shape[0] == 4
        assert mag.shape[1] == cfg.n_freq

    def test_config_defaults(self):
        cfg = STFTConfig()
        assert cfg.n_fft == 512
        assert cfg.hop_length == 256
        assert cfg.win_length == 512
        assert cfg.window == "hann"
        assert cfg.n_freq == 257

    def test_config_validation(self):
        with pytest.raises(ValueError):
            STFTConfig(n_fft=0)
        with pytest.raises(ValueError):
            STFTConfig(hop_length=0)
        with pytest.raises(ValueError):
            STFTConfig(win_length=1024, n_fft=512)
        with pytest.raises(ValueError):
            STFTConfig(window="invalid")

    def test_hamming_window(self):
        """Hamming window also works for STFT."""
        cfg = STFTConfig(window="hamming")
        x = torch.randn(8000)
        mag, phase = stft(x, cfg)
        y = istft(mag, phase, cfg, length=x.shape[0])
        assert torch.allclose(x, y, atol=1e-5)


class TestDBConversion:
    """Magnitude ↔ dB round-trip."""

    def test_roundtrip(self):
        mag = torch.rand(257, 63) + 0.01  # avoid near-zero
        db = mag_to_db(mag)
        recovered = db_to_mag(db)
        assert torch.allclose(mag, recovered, atol=1e-5)

    def test_zero_floor(self):
        """Near-zero magnitudes are floored, not -inf."""
        mag = torch.tensor([0.0, 1e-12, 1.0])
        db = mag_to_db(mag)
        assert torch.isfinite(db).all()


# ======================================================= segment tests


class TestSegments:
    """Fixed-length segment extraction."""

    def test_pad_short(self):
        """Short audio is zero-padded on the right."""
        x = torch.ones(100)
        y = pad_or_crop(x, 200)
        assert y.shape == (200,)
        assert y[:100].sum() == 100
        assert y[100:].sum() == 0

    def test_crop_long(self):
        """Long audio is cropped from the start."""
        x = torch.arange(200, dtype=torch.float32)
        y = pad_or_crop(x, 100)
        assert y.shape == (100,)
        assert (y == x[:100]).all()

    def test_exact_length(self):
        """Audio exactly the right length is returned unchanged."""
        x = torch.randn(100)
        y = pad_or_crop(x, 100)
        assert (y == x).all()

    def test_random_crop_shorter_than_length(self):
        """Random crop of a signal shorter than length → padded."""
        x = torch.ones(50)
        y = random_crop(x, 100)
        assert y.shape == (100,)
        assert y[:50].sum() == 50

    def test_random_crop_reproducible(self):
        """Same generator seed → same crop."""
        x = torch.randn(10000)
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        y1 = random_crop(x, 1000, g1)
        y2 = random_crop(x, 1000, g2)
        assert (y1 == y2).all()

    def test_random_crop_different_seeds(self):
        """Different seeds → (very likely) different crops."""
        x = torch.randn(10000)
        g1 = torch.Generator().manual_seed(1)
        g2 = torch.Generator().manual_seed(2)
        y1 = random_crop(x, 1000, g1)
        y2 = random_crop(x, 1000, g2)
        assert not (y1 == y2).all()

    def test_paired_crop_alignment(self):
        """Paired crop uses the same offset for clean and noisy."""
        clean = torch.arange(10000, dtype=torch.float32)
        noisy = torch.arange(10000, dtype=torch.float32) + 50000
        g = torch.Generator().manual_seed(42)
        c, n = paired_random_crop(clean, noisy, 1000, g)
        # The difference should be constant (50000) everywhere
        assert torch.allclose(n - c, torch.full((1000,), 50000.0))

    def test_paired_crop_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            paired_random_crop(torch.randn(100), torch.randn(200), 50)

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            pad_or_crop(torch.randn(10), 0)
        with pytest.raises(ValueError):
            random_crop(torch.randn(10), -1)


# ===================================================== noise mix tests


class TestNoiseMix:
    """Noise mixing at target SNR."""

    def test_snr_accuracy(self):
        """Mixed signal has approximately the requested SNR."""
        torch.manual_seed(0)
        clean = torch.randn(16000) * 0.5
        noise = torch.randn(16000)

        for target_snr in [0.0, 5.0, 10.0, 20.0, -5.0]:
            noisy = mix_at_snr(clean, noise, target_snr)
            measured = estimate_snr(clean, noisy)
            assert abs(measured - target_snr) < 0.5, (
                f"Target SNR {target_snr}, measured {measured:.2f}"
            )

    def test_silent_noise(self):
        """Silent noise returns clean unchanged."""
        clean = torch.randn(1000)
        noise = torch.zeros(1000)
        result = mix_at_snr(clean, noise, 10.0)
        assert torch.allclose(result, clean)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mix_at_snr(torch.randn(100), torch.randn(200), 10.0)

    def test_estimate_snr_clean(self):
        """Clean signal (no noise) → infinite SNR."""
        clean = torch.randn(1000)
        snr = estimate_snr(clean, clean)
        assert snr == float("inf")


# ==================================================== dataset tests


def _create_mock_dataset(root, speakers_train, speakers_test, n_per_speaker=3, sr=16000):
    """Create a mock VoiceBank+DEMAND directory tree with synthetic WAVs.

    Returns the root Path.
    """
    dirs = {
        "clean_trainset_28spk_wav": speakers_train,
        "noisy_trainset_28spk_wav": speakers_train,
        "clean_testset_wav": speakers_test,
        "noisy_testset_wav": speakers_test,
    }

    for dirname, speakers in dirs.items():
        d = root / dirname
        d.mkdir(parents=True, exist_ok=True)
        for spk in speakers:
            for i in range(n_per_speaker):
                fname = f"{spk}_{i + 1:03d}.wav"
                # Short sine wave (different freq per file for variety)
                t = np.linspace(0, 0.5, sr // 2, dtype=np.float32)
                freq = 220 + hash(fname) % 440
                audio = 0.5 * np.sin(2 * np.pi * freq * t)
                sf.write(str(d / fname), audio, sr)


class TestVoiceBankDEMANDDataset:
    """Dataset class with synthetic files."""

    TRAIN_SPEAKERS = ["p226", "p227", "p228", "p229", "p230"]
    TEST_SPEAKERS = ["p232", "p257"]

    @pytest.fixture()
    def mock_root(self, tmp_path):
        _create_mock_dataset(
            tmp_path,
            self.TRAIN_SPEAKERS,
            self.TEST_SPEAKERS,
            n_per_speaker=4,
        )
        return tmp_path

    def test_train_split(self, mock_root):
        from src.enhancement.dataset import DatasetConfig, VoiceBankDEMANDDataset

        cfg = DatasetConfig(root=str(mock_root), val_speakers=1, segment_length=4000)
        ds = VoiceBankDEMANDDataset(cfg, split="train")

        # 5 train speakers, hold out 1 → 4 speakers × 4 utterances
        assert len(ds) == 16
        assert len(ds.speakers) == 4

    def test_val_split(self, mock_root):
        from src.enhancement.dataset import DatasetConfig, VoiceBankDEMANDDataset

        cfg = DatasetConfig(root=str(mock_root), val_speakers=1, segment_length=4000)
        ds = VoiceBankDEMANDDataset(cfg, split="val")

        # Last 1 speaker → 4 utterances
        assert len(ds) == 4
        assert len(ds.speakers) == 1

    def test_test_split(self, mock_root):
        from src.enhancement.dataset import DatasetConfig, VoiceBankDEMANDDataset

        cfg = DatasetConfig(root=str(mock_root), val_speakers=1, segment_length=None)
        ds = VoiceBankDEMANDDataset(cfg, split="test")

        # 2 test speakers × 4 utterances
        assert len(ds) == 8
        assert len(ds.speakers) == 2

    def test_no_speaker_overlap(self, mock_root):
        """Train, val, and test splits have disjoint speaker sets."""
        from src.enhancement.dataset import DatasetConfig, VoiceBankDEMANDDataset

        cfg = DatasetConfig(root=str(mock_root), val_speakers=2)
        train_ds = VoiceBankDEMANDDataset(cfg, split="train")
        val_ds = VoiceBankDEMANDDataset(cfg, split="val")
        test_ds = VoiceBankDEMANDDataset(cfg, split="test")

        train_set = set(train_ds.speakers)
        val_set = set(val_ds.speakers)
        test_set = set(test_ds.speakers)

        assert train_set & val_set == set()
        assert train_set & test_set == set()
        assert val_set & test_set == set()

    def test_explicit_val_speakers(self, mock_root):
        from src.enhancement.dataset import DatasetConfig, VoiceBankDEMANDDataset

        cfg = DatasetConfig(root=str(mock_root), val_speakers=["p228"])
        val_ds = VoiceBankDEMANDDataset(cfg, split="val")
        assert val_ds.speakers == ["p228"]
        assert len(val_ds) == 4

    def test_getitem_shapes_training(self, mock_root):
        """Training __getitem__ returns fixed-length segments."""
        from src.enhancement.dataset import DatasetConfig, VoiceBankDEMANDDataset

        seg_len = 4000
        cfg = DatasetConfig(root=str(mock_root), segment_length=seg_len, val_speakers=0)
        ds = VoiceBankDEMANDDataset(cfg, split="train")

        noisy, clean = ds[0]
        assert noisy.shape == (seg_len,)
        assert clean.shape == (seg_len,)
        assert noisy.dtype == torch.float32

    def test_getitem_full_utterance_eval(self, mock_root):
        """Test split with segment_length=None returns full utterance."""
        from src.enhancement.dataset import DatasetConfig, VoiceBankDEMANDDataset

        cfg = DatasetConfig(root=str(mock_root), segment_length=None, val_speakers=0)
        ds = VoiceBankDEMANDDataset(cfg, split="test")

        noisy, clean = ds[0]
        # Full utterance — not a fixed length
        assert noisy.dim() == 1
        assert noisy.shape == clean.shape
        assert noisy.shape[0] > 0

    def test_invalid_split(self, mock_root):
        from src.enhancement.dataset import DatasetConfig, VoiceBankDEMANDDataset

        cfg = DatasetConfig(root=str(mock_root))
        with pytest.raises(ValueError, match="split"):
            VoiceBankDEMANDDataset(cfg, split="unknown")


class TestDataLoader:
    """DataLoader integration."""

    TRAIN_SPEAKERS = ["p226", "p227", "p228"]
    TEST_SPEAKERS = ["p232"]

    @pytest.fixture()
    def mock_root(self, tmp_path):
        _create_mock_dataset(
            tmp_path,
            self.TRAIN_SPEAKERS,
            self.TEST_SPEAKERS,
            n_per_speaker=4,
        )
        return tmp_path

    def test_train_dataloader_batching(self, mock_root):
        """Training DataLoader produces correctly shaped batches."""
        from src.enhancement.dataset import (
            DatasetConfig,
            VoiceBankDEMANDDataset,
            create_dataloader,
        )

        seg_len = 4000
        cfg = DatasetConfig(root=str(mock_root), segment_length=seg_len, val_speakers=0)
        ds = VoiceBankDEMANDDataset(cfg, split="train")
        dl = create_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False)

        batch = next(iter(dl))
        noisy, clean = batch
        assert noisy.shape == (4, seg_len)
        assert clean.shape == (4, seg_len)

    def test_eval_dataloader_variable_length(self, mock_root):
        """Eval DataLoader pads variable-length utterances correctly."""
        from src.enhancement.dataset import (
            DatasetConfig,
            VoiceBankDEMANDDataset,
            create_dataloader,
        )

        cfg = DatasetConfig(root=str(mock_root), segment_length=None, val_speakers=0)
        ds = VoiceBankDEMANDDataset(cfg, split="test")
        dl = create_dataloader(ds, batch_size=2, num_workers=0, pin_memory=False)

        batch = next(iter(dl))
        noisy, clean, lengths = batch
        assert noisy.dim() == 2
        assert clean.dim() == 2
        assert lengths.shape[0] == 2


class TestDatasetSummary:
    """Dataset summary statistics."""

    @pytest.fixture()
    def mock_root(self, tmp_path):
        _create_mock_dataset(
            tmp_path,
            ["p226", "p227", "p228"],
            ["p232"],
            n_per_speaker=5,
        )
        return tmp_path

    def test_summary_keys(self, mock_root):
        from src.enhancement.dataset import DatasetConfig, VoiceBankDEMANDDataset
        from src.enhancement.viz import dataset_summary

        cfg = DatasetConfig(root=str(mock_root), val_speakers=0)
        ds = VoiceBankDEMANDDataset(cfg, split="train")
        summary = dataset_summary(ds)

        assert summary["split"] == "train"
        assert summary["num_utterances"] == 15
        assert summary["num_speakers"] == 3
        assert len(summary["per_speaker_counts"]) == 3
        assert all(v == 5 for v in summary["per_speaker_counts"].values())
