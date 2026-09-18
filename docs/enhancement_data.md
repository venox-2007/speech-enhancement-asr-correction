# Speech Enhancement Data Pipeline

## Overview

This milestone adds the data infrastructure for training and evaluating
speech enhancement models.  It does **not** include model training — only
the dataset handling, audio preprocessing, STFT/iSTFT processing, and
visualization utilities that models will consume.

## Expected Dataset: VoiceBank+DEMAND

The VoiceBank+DEMAND dataset pairs clean speech from the Voice Bank
corpus with noise from the DEMAND database.  It is the standard
benchmark for supervised speech enhancement research.

### Download

The dataset is available from Edinburgh DataShare:

- **URL**: https://datashare.ed.ac.uk/handle/10283/2791
- **Size**: ~5 GB (uncompressed)
- **License**: Creative Commons Attribution 4.0

Download the following archives and extract them into a single directory:

```
data/voicebank_demand/
  clean_trainset_28spk_wav/    # 11,572 clean training utterances (28 speakers)
  noisy_trainset_28spk_wav/    # matching noisy training utterances
  clean_testset_wav/            # 824 clean test utterances (2 speakers)
  noisy_testset_wav/            # matching noisy test utterances
```

> [!IMPORTANT]
> Do NOT commit the dataset to Git.  The `.gitignore` already
> excludes `data/raw/`, `data/processed/`, and `*.wav`.

### Filename Convention

Filenames follow the pattern `p{speaker_id}_{utterance_id}.wav`.
For example, `p226_001.wav` is utterance 001 from speaker p226.

The speaker ID is used to ensure train/val/test splits are
speaker-disjoint (no data leakage).

## Train / Validation / Test Splits

The dataset ships pre-split into training (28 speakers) and test
(2 speakers).  This milestone adds a **validation split** carved from
the training speakers:

| Split | Source | Default Speakers |
|-------|--------|-----------------|
| Train | `clean_trainset_28spk_wav` minus val speakers | First 26 of 28 |
| Val   | `clean_trainset_28spk_wav`, held-out speakers | Last 2 of 28 |
| Test  | `clean_testset_wav` | All (as provided) |

### No Data Leakage

The split is **speaker-based**: a speaker's utterances appear in exactly
one partition.  This prevents the model from memorising speaker-specific
characteristics that do not generalise.

### Configuration

```python
from src.enhancement.dataset import DatasetConfig

# Default: hold out last 2 training speakers for validation
cfg = DatasetConfig(root="data/voicebank_demand", val_speakers=2)

# Explicit speaker list
cfg = DatasetConfig(root="data/voicebank_demand", val_speakers=["p256", "p257"])

# No validation split (all 28 speakers for training)
cfg = DatasetConfig(root="data/voicebank_demand", val_speakers=0)
```

## Audio Preprocessing

All audio is loaded at **16 kHz mono** using `soundfile` (same loader as
the ASR pipeline).  The existing `load_audio()` from
`src/preprocessing/audio_io.py` handles format detection, channel
downmix, and polyphase resampling.

## STFT Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_fft` | 512 | FFT size (32 ms at 16 kHz) |
| `hop_length` | 256 | Hop size (16 ms, 50% overlap) |
| `win_length` | 512 | Window length |
| `window` | `"hann"` | Window type |
| **Freq bins** | **257** | `n_fft // 2 + 1` |

These are configured via `STFTConfig` and shared across training,
evaluation, and inference.

### STFT → Magnitude + Phase

```python
from src.preprocessing.stft import STFTConfig, stft, istft

cfg = STFTConfig()
magnitude, phase = stft(audio_tensor, cfg)    # (F, N) or (B, F, N)
reconstructed = istft(magnitude, phase, cfg)  # round-trip
```

Enhancement models typically predict a clean magnitude spectrogram.
The noisy **phase** is reused during reconstruction (phase estimation
is a separate research problem).

### dB Conversion

```python
from src.preprocessing.stft import mag_to_db, db_to_mag

log_spec = mag_to_db(magnitude)        # 20 * log10(mag)
magnitude = db_to_mag(log_spec)        # inverse
```

## Segment Extraction

Training requires fixed-length inputs.  The default segment length is
**16,384 samples** (~1.02 s at 16 kHz), suitable for the 6 GB RTX 3050.

```python
from src.preprocessing.segments import paired_random_crop

# Random aligned crop (training)
clean_seg, noisy_seg = paired_random_crop(clean, noisy, 16384, generator=rng)

# Pad-or-crop (deterministic)
from src.preprocessing.segments import pad_or_crop
segment = pad_or_crop(audio, 16384)
```

The paired crop uses the **same random offset** for clean and noisy
so that alignment is preserved.

## Noise Mixing

The `noise_mix` module can create noisy mixtures at a target SNR:

```python
from src.preprocessing.noise_mix import mix_at_snr, estimate_snr

noisy = mix_at_snr(clean, noise, snr_db=10.0)
measured_snr = estimate_snr(clean, noisy)  # ≈ 10.0
```

This is useful for data augmentation and controlled experiments (the
VoiceBank+DEMAND dataset already provides pre-mixed noisy audio).

## PyTorch Dataset and DataLoader

```python
from src.enhancement.dataset import (
    DatasetConfig,
    VoiceBankDEMANDDataset,
    create_dataloader,
)

cfg = DatasetConfig(
    root="data/voicebank_demand",
    segment_length=16384,
    val_speakers=2,
)

train_ds = VoiceBankDEMANDDataset(cfg, split="train")
val_ds   = VoiceBankDEMANDDataset(cfg, split="val")
test_ds  = VoiceBankDEMANDDataset(cfg, split="test")

train_dl = create_dataloader(train_ds, batch_size=8)
val_dl   = create_dataloader(val_ds, batch_size=1)
```

### Training Mode

When `segment_length` is set and `split="train"`, each `__getitem__`
call returns a random crop of that length (different each epoch).

### Evaluation Mode

For `split="val"` or `split="test"`, the full utterance is returned.
The `create_dataloader` function automatically uses a variable-length
collation function that pads to the longest utterance in the batch and
returns `(noisy_batch, clean_batch, lengths)`.

## Visualization

```python
from src.enhancement.viz import plot_spectrogram, plot_comparison, dataset_summary

# Single spectrogram
ax = plot_spectrogram(magnitude, cfg)

# Side-by-side clean / noisy / enhanced
fig = plot_comparison(clean_wav, noisy_wav, enhanced_wav, cfg)
fig.savefig("comparison.png")

# Dataset statistics
summary = dataset_summary(train_ds)
print(summary)
```

## Module Layout

| Module | Description |
|--------|-------------|
| `src/preprocessing/stft.py` | `STFTConfig`, `stft`, `istft`, `mag_to_db`, `db_to_mag` |
| `src/preprocessing/segments.py` | `pad_or_crop`, `random_crop`, `paired_random_crop` |
| `src/preprocessing/noise_mix.py` | `mix_at_snr`, `estimate_snr` |
| `src/enhancement/dataset.py` | `VoiceBankDEMANDDataset`, `DatasetConfig`, `create_dataloader` |
| `src/enhancement/viz.py` | `plot_spectrogram`, `plot_comparison`, `dataset_summary` |
| `tests/test_enhancement_data.py` | All tests (synthetic audio, no real dataset required) |

## What This Does NOT Include

- No model training (CNN or U-Net)
- No speech quality metrics (PESQ, STOI, SI-SDR) — added later
- No experimental results
- No modifications to the ASR or correction pipeline
