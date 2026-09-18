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

| Split | Source | Speakers | Demographics |
|-------|--------|----------|--------------|
| Train | `clean_trainset_28spk_wav` minus val speakers | 26 speakers | 13 female, 13 male |
| Val   | `clean_trainset_28spk_wav`, held-out speakers | `["p286", "p287"]` (default) | 1 female (p286), 1 male (p287) |
| Test  | `clean_testset_wav` | `["p232", "p257"]` | 1 female (p257), 1 male (p232) |

### No Data Leakage

The split is **speaker-based**: a speaker's utterances appear in exactly
one partition.  This prevents the model from memorising speaker-specific
characteristics that do not generalise.

### Configuration

```python
from src.enhancement.dataset import DatasetConfig

# Default: hold out ["p286", "p287"] (1 female, 1 male)
cfg = DatasetConfig(root="data/voicebank_demand")

# Explicit speaker list
cfg = DatasetConfig(root="data/voicebank_demand", val_speakers=["p286", "p287"])

# Integer count: holds out the last N naturally sorted speakers
cfg = DatasetConfig(root="data/voicebank_demand", val_speakers=2)

# No validation split (all 28 speakers for training)
cfg = DatasetConfig(root="data/voicebank_demand", val_speakers=0)
```

## Audio Preprocessing & Offline Resampling

All audio is processed at **16 kHz mono** using `soundfile`.
The original Edinburgh DataShare release of VoiceBank+DEMAND is sampled
at 48 kHz (~11,572 files in train).  Performing polyphase resampling on
the fly in the DataLoader introduces severe CPU bottlenecks during training.

An offline script is provided to resample the dataset once on disk:

```bash
python -m src.preprocessing.resample_dataset \
    --input-dir data/voicebank_demand_raw \
    --output-dir data/voicebank_demand_16k \
    --target-sr 16000
```

This preserves the exact clean/noisy file pairing and subdirectory layout.

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

## Noise Mixing & Clipping Policy

The `noise_mix` module creates noisy mixtures at a target SNR:

```python
from src.preprocessing.noise_mix import mix_at_snr, estimate_snr

# Default policy: "warn" logs a warning if peak > 1.0 without distorting the SNR
noisy = mix_at_snr(clean, noise, snr_db=10.0, clipping_policy="warn")
measured_snr = estimate_snr(clean, noisy)  # ≈ 10.0
```

### Clipping Policies

At low SNR (e.g. 0 dB or negative SNR), additive noise can push amplitudes
past `[-1.0, 1.0]`.  The `clipping_policy` parameter provides explicit control:

| Policy | Behavior |
|--------|----------|
| `"warn"` (default) | Logs a warning with the peak amplitude; leaves mixture unscaled so the intended SNR is preserved. |
| `"error"` | Raises a `ValueError` if peak amplitude exceeds 1.0. |
| `"ignore"` | Leaves mixture unscaled silently. |
| `"normalize"` | Scales the mixture by `1.0 / peak`; prevents clipping while preserving relative SNR, but reduces overall gain. |

## PyTorch Dataset and DataLoader

```python
from src.enhancement.dataset import (
    DatasetConfig,
    VoiceBankDEMANDDataset,
    create_dataloader,
)

cfg = DatasetConfig(
    root="data/voicebank_demand_16k",
    segment_length=16384,
    val_speakers=["p286", "p287"],
    seed=42,
)

train_ds = VoiceBankDEMANDDataset(cfg, split="train")
val_ds   = VoiceBankDEMANDDataset(cfg, split="val")
test_ds  = VoiceBankDEMANDDataset(cfg, split="test")

train_dl = create_dataloader(train_ds, batch_size=8)
val_dl   = create_dataloader(val_ds, batch_size=1)
```

### Standardized Batch Output: `EnhancementBatch`

`create_dataloader` yields an `EnhancementBatch` NamedTuple consistently
across **train**, **val**, and **test**:

```python
for batch in train_dl:
    noisy, clean, lengths = batch
    # Or access via attributes:
    # batch.noisy   -> (B, T)
    # batch.clean   -> (B, T)
    # batch.lengths -> (B,) int64 valid audio lengths
```

- **Training**: segments are cropped reproducibly using `DatasetConfig.seed` (or worker-adjusted seeds).
- **Validation/Test**: full utterances are zero-padded to the longest in the batch; `batch.lengths` records original utterance sample counts.

## STFT Shape Constraint for Future CNN / U-Net

At $16\text{ kHz}$ with standard STFT parameters (`n_fft=512, hop_length=256`):
- A $16,384$-sample segment (~1.02 s) produces an STFT of shape $(257, 65)$ with PyTorch's default `center=True` reflection padding.
- **Divisibility Challenge**: A standard 4- or 5-layer 2D U-Net downsamples by powers of 2 ($2^4=16$ or $2^5=32$). Neither $257$ frequency bins nor $65$ time frames is evenly divisible by 16 or 32, leading to dimension mismatches across skip connections upon upsampling.
- **Decision Deferred**: We will deliberately decide the model/input convention before implementation rather than arbitrarily dropping frequency bins or frames now. Potential options include:
  1. *2D Reflection Padding / Cropping*: Handle boundary padding inside the network architecture to ensure symmetric skip connections.
  2. *Spectral Trimming*: Drop the 8 kHz Nyquist bin ($257 \to 256$) and adjust segment length or hop to yield 64 frames ($256 \times 64$).
  3. *Time-Domain Modeling*: Implement a 1D time-domain Wave-U-Net directly on the $(B, 1, 16384)$ waveforms, bypassing 2D STFT dimensions entirely.

## Visualization

```python
from src.enhancement.viz import plot_spectrogram, plot_comparison, dataset_summary

# Single spectrogram
ax = plot_spectrogram(magnitude, cfg)

# Side-by-side clean / noisy / enhanced (supports 1-D (T,) or batched (1, T))
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
| `src/preprocessing/noise_mix.py` | `mix_at_snr`, `estimate_snr`, clipping policies |
| `src/preprocessing/resample_dataset.py` | Offline 16 kHz dataset resampling utility |
| `src/enhancement/dataset.py` | `VoiceBankDEMANDDataset`, `DatasetConfig`, `create_dataloader`, `EnhancementBatch` |
| `src/enhancement/viz.py` | `plot_spectrogram`, `plot_comparison`, `dataset_summary` |
| `tests/test_enhancement_data.py` | Comprehensive tests (synthetic audio, no real dataset required) |

## What This Does NOT Include

- No model training (CNN or U-Net)
- No speech quality metrics (PESQ, STOI, SI-SDR) — added in future milestone
- No experimental results
- No modifications to the ASR or correction pipeline
