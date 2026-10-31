# Speech Enhancement Training Pipeline

## Overview

This milestone implements the supervised training infrastructure for the 2D U-Net speech enhancement model. It provides reproducible train and validation loops, optimization with automatic mixed precision (AMP), early stopping, checkpointing, and metric logging.

> [!IMPORTANT]
> **Research Status**: No full model training has been conducted yet. The pipeline has been verified via synthetic unit tests and a short real-data GPU smoke test. No performance claims are made. Full speech-quality evaluation (PESQ, STOI, SI-SDR) and benchmark results are deferred to subsequent evaluation milestones.

---

## Methodology & Optimization

### 1. Training Objective: Magnitude $L_1$ Loss
The primary baseline loss is computed directly on the predicted magnitude spectrogram $\hat{S} = |\hat{S}(f, t)|$ and ground-truth clean magnitude spectrogram $S = |S(f, t)|$:

$$\mathcal{L}_{\text{mag}} = \frac{1}{B \cdot F \cdot N} \sum_{b=1}^{B} \sum_{f=0}^{F-1} \sum_{t=0}^{N-1} \left| \hat{S}_{b}(f, t) - S_{b}(f, t) \right|$$

**Design Rationale**:
- **Linear Error Penalty**: Unlike $L_2$ (MSE) loss which squares spectral errors and heavily biases optimization toward high-energy vowel formants at the expense of quiet fricatives and consonants, $L_1$ loss applies equal proportional penalty across all energy levels.
- **Configurability**: The loss is implemented in `SpectralLoss` ([`src/enhancement/loss.py`](../src/enhancement/loss.py)) and can be configured to use `l1`, `mse`, `smooth_l1`, or power-law compression ($|S|^\alpha$ with $\alpha=0.5$).

### 2. Optimization Settings & Hyperparameter Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Optimizer** | `AdamW` | Decoupled weight decay regularization |
| **Initial Learning Rate** | `1e-3` | Standard initial rate for U-Net spectral models |
| **Weight Decay** | `1e-5` | $L_2$ weight regularization |
| **Batch Size (Train)** | `16` | Fixed $16,384$-sample segment batches (~1.024 s @ 16 kHz) |
| **Batch Size (Val)** | `1` | Full-utterance evaluation without padding overhead |
| **Gradient Clipping** | `5.0` | Max norm clipping via `clip_grad_norm_` |
| **Mixed Precision (AMP)** | `True` (on CUDA) | `torch.amp.autocast('cuda')` with `GradScaler` |
| **LR Scheduler** | `ReduceLROnPlateau` | Factor `0.5`, patience `5` epochs on validation loss |
| **Early Stopping** | `10` epochs | Halts training when validation loss fails to improve |
| **Max Epochs** | `50` | Maximum training iterations |
| **Random Seed** | `42` | Seed for Python, NumPy, PyTorch, and CUDA |

### 3. Validation Strategy & Safe Variable-Length Handling
- **Speaker-Disjoint Split**: Validation is performed exclusively on held-out speakers `p286` (female, 455 utterances) and `p287` (male, 417 utterances), ensuring zero speaker overlap with training.
- **Variable-Length Safety**: Validation utterances vary from 1.1 s to 8.6 s. Rather than padding large batches to the maximum utterance length in memory, validation defaults to `val_batch_size=1`. Each utterance is evaluated at its exact natural length without padding overhead or GPU memory spikes.

---

## Checkpointing & Artifacts

The `Trainer` automatically manages training artifacts in `--output-dir` (default: `results/enhancement_train/`):

1. **`checkpoint_latest.pt`**: Saved every epoch. Contains current epoch, full model weights, optimizer state, scheduler state, AMP scaler state, current best validation loss, and complete training history. Enables seamless resumption with `--resume`.
2. **`checkpoint_best.pt`**: Overwritten only when validation loss strictly achieves a new minimum.
3. **`history.json` & `history.csv`**: Updated every epoch, logging:
   - `epoch`: Current epoch index
   - `train_loss`: Mean training loss over all batches
   - `val_loss`: Mean validation loss
   - `val_l1`: Mean validation magnitude $L_1$ error
   - `val_mse`: Mean validation magnitude MSE error
   - `lr`: Active learning rate
   - `epoch_time_sec`: Elapsed duration for the epoch

---

## CLI Usage

### Standard Training Run

```bash
python -m src.enhancement.train \
    --data-dir data/processed/voicebank_demand_16k \
    --output-dir results/enhancement_train \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3 \
    --device cuda
```

### Resume Training from Checkpoint

```bash
python -m src.enhancement.train \
    --data-dir data/processed/voicebank_demand_16k \
    --resume results/enhancement_train/checkpoint_latest.pt
```

### GPU Smoke Test

To verify GPU execution, AMP, and checkpoint save/resume without running a full training epoch:

```bash
python -m src.enhancement.train --smoke-test --device cuda
```

---

## Real Dataset Baseline Experiment Results

The first controlled U-Net training run was executed on the processed VoiceBank+DEMAND dataset on an NVIDIA GeForce RTX 3050 6GB Laptop GPU.

### Exact Execution Command & Configuration
```bash
python -u -m src.enhancement.train \
    --data-dir data/processed/voicebank_demand_16k \
    --output-dir results/enhancement_train \
    --epochs 20 \
    --batch-size 16 \
    --val-batch-size 1 \
    --lr 1e-3 \
    --lr-patience 2 \
    --patience 5 \
    --seed 42 \
    --device cuda
```

### Empirical Training Metrics & Progression

- **Dataset**: 10,700 training utterances (26 speakers), 872 validation utterances (`p286` female, `p287` male)
- **Model Size**: 482,449 trainable parameters (~0.482 M, ~1.9 MB)
- **Peak GPU Memory**: **255.7 MB** (allocated) / 330.0 MB (reserved) — < 5% of 6 GB VRAM
- **Total Training Duration**: **1,112.0 seconds** (18.53 minutes)
- **Epoch Duration**: ~41.8 seconds per subsequent epoch (training and validation combined, excluding initial caching)

| Epoch | Train Loss ($L_1$) | Val Loss ($L_1$) | Val MSE | Active LR | Peak GPU | Epoch Time | Notes |
|:-----:|:------------------:|:----------------:|:-------:|:---------:|:--------:|:----------:|:-----:|
| 1 | 0.0612 | 0.0506 | 0.0615 | $1.00 \times 10^{-3}$ | 255.0 MB | 318.4 s | Initial caching |
| 2 | 0.0539 | 0.0474 | 0.0521 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.8 s | Best checkpoint |
| 3 | 0.0515 | 0.0468 | 0.0507 | $1.00 \times 10^{-3}$ | 255.7 MB | 39.1 s | Best checkpoint |
| 4 | 0.0499 | 0.0442 | 0.0441 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.4 s | Best checkpoint |
| 5 | 0.0487 | 0.0444 | 0.0473 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.6 s | |
| 6 | 0.0477 | 0.0433 | 0.0457 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.8 s | Best checkpoint |
| 7 | 0.0475 | 0.0424 | 0.0403 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.7 s | Best checkpoint |
| 8 | 0.0468 | 0.0421 | 0.0377 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.6 s | Best checkpoint |
| 9 | 0.0464 | 0.0410 | 0.0410 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.7 s | Best checkpoint |
| 10 | 0.0462 | 0.0415 | 0.0420 | $1.00 \times 10^{-3}$ | 255.7 MB | 42.1 s | |
| 11 | 0.0456 | 0.0407 | 0.0402 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.2 s | Best checkpoint |
| 12 | 0.0455 | 0.0415 | 0.0398 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.6 s | |
| 13 | 0.0452 | 0.0419 | 0.0432 | $1.00 \times 10^{-3}$ | 255.7 MB | 42.1 s | |
| 14 | 0.0449 | 0.0398 | 0.0341 | $1.00 \times 10^{-3}$ | 255.7 MB | 43.5 s | Best checkpoint |
| 15 | 0.0446 | 0.0403 | 0.0355 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.5 s | |
| 16 | 0.0445 | 0.0419 | 0.0396 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.5 s | |
| 17 | 0.0442 | 0.0396 | 0.0344 | $1.00 \times 10^{-3}$ | 255.7 MB | 41.5 s | Best checkpoint |
| 18 | 0.0441 | 0.0393 | 0.0348 | $1.00 \times 10^{-3}$ | 255.7 MB | 42.4 s | Best checkpoint |
| **19** | **0.0441** | **0.0387** | **0.0323** | $1.00 \times 10^{-3}$ | **255.7 MB** | **42.0 s** | **★ Best Checkpoint** |
| 20 | 0.0438 | 0.0400 | 0.0420 | $1.00 \times 10^{-3}$ | 255.7 MB | 42.1 s | Final epoch |

### Independent Checkpoint Verification
The best checkpoint saved at `results/enhancement_train/checkpoint_best.pt` was verified independently using `load_enhancer_checkpoint`:
- Successfully re-instantiated and loaded weights on both CPU and CUDA without needing training dependencies.
- Verified exact sample-length waveform inference on CPU and CUDA: input `(1, 16384)` $\to$ output `(1, 16384)`.

---

## Module Layout

| Module | Description |
|--------|-------------|
| `src/enhancement/loss.py` | `SpectralLoss` supporting L1, MSE, Smooth L1, power compression, and frame masking |
| `src/enhancement/trainer.py` | `Trainer` and `TrainingConfig` orchestrating training, AMP, validation, and checkpointing |
| `src/enhancement/train.py` | Command-line interface with argument parsing and `--smoke-test` utility |
| `tests/test_enhancement_training.py` | Synthetic tests for loss, training step, determinism, checkpoint roundtrip, and early stopping |

