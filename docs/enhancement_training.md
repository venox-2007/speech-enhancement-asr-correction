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
- **Configurability**: The loss is implemented in `SpectralLoss` ([`src/enhancement/loss.py`](file:///c:/Users/yugag/OneDrive/Documents/Sem%205/DL%20Project/src/enhancement/loss.py)) and can be configured to use `l1`, `mse`, `smooth_l1`, or power-law compression ($|S|^\alpha$ with $\alpha=0.5$).

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

## Module Layout

| Module | Description |
|--------|-------------|
| `src/enhancement/loss.py` | `SpectralLoss` supporting L1, MSE, Smooth L1, power compression, and frame masking |
| `src/enhancement/trainer.py` | `Trainer` and `TrainingConfig` orchestrating training, AMP, validation, and checkpointing |
| `src/enhancement/train.py` | Command-line interface with argument parsing and `--smoke-test` utility |
| `tests/test_enhancement_training.py` | Synthetic tests for loss, training step, determinism, checkpoint roundtrip, and early stopping |
