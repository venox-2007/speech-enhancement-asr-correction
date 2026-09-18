# Speech Enhancement Model Architecture: 2D U-Net Baseline

## Overview

This milestone introduces the speech enhancement baseline model: a convolutional **2D U-Net** operating on Short-Time Fourier Transform (STFT) magnitude spectrograms. The model processes noisy speech waveforms end-to-end via an integrated analysis-synthesis wrapper:

$$\text{Noisy Waveform } y(t) \xrightarrow{\text{STFT}} |Y(f, t)|, \angle Y(f, t) \xrightarrow{\text{U-Net}} |\hat{S}(f, t)| \xrightarrow{\text{iSTFT}(\cdot, \angle Y)} \hat{s}(t)$$

## Mathematical Formulation & Data Flow

### 1. Time-Frequency Representation
Given a discrete-time noisy speech signal $y \in \mathbb{R}^T$ sampled at $16\text{ kHz}$:

$$Y(f, t) = \text{STFT}(y; n_{\text{fft}}=512, R=256, w=\text{hann})$$

where $f \in \{0, \dots, 257\}$ denotes frequency bins ($n_{\text{fft}} / 2 + 1$) and $t \in \{0, \dots, N-1\}$ denotes time frames ($N = 1 + \lfloor T / R \rfloor$). The complex spectrogram is decomposed into:
- Magnitude spectrogram: $|Y(f, t)| \in \mathbb{R}_{\ge 0}^{F \times N}$
- Phase spectrogram: $\angle Y(f, t) \in (-\pi, \pi]^{F \times N}$

### 2. Spectrogram Enhancement: Ratio Masking vs Direct Prediction
The 2D U-Net accepts $|Y| \in \mathbb{R}^{B \times 1 \times F \times N}$ and outputs an estimated clean magnitude $|\hat{S}|$. Two operating modes are supported:

1. **Ratio Masking (`output_mode="mask"`, default)**:
   The network predicts a real-valued time-frequency mask $M(f, t)$:
   $$|\hat{S}(f, t)| = M(f, t) \odot |Y(f, t)|$$
   - **Sigmoid (`mask_act="sigmoid"`)**: $M(f, t) = \sigma(z) \in (0, 1)$. Models pure attenuation (suppressing noise energy without amplifying spectral bins). Standard for additive noise.
   - **ReLU (`mask_act="relu"`)**: $M(f, t) = \max(0, z) \in [0, \infty)$. Allows amplification of suppressed speech components.
   - **Softplus (`mask_act="softplus"`)**: $M(f, t) = \ln(1 + e^z) \in (0, \infty)$. Smooth non-negative gain.

2. **Direct Magnitude Estimation (`output_mode="direct"`)**:
   $$|\hat{S}(f, t)| = \max(0, \text{U-Net}(|Y(f, t)|))$$
   The network directly synthesizes magnitude values without explicit multiplicative coupling to the input.

### 3. Waveform Synthesis
The enhanced speech waveform is reconstructed via inverse STFT using the estimated clean magnitude and the **original noisy phase**:

$$\hat{s}(t) = \text{iSTFT}\left(|\hat{S}(f, t)| \cdot e^{j \angle Y(f, t)}; n_{\text{fft}}=512, R=256, w=\text{hann}\right)$$

Reusing the noisy phase is the standard convention in magnitude-only spectral enhancement baselines, as clean phase estimation is non-trivial and phase errors have secondary perceptual impact at moderate SNRs.

---

## Handling STFT Dimensions & Shape Decisions

### The Divisibility Challenge
At $16\text{ kHz}$ with standard parameters ($n_{\text{fft}}=512$, $R=256$):
- $16,384$ samples (~1.024 s) yields $(F, N) = (257, 65)$ with reflection padding (`center=True`).
- A $D$-layer U-Net downsamples by $2^D$. For depth 4, the downsampling divisor is $2^4 = 16$.
- Neither 257 nor 65 is divisible by 16 ($257 \equiv 1 \pmod{16}$, $65 \equiv 1 \pmod{16}$).

### The Zero-Discard Padding Strategy
Rather than discarding the 8 kHz Nyquist frequency bin ($257 \to 256$), `UNet2D` performs **dynamic internal padding**:

1. **Calculate Pad Lengths**:
   $$p_F = (2^D - (F \pmod{2^D})) \pmod{2^D} = (16 - 1) = 15$$
   $$p_N = (2^D - (N \pmod{2^D})) \pmod{2^D} = (16 - 1) = 15$$
2. **Boundary Padding**:
   $|Y|$ is padded from $(257, 65) \to (272, 80)$ using replicate padding. Both 272 ($17 \times 16$) and 80 ($5 \times 16$) are exact multiples of 16.
3. **Exact Down/Up Symmetry**:
   - $272 \to 136 \to 68 \to 34 \to 17$ (encoder)
   - $17 \to 34 \to 68 \to 136 \to 272$ (decoder)
   All skip connections align pixel-for-pixel without interpolation or size discrepancies.
4. **Unpad Slicing**:
   The output is sliced back to $(:, :, :F, :N)$ before returning, cleanly dropping boundary artifacts.

**Advantages**:
- Zero frequency bins discarded (all 257 bins from 0 Hz to 8 kHz are preserved and enhanced).
- Variable-length utterances supported seamlessly during evaluation.
- No boundary mismatch across skip connections.

---

## Architecture Specification

```text
Input: (B, 1, 257, N)
  ↓ Pad to (B, 1, 272, N_pad)
DoubleConv (1 -> 16)               ──────── skip0 (B, 16, 272, N_pad) ─────────┐
  ↓ MaxPool2d(2)                                                               │
DoubleConv (16 -> 32)              ──── skip1 (B, 32, 136, N_pad/2) ────┐      │
  ↓ MaxPool2d(2)                                                        │      │
DoubleConv (32 -> 64)              ─ skip2 (B, 64, 68, N_pad/4) ──┐     │      │
  ↓ MaxPool2d(2)                                                  │     │      │
DoubleConv (64 -> 128) [Bottleneck]                               │     │      │
  ↓ ConvTranspose2d(128 -> 64)                                    │     │      │
Cat + DoubleConv (128 -> 64)       <──────────────────────────────┘     │      │
  ↓ ConvTranspose2d(64 -> 32)                                           │      │
Cat + DoubleConv (64 -> 32)        <────────────────────────────────────┘      │
  ↓ ConvTranspose2d(32 -> 16)                                                  │
Cat + DoubleConv (32 -> 16)        <───────────────────────────────────────────┘
  ↓ Conv2d(16 -> 1, 1x1)
Mask / Direct Activation
  ↓ Unpad to (B, 1, 257, N)
Output: (B, 1, 257, N)
```

### Parameter Budget & RTX 3050 6 GB Fit

| Component | Default Channels | Parameters | Footprint (FP32) |
|-----------|------------------|------------|------------------|
| `UNet2D` (Depth 4) | `(16, 32, 64, 128)` | **482,449** (0.482 M) | ~1.9 MB |
| `WaveformUNetEnhancer` | Same | **482,449** (0.482 M) | ~1.9 MB |

With batch size 16 (16,384 samples/utterance), forward+backward activations require ~120 MB VRAM, fitting comfortably within the 6 GB VRAM limit of the RTX 3050 Laptop GPU.

---

## Python Usage

### 1. End-to-End Waveform Enhancement

```python
import torch
from src.enhancement.model import WaveformUNetEnhancer, EnhancementModelConfig

enhancer = WaveformUNetEnhancer()
noisy_wav = torch.randn(2, 16384)  # (B, T)
enhanced_wav = enhancer(noisy_wav)  # (B, T)
```

### 2. Direct Spectrogram Training

```python
from src.enhancement.unet import UNet2D, UNetConfig

unet = UNet2D(UNetConfig(output_mode="mask", mask_act="sigmoid"))
noisy_mag = torch.rand(8, 1, 257, 65)  # (B, 1, F, N)
enhanced_mag = unet(noisy_mag)         # (B, 1, F, N)
```

### 3. Parameter Inspection

```python
from src.enhancement.model import count_parameters, model_summary

counts = count_parameters(enhancer)
print(f"Trainable parameters: {counts['trainable']:,}")
print(model_summary(enhancer))
```

---

## Limitations

1. **Phase Invariance**: Only spectral magnitude is enhanced. High-frequency phase distortion under extreme low SNR (< 0 dB) is retained from the noisy signal.
2. **Temporal Context**: Standard 2D convolutions have a bounded receptive field determined by depth and kernel size. Longer contextual acoustic dependencies (e.g. non-stationary noises) may require recurrent, temporal attention, or dilated convolutions.
3. **Additive Noise Assumption**: Sigmoid ratio masking assumes the clean speech energy is strictly less than or equal to the noisy speech energy. Under non-additive conditions, `mask_act="relu"` or `output_mode="direct"` is preferred.
