# Speech Enhancement Objective Evaluation on Held-Out Test Set

## Overview

This milestone presents the rigorous objective evaluation of the trained 2D U-Net baseline speech enhancement model on the complete, held-out VoiceBank+DEMAND test set. 

> [!IMPORTANT]
> **Controlled Evaluation Protocol**:
> - **Model State**: Evaluated strictly from frozen weights in `results/enhancement_train/checkpoint_best.pt` (Epoch 19, validation loss $0.0387$). **Zero retraining was conducted**.
> - **Test Corpus**: All 824 paired utterances from `data/processed/voicebank_demand_16k/` test partition.
> - **Utterance Processing**: Full utterances are preserved without artificial chunking or segmentation. Reconstructed enhanced waveforms retain the exact sample length of the input.
> - **Zero Data Leakage**: Test speakers `p232` and `p257` are strictly disjoint from all 26 training speakers and both validation speakers (`p286`, `p287`).
> - **Reproducibility**: Complete per-utterance results are saved in `results/enhancement_eval/metrics_per_utterance.csv` and aggregated summary in `results/enhancement_eval/summary.json`.

---

## Evaluation Methodology & Metric Definitions

### 1. Short-Time Objective Intelligibility (STOI)
- **Definition**: Correlates the short-time temporal envelopes of clean and degraded speech across 1/3-octave bands (using standard 384 ms segments).
- **Scale**: Normalized score in $[0.0, 1.0]$. Higher values indicate higher expected speech intelligibility.
- **Implementation**: Computed via `pystoi.stoi` at 16 kHz.
- **Delta Metric**: $\Delta\text{STOI} = \text{STOI}_{\text{enhanced}} - \text{STOI}_{\text{noisy}}$.

### 2. Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
- **Definition**: Evaluates separation/denoising fidelity while remaining invariant to arbitrary global gain/scaling:
  $$s_{\text{target}} = \frac{\langle s, \hat{s} \rangle}{\|s\|^2 + \epsilon} \cdot s, \quad e_{\text{noise}} = \hat{s} - s_{\text{target}}$$
  $$\text{SI-SDR} = 10 \log_{10} \left( \frac{\|s_{\text{target}}\|^2 + \epsilon}{\|e_{\text{noise}}\|^2 + \epsilon} \right)$$
- **Scale**: Decibels (dB). Higher values indicate better signal preservation relative to residual noise and distortion.
- **Delta Metric**: $\Delta\text{SI-SDR} = \text{SI-SDR}_{\text{enhanced}} - \text{SI-SDR}_{\text{noisy}}$.

### 3. Signal-to-Noise Ratio (SNR) and SNR Improvement ($\Delta\text{SNR}$)
- **Definition**: Conventional energy ratio in decibels:
  $$\text{SNR} = 10 \log_{10} \left( \frac{\|s\|^2 + \epsilon}{\|\hat{s} - s\|^2 + \epsilon} \right)$$
- **Delta Metric**: $\Delta\text{SNR} = \text{SNR}_{\text{enhanced}} - \text{SNR}_{\text{noisy}}$.

### 4. Perceptual Evaluation of Speech Quality (PESQ)
- **Status & Windows Environment Policy**: The ITU-T P.862 standard PESQ algorithm requires native C compilation. On Windows environments without Microsoft Visual C++ Build Tools (`cl.exe`), binary wheels are not distributed on PyPI. 
- The evaluation module implements `calculate_pesq` with graceful degradation: it attempts to import `pesq` and computes wideband MOS ($[-0.5, 4.5]$) if compiled libraries are present; otherwise, it logs an informative notice and reports `None` without halting the pipeline.

---

## Empirical Benchmark Results

### 1. Overall Test Set Performance (824 Utterances)

| Metric | Noisy (Baseline) | U-Net Enhanced | Delta ($\Delta$) | Empirical Interpretation |
|:-------|:----------------:|:--------------:|:----------------:|:-------------------------|
| **STOI** | $0.9211 \pm 0.0709$ | **$0.9338 \pm 0.0598$** | **$+0.0127 \pm 0.0193$** | Mean intelligibility improved; standard deviation reduced |
| **SI-SDR (dB)** | $8.45 \pm 5.61$ dB | **$17.96 \pm 3.86$ dB** | **$+9.51 \pm 4.12$ dB** | Substantial **9.5 dB** reduction in residual noise & distortion |
| **SNR (dB)** | $8.45 \pm 5.61$ dB | **$18.05 \pm 3.77$ dB** | **$+9.60 \pm 4.12$ dB** | Substantial **9.6 dB** energy-domain signal quality improvement |
| **PESQ (MOS)** | *N/A* | *N/A* | *N/A* | Requires native C compiler extension on host |

**Distribution Highlights**:
- Median $\Delta\text{SI-SDR}$: **$+8.90\text{ dB}$** (Max: $+22.14\text{ dB}$)
- Median $\Delta\text{SNR}$: **$+9.05\text{ dB}$** (Max: $+22.16\text{ dB}$)
- Median $\Delta\text{STOI}$: **$+0.0074$** (Max: $+0.1158$)
- Processing Throughput: **15.6 utterances / second** on NVIDIA RTX 3050 Laptop GPU (52.8 seconds total for all 824 utterances).

---

### 2. Breakdown by Speaker

The VoiceBank+DEMAND test set contains two speakers: male speaker `p232` (393 utterances) and female speaker `p257` (431 utterances).

| Speaker | Sex | Utterances | Noisy STOI | Enhanced STOI | $\Delta$ STOI | Noisy SI-SDR | Enhanced SI-SDR | $\Delta$ SI-SDR | $\Delta$ SNR |
|:-------:|:---:|:----------:|:----------:|:-------------:|:-------------:|:------------:|:---------------:|:---------------:|:------------:|
| **p232** | Male | 393 | 0.9270 | **0.9404** | **+0.0134** | 8.45 dB | **18.51 dB** | **+10.06 dB** | **+10.15 dB** |
| **p257** | Female | 431 | 0.9157 | **0.9277** | **+0.0121** | 8.44 dB | **17.46 dB** | **+9.02 dB** | **+9.11 dB** |

**Observations**:
- Both male and female speakers exhibit consistent, significant noise reduction ($\ge 9.0\text{ dB}$).
- Speaker `p232` (male) achieved slightly higher gains ($+10.06\text{ dB}$ vs $+9.02\text{ dB}$ SI-SDR), which is consistent with the lower fundamental pitch ($F_0$) aligning well with the 256-frequency-bin STOI analysis windows.

---

### 3. Breakdown by Input Noise / SNR Condition

Utterances were binned by input SNR into standard speech benchmark tiers:

| Input SNR Tier | Utterances | Noisy STOI | Enhanced STOI | $\Delta$ STOI | Noisy SI-SDR | Enhanced SI-SDR | $\Delta$ SI-SDR | $\Delta$ SNR |
|:---------------|:----------:|:----------:|:-------------:|:-------------:|:------------:|:---------------:|:---------------:|:------------:|
| **$< 5$ dB (Low SNR)** | 218 | 0.8691 | **0.8931** | **+0.0240** | 1.15 dB | **14.34 dB** | **+13.19 dB** | **+13.37 dB** |
| **$5 - 10$ dB (Medium-Low)** | 209 | 0.9166 | **0.9329** | **+0.0163** | 6.36 dB | **17.29 dB** | **+10.93 dB** | **+11.01 dB** |
| **$10 - 15$ dB (Medium-High)** | 206 | 0.9425 | **0.9491** | **+0.0066** | 11.23 dB | **19.24 dB** | **+8.01 dB** | **+8.06 dB** |
| **$\ge 15$ dB (High SNR)** | 191 | 0.9622 | **0.9647** | **+0.0025** | 16.07 dB | **21.46 dB** | **+5.39 dB** | **+5.42 dB** |

**Key Research Takeaway**:
- **Monotonic Intelligibility Gain**: The U-Net provides its highest intelligibility gain ($\Delta\text{STOI} = +0.0240$) and noise suppression ($\Delta\text{SNR} = +13.37\text{ dB}$) in the most severely corrupted acoustic environments ($< 5\text{ dB}$ SNR).
- **Graceful High-SNR Plateau**: In high-SNR environments ($\ge 15\text{ dB}$), where the incoming signal is already intelligible (STOI $0.9622$), the model preserves speech fidelity without aggressive over-filtering ($\Delta\text{STOI} = +0.0025$) while still eliminating $+5.4\text{ dB}$ of residual stationary and non-stationary background noise.

---

## Leakage Verification & Integrity

To guarantee that benchmark results are uncontaminated by speaker memorization, speaker disjointness was validated across all 28 training/validation speakers and 2 test speakers:
- **Training Speakers** ($N=26$): `p226`, `p227`, `p228`, `p230`, `p231`, `p233`, `p234`, `p236`, `p239`, `p240`, `p243`, `p244`, `p250`, `p254`, `p256`, `p258`, `p259`, `p267`, `p268`, `p270`, `p273`, `p274`, `p276`, `p277`, `p278`, `p279`, `p282`.
- **Validation Speakers** ($N=2$): `p286`, `p287`.
- **Test Speakers** ($N=2$): `p232`, `p257`.
- **Intersection**: $\emptyset$ (zero speaker overlap across all three splits).

---

## Artifacts Generated

All evaluation outputs are saved strictly under the git-ignored `results/enhancement_eval/` directory:
1. `metrics_per_utterance.csv`: Full spreadsheet containing all 824 utterances with filename, speaker, duration, clean/noisy/enhanced metrics, and deltas.
2. `summary.json`: Machine-readable summary statistics with overall, per-speaker, and per-SNR condition breakdowns.
3. `audio/*.wav`: Complete directory containing all 824 full-length enhanced audio files at 16 kHz PCM_16.
4. `plots/*_eval.png`: Representative publication-quality 2x3 grid visualizations contrasting waveforms and spectrograms.

---

## Analysis & Limitations

### 1. Did the Speech Enhancement Baseline Help?
**Yes, unequivocally**:
- Signal-to-noise ratio increased by **$+9.60\text{ dB}$** on average across 824 held-out test utterances.
- Scale-invariant signal-to-distortion ratio improved by **$+9.51\text{ dB}$**.
- Speech intelligibility (STOI) improved across all condition tiers, showing its strongest impact ($+0.024$) in adverse low-SNR conditions.

### 2. Notable Limitations & Future Directions
- **Phase Invariance**: The current baseline model enhances only the STFT magnitude spectrogram and re-synthesizes the waveform using the noisy input phase. In low-SNR environments ($< 0\text{ dB}$), noisy phase distortion limits maximum achievable speech quality. Future iterations could explore complex spectral mapping or neural vocoder phase recovery.
- **PESQ Availability on Windows**: PESQ evaluation requires pre-compiled C wheels or a Linux/WSL environment. The pipeline handles this gracefully without failure, but deploying PESQ in CI/benchmarking on Windows requires external containerization or build tools.
- **ASR Downstream Impact**: Although objective metrics show substantial acoustic improvement, whether this translates to downstream Word Error Rate (WER) reduction on Whisper or Wav2Vec2 must be empirically established in the upcoming integration milestone.
