# Downstream Whisper ASR Evaluation: Speech Enhancement Impact

## Executive Summary

This milestone evaluated the impact of the trained U-Net speech enhancement model on downstream Automatic Speech Recognition (ASR) performance. We benchmarked OpenAI's **Whisper `base.en`** across all **824 held-out VoiceBank+DEMAND test utterances** comparing two conditions:
1. **Original noisy audio** (`noisy_testset_wav`)
2. **U-Net enhanced audio** (reconstructed via STFT $\to$ U-Net $\to$ iSTFT from `results/enhancement_train/checkpoint_best.pt`)

> [!IMPORTANT]
> **Key Empirical Finding**:
> Although the U-Net baseline achieved major objective acoustic improvements ($+9.60\text{ dB}$ SNR, $+9.51\text{ dB}$ SI-SDR, $+0.0127$ STOI), it **did not improve downstream Whisper ASR performance**.
> - **Noisy Corpus WER**: **7.23%** (449 total errors across 6,211 reference words)
> - **Enhanced Corpus WER**: **8.23%** (511 total errors across 6,211 reference words)
> - **Net Impact**: **$+1.00\%$ absolute degradation** in Word Error Rate (+13.8% relative error increase).

---

## Experimental Setup & Methodology

### 1. ASR Engine & Configuration
- **Model**: OpenAI Whisper `base.en` (74M parameters, English-only acoustic & language model).
- **Decoding Strategy**: Deterministic greedy decoding (`temperature=0.0`, `language="en"`).
- **Audio Preprocessing**: Standard 16 kHz mono float32 input fed directly into Whisper's 80-channel log-Mel spectrogram feature extractor.
- **Hardware Acceleration**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA).

### 2. Test Dataset & Reference Transcripts
- **Corpus**: Complete VoiceBank+DEMAND test partition ($N=824$ utterances, $6,211$ total reference words).
- **Held-Out Speakers**: `p232` (male, 393 utterances) and `p257` (female, 431 utterances).
- **Zero-Leakage Assurance**: Confirmed that test speakers `p232` and `p257` are strictly disjoint from all 26 training speakers and both validation speakers (`p286`, `p287`).
- **Ground Truth Transcripts**: Acquired from the official Edinburgh DataShare release (`testset_txt.zip`, handle `10283/2791`), matching every `.wav` 1:1.

### 3. Normalization & Scoring
- Text normalization was applied identically to references and hypotheses using standard `jiwer` transforms (lowercasing, punctuation stripping, whitespace normalization).
- Metric: Corpus-level Word Error Rate:
  $$\text{WER}_{\text{corpus}} = \frac{S + D + I}{N}$$
  where $S$ = Substitutions, $D$ = Deletions, $I$ = Insertions, and $N$ = Total reference words ($S + D + C$, where $C$ = Correct words / hits).

---

## Empirical Benchmark Results

### 1. Overall Test Set Performance (824 Utterances, 6,211 Words)

| Metric | Noisy Speech (Baseline) | U-Net Enhanced Speech | Absolute Delta ($\Delta$) | Relative Change |
|:-------|:-----------------------:|:---------------------:|:-------------------------:|:---------------:|
| **Corpus WER** | **7.23%** | **8.23%** | **+1.00%** | +13.8% errors |
| **Mean Per-Utterance WER** | **$7.59\% \pm 16.57\%$** | **$8.78\% \pm 18.83\%$** | **+1.19%** | +15.7% |
| **Substitutions ($S$)** | 330 | 369 | +39 | +11.8% |
| **Deletions ($D$)** | 69 | 80 | +11 | +15.9% |
| **Insertions ($I$)** | 50 | 62 | +12 | +24.0% |
| **Total Errors ($S+D+I$)** | 449 | 511 | +62 | +13.8% |
| **Correct Words ($C$)** | 5,812 | 5,762 | -50 | -0.86% |
| **Total Reference Words ($N$)** | 6,211 | 6,211 | - | - |

- **Execution Runtime**: 293.64 seconds (~4.89 minutes total for 1,648 transcriptions, ~2.8 utterances/second).
- **Cached Re-execution**: 0.31 seconds (>2,600 utterances/second via `asr_cache.json`).

---

### 2. Breakdown by Speaker

| Speaker | Sex | Utterances | Words | Noisy WER | Enhanced WER | $\Delta$ WER | Noisy Hits ($C$) | Enhanced Hits ($C$) | $\Delta$ Hits |
|:-------:|:---:|:----------:|:-----:|:---------:|:------------:|:------------:|:----------------:|:-------------------:|:-------------:|
| **p232** | Male | 393 | 2,991 | **5.18%** | 6.49% | **+1.30%** | 2,851 | 2,823 | -28 |
| **p257** | Female | 431 | 3,220 | **9.13%** | 9.84% | **+0.71%** | 2,961 | 2,939 | -22 |

Both male and female speakers exhibit higher error rates after enhancement, with speaker `p232` experiencing a slightly larger relative increase in substitutions.

---

### 3. Breakdown by Input SNR Condition Tier

| Input SNR Tier | Utterances | Words | Noisy WER | Enhanced WER | $\Delta$ WER | Noisy Errors | Enhanced Errors | $\Delta$ Errors |
|:---------------|:----------:|:-----:|:---------:|:------------:|:------------:|:------------:|:---------------:|:---------------:|
| **$< 5$ dB (Low SNR)** | 218 | 1,602 | **12.11%** | 15.11% | **+3.00%** | 194 | 242 | +48 |
| **$5 - 10$ dB (Medium-Low)** | 209 | 1,595 | **6.77%** | 7.59% | **+0.82%** | 108 | 121 | +13 |
| **$10 - 15$ dB (Medium-High)** | 206 | 1,535 | **4.50%** | 4.63% | **+0.13%** | 69 | 71 | +2 |
| **$\ge 15$ dB (High SNR)** | 191 | 1,479 | 5.27% | **5.21%** | **-0.07%** | 78 | 77 | -1 |

---

## Research Analysis: Why Did Enhancement Degrade Downstream ASR?

This result illustrates a well-documented phenomenon in modern speech processing research: **the disconnect between human-centric/signal-level speech enhancement and downstream foundation ASR models**.

### 1. Signal-Level Optimization vs. Phonetic Information Preservation
- The U-Net was trained with a **spectral magnitude $L_1$ objective** ($\mathcal{L}_{\text{mag}} = \| |\hat{S}| - |S| \|_1$).
- Magnitude masking prioritizes large spectral energy suppression to minimize total pixel error in the time-frequency plane. Consequently, quiet acoustic cues—such as unvoiced fricatives (`/s/`, `/f/`, `/th/`), burst releases in stop consonants (`/p/`, `/t/`, `/k/`), and formant transitions—are often over-attenuated or smoothed away.
- While this sounds cleaner to human listeners and yields substantial gains in SNR ($+9.6\text{ dB}$) and SI-SDR ($+9.5\text{ dB}$), it removes critical acoustic landmarks that Whisper's transformer encoder relies on for phoneme distinction.

### 2. Pre-Training Robustness of Whisper
- Whisper was pre-trained on **680,000 hours of diverse weakly supervised web audio**, which already contains substantial real-world additive noise, reverberation, background music, and varying recording conditions.
- As a result, Whisper's multi-head self-attention layers have learned internal invariant representations for speech corrupted by standard environmental noise.
- However, Whisper was **not trained on artificial non-linear artifacts** produced by neural magnitude masking (e.g. musical noise, spectral holes, and unaligned noisy phase). When presented with enhanced audio, the model encounters out-of-distribution acoustic distortions, leading to increased substitutions ($+39$) and deletions ($+11$).

### 3. Severity Correlates with Masking Aggressiveness
- In the **$< 5$ dB** regime, where the U-Net applied the most aggressive suppression ($\Delta\text{SNR} = +13.37\text{ dB}$), the WER penalty was highest: **$+3.00\%$** (from $12.11\%$ to $15.11\%$).
- Conversely, in the **$\ge 15$ dB** regime, where the input was already clean and minimal attenuation was applied, WER was effectively neutral (**$-0.07\%$**, $5.27\% \to 5.21\%$).

---

## Implications for the Multi-Stage Architecture

These empirical findings provide strong scientific justification for the architecture of our overall project:
1. **Frontend Enhancement Cannot Replace Downstream Error Correction**: Naive speech enhancement as a frontend to robust ASR models does not guarantee lower WER and can often harm transcription accuracy.
2. **The Critical Role of Phonetic Correction**: Rather than aggressively manipulating raw acoustic waveforms before ASR, post-ASR phonetic correction (via candidate generation, CMUdict phonetic distances, and language model contextual rescoring) operates directly in linguistic/phonetic space where ASR confidence and semantic likelihood can be modeled safely without introducing acoustic artifacts.
3. **Future Hybrid Opportunities**: Joint fine-tuning or feature-level enhancement (rather than waveform iSTFT reconstruction) could mitigate the acoustic mismatch in future iterations.

---

## Artifacts Generated

All ASR evaluation outputs are saved strictly under the git-ignored `results/asr_enhancement_eval/` directory:
- `results_per_utterance.csv`: Detailed 824-row CSV with references, noisy/enhanced hypotheses, per-utterance edit counts, WERs, SNRs, and tier labels.
- `summary.json`: Machine-readable summary statistics (overall, by speaker, and by SNR tier).
- `asr_cache.json`: Persistent transcription cache enabling instantaneous resume.
