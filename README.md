# Speech Enhancement & Context-Aware ASR Correction

An end-to-end deep learning and speech processing framework investigating the interaction between front-end spectral speech enhancement, downstream foundation automatic speech recognition (Whisper), and post-ASR context-aware phonetic correction.

---

## Table of Contents
- [Project Objective & Research Questions](#project-objective--research-questions)
- [System Architecture & Pipeline](#system-architecture--pipeline)
- [Consolidated Experimental Benchmark](#consolidated-experimental-benchmark)
- [Key Scientific Findings](#key-scientific-findings)
- [Interactive Streamlit Demo](#interactive-streamlit-demo)
- [Installation & Setup](#installation--setup)
- [Running Tests](#running-tests)
- [Documented Limitations](#documented-limitations)
- [Repository Structure](#repository-structure)

---

## Project Objective & Research Questions

This project empirically investigates three foundational questions at the intersection of deep learning speech enhancement and foundation speech recognition models:

1. **The Enhancement-ASR Question**: Does front-end spectral magnitude enhancement improve downstream transcription accuracy for a modern foundation ASR model (OpenAI Whisper `base.en`) on noisy speech?
2. **The Post-ASR Correction Question**: Can a context-aware phonetic language model corrector repair residual ASR errors without acoustic retraining, while avoiding catastrophic overcorrection?
3. **The End-to-End Pipeline Question**: Does a combined multi-stage system (Enhancement $\to$ Whisper $\to$ Context Correction) outperform an unprocessed noisy ASR baseline?

---

## System Architecture & Pipeline

```text
                               +---------------------------------------+
                               |        Input Audio Waveform y(t)      |
                               +---------------------------------------+
                                                   |
                        +--------------------------+--------------------------+
                        |                                                     |
             [ Branch A: Raw Baseline ]                            [ Branch B: U-Net Enhanced ]
                        |                                                     |
                        |                                           STFT (n_fft=512, hop=256)
                        |                                                     |
                        |                                           2D Convolutional U-Net
                        |                                           (Magnitude Masking, L1 loss)
                        |                                                     |
                        |                                           iSTFT Synthesis (input phase)
                        |                                                     |
                        |                                           [ Enhanced Audio s_hat(t) ]
                        |                                                     |
               OpenAI Whisper base.en                                OpenAI Whisper base.en
                        |                                                     |
               [ Raw Transcript ]                                    [ Enhanced Transcript ]
                        |                                                     |
          Context-Aware Post-ASR Correction                     Context-Aware Post-ASR Correction
          (CMUdict + GPT-2 + Safeguards)                        (CMUdict + GPT-2 + Safeguards)
                        |                                                     |
          +----------------------------+                        +-------------------------------+
          | Final Corrected Transcript |                        | Final Corrected Transcript    |
          +----------------------------+                        +-------------------------------+
```

> **Architectural Domain Separation**: The 2D U-Net operates strictly in the **audio domain** on STFT magnitude spectrograms to produce cleaned audio waveforms *before* Whisper ASR ingestion. It does not process or modify text. The phonetic/GPT-2 correction module operates strictly in the **text domain** as a *post-ASR* hypothesis refinement stage.

---

## Consolidated Experimental Benchmark

All configurations were evaluated on the complete official **824-utterance VoiceBank+DEMAND test set** ($6,211$ words; held-out speakers `p232` and `p257`) against official Edinburgh DataShare reference transcripts under standardized text normalization:

| Condition | End-to-End Pipeline | Corpus WER | Mean Utt WER | S | D | I | Hits ($C$) | Total Errors ($E$) | STOI | SI-SDR (dB) | SNR (dB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean Speech** | Clean Audio $\to$ Whisper `base.en` | **4.65%** | 5.05% | 202 | 55 | 32 | 5,954 | 289 | 1.0000 | N/A | N/A |
| **Noisy Speech** | Noisy Audio $\to$ Whisper `base.en` | **7.23%** | 7.59% | 330 | 69 | 50 | 5,812 | 449 | 0.9211 | 8.45 | 8.45 |
| **U-Net Enhanced** | Noisy $\to$ U-Net $\to$ Whisper `base.en` | **8.23%** | 8.78% | 369 | 80 | 62 | 5,762 | 511 | 0.9338 | 17.96 | 18.05 |
| **Noisy + Improved Corr** | Noisy $\to$ Whisper $\to$ Improved Corrector | **7.12%** | 7.51% | 323 | 69 | 50 | 5,819 | 442 | 0.9211 | 8.45 | 8.45 |
| **Enhanced + Improved Corr** | Noisy $\to$ U-Net $\to$ Whisper $\to$ Improved Corrector | **8.11%** | 8.66% | 362 | 80 | 62 | 5,769 | 504 | 0.9338 | 17.96 | 18.05 |

*Definitions: $S$ = Substitutions, $D$ = Deletions, $I$ = Insertions, Hits ($C$) = Correct words, Total Errors ($E = S + D + I$). Corpus WER $= E / 6,211$.*

### Paired Statistical Significance Analysis ($10,000$ Bootstrap Replications & Wilcoxon Signed-Rank)

| Comparison | Sample $\Delta\text{WER}$ | 95% Bootstrap CI | Bootstrap $p$-value | Bootstrap Sig. | Wilcoxon $p$-value | Wilcoxon Sig. | Non-Zero Pairs |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Noisy Raw vs Noisy Improved-Corr** | **$-0.11\%$** | $[-0.22\%, -0.02\%]$ | $p = 0.0356$ | **Yes** | $p = 0.1542$ | **No** | $11$ ($9$ imp / $2$ wor) |
| **Enhanced Raw vs Enhanced Improved-Corr** | **$-0.11\%$** | $[-0.22\%, -0.02\%]$ | $p = 0.0382$ | **Yes** | $p = 0.0754$ | **No** | $11$ ($9$ imp / $2$ wor) |
| **Noisy Raw vs Enhanced Raw** | **$+1.00\%$** | $[+0.35\%, +1.69\%]$ | $p = 0.0022$ | **Yes** | $p = 0.0066$ | **Yes** | $107$ ($37$ imp / $70$ wor) |
| **Noisy Raw vs Enhanced Improved-Corr** | **$+0.89\%$** | $[+0.22\%, +1.58\%]$ | $p = 0.0094$ | **Yes** | $p = 0.0213$ | **Yes** | $115$ ($45$ imp / $70$ wor) |

---

## Key Scientific Findings

1. **The Enhancement-ASR Disconnect**:
   U-Net magnitude enhancement delivered substantial signal fidelity gains ($+9.60\text{ dB}$ SNR, $+9.51\text{ dB}$ SI-SDR, $+0.0127$ STOI), yet caused a **statistically significant degradation** in downstream Whisper ASR accuracy ($+1.00$ percentage points in WER, $p = 0.0022$).
   - *Plausible Hypotheses*: Reusing corrupted noisy phase during iSTFT synthesis introduces non-linear phase distortions and musical noise artifacts. In addition, foundation models pre-trained on natural audio distributions are sensitive to non-linear spectral suppression artifacts that distort speech formants.
2. **Mitigating Post-ASR Overcorrection**:
   Unconstrained LM rescoring severely degraded WER ($+0.80$ percentage points) by substituting colloquial n-grams for correct words. Introducing Bayesian-inspired constraints (absolute score margin $\ge 3.5$, phonetic similarity floor $\ge 0.80$, retention prior $\beta = 1.0$) slashed overcorrections by $97\%$ and produced a measured **$-0.11$ percentage point error reduction** ($7$ net words repaired) across the corpus.
3. **End-to-End Pipeline Reality**:
   The full pipeline (Enhancement + Correction) reached $8.11\%$ WER, remaining significantly worse than the raw noisy baseline of $7.23\%$ ($p = 0.0094$). Front-end enhancement degradation could not be overcome by text-level post-processing alone.

---

## Interactive Streamlit Demo

A complete, production-grade Streamlit application is provided for demonstration and analysis.

### Launching the Web App:
```bash
streamlit run app.py
```

### Key UI Features:
- **Audio Ingestion & Live Recording**:
  - Upload custom audio files (`.wav`, `.mp3`, `.flac`, `.ogg`).
  - Select from curated VoiceBank+DEMAND benchmark test utterances.
  - **Live Microphone Recording Mode**: Record directly in the browser via native `st.audio_input` with instant playback, enhancement, transcription, and correction.
- **Audio Comparison**: Interactive audio players for original noisy speech and U-Net enhanced speech.
- **Visual Signal Analytics**: Dynamic side-by-side waveform and spectrogram visualizers.
- **Stage-by-Stage Latency Breakdown**: Tracks execution time across preprocessing, U-Net enhancement, Whisper ASR, and context-aware correction.
- **Word Correction Inspection**: Visual display of modified words with replacement confidence, phonetic similarity, and decision justifications.
- **Benchmark & Research Dashboard**: Interactive consolidated results table, paired statistical test summaries, and documented research conclusions.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.11
- NVIDIA GPU with $\ge 6\text{ GB}$ VRAM (e.g., RTX 3050) recommended; CPU fully supported.

### 2. Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/venox-2007/speech-enhancement-asr-correction.git
cd speech-enhancement-asr-correction

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install PyTorch matching your hardware (CUDA 12.8 example)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install project dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## Running Tests

The test suite contains **248 automated unit and integration tests** verifying audio preprocessing, dataset integrity, STFT/iSTFT math, U-Net architecture, Whisper ASR, phonetic candidate search, context scoring, statistical bootstrap tests, and demo services:

```bash
pytest
```


---

## Documented Limitations

1. **Single Enhancement Architecture**: Evaluated on a 2D convolutional U-Net baseline operating on magnitude STFT representations.
2. **Single Downstream ASR Model**: Restricted to OpenAI Whisper `base.en` with greedy decoding.
3. **Dataset Scope**: Evaluated exclusively on VoiceBank+DEMAND synthetically mixed read English speech.
4. **Noisy-Phase Reuse**: Magnitude-only enhancement without phase estimation during waveform synthesis.
5. **Magnitude-L1 Objective**: Optimized for spectral amplitude distance rather than perceptual or ASR-aligned loss functions.
6. **Development Set Size**: Safeguard thresholds calibrated on a 60-utterance synthetic development set.
7. **Small Absolute Correction Gain**: Net reduction of 7 words out of 6,211 words (-0.11 percentage points), leaving 98.67% of utterances unchanged.
8. **PESQ Unavailable on Windows**: Host environment lacked MSVC C++ build tools for the `pesq` C extension.
9. **No Claim of Generalization**: Findings reflect the specific models and evaluation sets tested.

---

## Repository Structure

```text
speech-enhancement-asr-correction/
├── app.py                             # Interactive Streamlit application
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Development & test dependencies
├── data/                              # Dataset storage (raw & 16kHz processed)
├── docs/                              # Milestone technical reports
│   ├── final_evaluation.md            # Final scientific benchmark & audit
│   ├── improved_correction_eval.md    # Post-ASR correction diagnosis & tuning
│   ├── asr_enhancement_eval.md        # Downstream Whisper evaluation
│   └── enhancement_eval.md            # Objective acoustic evaluation
├── results/                           # Ignored experimental logs, models & caches
│   ├── enhancement_train/             # Trained U-Net best checkpoint
│   └── final_analysis/                # Consolidated statistical analysis JSONs
├── src/
│   ├── asr/                           # Whisper ASR baseline and wrappers
│   ├── correction/                    # Phonetic search, GPT-2 scoring, pipeline
│   ├── demo/                          # Streamlit pipeline runner service
│   ├── enhancement/                   # 2D U-Net, dataset loaders, visualization
│   ├── evaluation/                    # WER, STOI, SI-SDR, paired bootstrap
│   └── preprocessing/                 # Audio I/O, resampling, STFT/iSTFT
└── tests/                             # Complete test suite (243 unit tests)
```
