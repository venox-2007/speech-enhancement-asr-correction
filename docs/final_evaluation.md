# Final Evaluation & Statistical Analysis: Speech Enhancement and Context-Aware ASR Correction

## Executive Summary

This report documents the final experimental evaluation and paired statistical analysis of the end-to-end pipeline for **Speech Enhancement and Context-Aware ASR Correction**. All evaluations were conducted on the complete official **824-utterance VoiceBank+DEMAND test set** ($6,211$ words; test speakers `p232` and `p257`) using official reference transcripts with standardized text normalization (lowercasing, punctuation removal, whitespace collapsing).

Five experimental configurations were evaluated under identical conditions:
1. **Clean Speech Benchmark**: Clean Audio $\to$ Whisper `base.en` (acoustic reference)
2. **Noisy Speech**: Unprocessed Noisy Audio $\to$ Whisper `base.en` (unprocessed baseline)
3. **U-Net Enhanced**: Noisy Audio $\to$ Trained U-Net $\to$ Whisper `base.en`
4. **Noisy + Improved Correction**: Noisy Audio $\to$ Whisper `base.en` $\to$ Improved Corrector
5. **Enhanced + Improved Correction**: Noisy Audio $\to$ Trained U-Net $\to$ Whisper `base.en` $\to$ Improved Corrector

> [!IMPORTANT]
> **Key Empirical Findings**:
> - **Clean Speech Reference**: Whisper `base.en` achieves **$4.65\%$ Corpus WER** ($5.05\%$ mean utterance WER; $289$ total errors across $6,211$ words) on clean test audio.
> - **The Enhancement-ASR Disconnect**: Spectral magnitude enhancement via U-Net produces substantial improvements in objective acoustic metrics ($+9.60\text{ dB}$ SNR, $+9.51\text{ dB}$ SI-SDR, $+0.0127$ STOI), but produces a **statistically significant degradation** in downstream Whisper ASR performance: WER increases from **$7.23\% \to 8.23\%$** ($+1.00$ percentage points; paired bootstrap $p = 0.0022$, Wilcoxon $p = 0.0066$).
> - **Post-ASR Correction Performance**: The improved context-aware correction pipeline yields an observed WER reduction of **$-0.11$ percentage points** ($7$ net words repaired out of $6,211$) on both noisy speech ($7.23\% \to 7.12\%$) and enhanced speech ($8.23\% \to 8.11\%$).
> - **Statistical Interpretation**:
>   - For corpus-level aggregate WER, the $10,000$-iteration paired bootstrap $95\%$ confidence interval excludes zero for both noisy ($[-0.22\%, -0.02\%]$, $p = 0.0356$) and enhanced ($[-0.22\%, -0.02\%]$, $p = 0.0382$) conditions.
>   - However, at the utterance level, the Wilcoxon signed-rank test does **not** reject the null hypothesis of zero median difference at $\alpha = 0.05$ ($p = 0.1542$ on noisy, $p = 0.0754$ on enhanced). Because $813$ out of $824$ utterances ($98.67\%$) remain unchanged, the effect is localized to a small subset ($11$ utterances modified: $9$ improved, $2$ worsened).

---

## 1. Consolidated Final Benchmark Table

All metrics were computed across all $824$ test utterances ($6,211$ words). Acoustic metrics (STOI, SI-SDR, SNR) were computed against matched clean speech.

| Condition | End-to-End Pipeline | Corpus WER | Mean Utt WER | S | D | I | Hits ($C$) | Total Errors ($E$) | STOI | SI-SDR (dB) | SNR (dB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean Speech** | Clean Audio $\to$ Whisper `base.en` | **4.65%** | 5.05% | 202 | 55 | 32 | 5,954 | 289 | 1.0000 | N/A | N/A |
| **Noisy Speech** | Noisy Audio $\to$ Whisper `base.en` | **7.23%** | 7.59% | 330 | 69 | 50 | 5,812 | 449 | 0.9211 | 8.45 | 8.45 |
| **U-Net Enhanced** | Noisy $\to$ U-Net $\to$ Whisper `base.en` | **8.23%** | 8.78% | 369 | 80 | 62 | 5,762 | 511 | 0.9338 | 17.96 | 18.05 |
| **Noisy + Improved Corr** | Noisy $\to$ Whisper $\to$ Improved Corrector | **7.12%** | 7.51% | 323 | 69 | 50 | 5,819 | 442 | 0.9211 | 8.45 | 8.45 |
| **Enhanced + Improved Corr** | Noisy $\to$ U-Net $\to$ Whisper $\to$ Improved Corrector | **8.11%** | 8.66% | 362 | 80 | 62 | 5,769 | 504 | 0.9338 | 17.96 | 18.05 |

*Definitions: $S$ = Substitutions, $D$ = Deletions, $I$ = Insertions, Hits ($C$) = Correct words, Total Errors ($E = S + D + I$). Corpus WER $= E / N_{\text{total}}$ where $N_{\text{total}} = 6,211$. Mean Utt WER $= \frac{1}{M} \sum_{i=1}^M \frac{e_i}{n_i}$ where $M = 824$.*

---

## 2. Paired Statistical Methodology and Results

### 2.1 Statistical Testing Framework

To evaluate whether differences between pipelines represent systematic shifts rather than sampling variability, two complementary paired statistical tests were conducted:

1. **Utterance-Level Paired Bootstrap for Corpus WER**:
   - Resamples paired utterances with replacement ($B = 10,000$ iterations, seed $42$).
   - In each replication $b$, the aggregate corpus WER delta is computed:
     $$\Delta^{(b)} = \frac{\sum_{i=1}^M e_{B, i}^{(b)}}{\sum_{i=1}^M n_i^{(b)}} - \frac{\sum_{i=1}^M e_{A, i}^{(b)}}{\sum_{i=1}^M n_i^{(b)}}$$
   - Percentile $95\%$ confidence interval: $[\text{percentile}_{2.5}(\Delta^{(b)}), \text{percentile}_{97.5}(\Delta^{(b)})]$.
   - Empirical two-sided $p$-value: $2 \times \min(P(\Delta^{(b)} \ge 0), P(\Delta^{(b)} \le 0))$.
   - *Interpretation*: Evaluates sampling variability of the overall corpus WER difference, weighting utterances proportionally to word count.

2. **Wilcoxon Signed-Rank Test on Utterance WER**:
   - Computes paired utterance error rate differences $d_i = \text{WER}_{B, i} - \text{WER}_{A, i}$.
   - Evaluates the non-parametric null hypothesis that the distribution of differences is symmetric about zero.
   - *Methodological Nuance*: When systems produce identical outputs on most utterances ($d_i = 0$), the Wilcoxon test ranks only the non-zero differences ($N_{\text{non-zero}}$). With small $N_{\text{non-zero}}$, power is limited.

### 2.2 Paired Comparison Results

| Comparison | Sample $\Delta\text{WER}$ | 95% Bootstrap CI | Bootstrap $p$-value | Bootstrap Sig. ($\alpha=0.05$) | Wilcoxon $W$ | Wilcoxon $p$-value | Wilcoxon Sig. ($\alpha=0.05$) | $N_{\text{non-zero}}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Noisy Raw vs Noisy Improved-Corr** | **$-0.11\%$** | $[-0.22\%, -0.02\%]$ | $p = 0.0356$ | **Yes** | $17.0$ | $p = 0.1542$ | **No** | 11 |
| **Enhanced Raw vs Enhanced Improved-Corr** | **$-0.11\%$** | $[-0.22\%, -0.02\%]$ | $p = 0.0382$ | **Yes** | $13.0$ | $p = 0.0754$ | **No** | 11 |
| **Noisy Raw vs Enhanced Raw** | **$+1.00\%$** | $[+0.35\%, +1.69\%]$ | $p = 0.0022$ | **Yes** | $2016.0$ | $p = 0.0066$ | **Yes** | 107 |
| **Noisy Raw vs Enhanced Improved-Corr** | **$+0.89\%$** | $[+0.22\%, +1.58\%]$ | $p = 0.0094$ | **Yes** | $2510.0$ | $p = 0.0213$ | **Yes** | 115 |

### 2.3 Synthesis of Statistical Evidence
- **Enhancement Degradation**: Both the paired bootstrap test ($p = 0.0022$) and the Wilcoxon signed-rank test ($p = 0.0066$, $N=107$) reject the null hypothesis at $\alpha = 0.01$. The evidence confirms that front-end U-Net magnitude enhancement degraded downstream Whisper `base.en` ASR on this test set.
- **Correction Stage Impact**: The bootstrap test on corpus WER rejects the null hypothesis at $\alpha = 0.05$ ($p = 0.0356$ and $p = 0.0382$), with the $95\%$ CI strictly below zero. However, the Wilcoxon test on per-utterance WER differences does not reach statistical significance ($p = 0.1542$ and $p = 0.0754$). This indicates that while the net word-level error reduction across the corpus is positive ($+7$ net words repaired), the intervention affects a small fraction of utterances ($11$ out of $824$), so a general shift across all utterances cannot be asserted.

---

## 3. Per-Utterance Outcome Distributions

| Comparison | Improved ($d_i < 0$) | Worsened ($d_i > 0$) | Unchanged ($d_i = 0$) | Mean $\Delta$ | Std $\Delta$ | Median $\Delta$ | Min $\Delta$ | Max $\Delta$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Noisy Raw vs Noisy Improved-Corr** | 9 (1.09%) | 2 (0.24%) | 813 (98.67%) | $-0.08\%$ | $1.48\%$ | $0.00\%$ | $-20.00\%$ | $+16.67\%$ |
| **Enhanced Raw vs Enhanced Improved-Corr** | 9 (1.09%) | 2 (0.24%) | 813 (98.67%) | $-0.12\%$ | $1.82\%$ | $0.00\%$ | $-33.33\%$ | $+14.29\%$ |
| **Noisy Raw vs Enhanced Raw** | 37 (4.49%) | 70 (8.50%) | 717 (87.01%) | $+1.19\%$ | $11.25\%$ | $0.00\%$ | $-80.00\%$ | $+100.00\%$ |
| **Noisy Raw vs Enhanced Improved-Corr** | 45 (5.46%) | 70 (8.50%) | 709 (86.04%) | $+1.06\%$ | $11.32\%$ | $0.00\%$ | $-80.00\%$ | $+100.00\%$ |

---

## 4. Scientific Discussion: Distinguishing Observations from Hypotheses

### 4.1 Experimentally Demonstrated Findings
1. **Acoustic Metrics vs Downstream Intelligibility**:
   The U-Net model successfully optimized spectral magnitude fidelity, producing an average gain of $+9.60\text{ dB}$ in SNR, $+9.51\text{ dB}$ in SI-SDR, and $+0.0127$ in STOI.
2. **ASR Performance Impact**:
   Despite these acoustic gains, Whisper `base.en` experienced an increase of $62$ word errors ($449 \to 511$, $+1.00$ percentage points in WER), with nearly double the number of utterances degrading ($70$) compared to improving ($37$).
3. **Correction Selectivity**:
   The improved corrector altered words in only $11$ of $824$ utterances, achieving $9$ beneficial corrections and $2$ degradations, substantially mitigating the overcorrection problem of the unconstrained baseline corrector (which degraded WER by $+0.80$ percentage points).

### 4.2 Hypotheses for the Enhancement-ASR Disconnect
The experimental results demonstrate *that* enhancement degrades Whisper `base.en` ASR, but do not isolate the individual causal mechanisms. Based on speech processing literature, we formulate the following plausible hypotheses:
- **Hypothesis 1 (Phase Mismatch and Non-Linear Synthesis Artifacts)**: The pipeline applies an enhanced magnitude spectrogram to the unenhanced, corrupted phase. Incongruence between magnitude and phase during iSTFT synthesis can produce non-linear spectral distortion and musical noise, which may disrupt learned acoustic features.
- **Hypothesis 2 (Acoustic Feature Distribution Shift)**: OpenAI Whisper was pre-trained on $680,000$ hours of diverse audio containing natural additive noise, rendering its convolutional frontend inherently robust to environmental noise. Processing audio through spectral subtraction or masking alters the distribution in ways unobserved during pre-training, potentially degrading encoder embeddings more than stationary additive noise.
- *Note*: These hypotheses are consistent with published observations in the speech enhancement literature, but testing them definitively would require controlled phase ablations (e.g., ideal phase synthesis) or ASR acoustic model fine-tuning.

---

## 5. Limitations

This study has several important limitations that define the scope of its conclusions:

1. **Single Enhancement Architecture**:
   The enhancement model evaluated is a single 2D convolutional U-Net baseline operating on magnitude STFT representations. Alternative architectures (e.g., complex-valued networks, Demucs, Conv-TasNet) were not evaluated.
2. **Single Downstream ASR Model**:
   ASR evaluation was restricted to OpenAI Whisper `base.en` with greedy decoding. Behavior on larger Whisper models (`small`, `medium`, `large-v3`) or acoustic models with distinct pre-training objectives (e.g., wav2vec 2.0, Conformer-CTC) was not examined.
3. **VoiceBank+DEMAND Dataset Scope**:
   Evaluation was conducted exclusively on the VoiceBank+DEMAND benchmark, which consists of read English speech synthetically mixed with selected noise recordings. Results cannot be assumed to generalize to spontaneous conversational speech, reverberant acoustics, or overlapping talkers.
4. **Noisy-Phase Reuse**:
   The enhancement pipeline modified only spectral magnitude while reusing the corrupted noisy phase for waveform reconstruction. Phase estimation or complex masking techniques were not employed.
5. **Magnitude-L1 Training Objective**:
   The U-Net was trained using an L1 loss on magnitude spectrograms. This objective penalizes absolute spectral amplitude deviations but does not incorporate perceptual metrics, phase information, or ASR-aligned loss functions.
6. **Correction Development Set Design**:
   Safeguard thresholds for the context-aware corrector were calibrated on a synthetic development set of $60$ controlled sentences. While this prevented tuning on the test set, optimal hyperparameters may vary for other acoustic distributions or domain vocabularies.
7. **Small Absolute Correction Gain**:
   The net error reduction achieved by the post-ASR corrector was $7$ words out of $6,211$ total words ($-0.11$ percentage points), leaving $98.67\%$ of test utterances unchanged.
8. **PESQ Metric Unavailable on Windows**:
   Due to the absence of MSVC C++ compiler tools in the host environment, the `pesq` package C extension could not be built. Objective acoustic evaluation was therefore restricted to STOI, SI-SDR, and SNR.
9. **No Claim of Generalization**:
   All findings reflect the specific models, hyperparameter configurations, and evaluation sets tested. No claims are made regarding generalization beyond these empirical conditions.

---

## 6. Final Research Conclusions

Addressing the primary research questions investigated in this project:

1. **Does speech enhancement as a front-end module improve downstream Whisper ASR accuracy on noisy speech?**
   **No**. Within the evaluated experimental setup (magnitude U-Net with noisy-phase iSTFT on VoiceBank+DEMAND), front-end speech enhancement significantly degraded downstream Whisper `base.en` accuracy, increasing WER from $7.23\%$ to $8.23\%$ ($+1.00$ percentage points, $p = 0.0022$). Objective signal-level improvements ($+9.60\text{ dB}$ SNR, $+9.51\text{ dB}$ SI-SDR) did not translate into improved recognition.
2. **Can a context-aware phonetic language-model corrector improve ASR outputs without acoustic model retraining?**
   **Yes, with modest effect**. When constrained by principled safeguards (score margin floor of $3.5$, phonetic similarity floor of $0.80$, retention prior of $1.0$), the post-ASR corrector produced a small, measured reduction of $-0.11$ percentage points in corpus WER on both noisy and enhanced speech ($7$ net words repaired, $9$ improved utterances vs $2$ worsened). This corpus WER delta was statistically distinguishable from zero under paired bootstrap resampling ($p < 0.05$), though not statistically significant at the utterance level under the Wilcoxon signed-rank test ($p > 0.05$).
3. **Does the full combined pipeline (Enhancement + Correction) outperform the original unprocessed noisy baseline?**
   **No**. The combined pipeline achieved a final WER of $8.11\%$, remaining significantly worse than the raw noisy baseline of $7.23\%$ ($+0.89$ percentage points, bootstrap $p = 0.0094$, Wilcoxon $p = 0.0213$). The post-ASR correction gain was insufficient to offset the degradation introduced by front-end spectral magnitude enhancement.

---

## 7. Reproducibility & Artifact Verification

- **Clean Baseline Cache**: `results/final_analysis/clean_asr_cache.json`
- **Statistical Summary JSON**: `results/final_analysis/final_evaluation_summary.json`
- **Analysis Implementation**: [`src/evaluation/statistical_analysis.py`](file:///c:/Users/yugag/OneDrive/Documents/Sem%205/DL%20Project/src/evaluation/statistical_analysis.py)
- **Unit Test Suite**: [`tests/test_statistical_analysis.py`](file:///c:/Users/yugag/OneDrive/Documents/Sem%205/DL%20Project/tests/test_statistical_analysis.py)
