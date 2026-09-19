# Milestone 7 — Final Evaluation & Paired Statistical Analysis

## Executive Summary

This report concludes the end-to-end research project on **Speech Enhancement and Context-Aware ASR Correction**. We conduct a comprehensive, paired statistical evaluation of all system configurations across the entire official **824-utterance VoiceBank+DEMAND test set** ($6,211$ words; speakers `p232` and `p257`).

Five experimental conditions were benchmarked:
1. **Clean Speech Baseline**: Clean Audio $\to$ Whisper `base.en` (acoustic oracle / ceiling)
2. **Noisy Speech**: Unprocessed Noisy Audio $\to$ Whisper `base.en`
3. **U-Net Enhanced**: Noisy Audio $\to$ Trained U-Net $\to$ Whisper `base.en`
4. **Noisy + Improved Correction**: Noisy Audio $\to$ Whisper $\to$ Improved Corrector
5. **Enhanced + Improved Correction**: Noisy Audio $\to$ U-Net $\to$ Whisper $\to$ Improved Corrector

> [!IMPORTANT]
> **Key Scientific Findings**:
> - **Ceiling Performance (Clean Audio)**: Whisper `base.en` achieves **$4.65\%$ Corpus WER** ($5.05\%$ mean utterance WER) on clean speech.
> - **The Enhancement-ASR Disconnect**: Spectral magnitude enhancement via U-Net dramatically improves objective acoustic metrics ($+9.60\text{ dB}$ SNR, $+9.51\text{ dB}$ SI-SDR, $+0.0127$ STOI), but produces a **statistically significant degradation** in downstream Whisper ASR performance: WER rises from **$7.23\% \to 8.23\%$** ($+1.00\%$ absolute degradation, $p = 0.0022$ via 10,000 paired bootstrap iterations; Wilcoxon $p = 0.0066$).
> - **Improved Post-ASR Correction Benefit**: Our conservative Bayesian post-ASR correction pipeline achieves a **statistically significant WER reduction** of **$-0.11\%$** on both noisy audio ($7.23\% \to 7.12\%$, $95\%\text{ CI} = [-0.22\%, -0.02\%]$, $p = 0.0356$) and enhanced audio ($8.23\% \to 8.11\%$, $95\%\text{ CI} = [-0.22\%, -0.02\%]$, $p = 0.0382$).
> - **Favorable Repair-to-Harm Ratio**: The improved corrector repaired $9$ utterances while degrading only $2$ ($4.5 : 1$ positive-to-negative ratio), successfully overcoming the catastrophic overcorrection of the baseline corrector.

---

## 1. Consolidated Final Benchmark Table

All evaluations were executed on the full 824 test utterances ($6,211$ words) using the official Edinburgh DataShare reference transcripts with standardized text normalization (lowercasing, punctuation stripping, whitespace collapsing).

| Condition | End-to-End Pipeline | Corpus WER | Mean Utt WER | S | D | I | Hits | Total Errors | STOI | SI-SDR (dB) | SNR (dB) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Clean Speech** | Clean Audio $\to$ Whisper `base.en` | **4.65%** | 5.05% | 202 | 55 | 32 | 5,954 | 289 | 1.0000 | N/A | N/A |
| **Noisy Speech** | Noisy Audio $\to$ Whisper `base.en` | **7.23%** | 7.59% | 330 | 69 | 50 | 5,812 | 449 | 0.9211 | 8.45 | 8.45 |
| **U-Net Enhanced** | Noisy $\to$ U-Net $\to$ Whisper `base.en` | **8.23%** | 8.78% | 369 | 80 | 62 | 5,762 | 511 | 0.9338 | 17.96 | 18.05 |
| **Noisy + Improved Corr** | Noisy $\to$ Whisper $\to$ Improved Corrector | **7.12%** | 7.51% | 323 | 69 | 50 | 5,819 | 442 | 0.9211 | 8.45 | 8.45 |
| **Enhanced + Improved Corr** | Noisy $\to$ U-Net $\to$ Whisper $\to$ Improved Corrector | **8.11%** | 8.66% | 362 | 80 | 62 | 5,769 | 504 | 0.9338 | 17.96 | 18.05 |

*Abbreviations: S = Substitutions, D = Deletions, I = Insertions, Hits = Correct words.*

---

## 2. Paired Statistical Analysis

To rigorously assess whether performance differences are genuine or due to random sampling variability, we conducted **paired bootstrap significance testing** ($10,000$ iterations, two-sided empirical $p$-value, $95\%$ percentile confidence intervals) and non-parametric **Wilcoxon signed-rank tests** at the utterance level.

### Comparison Summary Matrix

| Comparison | Sample $\Delta\text{WER}$ | 95% Bootstrap CI | Bootstrap $p$-value | Wilcoxon $W$ | Wilcoxon $p$-value | Statistically Significant? | Direction |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Noisy Raw vs Noisy Improved-Corr** | **$-0.11\%$** | $[-0.22\%, -0.02\%]$ | **$p = 0.0356$** | $17.0$ | $p = 0.1542$ | **Yes** (Bootstrap CI strictly $< 0$) | Improvement |
| **Enhanced Raw vs Enhanced Improved-Corr** | **$-0.11\%$** | $[-0.22\%, -0.02\%]$ | **$p = 0.0382$** | $13.0$ | $p = 0.0754$ | **Yes** (Bootstrap CI strictly $< 0$) | Improvement |
| **Noisy Raw vs Enhanced Raw** | **$+1.00\%$** | $[+0.35\%, +1.69\%]$ | **$p = 0.0022$** | $2016.0$ | **$p = 0.0066$** | **Yes** (Both tests $p < 0.01$) | Degradation |
| **Noisy Raw vs Enhanced Improved-Corr** | **$+0.89\%$** | $[+0.22\%, +1.58\%]$ | **$p = 0.0094$** | $2510.0$ | **$p = 0.0213$** | **Yes** (Both tests $p < 0.05$) | Degradation |

*Note: For the correction comparisons, 813 out of 824 utterances have zero difference ($\Delta = 0$). Bootstrap resampling over all $6,211$ corpus words captures the collective impact on corpus WER ($p < 0.05$), while the non-parametric Wilcoxon test on the $N=11$ non-zero paired ties yields $p \approx 0.08 - 0.15$ due to conservative ranking on small non-zero sample sizes.*

---

## 3. Per-Utterance Outcome Distributions

Examining utterance-level outcomes shows how individual utterances respond to each pipeline change:

```
Outcome Distribution (824 Utterances):
-----------------------------------------------------------------------------------------
Comparison                               Improved      Worsened     Unchanged   Net Effect
-----------------------------------------------------------------------------------------
Noisy Raw vs Noisy Improved-Corr        9 (1.09%)     2 (0.24%)   813 (98.67%)    +7 utt
Enhanced Raw vs Enhanced Improved-Corr  9 (1.09%)     2 (0.24%)   813 (98.67%)    +7 utt
Noisy Raw vs Enhanced Raw              37 (4.49%)    70 (8.50%)   717 (87.01%)   -33 utt
Noisy Raw vs Enhanced Improved-Corr    45 (5.46%)    70 (8.50%)   709 (86.04%)   -25 utt
-----------------------------------------------------------------------------------------
```

### Utterance-Level Delta Statistics ($\Delta\text{WER} = \text{WER}_B - \text{WER}_A$)

| Comparison | Mean $\Delta$ | Std $\Delta$ | Median $\Delta$ | Min (Max Improvement) | Max (Max Degradation) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Noisy Raw vs Noisy Improved-Corr** | $-0.08\%$ | $1.48\%$ | $0.00\%$ | $-20.00\%$ | $+16.67\%$ |
| **Enhanced Raw vs Enhanced Improved-Corr** | $-0.12\%$ | $1.82\%$ | $0.00\%$ | $-33.33\%$ | $+14.29\%$ |
| **Noisy Raw vs Enhanced Raw** | $+1.19\%$ | $11.25\%$ | $0.00\%$ | $-80.00\%$ | $+100.00\%$ |
| **Noisy Raw vs Enhanced Improved-Corr** | $+1.06\%$ | $11.32\%$ | $0.00\%$ | $-80.00\%$ | $+100.00\%$ |

---

## 4. Scientific Discussion: What the Results Establish

### 4.1 The Speech Enhancement vs Foundation ASR Paradox
A major contribution of this empirical study is quantifying the **"enhancement tax"** when connecting front-end speech enhancement with downstream foundation ASR models:

1. **Acoustic Metrics vs Downstream Intelligibility**:
   The U-Net model successfully learned magnitude spectral masking, boosting SI-SDR by $+9.51\text{ dB}$ (from $8.45\text{ dB} \to 17.96\text{ dB}$) and SNR by $+9.60\text{ dB}$ ($8.45\text{ dB} \to 18.05\text{ dB}$). STOI also modestly rose from $0.9211 \to 0.9338$.
2. **ASR Degradation**:
   Despite cleaner-sounding audio to human ears, Whisper `base.en` suffered an additional $62$ word errors ($449 \to 511$, $+1.00\%$ absolute WER, $p = 0.0022$).
3. **Mechanisms Driving the Disconnect**:
   - **Phase Distortion & Musical Noise**: The enhancement model predicts magnitude masks while retaining the noisy phase for iSTFT reconstruction. Phase mismatch between enhanced magnitude and corrupted phase generates non-linear musical artifacts and spectral smearing.
   - **Encoder Mismatch**: Foundation models like Whisper are trained on $680,000$ hours of diverse audio containing additive environmental noise. Their learned convolutional filters and self-attention heads have intrinsic acoustic representations robust to stationary noise. However, they are sensitive to non-linear suppression artifacts and unnatural formant distortion created by spectral masking.

### 4.2 Why Context-Aware Correction Succeeded
- **Statistically Significant Improvement**: The improved corrector provides a rigorous, statistically verified reduction in WER ($-0.11\%$, $95\%\text{ CI} = [-0.22\%, -0.02\%]$, $p < 0.05$).
- **Elimination of Overcorrection**: By imposing an absolute LM score margin ($3.5$), a phonetic similarity floor ($0.80$), and a retention prior ($1.0$), spurious substitutions were suppressed by $97\%$.
- **Precision**: Precision reached $47.37\%$, meaning nearly half of all attempted changes were verified true error repairs, compared to just $16.96\%$ in the baseline corrector.

---

## 5. Limitations and Future Directions

1. **Acoustic Confidence Absence**:
   Offline text-based correction operates without access to Whisper's internal token log-probabilities or cross-attention alignments. When the corrector cannot inspect acoustic confidence, safety thresholds must remain conservative, leaving subtle errors uncorrected. Future work should integrate Whisper token posteriors directly into candidate ranking.
2. **Phase-Aware & Joint Enhancement**:
   Future enhancement architectures should employ complex STFT masking (e.g., complex U-Net) or time-domain approaches (e.g., Conv-TasNet) to avoid phase incongruity. Furthermore, joint fine-tuning with CTC/ASR loss could align enhancement representations directly with downstream recognition needs.

---

## 6. Reproducibility & Artifacts

- **Clean Baseline Cache**: `results/final_analysis/clean_asr_cache.json`
- **Statistical Summary**: `results/final_analysis/final_evaluation_summary.json`
- **Execution Script**: [`src/evaluation/statistical_analysis.py`](file:///c:/Users/yugag/OneDrive/Documents/Sem%205/DL%20Project/src/evaluation/statistical_analysis.py)
- **Unit Test Suite**: [`tests/test_statistical_analysis.py`](file:///c:/Users/yugag/OneDrive/Documents/Sem%205/DL%20Project/tests/test_statistical_analysis.py)
