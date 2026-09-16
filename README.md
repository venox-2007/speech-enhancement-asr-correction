# Speech Enhancement + ASR Error Correction

Deep learning-based speech enhancement with context-aware phonetic post-ASR error correction under diverse noise conditions.

## Project objective

This project studies an end-to-end pipeline that:

1. Adds controlled noise to clean speech.
2. Enhances noisy speech using deep-learning models.
3. Transcribes speech using an automatic speech recognition (ASR) model.
4. Detects potentially incorrect ASR words using phonetic similarity.
5. Uses sentence context to decide whether a candidate correction is appropriate.
6. Measures the effect using speech-quality metrics and Word Error Rate (WER).

## Planned pipeline

```text
Clean Speech + Noise
        ↓
    Noisy Speech
        ↓
   CNN / U-Net
        ↓
  Enhanced Speech
        ↓
       ASR
        ↓
 Raw Transcription
        ↓
Phonetic Candidate Generation
        ↓
 Context-Aware Scoring
        ↓
 Correct / Keep Decision
        ↓
Corrected Transcription
        ↓
        WER
```

## Main research questions

- How robust are speech-enhancement models across different noise types and SNR levels?
- Does speech enhancement improve downstream ASR performance?
- Can phonetic similarity and contextual information correct residual ASR errors?
- When does post-ASR correction help, and when does it cause overcorrection?

## Planned stack

- Python 3.11
- PyTorch
- Torchaudio / Librosa
- Streamlit
- Whisper or another ASR model
- PESQ, STOI, SI-SDR
- JiWER for WER
- Git + GitHub

## Status

Early project setup. Experiments and implementations will be added incrementally.
