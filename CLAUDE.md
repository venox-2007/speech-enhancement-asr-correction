# Claude Code Project Instructions

## Project

This repository implements a student research project on robust speech enhancement and context-aware post-ASR error correction.

## Research objective

Study whether deep-learning-based speech enhancement improves ASR under diverse noise conditions and whether phonetic/context-aware post-processing can reduce residual transcription errors.

## Engineering principles

- Keep the code modular. Do not put the entire pipeline in one script.
- Prefer small, testable functions and clear interfaces between preprocessing, enhancement, ASR, correction, and evaluation.
- Keep research experiments reproducible through configuration files and saved metadata.
- Do not hard-code machine-specific paths.
- Use GPU when available and fall back to CPU cleanly.
- Avoid unnecessary dependencies.
- Never commit datasets, large audio files, model checkpoints, credentials, or API keys.

## Research integrity

- Never fabricate experimental results, metrics, datasets, citations, or model capabilities.
- Clearly distinguish measured results from expected outcomes.
- Preserve the actual experimental conditions when reporting results.
- Do not claim novelty merely because a technique has not yet been implemented in this repository.
- Record enough metadata to reproduce each experiment.

## Planned components

- Audio preprocessing and noise mixing
- STFT / iSTFT processing
- CNN enhancement baseline
- U-Net enhancement model
- ASR transcription
- Phonetic candidate generation
- Context-aware correction
- WER and speech-enhancement evaluation
- Streamlit demonstration interface

## Development order

1. Project setup and audio I/O
2. ASR baseline and WER
3. Phonetic candidate generation
4. Context-aware correction baseline
5. Correction evaluation and overcorrection analysis
6. CNN enhancement
7. U-Net enhancement
8. Combined end-to-end experiments
9. Streamlit research/demo UI

## Git workflow

- Make focused commits after each working milestone.
- Use descriptive commit messages.
- Keep `main` stable.
- Use feature branches for substantial changes when practical.
