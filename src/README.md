# Source Code

Run modules from the repository root, e.g. `python -m src.asr.baseline`.

Implemented:

- `preprocessing/audio_io.py` — audio loading, mono downmix, resampling
- `asr/whisper_asr.py` — Whisper ASR wrapper (`WhisperASR`)
- `asr/baseline.py` — ASR + WER command-line baseline
- `evaluation/wer.py` — WER with substitution/deletion/insertion counts
- `correction/pronunciation.py` — CMUdict pronunciation lookup
- `correction/phonetic_distance.py` — weighted phoneme edit distance
- `correction/candidates.py` — phonetic candidate generation (`PhoneticCandidateGenerator`)
- `utils/` — device selection, logging setup

Planned:

- `preprocessing/` — noise mixing, STFT/iSTFT
- `enhancement/` — CNN and U-Net speech enhancement models
- `correction/` — context-aware scoring and correct/keep decision
- `evaluation/` — speech-quality metrics (PESQ, STOI, SI-SDR)
