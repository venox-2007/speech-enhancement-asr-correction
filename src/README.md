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
- `correction/lm_scorer.py` — sentence log-probability with a local causal LM (GPT-2)
- `correction/context_scorer.py` — context-aware candidate scoring and correct/keep decision
- `correction/pipeline.py` — sentence-level correction pipeline (`CorrectionPipeline`)
- `correction/evaluate_corrections.py` — CLI demo with hand-crafted transcript pairs
- `evaluation/correction_eval.py` — correction evaluation with overcorrection tracking
- `preprocessing/stft.py` — STFT/iSTFT with configurable parameters
- `preprocessing/segments.py` — fixed-length segment extraction for training
- `preprocessing/noise_mix.py` — noise mixing at target SNR
- `preprocessing/resample_dataset.py` — offline 16 kHz dataset resampling utility
- `enhancement/dataset.py` — VoiceBank+DEMAND PyTorch dataset with speaker-based splits
- `enhancement/viz.py` — spectrogram plotting and dataset statistics
- `utils/` — device selection, logging setup

Planned:

- `enhancement/` — CNN and U-Net speech enhancement models
- `evaluation/` — speech-quality metrics (PESQ, STOI, SI-SDR)
