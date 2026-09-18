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
- `utils/` — device selection, logging setup

Planned:

- `preprocessing/` — noise mixing, STFT/iSTFT
- `enhancement/` — CNN and U-Net speech enhancement models
- `evaluation/` — speech-quality metrics (PESQ, STOI, SI-SDR)
