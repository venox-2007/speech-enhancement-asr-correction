# Milestone 1 — Whisper ASR Baseline and WER

```text
Audio file → Whisper → Transcript → WER against a reference transcript
```

## What the baseline measures

This baseline is the reference point for the rest of the project. It measures how
accurately an off-the-shelf Whisper model transcribes a given audio file,
**with no speech enhancement and no post-ASR correction**. Later milestones will
compare against it:

- clean speech → ASR (best-case WER),
- noisy speech → ASR (how much noise hurts),
- enhanced speech (CNN / U-Net) → ASR (whether enhancement helps ASR),
- enhanced speech → ASR → correction (whether correction reduces the remaining errors, and how often it overcorrects).

The comparisons are only meaningful if the model, decoding settings and
text normalization stay the same across conditions. Each run's JSON output
records these settings.

## Environment setup (Windows, Python 3.11)

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate            # Git Bash: source .venv/Scripts/activate
python -m pip install --upgrade pip
```

Install PyTorch **first**, choosing the build for your hardware
(see <https://pytorch.org/get-started/locally/>):

```bash
# NVIDIA GPU (CUDA 12.8 build — used during development of this milestone)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then the remaining dependencies:

```bash
pip install -r requirements-dev.txt   # or requirements.txt without pytest
```

Check the install:

```bash
python -c "import torch, whisper, jiwer; print(torch.__version__, torch.cuda.is_available())"
python -m pytest
```

### Notes

- **ffmpeg is not required** for WAV/FLAC/OGG input. Audio is decoded with
  `soundfile`, converted to mono and resampled to 16 kHz, then passed to Whisper
  as an array. (VoiceBank+DEMAND is distributed as WAV.)
- **Model download:** the first run of each Whisper model downloads its checkpoint
  to `~/.cache/whisper` (for example, `base.en` is about 139 MB and `small.en` is about 461 MB).
  Use `ASRConfig(download_root=...)` to change the cache location.

## Running the baseline

```bash
python -m src.asr.baseline --audio path/to/utterance.wav --reference "the reference transcript"
python -m src.asr.baseline --audio path/to/utterance.wav --reference-file path/to/utterance.txt
```

| Option | Default | Meaning |
|---|---|---|
| `--audio` | required | Audio file (WAV/FLAC/OGG) |
| `--reference` / `--reference-file` | none | Reference text; without it, only the transcript is printed |
| `--model` | `base.en` | Any Whisper model name (`tiny.en`, `small.en`, `medium.en`, `large-v3`, `turbo`, …) |
| `--device` | `auto` | `auto` / `cuda` / `cpu`. Requesting `cuda` without a GPU is an error, not a silent fallback |
| `--language` | `en` | Language code, or `auto` for Whisper's language detection |
| `--no-normalize` | off | Score raw text (case and punctuation count as errors) |
| `--show-alignment` | off | Print the word-level REF/HYP alignment |
| `--json-out` | none | Save transcript, segments, config and WER as JSON |
| `-v` | off | Debug logging |

Exit codes: `0` success, `2` bad input (missing file, undecodable audio, empty
reference), `1` model/runtime failure.

Decoding uses `temperature=0.0` (greedy decoding) so runs are reproducible. Whisper's
own default uses temperature fallback, which can give different output between runs.

## How WER is calculated

WER is computed with [JiWER](https://github.com/jitsi/jiwer) from the minimum
edit-distance word alignment between the reference and the hypothesis (the ASR output):

```text
WER = (S + D + I) / N
```

- **S**: substitutions (a reference word replaced by a different word)
- **D**: deletions (a reference word missing from the hypothesis)
- **I**: insertions (an extra word in the hypothesis)
- **N**: number of words in the reference (= S + D + H, where H is the number of correct words, or hits)

WER can be above 1.0 when there are many insertions. An empty reference raises an error, because WER is undefined.

**Text normalization** (on by default, applied identically to both texts):
lowercase → remove punctuation → collapse whitespace. Whisper outputs casing and
punctuation that references usually lack; without normalization, WER would mostly
measure formatting. Removing punctuation turns `don't` into `dont` on both
sides. Numbers are **not** normalized (`"2"` vs `"two"` counts as a substitution).
This may matter for VoiceBank transcripts and will be reviewed when the dataset is added.

## Python API (for the pipeline and the future Streamlit UI)

```python
from src.asr.whisper_asr import ASRConfig, WhisperASR
from src.evaluation.wer import compute_wer

asr = WhisperASR(ASRConfig(model_name="base.en"))   # model loads lazily, once
result = asr.transcribe("utt.wav")                   # or a mono float32 16 kHz numpy array
score = compute_wer("reference text", result.text)
print(result.text, score.wer, score.substitutions, score.deletions, score.insertions)
```

`transcribe()` also accepts arrays, so enhanced audio from a model can be passed
directly without writing it to disk.

## Module layout

| Module | Responsibility |
|---|---|
| `src/preprocessing/audio_io.py` | Decode audio, convert to mono, resample to 16 kHz |
| `src/utils/device.py` | CUDA/CPU selection |
| `src/utils/logging_utils.py` | Logging setup for entry points |
| `src/asr/whisper_asr.py` | `WhisperASR`, `ASRConfig`, `TranscriptionResult` |
| `src/evaluation/wer.py` | `compute_wer`, `normalize_text`, `alignment_report` |
| `src/asr/baseline.py` | Command-line interface |
