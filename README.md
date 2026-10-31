# Speech Enhancement and Context-Aware ASR Correction

A completed deep learning project that reduces noise with a trained U-Net, transcribes audio with pretrained Whisper, and evaluates pronunciation-aware transcript correction using pretrained GPT-2.

**Main finding:** the U-Net improves objective audio scores but increases Whisper word errors. Conservative text correction gives a small improvement in each audio path. The complete pipeline does not outperform Whisper on the original noisy audio.

## Submission and quick links

- [Final presentation: 12 slides, no speaker notes](presentation/Speech-Enhancement-ASR-SUBMISSION.pptx)
- [Saved demonstration replay](presentation/demo_backup/OPEN_DEMO_BACKUP.html): download and open locally to play embedded audio. This is a replay of actual runs, not live inference.
- [Trained U-Net checkpoint](artifacts/model/checkpoint_best.pt)
- [Full saved evaluation tables and summaries](artifacts/evaluation/)
- [Dataset source, license and download instructions](data/README.md)
- [Technical implementation](src/) and [tests](tests/)

## Run the demonstration from a fresh clone

Python 3.11 is the tested version. A GPU is optional. CPU inference is supported.

```bash
git clone https://github.com/venox-2007/speech-enhancement-asr-correction.git
cd speech-enhancement-asr-correction
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt alternative:
# .venv\Scripts\activate.bat
```

On Linux/macOS, use `source .venv/bin/activate`.

```bash
# CPU installation (works without an NVIDIA GPU):
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

# Restore the included checkpoint and six samples, then cache pretrained models:
python scripts/setup_demo.py --download-models

# Run the app:
python -m streamlit run app.py
```

For GPU acceleration, install matching PyTorch/torchaudio builds using the [official PyTorch selector](https://pytorch.org/get-started/locally/) instead of the CPU installation line.
The first model download requires internet. Whisper `base.en` and GPT-2 are cached by their respective libraries.
For additional audio-format support, install [FFmpeg](https://ffmpeg.org/download.html) and make it available on PATH.

After the initial setup, Windows users can double-click **launch_demo.cmd** for offline operation. This launcher expects `.venv`, the restored checkpoint and cached pretrained models.
Open **Settings > Load models for demo** to prepare the models before processing. The app supports WAV, MP3, FLAC and OGG uploads and browser microphone recordings of 0.1–60 seconds.
Recording creates a clip for processing. This is not continuous streaming transcription.

The six bundled samples are sufficient for a demonstration. Downloading the complete training dataset is optional for running the app.

## Pipeline

```text
                              Noisy audio, 16 kHz mono
                                /                 \
                    Original audio            STFT magnitude + phase
                         |                           |
                         |                    U-Net magnitude mask
                         |                           |
                         |                 iSTFT using noisy phase
                         |                           |
                 Whisper base.en              Whisper base.en
                         |                           |
                 Original transcript          Enhanced transcript
                         |                           |
                 Optional correction          Optional correction
                 CMUdict + GPT-2              CMUdict + GPT-2
```

U-Net operates on audio spectrogram magnitudes. It does not process text. Whisper transcribes waveforms and performs its own feature extraction. The separate corrector operates on text.

## Model and training

| Setting | Experiment value |
|---|---|
| Architecture | 2D U-Net, three pooling stages and three upsampling stages |
| Feature channels | 16, 32, 64, 128, then 64, 32, 16 |
| Convolution blocks | Two 3x3 convolutions, BatchNorm and LeakyReLU (slope 0.2) |
| Skip connections | Concatenation of matching encoder and decoder features |
| Output | Sigmoid mask in [0, 1] multiplied by noisy magnitude |
| Trainable parameters | 482,449 |
| STFT | FFT/window 512, Hann window, hop 256, 257 frequency bins |
| Training segments | 16,384 samples, approximately 1.024 seconds |
| Loss | Linear-magnitude L1 loss |
| Optimizer | AdamW, initial learning rate 0.001, weight decay 0.00001 |
| Batch size / epochs | 16 / 20 |
| Best checkpoint | Epoch 19, validation L1 approximately 0.038685 |
| Training hardware | NVIDIA RTX 3050, 6 GB VRAM, mixed precision |

This project trains U-Net. Whisper `base.en` and GPT-2 use pretrained weights without fine-tuning.
See [training history](artifacts/training/) and [checkpoint restoration checksums](artifacts/manifest.json).

## Pronunciation and context correction

CMUdict provides phoneme sequences. RapidFuzz shortlists similar pronunciations, and heuristic weighted phoneme distance ranks them. GPT-2 scores each candidate in the complete original sentence.

`combined score = sentence log probability + log(phonetic similarity) - replacement penalty`

The app requires similarity >= 0.80, a score margin >= 3.5 and a relative candidate probability >= 0.70. Replacements receive a penalty of 1.0. Each word is evaluated independently in the original transcript, preserving case and punctuation. Candidate probabilities are relative preferences, not calibrated acoustic confidence.

Example: `The piece process will be buried.` becomes `The peace process will be buried.`
A known failure changes the already correct word `bottle` to `battle`.

## Dataset and reproduction

[VoiceBank+DEMAND](https://doi.org/10.7488/ds/2117) supplies paired clean/noisy recordings. Its 28-speaker training pool has 11,572 recordings. We reserve speakers p286/p287 for validation, leaving 10,700 training recordings. The test set contains 824 recordings from p232/p257. The enhancer partitions are speaker-disjoint.

The repository includes six attributed 16 kHz demo samples, official archive URLs and checksums, the synthetic correction development set, and a compressed manifest of all 24,792 processed WAV files. The multi-gigabyte full dataset stays at the original publisher and can be restored with:

```bash
python scripts/download_dataset.py --split all --resample
python -m src.preprocessing.validate_dataset --all
```

For only the test set, use `--split test`. The downloader also restores official test reference transcripts. See [data/README.md](data/README.md) and [data/sources.json](data/sources.json).

To repeat the enhancement training with the main recorded settings:

```bash
python -m src.enhancement.train --data-dir data/processed/voicebank_demand_16k --output-dir results/enhancement_train --epochs 20 --batch-size 16 --lr 0.001 --patience 5
```

The seed defaults to 42. Numerical results can vary with hardware and library versions. Evaluate using the modules in [src/evaluation](src/evaluation/) and their `--help` options. Full experiment instructions appear in the [technical documentation](docs/). Earlier milestone documents describe development stages; the tables below and the bundled experiment artifacts describe the final evaluated system.

## Final results

All recognition conditions use the same **824 test recordings and 6,211 normalized reference words**. Corpus WER sums substitutions, deletions and insertions and divides by total reference words.

| Condition | Corpus WER | Substitutions | Deletions | Insertions | Total errors |
|---|---:|---:|---:|---:|---:|
| Clean audio + Whisper | 4.65% | 202 | 55 | 32 | 289 |
| Noisy audio + Whisper | 7.23% | 330 | 69 | 50 | 449 |
| Enhanced audio + Whisper | 8.23% | 369 | 80 | 62 | 511 |
| Noisy audio + Whisper + correction | 7.12% | 323 | 69 | 50 | 442 |
| Enhanced audio + Whisper + correction | 8.11% | 362 | 80 | 62 | 504 |

| Audio metric | Noisy | Enhanced | Mean gain |
|---|---:|---:|---:|
| STOI | 0.9211 | 0.9338 | +0.0127 |
| SI-SDR | 8.45 dB | 17.96 dB | +9.51 dB |
| SNR | 8.45 dB | 18.05 dB | +9.60 dB |

Correction removes seven net word errors per audio path, approximately **0.11 percentage points**. Stricter safeguards reduce incorrect replacements from 69 to 2 in the noisy branch and 67 to 2 in the enhanced branch, while also preventing some useful repairs. This is not a 97% reduction in overall WER.

Magnitude L1 optimizes spectral reconstruction, not recognition accuracy. Speech distortion, suppression artifacts and noisy-phase reuse are possible explanations for the WER increase. The experiments do not establish which explanation caused it. Objective metrics are not human listening-test results, and STOI is not recognition accuracy.

### Statistical checks and limits

| Comparison (second minus first) | WER change, percentage points | 95% paired bootstrap interval, percentage points | Bootstrap tail measure | Wilcoxon p |
|---|---:|---|---:|---:|
| Noisy / noisy + correction | -0.113 | [-0.223, -0.016] | 0.0356 | 0.1542 |
| Enhanced / enhanced + correction | -0.113 | [-0.222, -0.016] | 0.0382 | 0.0754 |
| Noisy / enhanced | +0.998 | [+0.345, +1.687] | 0.0022 | 0.0066 |
| Noisy / enhanced + correction | +0.886 | [+0.220, +1.584] | 0.0094 | 0.0213 |

The bootstrap uses 10,000 paired utterance resamples. Its reported tail measure is an empirical zero-crossing measure, not a formal null-resampling test. Wilcoxon does not reach 0.05 for either correction comparison. The small correction gain therefore needs broader validation. There is no multiple-comparison adjustment.

Further limitations: one dataset, two test speakers, a magnitude-only U-Net, one ASR model, and a 60-example synthetic correction development set. Earlier test error analysis informed correction safeguards, so correction development was not fully blind. PESQ was unavailable and is not reported. The work contributes implementation and evaluation rather than a new neural architecture or a state-of-the-art claim.

## Tests

```bash
python -m pip install -r requirements-dev.txt
# Unit/UI tests that do not require full benchmark data or a downloaded GPT-2:
python -m pytest -m "not lm and not dataset"
# Complete suite after preparing model caches and full data:
python -m pytest
```

All **265 tests passed** during submission verification on 31 October 2026. These include asset checksum/restoration tests, safe ZIP extraction checks, audio/model tests, correction evaluation, and Streamlit UI tests. The tested package versions are recorded in [artifacts/tested-environment.txt](artifacts/tested-environment.txt).

## Repository contents

```text
app.py                 Streamlit interface
launch_demo.cmd        Windows offline launcher
src/                   Enhancement, ASR, correction, preprocessing and evaluation
scripts/               Fresh-clone setup, official dataset download, replay generation
tests/                 Unit and integration checks
data/demo/             Six licensed demonstration recordings and references
data/sources.json      Official archive locations and checksums
data/processed_manifest.csv.gz  Manifest of the full processed benchmark
artifacts/model/       Trained U-Net checkpoint
artifacts/training/    Actual training history
artifacts/evaluation/  Complete saved summary JSONs and per-utterance CSVs
presentation/          Final submission deck, fonts and saved replay
docs/                  Technical reports and implementation details
```

The full dataset, third-party model caches, virtual environment, temporary renders and old slide drafts are excluded from Git. They are either reproducible from the supplied scripts or unnecessary for running the project. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and licensing.
