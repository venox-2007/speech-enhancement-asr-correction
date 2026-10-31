# Presentation rehearsal

The app runs locally and compares original audio, U-Net enhanced audio, Whisper transcripts, and conservative phonetic/GPT-2 correction. The interface contains only demo operations. Present the saved 824-utterance benchmark and methodology in the slides; a single live sample can behave differently.

## Verified on 25 October 2026

- The complete suite contains 261 passing tests, including real cached GPT-2 tests and Streamlit UI checks.
- All six curated samples completed both branches using real models on the RTX 3050. Warmed sample processing took 1.25–5.93 seconds. Offline model preparation took about 24 seconds in the checked run.
- WAV, MP3, FLAC, and OGG buffers decoded and completed the actual enhancement/ASR/correction pipeline.
- Enhancement/correction switches worked independently. A real CPU run also completed, taking about 11.4 seconds for `p257_096.wav` after preparation.
- Automated UI checks cover both input views, uploaded audio, supplied recording bytes, reference edits, HTML escaping, correction similarity, and saved-result behavior across navigation.
- The UI keeps the same absolute spectrogram scale across branches, shows actual hardware and correct model specifications, and supports downloading JSON results and enhanced WAV audio.
- The one-click offline launcher starts the app successfully. Physical microphone recording, audible playback, and projector readability still need a rehearsal with the actual equipment.

Detailed machine-readable outputs are saved locally in `results/presentation_readiness/audit.json`.

## Before the presentation

1. Keep this laptop's `.venv`, `data/processed/voicebank_demand_16k`, trained checkpoint in `results/enhancement_train`, and Whisper/Hugging Face model caches. GitHub now includes the trained U-Net checkpoint, six demo samples, saved summary/CSV results and full dataset download instructions. Run `python scripts/setup_demo.py --download-models` on a fresh clone. The full benchmark and pretrained model caches remain separate downloads.
2. Plug in the laptop and connect/test the room's audio output.
3. Double-click `launch_demo.cmd`, then open **http://127.0.0.1:8501** in Chrome or Edge. The launcher uses cached models with Hugging Face offline mode enabled. For a fresh machine, use the README setup and download the models online first.
4. Open **Settings** and click **Load models for demo**. Wait for the success message before starting. If a model is missing, close the launcher, run `streamlit run app.py` with internet access, load the models, and then retry the offline launcher.
5. Process `p232_032.wav` once to warm up GPU inference. Listen to both audio players. Confirm the transcript, correction chips, waveform/spectrogram comparison, timings, and WER table appear.
6. In **Record audio**, allow microphone access if prompted, record a short sentence, and process it. Physical microphone capture and projector/speaker operation need a rehearsal in the actual browser and room.

## Suggested demonstration sequence

- **Correction example:** `p232_032.wav`, with enhancement and correction enabled. In the checked run, original ASR produced “gullible homicide,” enhanced ASR produced “palpable homicide,” and both corrected to “culpable homicide.” Other transcription errors remained. A correction should not be presented as a perfect transcript.
- **Simple baseline:** `p232_001.wav` (“Please call Stella.”). An unchanged transcript is normal when safeguards retain the original words.
- **Known limitation:** `p232_043.wav` illustrates a documented risk of overcorrection (“bottle” versus “battle”). Inspect the actual run before showing it.
- **In your slides:** explain that SNR/STOI/SI-SDR improved while Whisper WER worsened. The best noisy-speech configuration was Whisper plus conservative correction, at 7.12% corpus WER.
- **In your slides:** explain the U-Net audio stage separately from the text correction stage. The U-Net has 482,449 parameters, channels 16/32/64/128, three pooling stages, and three upsampling stages. Whisper and GPT-2 are pretrained components.

## Claims to keep precise

- Noisy WER: **7.23%**. Enhanced WER: **8.23%**. Noisy + correction: **7.12%**. Enhanced + correction: **8.11%**.
- Correction repairs **seven net word errors out of 6,211 reference words**, about **0.11 percentage points**. Bootstrap results support a small gain; Wilcoxon does not reach significance for that correction comparison.
- SNR gain: **9.60 dB**. SI-SDR gain: **9.51 dB**. STOI gain: **0.0127**.
- PESQ is unavailable in the saved evaluation. Do not use numerical PESQ claims from older milestone documents.
- Audio artifacts and phase reuse are possible explanations for ASR degradation; the experiment does not prove the cause.
- The app processes recorded clips. It does not perform continuous streaming ASR. Demo clips are limited to 60 seconds.
- Candidate probability is relative to the proposed word alternatives. It is not calibrated ASR confidence.

## Remaining presentation material

The final submission deck is `presentation/Speech-Enhancement-ASR-SUBMISSION.pptx` (12 slides, no speaker notes). Use it instead of the original proposal or earlier drafts. It includes the corrected enhancement illustration, complete ASR/correction flow, implementation contribution, a worked correction example and qualified benchmark interpretation. It has no viva appendix, footer or page numbers. CNN comparisons and additional datasets were proposed rather than evaluated.

## Verification and backup on 28 October 2026

The full 261-test suite passed again, including all 23 focused pipeline and UI checks. The earlier restricted-sandbox stall was in Windows asyncio socketpair setup before the app executed. Allowing local loopback sockets resolved that test-environment issue without changing application behavior.

`scripts/prepare_presentation_backup.py` runs the actual cached models and saves audio, JSON and a self-contained HTML replay. The fresh GPU run reproduced the peace/piece success, the partial culpable-homicide repair and the bottle/battle overcorrection. Open `presentation/demo_backup/OPEN_DEMO_BACKUP.html` if live inference is unavailable. It contains embedded audio and clearly identifies itself as saved results. This backup is not a screen recording or a simulation of live inference.

See [presentation/START_HERE.md](../presentation/START_HERE.md) for the short rehearsal and demonstration sequence. The checkpoint, decoding configuration, correction thresholds and 824-recording benchmark remain unchanged.
