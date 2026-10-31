# Final submission presentation

Use **Speech-Enhancement-ASR-SUBMISSION.pptx**. It contains 12 slides, including the final GitHub link slide, with no speaker notes, footers or page numbers.

Double-click **OPEN_SIMPLE_PRESENTATION.cmd** or **OPEN_PRESENTATION.cmd** to open the deck with the included fonts temporarily available to PowerPoint. Keep the opener scripts and `assets/fonts/` beside the deck. Opening the PPTX directly on a computer without those fonts can cause substitution. Older local decks are previous drafts and are not part of the GitHub submission.

Repository: https://github.com/venox-2007/speech-enhancement-asr-correction

The repository includes all source code, the trained U-Net checkpoint, six attributed demonstration samples, saved evaluation results, tests and setup instructions. The complete VoiceBank+DEMAND dataset is restored from Edinburgh using `scripts/download_dataset.py`; full archives and virtual environments are not stored in Git.

## Demonstration

1. On a fresh clone, follow the root README and run `python scripts/setup_demo.py --download-models` online once.
2. Double-click `launch_demo.cmd` and open http://127.0.0.1:8501.
3. Use Settings to load the models, then process `p257_096.wav` to show piece/peace correction.
4. Explain that one curated example does not represent the full 824-recording benchmark.
5. `p232_043.wav` demonstrates the known bottle/battle overcorrection.

If live inference is unavailable, download and open **demo_backup/OPEN_DEMO_BACKUP.html**. It contains embedded audio and actual saved model outputs. Present it as a saved replay, not live inference.

## Main explanation

“I trained a U-Net to reduce noise in speech spectrograms. Whisper transcribes the original or enhanced audio, and a pronunciation/GPT-2 module checks word choices. Audio metrics improve after enhancement, but Whisper's WER increases. Correction removes seven net word errors in each path, so its overall gain is small.”
