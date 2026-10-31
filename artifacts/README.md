# Saved experiment artifacts

- `model/checkpoint_best.pt`: exact trained U-Net checkpoint used in the presentation (epoch 19).
- `training/`: complete 20-epoch history.
- `evaluation/`: full per-utterance CSVs, summary JSONs and small ASR caches for the reported experiment.
- `manifest.json`: SHA-256 checksums and restoration destinations for the checkpoint and six demo samples.

Run `python scripts/setup_demo.py` to restore the runtime paths without downloading the full dataset.
Whisper and GPT-2 remain pretrained components. Their caches are downloaded separately.
Large correction candidate caches, temporary renders, and smoke-test checkpoints are excluded.
The evaluation summaries retain their original experiment metadata; the project README explains limitations.
