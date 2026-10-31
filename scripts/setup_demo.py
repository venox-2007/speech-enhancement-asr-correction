"""Restore bundled demo assets and optionally cache pretrained models.

Run from the repository root: python scripts/setup_demo.py --download-models
The full benchmark is optional for the six bundled demonstration recordings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def restore_assets(root: Path = ROOT) -> int:
    manifest = json.loads((root / "artifacts/manifest.json").read_text(encoding="utf-8"))
    restored = 0
    for item in manifest["restore_files"]:
        source = root / item["source"]
        target = root / item["destination"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Bundled asset checksum mismatch: {item['source']}")
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                print(f"Keeping existing local file: {item['destination']}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        restored += 1
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-models", action="store_true",
                        help="Download/cache Whisper base.en and GPT-2 (internet required initially).")
    args = parser.parse_args()
    print(f"Restored {restore_assets()} bundled files. Existing local files were preserved.")
    if args.download_models:
        from src.asr.whisper_asr import ASRConfig, WhisperASR
        from src.correction.lm_scorer import CausalLMScorer
        WhisperASR(ASRConfig(device="cpu")).load()
        CausalLMScorer(device="cpu").load()
        print("Whisper base.en and GPT-2 are cached for offline use.")
    print("Start the app: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
