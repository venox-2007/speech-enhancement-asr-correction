"""Save real inference outputs and a self-contained offline presentation replay.

Run from the project root: .venv\Scripts\python.exe scripts/prepare_presentation_backup.py
The HTML is explicitly a saved replay, not a replacement inference application.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
from src.demo.pipeline_runner import PipelineRunner, tensor_to_wav_bytes
from src.evaluation.wer import compute_wer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "presentation/demo_backup")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    runner = PipelineRunner(checkpoint_path=ROOT / "results/enhancement_train/checkpoint_best.pt", device=args.device)
    runner.prepare()
    print("Cached models loaded.", flush=True)
    checkpoint = ROOT / "results/enhancement_train/checkpoint_best.pt"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": args.device,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "description": "Saved actual pipeline outputs. These curated cases do not replace the 824-recording benchmark.", "samples": []}
    sections = []
    cases = [("p257_096", "Context correction example"), ("p232_032", "Correction with remaining errors"),
             ("p232_043", "Known overcorrection limitation")]
    for name, title in cases:
        source = ROOT / f"data/processed/voicebank_demand_16k/noisy_testset_wav/{name}.wav"
        reference = (ROOT / f"data/processed/voicebank_demand_16k/testset_txt/{name}.txt").read_text().strip()
        result = runner.process(source)
        row = {"sample": name, "reference": reference, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), **result.to_dict()}
        transcript_fields = [("raw_transcript", "Original audio + Whisper"), ("corrected_raw_transcript", "Original audio + correction"),
                             ("enhanced_transcript", "Enhanced audio + Whisper"), ("corrected_enhanced_transcript", "Enhanced audio + correction")]
        row["wer"] = {key: compute_wer(reference, getattr(result, key)).wer for key, _ in transcript_fields}
        manifest["samples"].append(row)
        (args.output / f"{name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        players = []
        for label, audio in [("Original noisy audio", result.input_audio), ("U-Net enhanced audio", result.enhanced_audio)]:
            wav = tensor_to_wav_bytes(audio, result.sample_rate)
            suffix = "noisy" if label.startswith("Original") else "enhanced"
            (args.output / f"{name}-{suffix}.wav").write_bytes(wav)
            uri = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
            players.append(f'<div><h3>{label}</h3><audio controls preload="none" src="{uri}"></audio></div>')
        cards = []
        for key, label in transcript_fields:
            cards.append(f'<article><h3>{label}</h3><p>{escape(getattr(result, key))}</p><strong>WER: {row["wer"][key]*100:.1f}%</strong></article>')
        edits = []
        for branch, changes in [("Original audio", result.raw_corrections), ("Enhanced audio", result.enhanced_corrections)]:
            for change in changes:
                if change["was_changed"]:
                    edits.append(f'<li>{branch}: <del>{escape(change["original_token"])}</del> → <strong>{escape(change["corrected_token"])}</strong></li>')
        sections.append(f'<section id="{name}"><h2>{title} <small>{name}</small></h2><p class="reference"><b>Reference:</b> {escape(reference)}</p><div class="grid">{"".join(players)}</div><div class="grid">{"".join(cards)}</div><ul>{"".join(edits) or "<li>The safeguards retained the original words.</li>"}</ul><p class="muted">Recorded processing time: {result.timing.total_s:.2f} seconds on {args.device}. This is a saved run.</p></section>')
        print(name, row["wer"], flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    style = 'body{font-family:Segoe UI,Arial,sans-serif;background:#080e26;color:#ebecef;max-width:1180px;margin:auto;padding:28px}h1,h2{color:#fff}h1{font-size:34px}h2{font-size:27px}h3{color:#67beff;font-size:18px}p,li{font-size:19px;line-height:1.55}small,.muted{color:#aab6db}small{font-size:15px}a{color:#55d8c1;margin-right:24px}section{padding:24px 0 36px;border-top:1px solid #414a70}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}article,.reference{background:#151e3b;padding:18px;border-radius:9px}article strong{color:#55d8c1}audio{width:100%}.badge{color:#55d8c1;font-weight:600}del{color:#ff8898}@media(max-width:700px){.grid{grid-template-columns:1fr}}'
    navigation = ''.join('<a href="#' + n + '">' + escape(t) + '</a>' for n,t in cases)
    html = f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Speech enhancement: saved demo replay</title><style>{style}</style><h1>Speech Enhancement & ASR Correction</h1><p class="badge">Offline backup · Saved real model outputs</p><p>This page replays completed runs. It does not run models or simulate live inference. Audio is embedded, so this file works without the app, model caches, or internet.</p><nav>{navigation}</nav>{"".join(sections)}<p class="muted">Generated {escape(manifest["created_utc"])}. The full benchmark contains 824 recordings and 6,211 reference words. Curated examples show both successes and limitations.</p></html>'
    (args.output / "OPEN_DEMO_BACKUP.html").write_text(html, encoding="utf-8")
    print("Saved offline backup:", args.output / "OPEN_DEMO_BACKUP.html", flush=True)


if __name__ == "__main__":
    main()
