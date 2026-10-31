"""Presentation flows through Streamlit's real rendering/session-state layer."""

import io
from pathlib import Path

import streamlit as st
import pytest

import torch
from streamlit.testing.v1 import AppTest

from src.demo.pipeline_runner import PipelineResult, PipelineRunner, StageTiming, tensor_to_wav_bytes


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def presentation_app(tmp_path):
    """Use tiny fixtures so UI checks never require the benchmark or weights."""
    script = tmp_path / "app.py"
    script.write_text((ROOT / "app.py").read_text(encoding="utf-8"), encoding="utf-8")
    data = tmp_path / "data/processed/voicebank_demand_16k"
    (data / "noisy_testset_wav").mkdir(parents=True)
    (data / "testset_txt").mkdir()
    for index, name in enumerate(("p232_032", "p232_001")):
        (data / f"noisy_testset_wav/{name}.wav").write_bytes(
            tensor_to_wav_bytes(torch.sin(torch.arange(16000) * (0.1 + index * 0.05)) * 0.05)
        )
        (data / f"testset_txt/{name}.txt").write_text("the peace process", encoding="utf-8")
    return script


def _fake_result(self, audio_source, run_enhancement=True, run_correction=True):
    return PipelineResult(
        input_audio=torch.ones(16000) * 0.05,
        enhanced_audio=torch.ones(16000) * 0.025 if run_enhancement else None,
        sample_rate=16000, duration_s=1.0,
        raw_transcript="the <b>piece</b> process",
        enhanced_transcript="the piece process" if run_enhancement else None,
        corrected_raw_transcript="the peace process" if run_correction else None,
        corrected_enhanced_transcript="the peace process" if run_correction and run_enhancement else None,
        raw_corrections=[{
            "original_token": "piece", "corrected_token": "peace",
            "decision": {"best_word": "peace", "confidence": 0.98,
                         "alternatives": [{"word": "peace", "phonetic_similarity": 1.0}]},
        }] if run_correction else [],
        timing=StageTiming(asr_raw_s=0.2, asr_enhanced_s=0.3, total_s=0.8),
    )


def test_results_survive_navigation_but_hide_when_audio_changes(monkeypatch, presentation_app):
    monkeypatch.setattr(PipelineRunner, "process", _fake_result)
    at = AppTest.from_file(str(presentation_app), default_timeout=30).run()
    assert not at.exception
    assert at.button_group(key="navigation").options == ["Audio files", "Record audio"]
    assert not at.sidebar.children
    next(b for b in at.button if b.label in ("Process Audio", "Process Recording")).click().run()
    assert not at.error
    html = "\n".join(x.value for x in at.markdown)
    assert "&lt;b&gt;piece&lt;/b&gt;" in html
    assert "similarity: 1.00" in html
    assert len(at.dataframe) == 2  # stage timings and WER
    for page in ("Record audio", "Audio files"):
        at.button_group(key="navigation").set_value(page).run()
        assert not at.exception
        assert not at.error
    assert any("### Transcripts" == x.value for x in at.markdown)
    at.selectbox(key="demo_sample").select("p232_001.wav").run()
    assert not any("### Transcripts" == x.value for x in at.markdown)


def test_uploaded_audio_and_reference_edit_retain_results(monkeypatch, presentation_app):
    monkeypatch.setattr(PipelineRunner, "process", _fake_result)
    at = AppTest.from_file(str(presentation_app), default_timeout=30).run()
    at.radio[0].set_value("Upload audio").run()
    wav = tensor_to_wav_bytes(torch.ones(16000) * 0.05)
    at.file_uploader[0].set_value(("speech.wav", wav, "audio/wav")).run()
    at.text_input[0].set_value("the peace process").run()
    next(b for b in at.button if b.label in ("Process Audio", "Process Recording")).click().run()
    assert not at.exception and not at.error
    assert any("### Word Error Rate (WER) Evaluation" == x.value for x in at.markdown)
    at.button_group(key="navigation").set_value("Record audio").run()
    at.button_group(key="navigation").set_value("Audio files").run()
    assert not at.exception and not at.error
    assert any("### Transcripts" == x.value for x in at.markdown)
    at.text_input[0].set_value("!!!").run()
    assert not at.error
    assert any("Enter words" in x.value for x in at.warning)
    assert any("### Transcripts" == x.value for x in at.markdown)


def test_recorded_audio_can_be_processed_and_reference_edited(monkeypatch, presentation_app):
    monkeypatch.setattr(PipelineRunner, "process", _fake_result)
    at = AppTest.from_file(str(presentation_app), default_timeout=30).run()
    at.button_group(key="navigation").set_value("Record audio").run()
    recording = tensor_to_wav_bytes(torch.sin(torch.arange(16000) * 0.1) * 0.05)
    monkeypatch.setattr(st, "audio_input", lambda *args, **kwargs: io.BytesIO(recording))
    at.run()
    next(b for b in at.button if b.label in ("Process Audio", "Process Recording")).click().run()
    assert not at.exception and not at.error
    assert any("### Transcripts" == x.value for x in at.markdown)
    at.text_input[0].set_value("the peace process").run()
    assert not at.exception and not at.error
    assert any("### Word Error Rate (WER) Evaluation" == x.value for x in at.markdown)
    monkeypatch.setattr(st, "audio_input", lambda *args, **kwargs: None)
    at.button_group(key="navigation").set_value("Audio files").run()
    at.button_group(key="navigation").set_value("Record audio").run()
    assert not at.exception and not at.error
    assert any("### Transcripts" == x.value for x in at.markdown)
