"""ASR wrapper and CLI tests. The Whisper model is replaced by a fake, so no download is needed."""

import json

import numpy as np
import pytest
import soundfile as sf

from src.asr import baseline
from src.asr.whisper_asr import ASRConfig, ASRError, WhisperASR


class FakeModel:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return {"text": f"  {self.text} ", "language": "en",
                "segments": [{"start": 0.0, "end": 1.0, "text": self.text, "tokens": [1, 2]}]}


@pytest.fixture
def fake_asr(monkeypatch):
    model = FakeModel("The cat sat on a mat.")

    def fake_load(self):
        self._model = model

    monkeypatch.setattr(WhisperASR, "load", fake_load)
    return model


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "utt.wav"
    sf.write(path, np.zeros(16_000, dtype=np.float32), 16_000)
    return path


def test_transcribe_path(fake_asr, wav):
    result = WhisperASR(ASRConfig(device="cpu")).transcribe(wav)
    assert result.text == "The cat sat on a mat."
    assert result.audio_duration_s == pytest.approx(1.0)
    assert result.segments == [{"start": 0.0, "end": 1.0, "text": "The cat sat on a mat."}]
    assert fake_asr.calls[0]["fp16"] is False


def test_transcribe_rejects_bad_array(fake_asr):
    asr = WhisperASR(ASRConfig(device="cpu"))
    with pytest.raises(ValueError):
        asr.transcribe(np.zeros((2, 10), dtype=np.float32))


def test_unknown_model_name():
    with pytest.raises(ASRError):
        WhisperASR(ASRConfig(model_name="not-a-model", device="cpu")).load()


def test_invalid_device():
    with pytest.raises(ValueError):
        WhisperASR(ASRConfig(device="tpu"))


def test_cli_end_to_end(fake_asr, wav, tmp_path, capsys):
    out = tmp_path / "res" / "out.json"
    code = baseline.main(["--audio", str(wav), "--reference", "the cat sat on the mat",
                          "--device", "cpu", "--json-out", str(out)])
    assert code == 0
    record = json.loads(out.read_text())
    assert record["wer"]["substitutions"] == 1
    assert record["wer"]["wer"] == pytest.approx(1 / 6)
    assert "WER" in capsys.readouterr().out


def test_cli_reference_file(fake_asr, wav, tmp_path):
    ref = tmp_path / "ref.txt"
    ref.write_text("the cat sat on a mat\n", encoding="utf-8")
    assert baseline.main(["--audio", str(wav), "--reference-file", str(ref), "--device", "cpu"]) == 0


def test_cli_missing_audio(tmp_path):
    assert baseline.main(["--audio", str(tmp_path / "missing.wav"), "--device", "cpu"]) == 2
