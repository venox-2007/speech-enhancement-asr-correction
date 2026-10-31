"""Interactive speech enhancement, transcription, and correction demo."""

from __future__ import annotations

import io
import logging
import hashlib
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy.signal import stft as scipy_stft

# Safeguard: neutralize path inspection on torch.classes to prevent watcher issues
if hasattr(torch, "classes"):
    try:
        torch.classes.__path__ = []
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from src.demo.pipeline_runner import (
    PipelineResult,
    PipelineRunner,
    get_default_device,
    load_audio_from_source,
    tensor_to_wav_bytes,
)
from src.evaluation.wer import compute_wer

# ---------------------------------------------------------------------------
# Page Configuration & Visual Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Speech Enhancement",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
/* Modern AI SaaS Visual System */

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

code, pre, .mono-font {
    font-family: 'JetBrains Mono', monospace !important;
}

/* App Header styling */
.app-hero {
    padding: 0.25rem 0 1.25rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 1.25rem;
}

.app-brand {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #f8fafc;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.stMarkdown h1.app-brand { font-size: 1.8rem !important; }
[data-testid="stMainBlockContainer"] { padding-top: 2rem; }

.app-tagline {
    font-size: 0.92rem;
    color: #94a3b8;
    margin-top: 0.25rem;
    margin-bottom: 0.75rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(56, 189, 248, 0.08);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.22);
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #38bdf8;
    box-shadow: 0 0 6px rgba(56, 189, 248, 0.8);
}

/* Pipeline Progress Breadcrumb */
.pipeline-flow {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: #0f172a;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    font-size: 0.78rem;
    color: #94a3b8;
    overflow-x: auto;
    margin-top: 0.5rem;
}

.flow-step {
    color: #cbd5e1;
    font-weight: 500;
    white-space: nowrap;
}

.flow-arrow {
    color: #475569;
}

/* Card System */
.workspace-card {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.2rem;
}

.workspace-header {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Polished Transcript Container with strict full-wrapping */
.transcript-card {
    background: #0b0f19;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 1rem 1.15rem;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #f1f5f9;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    margin-bottom: 0.6rem;
}

.transcript-badge {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.55rem;
}

.badge-raw {
    background: rgba(148, 163, 184, 0.15);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.25);
}

.badge-enh {
    background: rgba(56, 189, 248, 0.12);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.25);
}

.badge-corr {
    background: rgba(52, 211, 153, 0.12);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.25);
}

/* Metric Cards */
.kpi-metric-box {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    text-align: left;
}

.kpi-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin: 0;
}

.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.15;
    margin: 0.3rem 0;
}

.kpi-sub {
    font-size: 0.76rem;
    color: #64748b;
    margin: 0;
}

/* Changed word chips */
.diff-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 0.28rem 0.65rem;
    border-radius: 6px;
    font-size: 0.82rem;
    margin: 0.25rem 0.25rem 0.25rem 0;
}

.diff-from {
    color: #f87171;
    text-decoration: line-through;
}

.diff-to {
    color: #4ade80;
    font-weight: 600;
}

.diff-arrow {
    color: #64748b;
}

.diff-meta {
    color: #94a3b8;
    font-size: 0.74rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Keep demo choices when their widgets are temporarily absent on another page.
for state_key in ("demo_sample", "demo_enhancement", "demo_correction", "live_enhancement", "live_correction",
                  "demo_source", "demo_reference", "live_reference"):
    if state_key in st.session_state:
        st.session_state[state_key] = st.session_state[state_key]

PROJECT_ROOT = Path(__file__).resolve().parent


def remember_audio(widget_key: str, saved_key: str) -> None:
    """Retain audio across navigation, and honor the recorder/uploader clear action."""
    audio = st.session_state.get(widget_key)
    if audio is None:
        st.session_state.pop(saved_key, None)
    else:
        st.session_state[saved_key] = audio.getvalue()


# ---------------------------------------------------------------------------
# Cached Model Lifecycles
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing neural network models in memory...")
def get_cached_pipeline_runner(
    checkpoint_path: str,
    whisper_model_name: str,
    device: str,
) -> PipelineRunner:
    """Instantiate and cache the pipeline runner across Streamlit sessions."""
    return PipelineRunner(
        checkpoint_path=checkpoint_path,
        whisper_model_name=whisper_model_name,
        device=device,
        load_models_immediately=False,
    )


# ---------------------------------------------------------------------------
# Signal Visualizations
# ---------------------------------------------------------------------------
def plot_audio_comparison(
    raw_audio: torch.Tensor,
    enh_audio: torch.Tensor | None,
    sr: int = 16_000,
) -> plt.Figure:
    """Generate high-contrast, production-grade waveform and spectrogram plots."""
    raw_np = raw_audio.detach().cpu().reshape(-1).numpy()
    time_raw = np.arange(len(raw_np)) / sr

    def draw_spectrogram(ax, audio):
        # Use the same absolute dB range on both branches for a fair comparison.
        signal = np.pad(audio, (0, max(0, 512 - len(audio))))
        frequencies, times, spectrum = scipy_stft(signal, fs=sr, nperseg=512, noverlap=256)
        db = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-10))
        ax.pcolormesh(times, frequencies, db, shading="auto", cmap="magma", vmin=-100, vmax=0)

    has_enh = enh_audio is not None
    num_cols = 2 if has_enh else 1

    fig, axes = plt.subplots(2, num_cols, figsize=(7 * num_cols, 4.4), sharex="col", squeeze=False)
    fig.patch.set_facecolor("#0b0f19")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#111827")
            ax.tick_params(colors="#94a3b8", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.15))
            ax.xaxis.label.set_color("#cbd5e1")
            ax.yaxis.label.set_color("#cbd5e1")
            ax.title.set_color("#f8fafc")
            ax.title.set_fontsize(10)
            ax.title.set_fontweight("600")

    # Raw Waveform
    axes[0, 0].plot(time_raw, raw_np, color="#38bdf8", linewidth=0.75, alpha=0.9)
    axes[0, 0].set_title("Input Signal Waveform")
    axes[0, 0].set_ylabel("Amplitude", fontsize=9)
    axes[0, 0].set_ylim(-1.05, 1.05)
    axes[0, 0].grid(True, color=(1.0, 1.0, 1.0), alpha=0.06, linestyle="--", linewidth=0.5)

    # Raw Spectrogram
    draw_spectrogram(axes[1, 0], raw_np)
    axes[1, 0].set_title("Input Spectrogram (STFT)")
    axes[1, 0].set_xlabel("Time (s)", fontsize=9)
    axes[1, 0].set_ylabel("Frequency (Hz)", fontsize=9)

    if has_enh and enh_audio is not None:
        enh_np = enh_audio.detach().cpu().reshape(-1).numpy()
        time_enh = np.arange(len(enh_np)) / sr

        # Enhanced Waveform
        axes[0, 1].plot(time_enh, enh_np, color="#34d399", linewidth=0.75, alpha=0.9)
        axes[0, 1].set_title("U-Net Enhanced Waveform")
        axes[0, 1].set_ylabel("Amplitude", fontsize=9)
        axes[0, 1].set_ylim(-1.05, 1.05)
        axes[0, 1].grid(True, color=(1.0, 1.0, 1.0), alpha=0.06, linestyle="--", linewidth=0.5)

        # Enhanced Spectrogram
        draw_spectrogram(axes[1, 1], enh_np)
        axes[1, 1].set_title("U-Net Enhanced Spectrogram (Cleaned)")
        axes[1, 1].set_xlabel("Time (s)", fontsize=9)
        axes[1, 1].set_ylabel("Frequency (Hz)", fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reusable Result Rendering
# ---------------------------------------------------------------------------
def render_pipeline_outputs(
    result: PipelineResult,
    input_wav_bytes: bytes,
    reference_text: str | None = None,
    run_enhancement: bool = True,
    run_correction: bool = True,
    is_live_recording: bool = False,
) -> None:
    """Render full transcript cascade, audio players, word diffs, and latency telemetry."""
    st.markdown("---")

    # 1. Transcript Flow Cascade
    st.markdown("### Transcripts")
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown("#### Original audio")
        st.markdown(
            '<div class="transcript-badge badge-raw">Original transcript</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="transcript-card">{escape(result.raw_transcript or "(no speech detected)")}</div>',
            unsafe_allow_html=True,
        )

        if run_correction and result.corrected_raw_transcript is not None:
            st.markdown(
                '<div class="transcript-badge badge-corr">Corrected transcript</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="transcript-card">{escape(result.corrected_raw_transcript or "(no speech detected)")}</div>',
                unsafe_allow_html=True,
            )

            if result.raw_corrections:
                st.markdown("**Changes:**")
                for c in result.raw_corrections:
                    dec = c.get("decision") or {}
                    sim = next((a["phonetic_similarity"] for a in dec.get("alternatives", []) if a["word"] == dec.get("best_word")), None)
                    sim_text = f"{sim:.2f}" if sim is not None else "N/A"
                    st.markdown(
                        f'<div class="diff-chip">'
                        f'<span class="diff-from">{escape(c["original_token"])}</span>'
                        f'<span class="diff-arrow">&rarr;</span>'
                        f'<span class="diff-to">{escape(c["corrected_token"])}</span>'
                        f'<span class="diff-meta">(candidate probability: {dec.get("confidence", 0):.2f}, similarity: {sim_text})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="font-size: 0.82rem; color: #94a3b8; font-style: italic; margin-top: 0.35rem;">'
                    'No changes needed under the correction safeguards.'
                    '</div>',
                    unsafe_allow_html=True,
                )

    with t_col2:
        st.markdown("#### Enhanced audio")
        if result.enhanced_transcript is not None:
            st.markdown(
                '<div class="transcript-badge badge-enh">Enhanced transcript</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="transcript-card">{escape(result.enhanced_transcript or "(no speech detected)")}</div>',
                unsafe_allow_html=True,
            )

            if run_correction and result.corrected_enhanced_transcript is not None:
                st.markdown(
                    '<div class="transcript-badge badge-corr">Corrected transcript</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="transcript-card">{escape(result.corrected_enhanced_transcript or "(no speech detected)")}</div>',
                    unsafe_allow_html=True,
                )

                if result.enhanced_corrections:
                    st.markdown("**Changes:**")
                    for c in result.enhanced_corrections:
                        dec = c.get("decision") or {}
                        sim = next((a["phonetic_similarity"] for a in dec.get("alternatives", []) if a["word"] == dec.get("best_word")), None)
                        sim_text = f"{sim:.2f}" if sim is not None else "N/A"
                        st.markdown(
                            f'<div class="diff-chip">'
                            f'<span class="diff-from">{escape(c["original_token"])}</span>'
                            f'<span class="diff-arrow">&rarr;</span>'
                            f'<span class="diff-to">{escape(c["corrected_token"])}</span>'
                            f'<span class="diff-meta">(candidate probability: {dec.get("confidence", 0):.2f}, similarity: {sim_text})</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div style="font-size: 0.82rem; color: #94a3b8; font-style: italic; margin-top: 0.35rem;">'
                        'No changes needed under the correction safeguards.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Speech enhancement is turned off.")

    # 2. Audio & Signal Comparison
    st.markdown("### Audio comparison")
    a_col1, a_col2 = st.columns(2)

    with a_col1:
        st.markdown("**Original audio**")
        st.audio(input_wav_bytes, format="audio/wav")

    with a_col2:
        if result.enhanced_audio is not None:
            st.markdown("**Enhanced audio**")
            enh_bytes = tensor_to_wav_bytes(result.enhanced_audio, result.sample_rate)
            st.audio(enh_bytes, format="audio/wav")
        else:
            st.info("Speech enhancement is turned off.")

    st.caption(f"Processed in {result.timing.total_s:.2f} seconds")
    with st.expander("Waveforms and spectrograms"):
        st.caption("Both spectrograms use the same color scale: −100 to 0 dB.")
        fig = plot_audio_comparison(result.input_audio, result.enhanced_audio, result.sample_rate)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with st.expander("Processing times"):
        st.dataframe(pd.DataFrame([
            {"Stage": name, "Seconds": round(seconds, 3)}
            for name, seconds in result.timing.to_dict().items()
        ]), hide_index=True, width="stretch")

    # 4. Word Error Rate (if reference supplied)
    if reference_text and not any(ch.isalnum() for ch in reference_text):
        st.warning("Enter words in the reference transcript to calculate WER.")
    elif reference_text:
        st.markdown("### Word Error Rate (WER) Evaluation")
        wer_records = []
        raw_eval = compute_wer(reference_text, result.raw_transcript)
        wer_records.append({
            "Configuration": "Raw Input Speech",
            "Word Error Rate": f"{raw_eval.wer * 100:.1f}%",
            "Substitutions": raw_eval.substitutions,
            "Deletions": raw_eval.deletions,
            "Insertions": raw_eval.insertions,
            "Hits": raw_eval.hits,
        })

        if result.corrected_raw_transcript is not None:
            raw_corr_eval = compute_wer(reference_text, result.corrected_raw_transcript)
            wer_records.append({
                "Configuration": "Raw Speech + Context Correction",
                "Word Error Rate": f"{raw_corr_eval.wer * 100:.1f}%",
                "Substitutions": raw_corr_eval.substitutions,
                "Deletions": raw_corr_eval.deletions,
                "Insertions": raw_corr_eval.insertions,
                "Hits": raw_corr_eval.hits,
            })

        if result.enhanced_transcript is not None:
            enh_eval = compute_wer(reference_text, result.enhanced_transcript)
            wer_records.append({
                "Configuration": "U-Net Enhanced Speech",
                "Word Error Rate": f"{enh_eval.wer * 100:.1f}%",
                "Substitutions": enh_eval.substitutions,
                "Deletions": enh_eval.deletions,
                "Insertions": enh_eval.insertions,
                "Hits": enh_eval.hits,
            })

        if result.corrected_enhanced_transcript is not None:
            enh_corr_eval = compute_wer(reference_text, result.corrected_enhanced_transcript)
            wer_records.append({
                "Configuration": "Enhanced Speech + Context Correction",
                "Word Error Rate": f"{enh_corr_eval.wer * 100:.1f}%",
                "Substitutions": enh_corr_eval.substitutions,
                "Deletions": enh_corr_eval.deletions,
                "Insertions": enh_corr_eval.insertions,
                "Hits": enh_corr_eval.hits,
            })

        st.dataframe(pd.DataFrame(wer_records), width="stretch")
    elif is_live_recording:
        st.caption("Add a reference transcript to calculate word error rate.")

    st.download_button("Download results (JSON)", json.dumps(result.to_dict(), indent=2),
                       file_name="speech_demo_results.json", mime="application/json")
    if result.enhanced_audio is not None:
        st.download_button("Download enhanced audio", tensor_to_wav_bytes(result.enhanced_audio, result.sample_rate),
                           file_name="enhanced_speech.wav", mime="audio/wav")


# ---------------------------------------------------------------------------
# Demo controls
# ---------------------------------------------------------------------------
default_ckpt = str(PROJECT_ROOT / "results/enhancement_train/checkpoint_best.pt")
ckpt_ok = Path(default_ckpt).is_file()
for option, default in (("demo_enhancement", ckpt_ok), ("demo_correction", True),
                        ("live_enhancement", ckpt_ok), ("live_correction", True)):
    st.session_state.setdefault(option, default)

st.markdown('<h1 class="app-brand">Speech Enhancement</h1>', unsafe_allow_html=True)
st.caption("Upload or record speech. Compare the audio and transcripts.")

with st.expander("Settings"):
    device_opts = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    selected_device = st.selectbox("Compute Device", options=device_opts)
    if st.button("Load models for demo"):
        try:
            with st.spinner("Loading models..."):
                get_cached_pipeline_runner(default_ckpt, "base.en", selected_device).prepare(run_enhancement=ckpt_ok)
            st.success("Models loaded. Ready to process speech.")
        except Exception as exc:
            st.error(f"Model preparation failed: {exc}")
            logger.exception("Model preparation failure")
    st.caption("Models load automatically when you process audio. You can load them here before the demo.")

active_tab = st.segmented_control(
    "Audio input", key="navigation",
    options=["Audio files", "Record audio"], default="Audio files",
    label_visibility="collapsed",
) or "Audio files"


# ===========================================================================
# PAGE 1: DEMO WORKSPACE
# ===========================================================================
if active_tab == "Audio files":
    col_src, col_opts = st.columns([2, 1])

    with col_opts:
        st.markdown('<div class="workspace-header">Processing</div>', unsafe_allow_html=True)
        run_enhancement = st.toggle(
            "Enhance speech",
            key="demo_enhancement",
            disabled=not ckpt_ok,
            help="Removes background acoustic noise via 2D magnitude spectrogram U-Net.",
        )
        run_correction = st.toggle(
            "Correct transcript",
            key="demo_correction",
            help="GPT-2 and CMUdict propose phonetic replacements and retain words unless the correction safeguards pass.",
        )

    with col_src:
        source_mode = st.radio(
            "Audio Source",
            ["Sample audio", "Upload audio"],
            key="demo_source",
            horizontal=True,
            label_visibility="collapsed",
        )

        audio_source: Any = None
        reference_text: str | None = None

        if source_mode == "Upload audio":
            uploaded = st.file_uploader(
                "Upload speech audio (WAV, MP3, FLAC, OGG)",
                type=["wav", "mp3", "flac", "ogg"],
                key="demo_upload",
                on_change=remember_audio,
                args=("demo_upload", "saved_upload"),
            )
            if uploaded is not None:
                audio_source = uploaded
                st.session_state["saved_upload"] = uploaded.getvalue()
            elif st.session_state.get("saved_upload") is not None:
                st.caption("Using your last uploaded audio. Upload a new file to replace it.")
                if st.button("Clear saved upload"):
                    st.session_state.pop("saved_upload", None)
                else:
                    audio_source = st.session_state["saved_upload"]
            custom_ref = st.text_input("Reference transcript (optional)", key="demo_reference")
            if custom_ref.strip():
                reference_text = custom_ref.strip()
        else:
            sample_dir = PROJECT_ROOT / "data/processed/voicebank_demand_16k/noisy_testset_wav"
            transcripts_dir = PROJECT_ROOT / "data/processed/voicebank_demand_16k/testset_txt"

            available = sorted([p.name for p in sample_dir.glob("*.wav")]) if sample_dir.is_dir() else []
            if available:
                curated = [
                    s for s in ["p232_032.wav", "p257_096.wav", "p232_001.wav", "p232_043.wav", "p257_010.wav", "p257_024.wav"]
                    if s in available
                ] or available[:8]

                selected_file = st.selectbox("Choose a sample", options=curated, key="demo_sample")
                audio_source = sample_dir / selected_file

                ref_file = transcripts_dir / f"{Path(selected_file).stem}.txt"
                if ref_file.is_file():
                    with open(ref_file, "r", encoding="utf-8") as f:
                        reference_text = f.read().strip()
            else:
                st.warning("No sample files found in data directory. Please upload an audio file.")

    if audio_source is not None:
        try:
            tensor_preview, prev_sr = load_audio_from_source(audio_source)
            wav_bytes_preview = tensor_to_wav_bytes(tensor_preview, prev_sr)
            signature = (hashlib.sha256(wav_bytes_preview).hexdigest(), run_enhancement, run_correction, selected_device)

            st.markdown('<div class="workspace-header">Audio preview</div>', unsafe_allow_html=True)
            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.audio(wav_bytes_preview, format="audio/wav")
                st.caption(f"Length: {len(tensor_preview)/prev_sr:.2f}s &bull; Sample Rate: {prev_sr} Hz mono")
            with p_col2:
                if reference_text:
                    st.markdown(
                        f'<div class="transcript-card" style="font-size: 0.88rem; color: #94a3b8;">'
                        f'<strong>Reference transcript:</strong><br>{escape(reference_text)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if st.button("Process Audio", type="primary", width="stretch"):
                runner = get_cached_pipeline_runner(
                    checkpoint_path=default_ckpt,
                    whisper_model_name="base.en",
                    device=selected_device,
                )

                with st.spinner("Enhancing and transcribing audio..."):
                    exec_result: PipelineResult = runner.process(
                        audio_source=audio_source,
                        run_enhancement=run_enhancement,
                        run_correction=run_correction,
                    )

                st.session_state["demo_result"] = (signature, exec_result)

            saved = st.session_state.get("demo_result")
            if saved and saved[0] == signature:
                render_pipeline_outputs(
                    result=saved[1],
                    input_wav_bytes=wav_bytes_preview,
                    reference_text=reference_text,
                    run_enhancement=run_enhancement,
                    run_correction=run_correction,
                    is_live_recording=False,
                )

        except Exception as exc:
            st.error(f"Execution Error: {exc}")
            logger.exception("Demo pipeline failure")


# ===========================================================================
# PAGE 2: LIVE RECORD STUDIO
# ===========================================================================
elif active_tab == "Record audio":
    r_col_rec, r_col_settings = st.columns([2, 1])

    with r_col_settings:
        st.markdown('<div class="workspace-header">Processing</div>', unsafe_allow_html=True)
        live_enh = st.toggle("Enhance speech", disabled=not ckpt_ok, key="live_enhancement")
        live_corr = st.toggle("Correct transcript", key="live_correction")

    with r_col_rec:
        st.markdown('<div class="workspace-header">Microphone Capture</div>', unsafe_allow_html=True)
        live_audio = st.audio_input("Record Audio", sample_rate=16000, key="live_audio",
                                   on_change=remember_audio, args=("live_audio", "saved_recording"))
        if live_audio is not None:
            st.session_state["saved_recording"] = live_audio.getvalue()
        elif st.session_state.get("saved_recording") is not None:
            st.caption("Using your last recording. Record again to replace it.")
            if st.button("Clear saved recording"):
                st.session_state.pop("saved_recording", None)
            else:
                live_audio = st.session_state["saved_recording"]
        optional_user_ref = st.text_input("Reference transcript (optional)", key="live_reference")
        ref_text = optional_user_ref.strip() if optional_user_ref.strip() else None

    if live_audio is not None:
        try:
            live_tensor, live_sr = load_audio_from_source(live_audio)
            live_bytes = tensor_to_wav_bytes(live_tensor, live_sr)
            signature = (hashlib.sha256(live_bytes).hexdigest(), live_enh, live_corr, selected_device)

            st.markdown('<div class="workspace-header">Audio preview</div>', unsafe_allow_html=True)
            prev_col1, prev_col2 = st.columns([1, 2])
            with prev_col1:
                st.audio(live_bytes, format="audio/wav")
                st.caption(f"Recorded: {len(live_tensor)/live_sr:.2f}s @ {live_sr} Hz")
            with prev_col2:
                if ref_text:
                    st.success(f"Reference transcript provided: '{ref_text}'")
                else:
                    st.caption("Add a reference transcript to compare word errors.")

            if st.button("Process Recording", type="primary", width="stretch"):
                runner = get_cached_pipeline_runner(
                    checkpoint_path=default_ckpt,
                    whisper_model_name="base.en",
                    device=selected_device,
                )

                with st.spinner("Enhancing and transcribing recording..."):
                    live_res: PipelineResult = runner.process(
                        audio_source=live_bytes,
                        run_enhancement=live_enh,
                        run_correction=live_corr,
                    )

                st.session_state["live_result"] = (signature, live_res)

            saved = st.session_state.get("live_result")
            if saved and saved[0] == signature:
                render_pipeline_outputs(
                    result=saved[1],
                    input_wav_bytes=live_bytes,
                    reference_text=ref_text,
                    run_enhancement=live_enh,
                    run_correction=live_corr,
                    is_live_recording=True,
                )

        except Exception as exc:
            st.error(f"Recording Error: {exc}")
            logger.exception("Live recording failure")
