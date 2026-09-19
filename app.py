"""Production-grade Streamlit application for Speech Enhancement and Context-Aware ASR Correction.

Provides:
- Interactive Demo: Professional workspace for audio enhancement, Whisper ASR, and context correction.
- Live Record: Focused browser microphone recording studio with real-time pipeline inference.
- Research Results: Analytical benchmark dashboard with verified experimental metrics and paired statistical tests.
- Methodology: Technical documentation detailing model architectures, configurations, and limitations.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

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
    page_title="Speech & ASR Pipeline | AI Speech Enhancement & Error Correction",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Modern AI SaaS Visual System */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

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
    raw_np = raw_audio.detach().cpu().squeeze().numpy()
    time_raw = np.linspace(0, len(raw_np) / sr, len(raw_np))

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
    axes[1, 0].specgram(raw_np, Fs=sr, NFFT=512, noverlap=256, cmap="magma")
    axes[1, 0].set_title("Input Spectrogram (STFT)")
    axes[1, 0].set_xlabel("Time (s)", fontsize=9)
    axes[1, 0].set_ylabel("Frequency (Hz)", fontsize=9)

    if has_enh and enh_audio is not None:
        enh_np = enh_audio.detach().cpu().squeeze().numpy()
        time_enh = np.linspace(0, len(enh_np) / sr, len(enh_np))

        # Enhanced Waveform
        axes[0, 1].plot(time_enh, enh_np, color="#34d399", linewidth=0.75, alpha=0.9)
        axes[0, 1].set_title("U-Net Enhanced Waveform")
        axes[0, 1].set_ylabel("Amplitude", fontsize=9)
        axes[0, 1].set_ylim(-1.05, 1.05)
        axes[0, 1].grid(True, color=(1.0, 1.0, 1.0), alpha=0.06, linestyle="--", linewidth=0.5)

        # Enhanced Spectrogram
        axes[1, 1].specgram(enh_np, Fs=sr, NFFT=512, noverlap=256, cmap="magma")
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
    st.markdown("### Comparative Transcription & Correction Outputs")
    st.markdown(
        '<div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">'
        '<strong style="color: #f8fafc;">Pipeline Architecture:</strong> Evaluates two independent parallel branches on the same audio. '
        '<strong style="color: #38bdf8;">Branch A</strong> feeds raw audio directly to Whisper. '
        '<strong style="color: #34d399;">Branch B</strong> enhances audio waveforms via 2D U-Net (STFT magnitude domain) <em>before</em> Whisper decoding. '
        'Context-aware correction operates strictly as a <strong>post-ASR text-domain stage</strong> on the generated transcripts.'
        '</div>',
        unsafe_allow_html=True,
    )
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown("#### Branch A: Raw Audio Pipeline (Baseline)")
        st.caption("Flow: Input Audio &rarr; Whisper base.en &rarr; Post-ASR Correction")
        st.markdown(
            '<div class="transcript-badge badge-raw">1. Raw ASR Transcript &bull; Whisper base.en</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="transcript-card">{result.raw_transcript or "(no speech detected)"}</div>',
            unsafe_allow_html=True,
        )

        if run_correction and result.corrected_raw_transcript:
            st.markdown(
                '<div class="transcript-badge badge-corr">2. Post-ASR Context Correction &bull; GPT-2 + CMUdict</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="transcript-card">{result.corrected_raw_transcript}</div>',
                unsafe_allow_html=True,
            )

            if result.raw_corrections:
                st.markdown("**Corrected Words in Raw Transcript:**")
                for c in result.raw_corrections:
                    dec = c.get("decision", {})
                    sim = dec.get("phonetic_similarity", 0.0)
                    st.markdown(
                        f'<div class="diff-chip">'
                        f'<span class="diff-from">{c["original_token"]}</span>'
                        f'<span class="diff-arrow">&rarr;</span>'
                        f'<span class="diff-to">{c["corrected_token"]}</span>'
                        f'<span class="diff-meta">(conf: {dec.get("confidence", 0):.2f}, sim: {sim:.2f}, {dec.get("reason")})</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="font-size: 0.82rem; color: #94a3b8; font-style: italic; margin-top: 0.35rem;">'
                    'No correction applied (all words retained under confidence and phonetic similarity safeguards).'
                    '</div>',
                    unsafe_allow_html=True,
                )

    with t_col2:
        st.markdown("#### Branch B: Enhanced Audio Pipeline")
        st.caption("Flow: Input Audio &rarr; U-Net (STFT) &rarr; Enhanced Audio &rarr; Whisper &rarr; Post-ASR Correction")
        if result.enhanced_transcript is not None:
            st.markdown(
                '<div class="transcript-badge badge-enh">1. Enhanced Audio ASR &bull; Whisper base.en</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="transcript-card">{result.enhanced_transcript or "(no speech detected)"}</div>',
                unsafe_allow_html=True,
            )

            if run_correction and result.corrected_enhanced_transcript:
                st.markdown(
                    '<div class="transcript-badge badge-corr">2. Post-ASR Context Correction &bull; GPT-2 + CMUdict</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="transcript-card">{result.corrected_enhanced_transcript}</div>',
                    unsafe_allow_html=True,
                )

                if result.enhanced_corrections:
                    st.markdown("**Corrected Words in Enhanced Transcript:**")
                    for c in result.enhanced_corrections:
                        dec = c.get("decision", {})
                        sim = dec.get("phonetic_similarity", 0.0)
                        st.markdown(
                            f'<div class="diff-chip">'
                            f'<span class="diff-from">{c["original_token"]}</span>'
                            f'<span class="diff-arrow">&rarr;</span>'
                            f'<span class="diff-to">{c["corrected_token"]}</span>'
                            f'<span class="diff-meta">(conf: {dec.get("confidence", 0):.2f}, sim: {sim:.2f}, {dec.get("reason")})</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div style="font-size: 0.82rem; color: #94a3b8; font-style: italic; margin-top: 0.35rem;">'
                        'No correction applied (all words retained under confidence and phonetic similarity safeguards).'
                        '</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Branch B (U-Net speech enhancement) was disabled or model checkpoint unavailable.")

    # 2. Audio & Signal Comparison
    st.markdown("### Audio Signals & Spectral Analysis")
    st.caption("Waveforms and STFT spectrograms synthesized in the audio domain. The 2D U-Net enhances audio waveforms prior to Whisper ASR decoding.")
    a_col1, a_col2 = st.columns(2)

    with a_col1:
        st.markdown("**Input Audio Playback**")
        st.audio(input_wav_bytes, format="audio/wav")

    with a_col2:
        if result.enhanced_audio is not None:
            st.markdown("**U-Net Enhanced Audio Playback**")
            enh_bytes = tensor_to_wav_bytes(result.enhanced_audio, result.sample_rate)
            st.audio(enh_bytes, format="audio/wav")
        else:
            st.info("Speech enhancement was disabled or model checkpoint unavailable.")

    fig = plot_audio_comparison(result.input_audio, result.enhanced_audio, result.sample_rate)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # 3. Telemetry & Performance
    st.markdown("### Latency & Telemetry")
    timing_cols = st.columns(4)
    with timing_cols[0]:
        st.metric("Preprocessing", f"{result.timing.preprocessing_s * 1000:.1f} ms")
    with timing_cols[1]:
        st.metric("Enhancement", f"{result.timing.enhancement_s * 1000:.1f} ms" if result.timing.enhancement_s > 0 else "Skipped")
    with timing_cols[2]:
        st.metric("Whisper ASR", f"{result.timing.asr_raw_s:.2f} s")
    with timing_cols[3]:
        st.metric("Total Latency", f"{result.timing.total_s:.2f} s")

    # 4. Word Error Rate (if reference supplied)
    if reference_text:
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

        if result.corrected_raw_transcript:
            raw_corr_eval = compute_wer(reference_text, result.corrected_raw_transcript)
            wer_records.append({
                "Configuration": "Raw Speech + Context Correction",
                "Word Error Rate": f"{raw_corr_eval.wer * 100:.1f}%",
                "Substitutions": raw_corr_eval.substitutions,
                "Deletions": raw_corr_eval.deletions,
                "Insertions": raw_corr_eval.insertions,
                "Hits": raw_corr_eval.hits,
            })

        if result.enhanced_transcript:
            enh_eval = compute_wer(reference_text, result.enhanced_transcript)
            wer_records.append({
                "Configuration": "U-Net Enhanced Speech",
                "Word Error Rate": f"{enh_eval.wer * 100:.1f}%",
                "Substitutions": enh_eval.substitutions,
                "Deletions": enh_eval.deletions,
                "Insertions": enh_eval.insertions,
                "Hits": enh_eval.hits,
            })

        if result.corrected_enhanced_transcript:
            enh_corr_eval = compute_wer(reference_text, result.corrected_enhanced_transcript)
            wer_records.append({
                "Configuration": "Enhanced Speech + Context Correction",
                "Word Error Rate": f"{enh_corr_eval.wer * 100:.1f}%",
                "Substitutions": enh_corr_eval.substitutions,
                "Deletions": enh_corr_eval.deletions,
                "Insertions": enh_corr_eval.insertions,
                "Hits": enh_corr_eval.hits,
            })

        st.dataframe(pd.DataFrame(wer_records), use_container_width=True)
    elif is_live_recording:
        st.caption("Live microphone mode operates in qualitative demonstration mode. Enter reference text to compute WER.")


# ---------------------------------------------------------------------------
# Sidebar: System Environment & Device Control
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### System Environment")
    device_avail = "cuda" if torch.cuda.is_available() else "cpu"
    device_opts = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    selected_device = st.selectbox(
        "Compute Device",
        options=device_opts,
        index=0,
        help="Selects device for PyTorch inference (RTX 3050 CUDA or CPU).",
    )

    default_ckpt = "results/enhancement_train/checkpoint_best.pt"
    ckpt_ok = Path(default_ckpt).is_file()
    ckpt_badge = '<span style="color: #4ade80;">Ready</span>' if ckpt_ok else '<span style="color: #f87171;">Missing</span>'
    hw_str = "NVIDIA RTX 3050 (6GB)" if torch.cuda.is_available() else "CPU"

    st.markdown(
        f'<div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">'
        f'<strong>Hardware:</strong> {hw_str}<br>'
        f'<strong>Whisper Model:</strong> <code>base.en</code> (greedy)<br>'
        f'<strong>U-Net Checkpoint:</strong> {ckpt_badge}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    with st.expander("Telemetry & Memory Footprint"):
        st.markdown(
            "- **VRAM Budget:** $\\approx 1.5\\text{ GB}$ peak\n"
            "- **ASR Framework:** OpenAI Whisper\n"
            "- **LM Scorer:** HuggingFace `gpt2` (local)\n"
            "- **U-Net Params:** $1.76\\text{ M}$ parameters"
        )


# ---------------------------------------------------------------------------
# Application Header & Clean Navigation
# ---------------------------------------------------------------------------
cuda_label = "CUDA &bull; RTX 3050" if selected_device == "cuda" else "CPU"
st.markdown(
    f"""
    <div class="app-hero">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <h1 class="app-brand">Speech & ASR Pipeline</h1>
                <div class="app-tagline">Deep-learning speech enhancement with context-aware ASR error correction.</div>
            </div>
            <div class="status-pill">
                <span class="status-dot"></span>
                <span>{cuda_label}</span>
            </div>
        </div>
        <div class="pipeline-flow" style="display: flex; flex-direction: column; align-items: flex-start; gap: 0.4rem; padding: 0.65rem 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                <span style="color: #38bdf8; font-weight: 600; font-size: 0.78rem;">Branch A (Raw):</span>
                <span class="flow-step">Raw Audio</span>
                <span class="flow-arrow">&rarr;</span>
                <span class="flow-step">Whisper base.en</span>
                <span class="flow-arrow">&rarr;</span>
                <span class="flow-step">Post-ASR Correction</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                <span style="color: #34d399; font-weight: 600; font-size: 0.78rem;">Branch B (Enhanced):</span>
                <span class="flow-step">Raw Audio</span>
                <span class="flow-arrow">&rarr;</span>
                <span class="flow-step" style="border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 4px; padding: 0.1rem 0.4rem; color: #34d399;">2D U-Net Audio Enhancement</span>
                <span class="flow-arrow">&rarr;</span>
                <span class="flow-step">Whisper base.en</span>
                <span class="flow-arrow">&rarr;</span>
                <span class="flow-step">Post-ASR Correction</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

active_tab = st.segmented_control(
    "Navigation",
    options=["Demo", "Live Record", "Research Results", "Methodology"],
    default="Demo",
    label_visibility="collapsed",
)
if active_tab is None:
    active_tab = "Demo"


# ===========================================================================
# PAGE 1: DEMO WORKSPACE
# ===========================================================================
if active_tab == "Demo":
    st.markdown("### Interactive Workspace")
    st.markdown(
        "Evaluate speech samples through the two comparative pipeline branches. "
        "Select a verified benchmark utterance from VoiceBank+DEMAND or upload a custom audio file."
    )

    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 1.25rem; font-size: 0.88rem; line-height: 1.55; color: #cbd5e1;">
            <div style="font-weight: 600; color: #f8fafc; margin-bottom: 0.35rem;">
                Pipeline Architecture & Comparative Evaluation
            </div>
            <div>
                This system evaluates two independent pathways on the same input audio:
                <br>&bull; <strong style="color: #38bdf8;">Branch A (Raw Baseline):</strong> Raw Audio &rarr; Whisper <code>base.en</code> &rarr; Raw Transcript &rarr; Post-ASR Context Correction.
                <br>&bull; <strong style="color: #34d399;">Branch B (Enhanced):</strong> Raw Audio &rarr; <strong>2D U-Net Audio Enhancement</strong> &rarr; Enhanced Audio &rarr; Whisper <code>base.en</code> &rarr; Enhanced Transcript &rarr; Post-ASR Context Correction.
            </div>
            <div style="margin-top: 0.55rem; padding-top: 0.55rem; border-top: 1px solid rgba(255, 255, 255, 0.08); font-size: 0.82rem; color: #94a3b8;">
                <strong style="color: #38bdf8;">Audio vs. Text Domain:</strong> The U-Net operates strictly on <em>acoustic waveforms</em> in the STFT magnitude domain to enhance audio <em>before</em> Whisper ASR transcription. It does not operate on text. Context-aware phonetic correction operates strictly on <em>transcribed text</em> after ASR decoding.
                <br><strong style="color: #f59e0b;">Verified Benchmark Finding:</strong> On 824 held-out test sentences, U-Net enhancement delivered marked acoustic gains (+9.60 dB SNR, +9.51 dB SI-SDR, +0.0127 STOI) but did <em>not</em> improve Whisper WER (7.23% &rarr; 8.23%). The post-ASR corrector provided a small, conservative error reduction (-0.11 pp WER), and the full pipeline (8.11% WER) did not outperform unprocessed noisy speech.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_src, col_opts = st.columns([2, 1])

    with col_opts:
        st.markdown('<div class="workspace-header">Pipeline Controls</div>', unsafe_allow_html=True)
        run_enhancement = st.toggle(
            "Speech Enhancement (U-Net)",
            value=ckpt_ok,
            disabled=not ckpt_ok,
            help="Removes background acoustic noise via 2D magnitude spectrogram U-Net.",
        )
        run_correction = st.toggle(
            "Context Correction (GPT-2 + Phonetics)",
            value=True,
            help="Performs phonetic candidate generation and context rescoring with Bayesian safeguards.",
        )

    with col_src:
        source_mode = st.radio(
            "Audio Source",
            ["Benchmark Utterance (VoiceBank+DEMAND)", "Upload Audio File"],
            horizontal=True,
            label_visibility="collapsed",
        )

        audio_source: Any = None
        reference_text: str | None = None

        if source_mode == "Upload Audio File":
            uploaded = st.file_uploader(
                "Upload speech audio (WAV, MP3, FLAC, OGG)",
                type=["wav", "mp3", "flac", "ogg"],
            )
            if uploaded:
                audio_source = uploaded
            custom_ref = st.text_input("Optional Reference Transcript:")
            if custom_ref.strip():
                reference_text = custom_ref.strip()
        else:
            sample_dir = Path("data/processed/voicebank_demand_16k/noisy_testset_wav")
            transcripts_dir = Path("data/processed/voicebank_demand_16k/testset_txt")

            available = sorted([p.name for p in sample_dir.glob("*.wav")]) if sample_dir.is_dir() else []
            if available:
                curated = [
                    s for s in ["p232_001.wav", "p232_015.wav", "p232_020.wav", "p257_001.wav", "p257_010.wav", "p257_024.wav"]
                    if s in available
                ] or available[:8]

                selected_file = st.selectbox("Select Test Sample", options=curated)
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

            st.markdown('<div class="workspace-header">Input Audio Card</div>', unsafe_allow_html=True)
            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.audio(wav_bytes_preview, format="audio/wav")
                st.caption(f"Length: {len(tensor_preview)/prev_sr:.2f}s &bull; Sample Rate: {prev_sr} Hz mono")
            with p_col2:
                if reference_text:
                    st.markdown(
                        f'<div class="transcript-card" style="font-size: 0.88rem; color: #94a3b8;">'
                        f'<strong>Ground-Truth Reference:</strong><br>{reference_text}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with st.expander("Input Signal Waveform & Spectrogram (Preview)", expanded=False):
                fig_prev = plot_audio_comparison(tensor_preview, None, prev_sr)
                st.pyplot(fig_prev, use_container_width=True)
                plt.close(fig_prev)

            if st.button("Process Audio", type="primary", use_container_width=True):
                runner = get_cached_pipeline_runner(
                    checkpoint_path=default_ckpt,
                    whisper_model_name="base.en",
                    device=selected_device,
                )

                with st.spinner("Processing audio through neural enhancement, Whisper ASR, and context rescoring..."):
                    exec_result: PipelineResult = runner.process(
                        audio_source=audio_source,
                        run_enhancement=run_enhancement,
                        run_correction=run_correction,
                    )

                render_pipeline_outputs(
                    result=exec_result,
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
elif active_tab == "Live Record":
    st.markdown("### Microphone Recording Studio")
    st.markdown(
        "Capture live microphone audio directly in your browser to evaluate real-time speech enhancement, "
        "Whisper ASR, and context-aware post-processing on your acoustic environment."
    )

    r_col_rec, r_col_settings = st.columns([2, 1])

    with r_col_settings:
        st.markdown('<div class="workspace-header">Studio Settings</div>', unsafe_allow_html=True)
        live_enh = st.toggle("U-Net Speech Enhancement", value=ckpt_ok, disabled=not ckpt_ok)
        live_corr = st.toggle("Context-Aware Correction", value=True)

    with r_col_rec:
        st.markdown('<div class="workspace-header">Microphone Capture</div>', unsafe_allow_html=True)
        live_audio = st.audio_input("Record Audio", sample_rate=16000)
        optional_user_ref = st.text_input("Reference Transcript (Optional, to evaluate exact WER):")
        ref_text = optional_user_ref.strip() if optional_user_ref.strip() else None

    if live_audio is not None:
        try:
            live_tensor, live_sr = load_audio_from_source(live_audio)
            live_bytes = tensor_to_wav_bytes(live_tensor, live_sr)

            st.markdown('<div class="workspace-header">Recorded Audio Preview</div>', unsafe_allow_html=True)
            prev_col1, prev_col2 = st.columns([1, 2])
            with prev_col1:
                st.audio(live_bytes, format="audio/wav")
                st.caption(f"Recorded: {len(live_tensor)/live_sr:.2f}s @ {live_sr} Hz")
            with prev_col2:
                if ref_text:
                    st.success(f"Reference transcript provided: '{ref_text}'")
                else:
                    st.caption("Qualitative demo mode active (no reference supplied).")

            with st.expander("Recorded Signal Waveform & Spectrogram (Preview)", expanded=False):
                fig_prev = plot_audio_comparison(live_tensor, None, live_sr)
                st.pyplot(fig_prev, use_container_width=True)
                plt.close(fig_prev)

            if st.button("Process Recording", type="primary", use_container_width=True):
                runner = get_cached_pipeline_runner(
                    checkpoint_path=default_ckpt,
                    whisper_model_name="base.en",
                    device=selected_device,
                )

                with st.spinner("Processing recording through enhancement, Whisper, and context rescoring..."):
                    live_res: PipelineResult = runner.process(
                        audio_source=live_bytes,
                        run_enhancement=live_enh,
                        run_correction=live_corr,
                    )

                render_pipeline_outputs(
                    result=live_res,
                    input_wav_bytes=live_bytes,
                    reference_text=ref_text,
                    run_enhancement=live_enh,
                    run_correction=live_corr,
                    is_live_recording=True,
                )

        except Exception as exc:
            st.error(f"Recording Error: {exc}")
            logger.exception("Live recording failure")


# ===========================================================================
# PAGE 3: RESEARCH RESULTS
# ===========================================================================
elif active_tab == "Research Results":
    st.markdown("### Experimental Benchmark & Statistical Analysis")
    st.markdown(
        "Verified experimental benchmark measured across all **824 held-out test utterances** ($6,211$ words; "
        "speakers `p232` and `p257`) of VoiceBank+DEMAND against Edinburgh DataShare official references."
    )

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            '<div class="kpi-metric-box">'
            '<div class="kpi-title">Clean Speech Baseline</div>'
            '<div class="kpi-value">4.65%</div>'
            '<div class="kpi-sub">Whisper base.en &bull; Clean audio</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            '<div class="kpi-metric-box">'
            '<div class="kpi-title">Noisy Speech Baseline</div>'
            '<div class="kpi-value">7.23%</div>'
            '<div class="kpi-sub">Unprocessed audio &bull; 8.45 dB SNR</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            '<div class="kpi-metric-box">'
            '<div class="kpi-title">U-Net Enhanced</div>'
            '<div class="kpi-value" style="color: #f87171;">8.23%</div>'
            '<div class="kpi-sub">+1.00 pp WER degradation (p = 0.0022)</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            '<div class="kpi-metric-box">'
            '<div class="kpi-title">Enhanced + Correction</div>'
            '<div class="kpi-value" style="color: #38bdf8;">8.11%</div>'
            '<div class="kpi-sub">-0.12 pp vs U-Net raw (p = 0.0382)</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Section 1: Benchmark Table
    st.markdown("#### 1. Consolidated Performance Table (824 Utterances, 6,211 Words)")
    benchmark_df = pd.DataFrame([
        {
            "Condition": "Clean Speech Baseline",
            "Pipeline": "Clean Audio → Whisper base.en",
            "Corpus WER": "4.65%",
            "Utt WER": "5.05%",
            "Sub (S)": 202,
            "Del (D)": 55,
            "Ins (I)": 32,
            "Hits (C)": 5954,
            "Total Errors": 289,
            "STOI": 1.0000,
            "SI-SDR": "N/A",
            "SNR": "N/A",
        },
        {
            "Condition": "Noisy Speech Baseline",
            "Pipeline": "Noisy Audio → Whisper base.en",
            "Corpus WER": "7.23%",
            "Utt WER": "7.59%",
            "Sub (S)": 330,
            "Del (D)": 69,
            "Ins (I)": 50,
            "Hits (C)": 5812,
            "Total Errors": 449,
            "STOI": 0.9211,
            "SI-SDR": "8.45 dB",
            "SNR": "8.45 dB",
        },
        {
            "Condition": "U-Net Enhanced",
            "Pipeline": "Noisy → U-Net → Whisper base.en",
            "Corpus WER": "8.23%",
            "Utt WER": "8.78%",
            "Sub (S)": 369,
            "Del (D)": 80,
            "Ins (I)": 62,
            "Hits (C)": 5762,
            "Total Errors": 511,
            "STOI": 0.9338,
            "SI-SDR": "17.96 dB",
            "SNR": "18.05 dB",
        },
        {
            "Condition": "Noisy + Improved Correction",
            "Pipeline": "Noisy → Whisper → Improved Corrector",
            "Corpus WER": "7.12%",
            "Utt WER": "7.51%",
            "Sub (S)": 323,
            "Del (D)": 69,
            "Ins (I)": 50,
            "Hits (C)": 5819,
            "Total Errors": 442,
            "STOI": 0.9211,
            "SI-SDR": "8.45 dB",
            "SNR": "8.45 dB",
        },
        {
            "Condition": "Enhanced + Improved Correction",
            "Pipeline": "Noisy → U-Net → Whisper → Improved Corrector",
            "Corpus WER": "8.11%",
            "Utt WER": "8.66%",
            "Sub (S)": 362,
            "Del (D)": 80,
            "Ins (I)": 62,
            "Hits (C)": 5769,
            "Total Errors": 504,
            "STOI": 0.9338,
            "SI-SDR": "17.96 dB",
            "SNR": "18.05 dB",
        },
    ])
    st.dataframe(benchmark_df, use_container_width=True)

    # Section 2: Key Discovery
    st.markdown("#### 2. The Enhancement-ASR Disconnect")
    disc_col1, disc_col2 = st.columns(2)
    with disc_col1:
        st.markdown(
            '<div class="workspace-card">'
            '<div class="workspace-header" style="color: #4ade80;">Acoustic Metric Gains</div>'
            '<p style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.6;">'
            'The 2D U-Net magnitude enhancer dramatically improved acoustic signal quality:<br>'
            '&bull; <strong>SNR Gain:</strong> +9.60 dB (8.45 dB &rarr; 18.05 dB)<br>'
            '&bull; <strong>SI-SDR Gain:</strong> +9.51 dB (8.45 dB &rarr; 17.96 dB)<br>'
            '&bull; <strong>STOI Improvement:</strong> +0.0127 (0.9211 &rarr; 0.9338)'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with disc_col2:
        st.markdown(
            '<div class="workspace-card">'
            '<div class="workspace-header" style="color: #f87171;">Downstream ASR Degradation</div>'
            '<p style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.6;">'
            'Despite cleaner audio, downstream Whisper WER significantly worsened:<br>'
            '&bull; <strong>Corpus WER:</strong> 7.23% &rarr; 8.23% (+1.00 pp, p = 0.0022)<br>'
            '&bull; <strong>Utterance Outcomes:</strong> 70 degraded vs 37 improved<br>'
            '&bull; <strong>Causal Factors:</strong> Reusing noisy phase creates musical synthesis artifacts that distort learned transformer acoustic tokens.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Section 3: Statistical Significance Matrix
    st.markdown("#### 3. Paired Statistical Significance (10,000 Bootstrap Replications & Wilcoxon)")
    stats_df = pd.DataFrame([
        {
            "Comparison": "Noisy Raw vs Noisy Improved-Corr",
            "Sample ΔWER": "-0.11 pp",
            "95% Bootstrap CI": "[-0.22%, -0.02%]",
            "Bootstrap p-val": "0.0356",
            "Bootstrap Sig (α=0.05)": "Yes",
            "Wilcoxon p-val": "0.1542",
            "Wilcoxon Sig (α=0.05)": "No",
            "Outcomes": "9 improved / 2 worsened / 813 tied",
        },
        {
            "Comparison": "Enhanced Raw vs Enhanced Improved-Corr",
            "Sample ΔWER": "-0.11 pp",
            "95% Bootstrap CI": "[-0.22%, -0.02%]",
            "Bootstrap p-val": "0.0382",
            "Bootstrap Sig (α=0.05)": "Yes",
            "Wilcoxon p-val": "0.0754",
            "Wilcoxon Sig (α=0.05)": "No",
            "Outcomes": "9 improved / 2 worsened / 813 tied",
        },
        {
            "Comparison": "Noisy Raw vs Enhanced Raw",
            "Sample ΔWER": "+1.00 pp",
            "95% Bootstrap CI": "[+0.35%, +1.69%]",
            "Bootstrap p-val": "0.0022",
            "Bootstrap Sig (α=0.05)": "Yes",
            "Wilcoxon p-val": "0.0066",
            "Wilcoxon Sig (α=0.05)": "Yes",
            "Outcomes": "37 improved / 70 worsened / 717 tied",
        },
        {
            "Comparison": "Noisy Raw vs Enhanced Improved-Corr",
            "Sample ΔWER": "+0.89 pp",
            "95% Bootstrap CI": "[+0.22%, +1.58%]",
            "Bootstrap p-val": "0.0094",
            "Bootstrap Sig (α=0.05)": "Yes",
            "Wilcoxon p-val": "0.0213",
            "Wilcoxon Sig (α=0.05)": "Yes",
            "Outcomes": "45 improved / 70 worsened / 709 tied",
        },
    ])
    st.dataframe(stats_df, use_container_width=True)


# ===========================================================================
# PAGE 4: METHODOLOGY
# ===========================================================================
elif active_tab == "Methodology":
    st.markdown("### Technical Architecture & Methodology")
    st.markdown(
        "Technical documentation covering deep-learning model specifications, mathematical formulation, "
        "training objectives, and verified research limitations."
    )

    st.markdown(
        """
```text
                               +---------------------------------------+
                               |        Input Audio Waveform y(t)      |
                               +---------------------------------------+
                                                   |
                        +--------------------------+--------------------------+
                        |                                                     |
             [ Branch A: Raw Baseline ]                            [ Branch B: U-Net Enhanced ]
                        |                                                     |
                        |                                           STFT (n_fft=512, hop=256)
                        |                                                     |
                        |                                           2D Convolutional U-Net
                        |                                           (Magnitude Masking, L1 loss)
                        |                                                     |
                        |                                           iSTFT Synthesis (input phase)
                        |                                                     |
                        |                                           [ Enhanced Audio s_hat(t) ]
                        |                                                     |
               OpenAI Whisper base.en                                OpenAI Whisper base.en
                        |                                                     |
               [ Raw Transcript ]                                    [ Enhanced Transcript ]
                        |                                                     |
          Context-Aware Post-ASR Correction                     Context-Aware Post-ASR Correction
          (CMUdict + GPT-2 + Safeguards)                        (CMUdict + GPT-2 + Safeguards)
                        |                                                     |
          +----------------------------+                        +-------------------------------+
          | Final Corrected Transcript |                        | Final Corrected Transcript    |
          +----------------------------+                        +-------------------------------+
```
        """
    )

    st.markdown(
        '<div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: #94a3b8;">'
        '<strong style="color: #38bdf8;">Architectural Domain Separation:</strong> The 2D U-Net operates strictly in the '
        '<strong>audio domain</strong> on STFT magnitude spectrograms to produce cleaned audio prior to Whisper ASR ingestion. '
        'It does <em>not</em> process or correct text. The phonetic/GPT-2 correction module operates strictly in the '
        '<strong>text domain</strong> on the ASR-decoded hypothesis string.'
        '</div>',
        unsafe_allow_html=True,
    )

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown(
            '<div class="workspace-card">'
            '<div class="workspace-header">Component Specifications</div>'
            '<p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">'
            '<strong>1. STFT Preprocessing:</strong> 16 kHz audio, n_fft=512, hop=256, Hann window.<br>'
            '<strong>2. Speech Enhancer:</strong> 2D U-Net with 4 downsampling and 4 upsampling convolutional stages. Channels: 32 &rarr; 64 &rarr; 128 &rarr; 256. Trained with magnitude L1 objective.<br>'
            '<strong>3. Foundation ASR:</strong> OpenAI Whisper <code>base.en</code>, greedy decoding (temperature=0.0).'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            '<div class="workspace-card">'
            '<div class="workspace-header">Correction Safeguards</div>'
            '<p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6;">'
            '<strong>1. Candidate Generation:</strong> CMU Pronouncing Dictionary with phonetic Levenshtein edit distance &le; 2.<br>'
            '<strong>2. Absolute Score Margin (&ge; 3.5):</strong> Rejects candidates unless log-probability improvement strictly exceeds 3.5.<br>'
            '<strong>3. Phonetic Similarity Floor (&ge; 0.80):</strong> Rejects semantically attractive but phonetically distant distractors.<br>'
            '<strong>4. Retention Bias (&beta; = 1.0):</strong> Penalizes non-original alternatives to honor ASR acoustic likelihood.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Documented Research Limitations")
    st.markdown(
        """
1. **Single Enhancement Architecture**: Evaluated on a 2D convolutional U-Net baseline operating on magnitude STFT representations.
2. **Single Downstream ASR Model**: Restricted to OpenAI Whisper `base.en` with greedy decoding.
3. **Dataset Scope**: Evaluated exclusively on VoiceBank+DEMAND synthetically mixed read English speech.
4. **Noisy-Phase Reuse**: Magnitude-only enhancement without phase estimation during waveform synthesis.
5. **Magnitude-L1 Objective**: Optimized for spectral amplitude distance rather than perceptual or ASR-aligned loss functions.
6. **Development Set Size**: Safeguard thresholds calibrated on a 60-utterance synthetic development set.
7. **Small Absolute Correction Gain**: Net reduction of 7 words out of 6,211 words (-0.11 percentage points), leaving 98.67% of utterances unchanged.
8. **PESQ Unavailable on Windows**: Host environment lacked MSVC C++ build tools for the `pesq` C extension.
9. **No Claim of Generalization**: Findings reflect the specific models and evaluation sets tested.
        """
    )
