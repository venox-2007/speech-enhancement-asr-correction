"""Streamlit web application for speech enhancement and context-aware ASR correction.

Provides:
1. Interactive Demo: Live audio upload or sample selection, U-Net enhancement, Whisper ASR, and context correction.
2. Benchmark & Results: The verified offline experimental benchmark across all 824 VoiceBank+DEMAND test utterances.
3. Architecture & Methodology: System design, model configurations, and documented scientific limitations.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.demo.pipeline_runner import (
    PipelineResult,
    PipelineRunner,
    get_default_device,
    load_audio_from_source,
    tensor_to_wav_bytes,
)
from src.evaluation.wer import compute_wer

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Speech Enhancement & ASR Correction",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached Model Lifecycles
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models into memory...")
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
# Waveform & Spectrogram Plotting Helpers
# ---------------------------------------------------------------------------
def plot_audio_comparison(
    raw_audio: torch.Tensor,
    enh_audio: torch.Tensor | None,
    sr: int = 16_000,
) -> plt.Figure:
    """Generate side-by-side waveform and spectrogram comparison plots."""
    raw_np = raw_audio.detach().cpu().squeeze().numpy()
    time_raw = np.linspace(0, len(raw_np) / sr, len(raw_np))

    has_enh = enh_audio is not None
    num_cols = 2 if has_enh else 1

    fig, axes = plt.subplots(2, num_cols, figsize=(7 * num_cols, 5), sharex="col", squeeze=False)
    fig.patch.set_facecolor("#0e1117")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#30363d")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

    # Raw Waveform
    axes[0, 0].plot(time_raw, raw_np, color="#58a6ff", linewidth=0.8)
    axes[0, 0].set_title("Input Audio Waveform")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_ylim(-1.05, 1.05)

    # Raw Spectrogram
    axes[1, 0].specgram(raw_np, Fs=sr, NFFT=512, noverlap=256, cmap="magma")
    axes[1, 0].set_title("Input Spectrogram")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Frequency (Hz)")

    if has_enh and enh_audio is not None:
        enh_np = enh_audio.detach().cpu().squeeze().numpy()
        time_enh = np.linspace(0, len(enh_np) / sr, len(enh_np))

        # Enhanced Waveform
        axes[0, 1].plot(time_enh, enh_np, color="#3fb950", linewidth=0.8)
        axes[0, 1].set_title("U-Net Enhanced Waveform")
        axes[0, 1].set_ylabel("Amplitude")
        axes[0, 1].set_ylim(-1.05, 1.05)

        # Enhanced Spectrogram
        axes[1, 1].specgram(enh_np, Fs=sr, NFFT=512, noverlap=256, cmap="magma")
        axes[1, 1].set_title("U-Net Enhanced Spectrogram")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Frequency (Hz)")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------------------------
st.sidebar.title("🎙️ Speech & ASR Pipeline")
st.sidebar.markdown(
    "Deep-learning speech enhancement paired with context-aware phonetic ASR correction."
)

page = st.sidebar.radio(
    "Navigate",
    [
        "🎙️ Interactive Pipeline Demo",
        "📊 Benchmark & Research Results",
        "🔬 Architecture & Methodology",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Configuration")

default_device = get_default_device()
device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
selected_device = st.sidebar.selectbox(
    "Compute Device",
    options=device_options,
    index=0,
    help="Select GPU (CUDA) or CPU for model execution.",
)

default_ckpt = "results/enhancement_train/checkpoint_best.pt"
checkpoint_exists = Path(default_ckpt).is_file()

st.sidebar.text_input(
    "Enhancement Checkpoint",
    value=default_ckpt,
    disabled=True,
    help="Path to trained U-Net weights.",
)

if checkpoint_exists:
    st.sidebar.success("🟢 U-Net Checkpoint Available")
else:
    st.sidebar.warning("⚠️ Checkpoint Missing (Enhancement unavailable)")

st.sidebar.markdown(
    f"**Whisper Model**: `base.en`  \n"
    f"**CUDA Status**: `{'Available (RTX 3050)' if torch.cuda.is_available() else 'Not available'}`"
)


# ===========================================================================
# PAGE 1: INTERACTIVE PIPELINE DEMO
# ===========================================================================
if page == "🎙️ Interactive Pipeline Demo":
    st.title("🎙️ Live Speech Enhancement & ASR Correction Pipeline")
    st.markdown(
        "Upload an audio recording or pick a benchmark sample to trace speech through "
        "the **4-stage pipeline**: Audio Input $\\to$ U-Net Enhancement $\\to$ Whisper ASR $\\to$ Context-Aware Correction."
    )

    col_input, col_config = st.columns([2, 1])

    with col_config:
        st.subheader("Pipeline Controls")
        run_enhancement = st.checkbox(
            "Enable U-Net Speech Enhancement",
            value=checkpoint_exists,
            disabled=not checkpoint_exists,
            help="Processes audio through the 2D U-Net spectral magnitude enhancer.",
        )
        run_correction = st.checkbox(
            "Enable Context-Aware Correction",
            value=True,
            help="Applies phonetic candidate generation and GPT-2 context scoring to the ASR transcript.",
        )

    with col_input:
        input_mode = st.radio(
            "Select Audio Source",
            ["Sample Benchmark Audio", "Upload Audio File"],
            horizontal=True,
        )

        audio_source = None
        reference_text: str | None = None

        if input_mode == "Upload Audio File":
            uploaded_file = st.file_uploader(
                "Upload Speech Audio (WAV, MP3, FLAC, OGG)",
                type=["wav", "mp3", "flac", "ogg"],
            )
            if uploaded_file is not None:
                audio_source = uploaded_file.getvalue()
        else:
            sample_dir = Path("data/processed/voicebank_demand_16k/noisy_testset_wav")
            transcripts_dir = Path("data/processed/voicebank_demand_16k/testset_txt")

            available_samples = []
            if sample_dir.is_dir():
                available_samples = sorted([p.name for p in sample_dir.glob("*.wav")])

            if available_samples:
                # Curated selection of diverse samples
                curated_samples = [
                    s for s in ["p232_001.wav", "p232_015.wav", "p232_020.wav", "p257_001.wav", "p257_010.wav", "p257_024.wav"]
                    if s in available_samples
                ] or available_samples[:10]

                selected_sample_name = st.selectbox(
                    "Choose a test utterance from VoiceBank+DEMAND:",
                    options=curated_samples,
                )
                sample_path = sample_dir / selected_sample_name
                audio_source = sample_path

                # Load reference text if available
                ref_path = transcripts_dir / f"{Path(selected_sample_name).stem}.txt"
                if ref_path.is_file():
                    with open(ref_path, "r", encoding="utf-8") as f:
                        reference_text = f.read().strip()
            else:
                st.warning("No sample files found in data directory. Please upload an audio file.")

    if audio_source is not None:
        try:
            # Preview Input Audio
            audio_tensor_preview, preview_sr = load_audio_from_source(audio_source)
            wav_bytes_preview = tensor_to_wav_bytes(audio_tensor_preview, preview_sr)

            st.markdown("### 🔊 Input Audio Preview")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.audio(wav_bytes_preview, format="audio/wav")
                st.caption(f"Duration: {len(audio_tensor_preview)/preview_sr:.2f}s | Sample Rate: {preview_sr} Hz")
            with col_b:
                if reference_text:
                    st.info(f"**Ground-Truth Reference Transcript:**\n`{reference_text}`")

            # Execution Button
            if st.button("🚀 Run End-to-End Pipeline", type="primary"):
                runner = get_cached_pipeline_runner(
                    checkpoint_path=default_ckpt,
                    whisper_model_name="base.en",
                    device=selected_device,
                )

                with st.spinner("Processing speech through all pipeline stages..."):
                    result: PipelineResult = runner.process(
                        audio_source=audio_source,
                        run_enhancement=run_enhancement,
                        run_correction=run_correction,
                    )

                st.success(f"Pipeline executed successfully in {result.timing.total_s:.2f}s!")

                # Stage 1 & 2: Audio & Enhancement Visuals
                st.markdown("---")
                st.subheader("1. Audio Signals & Speech Enhancement")

                col_raw, col_enh = st.columns(2)
                with col_raw:
                    st.markdown("**Input (Noisy) Audio**")
                    st.audio(wav_bytes_preview, format="audio/wav")

                with col_enh:
                    if result.enhanced_audio is not None:
                        st.markdown("**U-Net Enhanced Audio**")
                        enh_wav_bytes = tensor_to_wav_bytes(result.enhanced_audio, result.sample_rate)
                        st.audio(enh_wav_bytes, format="audio/wav")
                        st.caption(f"Enhancement latency: {result.timing.enhancement_s:.3f}s")
                    else:
                        st.info("Enhancement stage was skipped or unavailable.")

                fig = plot_audio_comparison(result.input_audio, result.enhanced_audio, result.sample_rate)
                st.pyplot(fig)
                plt.close(fig)

                # Stage 3 & 4: Transcriptions & Corrections
                st.markdown("---")
                st.subheader("2. ASR Transcriptions & Post-ASR Corrections")

                col_trans_raw, col_trans_enh = st.columns(2)

                with col_trans_raw:
                    st.markdown("#### Input Audio Pipeline")
                    st.markdown(f"**Whisper Raw ASR** ({result.timing.asr_raw_s:.2f}s):")
                    st.code(result.raw_transcript or "(empty)", language="text")

                    if run_correction and result.corrected_raw_transcript:
                        st.markdown(f"**Context-Corrected Transcript** ({result.timing.correction_raw_s:.2f}s):")
                        st.code(result.corrected_raw_transcript, language="text")

                        if result.raw_corrections:
                            st.markdown("**Corrected Words:**")
                            for c in result.raw_corrections:
                                dec = c.get("decision", {})
                                st.write(
                                    f"- Position {c['position']}: `'{c['original_token']}'` $\\to$ `'{c['corrected_token']}'` "
                                    f"(confidence: {dec.get('confidence', 0):.2f}, reason: `{dec.get('reason')}`)"
                                )
                        else:
                            st.caption("No words modified (retention criteria met for all tokens).")

                with col_trans_enh:
                    if result.enhanced_transcript is not None:
                        st.markdown("#### U-Net Enhanced Pipeline")
                        st.markdown(f"**Whisper Enhanced ASR** ({result.timing.asr_enhanced_s:.2f}s):")
                        st.code(result.enhanced_transcript or "(empty)", language="text")

                        if run_correction and result.corrected_enhanced_transcript:
                            st.markdown(f"**Context-Corrected Transcript** ({result.timing.correction_enhanced_s:.2f}s):")
                            st.code(result.corrected_enhanced_transcript, language="text")

                            if result.enhanced_corrections:
                                st.markdown("**Corrected Words:**")
                                for c in result.enhanced_corrections:
                                    dec = c.get("decision", {})
                                    st.write(
                                        f"- Position {c['position']}: `'{c['original_token']}'` $\\to$ `'{c['corrected_token']}'` "
                                        f"(confidence: {dec.get('confidence', 0):.2f}, reason: `{dec.get('reason')}`)"
                                    )
                            else:
                                st.caption("No words modified (retention criteria met for all tokens).")

                # Summary Comparison & Accuracy Metrics
                st.markdown("---")
                st.subheader("3. Pipeline Summary & Latency Breakdown")

                summary_rows = [
                    {
                        "Pipeline Path": "Raw Input Audio",
                        "ASR Transcript": result.raw_transcript,
                        "Corrected Transcript": result.corrected_raw_transcript or "N/A",
                        "ASR Time": f"{result.timing.asr_raw_s:.2f}s",
                        "Correction Time": f"{result.timing.correction_raw_s:.2f}s",
                    }
                ]
                if result.enhanced_transcript is not None:
                    summary_rows.append(
                        {
                            "Pipeline Path": "U-Net Enhanced Audio",
                            "ASR Transcript": result.enhanced_transcript,
                            "Corrected Transcript": result.corrected_enhanced_transcript or "N/A",
                            "ASR Time": f"{result.timing.asr_enhanced_s:.2f}s",
                            "Correction Time": f"{result.timing.correction_enhanced_s:.2f}s",
                        }
                    )
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                if reference_text:
                    st.markdown("#### Utterance Word Error Rate (WER)")
                    wer_rows = []
                    wer_raw = compute_wer(reference_text, result.raw_transcript)
                    wer_rows.append({"Condition": "Raw Input Speech", "WER": f"{wer_raw['wer']*100:.1f}%", "Errors (S/D/I)": f"{wer_raw['substitutions']}/{wer_raw['deletions']}/{wer_raw['insertions']}"})

                    if result.corrected_raw_transcript:
                        wer_raw_corr = compute_wer(reference_text, result.corrected_raw_transcript)
                        wer_rows.append({"Condition": "Raw Input + Correction", "WER": f"{wer_raw_corr['wer']*100:.1f}%", "Errors (S/D/I)": f"{wer_raw_corr['substitutions']}/{wer_raw_corr['deletions']}/{wer_raw_corr['insertions']}"})

                    if result.enhanced_transcript:
                        wer_enh = compute_wer(reference_text, result.enhanced_transcript)
                        wer_rows.append({"Condition": "U-Net Enhanced Speech", "WER": f"{wer_enh['wer']*100:.1f}%", "Errors (S/D/I)": f"{wer_enh['substitutions']}/{wer_enh['deletions']}/{wer_enh['insertions']}"})

                    if result.corrected_enhanced_transcript:
                        wer_enh_corr = compute_wer(reference_text, result.corrected_enhanced_transcript)
                        wer_rows.append({"Condition": "Enhanced + Correction", "WER": f"{wer_enh_corr['wer']*100:.1f}%", "Errors (S/D/I)": f"{wer_enh_corr['substitutions']}/{wer_enh_corr['deletions']}/{wer_enh_corr['insertions']}"})

                    st.dataframe(pd.DataFrame(wer_rows), use_container_width=True)

        except Exception as e:
            st.error(f"Error processing audio: {e}")
            logger.exception("Live pipeline processing error")


# ===========================================================================
# PAGE 2: BENCHMARK & EXPERIMENTAL RESULTS
# ===========================================================================
elif page == "📊 Benchmark & Research Results":
    st.title("📊 Verified Experimental Benchmark & Statistical Findings")

    st.warning(
        "⚠️ **Offline Experimental Results Notice**:  \n"
        "The metrics below were measured offline on the complete official **824-utterance VoiceBank+DEMAND test set** "
        "($6,211$ words; test speakers `p232` and `p257`). These results represent controlled scientific experiments "
        "and do not imply that an individual user's uploaded audio will reproduce these exact values."
    )

    # Consolidated Benchmark Table
    st.subheader("1. Consolidated Benchmark Results (824 Test Utterances, 6,211 Words)")
    benchmark_data = [
        {
            "Condition": "Clean Speech",
            "Pipeline": "Clean Audio → Whisper base.en",
            "Corpus WER": "4.65%",
            "Mean Utt WER": "5.05%",
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
            "Condition": "Noisy Speech",
            "Pipeline": "Noisy Audio → Whisper base.en",
            "Corpus WER": "7.23%",
            "Mean Utt WER": "7.59%",
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
            "Mean Utt WER": "8.78%",
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
            "Condition": "Noisy + Improved Corr",
            "Pipeline": "Noisy → Whisper → Improved Corrector",
            "Corpus WER": "7.12%",
            "Mean Utt WER": "7.51%",
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
            "Condition": "Enhanced + Improved Corr",
            "Pipeline": "Noisy → U-Net → Whisper → Improved Corrector",
            "Corpus WER": "8.11%",
            "Mean Utt WER": "8.66%",
            "Sub (S)": 362,
            "Del (D)": 80,
            "Ins (I)": 62,
            "Hits (C)": 5769,
            "Total Errors": 504,
            "STOI": 0.9338,
            "SI-SDR": "17.96 dB",
            "SNR": "18.05 dB",
        },
    ]
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)

    # Key Scientific Findings Callouts
    st.markdown("---")
    st.subheader("2. Key Scientific Findings & The Enhancement-ASR Disconnect")

    col_box1, col_box2 = st.columns(2)

    with col_box1:
        st.error(
            "#### 📉 The Enhancement-ASR Paradox\n"
            "While the U-Net substantially improved signal fidelity:\n"
            "- **SNR**: $+9.60\\text{ dB}$ ($8.45 \\to 18.05\\text{ dB}$)\n"
            "- **SI-SDR**: $+9.51\\text{ dB}$ ($8.45 \\to 17.96\\text{ dB}$)\n"
            "- **STOI**: $+0.0127$ ($0.9211 \\to 0.9338$)\n\n"
            "It **degraded downstream Whisper ASR** by $+1.00$ percentage points in WER ($7.23\\% \\to 8.23\\%$, $p = 0.0022$). "
            "Reusing corrupted phase during iSTFT reconstruction and non-linear spectral masking distort learned acoustic features."
        )

    with col_box2:
        st.success(
            "#### 🎯 Targeted Post-ASR Correction\n"
            "Applying Bayesian-inspired safeguards eliminated $97\\%$ of overcorrections:\n"
            "- **Score Margin Floor** ($3.5$)\n"
            "- **Phonetic Similarity Floor** ($0.80$)\n"
            "- **Retention Bias** ($1.0$)\n\n"
            "Yielded an observed **$-0.11$ percentage point WER reduction** on both noisy ($7.23\\% \\to 7.12\\%$) "
            "and enhanced ($8.23\\% \\to 8.11\\%$) speech, repairing $9$ utterances while worsening only $2$."
        )

    # Paired Statistical Analysis
    st.markdown("---")
    st.subheader("3. Paired Statistical Significance Testing (10,000 Bootstrap Replications)")

    stats_data = [
        {
            "Comparison": "Noisy Raw vs Noisy Improved-Corr",
            "Sample ΔWER": "-0.11 pp",
            "95% Bootstrap CI": "[-0.22%, -0.02%]",
            "Bootstrap p-val": "0.0356",
            "Bootstrap Sig (α=0.05)": "Yes",
            "Wilcoxon p-val": "0.1542",
            "Wilcoxon Sig (α=0.05)": "No",
            "Tied Utterances": "813 / 824 (98.7%)",
        },
        {
            "Comparison": "Enhanced Raw vs Enhanced Improved-Corr",
            "Sample ΔWER": "-0.11 pp",
            "95% Bootstrap CI": "[-0.22%, -0.02%]",
            "Bootstrap p-val": "0.0382",
            "Bootstrap Sig (α=0.05)": "Yes",
            "Wilcoxon p-val": "0.0754",
            "Wilcoxon Sig (α=0.05)": "No",
            "Tied Utterances": "813 / 824 (98.7%)",
        },
        {
            "Comparison": "Noisy Raw vs Enhanced Raw",
            "Sample ΔWER": "+1.00 pp",
            "95% Bootstrap CI": "[+0.35%, +1.69%]",
            "Bootstrap p-val": "0.0022",
            "Bootstrap Sig (α=0.05)": "Yes",
            "Wilcoxon p-val": "0.0066",
            "Wilcoxon Sig (α=0.05)": "Yes",
            "Degraded Utterances": "70 degraded vs 37 improved",
        },
        {
            "Comparison": "Noisy Raw vs Enhanced Improved-Corr",
            "Sample ΔWER": "+0.89 pp",
            "95% Bootstrap CI": "[+0.22%, +1.58%]",
            "Bootstrap p-val": "0.0094",
            "Bootstrap Sig (α=0.05)": "Yes",
            "Wilcoxon p-val": "0.0213",
            "Wilcoxon Sig (α=0.05)": "Yes",
            "Degraded Utterances": "70 degraded vs 45 improved",
        },
    ]
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

    st.info(
        "**Statistical Interpretation**: For the correction stage, bootstrap resampling across corpus word counts "
        "confirms an aggregate error reduction ($p < 0.05$). However, because $98.67\\%$ of utterances are left unchanged "
        "by design, the utterance-level Wilcoxon signed-rank test has small non-zero sample size ($N=11$) and does not "
        "reject the null hypothesis of median zero shift."
    )


# ===========================================================================
# PAGE 3: ARCHITECTURE & METHODOLOGY
# ===========================================================================
elif page == "🔬 Architecture & Methodology":
    st.title("🔬 Architecture, Methodology & Documented Limitations")

    st.subheader("1. End-to-End Pipeline Architecture")
    st.code(
        """
[ Noisy Waveform y(t) ]
         |
    STFT (n_fft=512, hop=256)
         |
    Magnitude Spectrogram |Y(f, t)|  +  Noisy Phase phi_Y(f, t)
         |                                     |
    2D Convolutional U-Net                     |
         |                                     |
    Enhanced Magnitude |S_hat(f, t)|           |
         |                                     |
         +-------------> iSTFT <---------------+
                           |
               [ Enhanced Waveform s_hat(t) ]
                           |
                   OpenAI Whisper base.en
                           |
                 [ Raw ASR Transcript ]
                           |
             Phonetic Candidate Generation (CMUdict)
                           |
             Context Scoring (GPT-2 + Bayesian Priors)
                           |
             [ Final Corrected ASR Transcript ]
        """,
        language="text",
    )

    st.markdown("---")
    st.subheader("2. Detailed Component Specifications")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(
            "#### Speech Enhancement: 2D U-Net\n"
            "- **Input Shape**: Magnitude spectrogram `(B, 1, 257, T)`\n"
            "- **Encoder**: 4 downsampling blocks (Conv2d + BatchNorm + LeakyReLU)\n"
            "- **Channels**: 32 → 64 → 128 → 256\n"
            "- **Decoder**: 4 upsampling blocks (ConvTranspose2d + Skip Connections)\n"
            "- **Loss Function**: Magnitude L1 distance\n"
            "- **Synthesis**: iSTFT using original noisy phase $\\phi_Y$\n"
        )

    with col_c2:
        st.markdown(
            "#### Post-ASR Context-Aware Correction\n"
            "- **Phonetic Shortlist**: CMU Pronouncing Dictionary with phonetic Levenshtein edit distance $\\le 2$\n"
            "- **Context Scorer**: Pre-trained causal LM (`gpt2`)\n"
            "- **Score Margin Floor**: $\\ge 3.5$ log-probability improvement required\n"
            "- **Phonetic Similarity Floor**: $\\ge 0.80$ similarity required\n"
            "- **Retention Bias**: $\\beta = 1.0$ penalty on non-original tokens\n"
        )

    st.markdown("---")
    st.subheader("3. Documented Research Limitations")
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
