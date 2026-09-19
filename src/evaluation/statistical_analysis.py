"""Statistical analysis and final evaluation benchmarking for speech enhancement and ASR correction.

Provides:
- Clean-test Whisper ASR baseline evaluation and caching
- Paired bootstrap confidence intervals (95%) for corpus WER differences
- Paired utterance-level outcome tracking (improved, worsened, unchanged)
- Wilcoxon signed-rank significance testing on utterance WER deltas
- Consolidated multi-condition benchmark table (Clean, Noisy, Enhanced, Noisy+Correction, Enhanced+Correction)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from src.asr.whisper_asr import ASRConfig, WhisperASR
from src.enhancement.dataset import _extract_speaker
from src.evaluation.wer import compute_wer
from src.preprocessing.transcripts import load_transcripts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical Analysis Functions
# ---------------------------------------------------------------------------

def paired_bootstrap_wer(
    errors_a: np.ndarray | list[int],
    errors_b: np.ndarray | list[int],
    ref_words: np.ndarray | list[int],
    num_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Utterance-level paired bootstrap confidence interval for corpus WER differences (B - A).

    Parameters
    ----------
    errors_a : list or np.ndarray
        Total word errors (S + D + I) per utterance for System A.
    errors_b : list or np.ndarray
        Total word errors (S + D + I) per utterance for System B.
    ref_words : list or np.ndarray
        Total reference words per utterance.
    num_bootstrap : int
        Number of bootstrap replications (default: 10,000).
    seed : int
        Random seed for reproducibility.
    alpha : float
        Significance level for confidence interval (default: 0.05 for 95% CI).

    Returns
    -------
    dict
        Sample delta, bootstrap mean, std err, 95% CI (lower, upper), p-value, and significance flag.
    """
    ea = np.asarray(errors_a, dtype=np.float64)
    eb = np.asarray(errors_b, dtype=np.float64)
    nw = np.asarray(ref_words, dtype=np.float64)

    n_samples = len(nw)
    if n_samples == 0 or np.sum(nw) == 0:
        raise ValueError("Cannot perform bootstrap on empty sample or zero reference words.")

    total_words = np.sum(nw)
    wer_a_sample = np.sum(ea) / total_words
    wer_b_sample = np.sum(eb) / total_words
    delta_sample = wer_b_sample - wer_a_sample

    rng = np.random.default_rng(seed)
    # Generate bootstrap index matrix of shape (num_bootstrap, n_samples)
    indices = rng.integers(0, n_samples, size=(num_bootstrap, n_samples))

    # Vectorized bootstrap evaluation
    # Sum across utterances for each bootstrap replication
    ea_boot = np.sum(ea[indices], axis=1)
    eb_boot = np.sum(eb[indices], axis=1)
    nw_boot = np.sum(nw[indices], axis=1)

    delta_boot = (eb_boot - ea_boot) / nw_boot

    ci_lower = float(np.percentile(delta_boot, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(delta_boot, 100 * (1 - alpha / 2)))

    # Empirical two-sided p-value: probability of delta crossing 0
    p_pos = np.mean(delta_boot >= 0)
    p_neg = np.mean(delta_boot <= 0)
    p_value = float(2 * min(p_pos, p_neg))
    p_value = min(1.0, max(0.0, p_value))

    # A difference is statistically significant at level alpha if 0 is not inside the CI
    is_significant = bool(ci_lower > 0 or ci_upper < 0)

    return {
        "wer_a": float(wer_a_sample),
        "wer_b": float(wer_b_sample),
        "delta_sample": float(delta_sample),
        "delta_boot_mean": float(np.mean(delta_boot)),
        "delta_boot_std": float(np.std(delta_boot)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": 1.0 - alpha,
        "p_value": p_value,
        "is_significant": is_significant,
        "num_bootstrap": num_bootstrap,
    }


def paired_utterance_outcomes(
    wers_a: list[float] | np.ndarray,
    wers_b: list[float] | np.ndarray,
    eps: float = 1e-7,
) -> dict[str, Any]:
    """Classify per-utterance outcomes into improved, worsened, and unchanged.

    Parameters
    ----------
    wers_a : list or np.ndarray
        Per-utterance WERs for System A.
    wers_b : list or np.ndarray
        Per-utterance WERs for System B.
    eps : float
        Tolerance for floating point equivalence.

    Returns
    -------
    dict
        Counts, percentages, and delta distribution summary statistics.
    """
    wa = np.asarray(wers_a, dtype=np.float64)
    wb = np.asarray(wers_b, dtype=np.float64)
    if len(wa) != len(wb):
        raise ValueError(f"Length mismatch: {len(wa)} vs {len(wb)}")

    deltas = wb - wa
    n = len(deltas)

    improved = int(np.sum(deltas < -eps))
    worsened = int(np.sum(deltas > eps))
    unchanged = int(np.sum(np.abs(deltas) <= eps))

    q25, q50, q75 = np.percentile(deltas, [25, 50, 75])

    return {
        "count": n,
        "improved": improved,
        "improved_pct": float(improved / n * 100) if n > 0 else 0.0,
        "worsened": worsened,
        "worsened_pct": float(worsened / n * 100) if n > 0 else 0.0,
        "unchanged": unchanged,
        "unchanged_pct": float(unchanged / n * 100) if n > 0 else 0.0,
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "delta_median": float(q50),
        "delta_min": float(np.min(deltas)),
        "delta_max": float(np.max(deltas)),
        "delta_q25": float(q25),
        "delta_q75": float(q75),
        "delta_iqr": float(q75 - q25),
    }


def paired_wilcoxon_test(
    wers_a: list[float] | np.ndarray,
    wers_b: list[float] | np.ndarray,
) -> dict[str, Any]:
    """Perform Wilcoxon signed-rank test on paired utterance-level WERs."""
    wa = np.asarray(wers_a, dtype=np.float64)
    wb = np.asarray(wers_b, dtype=np.float64)
    diff = wb - wa
    non_zero = diff[np.abs(diff) > 1e-7]

    if len(non_zero) == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_non_zero": 0,
            "is_significant": False,
        }

    try:
        res = stats.wilcoxon(wa, wb, zero_method="wilcox", alternative="two-sided")
        stat = float(res.statistic)
        pval = float(res.pvalue)
    except Exception as exc:
        logger.warning("Wilcoxon test encountered exception: %s. Returning fallback.", exc)
        stat = 0.0
        pval = 1.0

    return {
        "statistic": stat,
        "p_value": pval,
        "n_non_zero": len(non_zero),
        "is_significant": bool(pval < 0.05),
    }


# ---------------------------------------------------------------------------
# Clean Whisper Baseline
# ---------------------------------------------------------------------------

def run_clean_whisper_baseline(
    clean_dir: Path,
    transcripts: dict[str, str],
    cache_path: Path,
    device: str = "cuda",
    model_name: str = "base.en",
) -> dict[str, Any]:
    """Transcribe all clean testset audio files with Whisper and evaluate against references."""
    cache: dict[str, Any] = {}
    cache_path = Path(cache_path)
    if cache_path.is_file():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info("Loaded %d clean Whisper transcriptions from cache: %s", len(cache), cache_path)
        except Exception as exc:
            logger.warning("Could not read clean ASR cache (%s). Will rerun.", exc)
            cache = {}

    wav_files = sorted(clean_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {clean_dir}")

    needed = [f for f in wav_files if f.stem not in cache]
    if needed:
        logger.info("Transcribing %d clean test utterances with Whisper '%s' on %s...", len(needed), model_name, device)
        asr = WhisperASR(ASRConfig(model_name=model_name, device=device))
        t0 = time.time()
        for idx, wav_path in enumerate(needed, start=1):
            stem = wav_path.stem
            trans_res = asr.transcribe(wav_path)
            cache[stem] = {
                "clean_text": trans_res.text,
                "duration_s": trans_res.audio_duration_s,
            }
            if idx % 100 == 0 or idx == len(needed):
                logger.info("Clean ASR progress: %d/%d utterances (%.1fs)", idx, len(needed), time.time() - t0)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        logger.info("Saved clean ASR cache to %s", cache_path)

    # Compute WER and edit counts for each clean utterance
    records = []
    for wav_path in wav_files:
        stem = wav_path.stem
        ref = transcripts.get(stem, "")
        if not ref:
            raise ValueError(f"Missing reference transcript for '{stem}'")
        hyp = cache[stem]["clean_text"]
        wer_res = compute_wer(ref, hyp)
        records.append({
            "stem": stem,
            "reference": ref,
            "hypothesis": hyp,
            "ref_words": wer_res.reference_words,
            "wer": wer_res.wer,
            "substitutions": wer_res.substitutions,
            "deletions": wer_res.deletions,
            "insertions": wer_res.insertions,
            "hits": wer_res.hits,
        })

    total_words = sum(r["ref_words"] for r in records)
    tot_s = sum(r["substitutions"] for r in records)
    tot_d = sum(r["deletions"] for r in records)
    tot_i = sum(r["insertions"] for r in records)
    tot_c = sum(r["hits"] for r in records)
    corpus_wer = (tot_s + tot_d + tot_i) / total_words

    wers = [r["wer"] for r in records]

    return {
        "count": len(records),
        "total_words": total_words,
        "corpus_wer": float(corpus_wer),
        "substitutions": int(tot_s),
        "deletions": int(tot_d),
        "insertions": int(tot_i),
        "correct_words": int(tot_c),
        "utt_wer_mean": float(np.mean(wers)),
        "utt_wer_std": float(np.std(wers)),
        "utt_wer_median": float(np.median(wers)),
        "records": records,
    }


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run_final_analysis(
    clean_dir: Path,
    transcripts_dir: Path,
    improved_csv: Path,
    out_dir: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Execute complete statistical analysis and compile consolidated benchmark table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts = load_transcripts(transcripts_dir)
    logger.info("Loaded %d ground-truth reference transcripts.", len(transcripts))

    # 1. Clean test baseline
    clean_cache_path = out_dir / "clean_asr_cache.json"
    clean_results = run_clean_whisper_baseline(
        clean_dir=clean_dir,
        transcripts=transcripts,
        cache_path=clean_cache_path,
        device=device,
    )
    logger.info(
        "Clean Testset Baseline (824 utterances): Corpus WER = %.2f%% (S=%d, D=%d, I=%d, C=%d)",
        clean_results["corpus_wer"] * 100,
        clean_results["substitutions"],
        clean_results["deletions"],
        clean_results["insertions"],
        clean_results["correct_words"],
    )

    # 2. Load improved correction records
    import csv
    with open(improved_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    logger.info("Loaded %d per-utterance records from %s", len(reader), improved_csv)

    # Extract aligned arrays for all conditions across the exact same 824 utterances
    stems = [r["clean_file"].replace(".wav", "") for r in reader]
    clean_map = {r["stem"]: r for r in clean_results["records"]}

    ref_words = np.array([int(r["noisy_raw_ref_words"]) for r in reader], dtype=np.int64)

    # Condition arrays: errors = S + D + I
    clean_errors = np.array([clean_map[s]["substitutions"] + clean_map[s]["deletions"] + clean_map[s]["insertions"] for s in stems], dtype=np.int64)
    clean_wers = np.array([clean_map[s]["wer"] for s in stems], dtype=np.float64)

    noisy_raw_errors = np.array([int(r["noisy_raw_sub"]) + int(r["noisy_raw_del"]) + int(r["noisy_raw_ins"]) for r in reader], dtype=np.int64)
    noisy_raw_wers = np.array([float(r["noisy_raw_wer"]) for r in reader], dtype=np.float64)

    noisy_corr_errors = np.array([int(r["noisy_corr_sub"]) + int(r["noisy_corr_del"]) + int(r["noisy_corr_ins"]) for r in reader], dtype=np.int64)
    noisy_corr_wers = np.array([float(r["noisy_corr_wer"]) for r in reader], dtype=np.float64)

    enh_raw_errors = np.array([int(r["enhanced_raw_sub"]) + int(r["enhanced_raw_del"]) + int(r["enhanced_raw_ins"]) for r in reader], dtype=np.int64)
    enh_raw_wers = np.array([float(r["enhanced_raw_wer"]) for r in reader], dtype=np.float64)

    enh_corr_errors = np.array([int(r["enhanced_corr_sub"]) + int(r["enhanced_corr_del"]) + int(r["enhanced_corr_ins"]) for r in reader], dtype=np.int64)
    enh_corr_wers = np.array([float(r["enhanced_corr_wer"]) for r in reader], dtype=np.float64)

    # 3. Perform the 4 specified paired statistical comparisons
    comparisons = {
        "noisy_raw_vs_noisy_improved_corr": {
            "name": "Noisy Raw vs Noisy Improved-Correction",
            "system_a": "Noisy Raw",
            "system_b": "Noisy Improved-Correction",
            "errors_a": noisy_raw_errors,
            "errors_b": noisy_corr_errors,
            "wers_a": noisy_raw_wers,
            "wers_b": noisy_corr_wers,
        },
        "enhanced_raw_vs_enhanced_improved_corr": {
            "name": "Enhanced Raw vs Enhanced Improved-Correction",
            "system_a": "Enhanced Raw",
            "system_b": "Enhanced Improved-Correction",
            "errors_a": enh_raw_errors,
            "errors_b": enh_corr_errors,
            "wers_a": enh_raw_wers,
            "wers_b": enh_corr_wers,
        },
        "noisy_raw_vs_enhanced_raw": {
            "name": "Noisy Raw vs Enhanced Raw",
            "system_a": "Noisy Raw",
            "system_b": "Enhanced Raw",
            "errors_a": noisy_raw_errors,
            "errors_b": enh_raw_errors,
            "wers_a": noisy_raw_wers,
            "wers_b": enh_raw_wers,
        },
        "noisy_raw_vs_enhanced_improved_corr": {
            "name": "Noisy Raw vs Enhanced Improved-Correction",
            "system_a": "Noisy Raw",
            "system_b": "Enhanced Improved-Correction",
            "errors_a": noisy_raw_errors,
            "errors_b": enh_corr_errors,
            "wers_a": noisy_raw_wers,
            "wers_b": enh_corr_wers,
        },
    }

    statistical_results: dict[str, Any] = {}
    for comp_key, comp in comparisons.items():
        logger.info("Computing paired statistics for: %s", comp["name"])
        boot_res = paired_bootstrap_wer(
            comp["errors_a"],
            comp["errors_b"],
            ref_words,
            num_bootstrap=10000,
            seed=42,
            alpha=0.05,
        )
        outcomes_res = paired_utterance_outcomes(comp["wers_a"], comp["wers_b"])
        wilcox_res = paired_wilcoxon_test(comp["wers_a"], comp["wers_b"])

        statistical_results[comp_key] = {
            "description": comp["name"],
            "bootstrap": boot_res,
            "outcomes": outcomes_res,
            "wilcoxon": wilcox_res,
        }

    # 4. Consolidate Final Benchmark Table across all 5 conditions
    # Load acoustic metrics from enhancement_eval
    enh_eval_summary_path = Path("results/enhancement_eval/summary.json")
    acoustic_metrics = {}
    if enh_eval_summary_path.is_file():
        with open(enh_eval_summary_path, "r", encoding="utf-8") as f:
            enh_summary = json.load(f)
            ov = enh_summary.get("overall", {})
            acoustic_metrics = {
                "noisy_stoi": ov.get("noisy_stoi", {}).get("mean"),
                "enh_stoi": ov.get("enhanced_stoi", {}).get("mean"),
                "noisy_sisdr": ov.get("noisy_sisdr", {}).get("mean"),
                "enh_sisdr": ov.get("enhanced_sisdr", {}).get("mean"),
                "noisy_snr": ov.get("noisy_snr", {}).get("mean"),
                "enh_snr": ov.get("enhanced_snr", {}).get("mean"),
            }

    tot_words = int(np.sum(ref_words))
    final_table = [
        {
            "condition": "Clean Speech",
            "pipeline": "Clean Audio -> Whisper base.en",
            "corpus_wer": float(clean_results["corpus_wer"]),
            "mean_utt_wer": float(clean_results["utt_wer_mean"]),
            "substitutions": int(clean_results["substitutions"]),
            "deletions": int(clean_results["deletions"]),
            "insertions": int(clean_results["insertions"]),
            "total_errors": int(clean_results["substitutions"] + clean_results["deletions"] + clean_results["insertions"]),
            "correct_words": int(clean_results["correct_words"]),
            "total_words": tot_words,
            "stoi": 1.0,
            "sisdr_db": None,  # Reference clean audio
            "snr_db": None,
        },
        {
            "condition": "Noisy Speech",
            "pipeline": "Noisy Audio -> Whisper base.en",
            "corpus_wer": float(np.sum(noisy_raw_errors) / tot_words),
            "mean_utt_wer": float(np.mean(noisy_raw_wers)),
            "substitutions": int(sum(int(r["noisy_raw_sub"]) for r in reader)),
            "deletions": int(sum(int(r["noisy_raw_del"]) for r in reader)),
            "insertions": int(sum(int(r["noisy_raw_ins"]) for r in reader)),
            "total_errors": int(np.sum(noisy_raw_errors)),
            "correct_words": int(sum(int(r["noisy_raw_hits"]) for r in reader)),
            "total_words": tot_words,
            "stoi": acoustic_metrics.get("noisy_stoi"),
            "sisdr_db": acoustic_metrics.get("noisy_sisdr"),
            "snr_db": acoustic_metrics.get("noisy_snr"),
        },
        {
            "condition": "U-Net Enhanced",
            "pipeline": "Noisy -> U-Net -> Whisper base.en",
            "corpus_wer": float(np.sum(enh_raw_errors) / tot_words),
            "mean_utt_wer": float(np.mean(enh_raw_wers)),
            "substitutions": int(sum(int(r["enhanced_raw_sub"]) for r in reader)),
            "deletions": int(sum(int(r["enhanced_raw_del"]) for r in reader)),
            "insertions": int(sum(int(r["enhanced_raw_ins"]) for r in reader)),
            "total_errors": int(np.sum(enh_raw_errors)),
            "correct_words": int(sum(int(r["enhanced_raw_hits"]) for r in reader)),
            "total_words": tot_words,
            "stoi": acoustic_metrics.get("enh_stoi"),
            "sisdr_db": acoustic_metrics.get("enh_sisdr"),
            "snr_db": acoustic_metrics.get("enh_snr"),
        },
        {
            "condition": "Noisy + Improved Correction",
            "pipeline": "Noisy -> Whisper -> Improved Corrector",
            "corpus_wer": float(np.sum(noisy_corr_errors) / tot_words),
            "mean_utt_wer": float(np.mean(noisy_corr_wers)),
            "substitutions": int(sum(int(r["noisy_corr_sub"]) for r in reader)),
            "deletions": int(sum(int(r["noisy_corr_del"]) for r in reader)),
            "insertions": int(sum(int(r["noisy_corr_ins"]) for r in reader)),
            "total_errors": int(np.sum(noisy_corr_errors)),
            "correct_words": int(sum(int(r["noisy_corr_hits"]) for r in reader)),
            "total_words": tot_words,
            "stoi": acoustic_metrics.get("noisy_stoi"),
            "sisdr_db": acoustic_metrics.get("noisy_sisdr"),
            "snr_db": acoustic_metrics.get("noisy_snr"),
        },
        {
            "condition": "Enhanced + Improved Correction",
            "pipeline": "Noisy -> U-Net -> Whisper -> Improved Corrector",
            "corpus_wer": float(np.sum(enh_corr_errors) / tot_words),
            "mean_utt_wer": float(np.mean(enh_corr_wers)),
            "substitutions": int(sum(int(r["enhanced_corr_sub"]) for r in reader)),
            "deletions": int(sum(int(r["enhanced_corr_del"]) for r in reader)),
            "insertions": int(sum(int(r["enhanced_corr_ins"]) for r in reader)),
            "total_errors": int(np.sum(enh_corr_errors)),
            "correct_words": int(sum(int(r["enhanced_corr_hits"]) for r in reader)),
            "total_words": tot_words,
            "stoi": acoustic_metrics.get("enh_stoi"),
            "sisdr_db": acoustic_metrics.get("enh_sisdr"),
            "snr_db": acoustic_metrics.get("enh_snr"),
        },
    ]

    output_summary = {
        "utterance_count": len(reader),
        "total_words": tot_words,
        "clean_baseline": {k: v for k, v in clean_results.items() if k != "records"},
        "comparisons": statistical_results,
        "consolidated_table": final_table,
    }

    summary_file = out_dir / "final_evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)
    logger.info("Saved final evaluation summary to %s", summary_file)

    # Print summary tables to stdout
    print("\n" + "=" * 95)
    print("FINAL CONSOLIDATED BENCHMARK TABLE (824 Held-Out Test Utterances, 6,211 Words)")
    print("=" * 95)
    print(f"{'Condition':<30} | {'Corpus WER':<10} | {'Utt WER':<10} | {'S':<4} {'D':<4} {'I':<4} | {'Hits':<5} | {'STOI':<6} | {'SI-SDR':<7} | {'SNR':<7}")
    print("-" * 95)
    for row in final_table:
        stoi_str = f"{row['stoi']:.4f}" if row['stoi'] is not None else "N/A"
        sisdr_str = f"{row['sisdr_db']:.2f}dB" if row['sisdr_db'] is not None else "N/A"
        snr_str = f"{row['snr_db']:.2f}dB" if row['snr_db'] is not None else "N/A"
        print(f"{row['condition']:<30} | {row['corpus_wer']*100:>9.2f}% | {row['mean_utt_wer']*100:>9.2f}% | {row['substitutions']:<4} {row['deletions']:<4} {row['insertions']:<4} | {row['correct_words']:<5} | {stoi_str:<6} | {sisdr_str:<7} | {snr_str:<7}")

    print("\n" + "=" * 95)
    print("PAIRED STATISTICAL ANALYSIS RESULTS (10,000 Bootstrap Replications & Wilcoxon Signed-Rank)")
    print("=" * 95)
    for comp_key, res in statistical_results.items():
        b = res["bootstrap"]
        o = res["outcomes"]
        w = res["wilcoxon"]
        sig_str = "YES (p < 0.05)" if b["is_significant"] else "NO (p >= 0.05)"
        print(f"\n[{res['description']}]")
        print(f"  WER Delta:        {b['delta_sample']*100:+.2f}% ({b['wer_a']*100:.2f}% -> {b['wer_b']*100:.2f}%)")
        print(f"  95% Bootstrap CI: [{b['ci_lower']*100:+.2f}%, {b['ci_upper']*100:+.2f}%] | Bootstrap p-value = {b['p_value']:.4f}")
        print(f"  Stat. Significant: {sig_str}")
        print(f"  Per-Utt Outcomes: Improved: {o['improved']} ({o['improved_pct']:.1f}%), Worsened: {o['worsened']} ({o['worsened_pct']:.1f}%), Unchanged: {o['unchanged']} ({o['unchanged_pct']:.1f}%)")
        print(f"  Wilcoxon Test:    Statistic = {w['statistic']:.1f}, p-value = {w['p_value']:.4e} (N non-zero = {w['n_non_zero']})")

    return output_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Final statistical analysis and consolidated evaluation.")
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("data/processed/voicebank_demand_16k/clean_testset_wav"),
        help="Directory containing clean test WAV audio files",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("data/processed/voicebank_demand_16k/testset_txt"),
        help="Directory containing official test reference transcripts",
    )
    parser.add_argument(
        "--improved-csv",
        type=Path,
        default=Path("results/improved_correction_eval/results_per_utterance.csv"),
        help="CSV from improved correction evaluation milestone",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/final_analysis"),
        help="Directory to save final statistical outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Whisper inference if clean baseline not cached",
    )
    args = parser.parse_args()

    run_final_analysis(
        clean_dir=args.clean_dir,
        transcripts_dir=args.transcripts_dir,
        improved_csv=args.improved_csv,
        out_dir=args.out_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
