"""Tuning and development-set evaluation for context-aware ASR correction.

Evaluates candidate safeguard configurations on the controlled development set
(data/dev_correction_set.json) without touching the 824-utterance held-out test set.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.correction.context_scorer import ContextAwareScorer, ContextScorerConfig
from src.correction.lm_scorer import CausalLMScorer
from src.correction.pipeline import CorrectionPipeline
from src.evaluation.correction_eval import evaluate_correction
from src.evaluation.wer import compute_wer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DevEvalResult:
    config: dict[str, Any]
    total_words: int
    raw_wer: float
    corr_wer: float
    delta_wer: float
    successful: int
    overcorrections: int
    failed: int
    total_changes: int
    precision: float
    net_improvement: int  # successful - overcorrections - failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "total_words": self.total_words,
            "raw_wer": self.raw_wer,
            "corr_wer": self.corr_wer,
            "delta_wer": self.delta_wer,
            "successful": self.successful,
            "overcorrections": self.overcorrections,
            "failed": self.failed,
            "total_changes": self.total_changes,
            "precision": self.precision,
            "net_improvement": self.net_improvement,
        }


def evaluate_on_dev(
    pipeline: CorrectionPipeline,
    dev_data: list[dict[str, Any]],
) -> DevEvalResult:
    """Evaluate a pipeline configuration on the development dataset."""
    total_ref_words = 0
    raw_errors = 0
    corr_errors = 0

    successful = 0
    overcorrections = 0
    failed = 0
    unchanged_correct = 0
    unchanged_errors = 0

    for item in dev_data:
        ref = item["reference"]
        hyp = item["hypothesis"]

        corr_res = pipeline.correct(hyp)
        corr_hyp = corr_res.corrected

        raw_wer_res = compute_wer(ref, hyp)
        corr_wer_res = compute_wer(ref, corr_hyp)

        total_ref_words += raw_wer_res.reference_words
        raw_errors += (raw_wer_res.substitutions + raw_wer_res.deletions + raw_wer_res.insertions)
        corr_errors += (corr_wer_res.substitutions + corr_wer_res.deletions + corr_wer_res.insertions)

        eval_res = evaluate_correction(ref, hyp, corr_res)
        successful += eval_res.successful_corrections
        overcorrections += eval_res.overcorrections
        failed += eval_res.failed_corrections
        unchanged_correct += eval_res.unchanged_correct
        unchanged_errors += eval_res.unchanged_errors

    raw_wer = raw_errors / total_ref_words if total_ref_words > 0 else 0.0
    corr_wer = corr_errors / total_ref_words if total_ref_words > 0 else 0.0
    total_changes = successful + overcorrections + failed
    precision = (successful / total_changes) if total_changes > 0 else 0.0

    return DevEvalResult(
        config=pipeline._scorer.config.__dict__,
        total_words=total_ref_words,
        raw_wer=float(raw_wer),
        corr_wer=float(corr_wer),
        delta_wer=float(corr_wer - raw_wer),
        successful=successful,
        overcorrections=overcorrections,
        failed=failed,
        total_changes=total_changes,
        precision=float(precision),
        net_improvement=successful - overcorrections - failed,
    )


def run_grid_search(
    dev_path: Path,
    device: str = "cuda",
) -> list[DevEvalResult]:
    """Run grid search over candidate safeguard parameters on dev set."""
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    logger.info("Loaded %d dev sentences from %s", len(dev_data), dev_path)
    from src.correction.candidates import PhoneticCandidateGenerator
    generator = PhoneticCandidateGenerator(max_candidates=20)
    lm = CausalLMScorer(device=device, batch_size=32)

    # Grid of candidate configurations
    candidate_configs = [
        # Baseline (existing)
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=0.0, min_phonetic_similarity=0.0, retention_bias=0.0),
        # Pure threshold variations
        ContextScorerConfig(threshold=0.85, phonetic_weight=1.0, min_score_margin=0.0, min_phonetic_similarity=0.0, retention_bias=0.0),
        ContextScorerConfig(threshold=0.95, phonetic_weight=1.0, min_score_margin=0.0, min_phonetic_similarity=0.0, retention_bias=0.0),
        # Score margin safeguards
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=2.0, min_phonetic_similarity=0.0, retention_bias=0.0),
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=3.5, min_phonetic_similarity=0.0, retention_bias=0.0),
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=5.0, min_phonetic_similarity=0.0, retention_bias=0.0),
        # Phonetic similarity floor
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=0.0, min_phonetic_similarity=0.75, retention_bias=0.0),
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=0.0, min_phonetic_similarity=0.80, retention_bias=0.0),
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=0.0, min_phonetic_similarity=0.85, retention_bias=0.0),
        # Retention bias
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=0.0, min_phonetic_similarity=0.0, retention_bias=1.5),
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=0.0, min_phonetic_similarity=0.0, retention_bias=2.5),
        # Combined principled safeguards
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=2.5, min_phonetic_similarity=0.80, retention_bias=1.0),
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=3.5, min_phonetic_similarity=0.80, retention_bias=1.0),
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=3.5, min_phonetic_similarity=0.85, retention_bias=1.5),
        ContextScorerConfig(threshold=0.7, phonetic_weight=1.0, min_score_margin=4.0, min_phonetic_similarity=0.80, retention_bias=1.5),
    ]

    results: list[DevEvalResult] = []
    for i, cfg in enumerate(candidate_configs):
        logger.info(
            "Evaluating Config %d/%d: threshold=%.2f, margin=%.1f, min_sim=%.2f, bias=%.1f",
            i + 1,
            len(candidate_configs),
            cfg.threshold,
            cfg.min_score_margin,
            cfg.min_phonetic_similarity,
            cfg.retention_bias,
        )
        scorer = ContextAwareScorer(generator=generator, lm=lm, config=cfg)
        pipeline = CorrectionPipeline(scorer=scorer)
        res = evaluate_on_dev(pipeline, dev_data)
        results.append(res)
        logger.info(
            "  -> Dev WER: %.2f%% -> %.2f%% (delta=%+.2f%%) | Succ=%d, Over=%d, Fail=%d | Prec=%.1f%% | Net=%+d",
            res.raw_wer * 100,
            res.corr_wer * 100,
            res.delta_wer * 100,
            res.successful,
            res.overcorrections,
            res.failed,
            res.precision * 100,
            res.net_improvement,
        )

    # Sort by net improvement, then precision, then delta WER
    results.sort(key=lambda r: (r.net_improvement, r.precision, -r.delta_wer), reverse=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune ASR correction safeguards on dev set.")
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=Path("data/dev_correction_set.json"),
        help="Path to development dataset JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for LM inference (cuda or cpu)",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("results/tuning_dev_results.json"),
        help="Path to save tuning results JSON",
    )
    args = parser.parse_args()

    results = run_grid_search(args.dev_file, device=args.device)

    print("\n" + "=" * 80)
    print("TOP DEVELOPMENT SET CONFIGURATIONS")
    print("=" * 80)
    for idx, r in enumerate(results[:5], start=1):
        cfg = r.config
        print(
            f"Rank {idx}: margin={cfg['min_score_margin']}, min_sim={cfg['min_phonetic_similarity']}, "
            f"bias={cfg['retention_bias']}, thresh={cfg['threshold']} -> "
            f"WER {r.raw_wer*100:.2f}% -> {r.corr_wer*100:.2f}% (delta={r.delta_wer*100:+.2f}%) | "
            f"Succ={r.successful}, Over={r.overcorrections}, Fail={r.failed} | "
            f"Prec={r.precision*100:.1f}% | Net={r.net_improvement:+d}"
        )

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    logger.info("Saved tuning results to %s", args.out_file)


if __name__ == "__main__":
    main()
