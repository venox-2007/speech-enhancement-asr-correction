"""Demonstration of the ASR correction pipeline on hand-crafted examples.

.. warning::

   The transcript pairs below are **manually defined** for demonstration
   and smoke-testing purposes.  They are **not** experimental results from
   a real dataset and must not be cited as such.  Proper evaluation
   requires running the pipeline on a benchmark corpus (e.g.
   VoiceBank+DEMAND) with ground-truth transcripts.

Usage::

    python -m src.correction.evaluate_corrections
    python -m src.correction.evaluate_corrections --threshold 0.5 --verbose
    python -m src.correction.evaluate_corrections --json-out results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Sequence

from src.correction.context_scorer import ContextAwareScorer, ContextScorerConfig
from src.correction.pipeline import CorrectionPipeline
from src.evaluation.correction_eval import evaluate_correction
from src.utils.device import VALID_DEVICES
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------- demo pairs
# These are hand-crafted, NOT experimental data.
DEMO_PAIRS: list[dict[str, str]] = [
    {
        "reference": "I will send you the mail tomorrow",
        "hypothesis": "I will send you the male tomorrow",
        "note": "homophone: male → mail",
    },
    {
        "reference": "I can see the ship from here",
        "hypothesis": "I can sea the ship from here",
        "note": "homophone: sea → see",
    },
    {
        "reference": "Please write your name here",
        "hypothesis": "Please right your name here",
        "note": "homophone: right → write",
    },
    {
        "reference": "Turn right at the corner",
        "hypothesis": "Turn write at the corner",
        "note": "homophone: write → right",
    },
    {
        "reference": "The ship sailed across the sea",
        "hypothesis": "The ship sailed across the see",
        "note": "homophone: see → sea",
    },
    {
        "reference": "He is a male nurse",
        "hypothesis": "He is a male nurse",
        "note": "already correct — should not change",
    },
    {
        "reference": "Please check your mail",
        "hypothesis": "Please check your mail",
        "note": "already correct — should not change",
    },
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Demonstrate the ASR correction pipeline on hand-crafted examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "WARNING: These are hand-crafted demonstration pairs, not experimental\n"
            "results from a real dataset.  Do not cite them as benchmarks."
        ),
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=VALID_DEVICES,
        help="Compute device (default: auto)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Correction confidence threshold (default: 0.7)",
    )
    p.add_argument(
        "--phonetic-weight",
        type=float,
        default=1.0,
        help="Weight for phonetic similarity in scoring (default: 1.0)",
    )
    p.add_argument(
        "--json-out",
        metavar="PATH",
        help="Write full results to a JSON file",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def run(args: argparse.Namespace) -> list[dict]:
    """Run the demo and return a list of per-pair result dicts."""
    config = ContextScorerConfig(
        threshold=args.threshold,
        phonetic_weight=args.phonetic_weight,
    )
    scorer = ContextAwareScorer(config=config)
    pipeline = CorrectionPipeline(scorer)

    print("=" * 72)
    print("ASR Correction Pipeline — Demonstration")
    print("=" * 72)
    print()
    print("WARNING: The examples below are hand-crafted for demonstration.")
    print("         They are NOT experimental results from a real dataset.")
    print()

    all_results = []
    total_sc = 0
    total_oc = 0
    total_fc = 0

    for i, pair in enumerate(DEMO_PAIRS, 1):
        ref = pair["reference"]
        hyp = pair["hypothesis"]
        note = pair["note"]

        print(f"--- Example {i}: {note} ---")
        print(f"  Reference:  {ref}")
        print(f"  Hypothesis: {hyp}")

        correction = pipeline.correct(hyp)
        result = evaluate_correction(ref, hyp, correction)

        print(f"  Corrected:  {correction.corrected}")
        print(
            f"  WER: {result.wer_before.wer:.1%} -> {result.wer_after.wer:.1%}"
            f"  (delta {result.wer_delta:+.1%})"
        )
        print(
            f"  Successful: {result.successful_corrections}"
            f"  Overcorrections: {result.overcorrections}"
            f"  Failed: {result.failed_corrections}"
        )
        print()

        total_sc += result.successful_corrections
        total_oc += result.overcorrections
        total_fc += result.failed_corrections
        all_results.append(
            {
                "note": note,
                "evaluation": result.to_dict(),
            }
        )

    # --- Aggregate ---
    print("=" * 72)
    print("Aggregate (hand-crafted examples only — not a benchmark)")
    print(f"  Total successful corrections: {total_sc}")
    print(f"  Total overcorrections:        {total_oc}")
    print(f"  Total failed corrections:     {total_fc}")
    total = total_sc + total_oc + total_fc
    if total:
        print(f"  Correction precision:         {total_sc / total:.1%}")
    print("=" * 72)

    return all_results


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    try:
        results = run(args)
    except Exception:
        logger.exception("Demo failed")
        return 1

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "disclaimer": (
                        "Hand-crafted demonstration pairs. "
                        "NOT experimental results from a real dataset."
                    ),
                    "config": {
                        "threshold": args.threshold,
                        "phonetic_weight": args.phonetic_weight,
                        "device": args.device,
                    },
                    "pairs": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nResults written to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
