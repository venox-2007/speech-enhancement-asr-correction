"""Evaluation of ASR correction quality against a reference transcript.

Compares WER before and after correction and classifies every word-level
change as a successful correction, overcorrection, or failed correction.
Words that were not changed are classified as unchanged-correct or
unchanged-error so that the full picture of pipeline behaviour is visible.

Alignment between the hypothesis and reference is computed with
``jiwer.process_words`` so that insertions, deletions, and substitutions
are handled correctly.
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass, field
from typing import Any

import jiwer

from src.correction.pipeline import CorrectionResult
from src.evaluation.wer import NORMALIZE, WERResult, compute_wer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ outcome categories
SUCCESSFUL_CORRECTION = "successful_correction"
OVERCORRECTION = "overcorrection"
FAILED_CORRECTION = "failed_correction"
UNCHANGED_CORRECT = "unchanged_correct"
UNCHANGED_ERROR = "unchanged_error"


@dataclass(frozen=True)
class CorrectionOutcome:
    """Classification of what happened to a single hypothesis word."""

    position: int  # index in the hypothesis token list
    hyp_word: str  # normalized hypothesis word at this position
    corrected_word: str  # normalized corrected word at this position
    ref_word: str | None  # aligned reference word (None for insertions)
    category: str  # one of the category constants above


@dataclass
class CorrectionEvalResult:
    """Full evaluation of a correction run against a reference."""

    reference: str
    hypothesis: str
    corrected: str
    wer_before: WERResult
    wer_after: WERResult
    outcomes: list[CorrectionOutcome] = field(default_factory=list)

    @property
    def wer_delta(self) -> float:
        """WER change: negative means improvement."""
        return self.wer_after.wer - self.wer_before.wer

    @property
    def wer_improved(self) -> bool:
        return self.wer_after.wer < self.wer_before.wer

    @property
    def successful_corrections(self) -> int:
        return sum(1 for o in self.outcomes if o.category == SUCCESSFUL_CORRECTION)

    @property
    def overcorrections(self) -> int:
        return sum(1 for o in self.outcomes if o.category == OVERCORRECTION)

    @property
    def failed_corrections(self) -> int:
        return sum(1 for o in self.outcomes if o.category == FAILED_CORRECTION)

    @property
    def unchanged_correct(self) -> int:
        return sum(1 for o in self.outcomes if o.category == UNCHANGED_CORRECT)

    @property
    def unchanged_errors(self) -> int:
        return sum(1 for o in self.outcomes if o.category == UNCHANGED_ERROR)

    @property
    def total_changes(self) -> int:
        return self.successful_corrections + self.overcorrections + self.failed_corrections

    @property
    def correction_precision(self) -> float | None:
        """Fraction of changes that were successful.  None if no changes."""
        total = self.total_changes
        return self.successful_corrections / total if total else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "corrected": self.corrected,
            "wer_before": self.wer_before.to_dict(),
            "wer_after": self.wer_after.to_dict(),
            "wer_delta": round(self.wer_delta, 6),
            "wer_improved": self.wer_improved,
            "successful_corrections": self.successful_corrections,
            "overcorrections": self.overcorrections,
            "failed_corrections": self.failed_corrections,
            "unchanged_correct": self.unchanged_correct,
            "unchanged_errors": self.unchanged_errors,
            "total_changes": self.total_changes,
            "correction_precision": self.correction_precision,
            "outcomes": [
                {
                    "position": o.position,
                    "hyp_word": o.hyp_word,
                    "corrected_word": o.corrected_word,
                    "ref_word": o.ref_word,
                    "category": o.category,
                }
                for o in self.outcomes
            ],
        }


# ------------------------------------------------- normalisation helpers

# JiWER's RemovePunctuation uses string.punctuation
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_token(token: str) -> str:
    """Lowercase and strip punctuation (same transform as WER scoring)."""
    return token.lower().translate(_PUNCT_TABLE).strip()


# ------------------------------------------------- alignment mapping

def _build_hyp_to_ref_map(
    reference: str, hypothesis: str,
) -> dict[int, tuple[str, str | None]]:
    """Map each hypothesis word to its alignment type and reference word.

    Returns a dict ``{hyp_word_index: (alignment_type, ref_word)}``.

    *alignment_type* is ``"equal"``, ``"substitute"``, or ``"insert"``.
    *ref_word* is *None* for insertions (hypothesis word has no reference
    counterpart).
    """
    out = jiwer.process_words(
        reference,
        hypothesis,
        reference_transform=NORMALIZE,
        hypothesis_transform=NORMALIZE,
    )

    ref_words = out.references[0]
    mapping: dict[int, tuple[str, str | None]] = {}

    for chunk in out.alignments[0]:
        if chunk.type in ("equal", "substitute"):
            for h, r in zip(
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
                range(chunk.ref_start_idx, chunk.ref_end_idx),
            ):
                mapping[h] = (chunk.type, ref_words[r])
        elif chunk.type == "insert":
            for h in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                mapping[h] = ("insert", None)
        # "delete" chunks have no hypothesis words

    return mapping


def _token_to_norm_index(tokens: list[str]) -> dict[int, int]:
    """Map original token positions to normalized word-list positions.

    Tokens that become empty after normalisation (pure punctuation) are
    excluded, matching what JiWER does internally.
    """
    mapping: dict[int, int] = {}
    norm_idx = 0
    for tok_idx, token in enumerate(tokens):
        norm = _normalize_token(token)
        if norm:
            mapping[tok_idx] = norm_idx
            norm_idx += 1
    return mapping


# ------------------------------------------------- main evaluation function

def evaluate_correction(
    reference: str,
    hypothesis: str,
    correction: CorrectionResult,
) -> CorrectionEvalResult:
    """Evaluate a correction run against a reference transcript.

    Parameters
    ----------
    reference : str
        Ground-truth transcript.
    hypothesis : str
        Raw ASR output (before correction).
    correction : CorrectionResult
        Output of ``CorrectionPipeline.correct(hypothesis)``.

    Returns
    -------
    CorrectionEvalResult
        WER before/after, per-word outcome classification, and summary
        statistics.
    """
    wer_before = compute_wer(reference, hypothesis)
    wer_after = compute_wer(reference, correction.corrected)

    # Build alignment: normalised-hyp-word-index -> (type, ref_word)
    hyp_ref_map = _build_hyp_to_ref_map(reference, hypothesis)

    # Build mapping from pipeline token index to normalised word index
    hyp_tokens = hypothesis.split()
    tok_to_norm = _token_to_norm_index(hyp_tokens)

    outcomes: list[CorrectionOutcome] = []
    for wc in correction.word_corrections:
        norm_hyp = _normalize_token(wc.original_token)
        norm_cor = _normalize_token(wc.corrected_token)

        # Find the aligned reference word for this token position
        norm_idx = tok_to_norm.get(wc.position)
        if norm_idx is not None and norm_idx in hyp_ref_map:
            align_type, ref_word = hyp_ref_map[norm_idx]
        else:
            # Token normalised to empty (pure punctuation) or unmapped
            align_type, ref_word = None, None

        if not norm_hyp:
            # Pure-punctuation token — not evaluable
            continue

        was_correct = align_type == "equal"
        was_changed = norm_hyp != norm_cor

        if was_changed:
            if was_correct:
                category = OVERCORRECTION
            elif ref_word is not None and norm_cor == ref_word:
                category = SUCCESSFUL_CORRECTION
            else:
                category = FAILED_CORRECTION
        else:
            if was_correct:
                category = UNCHANGED_CORRECT
            else:
                category = UNCHANGED_ERROR

        outcomes.append(
            CorrectionOutcome(
                position=wc.position,
                hyp_word=norm_hyp,
                corrected_word=norm_cor,
                ref_word=ref_word,
                category=category,
            )
        )

    result = CorrectionEvalResult(
        reference=reference,
        hypothesis=hypothesis,
        corrected=correction.corrected,
        wer_before=wer_before,
        wer_after=wer_after,
        outcomes=outcomes,
    )

    logger.info(
        "WER %.2f%% -> %.2f%% (delta %+.2f%%) | "
        "corrections: %d successful, %d overcorrections, %d failed | "
        "unchanged: %d correct, %d errors",
        wer_before.wer * 100,
        wer_after.wer * 100,
        result.wer_delta * 100,
        result.successful_corrections,
        result.overcorrections,
        result.failed_corrections,
        result.unchanged_correct,
        result.unchanged_errors,
    )

    return result
