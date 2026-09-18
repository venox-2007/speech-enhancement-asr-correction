"""Sentence-level ASR correction pipeline.

Integrates phonetic candidate generation and context-aware scoring to
correct an entire ASR transcript word by word.

Design decision: corrections are **independent** — each word is scored
against the *original* sentence context, not the partially-corrected
one.  This avoids order-dependent cascading effects and makes per-word
analysis straightforward.  A future version could explore sequential
correction if order effects prove beneficial.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from src.correction.context_scorer import (
    ContextAwareScorer,
    ContextScorerConfig,
    ScoringDecision,
    replace_token,
    split_tokens,
    _split_affixes,
    match_case,
)
from src.correction.pronunciation import is_plain_word, normalize_word

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WordCorrection:
    """Record of a correction decision for a single word position."""

    position: int
    original_token: str
    corrected_token: str
    was_changed: bool
    decision: ScoringDecision | None  # None when the word was skipped

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "original_token": self.original_token,
            "corrected_token": self.corrected_token,
            "was_changed": self.was_changed,
            "decision": self.decision.to_dict() if self.decision else None,
        }


@dataclass
class CorrectionResult:
    """Result of correcting a full transcript."""

    original: str
    corrected: str
    word_corrections: list[WordCorrection] = field(default_factory=list)

    @property
    def num_words(self) -> int:
        """Total number of whitespace tokens in the original."""
        return len(self.word_corrections)

    @property
    def num_attempted(self) -> int:
        """Words for which the context scorer was invoked."""
        return sum(1 for wc in self.word_corrections if wc.decision is not None)

    @property
    def num_changed(self) -> int:
        """Words that were actually replaced."""
        return sum(1 for wc in self.word_corrections if wc.was_changed)

    @property
    def num_skipped(self) -> int:
        """Words skipped (non-alphabetic, no pronunciation, etc.)."""
        return sum(1 for wc in self.word_corrections if wc.decision is None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "corrected": self.corrected,
            "num_words": self.num_words,
            "num_attempted": self.num_attempted,
            "num_changed": self.num_changed,
            "num_skipped": self.num_skipped,
            "corrections": [wc.to_dict() for wc in self.word_corrections],
        }


class CorrectionPipeline:
    """Word-by-word ASR transcript correction.

    Each word in the hypothesis is independently evaluated using the
    context-aware scorer.  Words that are not plain English words
    (numbers, punctuation-only tokens) are skipped.

    Parameters
    ----------
    scorer : ContextAwareScorer or None
        Pre-configured scorer.  If *None*, a default scorer is created.
    """

    def __init__(self, scorer: ContextAwareScorer | None = None) -> None:
        self._scorer = scorer or ContextAwareScorer()

    def correct(self, hypothesis: str) -> CorrectionResult:
        """Correct an ASR hypothesis transcript.

        Parameters
        ----------
        hypothesis : str
            Raw ASR output sentence.

        Returns
        -------
        CorrectionResult
            Corrected transcript with per-word decision records.
        """
        if not hypothesis or not hypothesis.strip():
            return CorrectionResult(original=hypothesis, corrected=hypothesis)

        tokens = split_tokens(hypothesis)
        corrections: list[WordCorrection] = []
        corrected_tokens = list(tokens)

        for i, token in enumerate(tokens):
            word = normalize_word(token)
            if not word or not is_plain_word(word):
                # Skip non-alphabetic tokens (numbers, punctuation-only, etc.)
                corrections.append(
                    WordCorrection(
                        position=i,
                        original_token=token,
                        corrected_token=token,
                        was_changed=False,
                        decision=None,
                    )
                )
                continue

            # Score against the original sentence context (independent mode)
            decision = self._scorer.score(hypothesis, i)

            if decision.apply_correction:
                # Build the replacement token preserving punctuation and case
                prefix, core, suffix = _split_affixes(token)
                new_token = prefix + match_case(decision.best_word, core) + suffix
                corrected_tokens[i] = new_token

                corrections.append(
                    WordCorrection(
                        position=i,
                        original_token=token,
                        corrected_token=new_token,
                        was_changed=True,
                        decision=decision,
                    )
                )
                logger.info(
                    "Corrected word %d: %r -> %r (confidence=%.3f, reason=%s)",
                    i,
                    token,
                    new_token,
                    decision.confidence,
                    decision.reason,
                )
            else:
                corrections.append(
                    WordCorrection(
                        position=i,
                        original_token=token,
                        corrected_token=token,
                        was_changed=False,
                        decision=decision,
                    )
                )

        corrected = " ".join(corrected_tokens)
        return CorrectionResult(
            original=hypothesis,
            corrected=corrected,
            word_corrections=corrections,
        )
