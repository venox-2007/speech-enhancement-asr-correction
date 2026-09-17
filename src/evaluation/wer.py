"""Word Error Rate (WER) computation using JiWER.

WER = (S + D + I) / N, where S, D and I are the word substitutions,
deletions and insertions in the minimum-edit alignment, and N is the number
of words in the reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jiwer

# Applied identically to reference and hypothesis. Whisper output contains
# casing and punctuation that most references do not, so without this the WER
# would mostly measure formatting differences.
NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

# Only splits into words; used when normalization is disabled.
NO_NORMALIZE = jiwer.Compose(
    [jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()]
)


@dataclass(frozen=True)
class WERResult:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_words: int
    reference: str  # text after normalization
    hypothesis: str  # text after normalization
    normalized: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    """Return the normalized form used for WER scoring."""
    words = NORMALIZE(text)
    return " ".join(words[0]) if words else ""


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> WERResult:
    """Compute WER and edit counts for a single reference/hypothesis pair."""
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")

    transform = NORMALIZE if normalize else NO_NORMALIZE
    ref_words = transform(reference)[0] if reference.strip() else []
    if not ref_words:
        raise ValueError("Reference transcript is empty (after normalization); WER is undefined.")

    out = jiwer.process_words(
        reference,
        hypothesis,
        reference_transform=transform,
        hypothesis_transform=transform,
    )
    return WERResult(
        wer=float(out.wer),
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
        hits=out.hits,
        reference_words=len(ref_words),
        reference=" ".join(out.references[0]),
        hypothesis=" ".join(out.hypotheses[0]),
        normalized=normalize,
    )


def alignment_report(reference: str, hypothesis: str, normalize: bool = True) -> str:
    """Human-readable word alignment (useful for inspecting ASR errors)."""
    transform = NORMALIZE if normalize else NO_NORMALIZE
    out = jiwer.process_words(
        reference, hypothesis, reference_transform=transform, hypothesis_transform=transform
    )
    return jiwer.visualize_alignment(out, show_measures=False)
