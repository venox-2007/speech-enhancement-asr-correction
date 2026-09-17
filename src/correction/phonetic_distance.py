"""Weighted edit distance between ARPAbet phoneme sequences.

Substituting acoustically similar phonemes costs less than substituting
unrelated ones. The cost values are hand-set heuristics, not learned or tuned
values. Insertions and deletions always cost 1, and every substitution costs
at most 1, so the weighted distance never exceeds the plain edit distance.
"""

from __future__ import annotations

from typing import Sequence

VOWELS = frozenset(
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
)
PHONE_CLASS: dict[str, str] = {
    **{p: "vowel" for p in VOWELS},
    **{p: "stop" for p in "P B T D K G".split()},
    **{p: "affricate" for p in "CH JH".split()},
    **{p: "fricative" for p in "F V TH DH S Z SH ZH HH".split()},
    **{p: "nasal" for p in "M N NG".split()},
    **{p: "liquid" for p in "L R".split()},
    **{p: "glide" for p in "W Y".split()},
}

# Pairs differing mainly in voicing, or closely related vowels.
_CLOSE_PAIRS = [
    ("P", "B"), ("T", "D"), ("K", "G"), ("F", "V"), ("S", "Z"),
    ("TH", "DH"), ("SH", "ZH"), ("CH", "JH"), ("TH", "F"), ("DH", "D"),
    ("IH", "IY"), ("UH", "UW"), ("EH", "AE"), ("EH", "EY"), ("AA", "AO"),
    ("AH", "AA"), ("AH", "IH"), ("AH", "ER"), ("OW", "AO"), ("M", "N"), ("N", "NG"),
]
CLOSE_PAIRS = frozenset(frozenset(p) for p in _CLOSE_PAIRS)

INDEL_COST = 1.0
CLOSE_COST = 0.4
SAME_CLASS_COST = 0.7
DIFFERENT_CLASS_COST = 1.0


def substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if frozenset((a, b)) in CLOSE_PAIRS:
        return CLOSE_COST
    class_a, class_b = PHONE_CLASS.get(a), PHONE_CLASS.get(b)
    if class_a is not None and class_a == class_b:
        return SAME_CLASS_COST
    return DIFFERENT_CLASS_COST


def weighted_phoneme_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """Weighted Levenshtein distance between two phoneme sequences."""
    prev = [j * INDEL_COST for j in range(len(b) + 1)]
    for i, pa in enumerate(a, start=1):
        cur = [i * INDEL_COST]
        for j, pb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + INDEL_COST,
                cur[j - 1] + INDEL_COST,
                prev[j - 1] + substitution_cost(pa, pb),
            ))
        prev = cur
    return prev[-1]


def phonetic_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """Similarity in [0, 1]: 1 - weighted distance / length of the longer sequence."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return max(0.0, 1.0 - weighted_phoneme_distance(a, b) / longest)
