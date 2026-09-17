"""Phonetic candidate generation for post-ASR error correction.

Given one ASR output word, return vocabulary words that sound similar. This
module only *proposes* candidates. Deciding whether to replace the word
(context scoring, thresholds) belongs to a later stage.

Search runs in two stages:
1. Shortlist: fast plain phoneme edit distance (RapidFuzz) over the whole
   vocabulary, keeping pronunciations within ``allowed_edits`` edits.
2. Rank: weighted phoneme distance (see ``phonetic_distance``) on the shortlist.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from src.correction.phonetic_distance import PHONE_CLASS, phonetic_similarity, weighted_phoneme_distance
from src.correction.pronunciation import (
    Pronunciation,
    PronunciationDict,
    default_pronunciation_dict,
    is_plain_word,
    normalize_word,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    word: str
    pronunciation: Pronunciation
    distance: float  # weighted phoneme distance to the query (0 = homophone)
    similarity: float  # 1 - distance / longer length, in [0, 1]

    @property
    def is_homophone(self) -> bool:
        return self.distance == 0.0


@dataclass
class CandidateResult:
    token: str  # token as given
    word: str  # normalized token
    known: bool  # False if no pronunciation was found, so no candidates were searched
    pronunciations: list[Pronunciation] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def words(self) -> list[str]:
        return [c.word for c in self.candidates]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for c, cd in zip(self.candidates, d["candidates"]):
            cd["is_homophone"] = c.is_homophone
        return d


class PhoneticCandidateGenerator:
    """Find vocabulary words that sound like a given word.

    Args:
        pron_dict: pronunciation source (default: CMUdict).
        vocabulary: words allowed as candidates (e.g. a task/dataset vocabulary).
            Defaults to every plain alphabetic word in ``pron_dict``. CMUdict
            includes many rare words and names, so a restricted vocabulary
            gives more plausible candidates.
        max_edits: upper bound on plain phoneme edits in the shortlist stage.
            Short words get a tighter bound (see ``allowed_edits``).
        min_similarity: drop candidates below this phonetic similarity.
        max_candidates: maximum number of candidates returned per word.
    """

    def __init__(
        self,
        pron_dict: PronunciationDict | None = None,
        vocabulary: Iterable[str] | None = None,
        max_edits: int = 2,
        min_similarity: float = 0.5,
        max_candidates: int = 10,
    ) -> None:
        if max_edits < 0:
            raise ValueError("max_edits must be >= 0")
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [0, 1]")
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")

        self.pron_dict = pron_dict or default_pronunciation_dict()
        self.max_edits = max_edits
        self.min_similarity = min_similarity
        self.max_candidates = max_candidates

        self._symbols: dict[str, str] = {}
        self._phones: list[str] = []
        self._by_code: dict[str, set[str]] = defaultdict(set)
        self._build_index(vocabulary)
        self._codes = list(self._by_code)

    # ------------------------------------------------------------------ index
    def _encode(self, pron: Sequence[str]) -> str:
        """Map each phoneme to one character so RapidFuzz can compare strings quickly."""
        out = []
        for p in pron:
            if p not in self._symbols:
                self._symbols[p] = chr(0x4E00 + len(self._phones))
                self._phones.append(p)
            out.append(self._symbols[p])
        return "".join(out)

    def _decode(self, code: str) -> Pronunciation:
        return tuple(self._phones[ord(c) - 0x4E00] for c in code)

    def _build_index(self, vocabulary: Iterable[str] | None) -> None:
        if vocabulary is None:
            entries = ((w, p) for w, p in self.pron_dict.items() if is_plain_word(w))
        else:
            vocab = {normalize_word(w) for w in vocabulary} - {""}
            missing = sorted(w for w in vocab if w not in self.pron_dict)
            if missing:
                logger.warning(
                    "%d vocabulary word(s) have no pronunciation and were skipped (e.g. %s)",
                    len(missing), ", ".join(missing[:5]),
                )
            entries = ((w, self.pron_dict.lookup(w)) for w in sorted(vocab) if w in self.pron_dict)

        for word, prons in entries:
            for pron in prons:
                self._by_code[self._encode(pron)].add(word)
        logger.info("Indexed %d pronunciations", len(self._by_code))

    # ----------------------------------------------------------------- search
    def allowed_edits(self, length: int) -> int:
        """Edit budget for a pronunciation of ``length`` phonemes (1 for short words)."""
        return max(1, min(self.max_edits, length // 2)) if self.max_edits else 0

    def generate(self, token: str) -> CandidateResult:
        """Candidates for an ASR token. Unknown or non-word tokens return no candidates."""
        word = normalize_word(token)
        prons = self.pron_dict.lookup(word) if word else []
        if not prons:
            logger.debug("No pronunciation for %r; no candidates generated", token)
            return CandidateResult(token=token, word=word, known=False)

        candidates = self.candidates_for_pronunciations(prons, exclude={word})
        return CandidateResult(token=token, word=word, known=True,
                               pronunciations=prons, candidates=candidates)

    def generate_many(self, tokens: Iterable[str]) -> list[CandidateResult]:
        return [self.generate(t) for t in tokens]

    def candidates_for_pronunciations(
        self, prons: Sequence[Sequence[str]], exclude: set[str] | None = None
    ) -> list[Candidate]:
        """Candidates for explicit phoneme sequences (e.g. from a G2P model for unknown words)."""
        exclude = exclude or set()
        best: dict[str, Candidate] = {}
        for pron in prons:
            pron = tuple(pron)
            unknown = [p for p in pron if p not in PHONE_CLASS]
            if unknown:
                raise ValueError(f"Unknown ARPAbet phoneme(s): {unknown}")
            if not pron:
                continue
            for code in self._shortlist(pron):
                cand_pron = self._decode(code)
                dist = round(weighted_phoneme_distance(pron, cand_pron), 4)
                sim = round(phonetic_similarity(pron, cand_pron), 4)
                if sim < self.min_similarity:
                    continue
                for w in self._by_code[code]:
                    if w in exclude:
                        continue
                    if w not in best or dist < best[w].distance:
                        best[w] = Candidate(w, cand_pron, dist, sim)

        ranked = sorted(best.values(), key=lambda c: (c.distance, -c.similarity, c.word))
        return ranked[: self.max_candidates]

    def _shortlist(self, pron: Pronunciation) -> list[str]:
        budget = self.allowed_edits(len(pron))
        query = self._encode(pron)
        matches = process.extract(
            query, self._codes, scorer=Levenshtein.distance,
            score_cutoff=budget, limit=None,
        )
        return [code for code, _score, _idx in matches]
