"""Context-aware scoring of phonetic candidates for one ASR word.

For a target word in a sentence, the original word and each phonetic candidate
are placed into the sentence and scored with a language model:

    score(w) = log P_LM(sentence with w) + phonetic_weight * log(similarity(w))

A softmax over the scores of {original} ∪ candidates gives each alternative a
probability. The top alternative is applied only if it differs from the
original word and its probability reaches ``threshold``; otherwise the word is kept.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from src.correction.candidates import PhoneticCandidateGenerator
from src.correction.lm_scorer import CausalLMScorer, SentenceScorer
from src.correction.pronunciation import normalize_word

logger = logging.getLogger(__name__)


# Decision reasons
NO_PRONUNCIATION = "no_pronunciation"
NO_CANDIDATES = "no_candidates"
ORIGINAL_BEST = "original_best"
BELOW_THRESHOLD = "below_threshold"
BELOW_SCORE_MARGIN = "below_score_margin"
CORRECTED = "corrected"


@dataclass(frozen=True)
class ContextScorerConfig:
    threshold: float = 0.7  # min probability of the best candidate before a correction is applied
    phonetic_weight: float = 1.0  # weight of log(phonetic similarity); 0 = LM only
    max_candidates: int = 20  # candidates requested from the phonetic generator
    min_score_margin: float = 0.0  # min score(best) - score(original) required before replacing
    min_phonetic_similarity: float = 0.0  # min phonetic similarity required to consider candidate
    retention_bias: float = 0.0  # prior penalty applied to non-original candidates

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.phonetic_weight < 0:
            raise ValueError("phonetic_weight must be >= 0")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.min_score_margin < 0:
            raise ValueError("min_score_margin must be >= 0")
        if not 0.0 <= self.min_phonetic_similarity <= 1.0:
            raise ValueError("min_phonetic_similarity must be in [0, 1]")
        if self.retention_bias < 0:
            raise ValueError("retention_bias must be >= 0")


@dataclass(frozen=True)
class ScoredAlternative:
    word: str
    is_original: bool
    lm_log_prob: float
    phonetic_similarity: float
    score: float
    probability: float


@dataclass
class ScoringDecision:
    sentence: str
    word_index: int
    original_token: str
    original_word: str
    best_word: str
    confidence: float  # probability of best_word among the alternatives
    original_confidence: float  # probability of the original word
    apply_correction: bool
    reason: str
    corrected_sentence: str
    alternatives: list[ScoredAlternative] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ helpers
def split_tokens(sentence: str) -> list[str]:
    return sentence.split()


def _split_affixes(token: str) -> tuple[str, str, str]:
    """``'"Mail,'`` -> ``('"', 'Mail', ',')``."""
    core = normalize_word(token)
    if not core:
        return token, "", ""
    start = token.lower().find(core)
    return token[:start], token[start:start + len(core)], token[start + len(core):]


def match_case(word: str, template: str) -> str:
    if template.isupper() and len(template) > 1:
        return word.upper()
    if template[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def replace_token(tokens: list[str], index: int, word: str) -> str:
    """Sentence with ``tokens[index]`` replaced by ``word`` (keeps punctuation and casing)."""
    prefix, core, suffix = _split_affixes(tokens[index])
    new_tokens = list(tokens)
    new_tokens[index] = prefix + match_case(word, core) + suffix
    return " ".join(new_tokens)


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


# ------------------------------------------------------------------- scorer
class ContextAwareScorer:
    """Decide whether an ASR word should be replaced by one of its phonetic candidates."""

    def __init__(
        self,
        generator: PhoneticCandidateGenerator | None = None,
        lm: SentenceScorer | None = None,
        config: ContextScorerConfig | None = None,
    ) -> None:
        self.config = config or ContextScorerConfig()
        self.generator = generator or PhoneticCandidateGenerator(max_candidates=self.config.max_candidates)
        self.lm = lm or CausalLMScorer()

    def score_word(self, sentence: str, word: str, occurrence: int = 0) -> ScoringDecision:
        """Score the ``occurrence``-th (0-based) token in ``sentence`` matching ``word``."""
        target = normalize_word(word)
        matches = [i for i, t in enumerate(split_tokens(sentence)) if normalize_word(t) == target]
        if len(matches) <= occurrence:
            raise ValueError(f"Word {word!r} (occurrence {occurrence}) not found in sentence: {sentence!r}")
        return self.score(sentence, matches[occurrence])

    def score(self, sentence: str, word_index: int) -> ScoringDecision:
        """Score the whitespace token at ``word_index`` in ``sentence``."""
        tokens = split_tokens(sentence)
        if not tokens:
            raise ValueError("Sentence is empty")
        if not 0 <= word_index < len(tokens):
            raise IndexError(f"word_index {word_index} out of range for {len(tokens)} tokens")

        token = tokens[word_index]
        original = normalize_word(token)
        clean_sentence = " ".join(tokens)
        base = dict(sentence=clean_sentence, word_index=word_index, original_token=token,
                    original_word=original)
        keep = dict(base, best_word=original, corrected_sentence=clean_sentence,
                    confidence=1.0, original_confidence=1.0, apply_correction=False)

        result = self.generator.generate(token)
        if not result.known:
            return ScoringDecision(**keep, reason=NO_PRONUNCIATION)
        candidates = result.candidates[: self.config.max_candidates]
        if self.config.min_phonetic_similarity > 0.0:
            candidates = [c for c in candidates if c.similarity >= self.config.min_phonetic_similarity]
        if not candidates:
            return ScoringDecision(**keep, reason=NO_CANDIDATES)

        words = [original] + [c.word for c in candidates]
        similarities = [1.0] + [c.similarity for c in candidates]
        variants = [replace_token(tokens, word_index, w) for w in words]
        lm_scores = self.lm.score_sentences(variants)

        w = self.config.phonetic_weight
        bias = self.config.retention_bias
        scores = [
            lp + w * math.log(max(sim, 1e-9)) - (bias if i > 0 else 0.0)
            for i, (lp, sim) in enumerate(zip(lm_scores, similarities))
        ]
        probs = softmax(scores)
        orig_score = scores[0]
        alternatives = sorted(
            (ScoredAlternative(wd, i == 0, round(lp, 4), sim, round(sc, 4), prob)
             for i, (wd, lp, sim, sc, prob) in enumerate(zip(words, lm_scores, similarities, scores, probs))),
            key=lambda a: a.score, reverse=True,
        )

        best = alternatives[0]
        best_idx = words.index(best.word)
        if best.is_original:
            apply, reason = False, ORIGINAL_BEST
        elif (best.score - orig_score) < self.config.min_score_margin:
            apply, reason = False, BELOW_SCORE_MARGIN
        elif best.probability < self.config.threshold:
            apply, reason = False, BELOW_THRESHOLD
        else:
            apply, reason = True, CORRECTED

        logger.debug("%r -> %r (p=%.3f, %s)", original, best.word, best.probability, reason)
        return ScoringDecision(
            **base,
            best_word=best.word,
            corrected_sentence=variants[best_idx] if apply else clean_sentence,
            confidence=best.probability,
            original_confidence=probs[0],
            apply_correction=apply,
            reason=reason,
            alternatives=alternatives,
        )
