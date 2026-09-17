"""Context-aware scoring tests.

Logic tests use a fake LM (fast, offline). Tests marked ``lm`` use real GPT-2
and are skipped if the model cannot be loaded (e.g. offline with no cache).
Run only the fast tests with ``pytest -m "not lm"``.
"""

import math

import pytest

from src.correction.candidates import PhoneticCandidateGenerator
from src.correction.context_scorer import (
    BELOW_THRESHOLD,
    CORRECTED,
    NO_CANDIDATES,
    NO_PRONUNCIATION,
    ORIGINAL_BEST,
    ContextAwareScorer,
    ContextScorerConfig,
    match_case,
    replace_token,
    softmax,
)
from src.correction.lm_scorer import CausalLMScorer, LMError
from src.correction.pronunciation import PronunciationDict

PRON = PronunciationDict({
    "mail": [["M", "EY1", "L"]], "male": [["M", "EY1", "L"]], "nail": [["N", "EY1", "L"]],
    "see": [["S", "IY1"]], "sea": [["S", "IY1"]],
    "right": [["R", "AY1", "T"]], "write": [["R", "AY1", "T"]],
    "cat": [["K", "AE1", "T"]],
})


class FakeLM:
    """Returns a fixed log-prob per sentence (default -20); records calls."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score_sentences(self, sentences):
        self.calls.append(list(sentences))
        return [self.scores.get(s, -20.0) for s in sentences]


def make_scorer(scores, **config):
    gen = PhoneticCandidateGenerator(pron_dict=PRON, max_candidates=20)
    return ContextAwareScorer(gen, FakeLM(scores), ContextScorerConfig(**config))


# ------------------------------------------------------------------ helpers
def test_replace_token_keeps_punctuation_and_case():
    tokens = ["Send", "the", '"Male,"', "now"]
    assert replace_token(tokens, 2, "mail") == 'Send the "Mail," now'
    assert match_case("mail", "MALE") == "MAIL"
    assert match_case("mail", "male") == "mail"


def test_softmax():
    probs = softmax([0.0, 0.0])
    assert probs == pytest.approx([0.5, 0.5])
    assert sum(softmax([-1000.0, -1.0, -2.0])) == pytest.approx(1.0)


def test_config_validation():
    with pytest.raises(ValueError):
        ContextScorerConfig(threshold=1.5)
    with pytest.raises(ValueError):
        ContextScorerConfig(phonetic_weight=-1)
    with pytest.raises(ValueError):
        ContextScorerConfig(max_candidates=0)


# ---------------------------------------------------------- decision logic
def test_applies_confident_correction():
    scorer = make_scorer({"I got a mail today": -10.0})
    d = scorer.score_word("I got a male today", "male")
    assert d.best_word == "mail" and d.apply_correction and d.reason == CORRECTED
    assert d.corrected_sentence == "I got a mail today"
    assert d.confidence > 0.99 and d.original_confidence < 0.01
    assert d.alternatives[0].word == "mail"


def test_keeps_original_when_it_scores_best():
    scorer = make_scorer({"He is a male nurse": -10.0})
    d = scorer.score_word("He is a male nurse", "male")
    assert d.best_word == "male" and not d.apply_correction and d.reason == ORIGINAL_BEST
    assert d.corrected_sentence == "He is a male nurse"


def test_threshold_blocks_uncertain_correction():
    # mail slightly beats male: p(mail) = sigmoid(0.5) ~= 0.62
    scores = {"a male": -10.0, "a mail": -9.5}
    d = make_scorer(scores, threshold=0.7, phonetic_weight=0.0).score("a male", 1)
    assert d.best_word == "mail" and d.reason == BELOW_THRESHOLD
    assert not d.apply_correction and d.corrected_sentence == "a male"
    assert d.confidence == pytest.approx(1 / (1 + math.exp(-0.5)), abs=1e-3)

    d = make_scorer(scores, threshold=0.6, phonetic_weight=0.0).score("a male", 1)
    assert d.apply_correction


def test_threshold_one_never_corrects_ties():
    d = make_scorer({}, threshold=1.0).score("a male", 1)  # all alternatives tie
    assert not d.apply_correction


def test_phonetic_weight_penalizes_non_homophones():
    # nail (similarity < 1) gets the same LM score as mail; mail should win
    scorer = make_scorer({"a mail": -10.0, "a nail": -10.0}, threshold=0.0)
    d = scorer.score("a male", 1)
    assert d.best_word == "mail"
    nail = next(a for a in d.alternatives if a.word == "nail")
    assert nail.score < nail.lm_log_prob


def test_unknown_word_is_kept_without_lm_call():
    scorer = make_scorer({})
    d = scorer.score("the zzqx is here", 1)
    assert d.reason == NO_PRONUNCIATION and not d.apply_correction
    assert d.confidence == 1.0 and scorer.lm.calls == []


def test_word_without_candidates_is_kept():
    d = make_scorer({}).score("a cat", 1)
    assert d.reason == NO_CANDIDATES and not d.apply_correction


def test_bad_inputs():
    scorer = make_scorer({})
    with pytest.raises(ValueError):
        scorer.score("   ", 0)
    with pytest.raises(IndexError):
        scorer.score("a male", 5)
    with pytest.raises(ValueError):
        scorer.score_word("a male", "mail")


def test_score_word_occurrence_and_to_dict():
    scorer = make_scorer({})
    d = scorer.score_word("see and see", "see", occurrence=1)
    assert d.word_index == 2
    assert d.to_dict()["alternatives"][0]["word"] in {"see", "sea"}


# ------------------------------------------------------- real GPT-2 tests
@pytest.fixture(scope="module")
def real_scorer():
    lm = CausalLMScorer()
    try:
        lm.load()
    except LMError as exc:
        pytest.skip(f"GPT-2 unavailable: {exc}")
    return ContextAwareScorer(lm=lm)


@pytest.mark.lm
@pytest.mark.parametrize("sentence, word, expected", [
    ("I will send you a male tomorrow", "male", "mail"),
    ("I can sea the ship", "sea", "see"),
    ("We sailed across the see", "see", "sea"),
    ("Please right your name here", "right", "write"),
    ("Turn write at the corner", "write", "right"),
])
def test_lm_corrects_wrong_homophone(real_scorer, sentence, word, expected):
    d = real_scorer.score_word(sentence, word)
    assert d.apply_correction, d.alternatives[:3]
    assert d.best_word == expected


@pytest.mark.lm
@pytest.mark.parametrize("sentence, word", [
    ("Please check your mail", "mail"),
    ("He is a male nurse", "male"),
    ("I can see the ship", "see"),
    ("The ship sailed across the sea", "sea"),
    ("Please write your name here", "write"),
    ("Turn right at the corner", "right"),
    ("The weather is nice today", "weather"),
])
def test_lm_keeps_correct_word(real_scorer, sentence, word):
    d = real_scorer.score_word(sentence, word)
    assert not d.apply_correction, d.alternatives[:3]
    assert d.reason == ORIGINAL_BEST
    assert d.corrected_sentence == sentence


@pytest.mark.lm
def test_lm_scorer_prefers_fluent_sentence(real_scorer):
    good, bad = real_scorer.lm.score_sentences(["I can see the ship", "I can sea the ship"])
    assert good > bad
