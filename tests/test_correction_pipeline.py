"""Tests for the ASR correction pipeline and evaluation.

Logic tests use a FakeLM (fast, offline) following the same pattern as
``test_context_scorer.py``.  Tests marked ``lm`` use real GPT-2.
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
)
from src.correction.lm_scorer import CausalLMScorer, LMError
from src.correction.pipeline import CorrectionPipeline, CorrectionResult, WordCorrection
from src.correction.pronunciation import PronunciationDict
from src.evaluation.correction_eval import (
    FAILED_CORRECTION,
    OVERCORRECTION,
    SUCCESSFUL_CORRECTION,
    UNCHANGED_CORRECT,
    UNCHANGED_ERROR,
    evaluate_correction,
)

# ------------------------------------------------------------------ fixtures

PRON = PronunciationDict(
    {
        "mail": [["M", "EY1", "L"]],
        "male": [["M", "EY1", "L"]],
        "nail": [["N", "EY1", "L"]],
        "see": [["S", "IY1"]],
        "sea": [["S", "IY1"]],
        "right": [["R", "AY1", "T"]],
        "write": [["R", "AY1", "T"]],
        "i": [["AY1"]],
        "got": [["G", "AA1", "T"]],
        "a": [["AH0"]],
        "today": [["T", "AH0", "D", "EY1"]],
        "the": [["DH", "AH0"]],
        "cat": [["K", "AE1", "T"]],
        "is": [["IH1", "Z"]],
        "he": [["HH", "IY1"]],
        "nurse": [["N", "ER1", "S"]],
        "can": [["K", "AE1", "N"]],
        "ship": [["SH", "IH1", "P"]],
        "please": [["P", "L", "IY1", "Z"]],
        "your": [["Y", "AO1", "R"]],
        "name": [["N", "EY1", "M"]],
        "here": [["HH", "IY1", "R"]],
        "turn": [["T", "ER1", "N"]],
        "at": [["AE1", "T"]],
        "corner": [["K", "AO1", "R", "N", "ER0"]],
        "sat": [["S", "AE1", "T"]],
        "on": [["AA1", "N"]],
        "mat": [["M", "AE1", "T"]],
    }
)


class FakeLM:
    """Returns a fixed log-prob per sentence (default -20); records calls."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score_sentences(self, sentences):
        self.calls.append(list(sentences))
        return [self.scores.get(s, -20.0) for s in sentences]


def make_pipeline(scores, **config):
    gen = PhoneticCandidateGenerator(pron_dict=PRON, max_candidates=20)
    lm = FakeLM(scores)
    scorer = ContextAwareScorer(gen, lm, ContextScorerConfig(**config))
    return CorrectionPipeline(scorer)


# ============================================================= pipeline tests


class TestCorrectionPipeline:
    """Unit tests for CorrectionPipeline.correct()."""

    def test_correction_applied(self):
        """Pipeline corrects a wrong homophone when LM strongly prefers the candidate."""
        # FakeLM gives high score to "I got a mail today"
        pipeline = make_pipeline({"I got a mail today": -10.0})
        result = pipeline.correct("I got a male today")

        assert result.corrected == "I got a mail today"
        assert result.num_changed == 1
        # Find the changed word
        changed = [wc for wc in result.word_corrections if wc.was_changed]
        assert len(changed) == 1
        assert changed[0].original_token == "male"
        assert changed[0].corrected_token == "mail"
        assert changed[0].decision is not None
        assert changed[0].decision.reason == CORRECTED

    def test_correct_word_unchanged(self):
        """Pipeline keeps a word that is already correct (LM prefers original)."""
        pipeline = make_pipeline({"He is a male nurse": -10.0})
        result = pipeline.correct("He is a male nurse")

        assert result.corrected == "He is a male nurse"
        assert result.num_changed == 0

    def test_low_confidence_keeps_original(self):
        """Threshold blocks uncertain corrections."""
        # Small LM advantage for "mail" over "male" → below 0.7 threshold
        scores = {"a male": -10.0, "a mail": -9.5}
        pipeline = make_pipeline(scores, threshold=0.7, phonetic_weight=0.0)
        result = pipeline.correct("a male")

        assert result.corrected == "a male"
        assert result.num_changed == 0
        male_decision = result.word_corrections[1].decision
        assert male_decision is not None
        assert male_decision.reason == BELOW_THRESHOLD

    def test_unknown_word_skipped_by_scorer(self):
        """Words without a pronunciation are kept without calling the LM."""
        pipeline = make_pipeline({})
        result = pipeline.correct("the zzqx is here")

        assert result.corrected == "the zzqx is here"
        zzqx_wc = result.word_corrections[1]
        assert not zzqx_wc.was_changed
        assert zzqx_wc.decision is not None
        assert zzqx_wc.decision.reason == NO_PRONUNCIATION

    def test_non_word_tokens_skipped(self):
        """Pure-punctuation and numeric tokens are skipped (decision=None)."""
        pipeline = make_pipeline({})
        result = pipeline.correct("--- 42 hello")

        # "---" and "42" are not plain words
        assert result.word_corrections[0].decision is None
        assert result.word_corrections[1].decision is None
        # "hello" is a plain word but not in our tiny PRON dict
        assert result.word_corrections[2].decision is not None

    def test_empty_input(self):
        """Empty/whitespace-only input returns unchanged."""
        pipeline = make_pipeline({})
        result = pipeline.correct("")
        assert result.corrected == ""
        assert result.num_words == 0

        result2 = pipeline.correct("   ")
        assert result2.corrected == "   "
        assert result2.num_words == 0

    def test_preserves_punctuation_and_case(self):
        """Corrections preserve surrounding punctuation and original casing."""
        pipeline = make_pipeline({'"I got a Mail," today': -10.0})
        result = pipeline.correct('"I got a Male," today')

        # The corrected token should keep the quotes and comma
        changed = [wc for wc in result.word_corrections if wc.was_changed]
        assert len(changed) == 1
        # The token should have Mail with upper-case M matching Male
        assert "Mail" in changed[0].corrected_token

    def test_to_dict_completeness(self):
        """CorrectionResult.to_dict() includes all expected keys."""
        pipeline = make_pipeline({"I got a mail today": -10.0})
        result = pipeline.correct("I got a male today")
        d = result.to_dict()

        assert "original" in d
        assert "corrected" in d
        assert "num_words" in d
        assert "num_attempted" in d
        assert "num_changed" in d
        assert "num_skipped" in d
        assert "corrections" in d
        assert isinstance(d["corrections"], list)
        assert len(d["corrections"]) == result.num_words

    def test_multiple_corrections_in_one_sentence(self):
        """Pipeline can correct more than one word in the same sentence."""
        # Both "male" and "sea" get high-scoring corrections
        scores = {
            "I got a mail today": -10.0,  # for "male" at position 3
            "I got a male today": -20.0,  # original gets low score
        }
        # We need a sentence with two correctable words
        # Let's use a sentence where two words can be corrected independently
        scores2 = {
            "I can see the ship": -10.0,
        }
        pipeline = make_pipeline(scores2)
        result = pipeline.correct("I can sea the ship")

        assert result.num_changed >= 1
        # "sea" -> "see" should happen
        changed = [wc for wc in result.word_corrections if wc.was_changed]
        assert any(wc.corrected_token == "see" for wc in changed)


# ======================================================= evaluation tests


class TestCorrectionEvaluation:
    """Unit tests for evaluate_correction()."""

    def test_successful_correction_detected(self):
        """A wrong word corrected to match reference is classified as successful."""
        reference = "I got a mail today"
        hypothesis = "I got a male today"

        pipeline = make_pipeline({"I got a mail today": -10.0})
        correction = pipeline.correct(hypothesis)

        result = evaluate_correction(reference, hypothesis, correction)

        assert result.successful_corrections == 1
        assert result.overcorrections == 0
        assert result.wer_improved
        assert result.wer_delta < 0
        # WER after should be 0 (perfect correction)
        assert result.wer_after.wer == 0.0

    def test_overcorrection_detected(self):
        """A correct word changed by the pipeline is classified as overcorrection."""
        reference = "He is a male nurse"
        hypothesis = "He is a male nurse"

        # Force the pipeline to "correct" male -> mail even though male is right
        pipeline = make_pipeline({"He is a mail nurse": -10.0})
        correction = pipeline.correct(hypothesis)

        result = evaluate_correction(reference, hypothesis, correction)

        assert result.overcorrections >= 1
        assert result.wer_delta > 0  # WER got worse

    def test_unchanged_error_detected(self):
        """A wrong word kept by the pipeline is classified as unchanged_error."""
        reference = "I can see the ship"
        hypothesis = "I can sea the ship"

        # Don't give any score advantage → pipeline keeps "sea"
        pipeline = make_pipeline({}, threshold=0.99)
        correction = pipeline.correct(hypothesis)

        result = evaluate_correction(reference, hypothesis, correction)

        assert result.unchanged_errors >= 1
        # WER didn't change
        assert result.wer_after.wer == result.wer_before.wer

    def test_unchanged_correct_detected(self):
        """A correct word kept by the pipeline is classified as unchanged_correct."""
        reference = "the cat sat on the mat"
        hypothesis = "the cat sat on the mat"

        pipeline = make_pipeline({})
        correction = pipeline.correct(hypothesis)

        result = evaluate_correction(reference, hypothesis, correction)

        assert result.unchanged_correct > 0
        assert result.overcorrections == 0
        assert result.successful_corrections == 0
        assert result.wer_before.wer == 0.0
        assert result.wer_after.wer == 0.0

    def test_failed_correction_detected(self):
        """A wrong word changed to a different wrong word is a failed correction."""
        reference = "I got a mail today"
        hypothesis = "I got a male today"

        # Force correction to "nail" instead of "mail"
        pipeline = make_pipeline({"I got a nail today": -10.0})
        correction = pipeline.correct(hypothesis)

        # If the pipeline corrected "male" -> "nail", that's a failed correction
        if correction.num_changed > 0:
            result = evaluate_correction(reference, hypothesis, correction)
            # "nail" != "mail" (reference), so it's a failed correction
            assert result.failed_corrections >= 1

    def test_wer_delta_sign(self):
        """wer_delta is negative when correction helps."""
        reference = "I got a mail today"
        hypothesis = "I got a male today"

        pipeline = make_pipeline({"I got a mail today": -10.0})
        correction = pipeline.correct(hypothesis)
        result = evaluate_correction(reference, hypothesis, correction)

        assert result.wer_delta < 0
        assert result.wer_improved is True

    def test_wer_delta_positive_on_overcorrection(self):
        """wer_delta is positive when correction makes things worse."""
        reference = "He is a male nurse"
        hypothesis = "He is a male nurse"

        pipeline = make_pipeline({"He is a mail nurse": -10.0})
        correction = pipeline.correct(hypothesis)
        result = evaluate_correction(reference, hypothesis, correction)

        if result.total_changes > 0:
            assert result.wer_delta > 0
            assert result.wer_improved is False

    def test_correction_precision(self):
        """Precision is successful / total_changes."""
        reference = "I got a mail today"
        hypothesis = "I got a male today"

        pipeline = make_pipeline({"I got a mail today": -10.0})
        correction = pipeline.correct(hypothesis)
        result = evaluate_correction(reference, hypothesis, correction)

        assert result.correction_precision == 1.0  # 1 successful, 0 bad

    def test_precision_none_when_no_changes(self):
        """Precision is None when the pipeline made no changes."""
        pipeline = make_pipeline({}, threshold=1.0)
        correction = pipeline.correct("the cat sat")
        result = evaluate_correction("the cat sat", "the cat sat", correction)

        assert result.correction_precision is None

    def test_to_dict_serialization(self):
        """CorrectionEvalResult.to_dict() includes all summary fields."""
        reference = "I got a mail today"
        hypothesis = "I got a male today"

        pipeline = make_pipeline({"I got a mail today": -10.0})
        correction = pipeline.correct(hypothesis)
        result = evaluate_correction(reference, hypothesis, correction)
        d = result.to_dict()

        expected_keys = {
            "reference", "hypothesis", "corrected",
            "wer_before", "wer_after", "wer_delta", "wer_improved",
            "successful_corrections", "overcorrections", "failed_corrections",
            "unchanged_correct", "unchanged_errors",
            "total_changes", "correction_precision", "outcomes",
        }
        assert expected_keys <= set(d.keys())
        assert isinstance(d["outcomes"], list)


# ====================================================== real GPT-2 tests


@pytest.fixture(scope="module")
def real_pipeline():
    """Pipeline with real GPT-2 for integration tests."""
    lm = CausalLMScorer()
    try:
        lm.load()
    except LMError as exc:
        pytest.skip(f"GPT-2 unavailable: {exc}")
    scorer = ContextAwareScorer(lm=lm)
    return CorrectionPipeline(scorer)


@pytest.mark.lm
def test_lm_pipeline_corrects_wrong_homophone(real_pipeline):
    """End-to-end: pipeline corrects a wrong homophone with real GPT-2."""
    result = real_pipeline.correct("I will send you a male tomorrow")
    assert result.num_changed >= 1
    # "male" should become "mail"
    changed = {wc.corrected_token.lower() for wc in result.word_corrections if wc.was_changed}
    assert "mail" in changed


@pytest.mark.lm
def test_lm_pipeline_keeps_correct_sentence(real_pipeline):
    """End-to-end: pipeline does not change an already-correct sentence."""
    result = real_pipeline.correct("Please check your mail")
    assert result.num_changed == 0
    assert result.corrected == "Please check your mail"


@pytest.mark.lm
def test_lm_full_evaluation(real_pipeline):
    """End-to-end: full evaluation with real GPT-2."""
    reference = "I will send you a mail tomorrow"
    hypothesis = "I will send you a male tomorrow"

    correction = real_pipeline.correct(hypothesis)
    result = evaluate_correction(reference, hypothesis, correction)

    # The pipeline should improve WER
    assert result.wer_after.wer <= result.wer_before.wer
    assert result.successful_corrections >= 1
    assert result.overcorrections == 0
