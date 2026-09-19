"""Unit and integration tests for improved context-aware ASR correction safeguards."""

import pytest
from src.correction.context_scorer import (
    BELOW_SCORE_MARGIN,
    BELOW_THRESHOLD,
    CORRECTED,
    NO_CANDIDATES,
    ORIGINAL_BEST,
    ContextAwareScorer,
    ContextScorerConfig,
)
from src.correction.candidates import Candidate, CandidateResult, PhoneticCandidateGenerator
from src.correction.pronunciation import PronunciationDict
from src.evaluation.evaluate_improved_correction import aggregate_pipeline_stats
from src.evaluation.tune_correction_pipeline import DevEvalResult, evaluate_on_dev
from src.correction.pipeline import CorrectionPipeline


class DummyLM:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def score_sentences(self, sentences: list[str]) -> list[float]:
        return [self.scores.get(s, -20.0) for s in sentences]


def test_context_scorer_config_safeguards_validation() -> None:
    # Valid config
    cfg = ContextScorerConfig(
        threshold=0.7,
        phonetic_weight=1.0,
        max_candidates=20,
        min_score_margin=3.5,
        min_phonetic_similarity=0.80,
        retention_bias=1.0,
    )
    assert cfg.min_score_margin == 3.5
    assert cfg.min_phonetic_similarity == 0.80
    assert cfg.retention_bias == 1.0

    # Negative margin
    with pytest.raises(ValueError, match="min_score_margin"):
        ContextScorerConfig(min_score_margin=-0.5)

    # Invalid similarity
    with pytest.raises(ValueError, match="min_phonetic_similarity"):
        ContextScorerConfig(min_phonetic_similarity=1.5)

    # Negative bias
    with pytest.raises(ValueError, match="retention_bias"):
        ContextScorerConfig(retention_bias=-1.0)


def test_min_score_margin_blocks_marginal_candidate() -> None:
    pron_dict = PronunciationDict({
        "bottle": [["B", "AA1", "T", "AH0", "L"]],
        "battle": [["B", "AE1", "T", "AH0", "L"]],
    })
    generator = PhoneticCandidateGenerator(pron_dict=pron_dict, max_candidates=5)

    # LM gives battle a small advantage (+1.5)
    lm = DummyLM({
        "took a bottle": -10.0,
        "took a battle": -8.5,
    })

    # Without margin: battle beats bottle and threshold 0.7 is met
    cfg_loose = ContextScorerConfig(threshold=0.7, min_score_margin=0.0)
    scorer_loose = ContextAwareScorer(generator=generator, lm=lm, config=cfg_loose)
    d_loose = scorer_loose.score("took a bottle", 2)
    assert d_loose.apply_correction is True
    assert d_loose.best_word == "battle"
    assert d_loose.reason == CORRECTED

    # With margin=3.0: 1.5 < 3.0, so correction is rejected
    cfg_strict = ContextScorerConfig(threshold=0.7, min_score_margin=3.0)
    scorer_strict = ContextAwareScorer(generator=generator, lm=lm, config=cfg_strict)
    d_strict = scorer_strict.score("took a bottle", 2)
    assert d_strict.apply_correction is False
    assert d_strict.best_word == "battle"
    assert d_strict.reason == BELOW_SCORE_MARGIN


def test_min_phonetic_similarity_filters_dissimilar_candidate() -> None:
    pron_dict = PronunciationDict({
        "difference": [["D", "IH1", "F", "ER0", "AH0", "N", "S"]],
        "distance": [["D", "IH1", "S", "T", "AH0", "N", "S"]],
    })
    generator = PhoneticCandidateGenerator(pron_dict=pron_dict, max_candidates=5, min_similarity=0.5)

    # Huge LM advantage for distance (+10.0)
    lm = DummyLM({
        "the difference": -20.0,
        "the distance": -10.0,
    })

    # With min_phonetic_similarity=0.85: distance (sim ~0.76) is filtered out completely
    cfg = ContextScorerConfig(threshold=0.7, min_phonetic_similarity=0.85)
    scorer = ContextAwareScorer(generator=generator, lm=lm, config=cfg)
    d = scorer.score("the difference", 1)
    assert d.apply_correction is False
    assert d.reason == NO_CANDIDATES


def test_retention_bias_penalizes_alternatives() -> None:
    pron_dict = PronunciationDict({
        "floods": [["F", "L", "AH1", "D", "Z"]],
        "flood": [["F", "L", "AH1", "D"]],
    })
    generator = PhoneticCandidateGenerator(pron_dict=pron_dict, max_candidates=5)

    # LM difference is +2.0
    lm = DummyLM({
        "severe floods": -10.0,
        "severe flood": -8.0,
    })

    # With retention_bias=3.0, effective candidate score = -8.0 + log(sim) - 3.0 < -10.0
    # So original is best
    cfg = ContextScorerConfig(threshold=0.7, retention_bias=3.0)
    scorer = ContextAwareScorer(generator=generator, lm=lm, config=cfg)
    d = scorer.score("severe floods", 1)
    assert d.apply_correction is False
    assert d.best_word == "floods"
    assert d.reason == ORIGINAL_BEST


def test_aggregate_pipeline_stats_arithmetic() -> None:
    records = [
        {
            "test_raw_ref_words": 10,
            "test_raw_sub": 1,
            "test_raw_del": 0,
            "test_raw_ins": 0,
            "test_raw_hits": 9,
            "test_corr_sub": 0,
            "test_corr_del": 0,
            "test_corr_ins": 0,
            "test_corr_hits": 10,
            "test_raw_wer": 0.10,
            "test_corr_wer": 0.0,
            "test_delta_wer": -0.10,
            "test_successful": 1,
            "test_overcorrections": 0,
            "test_failed": 0,
            "test_unchanged_correct": 9,
            "test_unchanged_errors": 0,
        },
        {
            "test_raw_ref_words": 10,
            "test_raw_sub": 0,
            "test_raw_del": 0,
            "test_raw_ins": 0,
            "test_raw_hits": 10,
            "test_corr_sub": 1,
            "test_corr_del": 0,
            "test_corr_ins": 0,
            "test_corr_hits": 9,
            "test_raw_wer": 0.0,
            "test_corr_wer": 0.10,
            "test_delta_wer": 0.10,
            "test_successful": 0,
            "test_overcorrections": 1,
            "test_failed": 0,
            "test_unchanged_correct": 9,
            "test_unchanged_errors": 0,
        },
    ]
    stats = aggregate_pipeline_stats(records, "test")
    assert stats["count"] == 2
    assert stats["total_words"] == 20
    assert stats["raw_corpus_wer"] == 1 / 20
    assert stats["corrected_corpus_wer"] == 1 / 20
    assert stats["delta_corpus_wer"] == 0.0
    assert stats["successful_corrections"] == 1
    assert stats["overcorrections"] == 1
    assert stats["total_changes"] == 2
    assert stats["correction_precision"] == 0.50
