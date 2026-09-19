"""Unit tests for paired bootstrap and statistical analysis functions."""

import numpy as np
import pytest

from src.evaluation.statistical_analysis import (
    paired_bootstrap_wer,
    paired_utterance_outcomes,
    paired_wilcoxon_test,
)


def test_paired_bootstrap_wer_significant_difference() -> None:
    # System A has 5 errors per utterance, System B has 1 error per utterance
    # 50 utterances, 10 words each
    errors_a = [5] * 50
    errors_b = [1] * 50
    ref_words = [10] * 50

    res = paired_bootstrap_wer(errors_a, errors_b, ref_words, num_bootstrap=1000, seed=42)

    assert res["wer_a"] == pytest.approx(0.50)
    assert res["wer_b"] == pytest.approx(0.10)
    assert res["delta_sample"] == pytest.approx(-0.40)
    assert res["ci_lower"] < 0
    assert res["ci_upper"] < 0
    assert res["is_significant"] is True
    assert res["p_value"] < 0.01


def test_paired_bootstrap_wer_insignificant_difference() -> None:
    # System A and System B have identical error distributions with alternating +/- 1 errors
    errors_a = [2, 3, 2, 3] * 20
    errors_b = [3, 2, 3, 2] * 20
    ref_words = [10] * 80

    res = paired_bootstrap_wer(errors_a, errors_b, ref_words, num_bootstrap=1000, seed=42)

    assert res["wer_a"] == pytest.approx(0.25)
    assert res["wer_b"] == pytest.approx(0.25)
    assert res["delta_sample"] == pytest.approx(0.0)
    # 0 should be inside the confidence interval
    assert res["ci_lower"] <= 0.0 <= res["ci_upper"]
    assert res["is_significant"] is False
    assert res["p_value"] > 0.10


def test_bootstrap_reproducibility() -> None:
    errors_a = [3, 1, 0, 4, 2] * 10
    errors_b = [2, 1, 1, 3, 1] * 10
    ref_words = [10] * 50

    res1 = paired_bootstrap_wer(errors_a, errors_b, ref_words, num_bootstrap=500, seed=123)
    res2 = paired_bootstrap_wer(errors_a, errors_b, ref_words, num_bootstrap=500, seed=123)

    assert res1["ci_lower"] == res2["ci_lower"]
    assert res1["ci_upper"] == res2["ci_upper"]
    assert res1["p_value"] == res2["p_value"]


def test_paired_utterance_outcomes_calculation() -> None:
    wers_a = [0.10, 0.20, 0.05, 0.00]
    wers_b = [0.00, 0.30, 0.05, 0.00]

    outcomes = paired_utterance_outcomes(wers_a, wers_b)

    assert outcomes["count"] == 4
    assert outcomes["improved"] == 1  # 0.10 -> 0.00
    assert outcomes["worsened"] == 1  # 0.20 -> 0.30
    assert outcomes["unchanged"] == 2  # 0.05 -> 0.05, 0.00 -> 0.00
    assert outcomes["improved_pct"] == pytest.approx(25.0)
    assert outcomes["worsened_pct"] == pytest.approx(25.0)
    assert outcomes["unchanged_pct"] == pytest.approx(50.0)
    assert outcomes["delta_median"] == pytest.approx(0.0)


def test_paired_wilcoxon_test_handles_ties_and_differences() -> None:
    # All identical
    res_ties = paired_wilcoxon_test([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert res_ties["n_non_zero"] == 0
    assert res_ties["p_value"] == 1.0

    # Clear improvement
    wers_a = [0.20] * 20
    wers_b = [0.05] * 20
    res_diff = paired_wilcoxon_test(wers_a, wers_b)
    assert res_diff["n_non_zero"] == 20
    assert res_diff["p_value"] < 0.01
    assert res_diff["is_significant"] is True
