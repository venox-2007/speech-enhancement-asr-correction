import pytest

from src.evaluation.wer import alignment_report, compute_wer, normalize_text


def test_identical_after_normalization():
    r = compute_wer("The cat sat on the mat.", "the cat, sat on the MAT")
    assert r.wer == 0.0
    assert r.hits == 6 and r.reference_words == 6


def test_edit_counts():
    # ref: a b c d  | hyp: a x c d e  -> 1 substitution, 1 insertion
    r = compute_wer("a b c d", "a x c d e")
    assert (r.substitutions, r.deletions, r.insertions, r.hits) == (1, 0, 1, 3)
    assert r.wer == pytest.approx(2 / 4)


def test_deletion():
    r = compute_wer("one two three", "one three")
    assert r.deletions == 1 and r.wer == pytest.approx(1 / 3)


def test_empty_hypothesis_is_full_error():
    r = compute_wer("hello world", "")
    assert r.wer == 1.0 and r.deletions == 2


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        compute_wer("  ...  ", "anything")


def test_no_normalize_counts_case_differences():
    assert compute_wer("Hello world", "hello world", normalize=False).wer == pytest.approx(0.5)


def test_normalize_text():
    assert normalize_text("  Hello,   World! ") == "hello world"


def test_alignment_report_is_text():
    assert "REF" in alignment_report("a b", "a c")
