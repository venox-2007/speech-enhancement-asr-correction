import pytest

from src.correction.candidates import PhoneticCandidateGenerator
from src.correction.phonetic_distance import (
    phonetic_similarity,
    substitution_cost,
    weighted_phoneme_distance,
)
from src.correction.pronunciation import PronunciationDict, normalize_word


@pytest.fixture(scope="module")
def generator():
    """Full CMUdict vocabulary (loaded once for the module)."""
    return PhoneticCandidateGenerator(max_candidates=50)


# ---------------------------------------------------------------- pronunciation
def test_normalize_word():
    assert normalize_word("  Mail, ") == "mail"
    assert normalize_word('"Don\'t!"') == "don't"


def test_lookup_strips_stress():
    d = PronunciationDict({"mail": [["M", "EY1", "L"]]})
    assert d.lookup("MAIL") == [("M", "EY", "L")]
    assert d.lookup("unknownword") == []
    assert "mail" in d and len(d) == 1


# --------------------------------------------------------------------- distance
def test_substitution_costs_are_ordered():
    assert substitution_cost("T", "T") == 0
    assert substitution_cost("T", "D") < substitution_cost("T", "K") < substitution_cost("T", "IY")


def test_weighted_distance():
    assert weighted_phoneme_distance(("M", "EY", "L"), ("M", "EY", "L")) == 0
    assert weighted_phoneme_distance(("M", "EY", "L"), ("M", "EY")) == 1.0
    assert weighted_phoneme_distance((), ("S",)) == 1.0
    assert phonetic_similarity(("B", "AE", "T"), ("P", "AE", "T")) == pytest.approx(1 - 0.4 / 3)


# ------------------------------------------------------------------- candidates
@pytest.mark.parametrize("word, expected", [
    ("mail", "male"), ("male", "mail"),
    ("see", "sea"), ("sea", "see"),
    ("right", "write"), ("write", "right"),
    ("their", "there"), ("weather", "whether"),
])
def test_known_homophones(generator, word, expected):
    result = generator.generate(word)
    assert result.known
    match = next(c for c in result.candidates if c.word == expected)
    assert match.is_homophone and match.similarity == 1.0


def test_near_match_is_found_and_ranked_after_homophones(generator):
    result = generator.generate("right")
    words = result.words
    assert "ride" in words  # T -> D differs only in voicing
    assert words.index("write") < words.index("ride")
    distances = [c.distance for c in result.candidates]
    assert distances == sorted(distances)


def test_query_word_is_excluded(generator):
    assert "mail" not in generator.generate("mail").words


def test_token_punctuation_and_case(generator):
    result = generator.generate("Mail,")
    assert result.word == "mail" and "male" in result.words


@pytest.mark.parametrize("token", ["zzqxv", "42", "", "   ", "!!!"])
def test_unknown_tokens_are_handled(generator, token):
    result = generator.generate(token)
    assert result.known is False
    assert result.candidates == [] and result.pronunciations == []


def test_restricted_vocabulary():
    gen = PhoneticCandidateGenerator(vocabulary=["male", "mail", "nail", "sea", "qqqx"])
    assert gen.generate("mail").words == ["male", "nail"]
    assert gen.generate("see").words == ["sea"]


def test_max_candidates_and_similarity_threshold():
    gen = PhoneticCandidateGenerator(max_candidates=3, min_similarity=0.9)
    cands = gen.generate("right").candidates
    assert 0 < len(cands) <= 3
    assert all(c.similarity >= 0.9 for c in cands)


def test_explicit_pronunciation_for_oov(generator):
    # e.g. phonemes produced by a G2P model for a word missing from CMUdict
    words = [c.word for c in generator.candidates_for_pronunciations([("M", "EY", "L")])]
    assert {"mail", "male"} <= set(words)
    with pytest.raises(ValueError):
        generator.candidates_for_pronunciations([("XX",)])


def test_generate_many_and_to_dict(generator):
    results = generator.generate_many(["see", "zzqxv"])
    assert [r.known for r in results] == [True, False]
    d = results[0].to_dict()
    assert d["word"] == "see" and "is_homophone" in d["candidates"][0]


def test_invalid_arguments():
    with pytest.raises(ValueError):
        PhoneticCandidateGenerator(pron_dict=PronunciationDict({}), max_edits=-1)
    with pytest.raises(ValueError):
        PhoneticCandidateGenerator(pron_dict=PronunciationDict({}), min_similarity=1.5)
