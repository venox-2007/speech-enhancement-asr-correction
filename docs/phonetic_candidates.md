# Milestone 2 — Phonetic Candidate Generation

```text
ASR word → pronunciation (CMUdict) → similar-sounding vocabulary words → ranked candidates
```

This stage only **proposes** replacements for an ASR word. It does not decide whether
to replace anything. Context scoring, the confidence threshold, and the correct/keep
decision are later milestones. No LLM is used.

## Pronunciation resource

We use the [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
through the `cmudict` Python package:

- about 126k English words in ARPAbet (39 phonemes), bundled with the package; no download or model needed;
- pure Python, compatible with Python 3.11, and loads in about 1 s;
- stress markers are removed (`EY1` → `EY`), and all pronunciation variants of a word are used.

Licensing: the dictionary data is BSD-licensed, but the `cmudict` Python wrapper package
is GPL-3.0. It is only installed as a dependency, not bundled with this code. Check that this suits how the project will be distributed.

Neural grapheme-to-phoneme (G2P) models such as `g2p_en` were not used for now: they add heavier
dependencies and extra downloads. Words missing from CMUdict are handled as described below.

## Method

1. **Normalize the token**: lowercase it and strip surrounding punctuation (`"Mail,"` → `mail`).
2. **Look up** its pronunciations. If there are none, the word is reported as
   `known=False` with no candidates (see below).
3. **Shortlist**: compare against every pronunciation in the vocabulary using
   plain phoneme edit distance (RapidFuzz, implemented in C++). Each phoneme is mapped to one character
   so whole sequences can be compared as strings. The edit budget is
   `max(1, min(max_edits, len(pron) // 2))`, so short words get 1 edit and longer ones up to 2.
4. **Rank** the shortlist with a weighted phoneme edit distance:

   | Operation | Cost |
   |---|---|
   | identical phoneme | 0 |
   | close pair (voicing, e.g. `T/D`, `S/Z`; near vowels, e.g. `IH/IY`; `M/N`, …) | 0.4 |
   | same phoneme class (vowel, stop, fricative, nasal, …) | 0.7 |
   | different class | 1.0 |
   | insertion / deletion | 1.0 |

   `similarity = 1 − distance / length of the longer pronunciation`. Candidates
   below `min_similarity` (default 0.5) are dropped. Results are sorted by distance,
   so exact homophones (distance 0) come first, and are capped at `max_candidates` (default 10).

The cost values are **hand-set heuristics**. They have not been tuned or evaluated yet.

## Unknown words

Numbers, empty tokens, punctuation and words not in CMUdict return
`CandidateResult(known=False, candidates=[])`, so later stages leave these words
unchanged. If a G2P model is added later, its phonemes can be passed to
`candidates_for_pronunciations()`.

## Vocabulary

By default, every plain alphabetic CMUdict word can be a candidate. CMUdict contains many
rare words, names and spellings (for `mail`, the top candidates include `maile` and
`mayle` alongside `male`). A restricted vocabulary, such as the
dataset/task vocabulary or a frequency-filtered word list, gives more plausible candidates:

```python
from src.correction.candidates import PhoneticCandidateGenerator

gen = PhoneticCandidateGenerator()                        # full CMUdict
gen.generate("right").words        # ['reit', 'rite', 'wright', 'write', 'ride', ...]

gen = PhoneticCandidateGenerator(vocabulary=["male", "mail", "nail", "sea"])
gen.generate("mail").words         # ['male', 'nail']
```

Vocabulary words without a pronunciation are skipped with a warning. Choosing the vocabulary
(and possibly a word-frequency prior) is left to the context-scoring milestone.

## Output

`generate(token)` returns a `CandidateResult` with `token`, `word`, `known`,
`pronunciations` and `candidates`. Each `Candidate` has `word`, `pronunciation`,
`distance`, `similarity` and `is_homophone`. `to_dict()` gives a
JSON-serializable form.

## Limitations

- The edit budget limits recall: pronunciations differing by more than 1–2 phonemes are
  not considered, even when the weighted distance would be small.
- Only single words map to single words. Word-boundary errors (`ice cream` ↔ `I scream`) are not handled.
- Candidate quality has not been measured on real ASR errors yet. Tests only check
  hand-picked examples (`mail/male`, `see/sea`, `right/write`, `their/there`, `weather/whether`).
