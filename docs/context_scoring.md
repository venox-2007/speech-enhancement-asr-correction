# Milestone 3 — Context-Aware Candidate Scoring

```text
sentence + target ASR word
   → phonetic candidates (milestone 2)
   → each alternative placed into the sentence and scored by a pretrained LM
   → probabilities over {original} ∪ candidates
   → correct / keep decision (confidence threshold)
```

No LLM API is used. The language model is **GPT-2 (124M parameters)**, run locally with
Hugging Face `transformers`.

## Scoring

For the original word and each candidate `w`, the sentence is rewritten with `w`
in place of the target word (punctuation and capitalization are kept) and scored:

```text
score(w) = log P_LM(sentence with w)  +  phonetic_weight · log(similarity(w))
```

- `log P_LM` is the sum of GPT-2 token log-probabilities for the **whole sentence**,
  with a BOS token prepended. Words after the target are conditioned on it, so
  context on both sides affects the score (`"… across the see"` vs `"… across the sea"`).
  Words split into several tokens are handled naturally.
- `similarity` is the phonetic similarity from milestone 2 (1.0 for homophones and for the
  original word). The term penalizes candidates that sound less like the original.
- `softmax(score)` over all alternatives gives each one a probability.

## Decision

| Condition | `reason` | Correction applied |
|---|---|---|
| word has no pronunciation (OOV, number, punctuation) | `no_pronunciation` | no |
| no phonetic candidates | `no_candidates` | no |
| original word has the highest score | `original_best` | no |
| another word is best but its probability < `threshold` | `below_threshold` | no |
| another word is best with probability ≥ `threshold` | `corrected` | **yes** |

The threshold makes the system leave a word unchanged unless it is confident. This
matters for overcorrection: every correction applied to a word that was already right
adds a new error.

## Configuration (`ContextScorerConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `threshold` | 0.7 | Minimum probability of the best candidate before a correction is applied |
| `phonetic_weight` | 1.0 | Weight of the phonetic similarity term (0 = LM only) |
| `max_candidates` | 20 | Candidates requested from the phonetic generator |

These defaults are **untuned placeholders**. They should be chosen on a development split
when the correction evaluation (WER before/after, overcorrection rate) is implemented.
The LM model name, device and batch size are set on `CausalLMScorer`.

## Usage

```python
from src.correction.context_scorer import ContextAwareScorer, ContextScorerConfig

scorer = ContextAwareScorer(config=ContextScorerConfig(threshold=0.8))
d = scorer.score_word("I will send you a male tomorrow", "male")
d.best_word, d.confidence, d.apply_correction, d.reason, d.corrected_sentence
# score(sentence, word_index) targets a specific whitespace token instead.
```

`ScoringDecision` also contains `original_confidence` and the full ranked
`alternatives` list (LM log-prob, similarity, combined score, probability), plus
`to_dict()` for JSON output.

To use a different model, pass any object with
`score_sentences(list[str]) -> list[float]` as `lm=`. The tests use this to
substitute a fake LM.

## Requirements and model download

- Adds `transformers` (Apache-2.0).
- The first use downloads GPT-2 (about 550 MB) to the Hugging Face cache (`~/.cache/huggingface`).
  On Windows without Developer Mode, `huggingface_hub` warns that symlinks are not
  supported. The warning is harmless; the cache just uses more disk space.
- Runs on GPU when available (`device="auto"`), otherwise on CPU. CPU and GPU scores
  were checked to agree on a small example.

## Tests

`tests/test_context_scorer.py`:

- decision logic with a fake LM: correction, keep, threshold blocking, phonetic penalty, OOV, errors;
- tests marked `lm` that use real GPT-2 on hand-written sentences: corrections
  (`male→mail`, `sea→see`, `see→sea`, `right→write`, `write→right`) and **original-already-correct**
  cases (`mail`, `male`, `see`, `sea`, `write`, `right`, `weather` in correct contexts), which must
  be kept. Run `pytest -m "not lm"` to skip the model.

These are functional checks on a few hand-written sentences, **not an evaluation**.
No claim is made yet about WER improvement on real ASR output.

## Known limitations

- The whole sentence must be scored once per alternative (up to `1 + max_candidates` LM passes per word).
- The candidate pool comes from all of CMUdict, so it includes rare words. GPT-2 usually ranks them low,
  but a restricted vocabulary (milestone 2) is still advisable.
- Only one target word is scored at a time. Deciding *which* words to check, and handling several
  errors in one sentence, is left to the correction pipeline.
- GPT-2 is a small, general-domain, English-only model. Scores depend on how the text is
  formatted: ASR output without punctuation or casing may be scored differently from
  well-formatted text.
