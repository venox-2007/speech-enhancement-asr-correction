# Correction Pipeline and Evaluation

## Overview

The correction pipeline integrates phonetic candidate generation and
context-aware scoring into a single `CorrectionPipeline` class that
processes an ASR transcript word by word and produces a corrected
transcript with full per-word decision records.

The evaluation module (`correction_eval`) compares the corrected
transcript against a reference to measure WER improvement and classify
each word-level change.

## Architecture

```text
ASR Hypothesis
      │
      ▼
CorrectionPipeline.correct(hypothesis)
      │
      │  for each word in the hypothesis (independently):
      │
      ├── is_plain_word? ──no──> skip (decision = None)
      │        │
      │       yes
      │        ▼
      ├── ContextAwareScorer.score(sentence, word_index)
      │        │
      │        ├── PhoneticCandidateGenerator.generate(token)
      │        │       └── CMUdict + RapidFuzz shortlist + weighted distance rerank
      │        │
      │        └── CausalLMScorer (GPT-2) scores each candidate in context
      │                └── score = log P_LM(sentence) + phonetic_weight * log(similarity)
      │                └── softmax → probability → threshold decision
      │
      ▼
CorrectionResult
├── original: str
├── corrected: str
└── word_corrections: [WordCorrection, ...]
         ├── position, original_token, corrected_token
         ├── was_changed: bool
         └── decision: ScoringDecision (reason, confidence, alternatives)
```

## Independent Scoring

Each word is scored against the **original** sentence context, not a
partially-corrected sentence.  This is a deliberate design choice:

- **Reproducibility**: the result does not depend on word processing order.
- **Analyzability**: each correction decision can be evaluated in isolation.
- **Simplicity**: no cascading-error risk where one bad correction
  degrades context for subsequent words.

The trade-off is that corrections cannot benefit from earlier fixes.
For example, if two adjacent words are both wrong, correcting the first
might improve the LM context for the second — but independent scoring
misses this.  A future enhancement could add a sequential mode.

## Evaluation Methodology

Given a reference transcript, the evaluation module:

1. Computes **WER before** correction: `compute_wer(reference, hypothesis)`.
2. Computes **WER after** correction: `compute_wer(reference, corrected)`.
3. Uses JiWER word alignment to map each hypothesis word to the
   corresponding reference word.
4. Classifies each word into one of five outcome categories.

### Outcome Categories

| Category | ASR word | Pipeline action | Result |
|----------|----------|-----------------|--------|
| `successful_correction` | Wrong | Changed | Matches reference |
| `overcorrection` | Correct | Changed | Now wrong |
| `failed_correction` | Wrong | Changed | Still wrong (different word) |
| `unchanged_correct` | Correct | Kept | Still correct |
| `unchanged_error` | Wrong | Kept | Still wrong |

### Summary Metrics

- **WER delta** = WER_after − WER_before (negative = improvement)
- **Correction precision** = successful / (successful + overcorrections + failed)
  — measures how often a pipeline change was actually helpful.
- **Overcorrection rate** = overcorrections / total_changes
  — measures how often the pipeline damaged correct words.

## Scope and Limitations

The correction pipeline only handles **single-word substitution** errors.
It cannot:

- Fix insertion errors (extra words in the ASR output).
- Fix deletion errors (missing words).
- Handle multi-word confusions (e.g. "ice cream" ↔ "I scream").

These are structural limitations of the word-by-word approach.
Insertion and deletion errors require different correction strategies.

### Other known limitations

- **LM domain mismatch**: GPT-2 is trained on written text, but ASR
  output is typically unpunctuated spoken language.  This affects scoring
  quality, especially for informal or domain-specific speech.
- **Per-candidate LM pass**: each candidate requires a full sentence
  evaluation, making the pipeline slow for long sentences with many
  candidate-bearing words.
- **CMUdict coverage**: words not in CMUdict (~126k entries) cannot be
  corrected.  Proper nouns, technical terms, and neologisms are typically
  missing.

## Usage

### Python API

```python
from src.correction.pipeline import CorrectionPipeline
from src.evaluation.correction_eval import evaluate_correction

pipeline = CorrectionPipeline()
correction = pipeline.correct("I can sea the ship from here")

print(correction.corrected)      # "I can see the ship from here"
print(correction.num_changed)    # 1

# Evaluate against a reference
result = evaluate_correction(
    reference="I can see the ship from here",
    hypothesis="I can sea the ship from here",
    correction=correction,
)
print(f"WER: {result.wer_before.wer:.1%} -> {result.wer_after.wer:.1%}")
print(f"Successful corrections: {result.successful_corrections}")
print(f"Overcorrections: {result.overcorrections}")
```

### CLI Demo

```bash
python -m src.correction.evaluate_corrections
python -m src.correction.evaluate_corrections --threshold 0.5 --verbose
python -m src.correction.evaluate_corrections --json-out demo_results.json
```

The CLI runs on **hand-crafted** reference/hypothesis pairs for
demonstration only.  These are not experimental results.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `auto` | Compute device (`auto`, `cpu`, `cuda`) |
| `--threshold` | `0.7` | Minimum probability for applying a correction |
| `--phonetic-weight` | `1.0` | Weight for phonetic similarity in scoring |
| `--json-out PATH` | — | Write full results to a JSON file |
| `-v, --verbose` | — | Enable debug logging |

## Module Layout

| Module | Description |
|--------|-------------|
| `src/correction/pipeline.py` | `CorrectionPipeline`, `CorrectionResult`, `WordCorrection` |
| `src/evaluation/correction_eval.py` | `evaluate_correction`, `CorrectionEvalResult`, `CorrectionOutcome` |
| `src/correction/evaluate_corrections.py` | CLI demo entry point |
| `tests/test_correction_pipeline.py` | Unit and integration tests |

## Experimental Status

No experimental results have been produced yet.  The pipeline and
evaluation code are ready, but systematic evaluation on a benchmark
dataset (e.g. VoiceBank+DEMAND with Whisper transcripts) has not been
performed.
