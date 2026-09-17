"""Sentence log-probability scoring with a small pretrained causal language model.

The whole sentence is scored, so tokens *after* the target word also influence
the score (they are conditioned on it). The model runs locally; no API is used.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence

import torch

from src.utils.device import resolve_device

logger = logging.getLogger(__name__)

DEFAULT_LM = "gpt2"


class LMError(RuntimeError):
    """Raised when the language model cannot be loaded or run."""


class SentenceScorer(Protocol):
    """Anything that returns one total log-probability (natural log) per sentence."""

    def score_sentences(self, sentences: Sequence[str]) -> list[float]: ...


class CausalLMScorer:
    """Scores sentences as sum of token log-probabilities under a causal LM (default: GPT-2).

    The model is loaded lazily on first use and downloaded to the Hugging Face
    cache on first run.
    """

    def __init__(self, model_name: str = DEFAULT_LM, device: str = "auto", batch_size: int = 16) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer  # deferred: heavy import

        logger.info("Loading language model '%s' on %s...", self.model_name, self.device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except Exception as exc:  # network, unknown model id, corrupted cache...
            raise LMError(f"Failed to load language model '{self.model_name}': {exc}") from exc
        if tokenizer.bos_token_id is None:
            raise LMError(f"Model '{self.model_name}' has no BOS token; cannot score the first word.")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self._tokenizer = tokenizer
        self._model = model.to(self.device).eval()

    @torch.inference_mode()
    def score_sentences(self, sentences: Sequence[str]) -> list[float]:
        if not sentences:
            return []
        self.load()
        scores: list[float] = []
        for start in range(0, len(sentences), self.batch_size):
            scores.extend(self._score_batch(sentences[start:start + self.batch_size]))
        return scores

    def _score_batch(self, sentences: Sequence[str]) -> list[float]:
        tok = self._tokenizer
        # Prepend BOS so the first real token is also scored.
        texts = [tok.bos_token + s.strip() for s in sentences]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        logits = self._model(**enc).logits.float()

        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].to(log_probs.dtype)
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) * mask
        return token_lp.sum(dim=1).tolist()
