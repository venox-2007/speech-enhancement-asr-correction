"""Pronunciation lookup backed by the CMU Pronouncing Dictionary (ARPAbet).

Stress markers are removed (``EY1`` -> ``EY``), because ASR confusions rarely
depend on lexical stress.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Iterable, Iterator, Mapping

logger = logging.getLogger(__name__)

Pronunciation = tuple[str, ...]

_WORD_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")
_EDGE_PUNCT = "\"'.,!?;:()[]{}<>-–—…"


def normalize_word(token: str) -> str:
    """Lowercase an ASR token and strip surrounding punctuation (``"Mail,"`` -> ``"mail"``)."""
    return token.strip().lower().strip(_EDGE_PUNCT)


def is_plain_word(word: str) -> bool:
    """True for alphabetic words, optionally with one internal apostrophe (``don't``)."""
    return bool(_WORD_RE.match(word))


def strip_stress(phones: Iterable[str]) -> Pronunciation:
    return tuple(p.rstrip("012") for p in phones)


@lru_cache(maxsize=1)
def _load_cmudict() -> dict[str, list[list[str]]]:
    import cmudict  # deferred: parsing takes ~1s

    logger.info("Loading CMU Pronouncing Dictionary...")
    return cmudict.dict()


class PronunciationDict:
    """Word -> list of stress-free pronunciations."""

    def __init__(self, entries: Mapping[str, Iterable[Iterable[str]]] | None = None) -> None:
        raw = _load_cmudict() if entries is None else entries
        self._entries: dict[str, list[Pronunciation]] = {}
        for word, prons in raw.items():
            variants = list(dict.fromkeys(strip_stress(p) for p in prons))  # dedupe, keep order
            if variants:
                self._entries[word.lower()] = variants

    def __contains__(self, word: str) -> bool:
        return normalize_word(word) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, word: str) -> list[Pronunciation]:
        """Pronunciations for ``word``; an empty list if the word is not in the dictionary."""
        return list(self._entries.get(normalize_word(word), []))

    def items(self) -> Iterator[tuple[str, list[Pronunciation]]]:
        return iter(self._entries.items())


@lru_cache(maxsize=1)
def default_pronunciation_dict() -> PronunciationDict:
    """Shared CMUdict-backed instance (loaded once per process)."""
    return PronunciationDict()
