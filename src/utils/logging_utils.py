"""Logging setup for command-line entry points."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging. Library modules only create loggers; entry points call this."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
        force=True,
    )
