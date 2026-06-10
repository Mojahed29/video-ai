"""
Shared label -> P(fake) mapping for both the visual and audio classifiers.

A ``transformers`` classification pipeline returns a list of
``{"label": str, "score": float}`` entries. Real-vs-fake models in the wild use
inconsistent label wording and casing (``"Fake"``/``"Real"``,
``"deepfake"``/``"pristine"``, ``"spoof"``/``"bonafide"``,
``"synthetic"``/``"authentic"``, ...). Rather than hard-code one model's labels,
we match by substring so a swapped-in model keeps working without code changes.

Keeping this in one place means the visual and audio tracks stay consistent and
are covered by a single set of unit tests (see ``tests/test_labels.py``).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

# Substrings that identify the "fake / synthetic / generated" class.
FAKE_KEYWORDS: tuple[str, ...] = (
    "fake",
    "deepfake",
    "spoof",
    "synthetic",
    "generated",
    "ai",
    "tts",
)

# Substrings that identify the "real / authentic / human" class.
REAL_KEYWORDS: tuple[str, ...] = (
    "real",
    "authentic",
    "genuine",
    "pristine",
    "bonafide",
    "bona-fide",
    "human",
)


def _matches(label: str, keywords: Iterable[str]) -> bool:
    return any(keyword in label for keyword in keywords)


def fake_probability(pipe_output: Iterable[Mapping[str, object]]) -> Optional[float]:
    """
    Map a classification pipeline output to ``P(fake)`` in ``[0, 1]``.

    Returns ``None`` when neither a "fake" nor a "real" label can be identified,
    so callers can report an unrecognized label set rather than guess.
    """
    fake_score: Optional[float] = None
    real_score: Optional[float] = None

    for entry in pipe_output:
        label = str(entry.get("label", "")).lower()
        try:
            score = float(entry.get("score", 0.0))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue

        # Check "real" first: keywords like "ai" must not swallow "real".
        if _matches(label, REAL_KEYWORDS):
            real_score = score if real_score is None else max(real_score, score)
        elif _matches(label, FAKE_KEYWORDS):
            fake_score = score if fake_score is None else max(fake_score, score)

    if fake_score is not None:
        return fake_score
    if real_score is not None:
        return 1.0 - real_score
    return None
