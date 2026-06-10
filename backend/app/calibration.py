"""
Honest, opt-in score calibration.

These detectors output a raw probability that the model has *no* ground-truth
guarantee for, and we deliberately do **not** fabricate a fitted calibration
curve (that would require a labelled validation set we don't ship). What we do
offer two honest, transparent transforms:

1. *Temperature scaling* (``temperature_scale``): small fine-tuned deepfake
   detectors — the audio ones especially — are badly over-confident on ordinary
   compressed/resampled real-world media and pin the score near 0 or 100. A
   temperature > 1 softens the probability in logit space toward 50%. It cannot
   make a wrong model right (a logit of "0.999 fake" is so large that even a
   strong temperature only nudges it), but it tempers borderline readings.

2. *Uncertainty-aware shrinkage* (``calibrate_visual``/``calibrate_audio``):
   when a score is derived from very little signal (a couple of scored frames,
   under a few seconds of audio) it is pulled toward the neutral ``50`` prior.
   This is **opt-in** (``VIDEO_AI_CALIBRATION_ENABLED=true``) and identity by
   default, preserving the raw numbers.

No accuracy claim is implied by either transform — see the README disclaimer.
"""
from __future__ import annotations

import math
from typing import Optional

from .settings import Settings

NEUTRAL_SCORE = 50.0


def temperature_scale(prob: Optional[float], temperature: float) -> Optional[float]:
    """
    Soften an over-confident probability in logit space.

    ``temperature == 1.0`` is a no-op. ``temperature > 1`` pulls confident
    scores toward 0.5; ``temperature < 1`` sharpens them. Returns ``None``
    unchanged so callers can short-circuit on "not assessed".
    """
    if prob is None or temperature == 1.0:
        return prob
    p = min(max(prob, 1e-6), 1.0 - 1e-6)
    logit = math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-logit / max(1e-6, temperature)))


def _shrink(score: float, confidence: float, max_shrink: float) -> float:
    """
    Pull ``score`` toward :data:`NEUTRAL_SCORE` by up to ``max_shrink`` when
    ``confidence`` (in ``[0, 1]``) is low.

    confidence == 1.0  -> score unchanged.
    confidence == 0.0  -> score moved ``max_shrink`` of the way to 50.
    """
    confidence = max(0.0, min(1.0, confidence))
    shrink = max_shrink * (1.0 - confidence)
    return score + (NEUTRAL_SCORE - score) * shrink


def calibrate_visual(score: float, frames_scored: int, settings: Settings) -> float:
    """Calibrate a 0–100 visual score given how many frames were scored."""
    if not settings.calibration_enabled:
        return score
    full = max(1, settings.calibration_full_confidence_frames)
    confidence = frames_scored / full
    return round(_shrink(score, confidence, settings.calibration_max_shrink), 1)


def calibrate_audio(score: float, audio_seconds: float, settings: Settings) -> float:
    """Calibrate a 0–100 audio score given how many seconds were scored."""
    if not settings.calibration_enabled:
        return score
    full = max(0.1, settings.calibration_full_confidence_audio_seconds)
    confidence = audio_seconds / full
    return round(_shrink(score, confidence, settings.calibration_max_shrink), 1)
