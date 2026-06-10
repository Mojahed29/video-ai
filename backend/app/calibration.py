"""
Honest, opt-in score calibration.

These detectors output a raw probability that the model has *no* ground-truth
guarantee for, and we deliberately do **not** fabricate a fitted calibration
curve (that would require a labelled validation set we don't ship). What we do
offer instead is *uncertainty-aware shrinkage*: when a score is derived from
very little signal — only a couple of scored frames, or under a few seconds of
audio — it is pulled toward the neutral ``50`` prior, so the reported number
reflects how much the pipeline actually had to work with.

Calibration is **disabled by default** (identity transform), which preserves
the raw model numbers exactly. Enable it with ``VIDEO_AI_CALIBRATION_ENABLED=true``.
No accuracy claim is implied in either mode — see the README disclaimer.
"""
from __future__ import annotations

from .settings import Settings

NEUTRAL_SCORE = 50.0


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
