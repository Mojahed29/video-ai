"""Combine the visual and audio sub-scores into one overall 0-100 score."""
from __future__ import annotations

from . import config
from .schemas import ModalityResult


def fuse(visual: ModalityResult, audio: ModalityResult) -> tuple[float, str, str]:
    """
    Returns (overall_score, fusion_method_description, confidence_band).

    - Both assessed: weighted average (default 60% visual / 40% audio).
    - One assessed: use it alone, and say so.
    - Neither assessed: fall back to a neutral 50 with a "low" confidence band.
    """
    v_ok = visual.status == "assessed" and visual.score is not None
    a_ok = audio.status == "assessed" and audio.score is not None

    if v_ok and a_ok:
        overall = config.VISUAL_WEIGHT * visual.score + config.AUDIO_WEIGHT * audio.score
        method = (
            f"Weighted average: {config.VISUAL_WEIGHT:.0%} visual + {config.AUDIO_WEIGHT:.0%} audio"
        )
    elif v_ok:
        overall = visual.score
        method = "Visual score only (audio not assessed)"
    elif a_ok:
        overall = audio.score
        method = "Audio score only (visual not assessed)"
    else:
        overall = 50.0
        method = "Neither track could be assessed; neutral default returned"

    confidence = _confidence_band(visual, audio, v_ok, a_ok)
    return round(overall, 1), method, confidence


def _confidence_band(visual: ModalityResult, audio: ModalityResult, v_ok: bool, a_ok: bool) -> str:
    if v_ok and a_ok:
        return "high"
    if v_ok or a_ok:
        return "medium"
    return "low"
