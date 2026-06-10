"""Combine the visual and audio sub-scores into one overall 0–100 score."""
from __future__ import annotations

from .schemas import ConfidenceBand, ModalityResult
from .settings import settings


def fuse(visual: ModalityResult, audio: ModalityResult) -> tuple[float, str, ConfidenceBand]:
    """
    Returns ``(overall_score, fusion_method_description, confidence_band)``.

    - Both assessed: weighted average (default 60% visual / 40% audio).
    - One assessed: use it alone, and say so.
    - Neither assessed: fall back to a neutral 50 with a "low" confidence band.
    """
    v_ok = visual.status == "assessed" and visual.score is not None
    a_ok = audio.status == "assessed" and audio.score is not None

    if v_ok and a_ok:
        # Normalise weights so they always behave as a proper weighted average.
        total = settings.visual_weight + settings.audio_weight
        v_w = settings.visual_weight / total
        a_w = settings.audio_weight / total
        overall = v_w * visual.score + a_w * audio.score
        method = f"Weighted average: {v_w:.0%} visual + {a_w:.0%} audio"
    elif v_ok:
        overall = visual.score
        method = "Visual score only (audio not assessed)"
    elif a_ok:
        overall = audio.score
        method = "Audio score only (visual not assessed)"
    else:
        overall = 50.0
        method = "Neither track could be assessed; neutral default returned"

    return round(overall, 1), method, _confidence_band(v_ok, a_ok)


def _confidence_band(v_ok: bool, a_ok: bool) -> ConfidenceBand:
    if v_ok and a_ok:
        return "high"
    if v_ok or a_ok:
        return "medium"
    return "low"
