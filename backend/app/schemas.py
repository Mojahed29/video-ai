from typing import Literal, Optional

from pydantic import BaseModel


class ModalityResult(BaseModel):
    status: Literal["assessed", "not_assessed"]
    score: Optional[float] = None  # 0-100 probability the track is fake/synthetic
    reason: Optional[str] = None   # populated when status == "not_assessed"
    model_used: Optional[str] = None
    detail: Optional[str] = None   # short human-readable extra info (e.g. "18/24 frames had a detectable face")

    # --- Explainability extras (populated only where applicable & enabled) -------
    # Visual only: heuristic frame-to-frame "jitteriness" signal, see temporal_analysis.py
    temporal_consistency: Optional[dict] = None
    # Visual only: occlusion-based saliency heatmap of the most-suspicious face/frame
    saliency: Optional[dict] = None
    # Audio only: per-segment "how synthetic does THIS part sound" timeline
    timeline: Optional[list[dict]] = None
    # Raw {label, score} pairs returned by the classifier for the whole track/clip,
    # exposed so users can sanity-check how the 0-100 score was derived from the
    # model's own labels (useful when a score looks saturated/miscalibrated).
    raw_model_output: Optional[list[dict]] = None


class AnalyzeResponse(BaseModel):
    overall_score: float
    visual_subscore: Optional[float]
    audio_subscore: Optional[float]
    visual: ModalityResult
    audio: ModalityResult
    confidence_band: Literal["low", "medium", "high"]
    fusion_method: str
    frames_analyzed: int
    duration_seconds: float
    warnings: list[str] = []


class HealthResponse(BaseModel):
    status: str
    visual_model: str
    audio_model: str
    device: str
