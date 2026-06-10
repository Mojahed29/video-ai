"""Pydantic response models for the public API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ModalityStatus = Literal["assessed", "not_assessed"]
ConfidenceBand = Literal["low", "medium", "high"]


class TimelinePoint(BaseModel):
    """
    One point on a modality's explainability timeline.

    - Visual: one sampled frame (``t_start == t_end`` = the frame timestamp);
      ``note`` is ``"face"`` or ``"no face"``.
    - Audio: one scored window (``t_start``..``t_end`` seconds).

    ``score`` is the 0–100 likelihood-of-synthetic for that frame/window. These
    drive the per-frame heat strip and audio timeline in the UI; they explain
    *where in time* the model's suspicion came from, not per-pixel saliency.
    """

    t_start: float = Field(description="Window/frame start time in seconds.")
    t_end: float = Field(description="Window/frame end time in seconds.")
    score: float = Field(description="0–100 likelihood this segment is synthetic.")
    note: Optional[str] = None


class ModalityResult(BaseModel):
    """Result for a single track (visual or audio)."""

    status: ModalityStatus
    score: Optional[float] = Field(
        default=None, description="0–100 probability the track is fake/synthetic."
    )
    raw_score: Optional[float] = Field(
        default=None,
        description="Raw model probability (0–100) before temperature/calibration. "
        "Always present for the audio track so the raw signal sits beside the calibrated one.",
    )
    reason: Optional[str] = Field(
        default=None, description="Why the track was not assessed (status == not_assessed)."
    )
    model_used: Optional[str] = Field(
        default=None, description="Model ID that actually ran (primary or fallback)."
    )
    detail: Optional[str] = Field(
        default=None, description="Short human-readable note about how it was scored."
    )
    timeline: list[TimelinePoint] = Field(
        default_factory=list, description="Per-frame / per-window scores over time."
    )


class AnalyzeResponse(BaseModel):
    overall_score: float = Field(description="Fused 0–100 likelihood the video is AI-generated.")
    visual_subscore: Optional[float] = None
    audio_subscore: Optional[float] = None
    visual: ModalityResult
    audio: ModalityResult
    confidence_band: ConfidenceBand
    fusion_method: str
    calibrated: bool = Field(
        default=False, description="Whether uncertainty-aware calibration was applied."
    )
    frames_analyzed: int
    duration_seconds: float
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    visual_model: str
    audio_model: str
    device: str


class ErrorResponse(BaseModel):
    """Uniform error body returned for handled HTTP errors."""

    detail: str
