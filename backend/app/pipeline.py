"""End-to-end orchestration: probe -> extract -> score -> fuse."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from . import video_utils
from .audio_model import score_audio
from .fusion import fuse
from .schemas import AnalyzeResponse
from .settings import settings
from .visual_model import score_frames

logger = logging.getLogger("video_ai.pipeline")


class VideoTooLargeError(Exception):
    """Upload exceeds the configured size limit."""


class VideoTooLongError(Exception):
    """Upload exceeds the configured duration limit."""


def analyze_video(upload_path: Path, declared_size_bytes: int) -> AnalyzeResponse:
    """
    Run the full detection pipeline on a saved upload.

    Raises :class:`VideoTooLargeError` / :class:`VideoTooLongError` for limit
    violations and :class:`video_utils.UnreadableVideoError` for corrupt files;
    the API layer maps these to 413 / 422 respectively.
    """
    warnings: list[str] = []

    if declared_size_bytes > settings.max_file_size_bytes:
        raise VideoTooLargeError(
            f"File is {declared_size_bytes / (1024 * 1024):.1f} MB; "
            f"the limit is {settings.max_file_size_mb} MB."
        )

    info = video_utils.probe(upload_path)  # raises UnreadableVideoError on corrupt files
    duration = info["duration"]

    if duration > settings.max_duration_seconds:
        raise VideoTooLongError(
            f"Video is {duration:.1f}s long; the limit is {settings.max_duration_seconds:.0f}s."
        )
    if 0 < duration < settings.min_duration_seconds:
        warnings.append(
            f"Video is very short ({duration:.1f}s); results may be based on very little signal."
        )

    work_dir = Path(tempfile.mkdtemp(prefix="video_ai_"))
    try:
        frames = video_utils.extract_frames(upload_path, work_dir / "frames", duration)
        if not frames:
            warnings.append("Could not extract any frames from the video.")

        audio_path = video_utils.extract_audio(upload_path, work_dir) if info["has_audio"] else None
        if info["has_audio"] and audio_path is None:
            warnings.append("Audio stream was reported but could not be extracted.")

        visual_result = score_frames(frames)
        audio_result = score_audio(audio_path)

        overall, fusion_method, confidence = fuse(visual_result, audio_result)
        calibrated = settings.calibration_enabled and (
            visual_result.raw_score is not None or audio_result.raw_score is not None
        )

        return AnalyzeResponse(
            overall_score=overall,
            visual_subscore=visual_result.score,
            audio_subscore=audio_result.score,
            visual=visual_result,
            audio=audio_result,
            confidence_band=confidence,
            fusion_method=fusion_method,
            calibrated=calibrated,
            frames_analyzed=len(frames),
            duration_seconds=round(duration, 2),
            warnings=warnings,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
