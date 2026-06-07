"""End-to-end orchestration: save upload -> probe -> extract -> score -> fuse."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from . import config, video_utils
from .audio_model import score_audio
from .fusion import fuse
from .schemas import AnalyzeResponse, ModalityResult
from .visual_model import score_frames

logger = logging.getLogger("video_ai.pipeline")


class VideoTooLargeError(Exception):
    pass


class VideoTooLongError(Exception):
    pass


def analyze_video(upload_path: Path, declared_size_bytes: int) -> AnalyzeResponse:
    warnings: list[str] = []

    if declared_size_bytes > config.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise VideoTooLargeError(
            f"File is {declared_size_bytes / (1024 * 1024):.1f} MB; the limit is {config.MAX_FILE_SIZE_MB} MB."
        )

    info = video_utils.probe(upload_path)  # raises UnreadableVideoError on corrupt files
    duration = info["duration"]

    if duration > config.MAX_DURATION_SECONDS:
        raise VideoTooLongError(
            f"Video is {duration:.1f}s long; the limit is {config.MAX_DURATION_SECONDS}s."
        )
    if 0 < duration < config.MIN_DURATION_SECONDS:
        warnings.append(
            f"Video is very short ({duration:.1f}s); results may be based on very little signal."
        )

    work_dir = Path(tempfile.mkdtemp(prefix="video_ai_"))
    try:
        frame_paths = video_utils.extract_frames(upload_path, work_dir / "frames", duration)
        if not frame_paths:
            warnings.append("Could not extract any frames from the video.")

        audio_path = video_utils.extract_audio(upload_path, work_dir) if info["has_audio"] else None
        if info["has_audio"] and audio_path is None:
            warnings.append("Audio stream was reported but could not be extracted.")

        visual_result = score_frames(frame_paths)
        audio_result = score_audio(audio_path)

        overall, fusion_method, confidence = fuse(visual_result, audio_result)

        return AnalyzeResponse(
            overall_score=overall,
            visual_subscore=visual_result.score,
            audio_subscore=audio_result.score,
            visual=visual_result,
            audio=audio_result,
            confidence_band=confidence,
            fusion_method=fusion_method,
            frames_analyzed=len(frame_paths),
            duration_seconds=round(duration, 2),
            warnings=warnings,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
