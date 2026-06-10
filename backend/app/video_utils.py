"""ffmpeg/ffprobe helpers: probing, frame sampling and audio extraction."""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict

from .settings import settings

logger = logging.getLogger("video_ai.video")

_FFPROBE_TIMEOUT = 30
_FFMPEG_TIMEOUT = 120
_FRAME_TIMEOUT = 30


class UnreadableVideoError(Exception):
    """Raised when ffprobe cannot read the uploaded file (corrupt/unsupported)."""


class ProbeResult(TypedDict):
    duration: float
    has_video: bool
    has_audio: bool


@dataclass(frozen=True)
class ExtractedFrame:
    """A sampled video frame and the timestamp (seconds) it was taken from."""

    timestamp: float
    path: Path


def probe(path: Path) -> ProbeResult:
    """
    Return duration (seconds) and whether the file has usable video/audio streams.

    Raises :class:`UnreadableVideoError` for corrupt or unsupported files so the
    API can return a clean ``422`` instead of crashing.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_FFPROBE_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise UnreadableVideoError("ffprobe timed out reading the file.") from exc
    except FileNotFoundError as exc:
        raise UnreadableVideoError(
            "ffprobe is not installed or not on PATH. Install ffmpeg to analyze videos."
        ) from exc

    if result.returncode != 0 or not result.stdout.strip():
        raise UnreadableVideoError(result.stderr.strip()[:300] or "ffprobe could not read the file.")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise UnreadableVideoError(f"ffprobe returned unparseable output: {exc}") from exc

    fmt = data.get("format", {})
    streams = data.get("streams", [])
    try:
        duration = float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    if not has_video and duration <= 0:
        raise UnreadableVideoError("File has no readable video stream and no duration metadata.")

    return ProbeResult(duration=duration, has_video=has_video, has_audio=has_audio)


def extract_audio(path: Path, out_dir: Path) -> Optional[Path]:
    """
    Extract mono PCM WAV at the model's expected sample rate.

    Returns the WAV path, or ``None`` if extraction fails / yields no audio.
    """
    out_path = out_dir / "audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(settings.audio_sample_rate),
        "-f", "wav",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        logger.warning("Audio extraction timed out for %s", path)
        return None

    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        logger.info("No audio track extracted from %s (%s)", path, result.stderr.strip()[:200])
        return None
    return out_path


def extract_frames(
    path: Path,
    out_dir: Path,
    duration: float,
    num_frames: Optional[int] = None,
) -> list[ExtractedFrame]:
    """
    Extract ``num_frames`` evenly spaced JPEG frames, each tagged with its
    source timestamp (used for the visual explainability timeline).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, num_frames if num_frames is not None else settings.num_frames)

    if duration <= 0:
        # Duration metadata missing: fall back to a fixed 1 fps sample.
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-vf", "fps=1",
            "-vframes", str(n),
            str(out_dir / "frame_%03d.jpg"),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT)
        except subprocess.TimeoutExpired:
            return []
        # 1 fps => frame i is at ~i seconds.
        return [
            ExtractedFrame(timestamp=float(idx), path=fp)
            for idx, fp in enumerate(sorted(out_dir.glob("frame_*.jpg")))
        ]

    # Evenly spaced timestamps across (0, duration), avoiding the very first/last instants.
    timestamps = [duration * (i + 0.5) / n for i in range(n)]
    frames: list[ExtractedFrame] = []
    for idx, ts in enumerate(timestamps):
        frame_path = out_dir / f"frame_{idx:03d}.jpg"
        cmd_single = [
            "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(path),
            "-frames:v", "1", "-q:v", "2",
            str(frame_path),
        ]
        try:
            subprocess.run(cmd_single, capture_output=True, text=True, timeout=_FRAME_TIMEOUT)
        except subprocess.TimeoutExpired:
            continue
        if frame_path.exists() and frame_path.stat().st_size > 0:
            frames.append(ExtractedFrame(timestamp=round(ts, 2), path=frame_path))
    return frames
