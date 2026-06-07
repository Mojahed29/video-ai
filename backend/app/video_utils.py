"""ffmpeg/ffprobe helpers: probing, frame sampling and audio extraction."""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger("video_ai.video")


class UnreadableVideoError(Exception):
    """Raised when ffprobe cannot read the uploaded file (corrupt/unsupported)."""


def probe(path: Path) -> dict:
    """Return duration (seconds) and whether the file has a usable audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_type",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise UnreadableVideoError(f"ffprobe could not run: {exc}") from exc

    if result.returncode != 0 or not result.stdout.strip():
        raise UnreadableVideoError(f"ffprobe failed to read file: {result.stderr.strip()[:300]}")

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

    return {"duration": duration, "has_video": has_video, "has_audio": has_audio}


def extract_audio(path: Path, out_dir: Path) -> Optional[Path]:
    """Extract mono PCM WAV at the model's expected sample rate. Returns None on failure."""
    out_path = out_dir / "audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-vn",
        "-ac", "1",
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-f", "wav",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        logger.warning("Audio extraction timed out for %s", path)
        return None

    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        logger.info("No audio track extracted from %s (%s)", path, result.stderr.strip()[:200])
        return None
    return out_path


def extract_frames(path: Path, out_dir: Path, duration: float, num_frames: int = config.NUM_FRAMES) -> list[Path]:
    """Extract `num_frames` evenly spaced JPEG frames using ffmpeg `select` filter timestamps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, num_frames)

    if duration <= 0:
        # Fall back to a fixed fps sample if duration metadata is missing.
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-vf", f"fps=1",
            "-vframes", str(n),
            str(out_dir / "frame_%03d.jpg"),
        ]
    else:
        # Evenly spaced timestamps across (0, duration), avoiding the very first/last instants.
        timestamps = [duration * (i + 0.5) / n for i in range(n)]
        frames: list[Path] = []
        for idx, ts in enumerate(timestamps):
            frame_path = out_dir / f"frame_{idx:03d}.jpg"
            cmd_single = [
                "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(path),
                "-frames:v", "1", "-q:v", "2",
                str(frame_path),
            ]
            try:
                subprocess.run(cmd_single, capture_output=True, text=True, timeout=30)
            except subprocess.TimeoutExpired:
                continue
            if frame_path.exists() and frame_path.stat().st_size > 0:
                frames.append(frame_path)
        return frames

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return []
    return sorted(out_dir.glob("frame_*.jpg"))
