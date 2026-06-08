"""
Lightweight, model-agnostic explainability helpers.

VISUAL — occlusion-based saliency: mask coarse grid cells of the single most
suspicious detected face/frame one at a time, re-score each masked variant,
and use (baseline_score - masked_score) as that cell's "contribution" to the
fake-probability. Cells whose removal drops the fake score the most are
highlighted brightest, i.e. the regions the model relied on most. This works
with ANY image-classification pipeline (no architecture-specific gradient
hooks) at the cost of grid*grid extra forward passes — run on ONE crop, not
every frame, to stay cheap on CPU. See config.ENABLE_VISUAL_SALIENCY /
config.SALIENCY_GRID_SIZE.

AUDIO — temporal saliency: split the clip into a handful of equal,
non-overlapping windows and score each independently, producing a per-segment
"how synthetic does THIS part sound" timeline so users can see *where* in the
clip the signal is strongest rather than a single number for the whole track.
See config.ENABLE_AUDIO_TIMELINE / config.AUDIO_TIMELINE_*.

Both are heuristic, post-hoc explanations of the model's *behavior on this
input* — not a guarantee of *why* the underlying network made its decision.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Callable, Optional

import numpy as np
from PIL import Image

from . import config

logger = logging.getLogger("video_ai.explainability")

ImageScoreFn = Callable[[Image.Image], Optional[float]]
AudioScoreFn = Callable[[np.ndarray, int], Optional[float]]

_SALIENCY_INPUT_SIZE = 224
_OVERLAY_ALPHA = 0.45


def build_visual_saliency(image: Image.Image, score_fn: ImageScoreFn,
                          grid: int = config.SALIENCY_GRID_SIZE) -> Optional[dict]:
    try:
        base = image.convert("RGB").resize((_SALIENCY_INPUT_SIZE, _SALIENCY_INPUT_SIZE))
        baseline = score_fn(base)
        if baseline is None:
            return None

        width, height = base.size
        cell_w, cell_h = width // grid, height // grid
        mean_color = tuple(int(c) for c in np.asarray(base).reshape(-1, 3).mean(axis=0))

        contributions = np.zeros((grid, grid), dtype=np.float32)
        for row in range(grid):
            for col in range(grid):
                box = (col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h)
                masked = base.copy()
                masked.paste(Image.new("RGB", (box[2] - box[0], box[3] - box[1]), mean_color), box)
                masked_score = score_fn(masked)
                if masked_score is not None:
                    contributions[row, col] = max(0.0, baseline - masked_score)

        if contributions.max() <= 0:
            return None

        heat = contributions / contributions.max()
        heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((width, height), Image.BILINEAR)
        overlay = _tint_and_blend(base, heat_img)

        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "method": f"occlusion saliency ({grid}x{grid} grid)",
            "baseline_fake_probability": round(baseline * 100, 1),
            "image_base64": f"data:image/png;base64,{encoded}",
            "note": (
                "Heuristic, post-hoc explanation: brighter/red regions are where masking that "
                "part of the image reduced the model's fake-probability the most for THIS input "
                "— i.e. the regions it relied on most. Not an architecture-level explanation."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Visual saliency generation failed: %s", exc)
        return None


def _tint_and_blend(base: Image.Image, heat_gray: Image.Image, alpha: float = _OVERLAY_ALPHA) -> Image.Image:
    """Tint hot regions red and alpha-blend the tint over the original crop."""
    heat = np.asarray(heat_gray, dtype=np.float32)[..., None] / 255.0
    base_arr = np.asarray(base, dtype=np.float32)

    red_layer = np.zeros_like(base_arr)
    red_layer[..., 0] = 255.0

    blended = base_arr * (1 - alpha * heat) + red_layer * (alpha * heat)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def build_audio_timeline(samples: np.ndarray, sample_rate: int, score_fn: AudioScoreFn,
                         max_segments: int = config.AUDIO_TIMELINE_MAX_SEGMENTS) -> Optional[list[dict]]:
    duration = len(samples) / sample_rate
    if duration < config.AUDIO_TIMELINE_MIN_DURATION:
        return None

    segment_count = max(2, min(max_segments, int(duration // config.AUDIO_TIMELINE_MIN_SEGMENT_SECONDS)))
    segment_len = len(samples) // segment_count

    timeline = []
    for i in range(segment_count):
        start = i * segment_len
        end = len(samples) if i == segment_count - 1 else (i + 1) * segment_len
        chunk = samples[start:end]
        if chunk.size == 0:
            continue
        try:
            score = score_fn(chunk, sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Audio timeline segment scoring failed: %s", exc)
            score = None
        timeline.append({
            "start_seconds": round(start / sample_rate, 2),
            "end_seconds": round(end / sample_rate, 2),
            "score": round(score * 100, 1) if score is not None else None,
        })

    return timeline if any(seg["score"] is not None for seg in timeline) else None
