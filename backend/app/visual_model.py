"""
Visual track scoring: face detection (MTCNN) + binary real/fake image
classification (ViT-based, see ``settings.visual_model_id``).

For each sampled frame we detect faces and score every face crop; the most
fake-looking crop drives that frame's score. If no face is found, the full
frame is scored instead, so the model can still flag fully-synthetic (non-
deepfake) clips. The overall visual sub-score is the mean of the per-frame
scores; the per-frame scores are also returned as a timeline for the UI.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from PIL import Image

from . import calibration, model_runtime
from .labels import fake_probability
from .schemas import ModalityResult, TimelinePoint
from .settings import settings
from .video_utils import ExtractedFrame

logger = logging.getLogger("video_ai.visual")

_MIN_FACE_PX = 20  # ignore face boxes smaller than this on either side


def score_frames(frames: list[ExtractedFrame]) -> ModalityResult:
    """Score sampled frames and return a 0–100 visual likelihood + timeline."""
    if not frames:
        return ModalityResult(
            status="not_assessed", reason="No frames could be extracted from the video."
        )

    try:
        loaded = model_runtime.get_visual_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.error("Visual model failed to load: %s", exc)
        return ModalityResult(status="not_assessed", reason=f"Visual model failed to load: {exc}")

    detector = model_runtime.get_face_detector()
    temperature = settings.visual_calibration_temperature

    timeline: list[TimelinePoint] = []
    frame_scores: list[float] = []
    frames_with_face = 0

    for frame in frames:
        try:
            image = Image.open(frame.path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open frame %s: %s", frame.path, exc)
            continue

        crops = _detect_face_crops(detector, image)
        has_face = bool(crops)
        if has_face:
            frames_with_face += 1
        targets = crops if has_face else [image]

        per_target_scores: list[float] = []
        for target in targets:
            try:
                output = loaded.pipe(target, top_k=None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Visual classifier failed on a frame: %s", exc)
                continue
            prob = fake_probability(output)
            if prob is not None:
                per_target_scores.append(prob)

        if per_target_scores:
            # Worst case (most fake-looking face/frame) drives the per-frame score.
            frame_prob = max(per_target_scores)
            frame_scores.append(frame_prob)
            soft = calibration.temperature_scale(frame_prob, temperature)
            timeline.append(
                TimelinePoint(
                    t_start=frame.timestamp,
                    t_end=frame.timestamp,
                    score=round(soft * 100, 1),
                    note="face" if has_face else "no face",
                )
            )

    if not frame_scores:
        return ModalityResult(
            status="not_assessed",
            reason="The visual classifier produced no usable predictions for any sampled frame.",
            model_used=loaded.model_id,
        )

    raw_prob = sum(frame_scores) / len(frame_scores)
    raw = round(raw_prob * 100, 1)
    soft = round(calibration.temperature_scale(raw_prob, temperature) * 100, 1)
    score = calibration.calibrate_visual(soft, frames_scored=len(frame_scores), settings=settings)

    detail = (
        f"{frames_with_face}/{len(frames)} sampled frames had a detectable face "
        f"({len(frame_scores)} frames scored)."
        if frames_with_face
        else f"No face detected in any of the {len(frames)} sampled frames; "
             f"scored full frames instead ({len(frame_scores)} frames scored)."
    )

    return ModalityResult(
        status="assessed",
        score=score,
        raw_score=raw if score != raw else None,
        model_used=loaded.model_id,
        detail=detail,
        timeline=timeline,
    )


def _detect_face_crops(detector: Optional[Any], image: Image.Image) -> list[Image.Image]:
    if detector is None:
        return []
    try:
        boxes, _probs = detector.detect(image)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Face detection failed on a frame: %s", exc)
        return []

    if boxes is None:
        return []

    crops: list[Image.Image] = []
    width, height = image.size
    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < _MIN_FACE_PX or y2 - y1 < _MIN_FACE_PX:
            continue
        crops.append(image.crop((x1, y1, x2, y2)))
    return crops
