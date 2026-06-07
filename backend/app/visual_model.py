"""
Visual track scoring: face detection (MTCNN) + binary real/fake image
classification (ViT-based, see config.VISUAL_MODEL_ID).

If no face is found in a frame, the full frame is scored instead, so the
model still has a chance to flag fully-synthetic (non-deepfake) clips.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from . import model_runtime
from .schemas import ModalityResult

logger = logging.getLogger("video_ai.visual")


def _fake_probability(pipe_output: list[dict]) -> Optional[float]:
    """
    Map a `transformers` image-classification output (list of {label, score})
    to P(fake) in [0, 1]. Labels are matched by substring so this keeps working
    if a swapped-in model uses different label casing/wording (e.g. "Fake"/"Real",
    "deepfake"/"real", "synthetic"/"authentic").
    """
    fake_score = None
    real_score = None
    for entry in pipe_output:
        label = str(entry.get("label", "")).lower()
        score = float(entry.get("score", 0.0))
        if any(k in label for k in ("fake", "deepfake", "synthetic", "ai", "generated")):
            fake_score = score if fake_score is None else max(fake_score, score)
        elif any(k in label for k in ("real", "authentic", "genuine", "pristine")):
            real_score = score if real_score is None else max(real_score, score)

    if fake_score is not None:
        return fake_score
    if real_score is not None:
        return 1.0 - real_score
    return None


def score_frames(frame_paths: list[Path]) -> ModalityResult:
    if not frame_paths:
        return ModalityResult(status="not_assessed", reason="No frames could be extracted from the video.")

    try:
        loaded = model_runtime.get_visual_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.error("Visual model failed to load: %s", exc)
        return ModalityResult(status="not_assessed", reason=f"Visual model failed to load: {exc}")

    detector = model_runtime.get_face_detector()

    frame_scores: list[float] = []
    frames_with_face = 0

    for frame_path in frame_paths:
        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open frame %s: %s", frame_path, exc)
            continue

        crops = _detect_face_crops(detector, image)
        if crops:
            frames_with_face += 1
            targets = crops
        else:
            targets = [image]

        per_frame_face_scores = []
        for target in targets:
            try:
                output = loaded.pipe(target, top_k=None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Visual classifier failed on a frame: %s", exc)
                continue
            prob = _fake_probability(output)
            if prob is not None:
                per_frame_face_scores.append(prob)

        if per_frame_face_scores:
            # Worst case (most-fake-looking face/frame) drives the per-frame score.
            frame_scores.append(max(per_frame_face_scores))

    if not frame_scores:
        return ModalityResult(
            status="not_assessed",
            reason="The visual classifier produced no usable predictions for any sampled frame.",
            model_used=loaded.model_id,
        )

    avg_prob = sum(frame_scores) / len(frame_scores)
    detail = (
        f"{frames_with_face}/{len(frame_paths)} sampled frames had a detectable face "
        f"({len(frame_scores)} frames scored)."
        if frames_with_face
        else f"No face detected in any of the {len(frame_paths)} sampled frames; "
             f"scored full frames instead ({len(frame_scores)} frames scored)."
    )

    return ModalityResult(
        status="assessed",
        score=round(avg_prob * 100, 1),
        model_used=loaded.model_id,
        detail=detail,
    )


def _detect_face_crops(detector, image: Image.Image) -> list[Image.Image]:
    if detector is None:
        return []
    try:
        boxes, _probs = detector.detect(image)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Face detection failed on a frame: %s", exc)
        return []

    if boxes is None:
        return []

    crops = []
    width, height = image.size
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        crops.append(image.crop((x1, y1, x2, y2)))
    return crops
