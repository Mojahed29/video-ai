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

from . import config, explainability, model_runtime, temporal_analysis
from .schemas import ModalityResult

logger = logging.getLogger("video_ai.visual")


def fake_probability(pipe_output: list[dict]) -> Optional[float]:
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

    def score_image(image: Image.Image) -> Optional[float]:
        try:
            return fake_probability(loaded.pipe(image, top_k=None))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Visual classifier failed on an image: %s", exc)
            return None

    frame_scores: list[float] = []
    ordered_targets: list[Image.Image] = []  # one representative crop/frame per scored timestamp, in order
    most_suspicious: Optional[tuple[Image.Image, float]] = None
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

        scored_targets = [(t, p) for t in targets if (p := score_image(t)) is not None]
        if not scored_targets:
            continue

        # Worst case (most-fake-looking face/frame) drives this timestamp's score
        # and is what we track for temporal-consistency and saliency purposes.
        target_img, target_prob = max(scored_targets, key=lambda pair: pair[1])
        frame_scores.append(target_prob)
        ordered_targets.append(target_img)
        if most_suspicious is None or target_prob > most_suspicious[1]:
            most_suspicious = (target_img, target_prob)

    if not frame_scores:
        return ModalityResult(
            status="not_assessed",
            reason="The visual classifier produced no usable predictions for any sampled frame.",
            model_used=loaded.model_id,
        )

    classifier_avg = sum(frame_scores) / len(frame_scores)
    detail = (
        f"{frames_with_face}/{len(frame_paths)} sampled frames had a detectable face "
        f"({len(frame_scores)} frames scored)."
        if frames_with_face
        else f"No face detected in any of the {len(frame_paths)} sampled frames; "
             f"scored full frames instead ({len(frame_scores)} frames scored)."
    )

    # --- Temporal consistency: blend a heuristic frame-to-frame "jitteriness"
    # signal into the per-frame classifier average (see temporal_analysis.py).
    temporal = None
    final_prob = classifier_avg
    if config.TEMPORAL_SIGNAL_WEIGHT > 0:
        temporal = temporal_analysis.compute_temporal_signal(ordered_targets)
        if temporal is not None:
            w = config.TEMPORAL_SIGNAL_WEIGHT
            final_prob = (1 - w) * classifier_avg + w * temporal["inconsistency_score"]
            detail += (
                f" Frame-to-frame consistency nudged the score by "
                f"{(final_prob - classifier_avg) * 100:+.1f} points (weight {w:.0%})."
            )

    # --- Explainability: occlusion-based saliency heatmap of the single
    # most-suspicious detected face/frame (see explainability.py).
    saliency = None
    if config.ENABLE_VISUAL_SALIENCY and most_suspicious is not None:
        saliency = explainability.build_visual_saliency(most_suspicious[0], score_image)

    return ModalityResult(
        status="assessed",
        score=round(final_prob * 100, 1),
        model_used=loaded.model_id,
        detail=detail,
        temporal_consistency=temporal,
        saliency=saliency,
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
