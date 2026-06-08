"""
Audio track scoring: synthetic-speech detection (Wav2Vec2-based binary
real/fake classifier, see config.AUDIO_MODEL_ID).

Silence / music-only / missing-audio tracks are detected with a cheap
RMS-energy check before invoking the model, and reported as "not assessed"
rather than fed to a speech model that would produce meaningless output.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from . import config, explainability, model_runtime
from .schemas import ModalityResult

logger = logging.getLogger("video_ai.audio")


def _normalize_label(label: str) -> str:
    """
    Reduce a label to lowercase alphanumerics only, so variants like
    "bona-fide", "Bona Fide", "LABEL_0: spoof" all collapse to a comparable
    form ("bonafide", "bonafide", "label0spoof") before substring matching.
    Without this, a hyphenated/spaced "real" label (e.g. "bona-fide") silently
    fails the "bonafide" check, falls through to the fake-side keywords, and
    the track ends up scored as ~100% fake regardless of what the model
    actually predicted — which is the root cause of the "always 100%" bug.
    """
    return "".join(ch for ch in label.lower() if ch.isalnum())


def fake_probability(pipe_output: list[dict]) -> Optional[float]:
    """Same label-matching strategy as the visual model (see visual_model.fake_probability),
    plus normalization so punctuation/whitespace variants in label names don't break matching."""
    fake_score = None
    real_score = None
    for entry in pipe_output:
        label = _normalize_label(str(entry.get("label", "")))
        score = float(entry.get("score", 0.0))
        if any(k in label for k in ("fake", "spoof", "synthetic", "aigenerated", "generated", "tts", "deepfake")):
            fake_score = score if fake_score is None else max(fake_score, score)
        elif any(k in label for k in ("real", "bonafide", "bona", "authentic", "genuine", "human")):
            real_score = score if real_score is None else max(real_score, score)

    if fake_score is not None:
        return fake_score
    if real_score is not None:
        return 1.0 - real_score
    return None


def score_audio(audio_path: Optional[Path]) -> ModalityResult:
    if audio_path is None:
        return ModalityResult(status="not_assessed", reason="The video has no audio track.")

    try:
        samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        return ModalityResult(status="not_assessed", reason=f"Could not read extracted audio: {exc}")

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    if samples.size == 0:
        return ModalityResult(status="not_assessed", reason="Extracted audio track is empty.")

    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms < config.SILENCE_RMS_THRESHOLD:
        return ModalityResult(status="not_assessed", reason="Audio track is silent (or near-silent); no speech to assess.")

    try:
        loaded = model_runtime.get_audio_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.error("Audio model failed to load: %s", exc)
        return ModalityResult(status="not_assessed", reason=f"Audio model failed to load: {exc}")

    def score_chunk(chunk: np.ndarray, sr: int) -> Optional[float]:
        try:
            return fake_probability(loaded.pipe({"raw": chunk, "sampling_rate": sr}, top_k=None))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audio classifier failed on a chunk: %s", exc)
            return None

    try:
        raw_output = loaded.pipe({"raw": samples, "sampling_rate": sample_rate}, top_k=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Audio classifier failed on the full clip: %s", exc)
        raw_output = None

    prob = fake_probability(raw_output) if raw_output is not None else None
    if prob is None:
        return ModalityResult(
            status="not_assessed",
            reason="Audio classifier returned an unrecognized label set or failed on this clip.",
            model_used=loaded.model_id,
            raw_model_output=raw_output,
        )

    # --- Explainability: per-segment "how synthetic does THIS part sound" timeline.
    timeline = None
    if config.ENABLE_AUDIO_TIMELINE:
        timeline = explainability.build_audio_timeline(samples, sample_rate, score_chunk)

    detail = f"Scored {samples.size / sample_rate:.1f}s of audio at {sample_rate} Hz (RMS={rms:.4f})."
    if prob >= 0.97 or prob <= 0.03:
        detail += (
            " Note: this score is near the extreme of the scale — small fine-tuned audio "
            "classifiers can be poorly calibrated on compressed/resampled real-world audio "
            "that differs from their training distribution, so a near-0 or near-100 reading "
            "deserves extra skepticism (see raw_model_output for the model's own labels/scores)."
        )

    return ModalityResult(
        status="assessed",
        score=round(prob * 100, 1),
        model_used=loaded.model_id,
        detail=detail,
        timeline=timeline,
        raw_model_output=raw_output,
    )
