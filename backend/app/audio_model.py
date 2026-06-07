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

from . import config, model_runtime
from .schemas import ModalityResult

logger = logging.getLogger("video_ai.audio")


def _fake_probability(pipe_output: list[dict]) -> Optional[float]:
    """Same label-substring matching strategy as the visual model (see visual_model._fake_probability)."""
    fake_score = None
    real_score = None
    for entry in pipe_output:
        label = str(entry.get("label", "")).lower()
        score = float(entry.get("score", 0.0))
        if any(k in label for k in ("fake", "spoof", "synthetic", "ai", "generated", "tts")):
            fake_score = score if fake_score is None else max(fake_score, score)
        elif any(k in label for k in ("real", "bonafide", "authentic", "genuine", "human")):
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

    try:
        output = loaded.pipe({"raw": samples, "sampling_rate": sample_rate}, top_k=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Audio classifier failed: %s", exc)
        return ModalityResult(status="not_assessed", reason=f"Audio classifier failed on this clip: {exc}", model_used=loaded.model_id)

    prob = _fake_probability(output)
    if prob is None:
        return ModalityResult(
            status="not_assessed",
            reason="Audio classifier returned an unrecognized label set.",
            model_used=loaded.model_id,
        )

    return ModalityResult(
        status="assessed",
        score=round(prob * 100, 1),
        model_used=loaded.model_id,
        detail=f"Scored {samples.size / sample_rate:.1f}s of audio at {sample_rate} Hz (RMS={rms:.4f}).",
    )
