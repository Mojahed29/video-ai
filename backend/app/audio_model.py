"""
Audio track scoring: synthetic-speech detection (Wav2Vec2-based binary
real/fake classifier, see ``settings.audio_model_id``).

Silence / music-only / missing-audio tracks are detected with a cheap RMS-energy
check before invoking the model, and reported as "not assessed" rather than fed
to a speech model that would produce meaningless output.

The overall audio sub-score is computed over the whole clip. In addition, the
clip is split into a handful of windows that are each scored, producing a
timeline the UI plots so the user can see how the synthetic-speech likelihood
varies over time.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from . import calibration, model_runtime
from .labels import fake_probability
from .schemas import ModalityResult, TimelinePoint
from .settings import settings

logger = logging.getLogger("video_ai.audio")


def score_audio(audio_path: Optional[Path]) -> ModalityResult:
    """Score an extracted WAV file and return a 0–100 audio likelihood + timeline."""
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
    if rms < settings.silence_rms_threshold:
        return ModalityResult(
            status="not_assessed",
            reason="Audio track is silent (or near-silent); no speech to assess.",
        )

    try:
        loaded = model_runtime.get_audio_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.error("Audio model failed to load: %s", exc)
        return ModalityResult(status="not_assessed", reason=f"Audio model failed to load: {exc}")

    whole_clip = _classify(loaded.pipe, samples, sample_rate)
    if whole_clip is None:
        return ModalityResult(
            status="not_assessed",
            reason="Audio classifier returned an unrecognized label set or failed on this clip.",
            model_used=loaded.model_id,
        )

    total_seconds = samples.size / sample_rate
    raw = round(whole_clip * 100, 1)
    score = calibration.calibrate_audio(raw, audio_seconds=total_seconds, settings=settings)
    timeline = _build_timeline(loaded.pipe, samples, sample_rate, whole_clip)

    return ModalityResult(
        status="assessed",
        score=score,
        raw_score=raw if score != raw else None,
        model_used=loaded.model_id,
        detail=f"Scored {total_seconds:.1f}s of audio at {sample_rate} Hz (RMS={rms:.4f}).",
        timeline=timeline,
    )


def _classify(pipe, samples: np.ndarray, sample_rate: int) -> Optional[float]:
    """Run the classifier on a samples array and return P(fake), or None on failure."""
    try:
        output = pipe({"raw": samples, "sampling_rate": sample_rate}, top_k=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Audio classifier failed: %s", exc)
        return None
    return fake_probability(output)


def _build_timeline(pipe, samples: np.ndarray, sample_rate: int, whole_clip: float) -> list[TimelinePoint]:
    """Score evenly sized windows of the clip for the explainability timeline."""
    total_seconds = samples.size / sample_rate
    n_windows = min(
        settings.audio_timeline_windows,
        max(1, int(total_seconds // max(0.1, settings.audio_timeline_min_window_s))),
    )

    if n_windows <= 1:
        # Too short to split meaningfully: one window == the whole-clip score.
        return [TimelinePoint(t_start=0.0, t_end=round(total_seconds, 2), score=round(whole_clip * 100, 1))]

    timeline: list[TimelinePoint] = []
    window_samples = samples.size // n_windows
    for i in range(n_windows):
        start = i * window_samples
        end = samples.size if i == n_windows - 1 else (i + 1) * window_samples
        prob = _classify(pipe, samples[start:end], sample_rate)
        if prob is None:
            continue
        timeline.append(
            TimelinePoint(
                t_start=round(start / sample_rate, 2),
                t_end=round(end / sample_rate, 2),
                score=round(prob * 100, 1),
            )
        )
    return timeline
