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

    # Isolate speech with VAD so silence/music/noise isn't scored as "speech".
    scored, used_vad = _extract_speech(samples, sample_rate)
    if scored is None:
        return ModalityResult(
            status="not_assessed",
            reason="No speech detected in the audio (only silence, music, or noise).",
        )

    try:
        loaded = model_runtime.get_audio_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.error("Audio model failed to load: %s", exc)
        return ModalityResult(status="not_assessed", reason=f"Audio model failed to load: {exc}")

    raw_prob = _classify(loaded.pipe, scored, sample_rate)
    if raw_prob is None:
        return ModalityResult(
            status="not_assessed",
            reason="Audio classifier returned an unrecognized label set or failed on this clip.",
            model_used=loaded.model_id,
        )

    total_seconds = scored.size / sample_rate
    temperature = settings.audio_calibration_temperature
    soft_prob = calibration.temperature_scale(raw_prob, temperature)

    raw = round(raw_prob * 100, 1)
    score = calibration.calibrate_audio(round(soft_prob * 100, 1), audio_seconds=total_seconds, settings=settings)
    timeline = _build_timeline(loaded.pipe, scored, sample_rate, soft_prob, temperature)

    source = "detected speech (VAD)" if used_vad else "audio"
    detail = f"Scored {total_seconds:.1f}s of {source} at {sample_rate} Hz (RMS={rms:.4f})."
    if raw_prob >= 0.97 or raw_prob <= 0.03:
        detail += (
            " Heads-up: the model's raw score is near the extreme of the scale. Audio deepfake "
            "detectors can be poorly calibrated on out-of-distribution speech, so treat a near-0 "
            "or near-100 audio reading with extra skepticism."
        )

    return ModalityResult(
        status="assessed",
        score=score,
        raw_score=raw,  # always exposed alongside the calibrated score
        model_used=loaded.model_id,
        detail=detail,
        timeline=timeline,
    )


def _extract_speech(samples: np.ndarray, sample_rate: int) -> tuple[Optional[np.ndarray], bool]:
    """
    Return ``(speech_samples, used_vad)``.

    With Silero VAD available, concatenate only the detected speech regions; if
    too little speech is found, return ``(None, True)`` so the caller reports
    "no speech". If VAD is unavailable, return the original samples untouched.
    """
    vad = model_runtime.get_vad()
    if vad is None:
        return samples, False
    try:
        import torch
        from silero_vad import get_speech_timestamps

        wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
        spans = get_speech_timestamps(wav, vad, sampling_rate=sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.warning("VAD failed (%s); scoring the full audio instead.", exc)
        return samples, False

    if not spans:
        return None, True
    speech = np.concatenate([samples[s["start"]:s["end"]] for s in spans])
    if speech.size / sample_rate < settings.vad_min_speech_seconds:
        return None, True
    return speech, True


def _classify(pipe, samples: np.ndarray, sample_rate: int) -> Optional[float]:
    """Run the classifier on a samples array and return P(fake), or None on failure."""
    try:
        output = pipe({"raw": samples, "sampling_rate": sample_rate}, top_k=None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Audio classifier failed: %s", exc)
        return None
    return fake_probability(output)


def _build_timeline(
    pipe, samples: np.ndarray, sample_rate: int, soft_whole_clip: float, temperature: float
) -> list[TimelinePoint]:
    """Score evenly sized windows of the clip for the explainability timeline.

    Window scores are temperature-softened with the same factor as the overall
    score, so the timeline is on the same scale as the headline number.
    """
    total_seconds = samples.size / sample_rate
    n_windows = min(
        settings.audio_timeline_windows,
        max(1, int(total_seconds // max(0.1, settings.audio_timeline_min_window_s))),
    )

    if n_windows <= 1:
        # Too short to split meaningfully: one window == the whole-clip score.
        return [TimelinePoint(t_start=0.0, t_end=round(total_seconds, 2), score=round(soft_whole_clip * 100, 1))]

    timeline: list[TimelinePoint] = []
    window_samples = samples.size // n_windows
    for i in range(n_windows):
        start = i * window_samples
        end = samples.size if i == n_windows - 1 else (i + 1) * window_samples
        prob = _classify(pipe, samples[start:end], sample_rate)
        if prob is None:
            continue
        prob = calibration.temperature_scale(prob, temperature)
        timeline.append(
            TimelinePoint(
                t_start=round(start / sample_rate, 2),
                t_end=round(end / sample_rate, 2),
                score=round(prob * 100, 1),
            )
        )
    return timeline
