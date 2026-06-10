"""
Central, environment-driven configuration for the Video AI Detector.

All tunables live here and can be overridden via environment variables or a
``.env`` file (see ``.env.example``). Variables use the ``VIDEO_AI_`` prefix,
e.g. ``VIDEO_AI_NUM_FRAMES=16`` or ``VIDEO_AI_CORS_ALLOW_ORIGINS=https://a.com,https://b.com``.

To swap in a different pretrained model, change ``VIDEO_AI_VISUAL_MODEL_ID`` /
``VIDEO_AI_AUDIO_MODEL_ID`` (or edit the defaults below) — the rest of the
pipeline reads labels dynamically from each model's own config, so any binary
real/fake classifier exposed through the ``transformers`` pipeline API works
(``image-classification`` for visual, ``audio-classification`` for audio).
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="VIDEO_AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Visual (face / frame) deepfake classifier --------------------------
    # Vision Transformer fine-tuned for binary real-vs-fake image classification.
    # https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model
    visual_model_id: str = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    # Used only if the primary model fails to download/load.
    visual_model_fallback_id: str = "prithivMLmods/deepfake-detector-model-v1"

    # --- Audio (synthetic speech) classifier --------------------------------
    # Wav2Vec2-based binary real-vs-fake speech classifier.
    # https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2
    audio_model_id: str = "MelodyMachine/Deepfake-audio-detection-V2"
    # Used only if the primary model fails to download/load (the base model the
    # primary was fine-tuned from).
    audio_model_fallback_id: str = "mo-thecreator/Deepfake-audio-detection"

    # --- Pipeline parameters ------------------------------------------------
    num_frames: int = 24  # evenly spaced frames sampled from the video
    max_file_size_mb: int = 100
    max_duration_seconds: float = 60.0
    min_duration_seconds: float = 2.0  # below this we warn about "very short video"
    audio_sample_rate: int = 16_000  # required input rate for wav2vec2-family models
    silence_rms_threshold: float = 0.003  # below this average RMS, audio is treated as silent

    # Audio explainability timeline: the clip is split into at most this many
    # windows (each at least ``audio_timeline_min_window_s`` long) and each
    # window is scored so the UI can plot how the synthetic-speech likelihood
    # varies over time. The overall audio sub-score is still the whole-clip
    # score (behaviour preserved); the timeline is supplementary.
    audio_timeline_windows: int = 6
    audio_timeline_min_window_s: float = 2.0

    # --- Fusion -------------------------------------------------------------
    visual_weight: float = 0.6
    audio_weight: float = 0.4

    # --- Calibration --------------------------------------------------------
    # Honest, opt-in calibration. When enabled, scores derived from little
    # signal (few scored frames / very short audio) are shrunk toward the
    # neutral 50 prior, so a "90% fake" from a single frame is reported less
    # confidently than the same number from many frames. Default OFF preserves
    # the raw model behaviour exactly. No accuracy claim is implied either way.
    calibration_enabled: bool = False
    calibration_full_confidence_frames: int = 8
    calibration_full_confidence_audio_seconds: float = 3.0
    calibration_max_shrink: float = 0.35  # cap on how far a score is pulled toward 50

    # --- Runtime ------------------------------------------------------------
    force_cpu: bool = False  # set true to ignore an available CUDA GPU

    # --- API ----------------------------------------------------------------
    # Replaces the previous allow_origins=["*"]. The frontend is served from the
    # same origin, so the defaults only matter for cross-origin API consumers.
    cors_allow_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow ``A,B,C`` (env-friendly) in addition to a JSON list."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):  # let pydantic parse JSON lists
                return value
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()


# Convenient module-level singleton for imports: ``from .settings import settings``.
settings = get_settings()
