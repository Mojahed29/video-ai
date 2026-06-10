"""
Lazy, cached loading of the pretrained models with fallback handling.

Models are loaded once on first use and reused for every request. If the
primary model ID fails to download or load, we fall back to the documented
alternative and remember which one actually ended up being used, so the API can
report it honestly (``visual.model_used`` / ``audio.model_used`` and
``/api/health``). If both fail, the caller marks that modality
``"not_assessed"`` rather than crashing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from .settings import settings

logger = logging.getLogger("video_ai.models")

# torch / transformers are imported lazily inside the functions below so the
# package imports cheaply (and is testable) without the full ML stack present.


def get_device() -> str:
    """Return ``"cuda"`` when a GPU is available and not force-disabled, else ``"cpu"``."""
    import torch

    if not settings.force_cpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _device_index(device: str) -> int:
    """``transformers.pipeline`` expects -1 for CPU, the GPU index otherwise."""
    return 0 if device == "cuda" else -1


@dataclass(frozen=True)
class LoadedPipeline:
    pipe: Any
    model_id: str  # the model that actually loaded (primary or fallback)


def _load_with_fallback(task: str, primary_id: str, fallback_id: str, device: str) -> LoadedPipeline:
    from transformers import pipeline

    device_index = _device_index(device)
    try:
        logger.info("Loading %s model '%s' on %s ...", task, primary_id, device)
        pipe = pipeline(task, model=primary_id, device=device_index)
        return LoadedPipeline(pipe=pipe, model_id=primary_id)
    except Exception as exc:  # noqa: BLE001 - fall back on *any* load failure
        logger.warning(
            "Primary %s model '%s' failed to load (%s). Falling back to '%s'.",
            task, primary_id, exc, fallback_id,
        )
        try:
            pipe = pipeline(task, model=fallback_id, device=device_index)
            return LoadedPipeline(pipe=pipe, model_id=fallback_id)
        except Exception as exc2:  # noqa: BLE001
            logger.error("Fallback %s model '%s' also failed to load: %s", task, fallback_id, exc2)
            raise RuntimeError(
                f"Could not load any {task} model (tried '{primary_id}' and '{fallback_id}'): {exc2}"
            ) from exc2


@lru_cache(maxsize=1)
def get_visual_pipeline() -> LoadedPipeline:
    return _load_with_fallback(
        "image-classification",
        settings.visual_model_id,
        settings.visual_model_fallback_id,
        get_device(),
    )


@lru_cache(maxsize=1)
def get_audio_pipeline() -> LoadedPipeline:
    return _load_with_fallback(
        "audio-classification",
        settings.audio_model_id,
        settings.audio_model_fallback_id,
        get_device(),
    )


@lru_cache(maxsize=1)
def get_face_detector() -> Optional[Any]:
    """MTCNN face detector (facenet-pytorch). Returns ``None`` if it cannot load."""
    try:
        from facenet_pytorch import MTCNN

        return MTCNN(keep_all=True, device=get_device())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load MTCNN face detector (%s); will score full frames instead.", exc)
        return None


def warm_up() -> dict[str, str]:
    """Force-load every model once (startup / health check) and report what loaded."""
    info: dict[str, str] = {"device": get_device()}
    try:
        info["visual_model"] = get_visual_pipeline().model_id
    except Exception as exc:  # noqa: BLE001
        info["visual_model"] = f"FAILED: {exc}"
    try:
        info["audio_model"] = get_audio_pipeline().model_id
    except Exception as exc:  # noqa: BLE001
        info["audio_model"] = f"FAILED: {exc}"
    get_face_detector()
    return info
