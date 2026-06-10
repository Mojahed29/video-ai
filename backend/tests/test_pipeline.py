"""
Edge-case behaviour for the visual/audio scorers and the HTTP layer.

Model loading is mocked, so these tests run without downloading any weights.
The corrupt-file test uses the real ffprobe (skips if it isn't installed).
"""
import shutil

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")  # noqa: F841  (import side effect: skip if missing)

from app import audio_model, model_runtime, visual_model  # noqa: E402
from app.model_runtime import LoadedPipeline  # noqa: E402
from app.video_utils import ExtractedFrame  # noqa: E402


class FakePipe:
    """Stand-in for a transformers classification pipeline."""

    def __init__(self, output):
        self._output = output

    def __call__(self, *_args, **_kwargs):
        return self._output


def _fake_loaded(output, model_id="test/model"):
    return LoadedPipeline(pipe=FakePipe(output), model_id=model_id)


# --------------------------------------------------------------------------- #
# Audio edge cases
# --------------------------------------------------------------------------- #
def test_audio_none_is_not_assessed():
    result = audio_model.score_audio(None)
    assert result.status == "not_assessed"
    assert "no audio track" in result.reason.lower()


def test_audio_silent_is_not_assessed(monkeypatch):
    monkeypatch.setattr(audio_model.sf, "read", lambda *a, **k: (np.zeros(16000, dtype="float32"), 16000))
    result = audio_model.score_audio("dummy.wav")
    assert result.status == "not_assessed"
    assert "silent" in result.reason.lower()


def test_audio_empty_is_not_assessed(monkeypatch):
    monkeypatch.setattr(audio_model.sf, "read", lambda *a, **k: (np.array([], dtype="float32"), 16000))
    result = audio_model.score_audio("dummy.wav")
    assert result.status == "not_assessed"
    assert "empty" in result.reason.lower()


def test_audio_model_load_failure_is_not_assessed(monkeypatch):
    monkeypatch.setattr(audio_model.sf, "read", lambda *a, **k: (np.full(16000, 0.2, dtype="float32"), 16000))
    monkeypatch.setattr(model_runtime, "get_audio_pipeline", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    result = audio_model.score_audio("dummy.wav")
    assert result.status == "not_assessed"
    assert "failed to load" in result.reason.lower()


def test_audio_unrecognized_labels_is_not_assessed(monkeypatch):
    monkeypatch.setattr(audio_model.sf, "read", lambda *a, **k: (np.full(16000, 0.2, dtype="float32"), 16000))
    monkeypatch.setattr(model_runtime, "get_audio_pipeline", lambda: _fake_loaded([{"label": "class_0", "score": 1.0}]))
    result = audio_model.score_audio("dummy.wav")
    assert result.status == "not_assessed"


def test_audio_assessed_produces_score_and_timeline(monkeypatch):
    # 8s of audible audio -> multiple timeline windows.
    # Temperature 1.0 isolates this from the default audio softening.
    monkeypatch.setattr(audio_model.settings, "audio_calibration_temperature", 1.0)
    monkeypatch.setattr(audio_model.sf, "read", lambda *a, **k: (np.full(16000 * 8, 0.2, dtype="float32"), 16000))
    monkeypatch.setattr(
        model_runtime,
        "get_audio_pipeline",
        lambda: _fake_loaded([{"label": "fake", "score": 0.7}, {"label": "real", "score": 0.3}]),
    )
    result = audio_model.score_audio("dummy.wav")
    assert result.status == "assessed"
    assert result.score == pytest.approx(70.0)
    assert result.model_used == "test/model"
    assert len(result.timeline) >= 2  # windowed timeline
    assert all(0 <= p.score <= 100 for p in result.timeline)


def test_audio_temperature_softens_overconfident_score(monkeypatch):
    # A model that screams "100% fake" should be tempered below 100 when
    # temperature > 1 (the "always 100" mitigation).
    monkeypatch.setattr(audio_model.settings, "audio_calibration_temperature", 2.0)
    monkeypatch.setattr(audio_model.sf, "read", lambda *a, **k: (np.full(16000 * 4, 0.2, dtype="float32"), 16000))
    monkeypatch.setattr(
        model_runtime,
        "get_audio_pipeline",
        lambda: _fake_loaded([{"label": "fake", "score": 0.99}, {"label": "real", "score": 0.01}]),
    )
    result = audio_model.score_audio("dummy.wav")
    assert result.status == "assessed"
    assert result.score < 99.0  # softened
    assert result.raw_score == pytest.approx(99.0)  # original model score preserved
    assert "Softened" in (result.detail or "")


# --------------------------------------------------------------------------- #
# Visual edge cases
# --------------------------------------------------------------------------- #
def test_visual_no_frames_is_not_assessed():
    result = visual_model.score_frames([])
    assert result.status == "not_assessed"
    assert "no frames" in result.reason.lower()


def test_visual_model_load_failure_is_not_assessed(tmp_path):
    frame = _write_frame(tmp_path)
    import app.visual_model as vm

    def boom():
        raise RuntimeError("download failed")

    orig = model_runtime.get_visual_pipeline
    model_runtime.get_visual_pipeline = boom
    try:
        result = vm.score_frames([frame])
    finally:
        model_runtime.get_visual_pipeline = orig
    assert result.status == "not_assessed"
    assert "failed to load" in result.reason.lower()


def test_visual_no_face_falls_back_to_full_frame(monkeypatch, tmp_path):
    frames = [_write_frame(tmp_path, i) for i in range(3)]
    monkeypatch.setattr(model_runtime, "get_face_detector", lambda: None)
    monkeypatch.setattr(
        model_runtime,
        "get_visual_pipeline",
        lambda: _fake_loaded([{"label": "Fake", "score": 0.8}, {"label": "Real", "score": 0.2}]),
    )
    result = visual_model.score_frames(frames)
    assert result.status == "assessed"
    assert result.score == pytest.approx(80.0)
    assert "No face detected" in result.detail
    assert len(result.timeline) == 3
    assert all(p.note == "no face" for p in result.timeline)


def _write_frame(tmp_path, idx: int = 0) -> ExtractedFrame:
    from PIL import Image

    path = tmp_path / f"frame_{idx:03d}.jpg"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(path)
    return ExtractedFrame(timestamp=float(idx), path=path)


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def test_wrong_file_type_returns_400(client):
    resp = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_oversized_file_returns_413(client, monkeypatch):
    from app import main

    monkeypatch.setattr(main.settings, "max_file_size_mb", 0)  # any non-empty upload is "too large"
    resp = client.post("/api/analyze", files={"file": ("clip.mp4", b"x" * 1024, "video/mp4")})
    assert resp.status_code == 413
    assert "limit" in resp.json()["detail"].lower()


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")
def test_corrupt_file_returns_422(client):
    resp = client.post("/api/analyze", files={"file": ("clip.mp4", b"this is not a video", "video/mp4")})
    assert resp.status_code == 422
    assert "could not read" in resp.json()["detail"].lower()
