"""Fusion math + confidence bands."""
import pytest

from app import fusion
from app.schemas import ModalityResult


def assessed(score: float) -> ModalityResult:
    return ModalityResult(status="assessed", score=score)


def not_assessed(reason: str = "n/a") -> ModalityResult:
    return ModalityResult(status="not_assessed", reason=reason)


def test_both_assessed_weighted_average_high_confidence():
    overall, method, band = fusion.fuse(assessed(80.0), assessed(30.0))
    # default weights 0.6 / 0.4 -> 0.6*80 + 0.4*30 = 60.0
    assert overall == pytest.approx(60.0)
    assert band == "high"
    assert "visual" in method and "audio" in method


def test_visual_only_uses_visual_medium_confidence():
    overall, method, band = fusion.fuse(assessed(72.5), not_assessed())
    assert overall == pytest.approx(72.5)
    assert band == "medium"
    assert "Visual" in method


def test_audio_only_uses_audio_medium_confidence():
    overall, method, band = fusion.fuse(not_assessed(), assessed(41.0))
    assert overall == pytest.approx(41.0)
    assert band == "medium"
    assert "Audio" in method


def test_neither_assessed_neutral_low_confidence():
    overall, method, band = fusion.fuse(not_assessed(), not_assessed())
    assert overall == pytest.approx(50.0)
    assert band == "low"


def test_weights_are_normalised(monkeypatch):
    # Non-normalised weights (3:1) must still behave as a proper weighted average.
    monkeypatch.setattr(fusion.settings, "visual_weight", 3.0)
    monkeypatch.setattr(fusion.settings, "audio_weight", 1.0)
    overall, _method, _band = fusion.fuse(assessed(100.0), assessed(0.0))
    assert overall == pytest.approx(75.0)


def test_result_is_rounded_to_one_decimal():
    overall, _m, _b = fusion.fuse(assessed(33.33), assessed(66.66))
    assert overall == round(overall, 1)
