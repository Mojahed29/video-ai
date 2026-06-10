"""Uncertainty-aware calibration: identity by default, shrink-toward-50 when enabled."""
import pytest

from app import calibration
from app.settings import Settings


def _settings(**overrides) -> Settings:
    base = dict(
        calibration_enabled=True,
        calibration_full_confidence_frames=8,
        calibration_full_confidence_audio_seconds=3.0,
        calibration_max_shrink=0.35,
    )
    base.update(overrides)
    return Settings(**base)


def test_disabled_is_identity():
    s = Settings(calibration_enabled=False)
    assert calibration.calibrate_visual(83.7, frames_scored=1, settings=s) == 83.7
    assert calibration.calibrate_audio(12.4, audio_seconds=0.2, settings=s) == 12.4


def test_full_signal_leaves_score_unchanged():
    s = _settings()
    assert calibration.calibrate_visual(90.0, frames_scored=8, settings=s) == pytest.approx(90.0)
    assert calibration.calibrate_audio(90.0, audio_seconds=3.0, settings=s) == pytest.approx(90.0)


def test_no_signal_shrinks_by_max_toward_50():
    s = _settings()
    # confidence 0 -> shrink 0.35 -> 90 + (50-90)*0.35 = 76.0
    assert calibration.calibrate_visual(90.0, frames_scored=0, settings=s) == pytest.approx(76.0)


def test_partial_signal_shrinks_proportionally():
    s = _settings()
    # 4/8 frames -> confidence 0.5 -> shrink 0.175 -> 90 + (-40)*0.175 = 83.0
    assert calibration.calibrate_visual(90.0, frames_scored=4, settings=s) == pytest.approx(83.0)


def test_shrink_pulls_low_scores_up_toward_50():
    s = _settings()
    # 10 -> 10 + (50-10)*0.35 = 24.0 at zero confidence
    assert calibration.calibrate_audio(10.0, audio_seconds=0.0, settings=s) == pytest.approx(24.0)


def test_confidence_is_capped_at_one():
    s = _settings()
    # More than "full" signal must not push the score away from the raw value.
    assert calibration.calibrate_visual(90.0, frames_scored=100, settings=s) == pytest.approx(90.0)


# --- temperature scaling --------------------------------------------------- #
def test_temperature_one_is_identity():
    assert calibration.temperature_scale(0.9, 1.0) == pytest.approx(0.9)
    assert calibration.temperature_scale(0.02, 1.0) == pytest.approx(0.02)


def test_temperature_none_passthrough():
    assert calibration.temperature_scale(None, 1.5) is None


def test_temperature_above_one_pulls_toward_half():
    # 0.9 should move toward 0.5 (down) when softened.
    softened = calibration.temperature_scale(0.9, 2.0)
    assert 0.5 < softened < 0.9
    # 0.1 should move toward 0.5 (up).
    softened_low = calibration.temperature_scale(0.1, 2.0)
    assert 0.1 < softened_low < 0.5


def test_temperature_preserves_neutral_point():
    assert calibration.temperature_scale(0.5, 3.0) == pytest.approx(0.5)


def test_temperature_below_one_sharpens():
    assert calibration.temperature_scale(0.7, 0.5) > 0.7
