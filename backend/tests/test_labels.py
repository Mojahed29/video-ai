"""Label -> P(fake) mapping: the substring strategy that keeps swapped-in models working."""
import pytest

from app.labels import fake_probability


def test_explicit_fake_label_wins():
    out = [{"label": "Fake", "score": 0.83}, {"label": "Real", "score": 0.17}]
    assert fake_probability(out) == pytest.approx(0.83)


def test_real_label_is_inverted():
    # Only a "real" class present -> P(fake) = 1 - P(real).
    out = [{"label": "REAL", "score": 0.9}]
    assert fake_probability(out) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "fake_label",
    ["deepfake", "Synthetic-Speech", "AI-generated", "spoof", "tts", "GENERATED"],
)
def test_fake_synonyms_match(fake_label):
    out = [{"label": fake_label, "score": 0.6}, {"label": "bonafide", "score": 0.4}]
    assert fake_probability(out) == pytest.approx(0.6)


@pytest.mark.parametrize(
    "real_label",
    ["authentic", "Genuine", "pristine", "bonafide", "human"],
)
def test_real_synonyms_match(real_label):
    out = [{"label": real_label, "score": 0.7}]
    assert fake_probability(out) == pytest.approx(0.3)


def test_unrecognized_labels_return_none():
    out = [{"label": "class_0", "score": 0.5}, {"label": "class_1", "score": 0.5}]
    assert fake_probability(out) is None


def test_empty_output_returns_none():
    assert fake_probability([]) is None


def test_takes_max_when_multiple_fake_entries():
    out = [
        {"label": "fake-method-a", "score": 0.4},
        {"label": "fake-method-b", "score": 0.72},
    ]
    assert fake_probability(out) == pytest.approx(0.72)


def test_malformed_score_is_skipped_gracefully():
    out = [{"label": "fake", "score": "not-a-number"}, {"label": "real", "score": 0.8}]
    # The fake entry is unparseable -> falls back to the real entry.
    assert fake_probability(out) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "real_label",
    ["Bona-fide", "bona fide", "LABEL_1: Real", "Authentic ", "GENUINE"],
)
def test_punctuation_and_spacing_normalized(real_label):
    # Hyphens/spaces/prefixes must not break matching (regression for label-name variants).
    out = [{"label": real_label, "score": 0.85}]
    assert fake_probability(out) == pytest.approx(0.15)


def test_label_zero_one_only_is_unrecognized():
    # Bare LABEL_0 / LABEL_1 carry no real/fake meaning -> None (not_assessed).
    out = [{"label": "LABEL_0", "score": 0.6}, {"label": "LABEL_1", "score": 0.4}]
    assert fake_probability(out) is None
