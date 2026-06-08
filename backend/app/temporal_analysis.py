"""
Heuristic, model-agnostic temporal-consistency signal for the visual track.

Per-frame classifiers structurally cannot see one of the most commonly cited
deepfake/synthesis artifacts: frame-to-frame instability — blending-boundary
flicker, texture "swimming", or unnaturally static regions — that goes beyond
what natural motion and ordinary video compression produce.

We approximate "how erratic is the frame-to-frame change" with the coefficient
of variation (std / mean) of pixel-difference magnitude between consecutive
sampled frames, then squash it into a [0, 1] "inconsistency" score that nudges
the per-frame classifier's average at a small, configurable weight
(config.TEMPORAL_SIGNAL_WEIGHT).

This is a WEAK heuristic signal — not a standalone detector — meant to add
information the per-frame classifier cannot see on its own, not to replace it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

# Natural video (motion + compression noise) typically produces a coefficient
# of variation comfortably below this; values clearly above it indicate erratic,
# "jumpy" frame-to-frame change rather than smooth natural motion.
_EXPECTED_NATURAL_COV = 0.6
_SQUASH_STEEPNESS = 4.0


def compute_temporal_signal(ordered_images: list[Image.Image]) -> Optional[dict]:
    """
    `ordered_images` must be the same crop/frame chosen for scoring at each
    sampled timestamp, in chronological order. Returns None if there are too
    few frames to say anything meaningful, or if the sequence is degenerate
    (e.g. a static test pattern with zero frame-to-frame change).
    """
    if len(ordered_images) < 4:
        return None

    arrays = [
        np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
        for img in ordered_images
    ]
    diffs = [float(np.mean(np.abs(arrays[i] - arrays[i - 1]))) for i in range(1, len(arrays))]

    mean_diff = float(np.mean(diffs))
    if mean_diff < 1e-6:
        return None

    coefficient_of_variation = float(np.std(diffs) / mean_diff)
    inconsistency = float(1.0 / (1.0 + np.exp(-(coefficient_of_variation - _EXPECTED_NATURAL_COV) * _SQUASH_STEEPNESS)))

    return {
        "coefficient_of_variation": round(coefficient_of_variation, 3),
        "inconsistency_score": round(inconsistency, 3),
        "note": (
            "Heuristic signal: higher values mean more erratic frame-to-frame "
            "change than typical natural video motion + compression."
        ),
    }
