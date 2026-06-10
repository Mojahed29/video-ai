"""
Central configuration for the AI-video detector.

To swap in a different pretrained model, change ONE constant below
(VISUAL_MODEL_ID or AUDIO_MODEL_ID) — the rest of the pipeline reads
labels dynamically from the model's own config (id2label), so no other
code needs to change as long as the replacement model is a binary
real/fake classifier exposed through the `transformers` pipeline API
(image-classification for visual, audio-classification for audio).
"""

# --- Visual (face / frame) deepfake classifier -----------------------------
# Vision Transformer fine-tuned for binary real-vs-fake image classification.
# https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model
VISUAL_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
# Used only if the primary model fails to download/load.
VISUAL_MODEL_FALLBACK_ID = "prithivMLmods/deepfake-detector-model-v1"

# --- Audio (synthetic speech) classifier ------------------------------------
# Wav2Vec2-based binary real-vs-fake speech classifier.
# We default to the BASE model (mo-thecreator/Deepfake-audio-detection) rather
# than the narrowly fine-tuned V2: the fine-tune tends to be overconfident on
# ordinary compressed/resampled video audio and report "fake" with near-100%
# probability on genuine clips. The base model is generally better calibrated.
# https://huggingface.co/mo-thecreator/Deepfake-audio-detection
AUDIO_MODEL_ID = "mo-thecreator/Deepfake-audio-detection"
# Used only if the primary model fails to download/load.
# https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2
AUDIO_MODEL_FALLBACK_ID = "MelodyMachine/Deepfake-audio-detection-V2"

# Probability calibration for the audio classifier. Small fine-tuned speech
# deepfake models are often badly over-confident on real-world audio (they pin
# the score near 0 or 100). This applies temperature softening in logit space:
#   1.0  = no change
#   >1.0 = pull confident scores back toward 50% (softer, less saturated)
# Raising this makes a "100% fake" reading on genuine audio less extreme.
AUDIO_CALIBRATION_TEMPERATURE = 1.6

# --- Pipeline parameters -----------------------------------------------------
NUM_FRAMES = 24                 # evenly spaced frames sampled from the video
MAX_FILE_SIZE_MB = 100
MAX_DURATION_SECONDS = 60
MIN_DURATION_SECONDS = 2.0      # below this we warn about "very short video"
AUDIO_SAMPLE_RATE = 16_000      # required input rate for wav2vec2-family models
SILENCE_RMS_THRESHOLD = 0.003   # below this average RMS, audio is treated as silent

# --- Fusion -------------------------------------------------------------------
VISUAL_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4

# --- Temporal consistency (visual) -------------------------------------------
# Heuristic frame-to-frame "jitteriness" signal (see temporal_analysis.py) that
# is blended into the per-frame classifier's average. Set to 0 to disable.
TEMPORAL_SIGNAL_WEIGHT = 0.15

# --- Explainability -----------------------------------------------------------
# Occlusion-based saliency heatmap for the single most-suspicious detected
# face/frame (see explainability.py). Costs SALIENCY_GRID_SIZE^2 extra forward
# passes on ONE crop — not every frame — so it stays cheap on CPU.
ENABLE_VISUAL_SALIENCY = True
SALIENCY_GRID_SIZE = 4

# Per-segment "how synthetic does THIS part sound" timeline for the audio
# track (see explainability.py). Costs up to AUDIO_TIMELINE_MAX_SEGMENTS extra
# audio-classifier calls on short, non-overlapping chunks of the same clip.
ENABLE_AUDIO_TIMELINE = True
AUDIO_TIMELINE_MAX_SEGMENTS = 8
AUDIO_TIMELINE_MIN_DURATION = 3.0          # below this, a timeline isn't meaningful
AUDIO_TIMELINE_MIN_SEGMENT_SECONDS = 1.5   # each segment is at least this long

# Device selection: "cuda" if available else "cpu" (resolved at runtime in
# model_runtime.py — kept here only as an override switch).
FORCE_CPU = False
