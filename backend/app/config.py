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
# https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2
AUDIO_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
# Used only if the primary model fails to download/load (the model the
# primary was fine-tuned from).
AUDIO_MODEL_FALLBACK_ID = "mo-thecreator/Deepfake-audio-detection"

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

# Device selection: "cuda" if available else "cpu" (resolved at runtime in
# model_runtime.py — kept here only as an override switch).
FORCE_CPU = False
