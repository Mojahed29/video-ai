# Video AI Detector

A small full-stack app that estimates how likely an uploaded video is
**AI-generated or face-swapped**, by scoring its visual track and audio track
**separately** with pretrained open models, then fusing both probabilities
into a single 0–100 score.

> ⚠️ **This is a heuristic signal, not a verdict.** See [Disclaimer](#disclaimer).

---

## How it works

```
upload (mp4/mov/webm/mkv, ≤100MB, ≤60s)
   │
   ├─ ffprobe: duration, has-audio?, corrupt-file check
   │
   ├─ ffmpeg → N evenly spaced JPEG frames (default 24)
   │     └─ MTCNN face detection per frame
   │           ├─ face(s) found → crop & run visual classifier on each face
   │           └─ no face       → run visual classifier on the full frame
   │     └─ classifier average ──┐
   │     └─ + temporal-consistency "jitteriness" heuristic (weight 15%)
   │                              ├─→ VISUAL sub-score = P(fake)        →  0–100
   │     └─ + occlusion saliency heatmap of the most-suspicious face/frame
   │
   ├─ ffmpeg → mono 16kHz WAV
   │     └─ RMS silence check (skip model if track is silent/near-silent)
   │     └─ AUDIO sub-score = P(synthetic speech)                →  0–100
   │           (or "not assessed" if no audio / silent / model failure)
   │     └─ + per-segment "where in the clip" timeline
   │
   └─ FUSION: weighted average  (default 60% visual / 40% audio)
         · if one modality is missing → use the other alone, and say so
         · confidence band: high (both assessed) / medium (one) / low (neither)
```

The backend returns `overall_score`, both sub-scores, per-modality status,
the exact model IDs used (including whether a fallback model had to be
loaded), the number of frames analyzed, and a confidence band.

---

## Models used

Picked for being well-downloaded, maintained, binary real/fake classifiers
that load directly through `transformers.pipeline` — no training required,
inference only.

| Track  | Primary model | Fallback model |
|--------|---------------|----------------|
| Visual (deepfake / synthetic image) | [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) — ViT (`google/vit-base-patch16-224-in21k`) fine-tuned for binary Real/Fake image classification | [`prithivMLmods/deepfake-detector-model-v1`](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1) |
| Audio (synthetic / spoofed speech) | [`MelodyMachine/Deepfake-audio-detection-V2`](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2) — Wav2Vec2-based binary Real/Fake speech classifier | [`mo-thecreator/Deepfake-audio-detection`](https://huggingface.co/mo-thecreator/Deepfake-audio-detection) (the model the primary was fine-tuned from) |
| Face localization | [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) MTCNN (not a deepfake classifier — just crops faces for the visual model) | — |

If the primary model ID fails to download or load (e.g. it gets pulled,
renamed, or is unreachable), the app **automatically falls back** to the
documented alternative and the API/UI report **which model actually ran**
(`visual.model_used` / `audio.model_used` and the `/api/health` endpoint).
If both fail to load, that modality is reported as `"not_assessed"` with the
load error — the app does not crash.

### Swapping in a different model

Change exactly one constant in [`backend/app/config.py`](backend/app/config.py):

```python
VISUAL_MODEL_ID = "your-org/your-model"   # any binary real/fake image-classification model
AUDIO_MODEL_ID  = "your-org/your-model"   # any binary real/fake audio-classification model
```

Labels are matched by substring (`"fake"/"real"`, `"spoof"/"bonafide"`,
`"synthetic"/"authentic"`, …) in `visual_model.fake_probability` /
`audio_model.fake_probability`, so most binary real-vs-fake classifiers work
without further changes.

---

## Beyond the base score: temporal consistency & explainability

Two extra heuristic signals are layered on top of the raw classifier outputs.
Both are **model-agnostic** (they reuse the existing pipelines — no new
detector models to pick/verify) and **toggleable** in `config.py`.

### Temporal consistency (visual)
A single-frame classifier structurally cannot see one of the most commonly
cited deepfake artifacts: frame-to-frame instability (blending-boundary
flicker, texture "swimming"). `temporal_analysis.py` measures the coefficient
of variation of pixel-difference magnitude between the consecutive crops/frames
that were scored, squashes it into a `[0, 1]` "inconsistency" score, and blends
it into the visual sub-score at a small configurable weight
(`TEMPORAL_SIGNAL_WEIGHT`, default `0.15`). Steady natural motion scores near
zero; bursty, sporadic jumps (the kind blending-boundary flicker produces)
score high — this was verified with synthetic test sequences during
development. Returned as `visual.temporal_consistency`.

### Explainability overlays
- **Visual — occlusion saliency** (`visual.saliency`): the single
  most-suspicious detected face/frame is divided into a grid; each cell is
  masked and the image is re-scored. Cells whose removal drops the
  fake-probability the most are highlighted (red-tinted) in a heatmap returned
  as a base64 PNG — i.e. "here's what the model relied on for *this* image."
  Costs `SALIENCY_GRID_SIZE²` extra forward passes on **one** crop (default
  4×4 = 16), not every frame, to stay fast on CPU. Toggle with
  `ENABLE_VISUAL_SALIENCY`.
- **Audio — per-segment timeline** (`audio.timeline`): the clip is split into
  a handful of equal, non-overlapping windows, each independently re-scored,
  producing a "how synthetic does *this part* sound" timeline so users can see
  *where* the signal is strongest rather than one number for the whole track.
  Toggle with `ENABLE_AUDIO_TIMELINE` / tune segment count and length with
  `AUDIO_TIMELINE_*`.
- **Audio — raw classifier output** (`audio.raw_model_output`): the exact
  `{label, score}` pairs the audio model returned for the whole clip, exposed
  as-is so you can see precisely how the 0–100 score was derived from the
  model's *own* labels. This is most useful when a score looks saturated
  (near 0 or near 100): `audio.detail` adds an explicit caution note in that
  case, since small fine-tuned speech-deepfake classifiers can be poorly
  calibrated on compressed/resampled real-world audio that differs from their
  narrow training distribution — a confidently-wrong "100% fake" reading on
  ordinary audio is a calibration artifact, not a verdict.

> The saliency heatmap and audio timeline are **post-hoc, occlusion/segment-based explanations of the model's
> behavior on this specific input** — not architecture-level explanations
> (e.g. not gradient/attention-based) and not proof of *why* the network
> learned what it learned. They're meant to build calibrated trust ("here's
> what drove this number"), not to serve as independent evidence.

---

## Project structure

```
video-ai/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app, /api/analyze and /api/health routes
│   │   ├── config.py          ★ all tunables — model IDs, weights, limits
│   │   ├── pipeline.py        end-to-end orchestration + edge-case handling
│   │   ├── video_utils.py     ffmpeg/ffprobe: probing, frame & audio extraction
│   │   ├── visual_model.py    face detection + visual deepfake scoring
│   │   ├── audio_model.py     silence detection + synthetic-speech scoring
│   │   ├── temporal_analysis.py   heuristic frame-to-frame consistency signal
│   │   ├── explainability.py      occlusion saliency heatmap + audio timeline
│   │   ├── fusion.py          weighted-average fusion + confidence band
│   │   ├── model_runtime.py   lazy model loading, device selection, fallback
│   │   └── schemas.py         pydantic response models
│   └── requirements.txt
├── frontend/
│   ├── index.html             single-page UI (drag & drop, results card, "why this score?" panels)
│   ├── style.css
│   └── app.js                 upload, progress, result + explainability rendering
└── README.md
```

---

## Setup

### 1. Install ffmpeg (required — used for frame & audio extraction)

| OS | Command |
|----|---------|
| Ubuntu / Debian | `sudo apt-get update && sudo apt-get install -y ffmpeg` |
| macOS (Homebrew) | `brew install ffmpeg` |
| Windows | Download a build from [ffmpeg.org](https://ffmpeg.org/download.html), add the `bin/` folder to your `PATH` |

Verify with:

```bash
ffmpeg -version
ffprobe -version
```

### 2. Install Python dependencies

Python 3.11 recommended. From the `backend/` directory:

```bash
cd backend
pip install -r requirements.txt
```

This installs FastAPI, PyTorch (CPU build by default), `transformers`,
`facenet-pytorch` (MTCNN face detector), and audio I/O libraries.

> **Optional GPU acceleration:** if you have a CUDA-capable GPU, install a
> matching CUDA build of `torch`/`torchvision`/`torchaudio` from
> [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
> *before* running `pip install -r requirements.txt` (pip will then reuse
> your GPU build). The app auto-detects CUDA via `torch.cuda.is_available()`
> and reports the active device at `/api/health`.

### 3. Run

From the `backend/` directory:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** — the FastAPI app serves the frontend
directly (no separate frontend build/server, no CORS issues).

The first analysis request triggers a one-time download of the pretrained
models from Hugging Face Hub (cached under `~/.cache/huggingface` afterwards).
This requires an internet connection on first run only.

---

## API

### `GET /api/health`
Loads (or confirms already-loaded) models and reports which model IDs are
actually active and which compute device is in use:

```json
{
  "status": "ok",
  "visual_model": "prithivMLmods/Deep-Fake-Detector-v2-Model",
  "audio_model": "MelodyMachine/Deepfake-audio-detection-V2",
  "device": "cpu"
}
```

### `POST /api/analyze`
Multipart form upload, field name `file`. Returns:

```json
{
  "overall_score": 63.4,
  "visual_subscore": 71.2,
  "audio_subscore": 50.5,
  "visual": {
    "status": "assessed",
    "score": 71.2,
    "model_used": "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "detail": "18/24 sampled frames had a detectable face (24 frames scored). Frame-to-frame consistency nudged the score by +2.1 points (weight 15%).",
    "temporal_consistency": {
      "coefficient_of_variation": 0.74,
      "inconsistency_score": 0.62,
      "note": "Heuristic signal: higher values mean more erratic frame-to-frame change than typical natural video motion + compression."
    },
    "saliency": {
      "method": "occlusion saliency (4x4 grid)",
      "baseline_fake_probability": 71.2,
      "image_base64": "data:image/png;base64,iVBORw0KG...",
      "note": "Heuristic, post-hoc explanation: brighter/red regions are where masking that part of the image reduced the model's fake-probability the most for THIS input — i.e. the regions it relied on most. Not an architecture-level explanation."
    }
  },
  "audio": {
    "status": "assessed",
    "score": 50.5,
    "model_used": "MelodyMachine/Deepfake-audio-detection-V2",
    "detail": "Scored 5.8s of audio at 16000 Hz (RMS=0.0421).",
    "timeline": [
      { "start_seconds": 0.0, "end_seconds": 1.45, "score": 41.2 },
      { "start_seconds": 1.45, "end_seconds": 2.9, "score": 58.7 },
      { "start_seconds": 2.9, "end_seconds": 4.35, "score": 53.0 },
      { "start_seconds": 4.35, "end_seconds": 5.8, "score": 49.1 }
    ],
    "raw_model_output": [
      { "label": "bona-fide", "score": 0.495 },
      { "label": "spoof", "score": 0.505 }
    ]
  },
  "confidence_band": "high",
  "fusion_method": "Weighted average: 60% visual + 40% audio",
  "frames_analyzed": 24,
  "duration_seconds": 5.8,
  "warnings": []
}
```

---

## Sample run (documented end-to-end check)

A synthetic test clip was generated with ffmpeg's `testsrc`/`sine` filters
(no faces, a pure tone — i.e. a clip that exercises the "no face detected"
and "no speech" edge cases) and posted to a locally running instance:

```bash
ffmpeg -y -f lavfi -i "testsrc=duration=4:size=320x240:rate=10" \
       -f lavfi -i "sine=frequency=440:duration=4" \
       -c:v libx264 -c:a aac -shortest sample.mp4

curl -s -X POST http://127.0.0.1:8000/api/analyze -F "file=@sample.mp4;type=video/mp4"
```

Observed pipeline behavior (verified during development):

* `ffprobe` correctly read duration `4.0s`, `has_audio=true`.
* `ffmpeg` extracted 23 evenly spaced frames.
* No faces were found in the synthetic test pattern → the visual model fell
  back to scoring full frames, and `visual.detail` reported
  `"No face detected in any of the 24 sampled frames; scored full frames instead"`.
* The clip's audio was a constant tone (not silence, not speech) → the audio
  model still ran and was reported as `"assessed"` (a real speech clip would
  exercise the same code path with a meaningful synthetic-speech score).
* A 1-second no-audio clip produced `audio.status = "not_assessed"`,
  `reason = "The video has no audio track."`, and a
  `"Video is very short (1.0s); results may be based on very little signal."`
  warning — confirming the very-short-video and no-audio edge cases are
  handled without crashing.
* An intentionally corrupted `.mp4` (text bytes renamed to `.mp4`) returned
  HTTP `422` with `"Could not read video file: ... moov atom not found"`
  rather than crashing the server.
* `GET /api/health` returns `200` and reports the active device
  (`cpu`/`cuda`) and the model IDs that successfully loaded (or, on load
  failure, the fallback that was attempted and the resulting error — the
  modality is then marked `"not_assessed"` rather than crashing).
* `temporal_consistency` / `saliency` / `timeline` were validated with
  standalone unit checks (mocked scoring functions, since real model weights
  require Hugging Face Hub access): a synthetic "steady camera pan" sequence
  produced `coefficient_of_variation ≈ 0.0` (inconsistency `0.08`), while a
  "bursty flicker" sequence (mostly-static frames punctuated by sudden
  region-wide jumps — modeling blending-boundary artifacts) produced
  `coefficient_of_variation ≈ 1.10` (inconsistency `0.88`), confirming the
  heuristic separates steady natural motion from erratic frame-to-frame change
  as intended. The occlusion-saliency and audio-timeline builders were
  similarly confirmed to produce valid base64 PNG overlays and per-segment
  score lists end-to-end.

> Note: actual `visual_subscore`/`audio_subscore` numbers depend on the real
> pretrained model weights, which are downloaded from Hugging Face Hub on
> first run — they are not bundled with this repo (per the "no manual dataset
> downloads, inference only" requirement).

---

## A note on saturated audio scores (e.g. "always near 100")

Label sets returned by `transformers` audio-classification pipelines vary in
spelling/punctuation/casing across model versions ("Bona-Fide" vs "bonafide"
vs "LABEL_0: real", …). `audio_model.fake_probability` now **normalizes**
labels (lowercases and strips all non-alphanumeric characters) before the
substring match, so a label like `"Bona-Fide"` correctly matches the
`"bonafide"` keyword instead of silently falling through to the fake-side
keywords and producing a near-100% reading regardless of what the model
actually predicted (the same fix was mirrored into `visual_model.py` for
consistency).

Beyond label-matching bugs, a persistently saturated reading can also be a
genuine **calibration problem**: small fine-tuned speech-deepfake classifiers
are trained on narrow datasets (often studio-quality TTS/voice-clone samples
vs. genuine recordings) and can generalize poorly to compressed, resampled,
background-noisy real-world video audio — confidently mislabeling ordinary
speech as "spoof". To make this diagnosable without needing to read server
logs:
- `audio.raw_model_output` now exposes the exact `{label, score}` pairs the
  model returned for the clip (also rendered in the UI's "Where in the clip?"
  panel), so you can see precisely how the 0–100 number was derived; and
- `audio.detail` appends an explicit caution note whenever the score lands
  at the extreme ends of the scale (≥ 97 or ≤ 3), reminding you that a
  confidently-saturated reading on ordinary audio is more likely a
  calibration artifact of a narrowly-trained model than a verdict.

---

## Edge cases handled

| Case | Behavior |
|------|----------|
| No face detected in any frame | Falls back to scoring full frames; `visual.detail` says so |
| No audio track | `audio.status = "not_assessed"`, `reason = "The video has no audio track."` |
| Silent / near-silent audio (incl. music-only, via a cheap RMS check) | `audio.status = "not_assessed"`, `reason` explains it was skipped |
| Corrupt / unreadable file | `422 Unprocessable Entity` with a clear ffprobe-derived message; never crashes |
| Very short video (< 2s) | Still analyzed; a warning is added (`warnings: [...]`) |
| Oversized file (> 100MB) or too long (> 60s) | `413 Payload Too Large` with a clear message, rejected before any model runs |
| Wrong file type | `400 Bad Request` listing allowed extensions |
| Model fails to load | Falls back to the documented alternative; if both fail, the modality is `"not_assessed"` with the load error message — never crashes |

---

## Disclaimer

The score reflects the **likelihood that the video matches generation
patterns the underlying detectors were trained to recognize** — it is
**not proof** that a video is real or AI-generated. These detectors
generalize poorly to generation methods they have never seen, and confident-
looking numbers can still be wrong in either direction. **Do not treat the
0–100 number as authoritative ground truth.** This disclaimer is shown
prominently in the UI on every result.
