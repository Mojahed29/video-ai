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
   │     └─ VISUAL sub-score = mean P(fake) over scored frames   →  0–100
   │
   ├─ ffmpeg → mono 16kHz WAV
   │     └─ RMS silence check (skip model if track is silent/near-silent)
   │     └─ AUDIO sub-score = P(synthetic speech)                →  0–100
   │           (or "not assessed" if no audio / silent / model failure)
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
`"synthetic"/"authentic"`, …) in `visual_model._fake_probability` /
`audio_model._fake_probability`, so most binary real-vs-fake classifiers work
without further changes.

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
│   │   ├── fusion.py          weighted-average fusion + confidence band
│   │   ├── model_runtime.py   lazy model loading, device selection, fallback
│   │   └── schemas.py         pydantic response models
│   └── requirements.txt
├── frontend/
│   ├── index.html             single-page UI (drag & drop, results card)
│   ├── style.css
│   └── app.js                 upload, progress, result rendering
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
    "detail": "18/24 sampled frames had a detectable face (24 frames scored)."
  },
  "audio": {
    "status": "assessed",
    "score": 50.5,
    "model_used": "MelodyMachine/Deepfake-audio-detection-V2",
    "detail": "Scored 5.8s of audio at 16000 Hz (RMS=0.0421)."
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

> Note: actual `visual_subscore`/`audio_subscore` numbers depend on the real
> pretrained model weights, which are downloaded from Hugging Face Hub on
> first run — they are not bundled with this repo (per the "no manual dataset
> downloads, inference only" requirement).

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
