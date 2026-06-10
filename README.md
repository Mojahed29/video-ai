# Video AI Detector

Estimate how likely an uploaded video is **AI-generated or face-swapped** by
scoring its **visual** and **audio** tracks *separately* with pretrained open
models, then fusing both probabilities into a single 0–100 score — with the
supporting evidence shown over time.

> ⚠️ **This is an estimate, not proof.** It is a heuristic signal, not a verdict.
> See [Honest limitations](#honest-limitations).

FastAPI backend (inference only — no training) + a framework-light, fully
responsive, accessible frontend themed with the **Telecommunications Regulatory
Authority (TRA) of Oman** brand palette.

---

## Table of contents

- [What it does](#what-it-does)
- [Models used](#models-used)
- [Quick start (Docker)](#quick-start-docker)
- [Local development](#local-development)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [The interface](#the-interface)
- [Design notes & color tokens](#design-notes--color-tokens)
- [Explainability & calibration](#explainability--calibration)
- [Edge cases handled](#edge-cases-handled)
- [Tests](#tests)
- [Project structure](#project-structure)
- [Honest limitations](#honest-limitations)

---

## What it does

```
upload (mp4/mov/webm/mkv, ≤100 MB, ≤60 s)
   │
   ├─ ffprobe: duration, has-audio?, corrupt-file check
   │
   ├─ ffmpeg → N evenly spaced JPEG frames (default 24)
   │     └─ MTCNN face detection per frame
   │           ├─ face(s) found → crop & run the visual classifier on each face
   │           └─ no face       → run the visual classifier on the full frame
   │     └─ VISUAL sub-score = mean P(fake) over scored frames        → 0–100
   │        (+ per-frame timeline for the heat strip)
   │
   ├─ ffmpeg → mono 16 kHz WAV
   │     └─ RMS silence check (skip the model if the track is silent)
   │     └─ AUDIO sub-score = P(synthetic speech) over the whole clip  → 0–100
   │        (+ per-window timeline; "not assessed" if no/silent audio)
   │
   └─ FUSION: weighted average (default 60% visual / 40% audio)
         · one modality missing → use the other alone, and say so
         · confidence band: high (both) / medium (one) / low (neither)
         · optional uncertainty-aware calibration (off by default)
```

The API returns the `overall_score`, both sub-scores, per-modality status, the
exact model IDs used (including whether a fallback model had to be loaded), the
number of frames analyzed, a confidence band, and per-frame / per-window
timelines for the explainability panels.

---

## Models used

Chosen for being well-downloaded, maintained, binary real/fake classifiers that
load directly through `transformers.pipeline` — no training required.

| Track | Primary model | Fallback model |
|-------|---------------|----------------|
| **Visual** (deepfake / synthetic image) | [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model) — ViT fine-tuned for binary Real/Fake image classification | [`prithivMLmods/deepfake-detector-model-v1`](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1) |
| **Audio** (synthetic / spoofed speech) | [`mo-thecreator/Deepfake-audio-detection`](https://huggingface.co/mo-thecreator/Deepfake-audio-detection) — Wav2Vec2-based binary Real/Fake speech classifier (better calibrated than the V2 fine-tune) | [`MelodyMachine/Deepfake-audio-detection-V2`](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2) |
| **Face localization** | [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) MTCNN — *not* a deepfake classifier, only crops faces for the visual model | — |

If a primary model ID fails to download or load (pulled, renamed, unreachable),
the app **automatically falls back** to the documented alternative and the
API/UI report **which model actually ran** (`visual.model_used`,
`audio.model_used`, and `/api/health`). If both fail to load, that modality is
reported as `"not_assessed"` with the load error — the app does not crash.

**Swapping models** is a config change only (see [Configuration](#configuration)).
Labels are matched by substring (`fake`/`real`, `spoof`/`bonafide`,
`synthetic`/`authentic`, …) in [`backend/app/labels.py`](backend/app/labels.py),
so most binary real-vs-fake classifiers work without code changes.

---

## Quick start (Docker)

The image bundles **ffmpeg** and serves the backend + frontend together.

```bash
docker compose up --build
```

Then open **http://localhost:8000**.

The first analysis downloads the pretrained models from the Hugging Face Hub
(cached in the `hf-models` volume, so restarts are fast). This needs internet
on first run only. The image uses the **CPU build of PyTorch** by default.

---

## Local development

### 1. Install ffmpeg (required)

| OS | Command |
|----|---------|
| Ubuntu / Debian | `sudo apt-get update && sudo apt-get install -y ffmpeg` |
| macOS (Homebrew) | `brew install ffmpeg` |
| Windows | Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add `bin/` to `PATH` |

Verify: `ffmpeg -version` and `ffprobe -version`.

### 2. Install Python dependencies (Python 3.11 recommended)

```bash
cd backend
pip install -r requirements.txt
```

> **Optional GPU acceleration:** install a CUDA build of
> `torch`/`torchvision`/`torchaudio` from
> [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
> *before* `pip install -r requirements.txt`. The app auto-detects CUDA via
> `torch.cuda.is_available()` and reports the active device at `/api/health`.

### 3. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — FastAPI serves the frontend on the same origin
(no separate frontend server, no CORS issues).

---

## Configuration

All settings are environment-driven via [`backend/app/settings.py`](backend/app/settings.py)
(pydantic-settings). Copy [`backend/.env.example`](backend/.env.example) to
`backend/.env` and override what you need. Every variable uses the `VIDEO_AI_`
prefix.

| Variable | Default | Purpose |
|----------|---------|---------|
| `VIDEO_AI_VISUAL_MODEL_ID` | `prithivMLmods/Deep-Fake-Detector-v2-Model` | Visual classifier |
| `VIDEO_AI_AUDIO_MODEL_ID` | `MelodyMachine/Deepfake-audio-detection-V2` | Audio classifier |
| `VIDEO_AI_*_FALLBACK_ID` | (documented alternates) | Used only on load failure |
| `VIDEO_AI_NUM_FRAMES` | `24` | Frames sampled per video |
| `VIDEO_AI_MAX_FILE_SIZE_MB` | `100` | Upload size limit |
| `VIDEO_AI_MAX_DURATION_SECONDS` | `60` | Duration limit |
| `VIDEO_AI_VISUAL_WEIGHT` / `_AUDIO_WEIGHT` | `0.6` / `0.4` | Fusion weights (normalized) |
| `VIDEO_AI_AUDIO_CALIBRATION_TEMPERATURE` | `1.5` | Soften over-confident audio scores (1.0 = off) |
| `VIDEO_AI_CALIBRATION_ENABLED` | `false` | Uncertainty-aware calibration (see below) |
| `VIDEO_AI_FORCE_CPU` | `false` | Ignore an available GPU |
| `VIDEO_AI_CORS_ALLOW_ORIGINS` | `localhost:8000, 127.0.0.1:8000` | Comma-separated origins (replaces `*`) |

CORS is restricted to the configured origins and `GET`/`POST` only. Because the
frontend is same-origin, the defaults are all you need for normal use; set real
origins only if you expose the API to other sites.

---

## API reference

### `GET /api/health`

Loads (or confirms) the models and reports which IDs are active and the compute
device:

```json
{
  "status": "ok",
  "visual_model": "prithivMLmods/Deep-Fake-Detector-v2-Model",
  "audio_model": "MelodyMachine/Deepfake-audio-detection-V2",
  "device": "cpu"
}
```

### `POST /api/analyze`

Multipart form upload, field name `file`. Returns an `AnalyzeResponse`:

```json
{
  "overall_score": 65.9,
  "visual_subscore": 70.8,
  "audio_subscore": 58.5,
  "visual": {
    "status": "assessed",
    "score": 70.8,
    "raw_score": null,
    "model_used": "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "detail": "18/24 sampled frames had a detectable face (24 frames scored).",
    "timeline": [{ "t_start": 1.2, "t_end": 1.2, "score": 64.0, "note": "face" }]
  },
  "audio": {
    "status": "assessed",
    "score": 58.5,
    "model_used": "MelodyMachine/Deepfake-audio-detection-V2",
    "detail": "Scored 8.0s of audio at 16000 Hz (RMS=0.0421).",
    "timeline": [{ "t_start": 0.0, "t_end": 2.0, "score": 32.0, "note": null }]
  },
  "confidence_band": "high",
  "fusion_method": "Weighted average: 60% visual + 40% audio",
  "calibrated": false,
  "frames_analyzed": 24,
  "duration_seconds": 8.0,
  "warnings": []
}
```

**Status codes:** `400` wrong file type · `413` too large / too long ·
`422` corrupt / unreadable · `500` unexpected. Each returns `{ "detail": "…" }`.

Interactive docs are available at `/docs` (Swagger) and `/redoc`.

---

## The interface

A single-page app with an explicit state machine (idle → selected → working →
success | error). All states are designed, not afterthoughts:

- **Idle** — branded dropzone with drag-and-drop, click, or keyboard selection
  and format/limit chips.
- **Selected** — a file card (name, size, type) with **Analyze** / **Choose a
  different file**; obviously-invalid files are rejected client-side before the
  ~1-minute upload.
- **Working** — a **real** upload progress bar followed by a staged analysis
  indicator (*Uploading → Extracting frames → Scoring visual → Scoring audio →
  Fusing results*) with a checklist that ticks off. Re-submission is disabled
  and a **Cancel** button aborts the request.
- **Success** — an arc **score gauge**, a plain-language verdict, the confidence
  band, the two sub-scores with meters and the model that produced each, and the
  explainability panels.
- **Errors & "not assessed"** — every error case has a tailored message; a
  modality that can't be scored shows a clear "Not assessed" state with the
  reason instead of a fake number.

**Accessibility:** semantic landmarks, ARIA labels, a keyboard-operable
dropzone, visible focus rings, `aria-live` result/progress regions, and respect
for `prefers-reduced-motion` and `forced-colors`. **Responsive:** mobile-first,
verified from 375 px phones to desktop with no horizontal scroll and ≥ 44 px tap
targets. The "estimate, not proof" disclaimer is shown prominently at all times.

---

## Design notes & color tokens

The theme is built on the **official TRA of Oman brand palette**, taken directly
from the design tokens published on the TRA website (`tra.gov.om`) — a deep
"TRA blue" with teal/cyan accents (not the green many assume from the flag).
These are codified as CSS custom properties in
[`frontend/css/tokens.css`](frontend/css/tokens.css).

| Token | Value | Role |
|-------|-------|------|
| `--tra-600` / `--brand` | `#234F9C` | Primary "TRA blue" |
| `--tra-700` … `--tra-900` | `#1d4282` `#19366b` `#142a54` | Strong / deep brand, headings |
| `--teal` / `--cyan` | `#00ABB1` / `#33DEF2` | Accents (decorative) |
| `--teal-text` | `#007E83` | AA-safe teal for text/icons |
| `--bg` (+ gradient) | `#eef3fb` | Soft, layered page background |
| `--surface` / `--surface-alt` | `#ffffff` / `#F8FCFF` | Cards |
| `--text` / `--text-body` / `--muted` | `#16223F` / `#2C3A59` / `#586A8C` | Text ramp |
| `--success` / `--warn` / `--danger` | `#0E7A52` / `#B45309` / `#C0362C` | Status + score scale |

Every text/background pair was checked for **WCAG-AA contrast** (≥ 4.5:1 for body
text). The design also uses a consistent spacing scale, type scale, radii,
shadow set, and motion tokens. The original theme — a bright-blue page
background behind near-black panels — has been replaced entirely with a coherent,
intentional light system.

> If the TRA brand is ever unavailable, the documented fallback is the Omani
> national flag palette: white `#FFFFFF`, red `#DB161B`, green `#008000`.

---

## Explainability & calibration

**Temporal evidence (honest).** The visual model already scores each sampled
frame and the audio model scores windows of the clip; those per-unit scores are
returned and rendered as:

- a **per-frame likelihood heat strip** — *where in time* the model's suspicion
  came from (this is a timeline heatmap, **not** a per-pixel saliency map); and
- a **windowed audio timeline** — how the synthetic-speech likelihood varies
  across the clip.

The overall sub-scores are unchanged by these views — they are explanations of
the same numbers, not new ones.

**Calibration (opt-in, off by default).** We deliberately do **not** ship a
fabricated calibration curve (that needs a labelled validation set we don't
have). Instead, `VIDEO_AI_CALIBRATION_ENABLED=true` enables *uncertainty-aware
shrinkage*: scores derived from little signal (few scored frames, very short
audio) are pulled toward the neutral 50 prior, so a "90% fake" from a single
frame is reported less confidently than the same number from many frames. When
it changes a score, the API exposes the pre-calibration value as `raw_score` and
the UI shows "raw → calibrated". With calibration off, the raw model numbers are
reported exactly. **No accuracy claim is implied in either mode.**

---

## Troubleshooting: audio always reads ~100% AI

Small Wav2Vec2 deepfake-audio detectors are notoriously **over-confident** on
ordinary recorded/compressed audio and can pin the audio sub-score near 100 on
genuine clips (which then drags the fused score up). This is a model-quality
limitation, not a bug in the scoring. Mitigations, in order of impact:

1. **Use the better-calibrated base model** (now the default):
   `VIDEO_AI_AUDIO_MODEL_ID=mo-thecreator/Deepfake-audio-detection`. The V2
   fine-tune is more saturated; it remains available as the fallback.
2. **Raise the softening temperature**, e.g. `VIDEO_AI_AUDIO_CALIBRATION_TEMPERATURE=2.5`.
   Temperature tempers borderline scores but, by design, only nudges extreme
   ones (a raw 0.999 is a very large logit) — so it complements, not replaces, a
   better model.
3. **Lean on the visual track / lower the audio weight**, e.g.
   `VIDEO_AI_VISUAL_WEIGHT=0.75` and `VIDEO_AI_AUDIO_WEIGHT=0.25`.
4. **Swap in any other** `audio-classification` real/fake model via
   `VIDEO_AI_AUDIO_MODEL_ID` — label matching is tolerant of naming variants.

The result now also shows the **raw vs. softened** audio score and flags when the
model's raw reading is near an extreme, so a saturated value is visible rather
than hidden.

---

## Edge cases handled

| Case | Behavior |
|------|----------|
| No face detected in any frame | Falls back to scoring full frames; `visual.detail` says so |
| No audio track | `audio.status = "not_assessed"`, reason explains it |
| Silent / near-silent / music-only audio (cheap RMS check) | `not_assessed`; the model is skipped |
| Corrupt / unreadable file | `422` with a clear ffprobe-derived message; never crashes |
| Very short video (< 2 s) | Still analyzed; a warning is added |
| Oversized (> 100 MB) or too long (> 60 s) | `413`, rejected before any model runs |
| Wrong file type | `400` listing allowed extensions (also caught client-side) |
| Model fails to load | Falls back to the alternate; if both fail, `not_assessed` with the error — never crashes |

---

## Tests

The suite mocks the model pipelines, so it runs **without** downloading any
weights (and skips the audio tests cleanly if `soundfile` isn't installed). The
corrupt-file test uses the real `ffprobe`.

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

Coverage: label → P(fake) matching, fusion math + confidence bands, calibration
math (identity + shrinkage), and pipeline/HTTP edge cases (no frames, silent /
empty / missing audio, model-load failure, unrecognized labels, and the
`400` / `413` / `422` responses).

---

## Project structure

```
video-ai/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app factory + /api routes + static UI
│   │   ├── settings.py        ★ env-driven config (model IDs, weights, limits, CORS)
│   │   ├── pipeline.py        end-to-end orchestration + edge-case handling
│   │   ├── video_utils.py     ffmpeg/ffprobe: probing, frame & audio extraction
│   │   ├── visual_model.py    face detection + visual scoring + per-frame timeline
│   │   ├── audio_model.py     silence check + audio scoring + windowed timeline
│   │   ├── labels.py          shared label → P(fake) mapping
│   │   ├── calibration.py     opt-in uncertainty-aware calibration
│   │   ├── fusion.py          weighted-average fusion + confidence band
│   │   ├── model_runtime.py   lazy model loading, device selection, fallback
│   │   └── schemas.py         pydantic response models
│   ├── tests/                 pytest suite (models mocked)
│   ├── requirements.txt       pinned deps (CPU torch by default)
│   ├── requirements-dev.txt   + pytest, httpx
│   ├── pyproject.toml         pytest config
│   └── .env.example
├── frontend/
│   ├── index.html             semantic, accessible single-page UI
│   ├── css/                   tokens.css · base.css · components.css
│   └── js/                    main · api · dropzone · progress · charts · results · …
├── Dockerfile                 single image, ffmpeg + CPU torch
├── docker-compose.yml         one-command run + model cache volume
└── README.md
```

---

## Honest limitations

The score reflects the **likelihood that the video matches generation patterns
the underlying detectors were trained to recognize** — it is **not proof** that
a video is real or AI-generated:

- These detectors **generalize poorly** to generation methods they have never
  seen. A brand-new model can fool them in either direction.
- The audio model assesses **speech**; music-only or silent tracks are reported
  as "not assessed", not as "real".
- Raw probabilities are **not calibrated to a ground-truth accuracy** (see the
  calibration note above).
- Confident-looking numbers can still be wrong. **Do not treat the 0–100 number
  as authoritative ground truth** — use it as one signal among many. This
  disclaimer is shown prominently in the UI on every result.
