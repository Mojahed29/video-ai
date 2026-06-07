from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, model_runtime
from .pipeline import VideoTooLargeError, VideoTooLongError, analyze_video
from .schemas import HealthResponse
from .video_utils import UnreadableVideoError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("video_ai.api")

app = FastAPI(title="Video AI Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska"}
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    info = model_runtime.warm_up()
    return HealthResponse(
        status="ok",
        visual_model=info.get("visual_model", "unknown"),
        audio_model=info.get("audio_model", "unknown"),
        device=info.get("device", "unknown"),
    )


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or 'unknown'}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="video_ai_upload_"))
    tmp_path = tmp_dir / f"upload{suffix}"

    size = 0
    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    try:
        with tmp_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {config.MAX_FILE_SIZE_MB} MB limit.",
                    )
                out.write(chunk)
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Could not read upload: {exc}") from exc
    finally:
        await file.close()

    try:
        result = analyze_video(tmp_path, declared_size_bytes=size)
        return result
    except VideoTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except VideoTooLongError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnreadableVideoError as exc:
        raise HTTPException(status_code=422, detail=f"Could not read video file: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error analyzing %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Unexpected error while analyzing video: {exc}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Static frontend (served from the same origin to avoid CORS in normal use) ---
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
