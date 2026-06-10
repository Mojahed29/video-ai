"""FastAPI application: ``/api/analyze`` and ``/api/health`` plus the static UI."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import model_runtime
from .pipeline import VideoTooLargeError, VideoTooLongError, analyze_video
from .schemas import AnalyzeResponse, HealthResponse
from .settings import settings
from .video_utils import UnreadableVideoError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("video_ai.api")

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Video AI Detector",
        version="2.0.0",
        description=(
            "Estimates how likely an uploaded video is AI-generated or face-swapped "
            "by scoring its visual and audio tracks separately, then fusing them. "
            "Heuristic signal, not a verdict."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        info = model_runtime.warm_up()
        return HealthResponse(
            status="ok",
            visual_model=info.get("visual_model", "unknown"),
            audio_model=info.get("audio_model", "unknown"),
            device=info.get("device", "unknown"),
        )

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{suffix or 'unknown'}'. "
                    f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
                ),
            )

        tmp_dir = Path(tempfile.mkdtemp(prefix="video_ai_upload_"))
        tmp_path = tmp_dir / f"upload{suffix}"
        size = await _save_upload(file, tmp_path, tmp_dir)

        try:
            return analyze_video(tmp_path, declared_size_bytes=size)
        except (VideoTooLargeError, VideoTooLongError) as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except UnreadableVideoError as exc:
            raise HTTPException(status_code=422, detail=f"Could not read video file: {exc}") from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error analyzing %s", file.filename)
            raise HTTPException(
                status_code=500, detail=f"Unexpected error while analyzing video: {exc}"
            ) from exc
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    _mount_frontend(app)
    return app


async def _save_upload(file: UploadFile, tmp_path: Path, tmp_dir: Path) -> int:
    """Stream the upload to disk, enforcing the size limit. Returns bytes written."""
    size = 0
    try:
        with tmp_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_file_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.max_file_size_mb} MB limit.",
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
    return size


def _mount_frontend(app: FastAPI) -> None:
    """Serve the static frontend from the same origin (avoids CORS in normal use)."""
    if not FRONTEND_DIR.exists():
        return
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))


app = create_app()
