# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Video AI Detector — single-image deployment.
# Serves the FastAPI backend + the static frontend from one container.
# Defaults to the CPU build of PyTorch (see the torch install step).
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

# System deps: ffmpeg/ffprobe (frame + audio extraction), libsndfile (soundfile).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    # Hugging Face download reliability: the Xet CDN can time out on large
    # files, so use the standard CDN with a longer per-request timeout.
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DOWNLOAD_TIMEOUT=60

WORKDIR /app/backend

# Install dependencies first (better layer caching).
COPY backend/requirements.txt ./requirements.txt
# CPU-only torch from the PyTorch CPU index keeps the image small. To use a GPU
# instead, build with --build-arg TORCH_INDEX=... or install a CUDA torch here.
RUN pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    # Remove the Xet transfer client if present: its CDN times out on large
    # files; without it huggingface_hub uses the reliable standard download.
    && pip uninstall -y hf_xet hf-xet 2>/dev/null || true

# Application code.
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# Model cache directory (mount a volume here to persist downloads).
RUN mkdir -p /models && useradd -m appuser && chown -R appuser /models /app
USER appuser

EXPOSE 8000

# Liveness check that does NOT trigger a model download (serves the SPA shell).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
