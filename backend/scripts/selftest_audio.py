#!/usr/bin/env python3
"""
Audio self-test / pass-fail gate for the deepfake-audio scorer.

Scores a set of labelled clips through the REAL pipeline (the configured model
+ VAD + temperature + calibration) and prints the raw and calibrated scores for
each. The gate: a genuine human-speech clip — including an ARABIC one — must NOT
read as confidently fake.

Clips are discovered from a directory (``--clips-dir`` or
``$VIDEO_AI_SELFTEST_CLIPS``); filenames starting with ``genuine`` are treated
as real speech and ``ai``/``fake`` as synthetic. With ``--make-clips`` the
script will best-effort generate them on macOS (``say`` for TTS) and download a
real Arabic human clip from Wikimedia Commons.

Usage:
    cd backend
    python scripts/selftest_audio.py --make-clips
    python scripts/selftest_audio.py --clips-dir /path/to/clips
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.audio_model import score_audio  # noqa: E402
from app.settings import settings  # noqa: E402

# A genuine clip must come in below this calibrated score to pass.
GENUINE_MAX = 60.0
UA = {"User-Agent": "video-ai-selftest/1.0 (research)"}


def _ffmpeg(src: Path, dst: Path, seconds: int = 8) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(settings.audio_sample_rate),
             "-t", str(seconds), str(dst), "-loglevel", "error"],
            check=True, timeout=60,
        )
        return dst.exists()
    except Exception as exc:  # noqa: BLE001
        print(f"  ffmpeg failed for {src.name}: {exc}")
        return False


def make_clips(out: Path) -> None:
    """Best-effort generation of test clips (macOS + network required)."""
    out.mkdir(parents=True, exist_ok=True)
    # AI / TTS clips via macOS `say` (English + Arabic if the voices exist).
    tts = [("Daniel", "ai_en_tts", "This is a synthetic voice produced by a text to speech engine."),
           ("Majed", "ai_ar_tts", "مرحبا، هذا صوت اصطناعي تم إنشاؤه بواسطة محرك تحويل النص إلى كلام.")]
    for voice, name, text in tts:
        aiff = out / f"{name}.aiff"
        try:
            subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True, timeout=60)
            _ffmpeg(aiff, out / f"{name}.wav")
            aiff.unlink(missing_ok=True)
            print(f"  generated {name}.wav (voice={voice})")
        except Exception as exc:  # noqa: BLE001
            print(f"  could not generate {name} (voice {voice}): {exc}")
    # Genuine human Arabic from Wikimedia Commons.
    try:
        _download_commons_arabic(out / "genuine_ar_human.wav")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not fetch genuine Arabic clip: {exc}")


def _commons_api(params: dict) -> dict:
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"})
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40))


def _download_commons_arabic(dst: Path) -> None:
    members = _commons_api({"action": "query", "list": "categorymembers",
                            "cmtitle": "Category:Audio files in Arabic", "cmtype": "file", "cmlimit": "40"})
    for c in members["query"]["categorymembers"]:
        info = _commons_api({"action": "query", "titles": c["title"], "prop": "imageinfo",
                             "iiprop": "url|mediatype"})
        page = next(iter(info["query"]["pages"].values()))
        ii = page["imageinfo"][0]
        if ii.get("mediatype") not in (None, "AUDIO"):
            continue
        raw = dst.with_suffix(".src")
        data = urllib.request.urlopen(urllib.request.Request(ii["url"], headers=UA), timeout=90).read()
        raw.write_bytes(data)
        if _ffmpeg(raw, dst, seconds=10):
            raw.unlink(missing_ok=True)
            print(f"  fetched genuine Arabic clip: {c['title']}")
            return
    raise RuntimeError("no suitable Arabic clip found")


def label_of(path: Path) -> str:
    n = path.stem.lower()
    if n.startswith("genuine") or n.startswith("real"):
        return "genuine"
    if n.startswith(("ai", "fake", "tts", "synthetic")):
        return "ai"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", default=os.environ.get("VIDEO_AI_SELFTEST_CLIPS"))
    ap.add_argument("--make-clips", action="store_true")
    args = ap.parse_args()

    clips_dir = Path(args.clips_dir) if args.clips_dir else BACKEND_DIR / "scripts" / "selftest_clips"
    if args.make_clips:
        make_clips(clips_dir)
    if not clips_dir.exists():
        print(f"No clips dir at {clips_dir}. Run with --make-clips or set --clips-dir.")
        return 2

    wavs = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"})
    if not wavs:
        print(f"No audio clips in {clips_dir}.")
        return 2

    print(f"\nModel: {settings.audio_model_id}")
    print(f"Temperature: {settings.audio_calibration_temperature} | VAD: {settings.use_vad}\n")
    print(f"{'clip':30} {'label':8} {'status':13} {'raw%':>6} {'calib%':>7}")
    print("-" * 72)

    failures = []
    tmp = clips_dir / "_normalized"
    tmp.mkdir(exist_ok=True)
    for wav in wavs:
        label = label_of(wav)
        # Normalize to 16 kHz mono wav (mirrors the real pipeline's ffmpeg step),
        # so mp3/m4a/etc. are handled the same way an uploaded video would be.
        norm = tmp / (wav.stem + ".wav")
        r = score_audio(norm if _ffmpeg(wav, norm, seconds=30) else wav)
        raw = f"{r.raw_score:.1f}" if r.raw_score is not None else "—"
        calib = f"{r.score:.1f}" if r.score is not None else "—"
        print(f"{wav.name:30} {label:8} {r.status:13} {raw:>6} {calib:>7}"
              + (f"   ({r.reason})" if r.reason else ""))
        if label == "genuine" and r.status == "assessed" and r.score is not None and r.score >= GENUINE_MAX:
            failures.append((wav.name, r.score))

    print()
    if failures:
        for name, sc in failures:
            print(f"FAIL: genuine clip '{name}' read as {sc:.1f}% fake (>= {GENUINE_MAX}).")
        print("\nGATE FAILED — the model still saturates on genuine speech. Try the next model.")
        return 1
    print(f"GATE PASSED — all genuine clips scored below {GENUINE_MAX}% fake.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
