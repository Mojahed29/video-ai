"""Settings parsing — especially CORS origins from env (regression: Docker crash)."""
import pytest

from app.settings import Settings


def test_cors_origins_default():
    s = Settings()
    assert s.cors_origins == ["http://localhost:8000", "http://127.0.0.1:8000"]


def test_cors_origins_from_comma_separated_string():
    # This is exactly what docker-compose passes; it used to crash on a json.loads().
    s = Settings(cors_allow_origins="http://localhost:8000,http://127.0.0.1:8000")
    assert s.cors_origins == ["http://localhost:8000", "http://127.0.0.1:8000"]


def test_cors_origins_single_value():
    s = Settings(cors_allow_origins="https://only.example")
    assert s.cors_origins == ["https://only.example"]


def test_cors_origins_from_json_list():
    s = Settings(cors_allow_origins='["https://a.com", "https://b.com"]')
    assert s.cors_origins == ["https://a.com", "https://b.com"]


def test_cors_origins_empty_string():
    s = Settings(cors_allow_origins="")
    assert s.cors_origins == []


def test_cors_origins_strips_whitespace():
    s = Settings(cors_allow_origins=" https://a.com , https://b.com ")
    assert s.cors_origins == ["https://a.com", "https://b.com"]


def test_cors_env_var_is_read(monkeypatch):
    monkeypatch.setenv("VIDEO_AI_CORS_ALLOW_ORIGINS", "https://x.com,https://y.com")
    s = Settings()
    assert s.cors_origins == ["https://x.com", "https://y.com"]


def test_max_file_size_bytes():
    assert Settings(max_file_size_mb=10).max_file_size_bytes == 10 * 1024 * 1024
