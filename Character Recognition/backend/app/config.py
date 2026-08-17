"""
FastAPI application configuration using pydantic-settings.
Reads from environment variables and .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "AI Handwritten OCR System"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = True

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8002
    api_reload: bool = True
    allowed_origins: str = "http://localhost:5175,http://127.0.0.1:5175"

    # Database
    database_url: str = "sqlite:///./ocr_system.db"

    # Uploads
    upload_dir: str = "backend/uploads"
    max_upload_size_mb: int = 10
    allowed_extensions: str = "jpg,jpeg,png,bmp,tiff,webp"

    # ML
    models_dir: str = "models/saved_models"
    default_model: str = "auto"
    device: str = "auto"

    # OCR engines
    easyocr_enabled: bool = True
    tesseract_enabled: bool = True
    custom_model_enabled: bool = True
    easyocr_languages: str = "en"

    # Security
    secret_key: str = "change-me-in-production"
    rate_limit_requests: int = 100

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def allowed_extensions_set(self) -> set:
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",") if ext.strip()}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache()
def get_settings() -> Settings:
    return Settings()
