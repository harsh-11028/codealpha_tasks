"""
utils/validators.py — Input validation utilities for audio files and API requests.
"""

import os
import io
from pathlib import Path
from typing import Optional
import filetype

from utils.logger import get_logger

logger = get_logger(__name__)

# Allowed MIME types for audio uploads
ALLOWED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",        # MP3
    "audio/mp3",
    "audio/ogg",
    "audio/x-m4a",
    "audio/mp4",
    "audio/aac",
    "audio/flac",
    "audio/x-flac",
    # Browser MediaRecorder output formats
    "audio/webm",
    "video/webm",        # Chrome labels webm recordings as video/webm
    "audio/webm;codecs=opus",
    "audio/ogg;codecs=opus",
}

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".m4a", ".aac", ".flac", ".webm"}

MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024
MAX_AUDIO_DURATION_SEC = int(os.getenv("MAX_AUDIO_DURATION", "30"))


def validate_audio_file(
    file_bytes: bytes,
    filename: str,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> tuple[bool, str]:
    """
    Validate an uploaded audio file.

    Returns:
        (is_valid, error_message) — error_message is empty string if valid.
    """
    # 1. Size check
    size = len(file_bytes)
    if size == 0:
        return False, "File is empty."
    if size > max_size_bytes:
        mb = max_size_bytes / (1024 * 1024)
        return False, f"File exceeds maximum size of {mb:.0f} MB."

    # 2. Extension check
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(ALLOWED_EXTENSIONS)
        return False, f"File extension '{ext}' not allowed. Accepted: {allowed}"

    # 3. MIME type check (magic bytes — prevents extension spoofing)
    # webm files may not have magic bytes detectable by filetype, so check extension first
    if ext == ".webm":
        logger.info(f"Audio validation passed (webm): {filename} ({size / 1024:.1f} KB)")
        return True, ""

    kind = filetype.guess(file_bytes)
    if kind is None:
        # Some valid audio formats (e.g., raw webm/opus) aren't recognised by filetype.
        # Fall back to extension-only validation for known browser recording types.
        logger.warning(f"filetype could not detect MIME for {filename}, accepting on extension.")
        return True, ""
    # Strip codec params (e.g. 'audio/webm;codecs=opus' → 'audio/webm') for matching
    mime_base = kind.mime.split(";")[0].strip()
    if mime_base not in ALLOWED_MIME_TYPES:
        return False, f"File MIME type '{kind.mime}' is not an accepted audio format."

    # 4. Content sanity — very basic header check
    # WAV files start with RIFF
    # MP3 files start with ID3 or 0xFF 0xFB
    # OGG files start with OggS
    if ext == ".wav" and not file_bytes[:4] == b"RIFF":
        return False, "Invalid WAV file header."

    logger.info(f"Audio validation passed: {filename} ({size / 1024:.1f} KB, {kind.mime})")
    return True, ""


def sanitize_filename(filename: str) -> str:
    """
    Strip dangerous characters from filename, keeping only safe chars.
    """
    import re
    stem = Path(filename).stem
    ext = Path(filename).suffix.lower()
    safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)
    safe_stem = safe_stem[:64]  # max length
    return f"{safe_stem}{ext}"
