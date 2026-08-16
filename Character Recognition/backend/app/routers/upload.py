"""
Upload routers — POST /upload, POST /webcam.
"""

from __future__ import annotations

import base64
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.database import get_db
from backend.app.middleware.security import rate_limit_middleware, validate_image_upload
from backend.app.schemas.prediction import UploadResponse, WebcamRequest

router = APIRouter(prefix="/api", tags=["upload"])
settings = get_settings()


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload an image for OCR",
)
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(default_factory=lambda: str(uuid.uuid4())),
    db: Session = Depends(get_db),
):
    """
    Upload an image and receive a session_id + image_url.
    Use the session_id in subsequent predict calls.
    """
    rate_limit_middleware(request)

    filename = file.filename or "upload.png"
    content_type = file.content_type or "image/png"

    image_bytes = await file.read()
    validate_image_upload(content_type, len(image_bytes), filename)

    # Save with unique name
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
    unique_name = f"{session_id}.{ext}"
    save_path = settings.upload_path / unique_name
    save_path.write_bytes(image_bytes)

    image_url = f"/api/images/{unique_name}"
    return UploadResponse(
        session_id=session_id,
        image_url=image_url,
        filename=filename,
        size_bytes=len(image_bytes),
    )


@router.get(
    "/images/{filename}",
    summary="Retrieve an uploaded image by filename",
)
async def get_image(filename: str):
    """Serve a previously uploaded image file."""
    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name
    file_path = settings.upload_path / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(str(file_path))


@router.post(
    "/webcam",
    response_model=UploadResponse,
    summary="Upload a base64-encoded webcam frame",
)
async def upload_webcam(
    request: Request,
    payload: WebcamRequest,
    db: Session = Depends(get_db),
):
    """
    Accept a base64-encoded image from webcam capture.
    Returns session_id + image_url same as /upload.
    """
    rate_limit_middleware(request)

    session_id = payload.session_id or str(uuid.uuid4())

    try:
        image_bytes = base64.b64decode(payload.image_base64)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image data.",
        )

    # Save webcam frame as PNG
    filename = f"{session_id}_webcam.png"
    save_path = settings.upload_path / filename
    save_path.write_bytes(image_bytes)

    return UploadResponse(
        session_id=session_id,
        image_url=f"/api/images/{filename}",
        filename="webcam_capture.png",
        size_bytes=len(image_bytes),
    )
