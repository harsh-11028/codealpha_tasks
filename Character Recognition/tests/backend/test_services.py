"""
Unit tests for backend business logic services (OCRService & ExportService).
"""

import pytest
from backend.app.services.export_service import ExportService
from backend.app.services.model_service import ModelService


def test_export_service_txt():
    svc = ExportService()
    text = "Line 1: Neural OCR Recognition\nLine 2: Deep Learning Models"
    content, media_type, ext = svc.export(text, format="txt", title="Test Export")
    assert media_type == "text/plain; charset=utf-8"
    assert ext == ".txt"
    assert isinstance(content, bytes)
    assert content.decode("utf-8") == text


def test_export_service_pdf():
    svc = ExportService()
    text = "Sample Document Title\nRecognized handwritten paragraph output."
    content, media_type, ext = svc.export(text, format="pdf", title="OCR Report")
    assert media_type == "application/pdf"
    assert ext == ".pdf"
    assert isinstance(content, bytes)
    assert len(content) > 0


def test_export_service_docx():
    svc = ExportService()
    text = "Export to Microsoft Word DOCX format test."
    content, media_type, ext = svc.export(text, format="docx", title="Word Report")
    assert ext == ".docx"
    assert isinstance(content, bytes)
    assert len(content) > 0


def test_model_service():
    svc = ModelService()
    # verify model info methods return structured lists/strings without crashing
    assert isinstance(svc.get_active_model(), str)
    assert isinstance(svc.get_loaded_models(), list)
    assert isinstance(svc.get_model_info(), list)
