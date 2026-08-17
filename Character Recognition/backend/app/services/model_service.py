"""
Model service — manages ML model loading and info for the API.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from models.ocr.pipeline import get_pipeline
from models.utils.model_selector import MODEL_REGISTRY

logger = logging.getLogger(__name__)


class ModelService:
    """Provides model information and management for the API."""

    def get_loaded_models(self) -> List[str]:
        from models.ocr.pipeline import _default_pipeline
        if _default_pipeline is None or not _default_pipeline._initialized:
            return []
        if _default_pipeline._predictor and _default_pipeline._predictor.selector:
            return _default_pipeline._predictor.selector.get_loaded_models()
        return []

    def get_model_info(self) -> List[Dict]:
        from models.ocr.pipeline import _default_pipeline
        if _default_pipeline is None or not _default_pipeline._initialized:
            return []
        if _default_pipeline._predictor:
            return _default_pipeline._predictor.get_model_info()
        return []

    def get_active_model(self) -> str:
        loaded = self.get_loaded_models()
        return loaded[0] if loaded else "none"

    def is_ready(self) -> bool:
        from models.ocr.pipeline import _default_pipeline
        return _default_pipeline is not None and _default_pipeline._initialized
