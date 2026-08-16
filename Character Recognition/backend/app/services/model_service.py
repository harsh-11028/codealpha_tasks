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
        pipeline = get_pipeline()
        if pipeline._predictor and pipeline._predictor.selector:
            return pipeline._predictor.selector.get_loaded_models()
        return []

    def get_model_info(self) -> List[Dict]:
        pipeline = get_pipeline()
        if pipeline._predictor:
            return pipeline._predictor.get_model_info()
        return []

    def get_active_model(self) -> str:
        loaded = self.get_loaded_models()
        return loaded[0] if loaded else "none"

    def is_ready(self) -> bool:
        return get_pipeline()._initialized
