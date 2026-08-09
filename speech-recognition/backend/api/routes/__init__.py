"""api/routes package — exports all router modules."""
from api.routes import health, metrics, model_info, predict, upload

__all__ = ["health", "metrics", "model_info", "predict", "upload"]
