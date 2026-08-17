from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel

class PredictionModel(BaseModel):
    user_id: str
    disease: str
    input_data: Dict[str, Any]
    prediction: int
    probability: float
    model_used: str
    created_at: datetime = datetime.utcnow()
