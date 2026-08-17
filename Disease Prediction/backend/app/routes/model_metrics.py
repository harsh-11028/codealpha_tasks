import json
import os
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.response import APIResponse
from app.middleware.auth_middleware import get_current_user
from app.config import settings

router = APIRouter(prefix="/models", tags=["model metrics"])

@router.get("", response_model=APIResponse)
async def list_models(current_user: dict = Depends(get_current_user)):
    diseases = ["heart", "diabetes", "breast_cancer"]
    return APIResponse(
        success=True,
        message="Available models retrieved",
        data={"diseases": diseases}
    )

@router.get("/{disease}/performance", response_model=APIResponse)
async def get_model_performance(disease: str, current_user: dict = Depends(get_current_user)):
    if disease not in ["heart", "diabetes", "breast_cancer"]:
        raise HTTPException(status_code=404, detail="Disease not found")
        
    metrics_path = os.path.join(settings.ML_MODELS_PATH, disease, "all_metrics.json")
    
    if not os.path.exists(metrics_path):
        return APIResponse(
            success=False,
            message="Metrics not available for this disease yet.",
            data=[]
        )
        
    try:
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
            
        return APIResponse(
            success=True,
            message=f"Model metrics for {disease} retrieved",
            data=metrics_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {str(e)}")
