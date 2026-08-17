from fastapi import APIRouter, Depends, HTTPException, Query
from app.schemas.response import APIResponse
from app.middleware.auth_middleware import get_current_user
from app.database import get_database
from bson import ObjectId
from typing import Optional, List

router = APIRouter(prefix="/predictions", tags=["history"])

@router.get("", response_model=APIResponse)
async def get_predictions(
    disease: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    query = {"user_id": current_user["id"]}
    if disease:
        query["disease"] = disease
        
    cursor = db.predictions.find(query).sort("created_at", -1).skip(skip).limit(limit)
    predictions = await cursor.to_list(length=limit)
    
    # Format for response
    for p in predictions:
        p["id"] = str(p.pop("_id"))
        
    total = await db.predictions.count_documents(query)
    
    return APIResponse(
        success=True,
        message="Predictions retrieved successfully",
        data={
            "predictions": predictions,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    )

@router.get("/{id}", response_model=APIResponse)
async def get_prediction(
    id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    try:
        prediction = await db.predictions.find_one({"_id": ObjectId(id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid prediction ID format")
        
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
        
    if prediction["user_id"] != current_user["id"] and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to access this prediction")
        
    prediction["id"] = str(prediction.pop("_id"))
    
    return APIResponse(
        success=True,
        message="Prediction retrieved successfully",
        data=prediction
    )

@router.delete("/{id}", response_model=APIResponse)
async def delete_prediction(
    id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    try:
        prediction = await db.predictions.find_one({"_id": ObjectId(id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid prediction ID format")
        
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
        
    if prediction["user_id"] != current_user["id"] and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this prediction")
        
    await db.predictions.delete_one({"_id": ObjectId(id)})
    
    return APIResponse(
        success=True,
        message="Prediction deleted successfully"
    )
