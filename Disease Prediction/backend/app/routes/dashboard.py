from fastapi import APIRouter, Depends
from app.schemas.response import APIResponse
from app.middleware.auth_middleware import get_current_user
from app.database import get_database

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

@router.get("/stats", response_model=APIResponse)
async def get_dashboard_stats(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    user_id = current_user["id"]
    
    # Run aggregations concurrently or sequentially
    total = await db.predictions.count_documents({"user_id": user_id})
    heart_count = await db.predictions.count_documents({"user_id": user_id, "disease": "heart"})
    diabetes_count = await db.predictions.count_documents({"user_id": user_id, "disease": "diabetes"})
    breast_cancer_count = await db.predictions.count_documents({"user_id": user_id, "disease": "breast_cancer"})
    
    positive_count = await db.predictions.count_documents({"user_id": user_id, "prediction": 1})
    negative_count = await db.predictions.count_documents({"user_id": user_id, "prediction": 0})
    
    # Recent predictions
    recent_cursor = db.predictions.find({"user_id": user_id}).sort("created_at", -1).limit(5)
    recent = await recent_cursor.to_list(length=5)
    for r in recent:
        r["id"] = str(r.pop("_id"))
        
    # Disease distribution
    disease_distribution = [
        {"name": "Heart Disease", "value": heart_count},
        {"name": "Diabetes", "value": diabetes_count},
        {"name": "Breast Cancer", "value": breast_cancer_count}
    ]
    
    # Group by date for trends
    pipeline = [
        {"$match": {"user_id": user_id}},
        {
            "$group": {
                "_id": {
                    "$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}
                },
                "count": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}},
        {"$limit": 30}
    ]
    trend_cursor = db.predictions.aggregate(pipeline)
    trends = await trend_cursor.to_list(length=30)
    
    prediction_trend = [{"date": t["_id"], "count": t["count"]} for t in trends]
    
    return APIResponse(
        success=True,
        message="Dashboard stats retrieved",
        data={
            "total_predictions": total,
            "heart_predictions": heart_count,
            "diabetes_predictions": diabetes_count,
            "breast_cancer_predictions": breast_cancer_count,
            "positive_predictions": positive_count,
            "negative_predictions": negative_count,
            "recent_predictions": recent,
            "disease_distribution": disease_distribution,
            "prediction_trend": prediction_trend
        }
    )
