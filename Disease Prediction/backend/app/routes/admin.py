from fastapi import APIRouter, Depends
from app.schemas.response import APIResponse
from app.middleware.auth_middleware import require_admin
from app.database import get_database

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/stats", response_model=APIResponse)
async def get_admin_stats(
    current_user: dict = Depends(require_admin),
    db = Depends(get_database)
):
    total_users = await db.users.count_documents({})
    total_predictions = await db.predictions.count_documents({})
    
    pipeline = [
        {"$group": {"_id": "$disease", "count": {"$sum": 1}}}
    ]
    disease_stats_cursor = db.predictions.aggregate(pipeline)
    disease_stats = await disease_stats_cursor.to_list(length=10)
    
    stats_formatted = {item["_id"]: item["count"] for item in disease_stats if item["_id"]}
    
    return APIResponse(
        success=True,
        message="Admin stats retrieved",
        data={
            "total_users": total_users,
            "total_predictions": total_predictions,
            "disease_breakdown": stats_formatted
        }
    )

@router.get("/users", response_model=APIResponse)
async def get_all_users(
    skip: int = 0,
    limit: int = 50,
    current_user: dict = Depends(require_admin),
    db = Depends(get_database)
):
    cursor = db.users.find({}, {"password_hash": 0}).skip(skip).limit(limit)
    users = await cursor.to_list(length=limit)
    
    for u in users:
        u["id"] = str(u.pop("_id"))
        
    total = await db.users.count_documents({})
    
    return APIResponse(
        success=True,
        message="Users retrieved successfully",
        data={
            "users": users,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    )
