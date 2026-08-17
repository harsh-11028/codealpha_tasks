from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.prediction import HeartPredictionInput, DiabetesPredictionInput, BreastCancerPredictionInput, PredictionResponse
from app.middleware.auth_middleware import get_current_user
from app.database import get_database
from app.ml.predictor import predictor
from datetime import datetime

router = APIRouter(prefix="/predict", tags=["prediction"])

async def handle_prediction(disease: str, input_data: dict, user_id: str, db) -> PredictionResponse:
    try:
        prediction, probability, model_name = predictor.predict(disease, input_data)
        
        # Save to DB
        prediction_doc = {
            "user_id": user_id,
            "disease": disease,
            "input_data": input_data,
            "prediction": prediction,
            "probability": probability,
            "model_used": model_name,
            "created_at": datetime.utcnow()
        }
        
        result = await db.predictions.insert_one(prediction_doc)
        
        label = "Positive" if prediction == 1 else "Negative"
        
        return PredictionResponse(
            success=True,
            disease=disease,
            prediction=prediction,
            label=label,
            probability=probability,
            model=model_name,
            message=f"{disease.replace('_', ' ').title()} prediction completed",
            prediction_id=str(result.inserted_id)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/heart", response_model=PredictionResponse)
async def predict_heart(
    data: HeartPredictionInput,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    return await handle_prediction("heart", data.model_dump(), current_user["id"], db)

@router.post("/diabetes", response_model=PredictionResponse)
async def predict_diabetes(
    data: DiabetesPredictionInput,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    return await handle_prediction("diabetes", data.model_dump(), current_user["id"], db)

@router.post("/breast-cancer", response_model=PredictionResponse)
async def predict_breast_cancer(
    data: BreastCancerPredictionInput,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    return await handle_prediction("breast_cancer", data.model_dump(), current_user["id"], db)
