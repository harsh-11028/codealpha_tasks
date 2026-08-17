from fastapi import APIRouter, Depends, HTTPException
from app.schemas.response import APIResponse
from app.middleware.auth_middleware import get_current_user

router = APIRouter(prefix="/diseases", tags=["diseases"])

DISEASES_DATA = {
    "heart": {
        "id": "heart",
        "name": "Heart Disease",
        "description": "Heart disease describes a range of conditions that affect your heart.",
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    },
    "diabetes": {
        "id": "diabetes",
        "name": "Diabetes",
        "description": "Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high.",
        "features": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    },
    "breast_cancer": {
        "id": "breast_cancer",
        "name": "Breast Cancer",
        "description": "Breast cancer is a disease in which cells in the breast grow out of control.",
        "features": ["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension", "radius_error", "texture_error", "perimeter_error", "area_error", "smoothness_error", "compactness_error", "concavity_error", "concave_points_error", "symmetry_error", "fractal_dimension_error", "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness", "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"]
    }
}

@router.get("", response_model=APIResponse)
async def get_diseases(current_user: dict = Depends(get_current_user)):
    return APIResponse(
        success=True,
        message="Diseases list retrieved",
        data=list(DISEASES_DATA.values())
    )

@router.get("/{disease}", response_model=APIResponse)
async def get_disease(disease: str, current_user: dict = Depends(get_current_user)):
    if disease not in DISEASES_DATA:
        raise HTTPException(status_code=404, detail="Disease not found")
        
    return APIResponse(
        success=True,
        message="Disease details retrieved",
        data=DISEASES_DATA[disease]
    )
