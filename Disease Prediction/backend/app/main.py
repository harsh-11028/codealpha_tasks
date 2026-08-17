from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.config import settings
from app.database import connect_to_mongo, close_mongo_connection
from app.ml.predictor import predictor

from app.routes import auth, predict, history, dashboard, model_metrics, diseases, admin

app = FastAPI(
    title="AI Disease Prediction API",
    description="Backend API for AI-based disease prediction system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()
    predictor.load_models()

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()

# Include routers
app.include_router(auth.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(model_metrics.router, prefix="/api")
app.include_router(diseases.router, prefix="/api")
app.include_router(admin.router, prefix="/api")

@app.get("/api/health", tags=["health"])
async def health_check():
    return {"status": "ok", "message": "API is running"}

# Global Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Special handling for our custom 404
    if exc.status_code == 404 and exc.detail == "Not Found":
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "Resource not found", "detail": "The requested endpoint does not exist."}
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": exc.detail, "detail": str(exc.detail)}
    )

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=404,
        content={"success": False, "message": "Resource not found", "detail": "The requested endpoint does not exist."}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"success": False, "message": "Validation Error", "detail": str(exc.errors())}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal Server Error", "detail": str(exc)}
    )
