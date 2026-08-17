from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.auth import UserCreate, UserLogin, UserResponse, Token
from app.schemas.response import APIResponse
from app.utils.helpers import get_password_hash, verify_password, create_access_token
from app.database import get_database
from app.middleware.auth_middleware import get_current_user
from bson import ObjectId
from datetime import datetime

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db=Depends(get_database)):
    """Register a new user."""
    # Check for existing email
    existing_user = await db.users.find_one({"email": user_data.email.lower()})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_dict = {
        "name": user_data.name,
        "email": user_data.email.lower(),
        "password_hash": get_password_hash(user_data.password),
        "role": "user",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    result = await db.users.insert_one(user_dict)
    user_id = str(result.inserted_id)

    # Generate token on registration
    access_token = create_access_token(
        data={"user_id": user_id, "email": user_dict["email"], "role": "user"}
    )

    return APIResponse(
        success=True,
        message="User registered successfully",
        data={
            "user": {
                "id": user_id,
                "name": user_dict["name"],
                "email": user_dict["email"],
                "role": user_dict["role"],
            },
            "access_token": access_token,
            "token_type": "bearer",
        },
    )


@router.post("/login", response_model=APIResponse)
async def login(credentials: UserLogin, db=Depends(get_database)):
    """Login with email and password (JSON body)."""
    user = await db.users.find_one({"email": credentials.email.lower()})

    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={
            "user_id": str(user["_id"]),
            "email": user["email"],
            "role": user.get("role", "user"),
        }
    )

    return APIResponse(
        success=True,
        message="Login successful",
        data={
            "user": {
                "id": str(user["_id"]),
                "name": user["name"],
                "email": user["email"],
                "role": user.get("role", "user"),
            },
            "access_token": access_token,
            "token_type": "bearer",
        },
    )


@router.get("/me", response_model=APIResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get the current authenticated user's profile."""
    return APIResponse(
        success=True,
        message="User profile retrieved",
        data={
            "id": current_user["id"],
            "name": current_user["name"],
            "email": current_user["email"],
            "role": current_user.get("role", "user"),
            "created_at": str(current_user.get("created_at", "")),
        },
    )
