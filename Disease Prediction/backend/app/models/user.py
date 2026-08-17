from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr

class UserInDB(BaseModel):
    name: str
    email: EmailStr
    password_hash: str
    role: str = "user"
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
