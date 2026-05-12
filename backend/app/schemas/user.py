from app.schemas.auth import SignUpIn
from pydantic import BaseModel, EmailStr, ConfigDict
from datetime import datetime
from typing import Optional

class UserUpdateIn(SignUpIn):
    class Config:
        from_attributes = True

class UserBase(BaseModel):
    email: EmailStr
    is_active: Optional[bool] = True

class UserOut(UserBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)