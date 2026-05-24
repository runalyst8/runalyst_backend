from pydantic import BaseModel, EmailStr, ConfigDict
from datetime import datetime
from typing import Optional

class UserUpdateIn(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = None

    class Config:
        from_attributes = True

class UserBase(BaseModel):
    email: EmailStr
    is_active: Optional[bool] = True

class UserOut(UserBase):
    id: int
    auth_provider: str = "local"
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)