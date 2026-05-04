from typing import Optional
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from app.core.enums import Gender, ExperienceLevel, RunningGoal

class SignUpIn(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword123"
            }
        }

class ProfileUpdateIn(BaseModel):
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    bio: Optional[str] = None
    gender: Optional[Gender] = None
    experience_level: Optional[ExperienceLevel] = None
    running_goal: Optional[RunningGoal] = None
    has_injuries: Optional[bool] = None


class ProfileOut(ProfileUpdateIn):
    class Config:
        from_attributes = True


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: int
    email: EmailStr
    is_active: bool
    created_at: datetime
    profile: Optional[ProfileOut] = None 
    
    class Config:
        from_attributes = True

class PasswordResetRequestIn(BaseModel):
    email: EmailStr

class PasswordResetIn(BaseModel):
    token: str
    new_password: str

class SendVerificationIn(BaseModel):
    email: EmailStr

class VerifyEmailIn(BaseModel):
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6)
