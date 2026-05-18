from typing import Optional
from pydantic import BaseModel, EmailStr, Field

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

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class GoogleAuthIn(BaseModel):
    token: str


class AppleAuthIn(BaseModel):
    identity_token: str
    email: Optional[str] = None
    full_name: Optional[str] = None

class TokenRefreshRequest(BaseModel):
    refresh_token: str