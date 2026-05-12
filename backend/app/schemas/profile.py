from typing import Optional
from pydantic import BaseModel
from app.core.enums import Gender, ExperienceLevel, RunningGoal

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
        orm_mode = True

