from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict


class RunCreateIn(BaseModel):
    video_path: str
    title: Optional[str] = None

class RunOut(BaseModel):
    id: int
    title: Optional[str]
    video_path: str
    created_at: datetime
    user_id: int

    class Config:
        from_attributes = True

class RunAllOut(BaseModel):
    runs: Dict[int, RunOut]