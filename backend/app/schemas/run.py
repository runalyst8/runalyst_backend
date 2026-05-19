from pydantic import BaseModel, computed_field
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

    @computed_field
    @property
    def thumbnail_path(self) -> str:
        if not self.video_path:
            return ""

        if self.video_path.endswith(".mp4"):
            return self.video_path[:-4] + ".jpg"

        return self.video_path + ".jpg"

class RunAllOut(BaseModel):
    runs: Dict[int, RunOut]