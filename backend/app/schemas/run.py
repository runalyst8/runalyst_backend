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

        path = self.video_path.replace("user_videos_test", "video_thumbnails")
        if path.endswith(".mp4"):
            path = path[:-4] + ".jpg"
        return path

class RunAllOut(BaseModel):
    runs: Dict[int, RunOut]