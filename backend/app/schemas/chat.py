from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class VideoSummary(BaseModel):
    user_id: str
    video_id: str
    title: str | None = None
    thumbnail_url: str | None = None
    created_at: datetime | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    analysis: dict[str, Any] | str | None = None


class VideoListResponse(BaseModel):
    session_id: str
    user_id: str
    videos: list[VideoSummary]


class SelectVideoRequest(BaseModel):
    session_id: str
    video_id: str


class SelectVideoResponse(BaseModel):
    ok: bool
    selected_video: VideoSummary


class ChatMessageRequest(BaseModel):
    session_id: str
    message: str


class ChatMessageResponse(BaseModel):
    answer: str
