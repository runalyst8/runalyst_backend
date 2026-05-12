from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.deps.auth import get_current_user_id
from app.deps.db import get_db
from app.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    SelectVideoRequest,
    SelectVideoResponse,
    VideoListResponse,
)
from app.services.chat import service as chat_service

router = APIRouter()


@router.get("/videos", response_model=VideoListResponse, status_code=status.HTTP_200_OK)
def get_videos(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> VideoListResponse:
    session_id, videos = chat_service.list_user_videos(db, user_id=int(user_id))
    return VideoListResponse(session_id=session_id, user_id=str(user_id), videos=videos)


@router.post("/videos/select", response_model=SelectVideoResponse, status_code=status.HTTP_200_OK)
def select_video(
    payload: SelectVideoRequest,
    user_id: int = Depends(get_current_user_id),
) -> SelectVideoResponse:
    selected_video = chat_service.select_video(
        session_id=payload.session_id,
        user_id=int(user_id),
        video_id=payload.video_id,
    )
    return SelectVideoResponse(ok=True, selected_video=selected_video)


@router.post("/message", response_model=ChatMessageResponse, status_code=status.HTTP_200_OK)
async def chat(
    payload: ChatMessageRequest,
    user_id: int = Depends(get_current_user_id),
) -> ChatMessageResponse:
    answer = await chat_service.answer_chat_message(
        session_id=payload.session_id,
        user_id=int(user_id),
        message=payload.message,
    )
    return ChatMessageResponse(answer=answer)
