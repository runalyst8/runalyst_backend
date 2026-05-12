import re
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session, joinedload

from app.models.run import Run
from app.services.chat.ollama_client import ask_ollama
from app.services.chat.session_store import session_store


NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Za-zÇĞİÖŞÜçğıöşü0-9_-]+)", re.IGNORECASE),
    re.compile(r"\bbenim adım\s+([A-Za-zÇĞİÖŞÜçğıöşü0-9_-]+)", re.IGNORECASE),
    re.compile(r"\badım\s+([A-Za-zÇĞİÖŞÜçğıöşü0-9_-]+)", re.IGNORECASE),
    re.compile(r"\bben\s+([A-Za-zÇĞİÖŞÜçğıöşü0-9_-]+)", re.IGNORECASE),
]
NAME_QUESTION_PATTERNS = [
    re.compile(r"\bwhat is my name\b", re.IGNORECASE),
    re.compile(r"\bwhat's my name\b", re.IGNORECASE),
    re.compile(r"\bremember my name\b", re.IGNORECASE),
    re.compile(r"\bknow my name\b", re.IGNORECASE),
    re.compile(r"\bdo not remember my name\b", re.IGNORECASE),
    re.compile(r"\bdon't remember my name\b", re.IGNORECASE),
    re.compile(r"\badim ne\b", re.IGNORECASE),
    re.compile(r"\badım ne\b", re.IGNORECASE),
    re.compile(r"\bbenim adim ne\b", re.IGNORECASE),
    re.compile(r"\bbenim adım ne\b", re.IGNORECASE),
]


def update_memory_from_message(memory: dict[str, str], message: str) -> None:
    for pattern in NAME_PATTERNS:
        match = pattern.search(message)
        if match:
            memory["name"] = match.group(1).strip()
            return


def answer_from_memory(memory: dict[str, str], message: str) -> str | None:
    if any(pattern.search(message) for pattern in NAME_QUESTION_PATTERNS):
        name = memory.get("name")
        if name:
            return f"Your name is {name}."
        return "I do not know your name yet."

    return None


def run_to_video_summary(run: Run) -> dict[str, Any]:
    analysis_result = run.analysis_result
    metadata: dict[str, Any] = {
        "video_path": run.video_path,
        "status": run.status,
    }
    if analysis_result:
        metadata["fps"] = analysis_result.fps

    return {
        "user_id": str(run.user_id),
        "video_id": str(run.id),
        "title": run.title or f"Analysis {run.id}",
        "thumbnail_url": None,
        "created_at": analysis_result.created_at if analysis_result else run.created_at,
        "metadata": metadata,
        "analysis": analysis_result.modules if analysis_result else None,
    }


def list_user_videos(db: Session, *, user_id: int) -> tuple[str, list[dict[str, Any]]]:
    runs = (
        db.query(Run)
        .options(joinedload(Run.analysis_result))
        .filter(Run.user_id == user_id, Run.analysis_result.has())
        .order_by(Run.created_at.desc())
        .all()
    )
    videos = [run_to_video_summary(run) for run in runs]
    session_id, _ = session_store.create(str(user_id), videos)
    return session_id, videos


def select_video(*, session_id: str, user_id: int, video_id: str) -> dict[str, Any]:
    state = session_store.get(session_id, str(user_id))
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found. Reload the video list.")

    selected_video = state.videos_by_id.get(video_id)
    if not selected_video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video was not found in this session.")

    state.selected_video_id = video_id
    state.chat_history.clear()
    return selected_video


async def answer_chat_message(*, session_id: str, user_id: int, message: str) -> str:
    state = session_store.get(session_id, str(user_id))
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found. Reload the video list.")
    if not state.selected_video_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Select a video first.")

    selected_video = state.videos_by_id[state.selected_video_id]
    update_memory_from_message(state.memory, message)

    answer = answer_from_memory(state.memory, message)
    if not answer:
        try:
            answer = await ask_ollama(
                video=selected_video,
                message=message,
                history=state.chat_history,
                memory=state.memory,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not get a response from Ollama: {exc}",
            ) from exc

    state.chat_history.append({"role": "user", "content": message})
    state.chat_history.append({"role": "assistant", "content": answer})
    return answer
