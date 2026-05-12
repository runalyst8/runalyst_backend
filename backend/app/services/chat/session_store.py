from dataclasses import dataclass, field
from time import time
from typing import Any
from uuid import uuid4


SESSION_TTL_SECONDS = 60 * 60


@dataclass
class SessionState:
    user_id: str
    videos_by_id: dict[str, dict[str, Any]]
    selected_video_id: str | None = None
    chat_history: list[dict[str, str]] = field(default_factory=list)
    memory: dict[str, str] = field(default_factory=dict)
    updated_at: float = field(default_factory=time)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create(self, user_id: str, videos: list[dict[str, Any]]) -> tuple[str, SessionState]:
        self._cleanup()
        session_id = str(uuid4())
        state = SessionState(
            user_id=user_id,
            videos_by_id={str(video["video_id"]): video for video in videos},
        )
        self._sessions[session_id] = state
        return session_id, state

    def get(self, session_id: str, user_id: str) -> SessionState | None:
        self._cleanup()
        state = self._sessions.get(session_id)
        if not state or state.user_id != user_id:
            return None
        state.updated_at = time()
        return state

    def _cleanup(self) -> None:
        now = time()
        expired = [
            session_id
            for session_id, state in self._sessions.items()
            if now - state.updated_at > SESSION_TTL_SECONDS
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)


session_store = SessionStore()
