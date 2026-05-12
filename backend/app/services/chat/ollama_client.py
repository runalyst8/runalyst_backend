from typing import Any

import httpx

from app.core.config import settings


def build_video_context(video: dict[str, Any], memory: dict[str, str]) -> str:
    memory_lines = "\n".join(f"- {key}: {value}" for key, value in memory.items())
    memory_context = memory_lines or "- no user-specific memory has been stored yet"

    return (
        "You are an expert assistant helping the user understand running video analysis results.\n"
        "Base your answers on the selected video's metadata and full analysis JSON. "
        "If you make an assumption, state that clearly. "
        "Answer in English, keep responses concise and actionable, and do not reveal your reasoning process. "
        "Remember user-specific facts from the conversation memory in later replies.\n\n"
        f"Conversation memory:\n{memory_context}\n\n"
        "Selected video information:\n"
        f"- user_id: {video.get('user_id')}\n"
        f"- video_id: {video.get('video_id')}\n"
        f"- title: {video.get('title')}\n"
        f"- created_at: {video.get('created_at')}\n"
        f"- metadata: {video.get('metadata')}\n"
        f"- analysis JSON: {video.get('analysis')}"
    )


async def ask_ollama(
    *,
    video: dict[str, Any],
    message: str,
    history: list[dict[str, str]],
    memory: dict[str, str],
) -> str:
    messages = [{"role": "system", "content": build_video_context(video, memory)}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": message})

    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 900,
                },
            },
        )
        response.raise_for_status()

    data = response.json()
    message_data = data.get("message", {})
    answer = message_data.get("content", "").strip()
    if answer:
        return answer

    thinking = message_data.get("thinking", "").strip()
    if thinking:
        return thinking

    return "Ollama returned an empty response. Please try again."
