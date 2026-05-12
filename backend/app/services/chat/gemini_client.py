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


def history_to_gemini_contents(history: list[dict[str, str]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in history[-10:]:
        content = message.get("content", "").strip()
        if not content:
            continue

        role = "model" if message.get("role") == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": content}]})

    return contents


async def ask_gemini(
    *,
    video: dict[str, Any],
    message: str,
    history: list[dict[str, str]],
    memory: dict[str, str],
) -> str:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    contents = history_to_gemini_contents(history)
    contents.append({"role": "user", "parts": [{"text": message}]})

    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{settings.GEMINI_BASE_URL}/v1beta/models/{settings.GEMINI_MODEL}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": settings.GEMINI_API_KEY,
            },
            json={
                "systemInstruction": {
                    "parts": [{"text": build_video_context(video, memory)}],
                },
                "contents": contents,
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 900,
                },
            },
        )
        response.raise_for_status()

    data = response.json()
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "\n".join(part.get("text", "") for part in parts).strip()
        if answer:
            return answer

    return "Gemini returned an empty response. Please try again."
