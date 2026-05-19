from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# JSON schema enforced on Gemini's output
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "issues": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "issue_key":       {"type": "STRING"},
                    "name":            {"type": "STRING"},
                    "severity":        {"type": "STRING"},
                    "impact":          {"type": "STRING"},
                    "exercises": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "name":             {"type": "STRING"},
                                "type":             {"type": "STRING"},
                                "duration_or_reps": {"type": "STRING"},
                                "rationale":        {"type": "STRING"},
                            },
                            "required": ["name", "type", "duration_or_reps", "rationale"],
                        },
                    },
                    "drills": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "name":     {"type": "STRING"},
                                "duration": {"type": "STRING"},
                                "cue":      {"type": "STRING"},
                            },
                            "required": ["name", "duration", "cue"],
                        },
                    },
                    "technique_cues": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
                "required": ["issue_key", "name", "severity", "impact", "exercises", "drills", "technique_cues"],
            },
        },
        "summary": {"type": "STRING"},
    },
    "required": ["issues", "summary"],
}


def _build_system_prompt(profile: dict[str, Any] | None) -> str:
    profile_lines = ""
    if profile:
        parts = []
        if profile.get("age"):
            parts.append(f"age {profile['age']}")
        if profile.get("experience_level"):
            parts.append(f"{profile['experience_level']} runner")
        if profile.get("running_goal"):
            parts.append(f"goal: {profile['running_goal'].replace('_', ' ')}")
        if profile.get("has_injuries"):
            parts.append("has existing injuries — recommend conservative progressions")
        if parts:
            profile_lines = f"\nRunner profile: {', '.join(parts)}."

    return (
        "You are a certified running coach and sports physiotherapist. "
        "You will be given a list of biomechanical issues detected in a runner's gait analysis. "
        "For each issue, provide 2–3 targeted exercises, 1–2 running drills, and 2–3 technique cues. "
        "Tailor the intensity and complexity to the runner's profile if provided. "
        "Be specific: include sets/reps or durations, not vague instructions. "
        "Output must be valid JSON matching the provided schema."
        + profile_lines
    )


def _build_user_prompt(issues: list[dict[str, Any]]) -> str:
    lines = ["Detected gait issues (ordered by priority):\n"]
    for i, issue in enumerate(issues, 1):
        lines.append(
            f"{i}. [{issue['severity'].upper()}] {issue['name']}\n"
            f"   Measured: {issue['measured_value']} (threshold: {issue['threshold']})\n"
            f"   Context: {issue['context']}"
        )
    lines.append(
        "\nFor each issue, provide targeted exercises, drills, and technique cues. "
        "Also write a concise overall summary (2–3 sentences) the runner should read first."
    )
    return "\n".join(lines)


async def generate_recommendations(
    *,
    issues: list[dict[str, Any]],
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    system_prompt = _build_system_prompt(profile)
    user_prompt = _build_user_prompt(issues)

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{settings.GEMINI_BASE_URL}/v1beta/models/{settings.GEMINI_MODEL}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": settings.GEMINI_API_KEY,
            },
            json={
                "systemInstruction": {
                    "parts": [{"text": system_prompt}],
                },
                "contents": [
                    {"role": "user", "parts": [{"text": user_prompt}]},
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 4096,
                    "responseMimeType": "application/json",
                    "responseSchema": _RESPONSE_SCHEMA,
                },
            },
        )
        response.raise_for_status()

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    raw_text = "\n".join(p.get("text", "") for p in parts).strip()
    if not raw_text:
        raise ValueError("Gemini returned empty content")

    return json.loads(raw_text)
