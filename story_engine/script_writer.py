"""Narration writing.

Two jobs:
  * `write_narration()` turns story beats into narration lines (Gemini, or the
    beat text itself as the offline fallback — never silently rewritten).
  * `estimate_duration()` converts words to seconds so scene lengths match the
    spoken audio.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
MODELS = [os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash-lite",
          "gemini-2.5-flash-lite", "gemini-flash-lite-latest"]

WORDS_PER_SECOND = 2.6

PROMPT = """Write narration for a vertical animated explainer short.

STORY:
\"\"\"{story}\"\"\"

BEATS (one per sentence, keep the same order and count):
{beats}

Return ONLY JSON: {{"lines": ["narration line 1", "narration line 2", ...]}}

Rules:
- exactly {count} lines, one per beat
- spoken English, short sentences, 8-18 words each
- no stage directions, no quotes, no markdown, no emojis
- keep every fact of the story; do not invent new events
"""


def estimate_duration(text: str, min_seconds: float = 2.0, max_seconds: float = 12.0) -> float:
    words = max(1, len((text or "").split()))
    seconds = words / WORDS_PER_SECOND
    return float(max(min_seconds, min(max_seconds, seconds)))


def _fallback_lines(beats: List[dict]) -> List[str]:
    lines = []
    for beat in beats:
        text = (beat.get("text") or "").strip()
        text = re.sub(r"\s+", " ", text)
        lines.append(text)
    return lines


def gemini_lines(story: str, beats: List[dict]) -> Optional[List[str]]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key or requests is None:
        return None
    beat_block = "\n".join(f"{i + 1}. {b.get('text', '')}" for i, b in enumerate(beats))
    body = {
        "contents": [{"role": "user", "parts": [{"text": PROMPT.format(
            story=story[:5000], beats=beat_block, count=len(beats))}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000,
                             "responseMimeType": "application/json"},
    }
    for model in MODELS:
        try:
            resp = requests.post(f"{API_ROOT}/{model}:generateContent",
                                 params={"key": key}, json=body, timeout=120)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        try:
            raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
            lines = json.loads(raw).get("lines") or []
            if len(lines) == len(beats):
                return [str(x).strip() for x in lines]
        except Exception:
            continue
    return None


def write_narration(story: str, beats: List[dict], use_ai: bool = True) -> Dict:
    lines = gemini_lines(story, beats) if use_ai else None
    source = "gemini"
    if not lines:
        lines = _fallback_lines(beats)
        source = "story-text"
    total = sum(estimate_duration(line) for line in lines)
    return {
        "lines": lines,
        "source": source,
        "total_seconds": round(total, 2),
        "words": sum(len(line.split()) for line in lines),
    }


def save(script: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(script, f, indent=2, ensure_ascii=False)
    return path
