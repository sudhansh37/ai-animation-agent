"""Subtitle rendering and SRT export.

Captions are burned into the finished frame (after the camera, so they never
scale or drift) in the safe area at the bottom, wrapped automatically and
outlined for contrast against any background.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from render_engine.draw_utils import font, wrap_text

SAFE_BOTTOM = 0.885    # keep captions above the platform UI
MAX_WIDTH = 0.86       # fraction of frame width


def _timestamp(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def draw_caption(img: Image.Image, text: str, highlight_word: Optional[str] = None) -> None:
    """Draw one caption block near the bottom of the frame."""
    text = (text or "").strip()
    if not text:
        return
    w, h = img.size
    size = max(28, int(w * 0.056))
    fnt = font(size, bold=True)
    lines = wrap_text(text, fnt, int(w * MAX_WIDTH))
    if not lines:
        return

    line_h = int(size * 1.28)
    block_h = line_h * len(lines)
    y0 = int(h * SAFE_BOTTOM) - block_h

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    for i, line in enumerate(lines):
        tw = fnt.getlength(line)
        x = (w - tw) / 2
        y = y0 + i * line_h
        # soft dark plate for contrast
        d.rounded_rectangle([x - size * 0.28, y - size * 0.10,
                             x + tw + size * 0.28, y + size * 1.06],
                            radius=size * 0.22, fill=(0, 0, 0, 120))
        # outline then fill
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)):
            d.text((x + dx, y + dy), line, font=fnt, fill=(0, 0, 0, 235))
        d.text((x, y), line, font=fnt, fill=(255, 255, 255, 255))

        if highlight_word:
            idx = line.lower().find(highlight_word.lower())
            if idx >= 0:
                pre = line[:idx]
                word = line[idx:idx + len(highlight_word)]
                wx = x + fnt.getlength(pre)
                ww = fnt.getlength(word)
                d.rounded_rectangle([wx - 3, y + size * 0.06, wx + ww + 3, y + size * 1.0],
                                    radius=size * 0.12, fill=(255, 224, 102, 110))

    img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert(img.mode), (0, 0))


def write_srt(cues: Sequence[dict], path: str) -> str:
    """cues = [{"start": s, "end": s, "text": "..."}]"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, 1):
            f.write(f"{i}\n{_timestamp(cue['start'])} --> {_timestamp(cue['end'])}\n{cue['text'].strip()}\n\n")
    return path


def cues_from_scenes(scenes: Sequence[dict]) -> List[dict]:
    """One caption per scene, split into <=12-word chunks when the line is long."""
    cues: List[dict] = []
    for scene in scenes:
        text = (scene.get("caption") or scene.get("narration") or "").strip()
        if not text:
            continue
        start = float(scene.get("start", 0.0))
        end = start + float(scene.get("duration", 1.0))
        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        for word in words:
            current.append(word)
            if len(current) >= 11 or word.endswith((".", "!", "?")):
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        if len(chunks) <= 1:
            cues.append({"start": start, "end": end, "text": text})
        else:
            span = (end - start) / len(chunks)
            for i, chunk in enumerate(chunks):
                cues.append({"start": start + i * span, "end": start + (i + 1) * span, "text": chunk})
    return cues


def caption_for_time(cues: Sequence[dict], t: float) -> Optional[str]:
    for cue in cues:
        if cue["start"] <= t < cue["end"]:
            return cue["text"]
    return None
