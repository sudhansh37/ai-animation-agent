"""Scene transitions: cut, fade, cross-fade, slide, flash, iris."""

from __future__ import annotations

from typing import Optional

from PIL import Image, ImageDraw

from render_engine.draw_utils import ease

TRANSITIONS = ["cut", "fade", "crossfade", "slide", "flash", "iris"]


def apply_transition(current: Image.Image, previous: Optional[Image.Image], kind: str,
                     p: float, duration_frames: int = 12) -> Image.Image:
    """Blend `current` with the outgoing frame `previous` during the first frames.

    `p` is progress through the *transition* (0..1), not through the scene.
    """
    kind = (kind or "cut").lower()
    if kind == "cut" or previous is None or p >= 1.0:
        return current

    e = ease("smooth", p)
    w, h = current.size

    if kind == "fade":
        black = Image.new("RGB", current.size, (0, 0, 0))
        return Image.blend(black, current, e)

    if kind == "crossfade":
        return Image.blend(previous.convert(current.mode), current, e)

    if kind == "slide":
        out = Image.new(current.mode, current.size)
        offset = int(w * (1 - e))
        out.paste(current, (offset, 0))
        out.paste(previous, (offset - w, 0))
        return out

    if kind == "flash":
        white = Image.new("RGB", current.size, (255, 255, 255))
        amount = 1.0 - abs(2 * e - 1)
        return Image.blend(current, white, amount * 0.85)

    if kind == "iris":
        mask = Image.new("L", current.size, 0)
        d = ImageDraw.Draw(mask)
        radius = int((w ** 2 + h ** 2) ** 0.5 * e)
        d.ellipse([w / 2 - radius, h / 2 - radius, w / 2 + radius, h / 2 + radius], fill=255)
        black = Image.new("RGB", current.size, (0, 0, 0))
        return Image.composite(current, black, mask)

    return current
