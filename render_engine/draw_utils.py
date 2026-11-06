"""Shared vector-drawing helpers on top of Pillow.

Everything the renderer draws — characters, backgrounds, objects, subtitles —
goes through these primitives, so line weights, outlines and shading stay
consistent across the whole video.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

Point = Tuple[float, float]
RGBA = Tuple[int, int, int, int]

# --------------------------------------------------------------------------- #
# colour helpers
# --------------------------------------------------------------------------- #


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    v = (value or "#000000").strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) < 6:
        v = v.ljust(6, "0")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: Sequence[int]) -> str:
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def shade(colour: str, amount: float) -> str:
    """amount > 0 lightens, < 0 darkens."""
    r, g, b = hex_to_rgb(colour)
    f = lambda x: max(0, min(255, int(x + (255 - x) * amount if amount >= 0 else x * (1 + amount))))
    return rgb_to_hex((f(r), f(g), f(b)))


def mix(a: str, b: str, t: float) -> str:
    ar, ag, ab = hex_to_rgb(a)
    br, bg, bb = hex_to_rgb(b)
    return rgb_to_hex((ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t))


def luminance(colour: str) -> float:
    r, g, b = hex_to_rgb(colour)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #

FONT_CANDIDATES = [
    ("assets/fonts/DejaVuSans-Bold.ttf", "assets/fonts/DejaVuSans.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ("/Library/Fonts/Arial Bold.ttf", "/Library/Fonts/Arial.ttf"),
    ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"),
]

_font_cache: dict = {}


def resolve_font(bold: bool = True) -> str:
    for bold_path, regular_path in FONT_CANDIDATES:
        path = bold_path if bold else regular_path
        if os.path.exists(path):
            return path
        if os.path.exists(regular_path):
            return regular_path
    raise RuntimeError(
        "No usable TTF font found. Put DejaVuSans-Bold.ttf in assets/fonts/ "
        "(the repo ships one) or install fonts-dejavu."
    )


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(resolve_font(bold), size)
    return _font_cache[key]


def wrap_text(text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = (text or "").split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if fnt.getlength(trial) <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


def capsule(draw: ImageDraw.ImageDraw, p1: Point, p2: Point, width: float,
            fill: str, outline: str | None = None, outline_width: float = 0.0) -> None:
    """A rounded limb segment: an ink outline with the fill on top."""
    w = max(1, int(round(width)))
    if outline and outline_width > 0:
        ow = int(round(w + outline_width * 2))
        draw.line([p1, p2], fill=outline, width=ow, joint="curve")
        r = ow / 2.0
        draw.ellipse([p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r], fill=outline)
        draw.ellipse([p2[0] - r, p2[1] - r, p2[0] + r, p2[1] + r], fill=outline)
    draw.line([p1, p2], fill=fill, width=w, joint="curve")
    r = w / 2.0
    draw.ellipse([p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r], fill=fill)
    draw.ellipse([p2[0] - r, p2[1] - r, p2[0] + r, p2[1] + r], fill=fill)


def ellipse(draw: ImageDraw.ImageDraw, cx: float, cy: float, rx: float, ry: float,
            fill: str, outline: str | None = None, outline_width: float = 0.0,
            rotation: float = 0.0) -> None:
    rx, ry = abs(rx), abs(ry)
    if rx < 0.5 or ry < 0.5:
        return
    box = [cx - rx, cy - ry, cx + rx, cy + ry]
    if outline and outline_width > 0:
        draw.ellipse(box, fill=fill, outline=outline, width=int(round(outline_width)))
    else:
        draw.ellipse(box, fill=fill)


def polygon(draw: ImageDraw.ImageDraw, points: Iterable[Point], fill: str,
            outline: str | None = None, outline_width: float = 0.0) -> None:
    pts = [tuple(p) for p in points]
    draw.polygon(pts, fill=fill)
    if outline and outline_width > 0:
        draw.line(pts + [pts[0]], fill=outline, width=int(round(outline_width)), joint="curve")


def rounded_rect(draw: ImageDraw.ImageDraw, box: Sequence[float], radius: float,
                 fill: str, outline: str | None = None, outline_width: float = 0.0) -> None:
    x0, y0, x1, y1 = box
    # poses can invert a box (rotated bodies, extreme leans) — normalise instead
    # of letting Pillow raise mid-frame
    x0, x1 = (x0, x1) if x1 >= x0 else (x1, x0)
    y0, y1 = (y0, y1) if y1 >= y0 else (y1, y0)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return
    radius = max(0.0, min(radius, (x1 - x0) / 2, (y1 - y0) / 2))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill,
                           outline=outline, width=int(round(outline_width)) if outline else 0)


def vertical_gradient(size: Tuple[int, int], top: str, bottom: str) -> Image.Image:
    w, h = size
    base = Image.new("RGB", (1, h))
    px = base.load()
    tr, tg, tb = hex_to_rgb(top)
    br, bg, bb = hex_to_rgb(bottom)
    for y in range(h):
        t = y / max(1, h - 1)
        px[0, y] = (int(tr + (br - tr) * t), int(tg + (bg - tg) * t), int(tb + (bb - tb) * t))
    return base.resize((w, h), Image.BILINEAR)


def radial_glow(size: int, colour: str, strength: float = 1.0) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r, g, b = hex_to_rgb(colour)
    steps = 24
    for i in range(steps, 0, -1):
        t = i / steps
        alpha = int(120 * strength * (1 - t) ** 1.6)
        rad = size / 2 * t
        d.ellipse([size / 2 - rad, size / 2 - rad, size / 2 + rad, size / 2 + rad],
                  fill=(r, g, b, alpha))
    return img.filter(ImageFilter.GaussianBlur(size * 0.03))


_shadow_cache: dict = {}
_vignette_cache: dict = {}


def soft_shadow(img: Image.Image, centre: Point, rx: float, ry: float,
                alpha: int = 90, blur: float = 12.0) -> None:
    """Paints a blurred contact shadow under a character, in place.

    The blur happens on a small cached tile: blurring the full frame for every
    character on every frame is what makes naive renderers crawl.
    """
    rx, ry = max(2.0, abs(rx)), max(2.0, abs(ry))
    pad = int(max(4.0, blur * 2.5))
    tw, th = int(rx * 2 + pad * 2), int(ry * 2 + pad * 2)
    key = (tw, th, int(blur), int(alpha))
    tile = _shadow_cache.get(key)
    if tile is None:
        base = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        d = ImageDraw.Draw(base)
        d.ellipse([pad, pad, tw - pad, th - pad], fill=(10, 8, 20, alpha))
        tile = base.filter(ImageFilter.GaussianBlur(max(1.0, blur)))
        _shadow_cache[key] = tile
    pos = (int(centre[0] - tw / 2), int(centre[1] - th / 2))
    if img.mode == "RGBA":
        img.alpha_composite(tile, pos)
    else:
        base = img.convert("RGBA")
        base.alpha_composite(tile, pos)
        img.paste(base.convert(img.mode), (0, 0))


def vignette(img: Image.Image, strength: float = 0.35) -> Image.Image:
    """Darken the corners a little; safe for RGB and RGBA input.

    Uses a cached multiply map: one C-level pass instead of two full-frame blends.
    """
    mode = img.mode
    rgb = img.convert("RGB")
    w, h = rgb.size
    key = (w, h, round(strength, 3))
    darkmap = _vignette_cache.get(key)
    if darkmap is None:
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        d.ellipse([-w * 0.25, -h * 0.2, w * 1.25, h * 1.2], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(max(1.0, min(w, h) * 0.06)))
        floor = int(255 * (1.0 - strength))
        ramp = mask.point(lambda v: floor + (255 - floor) * v // 255)
        darkmap = Image.merge("RGB", (ramp, ramp, ramp))
        _vignette_cache[key] = darkmap
    out = ImageChops.multiply(rgb, darkmap)
    return out.convert(mode)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


_EASE = {
    "linear": lambda t: t,
    "ease_in": lambda t: t * t * t,
    "ease_out": lambda t: 1 - (1 - t) ** 3,
    "ease_in_out": lambda t: 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2,
    "smooth": lambda t: t * t * (3 - 2 * t),
}


def ease(name: str, t: float) -> float:
    t = max(0.0, min(1.0, t))
    return _EASE.get(name or "ease_in_out", _EASE["ease_in_out"])(t)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
