"""Procedural backgrounds.

Backgrounds are drawn per frame so parallax and lighting can move, but they use
a small, fixed set of shapes so the cost stays low. A user-supplied background
image always wins: put `assets/backgrounds/<kind>.png` (or set
`"image": "path.png"` on the scene) and it is used as-is, with the characters
composited on top — the reference is preserved and nothing is added to it.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

from render_engine.draw_utils import (ellipse, hex_to_rgb, mix, polygon, radial_glow,
                                      rounded_rect, shade, vertical_gradient)

KINDS = [
    "lab", "prison_cell", "prison_corridor", "prison_yard", "office",
    "city", "forest", "plain", "night_sky", "room", "wall",
    # added for the reference-style clone
    "cell", "paint_shop", "corridor", "canteen",
]

PALETTES: Dict[str, List[str]] = {
    "lab": ["#1b2436", "#33415c", "#8fa6c8"],
    "prison_cell": ["#2a2f38", "#4a525e", "#8d99a8"],
    "prison_corridor": ["#242a33", "#3e4753", "#7c8794"],
    "prison_yard": ["#4a6a8a", "#8fa6b8", "#c8c0a8"],
    "office": ["#2a3242", "#4a5468", "#b8c2d4"],
    "city": ["#1a2438", "#3a4a68", "#8aa0c0"],
    "forest": ["#243a2a", "#3f6a44", "#9ac08a"],
    "plain": ["#5a7ab0", "#a8c0d8", "#d8e0c0"],
    "night_sky": ["#0e1630", "#243a68", "#5a7ab0"],
    "room": ["#3a3242", "#5a5060", "#b8a8b0"],
    "wall": ["#333a44", "#4a525e", "#9aa4b0"],
    "cell": ["#20404f", "#2f5d70", "#8fb8c8"],
    "paint_shop": ["#dfe4e0", "#b9c4c0", "#6d7a76"],
    "corridor": ["#1d3442", "#2a4c5c", "#9ec4d2"],
    "canteen": ["#22323f", "#33566a", "#a8c4d2"],
}


def _bricks(draw: ImageDraw.ImageDraw, w: int, h: int, top: float, bottom: float,
            colour: str, mortar: str, bw: float = 0.075, bh: float = 0.030) -> None:
    """Running-bond brick wall — the reference rooms are all tiled like this."""
    draw.rectangle([0, top, w, bottom], fill=colour)
    rows = max(1, int((bottom - top) / (h * bh)))
    for r in range(rows + 1):
        y = top + r * h * bh
        offset = (w * bw * 0.5) if r % 2 else 0
        draw.line([(0, y), (w, y)], fill=mortar, width=max(1, int(h * 0.0035)))
        x = -offset
        while x < w:
            draw.line([(x, y), (x, y + h * bh)], fill=mortar, width=max(1, int(h * 0.0035)))
            x += w * bw


def _ceiling_light(img: Image.Image, x: float, y: float, w: float, h: float) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    rounded_rect(d, [x - w * 0.5, y, x + w * 0.5, y + h * 0.10], w * 0.03, "#e8eef2", "#8fa4b0", max(1, w * 0.012))
    for i in range(3):
        d.rectangle([x - w * 0.42, y + h * 0.02 + i * h * 0.022, x + w * 0.42, y + h * 0.036 + i * h * 0.022],
                    fill=(255, 252, 235, 235))


def _window(img: Image.Image, x: float, y: float, w: float, h: float, bars: int = 4) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    rounded_rect(d, [x - w / 2, y, x + w / 2, y + h], w * 0.02, "#cfe4ef", "#6f8b9a", max(2, w * 0.02))
    for i in range(1, bars):
        bx = x - w / 2 + i * w / bars
        d.line([(bx, y), (bx, y + h)], fill="#8fa4b0", width=max(2, int(w * 0.018)))
    d.line([(x - w / 2, y + h * 0.5), (x + w / 2, y + h * 0.5)], fill="#8fa4b0", width=max(2, int(w * 0.018)))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _floor(draw: ImageDraw.ImageDraw, w: int, h: int, horizon: float, colour: str,
           lines: int = 7, line_colour: Optional[str] = None) -> None:
    """Floor plane with soft perspective lines."""
    draw.rectangle([0, horizon, w, h], fill=colour)
    lc = line_colour or shade(colour, -0.18)
    cx = w / 2
    for i in range(-lines, lines + 1):
        x_far = cx + i * (w * 0.045)
        x_near = cx + i * (w * 0.34)
        draw.line([(x_far, horizon), (x_near, h)], fill=lc, width=max(1, int(w * 0.0016)))
    for i in range(1, 6):
        t = (i / 6.0) ** 2.1
        y = horizon + (h - horizon) * t
        draw.line([(0, y), (w, y)], fill=lc, width=max(1, int(w * 0.0012)))


def _light_shaft(img: Image.Image, x: float, y: float, w: float, h: float, colour: str = "#fff6d0",
                 alpha: int = 46) -> None:
    """Soft beam built from a few nested polygons — no per-frame blur."""
    d = ImageDraw.Draw(img, "RGBA")
    r, g, b = hex_to_rgb(colour)
    steps = 4
    for i in range(steps):
        t = i / steps
        spread = 0.10 + 0.16 * t
        a = int(alpha * (0.25 + 0.75 * (1 - t)) / steps)
        d.polygon([(x - w * spread * 0.5, y), (x + w * spread * 0.5, y),
                   (x + w * spread * 1.5, y + h), (x - w * spread * 1.5, y + h)],
                  fill=(r, g, b, max(2, a)))


def _bars(img: Image.Image, x0: float, x1: float, y0: float, y1: float, spacing: float,
          width: float, colour: str = "#59616e") -> None:
    d = ImageDraw.Draw(img, "RGBA")
    for x in [x0 + i * spacing for i in range(int((x1 - x0) / spacing) + 1)]:
        d.line([(x, y0), (x, y1)], fill=colour, width=int(width))
    d.line([(x0, y0), (x1, y0)], fill=colour, width=int(width * 1.1))
    d.line([(x0, y1), (x1, y1)], fill=colour, width=int(width * 1.1))


# --------------------------------------------------------------------------- #
# the background library
# --------------------------------------------------------------------------- #

def draw_background(img: Image.Image, kind: str, palette: Optional[List[str]] = None,
                    t: float = 0.0, parallax: Tuple[float, float] = (0.0, 0.0),
                    extra: Optional[dict] = None) -> None:
    kind = (kind or "plain").lower()
    extra = extra or {}
    colours = palette or PALETTES.get(kind, PALETTES["plain"])
    top, mid, low = (colours + ["#888888"] * 3)[:3]
    w, h = img.size

    if img.mode != "RGBA":
        img = img.convert("RGBA") if img.mode == "RGB" else img
    d = ImageDraw.Draw(img, "RGBA")

    horizon = h * 0.66
    ox = parallax[0] * w
    oy = parallax[1] * h

    if kind == "lab":
        bg = vertical_gradient((w, h), top, mid).convert("RGBA")
        img.paste(bg, (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        _floor(d, w, h, horizon, shade(mid, -0.22))
        # back wall panels
        for i in range(6):
            x = ox * 0.4 + i * w / 6
            d.rectangle([x, h * 0.10, x + w / 6 - w * 0.008, horizon], outline=shade(mid, 0.10), width=max(2, int(w * 0.003)))
        # workbench
        rounded_rect(d, [w * 0.08, horizon - h * 0.06, w * 0.92, horizon + h * 0.02], w * 0.01,
                     shade(mid, -0.10), shade(mid, -0.45), w * 0.004)
        # beakers on the bench
        for i, col in enumerate(["#7fe0c0", "#ffd166", "#8ab6ff", "#ff8fa3"]):
            bx = w * (0.18 + i * 0.16)
            by = horizon - h * 0.06
            rounded_rect(d, [bx - w * 0.02, by - h * 0.06, bx + w * 0.02, by], w * 0.008, col, "#0f1622", w * 0.003)
            ellipse(d, bx, by - h * 0.062, w * 0.02, h * 0.008, shade(col, 0.25), "#0f1622", w * 0.003)
        # wall screen
        rounded_rect(d, [w * 0.62, h * 0.16, w * 0.90, h * 0.34], w * 0.01, "#12303a", "#0b1a20", w * 0.004)
        for i in range(4):
            y = h * (0.19 + i * 0.035)
            d.line([(w * 0.65, y), (w * (0.72 + 0.04 * math.sin(t + i))), y],
                   fill="#3fe0c8", width=max(2, int(h * 0.004)))
        _light_shaft(img, w * 0.5, h * 0.05, w * 0.30, h * 0.55, "#cfe6ff", 26)

    elif kind == "prison_cell":
        bg = vertical_gradient((w, h), top, shade(mid, -0.15)).convert("RGBA")
        img.paste(bg, (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        _floor(d, w, h, horizon, shade(mid, -0.30))
        # bunk
        rounded_rect(d, [w * 0.10, horizon - h * 0.10, w * 0.52, horizon - h * 0.02], w * 0.008,
                     "#6b7280", "#2a2f38", w * 0.004)
        rounded_rect(d, [w * 0.12, horizon - h * 0.135, w * 0.50, horizon - h * 0.095], w * 0.012,
                     "#c9c2b4", "#2a2f38", w * 0.004)
        # barred window with light
        wx0, wy0, wx1, wy1 = w * 0.62, h * 0.18, w * 0.88, h * 0.38
        d.rectangle([wx0, wy0, wx1, wy1], fill="#8fb6d8", outline="#2a2f38", width=int(w * 0.006))
        _bars(img, wx0, wx1, wy0, wy1, (wx1 - wx0) / 4, w * 0.008, "#3a4048")
        _light_shaft(img, (wx0 + wx1) / 2, wy1, w * 0.26, h * 0.45, "#ffe9b0", 40)
        # bucket
        rounded_rect(d, [w * 0.74, horizon + h * 0.02, w * 0.84, horizon + h * 0.09], w * 0.01,
                     "#5a6270", "#232830", w * 0.004)

    elif kind == "prison_corridor":
        bg = vertical_gradient((w, h), top, mid).convert("RGBA")
        img.paste(bg, (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        # converging walls
        vanish = (w * 0.5 + ox * 0.3, h * 0.46)
        polygon(d, [(0, 0), (vanish[0], vanish[1] - h * 0.06), (vanish[0], vanish[1] + h * 0.06), (0, h)],
                shade(mid, -0.18))
        polygon(d, [(w, 0), (vanish[0], vanish[1] - h * 0.06), (vanish[0], vanish[1] + h * 0.06), (w, h)],
                shade(mid, -0.26))
        polygon(d, [(0, h), (vanish[0], vanish[1] + h * 0.06), (w, h)], shade(mid, -0.34))
        # doors down the corridor
        for i in range(5):
            k = (i + 1) / 6
            dw = w * 0.10 * (1 - k * 0.6)
            dh = h * 0.30 * (1 - k * 0.6)
            x = w * 0.5 + (vanish[0] - w * 0.5) * k - dw / 2 + (1 if i % 2 else -1) * w * 0.12 * (1 - k)
            y = vanish[1] + h * 0.02 - dh * 0.15
            rounded_rect(d, [x, y - dh, x + dw, y], dw * 0.08, shade(mid, -0.05), "#1a1f26", w * 0.004)
        # ceiling lights
        for i in range(3):
            y = h * (0.06 + i * 0.10)
            rounded_rect(d, [w * 0.38, y, w * 0.62, y + h * 0.012], w * 0.006, "#fff2c4", None, 0)
            _light_shaft(img, w * 0.5, y + h * 0.012, w * 0.30, h * 0.30, "#fff2c4", 22)

    elif kind == "prison_yard":
        img.paste(vertical_gradient((w, h), top, low).convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        # ground
        polygon(d, [(0, horizon), (w, horizon), (w, h), (0, h)], shade("#b8a888", -0.05))
        # wall + fence
        d.rectangle([0, horizon - h * 0.16, w, horizon], shade("#8d99a8", -0.25))
        for x in range(0, w, int(w * 0.05)):
            d.line([(x, horizon - h * 0.30), (x, horizon - h * 0.02)], fill="#4a525e", width=int(w * 0.004))
        d.line([(0, horizon - h * 0.30), (w, horizon - h * 0.30)], fill="#4a525e", width=int(w * 0.006))
        d.line([(0, horizon - h * 0.16), (w, horizon - h * 0.16)], fill="#4a525e", width=int(w * 0.006))
        # watch tower
        tx = w * 0.78
        rounded_rect(d, [tx - w * 0.05, horizon - h * 0.34, tx + w * 0.05, horizon - h * 0.24], w * 0.008,
                     "#5a6270", "#2a2f38", w * 0.004)
        d.line([(tx - w * 0.03, horizon - h * 0.24), (tx - w * 0.05, horizon)], fill="#3a4048", width=int(w * 0.008))
        d.line([(tx + w * 0.03, horizon - h * 0.24), (tx + w * 0.05, horizon)], fill="#3a4048", width=int(w * 0.008))

    elif kind == "office":
        img.paste(vertical_gradient((w, h), top, mid).convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        _floor(d, w, h, horizon, shade(mid, -0.28))
        rounded_rect(d, [w * 0.15, horizon - h * 0.05, w * 0.85, horizon + h * 0.01], w * 0.008,
                     "#8a6a4a", "#3a2a1c", w * 0.004)
        rounded_rect(d, [w * 0.55, h * 0.18, w * 0.85, h * 0.36], w * 0.006, "#22303c", "#101820", w * 0.004)

    elif kind == "city":
        img.paste(vertical_gradient((w, h), top, low).convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        for i in range(9):
            bw = w * (0.09 + (i % 3) * 0.03)
            bh = h * (0.20 + ((i * 37) % 50) / 100)
            x = i * w * 0.11 - ox * 0.2
            d.rectangle([x, horizon - bh, x + bw, horizon], fill=shade(mid, -0.12 - (i % 3) * 0.06))
            for wy in range(int(bh / (h * 0.05))):
                for wx in range(2):
                    d.rectangle([x + bw * (0.22 + wx * 0.36), horizon - bh + h * 0.03 + wy * h * 0.05,
                                 x + bw * (0.38 + wx * 0.36), horizon - bh + h * 0.05 + wy * h * 0.05],
                                fill="#ffe9a8" if (wx + wy + i) % 3 else "#2a3a4a")
        d.rectangle([0, horizon, w, h], fill=shade(mid, -0.35))

    elif kind == "forest":
        img.paste(vertical_gradient((w, h), top, low).convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, horizon, w, h], fill=shade("#3f6a44", -0.15))
        for i in range(11):
            x = (i * w * 0.1 + ox * 0.3) % w
            th = h * (0.28 + ((i * 53) % 30) / 100)
            polygon(d, [(x, horizon - th), (x - w * 0.05, horizon), (x + w * 0.05, horizon)],
                    shade("#2f5a36", -0.05 - (i % 3) * 0.05))

    elif kind == "night_sky":
        img.paste(vertical_gradient((w, h), top, low).convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        for i in range(70):
            x = ((i * 137) % w)
            y = ((i * 271) % int(h * 0.7))
            r = 1.6 + (i % 3)
            d.ellipse([x, y, x + r, y + r], fill="#ffffff" if i % 4 else "#ffe9a8")
        ellipse(d, w * 0.78, h * 0.14, w * 0.09, w * 0.09, "#f2f0e0", None, 0)
        d.rectangle([0, horizon, w, h], fill=shade(mid, -0.4))

    elif kind == "room":
        img.paste(vertical_gradient((w, h), top, mid).convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        _floor(d, w, h, horizon, shade("#7a5a3a", -0.2))
        rounded_rect(d, [w * 0.58, h * 0.20, w * 0.86, h * 0.42], w * 0.008, "#a8c8e0", "#2a3038", w * 0.005)
        d.line([(w * 0.72, h * 0.20), (w * 0.72, h * 0.42)], fill="#2a3038", width=int(w * 0.005))
        rounded_rect(d, [w * 0.10, horizon - h * 0.02, w * 0.40, horizon + h * 0.06], w * 0.01,
                     "#6a4a3a", "#2a1c14", w * 0.004)

    elif kind == "cell":
        img.paste(vertical_gradient((w, h), shade(top, 0.10), mid).convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        _bricks(d, w, h, h * 0.16, h * 0.80, mid, shade(mid, -0.30))
        _floor(d, w, h, h * 0.80, shade(mid, -0.22), lines=4, line_colour=shade(mid, -0.34))
        _ceiling_light(img, w * 0.55, h * 0.02, w * 0.34, h * 0.10)
        _window(img, w * 0.30, h * 0.30, w * 0.26, h * 0.16, bars=3)
        _light_shaft(img, w * 0.30, h * 0.46, w * 0.22, h * 0.38, "#eaf6ff", 20)
        # painted patch on the wall
        patch = float(extra.get("paint", 0.0) or 0.0)
        if patch > 0:
            d.rectangle([w * 0.52, h * 0.30, w * 0.52 + w * 0.22 * patch, h * 0.72],
                        fill=extra.get("paint_colour", "#e0218a"))
        # paint tray on the floor
        rounded_rect(d, [w * 0.18, h * 0.84, w * 0.40, h * 0.90], w * 0.01,
                     "#c8ced4", "#5a6a72", max(2, w * 0.006))
        d.rectangle([w * 0.20, h * 0.85, w * 0.38, h * 0.885], fill=extra.get("paint_colour", "#e0218a"))

    elif kind == "paint_shop":
        img.paste(vertical_gradient((w, h), top, shade(mid, 0.08)).convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        _bricks(d, w, h, h * 0.10, h * 0.66, shade(low, 0.35), shade(low, 0.10), bw=0.10, bh=0.026)
        _floor(d, w, h, h * 0.66, shade("#5a6b74", -0.05), lines=4, line_colour=shade("#5a6b74", -0.2))
        # window with mullions
        _window(img, w * 0.66, h * 0.20, w * 0.30, h * 0.22, bars=3)
        # shelves with paint cans
        for row in range(2):
            sy = h * (0.26 + row * 0.12)
            d.rectangle([w * 0.10, sy, w * 0.44, sy + h * 0.012], fill="#6a5238")
            for i in range(5):
                cx = w * (0.13 + i * 0.062)
                colour = ["#d94f4f", "#4f8fd9", "#f2c14b", "#7fd1a0", "#b47fd1"][(i + row) % 5]
                rounded_rect(d, [cx, sy - h * 0.035, cx + w * 0.038, sy], w * 0.006, colour, "#3a3038", max(1, w * 0.004))
        # PAINT SHOP sign
        rounded_rect(d, [w * 0.12, h * 0.19, w * 0.36, h * 0.235], w * 0.008, "#f4f1ea", "#3a3038", max(1, w * 0.005))
        # work table
        rounded_rect(d, [w * 0.05, h * 0.62, w * 0.95, h * 0.70], w * 0.012, "#b07a3c", "#4a3018", max(2, w * 0.008))
        for lx in (0.14, 0.86):
            d.rectangle([w * lx - w * 0.012, h * 0.70, w * lx + w * 0.012, h * 0.86], fill="#6a4a28")
        # mixing mat + cans on the table
        rounded_rect(d, [w * 0.24, h * 0.63, w * 0.76, h * 0.695], w * 0.006, "#4f8f6a", None, 0)
        for i, colour in enumerate(["#e8e8e8", "#d94f4f"]):
            cx = w * (0.34 + i * 0.20)
            rounded_rect(d, [cx - w * 0.045, h * 0.56, cx + w * 0.045, h * 0.63], w * 0.01, "#c8ced4", "#5a6a72", max(1, w * 0.005))
            ellipse(d, cx, h * 0.56, w * 0.045, h * 0.010, colour, "#5a6a72", max(1, w * 0.004))

    elif kind == "corridor":
        img.paste(vertical_gradient((w, h), shade(top, 0.08), mid).convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        _bricks(d, w, h, h * 0.18, h * 0.78, mid, shade(mid, -0.28), bw=0.085, bh=0.028)
        _floor(d, w, h, h * 0.78, shade("#8a9aa2", 0.10), lines=3, line_colour=shade("#8a9aa2", -0.10))
        # yellow safety line
        d.rectangle([0, h * 0.86, w, h * 0.895], fill="#e8c33a")
        # cell doors
        for i, dx in enumerate((0.18, 0.62)):
            door_x = w * dx
            inner = extra.get("pink_doors", True)
            rounded_rect(d, [door_x, h * 0.30, door_x + w * 0.26, h * 0.80], w * 0.01,
                         "#3d6b7a" if not inner else "#c8d4d8", "#1b2a30", max(2, w * 0.007))
            # opening with the pink cell visible
            d.rectangle([door_x + w * 0.03, h * 0.34, door_x + w * 0.23, h * 0.78],
                        fill="#ef4f9a" if inner else "#22343c")
            d.rectangle([door_x + w * 0.06, h * 0.40, door_x + w * 0.20, h * 0.52],
                        fill="#d8e8f0" if inner else "#1b2a30")
            # plate
            rounded_rect(d, [door_x + w * 0.07, h * 0.26, door_x + w * 0.19, h * 0.295], w * 0.004,
                         "#e8eef2", "#3a4a52", max(1, w * 0.004))
        _ceiling_light(img, w * 0.50, h * 0.02, w * 0.30, h * 0.09)
        _ceiling_light(img, w * 0.14, h * 0.06, w * 0.18, h * 0.07)

    elif kind == "canteen":
        img.paste(vertical_gradient((w, h), top, mid).convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        _bricks(d, w, h, h * 0.20, h * 0.62, shade(mid, 0.10), shade(mid, -0.15), bw=0.09, bh=0.028)
        _floor(d, w, h, h * 0.62, shade("#9aa8ae", 0.05), lines=4, line_colour=shade("#9aa8ae", -0.12))
        # mezzanine railing + pod sign
        for i in range(14):
            d.line([(w * i / 13, h * 0.16), (w * i / 13, h * 0.24)], fill="#7a8f9a", width=max(1, int(w * 0.004)))
        d.line([(0, h * 0.16), (w, h * 0.16)], fill="#7a8f9a", width=max(2, int(w * 0.008)))
        rounded_rect(d, [w * 0.34, h * 0.10, w * 0.66, h * 0.145], w * 0.006, "#dfe8ee", "#3a4a52", max(1, w * 0.004))
        # canteen tables
        for i, ty in enumerate((0.66, 0.80)):
            rounded_rect(d, [w * (0.04 + i * 0.02), h * ty, w * (0.96 - i * 0.02), h * (ty + 0.03)],
                         w * 0.01, "#8fb8d8", "#3a5a6a", max(2, w * 0.006))
            for lx in (0.12, 0.88):
                d.rectangle([w * lx, h * (ty + 0.03), w * lx + w * 0.012, h * (ty + 0.12)], fill="#5a7a8a")
        _ceiling_light(img, w * 0.5, h * 0.02, w * 0.36, h * 0.08)

    else:  # plain / wall
        img.paste(vertical_gradient((w, h), top, mid).convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        _floor(d, w, h, horizon, shade(mid, -0.25))

    # ---- shared grading: pink wash (the story's paint), lighting, haze ----
    pink = float(extra.get("pink", 0.0) or 0.0)
    if pink > 0.001:
        layer = Image.new("RGBA", img.size, (226, 116, 164, int(70 * min(1.0, pink))))
        img.alpha_composite(layer)

    if extra.get("glow"):
        g = radial_glow(int(w * 0.9), extra["glow"], 0.9)
        img.alpha_composite(g, (int(w * 0.05 + ox), int(h * 0.12 + oy)))



def load_image_background(path: str, size: Tuple[int, int], parallax: Tuple[float, float] = (0, 0)) -> Optional[Image.Image]:
    """Load a user background and cover-fit it, preserving the original art."""
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return None
    w, h = size
    scale = max(w / img.width, h / img.height) * 1.08  # slight overscan for parallax
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    left = int((img.width - w) * (0.5 + parallax[0]))
    top = int((img.height - h) * (0.5 + parallax[1]))
    return img.crop((max(0, left), max(0, top), max(0, left) + w, max(0, top) + h))
