"""Facial expressions drawn on top of the head.

Each expression is a small set of shapes (brows, eyes, mouth, extras) so it
composes with any head size, colour or facing direction.
"""

from __future__ import annotations

import math
from typing import Dict

from PIL import ImageDraw

from render_engine.draw_utils import ellipse, polygon, shade

EXPRESSIONS = ["neutral", "happy", "sad", "angry", "surprised", "scared", "confused", "laughing", "crying"]

# brow slope (positive = inner ends up), eye scale, mouth style, extras
EXPRESSION_TABLE: Dict[str, dict] = {
    "neutral":    {"brow": 0.00, "brow_dy": 0.00, "eye": 1.00, "mouth": "line",     "extra": None},
    "happy":      {"brow": -0.18, "brow_dy": -0.02, "eye": 1.05, "mouth": "smile",   "extra": "blush"},
    "sad":        {"brow": 0.28, "brow_dy": 0.02, "eye": 0.88, "mouth": "frown",    "extra": None},
    "angry":      {"brow": -0.42, "brow_dy": 0.03, "eye": 0.82, "mouth": "grit",     "extra": "anger"},
    "surprised":  {"brow": -0.30, "brow_dy": -0.05, "eye": 1.28, "mouth": "o",       "extra": None},
    "scared":     {"brow": 0.34, "brow_dy": -0.03, "eye": 1.30, "mouth": "gasp",     "extra": "sweat"},
    "confused":   {"brow": 0.16, "brow_dy": 0.00, "eye": 1.00, "mouth": "wave",     "extra": "question"},
    "laughing":   {"brow": -0.22, "brow_dy": -0.03, "eye": 0.70, "mouth": "laugh",   "extra": "blush"},
    "crying":     {"brow": 0.30, "brow_dy": 0.03, "eye": 0.80, "mouth": "frown",    "extra": "tears"},
}


def draw_face(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float,
              skin: str, expression: str = "neutral", mouth_open: float = 0.0,
              blink: float = 0.0, facing: int = 1, head_turn: float = 0.0,
              ink: str = "#241726", line: float = 0.0) -> None:
    """Draw eyes, brows and mouth inside a head of radius r centred at (cx, cy)."""
    cfg = EXPRESSION_TABLE.get(expression, EXPRESSION_TABLE["neutral"])
    ink_w = max(1.0, line or r * 0.10)

    eye_dx = r * 0.38
    eye_y = cy - r * 0.02
    eye_r = r * 0.20 * cfg["eye"]
    open_amt = max(0.05, (1.0 - blink) * 1.0)

    # eye whites + irises
    for side in (-1, 1):
        ex = cx + side * eye_dx + head_turn * r * 0.10
        rx = eye_r * 0.92
        ry = eye_r * open_amt
        ellipse(draw, ex, eye_y, rx, ry, "#ffffff", ink, ink_w)
        if open_amt > 0.28:
            pupil_dx = head_turn * r * 0.05 * facing
            ellipse(draw, ex + pupil_dx, eye_y + ry * 0.08, eye_r * 0.58, eye_r * 0.66 * open_amt, ink)
            ellipse(draw, ex + pupil_dx - eye_r * 0.20, eye_y - eye_r * 0.24, eye_r * 0.16, eye_r * 0.16, "#ffffff")

    # brows
    brow_y = eye_y - r * 0.40 + cfg["brow_dy"] * r
    for side in (-1, 1):
        bx = cx + side * eye_dx
        slope = cfg["brow"] * r * 0.34
        x1, x2 = bx - r * 0.22, bx + r * 0.22
        y1 = brow_y + (slope if side < 0 else -slope) * 0.5
        y2 = brow_y - (slope if side < 0 else -slope) * 0.5
        draw.line([(x1, y1), (x2, y2)], fill=ink, width=max(2, int(round(ink_w * 0.9))), joint="curve")

    # mouth
    my = cy + r * 0.42
    mw = r * 0.34
    if mouth_open > 0.08:
        h = r * (0.10 + 0.24 * mouth_open)
        ellipse(draw, cx, my, mw * 0.72, h, "#5d1f2b", ink, ink_w * 0.8)
    else:
        style = cfg["mouth"]
        if style == "smile":
            draw.arc([cx - mw, my - mw * 0.9, cx + mw, my + mw * 0.9], 20, 160, fill=ink,
                     width=max(2, int(round(ink_w))))
        elif style == "laugh":
            ellipse(draw, cx, my, mw * 0.95, mw * 0.8, "#5d1f2b", ink, ink_w * 0.8)
            draw.arc([cx - mw, my - mw * 0.2, cx + mw, my + mw * 1.3], 200, 340, fill="#ffffff",
                     width=max(1, int(round(ink_w * 0.5))))
        elif style == "frown":
            draw.arc([cx - mw, my, cx + mw, my + mw * 1.6], 200, 340, fill=ink,
                     width=max(2, int(round(ink_w))))
        elif style == "grit":
            polygon(draw, [(cx - mw, my - mw * 0.35), (cx + mw, my - mw * 0.35),
                           (cx + mw, my + mw * 0.35), (cx - mw, my + mw * 0.35)],
                    "#f7f2ea", ink, ink_w * 0.7)
            draw.line([(cx - mw * 0.3, my - mw * 0.35), (cx - mw * 0.3, my + mw * 0.35)], fill=ink, width=2)
            draw.line([(cx + mw * 0.3, my - mw * 0.35), (cx + mw * 0.3, my + mw * 0.35)], fill=ink, width=2)
        elif style == "o":
            ellipse(draw, cx, my, mw * 0.42, mw * 0.52, "#5d1f2b", ink, ink_w * 0.8)
        elif style == "gasp":
            ellipse(draw, cx, my, mw * 0.5, mw * 0.68, "#5d1f2b", ink, ink_w * 0.8)
        elif style == "wave":
            draw.line([(cx - mw, my + mw * 0.2), (cx - mw * 0.3, my - mw * 0.2),
                       (cx + mw * 0.3, my + mw * 0.2), (cx + mw, my - mw * 0.2)],
                      fill=ink, width=max(2, int(round(ink_w))), joint="curve")
        else:
            draw.line([(cx - mw * 0.8, my), (cx + mw * 0.8, my)], fill=ink,
                      width=max(2, int(round(ink_w))), joint="curve")

    # extras
    extra = cfg["extra"]
    if extra == "blush":
        for side in (-1, 1):
            ellipse(draw, cx + side * r * 0.52, cy + r * 0.16, r * 0.16, r * 0.09,
                    shade(skin, -0.05), None, 0)
    elif extra == "sweat":
        sx, sy = cx + r * 0.62, cy - r * 0.34
        polygon(draw, [(sx, sy - r * 0.16), (sx + r * 0.09, sy + r * 0.06),
                       (sx, sy + r * 0.14), (sx - r * 0.09, sy + r * 0.06)],
                "#9fd8ff", "#3d7ea6", ink_w * 0.5)
    elif extra == "anger":
        ax, ay = cx + r * 0.60, cy - r * 0.42
        for i in range(2):
            draw.line([(ax - r * 0.10, ay + i * r * 0.10), (ax + r * 0.10, ay + i * r * 0.10 - r * 0.06)],
                      fill="#d8392b", width=max(2, int(round(ink_w * 0.8))))
    elif extra == "tears":
        for side in (-1, 1):
            tx = cx + side * eye_dx
            polygon(draw, [(tx - r * 0.07, eye_y + r * 0.2), (tx + r * 0.07, eye_y + r * 0.2),
                           (tx, eye_y + r * 0.5)], "#9fd8ff", "#3d7ea6", ink_w * 0.4)
    elif extra == "question":
        qx, qy = cx + r * 0.85, cy - r * 0.85
        draw.text((qx, qy), "?", fill="#ffe066", font=None)
