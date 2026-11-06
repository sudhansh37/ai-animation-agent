"""Props and object interaction.

Objects are drawn relative to an anchor. When an object is attached to a hand
(`"attached_to": "scientist.hand_r"`), the anchor follows the rig's hand joint
every frame, so a vial or clipboard stays in the hand as the arm swings — the
object interaction the spec asks for, without a physics engine.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from PIL import Image, ImageDraw

from render_engine.draw_utils import (capsule, ellipse, hex_to_rgb, polygon,
                                      rounded_rect, shade)

KINDS = ["door", "vial", "clipboard", "baton", "bucket", "brush", "box", "cup",
         "phone", "key", "chair", "table", "sign", "bars", "bed", "paint_can", "ladder"]

INK = "#241726"


def draw_object(img: Image.Image, kind: str, anchor: Tuple[float, float], scale: float,
                state: Optional[dict] = None, rotation: float = 0.0) -> None:
    """`scale` is pixels per world unit (character height); objects are ~0.1-0.5 of that."""
    state = state or {}
    d = ImageDraw.Draw(img, "RGBA")
    x, y = anchor
    s = scale
    ink_w = max(1.0, s * 0.010)
    kind = (kind or "box").lower()

    if kind == "vial":
        w, h = s * 0.030, s * 0.090
        rounded_rect(d, [x - w, y - h, x + w, y], w * 0.5, "#bfe9ff", INK, ink_w)
        rounded_rect(d, [x - w * 0.85, y - h * 0.55, x + w * 0.85, y - h * 0.05], w * 0.4,
                     state.get("liquid", "#ff7fb0"), INK, ink_w * 0.7)
        rounded_rect(d, [x - w * 0.5, y - h * 1.15, x + w * 0.5, y - h], w * 0.2, "#8a8f9a", INK, ink_w * 0.7)

    elif kind == "clipboard":
        w, h = s * 0.075, s * 0.100
        rounded_rect(d, [x - w, y - h, x + w, y + h], w * 0.12, "#c8a878", INK, ink_w)
        rounded_rect(d, [x - w * 0.78, y - h * 0.82, x + w * 0.78, y + h * 0.82], w * 0.06, "#f6f2e8", INK, ink_w * 0.5)
        for i in range(3):
            yy = y - h * 0.45 + i * h * 0.36
            d.line([(x - w * 0.55, yy), (x + w * 0.55, yy)], fill="#8a94a8", width=max(1, int(ink_w * 0.8)))
        rounded_rect(d, [x - w * 0.3, y - h * 1.05, x + w * 0.3, y - h * 0.85], w * 0.1, "#9aa2b0", INK, ink_w * 0.6)

    elif kind == "baton":
        capsule(d, (x, y), (x + s * 0.008, y - s * 0.18), s * 0.020, "#3a3f4a", INK, ink_w)

    elif kind == "bucket":
        w, h = s * 0.070, s * 0.085
        polygon(d, [(x - w, y - h), (x + w, y - h), (x + w * 0.8, y), (x - w * 0.8, y)],
                state.get("colour", "#8a929e"), INK, ink_w)
        ellipse(d, x, y - h, w * 0.85, h * 0.22, shade(state.get("colour", "#8a929e"), 0.2), INK, ink_w * 0.7)

    elif kind == "brush":
        capsule(d, (x, y), (x, y - s * 0.16), s * 0.014, "#a8743a", INK, ink_w)
        rounded_rect(d, [x - s * 0.016, y - s * 0.20, x + s * 0.016, y - s * 0.155], s * 0.004,
                     state.get("paint", "#ff7fb0"), INK, ink_w * 0.6)

    elif kind == "paint_can":
        w, h = s * 0.080, s * 0.100
        rounded_rect(d, [x - w, y - h, x + w, y], w * 0.12, "#b0b6c0", INK, ink_w)
        ellipse(d, x, y - h, w * 0.92, h * 0.20, "#d8dde6", INK, ink_w * 0.7)
        ellipse(d, x, y - h, w * 0.62, h * 0.13, state.get("colour", "#ff7fb0"), INK, ink_w * 0.5)

    elif kind == "door":
        w, h = s * 0.16, s * 0.42
        rounded_rect(d, [x - w, y - h, x + w, y], w * 0.06, "#6a5238", INK, ink_w)
        ellipse(d, x + w * 0.7, y - h * 0.45, w * 0.07, w * 0.07, "#d8b25a", INK, ink_w * 0.6)
        if state.get("open", 0) > 0.01:
            ow = w * 2 * float(state["open"])
            polygon(d, [(x + w, y - h), (x + w + ow, y - h * 0.92), (x + w + ow, y - h * 0.08), (x + w, y)],
                    shade("#6a5238", -0.25), INK, ink_w)

    elif kind == "box":
        w, h = s * 0.080, s * 0.075
        rounded_rect(d, [x - w, y - h, x + w, y], w * 0.1, state.get("colour", "#c8a878"), INK, ink_w)
        d.line([(x - w, y - h * 0.62), (x + w, y - h * 0.62)], fill=INK, width=int(ink_w))

    elif kind == "cup":
        w, h = s * 0.030, s * 0.045
        rounded_rect(d, [x - w, y - h, x + w, y], w * 0.3, "#f0f0f0", INK, ink_w)

    elif kind == "phone":
        w, h = s * 0.028, s * 0.055
        rounded_rect(d, [x - w, y - h, x + w, y + h], w * 0.25, "#2a2f38", INK, ink_w)
        rounded_rect(d, [x - w * 0.8, y - h * 0.75, x + w * 0.8, y + h * 0.7], w * 0.15, "#7fd8ff", None, 0)

    elif kind == "key":
        capsule(d, (x, y), (x + s * 0.06, y), s * 0.010, "#d8b25a", INK, ink_w)
        ellipse(d, x, y, s * 0.018, s * 0.018, "#d8b25a", INK, ink_w)
        d.line([(x + s * 0.05, y), (x + s * 0.05, y + s * 0.018)], fill="#d8b25a", width=int(ink_w * 1.4))

    elif kind == "chair":
        w, h = s * 0.09, s * 0.16
        rounded_rect(d, [x - w, y - h, x + w, y - h * 0.55], w * 0.1, "#8a6a4a", INK, ink_w)
        rounded_rect(d, [x - w, y - h * 0.55, x + w, y - h * 0.42], w * 0.1, "#a07a52", INK, ink_w)
        for dx in (-w * 0.85, w * 0.85):
            d.line([(x + dx, y - h * 0.42), (x + dx, y)], fill=INK, width=int(ink_w * 1.6))

    elif kind == "table":
        w, h = s * 0.20, s * 0.14
        rounded_rect(d, [x - w, y - h, x + w, y - h * 0.82], w * 0.05, "#8a6a4a", INK, ink_w)
        for dx in (-w * 0.85, w * 0.85):
            d.line([(x + dx, y - h * 0.82), (x + dx, y)], fill=INK, width=int(ink_w * 1.8))

    elif kind == "sign":
        w, h = s * 0.14, s * 0.09
        rounded_rect(d, [x - w, y - h, x + w, y], w * 0.1, state.get("colour", "#e8e2d0"), INK, ink_w)
        d.line([(x, y), (x, y + s * 0.16)], fill="#6a6a72", width=int(ink_w * 1.8))

    elif kind == "bars":
        for i in range(5):
            bx = x - s * 0.18 + i * s * 0.09
            capsule(d, (bx, y), (bx, y - s * 0.9), s * 0.014, "#59616e", INK, ink_w * 0.8)

    elif kind == "bed":
        w, h = s * 0.20, s * 0.10
        rounded_rect(d, [x - w, y - h, x + w, y], w * 0.08, "#6b7280", INK, ink_w)
        rounded_rect(d, [x - w * 0.9, y - h * 1.25, x + w * 0.9, y - h], w * 0.08, "#c9c2b4", INK, ink_w)

    elif kind == "ladder":
        for dx in (-s * 0.05, s * 0.05):
            d.line([(x + dx, y), (x + dx, y - s * 0.5)], fill=INK, width=int(ink_w * 1.6))
        for i in range(5):
            yy = y - s * 0.05 - i * s * 0.10
            d.line([(x - s * 0.05, yy), (x + s * 0.05, yy)], fill="#a8743a", width=int(ink_w * 1.6))

    else:  # unknown prop: a small crate, never a crash
        w, h = s * 0.06, s * 0.06
        rounded_rect(d, [x - w, y - h, x + w, y], w * 0.15, "#a08a6a", INK, ink_w)
