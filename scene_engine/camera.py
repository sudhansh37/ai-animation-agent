"""Virtual camera for a vertical 9:16 frame.

The scene is drawn onto an oversized *world* canvas; the camera crops a window
out of it and resizes that to the output resolution. Zoom, pan and tilt are
therefore real (the framing changes), and every camera move is computed for the
1080x1920 composition — the window is always 9:16 so nothing is ever stretched
into a landscape frame.
"""

from __future__ import annotations

import math
from typing import Tuple

from PIL import Image

from render_engine.draw_utils import ease

MOVES = [
    "static", "push_in", "pull_out", "pan_left", "pan_right", "tilt_up", "tilt_down",
    "track", "dolly", "low_angle", "high_angle", "side_angle", "over_shoulder",
]

ANGLES = {
    "low_angle": {"pan_y": 0.06, "zoom": 1.04},
    "high_angle": {"pan_y": -0.07, "zoom": 1.02},
    "side_angle": {"pan_x": 0.05, "zoom": 1.02},
    "over_shoulder": {"pan_x": -0.07, "pan_y": 0.02, "zoom": 1.10},
}


class Camera:
    def __init__(self, out_w: int = 1080, out_h: int = 1920, world_scale: float = 1.35):
        self.out_w = out_w
        self.out_h = out_h
        self.scale = world_scale
        self.world_w = int(round(out_w * world_scale))
        self.world_h = int(round(out_h * world_scale))

    # ------------------------------------------------------------------ #
    def state(self, move: str, p: float, spec: dict | None = None) -> dict:
        """Camera parameters for progress p (0..1) through a shot."""
        spec = spec or {}
        move = (move or "static").lower().replace("-", "_").replace(" ", "_")
        zoom = float(spec.get("zoom", 1.0) or 1.0)
        pan_x = float(spec.get("pan_x", 0.0) or 0.0)
        pan_y = float(spec.get("pan_y", 0.0) or 0.0)
        roll = float(spec.get("roll", 0.0) or 0.0)

        e = ease(spec.get("easing", "ease_in_out"), p)
        if move == "push_in":
            zoom *= 1.0 + 0.14 * e
        elif move == "pull_out":
            zoom *= 1.16 - 0.14 * e
        elif move == "dolly":
            zoom *= 1.0 + 0.22 * e
            pan_y += 0.02 * e
        elif move == "pan_left":
            pan_x += 0.07 * (0.5 - e)
        elif move == "pan_right":
            pan_x += 0.07 * (e - 0.5)
        elif move == "track":
            pan_x += 0.05 * math.sin(e * math.pi)
        elif move == "tilt_up":
            pan_y += -0.05 + 0.10 * e
        elif move == "tilt_down":
            pan_y += 0.05 - 0.10 * e
        elif move in ANGLES:
            a = ANGLES[move]
            zoom *= a.get("zoom", 1.0)
            pan_x += a.get("pan_x", 0.0)
            pan_y += a.get("pan_y", 0.0)
            if move == "low_angle":
                roll += 0.0

        # handheld drift keeps the frame alive
        zoom *= 1.0 + 0.0015 * math.sin(p * 9.0)
        pan_x += 0.0012 * math.sin(p * 7.0 + 1.2)
        pan_y += 0.0010 * math.sin(p * 5.5 + 0.4)

        return {"zoom": max(1.0, zoom), "pan_x": pan_x, "pan_y": pan_y, "roll": roll}

    # ------------------------------------------------------------------ #
    def crop_box(self, state: dict) -> Tuple[int, int, int, int]:
        zoom = max(1.0, state["zoom"])
        win_w = self.world_w / zoom
        win_h = self.world_h / zoom
        cx = self.world_w * (0.5 + state["pan_x"])
        cy = self.world_h * (0.5 + state["pan_y"])
        left = cx - win_w / 2
        top = cy - win_h / 2
        left = max(0, min(self.world_w - win_w, left))
        top = max(0, min(self.world_h - win_h, top))
        return (int(round(left)), int(round(top)), int(round(left + win_w)), int(round(top + win_h)))

    def apply(self, world_img: Image.Image, state: dict) -> Image.Image:
        box = self.crop_box(state)
        crop = world_img.crop(box)
        # area-average (BOX) when shrinking: sharper than bilinear and ~3x faster
        # than Lanczos, which matters when every frame goes through here
        method = Image.BOX if crop.width > self.out_w * 1.12 else Image.BICUBIC
        frame = crop.resize((self.out_w, self.out_h), method)
        roll = state.get("roll", 0.0)
        if roll:
            frame = frame.rotate(math.degrees(roll), resample=Image.BICUBIC, expand=False)
        return frame


def camera_list() -> list:
    return MOVES
