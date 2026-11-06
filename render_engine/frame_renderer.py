"""Composes a single frame: background → props → characters → camera → captions.

One frame is a pure function of (scene, t), so rendering can be interrupted and
resumed at any scene boundary and the output is identical.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter

from character_engine.animation import (action_expression, blink_amount, mouth_amount,
                                        pose_for, resolve_action)
from character_engine.character_rig import CharacterRig
from render_engine.draw_utils import ease, vignette
from scene_engine import backgrounds, objects
from scene_engine.camera import Camera
from scene_engine.subtitles import caption_for_time

GROUND_FRACTION = 0.87
CHAR_HEIGHT_FRACTION = 0.50


class FrameRenderer:
    def __init__(self, characters: Dict[str, dict], camera: Camera, cues: Optional[List[dict]] = None,
                 fps: int = 30):
        self.characters = characters or {}
        self.camera = camera
        self.cues = cues or []
        self.fps = fps
        self.rigs = {cid: CharacterRig(spec) for cid, spec in self.characters.items()}
        self._bg_cache: Dict[str, Image.Image] = {}

    # ------------------------------------------------------------------ #
    def _background(self, scene: dict, cam_state: dict) -> Image.Image:
        """Cached painted background, composited with a small parallax offset.

        Drawing every wall, bar and light beam from scratch on every frame is the
        single most expensive thing a renderer like this can do, so each distinct
        background is painted once per run into an oversized canvas and then
        blitted with an offset.
        """
        w, h = self.camera.world_w, self.camera.world_h
        bg_spec = scene.get("background") or {}

        image_path = bg_spec.get("image")
        if image_path and not os.path.isabs(image_path):
            image_path = os.path.join(scene.get("_project_dir", "."), image_path)
        if image_path and os.path.exists(image_path):
            cached = self._bg_cache.get(image_path)
            if cached is None:
                cached = backgrounds.load_image_background(
                    image_path, (w, h), (cam_state["pan_x"], cam_state["pan_y"]))
                if cached is not None:
                    self._bg_cache[image_path] = cached
            if cached is not None:
                return cached.copy()

        kind = bg_spec.get("kind", "plain")
        palette = tuple(bg_spec.get("palette") or ())
        pink = round(float(bg_spec.get("pink", 0.0) or 0.0), 1)
        glow = bg_spec.get("glow")
        key = ("painted", kind, palette, pink, glow, w, h)

        margin = int(w * 0.06)
        entry = self._bg_cache.get(key)
        if entry is None:
            big = Image.new("RGBA", (w + margin * 2, h + margin * 2), (0, 0, 0, 255))
            backgrounds.draw_background(big, kind, list(palette) or None, t=0.0,
                                        parallax=(0.0, 0.0),
                                        extra={"pink": pink, "glow": glow})
            entry = (big, margin)
            self._bg_cache[key] = entry

        big, margin = entry
        # the painted canvas is opaque, so a crop is the parallax — far cheaper
        # than compositing a 1700x3100 layer every frame
        dx = int(margin - cam_state["pan_x"] * w * 0.45)
        dy = int(margin - cam_state["pan_y"] * h * 0.45)
        dx = max(0, min(big.width - w, dx))
        dy = max(0, min(big.height - h, dy))
        return big.crop((dx, dy, dx + w, dy + h))

    # ------------------------------------------------------------------ #
    def _character_ground(self, entry: dict, t: float, duration: float) -> Tuple[float, float, float]:
        """Returns (x_px, ground_y_px, height_px) for a character entry."""
        w, h = self.camera.world_w, self.camera.world_h
        x_norm = float(entry.get("x", 0.0) or 0.0)
        if entry.get("from_x") is not None and entry.get("to_x") is not None:
            p = min(1.0, max(0.0, t / max(duration, 1e-6)))
            x_norm = float(entry["from_x"]) + (float(entry["to_x"]) - float(entry["from_x"])) * ease("smooth", p)
        depth = float(entry.get("depth", 1.0) or 1.0)
        scale = float(entry.get("scale", 1.0) or 1.0)
        x_px = w * (0.5 + x_norm * 0.30)
        ground_y = h * (GROUND_FRACTION + float(entry.get("y", 0.0) or 0.0)) + (depth - 1.0) * h * 0.04
        height_px = h * CHAR_HEIGHT_FRACTION * scale * (0.82 + 0.18 * depth)
        return x_px, ground_y, height_px

    # ------------------------------------------------------------------ #
    def render(self, scene: dict, t_local: float, t_global: float,
               prev_frame: Optional[Image.Image] = None) -> Image.Image:
        duration = float(scene.get("duration", 1.0))
        p = min(1.0, max(0.0, t_local / duration)) if duration else 0.0
        cam_state = self.camera.state((scene.get("camera") or {}).get("move", "static"), p,
                                      scene.get("camera") or {})

        world = self._background(scene, cam_state)
        draw = ImageDraw.Draw(world, "RGBA")

        entries = list(scene.get("characters") or [])
        # far characters first
        entries.sort(key=lambda e: float(e.get("depth", 1.0) or 1.0))

        # props that sit behind the cast
        prop_state = scene.get("_prop_state") or {}
        for prop in scene.get("objects") or []:
            if prop.get("layer", "front") == "back":
                self._draw_prop(world, prop, {}, cam_state, t_local)

        joints_by_char: Dict[str, dict] = {}
        for entry in entries:
            cid = entry.get("id")
            rig = self.rigs.get(cid)
            if rig is None:
                continue
            x_px, ground_y, height_px = self._character_ground(entry, t_local, duration)
            action = resolve_action(entry.get("action", "stand"))
            pose = pose_for(action, t_local, duration, blend=0.0,
                            previous=entry.get("previous_action"))
            expression = entry.get("expression") or action_expression(action) or "neutral"
            talking = bool(scene.get("narration")) and entry.get("talking", True) is not False
            state = {
                "pose": pose,
                "facing": entry.get("facing", 1),
                "expression": expression,
                "mouth": mouth_amount(t_local, talking) if action in ("talk", "stand", "idle", "walk", "sit") else 0.0,
                "blink": blink_amount(t_global, hash(cid) % 7),
            }
            joints = rig.draw(world, (x_px, ground_y), height_px, state)
            joints_by_char[cid] = {"joints": joints, "height": height_px, "x": x_px, "ground_y": ground_y}

        # props in front, including hand-attached ones
        for prop in scene.get("objects") or []:
            if prop.get("layer", "front") != "back":
                self._draw_prop(world, prop, joints_by_char, cam_state, t_local)

        frame = self.camera.apply(world, cam_state)

        # captions + grade
        caption = caption_for_time(self.cues, t_global)
        if caption:
            from scene_engine.subtitles import draw_caption
            draw_caption(frame, caption)
        return vignette(frame, 0.22) if scene.get("vignette", True) else frame

    # ------------------------------------------------------------------ #
    def _draw_prop(self, world: Image.Image, prop: dict, joints_by_char: dict,
                   cam_state: dict, t: float) -> None:
        anchor_spec = str(prop.get("attached_to") or "")
        scale = float(prop.get("scale", 1.0) or 1.0)
        rotation = float(prop.get("rotation", 0.0) or 0.0)
        w, h = self.camera.world_w, self.camera.world_h

        if "." in anchor_spec:
            cid, joint_name = anchor_spec.split(".", 1)
            info = joints_by_char.get(cid)
            if info and joint_name in info["joints"]:
                anchor = info["joints"][joint_name]
                base_scale = info["height"]
            else:
                anchor = (w * (0.5 + float(prop.get("x", 0.0)) * 0.30), h * GROUND_FRACTION)
                base_scale = h * CHAR_HEIGHT_FRACTION
        else:
            anchor = (w * (0.5 + float(prop.get("x", 0.0) or 0.0) * 0.30),
                      h * (GROUND_FRACTION + float(prop.get("y", 0.0) or 0.0)))
            base_scale = h * CHAR_HEIGHT_FRACTION * scale

        objects.draw_object(world, prop.get("kind", "box"), anchor,
                            base_scale * scale, prop.get("state") or {}, rotation)
