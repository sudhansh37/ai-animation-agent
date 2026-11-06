"""The 2D skeletal character rig.

A character is a small bone hierarchy (hips → spine → neck → head, plus arms and
legs). Poses are joint angles; the rig turns those angles into drawn shapes, so
limbs rotate at the joints instead of the whole sprite sliding around.

Local space: character height = 1.0, feet at y = 0, y grows upward.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from character_engine.expressions import draw_face
from render_engine.draw_utils import (capsule, ellipse, hex_to_rgb, polygon, rounded_rect,
                                      shade, soft_shadow)

Point = Tuple[float, float]

# skeleton proportions (fraction of total height)
HIP_Y = 0.480
CHEST_Y = 0.700
NECK_Y = 0.765
HEAD_CY = 0.852
HEAD_R = 0.105
SHOULDER_Y = 0.715
SHOULDER_DX = 0.088
HIP_DX = 0.052
THIGH = 0.245
SHIN = 0.235
UPPER_ARM = 0.155
FOREARM = 0.145

INK = "#241726"

DEFAULT_SPEC = {
    "id": "character",
    "name": "Character",
    "skin": "#e8b48a",
    "hair": "#2b2430",
    "hairStyle": "short",
    "outfit": "#3f6fa8",
    "outfitStyle": "plain",
    "accent": "#f0e9dc",
    "pants": "#33384a",
    "shoe": "#221c26",
    "headgear": "none",
    "accessory": "none",
    "build": 1.0,
    "height": 1.0,
}


def _down(origin: Point, angle: float, length: float) -> Point:
    """Limbs: point `length` from origin, angle 0 = straight down, + = forward.

    Local space has y growing upward, so "down" subtracts from y.
    """
    return (origin[0] + math.sin(angle) * length, origin[1] - math.cos(angle) * length)


def _up(origin: Point, angle: float, length: float) -> Point:
    """Spine: point `length` from origin, angle 0 = straight up, + = leaning forward."""
    return (origin[0] + math.sin(angle) * length, origin[1] + math.cos(angle) * length)


def _rot(p: Point, origin: Point, angle: float) -> Point:
    dx, dy = p[0] - origin[0], p[1] - origin[1]
    ca, sa = math.cos(angle), math.sin(angle)
    return (origin[0] + dx * ca - dy * sa, origin[1] + dx * sa + dy * ca)


class CharacterRig:
    """Draws one character for one frame. Stateless apart from the spec."""

    def __init__(self, spec: dict):
        self.spec = {**DEFAULT_SPEC, **(spec or {})}
        self.height = float(self.spec.get("height", 1.0) or 1.0)
        self.build = float(self.spec.get("build", 1.0) or 1.0)

    # ------------------------------------------------------------------ #
    # skeleton
    # ------------------------------------------------------------------ #
    def skeleton(self, pose: dict) -> Dict[str, Point]:
        bob = pose.get("bob", 0.0)
        hip_drop = pose.get("hip_drop", 0.0)
        lean = pose.get("lean", 0.0)
        body_rot = pose.get("body_rot", 0.0)
        head_tilt = pose.get("head_tilt", 0.0)
        head_turn = pose.get("head_turn", 0.0)

        hip = (0.0, HIP_Y - hip_drop + bob)
        chest = _up(hip, lean, CHEST_Y - HIP_Y)
        neck = _up(chest, lean, NECK_Y - CHEST_Y)
        head_c = _up(neck, lean + head_tilt, HEAD_CY - NECK_Y)

        perp = (math.cos(lean), -math.sin(lean))
        sh_l = (chest[0] - perp[0] * SHOULDER_DX, chest[1] - perp[1] * SHOULDER_DX)
        sh_r = (chest[0] + perp[0] * SHOULDER_DX, chest[1] + perp[1] * SHOULDER_DX)
        hip_l = (hip[0] - perp[0] * HIP_DX, hip[1] - perp[1] * HIP_DX)
        hip_r = (hip[0] + perp[0] * HIP_DX, hip[1] + perp[1] * HIP_DX)

        joints: Dict[str, Point] = {
            "hip": hip, "chest": chest, "neck": neck, "head": head_c,
            "shoulder_l": sh_l, "shoulder_r": sh_r, "hip_l": hip_l, "hip_r": hip_r,
        }

        # arms
        for side, key in ((-1, "l"), (1, "r")):
            sh = sh_l if side < 0 else sh_r
            a1 = pose.get(f"arm_{key}", 0.0) + lean
            a2 = a1 + pose.get(f"elbow_{key}", 0.0)
            elbow = _down(sh, a1, UPPER_ARM)
            hand = _down(elbow, a2, FOREARM)
            joints[f"elbow_{key}"] = elbow
            joints[f"hand_{key}"] = hand

        # legs
        for side, key in ((-1, "l"), (1, "r")):
            hj = hip_l if side < 0 else hip_r
            t = pose.get(f"thigh_{key}", 0.0)
            k = t - pose.get(f"knee_{key}", 0.0)
            knee = _down(hj, t, THIGH)
            ankle = _down(knee, k, SHIN)
            joints[f"knee_{key}"] = knee
            joints[f"ankle_{key}"] = ankle

        if body_rot:
            base = hip
            joints = {k: _rot(v, base, body_rot) for k, v in joints.items()}

        joints["head_turn"] = (head_turn, 0.0)  # type: ignore[assignment]
        return joints

    # ------------------------------------------------------------------ #
    # drawing
    # ------------------------------------------------------------------ #
    def draw(self, img: Image.Image, ground: Point, height_px: float, state: dict) -> Dict[str, Point]:
        """Draw the character and return the world-space joint map (for objects)."""
        if (self.spec.get("bodyStyle") or "").lower() == "stick":
            return self._draw_stick(img, ground, height_px, state)
        pose = state.get("pose") or {}
        facing = 1 if state.get("facing", 1) >= 0 else -1
        H = height_px * self.height
        ox, oy = ground

        joints = self.skeleton(pose)
        head_turn = float(joints.pop("head_turn")[0])  # type: ignore[index]

        def P(p: Point) -> Point:
            return (ox + p[0] * H * facing, oy - p[1] * H)

        world = {k: P(v) for k, v in joints.items()}  # type: ignore[arg-type]

        draw = ImageDraw.Draw(img, "RGBA")
        ink_w = max(1.0, H * 0.011)
        spec = self.spec

        skin = spec["skin"]
        hair = spec["hair"]
        outfit = spec["outfit"]
        pants = spec["pants"]
        accent = spec.get("accent") or shade(outfit, 0.4)
        back_tint = -0.16

        # contact shadow
        soft_shadow(img, (ox, oy), H * 0.16, H * 0.035, alpha=95, blur=H * 0.02)
        draw = ImageDraw.Draw(img, "RGBA")

        limb_w = H * 0.048 * self.build
        arm_w = H * 0.041 * self.build

        # ---- back arm & leg (drawn first, slightly darkened) ----
        capsule(draw, world["shoulder_l"], world["elbow_l"], arm_w,
                shade(outfit, back_tint), INK, ink_w)
        capsule(draw, world["elbow_l"], world["hand_l"], arm_w * 0.86,
                shade(outfit, back_tint), INK, ink_w)
        ellipse(draw, *world["hand_l"], arm_w * 0.62, arm_w * 0.62,
                shade(skin, back_tint), INK, ink_w * 0.8)
        capsule(draw, world["hip_l"], world["knee_l"], limb_w,
                shade(pants, back_tint), INK, ink_w)
        capsule(draw, world["knee_l"], world["ankle_l"], limb_w * 0.88,
                shade(pants, back_tint), INK, ink_w)
        self._foot(draw, world["ankle_l"], H, shade(spec["shoe"], back_tint), ink_w, facing)

        # ---- torso ----
        hip, chest = world["hip"], world["chest"]
        neck = world["neck"]
        torso_w = H * 0.096 * self.build
        self._torso(draw, hip, chest, torso_w, outfit, accent, spec, ink_w)

        # ---- front leg ----
        capsule(draw, world["hip_r"], world["knee_r"], limb_w, pants, INK, ink_w)
        capsule(draw, world["knee_r"], world["ankle_r"], limb_w * 0.88, pants, INK, ink_w)
        self._foot(draw, world["ankle_r"], H, spec["shoe"], ink_w, facing)

        # ---- front arm ----
        capsule(draw, world["shoulder_r"], world["elbow_r"], arm_w, outfit, INK, ink_w)
        capsule(draw, world["elbow_r"], world["hand_r"], arm_w * 0.86, outfit, INK, ink_w)
        ellipse(draw, *world["hand_r"], arm_w * 0.62, arm_w * 0.62, skin, INK, ink_w * 0.8)

        # ---- neck + head ----
        capsule(draw, chest, neck, H * 0.036, shade(skin, -0.10), INK, ink_w * 0.7)
        head = world["head"]
        r = HEAD_R * H
        self._head(img, draw, head, r, state, head_turn, facing, ink_w)

        # expose hand anchors for object attachment
        world["hand_l_tip"] = world["hand_l"]
        world["hand_r_tip"] = world["hand_r"]
        world["head_top"] = (head[0], head[1] - r)
        return world

    # ------------------------------------------------------------------ #
    def _draw_stick(self, img: Image.Image, ground: Point, height_px: float,
                    state: dict) -> Dict[str, Point]:
        """Stick-figure body: oversized head, thin black limbs, slim torso.

        This is the look of the reference short: a big cream head with heavy
        brows on a nearly weightless body.
        """
        pose = state.get("pose") or {}
        facing = 1 if state.get("facing", 1) >= 0 else -1
        spec = self.spec
        H = height_px * self.height
        ox, oy = ground

        joints = self.skeleton(pose)
        head_turn = float(joints.pop("head_turn")[0])  # type: ignore[index]

        def P(p: Point) -> Point:
            return (ox + p[0] * H * facing, oy - p[1] * H)

        world = {k: P(v) for k, v in joints.items()}  # type: ignore[arg-type]
        draw = ImageDraw.Draw(img, "RGBA")

        ink = spec.get("ink", "#1b1b22")
        limb = H * float(spec.get("limb", 0.019))
        ink_w = max(1.0, limb * 0.55)
        skin = spec["skin"]
        outfit = spec["outfit"]
        accent = spec.get("accent") or "#8a5a2b"
        torso_w = H * float(spec.get("torso", 0.062))

        soft_shadow(img, (ox, oy), H * 0.14, H * 0.028, alpha=90, blur=H * 0.018)
        draw = ImageDraw.Draw(img, "RGBA")

        # back limbs
        capsule(draw, world["shoulder_l"], world["elbow_l"], limb, ink, None, 0)
        capsule(draw, world["elbow_l"], world["hand_l"], limb * 0.9, ink, None, 0)
        capsule(draw, world["hip_l"], world["knee_l"], limb, ink, None, 0)
        capsule(draw, world["knee_l"], world["ankle_l"], limb * 0.92, ink, None, 0)
        self._foot(draw, world["ankle_l"], H, ink, 0, facing)

        # torso
        hip, chest = world["hip"], world["chest"]
        capsule(draw, hip, chest, torso_w, outfit, ink, ink_w)
        # tie / badge
        if spec.get("tie", True):
            neck = world["neck"]
            tie_top = (neck[0], neck[1] + (chest[1] - neck[1]) * 0.05)
            tie_mid = (chest[0], chest[1] + (hip[1] - chest[1]) * 0.55)
            polygon(draw, [(tie_top[0] - torso_w * 0.30, tie_top[1]),
                           (tie_top[0] + torso_w * 0.30, tie_top[1]),
                           (tie_mid[0] + torso_w * 0.22, tie_mid[1]),
                           (tie_mid[0], tie_mid[1] + torso_w * 0.55),
                           (tie_mid[0] - torso_w * 0.22, tie_mid[1])],
                    accent, ink, ink_w * 0.6)
        if spec.get("badge"):
            bx = chest[0] + torso_w * 0.55 * facing
            by = chest[1] + (hip[1] - chest[1]) * 0.30
            polygon(draw, [(bx, by - H * 0.016), (bx + H * 0.011, by - H * 0.005),
                           (bx + H * 0.007, by + H * 0.011), (bx - H * 0.007, by + H * 0.011),
                           (bx - H * 0.011, by - H * 0.005)], "#f2c14b", ink, ink_w * 0.5)

        # front limbs
        capsule(draw, world["hip_r"], world["knee_r"], limb, ink, None, 0)
        capsule(draw, world["knee_r"], world["ankle_r"], limb * 0.92, ink, None, 0)
        self._foot(draw, world["ankle_r"], H, ink, 0, facing)
        capsule(draw, world["shoulder_r"], world["elbow_r"], limb, ink, None, 0)
        capsule(draw, world["elbow_r"], world["hand_r"], limb * 0.9, ink, None, 0)
        ellipse(draw, *world["hand_r"], limb * 1.25, limb * 1.25, ink, None, 0)

        # head (much bigger than the default rig)
        head = world["head"]
        scale = float(spec.get("head_scale", 1.55))
        r = HEAD_R * H * scale
        rx, ry = r * 0.92, r * 1.02
        hx, hy = head
        neck = world["neck"]
        capsule(draw, chest, neck, torso_w * 0.45, skin, ink, ink_w * 0.8)
        ellipse(draw, hx, hy, rx, ry, skin, ink, max(1.5, ink_w * 1.5))

        # hair / cap
        hair = spec["hair"]
        style = spec.get("hairStyle", "short")
        if style == "cap":
            draw.chord([hx - rx * 1.04, hy - ry * 1.12, hx + rx * 1.04, hy + ry * 0.18],
                       180, 360, fill=spec.get("accent", "#22304f"), outline=ink, width=int(max(2, ink_w * 1.6)))
            polygon(draw, [(hx + facing * rx * 0.1, hy - ry * 0.55),
                           (hx + facing * rx * 1.35, hy - ry * 0.42),
                           (hx + facing * rx * 1.30, hy - ry * 0.28),
                           (hx + facing * rx * 0.1, hy - ry * 0.34)],
                    spec.get("accent", "#22304f"), ink, max(1.5, ink_w))
        elif style != "bald":
            draw.chord([hx - rx * 1.02, hy - ry * 1.10, hx + rx * 1.02, hy + ry * 0.05],
                       180, 360, fill=hair, outline=ink, width=int(max(2, ink_w * 1.4)))
            # a small tuft, like the reference scientist
            polygon(draw, [(hx - rx * 0.1, hy - ry * 1.02), (hx + rx * 0.35, hy - ry * 1.22),
                           (hx + rx * 0.30, hy - ry * 0.92)], hair, ink, max(1.2, ink_w * 0.8))

        draw_face(draw, hx + facing * rx * 0.05, hy + ry * 0.06, r * 0.86, skin,
                  expression=state.get("expression", "neutral"),
                  mouth_open=float(state.get("mouth", 0.0)) * 0.55,
                  blink=float(state.get("blink", 0.0)),
                  facing=facing, head_turn=head_turn, ink=ink, line=max(1.6, ink_w * 1.5))

        world["hand_l_tip"] = world["hand_l"]
        world["hand_r_tip"] = world["hand_r"]
        world["head_top"] = (head[0], head[1] - r)
        return world

    # ------------------------------------------------------------------ #
    def _foot(self, draw: ImageDraw.ImageDraw, ankle: Point, H: float, colour: str,
              ink_w: float, facing: int) -> None:
        toe = (ankle[0] + H * 0.045 * facing, ankle[1] + H * 0.004)
        capsule(draw, ankle, toe, H * 0.038, colour, INK, ink_w)

    def _torso(self, draw: ImageDraw.ImageDraw, hip: Point, chest: Point, width: float,
               outfit: str, accent: str, spec: dict, ink_w: float) -> None:
        dx = chest[0] - hip[0]
        dy = chest[1] - hip[1]
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        px, py = -uy, ux  # perpendicular

        w_top = width * 1.05
        w_bot = width * 0.86
        top = (chest[0] + ux * width * 0.30, chest[1] + uy * width * 0.30)
        pts = [
            (top[0] + px * w_top, top[1] + py * w_top),
            (top[0] - px * w_top, top[1] - py * w_top),
            (hip[0] - px * w_bot, hip[1] - py * w_bot),
            (hip[0] + px * w_bot, hip[1] + py * w_bot),
        ]
        # rounded shoulders + hips
        polygon(draw, pts, outfit, INK, ink_w)
        ellipse(draw, top[0], top[1], w_top * 0.55, w_top * 0.55, outfit, INK, ink_w)
        ellipse(draw, hip[0], hip[1], w_bot * 0.42, w_bot * 0.42, outfit, INK, ink_w)

        style = spec.get("outfitStyle", "plain")
        if style == "coat":
            # lab coat: centre seam + lapels + pockets
            draw.line([(chest[0], chest[1] + width * 0.1), (hip[0], hip[1])],
                      fill=shade(outfit, -0.35), width=max(2, int(ink_w * 0.7)))
            polygon(draw, [(top[0] - px * w_top * 0.1, top[1] - py * w_top * 0.1),
                           (top[0] - px * w_top * 0.85, top[1] - py * w_top * 0.85),
                           (chest[0] - px * w_top * 0.2, chest[1] + width * 0.55)],
                    accent, INK, ink_w * 0.6)
            polygon(draw, [(top[0] + px * w_top * 0.1, top[1] + py * w_top * 0.1),
                           (top[0] + px * w_top * 0.85, top[1] + py * w_top * 0.85),
                           (chest[0] + px * w_top * 0.2, chest[1] + width * 0.55)],
                    accent, INK, ink_w * 0.6)
        elif style == "uniform":
            rounded_rect(draw, [hip[0] - width * 0.55, chest[1] + width * 0.15,
                                hip[0] + width * 0.55, hip[1] + width * 0.25], width * 0.2,
                         shade(outfit, -0.25), INK, ink_w * 0.5)
            draw.line([(chest[0], chest[1] + width * 0.2), (hip[0], hip[1])],
                      fill=shade(outfit, -0.4), width=max(2, int(ink_w * 0.7)))
        elif style == "striped":
            for i in range(4):
                y = chest[1] + width * (0.15 + i * 0.55)
                draw.line([(hip[0] - width * 0.9, y), (hip[0] + width * 0.9, y)],
                          fill=accent, width=max(2, int(width * 0.22)))
        elif style == "armor":
            ellipse(draw, chest[0], chest[1] + width * 0.35, w_top * 0.85, w_top * 0.6,
                    shade(outfit, 0.15), INK, ink_w * 0.7)
        elif style == "dress":
            polygon(draw, [(hip[0] - width * 0.95, hip[1] + width * 0.5),
                           (hip[0] + width * 0.95, hip[1] + width * 0.5),
                           (hip[0] + width * 0.6, hip[1] - width * 0.2),
                           (hip[0] - width * 0.6, hip[1] - width * 0.2)],
                    outfit, INK, ink_w)
        elif style == "suit":
            polygon(draw, [(top[0] - px * w_top * 0.35, top[1] - py * w_top * 0.35),
                           (top[0] - px * w_top * 0.05, top[1] - py * w_top * 0.05),
                           (chest[0] - px * w_top * 0.05, chest[1] + width * 0.8)],
                    accent, INK, ink_w * 0.5)
        else:
            # plain shirt: collar + hem
            polygon(draw, [(top[0] - px * w_top * 0.35, top[1] - py * w_top * 0.35),
                           (top[0], top[1] + width * 0.35),
                           (top[0] + px * w_top * 0.35, top[1] + py * w_top * 0.35)],
                    accent, INK, ink_w * 0.5)

    def _head(self, img: Image.Image, draw: ImageDraw.ImageDraw, head: Point, r: float,
              state: dict, head_turn: float, facing: int, ink_w: float) -> None:
        spec = self.spec
        skin = spec["skin"]
        hair = spec["hair"]
        hx, hy = head
        rx, ry = r * 1.02, r * 1.08

        # ears
        for side in (-1, 1):
            ellipse(draw, hx + side * rx * 0.95, hy + ry * 0.05, r * 0.16, r * 0.20,
                    shade(skin, -0.08), INK, ink_w * 0.7)

        ellipse(draw, hx, hy, rx, ry, skin, INK, ink_w)

        # hair
        style = spec.get("hairStyle", "short")
        if style != "bald":
            if style == "long":
                polygon(draw, [(hx - rx * 1.02, hy - ry * 0.1), (hx + rx * 1.02, hy - ry * 0.1),
                               (hx + rx * 1.05, hy + ry * 1.15), (hx - rx * 1.05, hy + ry * 1.15)],
                        hair, INK, ink_w)
                draw.ellipse([hx - rx, hy - ry, hx + rx, hy + ry * 0.2], fill=skin)
                draw.arc([hx - rx, hy - ry, hx + rx, hy + ry * 0.2], 180, 360, fill=INK,
                         width=int(ink_w))
                draw.chord([hx - rx * 1.02, hy - ry * 1.05, hx + rx * 1.02, hy + ry * 0.35],
                           180, 360, fill=hair, outline=INK, width=int(ink_w))
            elif style == "bun":
                ellipse(draw, hx + facing * rx * 0.55, hy - ry * 1.02, r * 0.30, r * 0.30,
                        hair, INK, ink_w * 0.8)
                draw.chord([hx - rx * 1.02, hy - ry * 1.08, hx + rx * 1.02, hy + ry * 0.45],
                           180, 360, fill=hair, outline=INK, width=int(ink_w))
            elif style == "spiky":
                pts = [(hx - rx, hy - ry * 0.25)]
                n = 6
                for i in range(n + 1):
                    t = i / n
                    x = hx - rx + 2 * rx * t
                    y = hy - ry * (1.05 if i % 2 else 0.62)
                    pts.append((x, y))
                pts.append((hx + rx, hy - ry * 0.25))
                polygon(draw, pts, hair, INK, ink_w)
            elif style == "curly":
                for i in range(7):
                    a = math.pi + i / 6 * math.pi
                    ellipse(draw, hx + math.cos(a) * rx * 0.92, hy + math.sin(a) * ry * 0.92,
                            r * 0.26, r * 0.26, hair, INK, ink_w * 0.7)
            elif style == "balding":
                draw.chord([hx - rx * 1.01, hy - ry * 1.02, hx + rx * 1.01, hy + ry * 0.3],
                           190, 350, fill=hair, outline=INK, width=int(ink_w))
            else:  # short
                draw.chord([hx - rx * 1.02, hy - ry * 1.06, hx + rx * 1.02, hy + ry * 0.30],
                           180, 360, fill=hair, outline=INK, width=int(ink_w))

        # headgear
        gear = spec.get("headgear", "none")
        if gear == "cap":
            draw.chord([hx - rx * 1.05, hy - ry * 1.12, hx + rx * 1.05, hy + ry * 0.25],
                       180, 360, fill=spec.get("accent", "#2f4a7a"), outline=INK, width=int(ink_w))
            polygon(draw, [(hx + facing * rx * 0.2, hy - ry * 0.55),
                           (hx + facing * rx * 1.35, hy - ry * 0.42),
                           (hx + facing * rx * 1.30, hy - ry * 0.30),
                           (hx + facing * rx * 0.2, hy - ry * 0.36)],
                    spec.get("accent", "#2f4a7a"), INK, ink_w * 0.7)
        elif gear == "hardhat":
            draw.chord([hx - rx * 1.15, hy - ry * 1.25, hx + rx * 1.15, hy + ry * 0.2],
                       180, 360, fill="#f2c14b", outline=INK, width=int(ink_w))
            draw.line([(hx - rx * 1.25, hy - ry * 0.55), (hx + rx * 1.25, hy - ry * 0.55)],
                      fill="#f2c14b", width=int(r * 0.16))
        elif gear == "hood":
            draw.chord([hx - rx * 1.25, hy - ry * 1.2, hx + rx * 1.25, hy + ry * 1.1],
                       185, 355, fill=shade(spec["outfit"], -0.2), outline=INK, width=int(ink_w))

        # face
        draw_face(draw, hx + facing * r * 0.06, hy, r, skin,
                  expression=state.get("expression", "neutral"),
                  mouth_open=float(state.get("mouth", 0.0)),
                  blink=float(state.get("blink", 0.0)),
                  facing=facing, head_turn=head_turn, ink=INK, line=ink_w)

        # glasses
        if spec.get("accessory") == "glasses":
            for side in (-1, 1):
                ellipse(draw, hx + side * r * 0.38 + facing * r * 0.06, hy - r * 0.02,
                        r * 0.26, r * 0.22, (255, 255, 255, 30), INK, ink_w * 0.8)
            draw.line([(hx - r * 0.12, hy - r * 0.02), (hx + r * 0.12, hy - r * 0.02)],
                      fill=INK, width=int(ink_w * 0.8))
