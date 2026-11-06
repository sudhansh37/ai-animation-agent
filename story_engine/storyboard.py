"""Storyboard generation: turns analysis + narration into the scene list the
renderer consumes.

Each scene carries everything the spec asks for: narration, background, camera
angle and move, character poses and expressions, props, sound effects, caption
text and a transition. Timing comes from the narration length, so the visuals
follow the voice.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from story_engine.script_writer import estimate_duration

# --------------------------------------------------------------------------- #
# character templates by role
# --------------------------------------------------------------------------- #

ROLE_TEMPLATES: Dict[str, dict] = {
    "scientist": {"skin": "#e8b48a", "hair": "#3a3038", "hairStyle": "short",
                  "outfit": "#f4f1ea", "outfitStyle": "coat", "accent": "#3f6fa8",
                  "pants": "#3a4152", "headgear": "none", "accessory": "glasses", "build": 0.95},
    "guard": {"skin": "#d8a078", "hair": "#221c22", "hairStyle": "short",
              "outfit": "#2f3f5f", "outfitStyle": "uniform", "accent": "#1c2740",
              "pants": "#232a3a", "headgear": "cap", "accessory": "none", "build": 1.12},
    "prisoner": {"skin": "#c98a5e", "hair": "#2a1c16", "hairStyle": "balding",
                 "outfit": "#e0dcc8", "outfitStyle": "striped", "accent": "#d9822b",
                 "pants": "#6a6252", "headgear": "none", "accessory": "none", "build": 1.0},
    "inmate": {"skin": "#b87a4e", "hair": "#1a1414", "hairStyle": "bald",
               "outfit": "#c8c4b0", "outfitStyle": "striped", "accent": "#b5476f",
               "pants": "#5a5448", "headgear": "none", "accessory": "none", "build": 1.28},
    "teacher": {"skin": "#e8c09a", "hair": "#4a3428", "hairStyle": "bun",
                "outfit": "#7a5aa8", "outfitStyle": "dress", "accent": "#f0e9dc",
                "pants": "#4a3a5a", "headgear": "none", "accessory": "glasses", "build": 0.95},
    "student": {"skin": "#f0c9a8", "hair": "#2a1e18", "hairStyle": "curly",
                "outfit": "#4a8c5a", "outfitStyle": "plain", "accent": "#f2c14b",
                "pants": "#33415c", "headgear": "none", "accessory": "none", "build": 0.88},
    "king": {"skin": "#e8b48a", "hair": "#5a4028", "hairStyle": "long",
             "outfit": "#8e3b5f", "outfitStyle": "dress", "accent": "#f2c14b",
             "pants": "#4a2030", "headgear": "none", "accessory": "none", "build": 1.05},
    "robot": {"skin": "#9aa4b2", "hair": "#6f7b8a", "hairStyle": "bald",
              "outfit": "#8b98a8", "outfitStyle": "armor", "accent": "#3fe0c8",
              "pants": "#5a6270", "headgear": "none", "accessory": "none", "build": 1.15},
    "default": {"skin": "#e8b48a", "hair": "#2b2430", "hairStyle": "short",
                "outfit": "#3f6fa8", "outfitStyle": "plain", "accent": "#f0e9dc",
                "pants": "#33384a", "headgear": "none", "accessory": "none", "build": 1.0},
}

PALETTES = {
    "lab": ["#18202f", "#33415c", "#8fa6c8"],
    "prison_cell": ["#2a2f38", "#4a525e", "#8d99a8"],
    "prison_corridor": ["#242a33", "#3e4753", "#7c8794"],
    "prison_yard": ["#4a6a8a", "#8fa6b8", "#c8c0a8"],
    "office": ["#2a3242", "#4a5468", "#b8c2d4"],
    "city": ["#1a2438", "#3a4a68", "#8aa0c0"],
    "forest": ["#243a2a", "#3f6a44", "#9ac08a"],
    "room": ["#3a3242", "#5a5060", "#b8a8b0"],
    "night_sky": ["#0e1630", "#243a68", "#5a7ab0"],
    "plain": ["#5a7ab0", "#a8c0d8", "#d8e0c0"],
    "wall": ["#333a44", "#4a525e", "#9aa4b0"],
}

CAMERA_CYCLE = ["push_in", "medium", "pan_right", "low_angle", "over_shoulder",
                "track", "high_angle", "pull_out", "side_angle", "dolly"]

CAMERA_ZOOM = {
    "push_in": 1.06, "medium": 1.14, "pan_right": 1.10, "low_angle": 1.08,
    "over_shoulder": 1.20, "track": 1.12, "high_angle": 1.04, "pull_out": 1.10,
    "side_angle": 1.12, "dolly": 1.08, "close_up": 1.30,
}

SFX_BY_ACTION = {
    "walk": ["footsteps"], "run": ["footsteps"], "jump": ["whoosh", "thud"],
    "fight": ["punch", "impact"], "fall": ["impact"], "open_door": ["door_open"],
    "close_door": ["door_close"], "pick_up": ["object_move"], "drop": ["object_move"],
    "push": ["object_move"], "pull": ["object_move"], "cry": ["ambience"],
    "laugh": ["ting"], "sit": ["object_move"], "point": ["whoosh"],
    "talk": [], "stand": [], "look_around": [], "wave": ["ting"],
}

EMOTION_TO_EXPRESSION = {
    "neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry",
    "surprised": "surprised", "scared": "scared", "confused": "confused",
    "laughing": "laughing", "crying": "crying",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s or fallback


def build_characters(analysis: Dict, style: Dict) -> List[dict]:
    chars: List[dict] = []
    for i, c in enumerate(analysis.get("characters") or []):
        key = _slug(c.get("id") or c.get("name") or f"char{i}", f"char{i}")
        template_key = key if key in ROLE_TEMPLATES else "default"
        for role in ROLE_TEMPLATES:
            if role != "default" and role in key:
                template_key = role
                break
        spec = dict(ROLE_TEMPLATES[template_key])
        spec.update({
            "id": key,
            "name": c.get("name") or key.replace("_", " ").title(),
            "role": c.get("role", "support"),
        })
        # honour an explicit style override from the project file
        override = (style.get("characters") or {}).get(key)
        if isinstance(override, dict):
            spec.update({k: v for k, v in override.items() if v is not None})
        chars.append(spec)
    if not chars:
        chars = [{**ROLE_TEMPLATES["default"], "id": "hero", "name": "Hero", "role": "protagonist"}]
    return chars


PLURAL_WORDS = ("prisoners", "people", "crowd", "everyone", "they", "them", "inmates", "students")


def _pick_characters(beat: Dict, characters: List[dict], scene_index: int = 0) -> List[dict]:
    """Choose who is on screen: the named cast, plus a rotating crowd when the
    sentence talks about a group — so every character gets screen time."""
    hero = next((c for c in characters if c.get("role") == "protagonist"), characters[0])
    text = (beat.get("text") or "").lower()
    named = [c for c in characters if c["id"] in (beat.get("characters") or [])]

    wanted: List[dict] = []
    for c in named:
        if c not in wanted:
            wanted.append(c)

    others = [c for c in characters if c["id"] != hero["id"]]
    if any(word in text for word in PLURAL_WORDS) and others:
        for offset in range(2):
            candidate = others[(scene_index + offset) % len(others)]
            if candidate not in wanted:
                wanted.append(candidate)

    # whoever the sentence names first is the actor of that shot
    positions = {c["id"]: text.find(c["id"].lower()) for c in named}
    positions = {k: v for k, v in positions.items() if v >= 0}
    actor_id = min(positions, key=positions.get) if positions else hero["id"]
    actor = next((c for c in characters if c["id"] == actor_id), hero)
    ordered = [actor]
    if hero["id"] != actor_id:
        ordered.append(hero)
    for c in wanted:
        if all(c["id"] != o["id"] for o in ordered):
            ordered.append(c)
    wanted = ordered

    if len(wanted) < 2 and others:
        candidate = others[scene_index % len(others)]
        if candidate not in wanted:
            wanted.append(candidate)
    return wanted[:3]


def _placement(count: int) -> List[tuple]:
    """Stage positions, spaced so characters do not overlap in a 9:16 frame."""
    if count <= 1:
        return [(0.0, 1.0, 1)]
    if count == 2:
        return [(-0.62, 1.0, 1), (0.62, 1.0, -1)]
    return [(-1.0, 0.9, 1), (0.0, 1.0, 1), (1.0, 0.9, -1)]


def _props_for_beat(beat: Dict, characters: List[dict]) -> List[dict]:
    props: List[dict] = []
    objects = beat.get("objects") or []
    holder = characters[0]["id"] if characters else None
    for obj in objects[:2]:
        kind = _slug(obj, "box")
        if kind in ("vial", "clipboard", "baton", "phone", "cup", "key", "brush"):
            props.append({"kind": kind, "attached_to": f"{holder}.hand_r", "scale": 1.0})
        else:
            props.append({"kind": kind, "x": 0.42, "y": 0.0, "scale": 1.0})
    return props


def _background_for(beat: Dict, previous_kind: str, paint_level: float) -> dict:
    kind = beat.get("location") or previous_kind or "plain"
    if kind not in PALETTES:
        kind = "plain"
    text = (beat.get("text") or "").lower()
    pink = paint_level
    if "pink" in text:
        pink = min(1.0, pink + 0.55)
    if "paint" in text and ("wall" in text or "cell" in text or "corridor" in text):
        pink = min(1.0, pink + 0.35)
    return {"kind": kind, "palette": PALETTES[kind], "pink": round(pink, 2)}


# --------------------------------------------------------------------------- #
# main builder
# --------------------------------------------------------------------------- #

def build_storyboard(analysis: Dict, narration: Dict, options: Optional[dict] = None) -> Dict:
    options = options or {}
    fps = int(options.get("fps", 30))
    width = int(options.get("width", 1080))
    height = int(options.get("height", 1920))
    scene_pad = float(options.get("scene_pad", 0.45))
    style = options.get("style") or {}

    cast_override = options.get("cast") or []
    if cast_override:
        characters = []
        for i, spec in enumerate(cast_override):
            merged = dict(ROLE_TEMPLATES["default"])
            role_key = next((r for r in ROLE_TEMPLATES if r != "default" and r in str(spec.get("id", ""))), None)
            if role_key:
                merged.update(ROLE_TEMPLATES[role_key])
            merged.update({k: v for k, v in spec.items() if v is not None})
            merged.setdefault("id", f"char{i}")
            merged.setdefault("name", merged["id"].title())
            merged.setdefault("role", "protagonist" if i == 0 else "support")
            characters.append(merged)
    else:
        characters = build_characters(analysis, style)
    beats = analysis.get("beats") or []
    lines = narration.get("lines") or []
    if len(lines) < len(beats):
        lines = list(lines) + [b.get("text", "") for b in beats[len(lines):]]

    scenes: List[dict] = []
    cursor = 0.0
    previous_kind = (analysis.get("locations") or [{}])[0].get("kind", "plain")
    paint_level = 0.0
    previous_action = "stand"

    for i, beat in enumerate(beats):
        line = (lines[i] if i < len(lines) else beat.get("text", "")).strip()
        duration = estimate_duration(line) + scene_pad
        cast = _pick_characters(beat, characters, i)
        placements = _placement(len(cast))
        action = beat.get("action") or "stand"
        emotion = EMOTION_TO_EXPRESSION.get(beat.get("emotion") or "neutral", "neutral")
        text_low = (beat.get("text") or "").lower()
        if "aggress" in text_low and emotion in ("neutral", "angry"):
            action, emotion = "fight", "angry"
        if "calm" in text_low and action == "fight":
            action, emotion = "stand", "happy"
        if action in ("cry", "laugh") and emotion == "neutral":
            emotion = "crying" if action == "cry" else "laughing"

        bg = _background_for(beat, previous_kind, paint_level)
        paint_level = bg["pink"]
        previous_kind = bg["kind"]

        camera_move = CAMERA_CYCLE[i % len(CAMERA_CYCLE)]
        if i == 0:
            camera_move = "pull_out"
        if i == len(beats) - 1:
            camera_move = "push_in"

        scene_chars = []
        # with three actors in a 9:16 frame the outer two are drawn smaller so
        # shoulders never overlap
        crowd_scale = 0.80 if len(cast) >= 3 else 1.0
        for pos_index, ((x, depth, facing), char) in enumerate(zip(placements, cast)):
            scene_chars.append({
                "id": char["id"],
                "x": round(x, 3),
                "depth": depth,
                "facing": facing,
                "scale": crowd_scale if pos_index != 1 or len(cast) < 3 else 1.0,
                "action": action if char is cast[0] else ("stand" if action in ("talk",) else action),
                "expression": emotion if char is cast[0] else "neutral",
                "previous_action": previous_action,
                "talking": char is cast[0],
            })

        sfx = [{"at": 0.25, "name": name, "gain": 0.7}
               for name in SFX_BY_ACTION.get(action, [])][:2]
        if bg["kind"] in ("prison_yard", "prison_corridor", "city"):
            sfx.append({"at": 0.0, "name": "ambience", "gain": 0.35})
        if bg["pink"] > 0.3:
            sfx.append({"at": 0.15, "name": "machine", "gain": 0.25})

        scenes.append({
            "id": f"s{i + 1:02d}",
            "index": i,
            "start": round(cursor, 3),
            "duration": round(duration, 3),
            "narration": line,
            "background": bg,
            "camera": {
                "move": camera_move,
                "zoom": CAMERA_ZOOM.get(camera_move, 1.1),
                "easing": "ease_in_out",
            },
            "characters": scene_chars,
            "objects": _props_for_beat(beat, cast),
            "sfx": sfx,
            "transition": "fade" if i == 0 else ("iris" if i == len(beats) - 1 else "crossfade"),
            "vignette": True,
        })
        cursor += duration
        previous_action = action

    total = round(cursor, 3)
    return {
        "title": analysis.get("title") or "Untitled",
        "fps": fps,
        "width": width,
        "height": height,
        "duration": total,
        "mood": options.get("mood") or ("dark" if paint_level > 0.4 else "tension"),
        "style": {"look": "stylized 2D vector", **style},
        "characters": characters,
        "scenes": scenes,
    }


def save(storyboard: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(storyboard, f, indent=2, ensure_ascii=False)
    return path


def summarise(storyboard: Dict) -> str:
    lines = [f"Storyboard: {storyboard['title']}  ({storyboard['duration']:.1f}s, "
             f"{len(storyboard['scenes'])} scenes, {len(storyboard['characters'])} characters)"]
    for scene in storyboard["scenes"]:
        cast = ", ".join(f"{c['id']}:{c['action']}/{c['expression']}" for c in scene["characters"])
        lines.append(f"  {scene['id']}  {scene['start']:6.2f}s +{scene['duration']:4.2f}s  "
                     f"{scene['background']['kind']:<16} {scene['camera']['move']:<13} {cast}")
        lines.append(f"        \"{scene['narration'][:88]}\"")
    return "\n".join(lines)
