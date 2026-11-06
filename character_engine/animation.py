"""Keyframe curves, easing and the action library.

An *action* is a pure function of (t, duration) that returns a pose dict for the
rig: joint angles in radians, plus a few body-level offsets. Because it is pure,
any frame can be re-rendered identically after an interruption.

Actions implemented: stand, idle, walk, run, jump, sit, talk, point, wave,
fight, fall, cry, laugh, look_around, pick_up, drop, open_door, close_door,
push, pull.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

TAU = math.pi * 2

# --------------------------------------------------------------------------- #
# easing
# --------------------------------------------------------------------------- #

EASINGS: Dict[str, Callable[[float], float]] = {
    "linear": lambda t: t,
    "ease_in": lambda t: t * t * t,
    "ease_out": lambda t: 1 - (1 - t) ** 3,
    "ease_in_out": lambda t: 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2,
    "smooth": lambda t: t * t * (3 - 2 * t),
    "back_out": lambda t: 1 + 2.2 * (t - 1) ** 3 + 1.2 * (t - 1) ** 2,
    "elastic_out": lambda t: t if t in (0.0, 1.0) else 2 ** (-9 * t) * math.sin((t * 10 - 0.75) * (TAU / 3)) + 1,
}


def ease(name: str, t: float) -> float:
    t = max(0.0, min(1.0, t))
    return EASINGS.get(name, EASINGS["ease_in_out"])(t)


def keyframe(t: float, keys: List[tuple], easing: str = "ease_in_out") -> float:
    """keys = [(time, value), ...] sorted by time."""
    if not keys:
        return 0.0
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    for i in range(len(keys) - 1):
        t0, v0 = keys[i]
        t1, v1 = keys[i + 1]
        if t0 <= t <= t1:
            span = (t1 - t0) or 1e-6
            return v0 + (v1 - v0) * ease(easing, (t - t0) / span)
    return keys[-1][1]


# --------------------------------------------------------------------------- #
# secondary motion
# --------------------------------------------------------------------------- #

def blink_amount(t: float, offset: float = 0.0, period: float = 3.6) -> float:
    phase = ((t + offset) % period) / period
    start, dur = 0.60, 0.045
    if phase < start or phase > start + dur:
        return 0.0
    return math.sin((phase - start) / dur * math.pi)


def mouth_amount(t: float, talking: bool = True, rate: float = 7.0) -> float:
    if not talking:
        return 0.0
    a = 0.5 + 0.5 * math.sin(t * rate)
    b = 0.5 + 0.5 * math.sin(t * rate * 1.7 + 1.3)
    return max(0.0, min(1.0, (a * 0.6 + b * 0.4) * 0.95))


# --------------------------------------------------------------------------- #
# the action library
# --------------------------------------------------------------------------- #

BASE = dict(thigh_l=0.0, thigh_r=0.0, knee_l=0.05, knee_r=0.05,
            arm_l=-0.10, arm_r=0.10, elbow_l=0.15, elbow_r=0.15,
            lean=0.0, bob=0.0, hip_drop=0.0, body_rot=0.0,
            head_tilt=0.0, head_turn=0.0)


def _p(**kw) -> dict:
    out = dict(BASE)
    out.update(kw)
    return out


def a_stand(t: float, d: float) -> dict:
    breath = math.sin(t * 1.9) * 0.006
    return _p(bob=breath, arm_l=-0.12 + breath * 0.4, arm_r=0.12 - breath * 0.4,
              head_turn=math.sin(t * 0.6) * 0.12)


def a_idle(t: float, d: float) -> dict:
    sway = math.sin(t * 1.2)
    return _p(bob=sway * 0.008, arm_l=-0.14 + sway * 0.03, arm_r=0.14 - sway * 0.03,
              elbow_l=0.22, elbow_r=0.22, head_turn=math.sin(t * 0.5) * 0.2,
              head_tilt=math.sin(t * 0.8) * 0.03)


def a_walk(t: float, d: float) -> dict:
    ph = t * TAU * 1.55
    s = math.sin(ph)
    return _p(thigh_l=0.42 * s, thigh_r=-0.42 * s,
              knee_l=max(0.0, -s) * 0.75 + 0.08, knee_r=max(0.0, s) * 0.75 + 0.08,
              arm_l=-0.42 * s, arm_r=0.42 * s, elbow_l=0.25, elbow_r=0.25,
              bob=abs(math.cos(ph)) * 0.014 - 0.006, lean=0.05,
              head_turn=math.sin(t * 0.9) * 0.1)


def a_run(t: float, d: float) -> dict:
    ph = t * TAU * 2.4
    s = math.sin(ph)
    return _p(thigh_l=0.78 * s, thigh_r=-0.78 * s,
              knee_l=max(0.0, -s) * 1.35 + 0.25, knee_r=max(0.0, s) * 1.35 + 0.25,
              arm_l=-0.85 * s, arm_r=0.85 * s, elbow_l=1.15, elbow_r=1.15,
              bob=abs(math.cos(ph)) * 0.03, lean=0.18)


def a_jump(t: float, d: float) -> dict:
    p = t / max(d, 1e-6)
    lift = math.sin(min(1.0, p) * math.pi)
    tuck = 0.5 + 0.5 * math.cos(min(1.0, p) * math.pi)
    return _p(thigh_l=0.55 * tuck, thigh_r=0.55 * tuck,
              knee_l=1.1 * tuck, knee_r=1.1 * tuck,
              arm_l=-1.9 * lift, arm_r=1.9 * lift,
              elbow_l=0.4, elbow_r=0.4, bob=0.16 * lift, lean=0.05)


def a_sit(t: float, d: float) -> dict:
    return _p(hip_drop=0.20, thigh_l=1.35, thigh_r=1.30, knee_l=1.25, knee_r=1.28,
              arm_l=-0.55, arm_r=0.55, elbow_l=1.1, elbow_r=1.1,
              lean=-0.05, head_turn=math.sin(t * 0.7) * 0.15)


def a_talk(t: float, d: float) -> dict:
    g = math.sin(t * 2.1)
    return _p(arm_r=0.55 + g * 0.35, elbow_r=0.9 + g * 0.25,
              arm_l=-0.25, elbow_l=0.5,
              head_turn=math.sin(t * 1.3) * 0.18, head_tilt=math.sin(t * 1.7) * 0.04,
              bob=math.sin(t * 2.1) * 0.006)


def a_point(t: float, d: float) -> dict:
    return _p(arm_r=1.55 + math.sin(t * 2.4) * 0.06, elbow_r=0.05,
              arm_l=-0.2, elbow_l=0.3, lean=0.04, head_turn=0.25)


def a_wave(t: float, d: float) -> dict:
    return _p(arm_r=2.35 + math.sin(t * 7.5) * 0.30, elbow_r=0.25,
              arm_l=-0.15, elbow_l=0.2, head_tilt=-0.04)


def a_fight(t: float, d: float) -> dict:
    p = (t * 2.2) % 1.0
    thrust = max(0.0, math.sin(p * math.pi))
    return _p(arm_r=0.75 + thrust * 0.30, elbow_r=1.95 - thrust * 1.15,
              arm_l=-0.50, elbow_l=1.90, lean=0.10 + thrust * 0.12,
              thigh_l=0.25, thigh_r=-0.25, knee_l=0.35, knee_r=0.35,
              head_tilt=0.06)


def a_fall(t: float, d: float) -> dict:
    p = min(1.0, t / max(d, 1e-6))
    return _p(body_rot=-1.35 * ease("ease_in", p), arm_l=-2.0, arm_r=2.0,
              elbow_l=0.3, elbow_r=0.3, thigh_l=0.5, thigh_r=0.2,
              knee_l=0.6, knee_r=0.4, bob=-0.10 * p)


def a_cry(t: float, d: float) -> dict:
    shake = math.sin(t * 9.0) * 0.03
    return _p(arm_l=-1.15, arm_r=1.15, elbow_l=1.9, elbow_r=1.9,
              head_tilt=0.22, lean=0.12, bob=shake * 0.02,
              thigh_l=0.05, thigh_r=-0.05, knee_l=0.2, knee_r=0.2)


def a_laugh(t: float, d: float) -> dict:
    bounce = abs(math.sin(t * 4.2))
    return _p(head_tilt=-0.22, bob=bounce * 0.02, lean=-0.06,
              arm_l=-0.9, arm_r=0.9, elbow_l=1.3, elbow_r=1.3,
              thigh_l=0.08, thigh_r=-0.08, knee_l=0.15, knee_r=0.15)


def a_look_around(t: float, d: float) -> dict:
    return _p(head_turn=math.sin(t * 1.5) * 0.9, head_tilt=math.sin(t * 1.1) * 0.05,
              arm_l=-0.2, arm_r=0.2, elbow_l=0.35, elbow_r=0.35)


def a_pick_up(t: float, d: float) -> dict:
    p = t / max(d, 1e-6)
    bend = math.sin(min(1.0, p) * math.pi)
    return _p(lean=0.75 * bend, hip_drop=0.10 * bend,
              arm_r=0.9 * bend + 0.1, elbow_r=0.25,
              arm_l=-0.5 * bend, elbow_l=0.6,
              thigh_l=0.45 * bend, thigh_r=0.20 * bend,
              knee_l=0.9 * bend, knee_r=0.55 * bend, head_tilt=0.25 * bend)


def a_drop(t: float, d: float) -> dict:
    p = t / max(d, 1e-6)
    release = ease("ease_out", min(1.0, p * 1.5))
    return _p(arm_r=1.1 - release * 0.6, elbow_r=0.3 + release * 0.4,
              arm_l=-0.3, elbow_l=0.4, lean=0.06 * (1 - release))


def a_open_door(t: float, d: float) -> dict:
    p = min(1.0, t / max(d, 1e-6))
    reach = ease("ease_out", min(1.0, p * 1.4))
    return _p(arm_r=1.35 * reach, elbow_r=0.5 - reach * 0.35,
              arm_l=-0.2, elbow_l=0.3, lean=0.10 * reach,
              thigh_l=0.12, thigh_r=-0.10, knee_l=0.2, knee_r=0.15)


def a_close_door(t: float, d: float) -> dict:
    p = min(1.0, t / max(d, 1e-6))
    push = ease("ease_in_out", p)
    return _p(arm_r=1.5 - push * 0.5, elbow_r=0.2 + push * 0.9,
              arm_l=-0.2, elbow_l=0.3, lean=0.12 - push * 0.06,
              thigh_l=0.15, thigh_r=-0.12, knee_l=0.25, knee_r=0.2)


def a_push(t: float, d: float) -> dict:
    p = (t * 0.9) % 1.0
    effort = 0.5 + 0.5 * math.sin(p * math.pi)
    return _p(arm_l=-1.35, arm_r=1.35, elbow_l=0.15, elbow_r=0.15,
              lean=0.22 + effort * 0.08,
              thigh_l=0.42, thigh_r=-0.30, knee_l=0.55, knee_r=0.25,
              head_tilt=0.12)


def a_pull(t: float, d: float) -> dict:
    p = (t * 0.9) % 1.0
    effort = 0.5 + 0.5 * math.sin(p * math.pi)
    return _p(arm_l=-1.2, arm_r=1.2, elbow_l=1.2, elbow_r=1.2,
              lean=-0.20 - effort * 0.08,
              thigh_l=0.30, thigh_r=-0.42, knee_l=0.25, knee_r=0.55,
              head_tilt=-0.10)


ACTIONS: Dict[str, Callable[[float, float], dict]] = {
    "stand": a_stand,
    "idle": a_idle,
    "walk": a_walk,
    "run": a_run,
    "jump": a_jump,
    "sit": a_sit,
    "talk": a_talk,
    "point": a_point,
    "wave": a_wave,
    "fight": a_fight,
    "fall": a_fall,
    "cry": a_cry,
    "laugh": a_laugh,
    "look_around": a_look_around,
    "pick_up": a_pick_up,
    "drop": a_drop,
    "open_door": a_open_door,
    "close_door": a_close_door,
    "push": a_push,
    "pull": a_pull,
}

ACTION_ALIASES = {
    "standing": "stand", "walking": "walk", "running": "run", "jumping": "jump",
    "sitting": "sit", "talking": "talk", "pointing": "point", "waving": "wave",
    "fighting": "fight", "punch": "fight", "falling": "fall", "crying": "cry",
    "laughing": "laugh", "looking_around": "look_around", "look": "look_around",
    "picking_up": "pick_up", "pickup": "pick_up", "dropping": "drop",
    "opening_door": "open_door", "closing_door": "close_door",
    "pushing": "push", "pulling": "pull",
}


def resolve_action(name: str) -> str:
    n = (name or "stand").strip().lower().replace("-", "_").replace(" ", "_")
    n = ACTION_ALIASES.get(n, n)
    return n if n in ACTIONS else "stand"


def pose_for(action: str, t: float, duration: float, blend: float = 0.0,
             previous: str | None = None) -> dict:
    """Pose for `action` at time t, optionally blended in from `previous`."""
    a = ACTIONS[resolve_action(action)]
    pose = a(t, duration)
    if previous and blend > 0:
        b = ACTIONS[resolve_action(previous)]
        prev_pose = b(max(0.0, t - 0.001), duration)
        pose = {k: prev_pose.get(k, 0.0) * (1 - blend) + pose.get(k, 0.0) * blend
                for k in set(pose) | set(prev_pose)}
    return pose


def action_expression(action: str) -> str | None:
    """A sensible default expression for an action, unless the scene says otherwise."""
    return {
        "cry": "crying", "laugh": "laughing", "fight": "angry", "fall": "scared",
        "push": "determined", "pull": "determined", "pick_up": "neutral",
        "wave": "happy", "talk": "neutral", "point": "neutral",
    }.get(resolve_action(action))
