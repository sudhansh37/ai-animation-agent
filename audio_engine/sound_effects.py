"""Procedurally synthesised sound effects (no downloaded assets required).

Every effect is generated with NumPy, so the repo stays small and there are no
licensing questions. If a real recording is dropped into assets/sounds/<name>.wav
it is used instead of the synth version.
"""

from __future__ import annotations

import math
import os
import wave
from typing import Dict, List, Optional

import numpy as np

SR = 44100
ASSET_DIR = "assets/sounds"

NAMES = ["footsteps", "door_open", "door_close", "punch", "impact", "object_move",
         "machine", "ambience", "whoosh", "thud", "alarm", "splash", "ting", "crowd"]

_cache: Dict[str, np.ndarray] = {}


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _env(n: int, attack: float = 0.01, decay: float = 6.0, sr: int = SR) -> np.ndarray:
    a = max(1, int(attack * sr))
    env = np.exp(-np.linspace(0, 1, n) * decay)
    env[:a] *= np.linspace(0, 1, a)
    return env


def _lowpass(x: np.ndarray, k: int = 9) -> np.ndarray:
    if k < 2:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def _highpass(x: np.ndarray, k: int = 41) -> np.ndarray:
    return x - _lowpass(x, k)


def _noise(n: int, seed: int = 1) -> np.ndarray:
    return _rng(seed).uniform(-1, 1, n)


def _wav_from_asset(name: str, sr: int) -> Optional[np.ndarray]:
    path = os.path.join(ASSET_DIR, f"{name}.wav")
    if not os.path.exists(path):
        return None
    try:
        with wave.open(path, "rb") as w:
            ch, sw, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
        if sw != 2:
            return None
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        if rate != sr:
            idx = np.linspace(0, len(data) - 1, int(len(data) * sr / rate))
            data = np.interp(idx, np.arange(len(data)), data)
        return data
    except Exception:
        return None


def build(name: str, sr: int = SR) -> np.ndarray:
    """Synthesise one effect, mono float32 in [-1, 1]."""
    name = (name or "").lower().strip()

    if name == "footsteps":
        n = int(0.9 * sr)
        out = np.zeros(n)
        for i, offset in enumerate((0.0, 0.34)):
            start = int(offset * sr)
            seg_n = int(0.16 * sr)
            tt = np.arange(seg_n) / sr
            thud = np.sin(2 * math.pi * 95 * tt) * np.exp(-tt * 30)
            click = _highpass(_noise(seg_n, seed=3 + i), 21) * np.exp(-tt * 60) * 0.4
            end = min(n, start + seg_n)
            out[start:end] += (thud + click)[: end - start] * (0.9 if i == 0 else 0.7)
        return out * 0.7

    if name in ("door_open", "door_close"):
        n = int(1.1 * sr)
        tt = np.arange(n) / sr
        creak_f = 220 + 90 * np.sin(2 * math.pi * 1.6 * tt) - 60 * tt
        creak = np.sin(2 * math.pi * np.cumsum(creak_f) / sr) * 0.5
        creak = _lowpass(creak, 15) * np.exp(-tt * 2.2)
        thump = np.sin(2 * math.pi * 70 * tt) * np.exp(-tt * 18) * (0.9 if name == "door_close" else 0.3)
        click = _noise(n, seed=7) * np.exp(-tt * 90) * (0.5 if name == "door_close" else 0.1)
        return (creak * 0.6 + thump + click) * 0.8

    if name in ("punch", "impact", "thud"):
        n = int(0.28 * sr)
        tt = np.arange(n) / sr
        body = np.sin(2 * math.pi * (110 if name == "thud" else 85) * tt) * np.exp(-tt * 26)
        slap = _highpass(_noise(n, seed=11), 11) * np.exp(-tt * 45) * (0.7 if name == "punch" else 0.4)
        return (body + slap) * 0.85

    if name == "object_move":
        n = int(0.5 * sr)
        tt = np.arange(n) / sr
        scrape = _highpass(_lowpass(_noise(n, seed=13), 5), 61)
        mod = 0.5 + 0.5 * np.sin(2 * math.pi * 13 * tt)
        return scrape * mod * np.exp(-tt * 3.2) * 0.6

    if name == "machine":
        n = int(1.6 * sr)
        tt = np.arange(n) / sr
        hum = np.sin(2 * math.pi * 60 * tt) * 0.4 + np.sin(2 * math.pi * 120 * tt) * 0.2
        hum *= 0.7 + 0.3 * np.sin(2 * math.pi * 4 * tt)
        clatter = _noise(n, seed=17) * 0.05 * (0.5 + 0.5 * np.sin(2 * math.pi * 7 * tt))
        env = np.minimum(1, np.minimum(tt / 0.15, (n / sr - tt) / 0.2))
        return (hum + clatter) * np.clip(env, 0, 1) * 0.5

    if name == "ambience":
        n = int(6.0 * sr)
        tt = np.arange(n) / sr
        bed = _lowpass(_noise(n, seed=19), 601)
        bed = bed / (np.max(np.abs(bed)) or 1)
        rumble = np.sin(2 * math.pi * 42 * tt) * 0.08
        env = np.minimum(1, np.minimum(tt / 1.2, (n / sr - tt) / 1.2))
        return (bed * 0.35 + rumble) * np.clip(env, 0, 1)

    if name == "whoosh":
        n = int(0.6 * sr)
        tt = np.arange(n) / sr
        x = _highpass(_noise(n, seed=23), 41)
        return x * np.exp(-((tt - 0.2) ** 2) / 0.008) * 0.7

    if name == "alarm":
        n = int(1.4 * sr)
        tt = np.arange(n) / sr
        tone = np.sin(2 * math.pi * 740 * tt) * (0.5 + 0.5 * np.sign(np.sin(2 * math.pi * 3 * tt)))
        return tone * np.exp(-tt * 1.2) * 0.4

    if name == "splash":
        n = int(0.8 * sr)
        tt = np.arange(n) / sr
        x = _highpass(_noise(n, seed=29), 5) * np.exp(-tt * 8)
        bubbles = sum(np.sin(2 * math.pi * (300 + i * 130) * tt) * np.exp(-tt * (10 + i * 3)) * 0.12
                      for i in range(4))
        return (x * 0.7 + bubbles) * 0.8

    if name == "ting":
        n = int(0.7 * sr)
        tt = np.arange(n) / sr
        return (np.sin(2 * math.pi * 1760 * tt) + 0.4 * np.sin(2 * math.pi * 2640 * tt)) * np.exp(-tt * 9) * 0.6

    if name == "crowd":
        n = int(3.0 * sr)
        tt = np.arange(n) / sr
        voices = sum(_lowpass(_noise(n, seed=31 + i), 101) for i in range(4)) / 4.0
        mod = 0.6 + 0.4 * np.sin(2 * math.pi * 0.7 * tt)
        env = np.minimum(1, np.minimum(tt / 0.6, (n / sr - tt) / 0.8))
        return voices * mod * np.clip(env, 0, 1) * 0.35

    # unknown name → neutral blip, never silence-with-a-crash
    n = int(0.15 * sr)
    tt = np.arange(n) / sr
    return np.sin(2 * math.pi * 660 * tt) * np.exp(-tt * 30) * 0.35


def get(name: str, sr: int = SR) -> np.ndarray:
    key = f"{name}@{sr}"
    if key in _cache:
        return _cache[key]
    asset = _wav_from_asset(name, sr)
    data = asset if asset is not None else build(name, sr)
    _cache[key] = data
    return data


def write_all(out_dir: str, sr: int = SR) -> List[str]:
    """Materialise every effect as a wav (useful for inspecting / reusing them)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name in NAMES:
        data = get(name, sr)
        path = os.path.join(out_dir, f"{name}.wav")
        ints = (np.clip(data, -1, 1) * 32767).astype("<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(ints.tobytes())
        paths.append(path)
    return paths
