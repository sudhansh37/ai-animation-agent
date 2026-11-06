"""Builds the final stereo track: narration + sound effects + music bed.

Rules
  * narration is placed at each scene's start and never retimed
  * music sits at -18 dB under the voice and is ducked further while narration plays
  * SFX sit at -10 dB with their peaks capped relative to the voice peak
  * a soft limiter touches only the summed mix
"""

from __future__ import annotations

import json
import os
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

from audio_engine import sound_effects

SR = 44100


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #

def read_wav(path: str, sr: int = SR) -> np.ndarray:
    with wave.open(path, "rb") as w:
        ch, sw, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"{path}: unsupported sample width {sw}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if rate != sr:
        idx = np.linspace(0, len(data) - 1, max(1, int(len(data) * sr / rate)))
        data = np.interp(idx, np.arange(len(data)), data)
    return data


def write_wav(path: str, data: np.ndarray, sr: int = SR) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    stereo = np.stack([data, data], axis=1) if data.ndim == 1 else data
    ints = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(ints.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(ints.tobytes())
    return path


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x), dtype=x.dtype)])


def _db(x: float) -> float:
    return 20 * np.log10(x) if x > 1e-9 else -120.0


# --------------------------------------------------------------------------- #
# music bed
# --------------------------------------------------------------------------- #

def music_bed(duration: float, sr: int = SR, mood: str = "tension") -> np.ndarray:
    """A slow, low pad — deliberately simple so it never fights the narration."""
    n = int(max(0.5, duration) * sr)
    t = np.arange(n) / sr
    roots = {"tension": [110.0, 130.8, 98.0], "calm": [130.8, 164.8, 110.0],
             "hopeful": [146.8, 185.0, 220.0], "dark": [82.4, 98.0, 73.4]}
    chord = roots.get(mood, roots["tension"])
    pad = np.zeros(n)
    for i, f in enumerate(chord):
        for detune in (0.997, 1.0, 1.004):
            pad += np.sin(2 * math.pi * f * detune * t + i * 0.7) * (0.12 / (i + 1))
    # slow amplitude movement + gentle low-pass
    lfo = 0.6 + 0.4 * np.sin(2 * math.pi * 0.06 * t)
    pad *= lfo
    kernel = np.ones(1201) / 1201
    pad = np.convolve(pad, kernel, mode="same")
    fade = int(min(2.0, duration / 4) * sr)
    if fade > 2:
        pad[:fade] *= np.linspace(0, 1, fade)
        pad[-fade:] *= np.linspace(1, 0, fade)
    peak = np.max(np.abs(pad)) or 1.0
    return pad / peak * 0.5


import math  # noqa: E402  (used by music_bed)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

def build_track(scenes: List[dict], narration_files: Dict[str, str], total_duration: float,
                out_path: str, sr: int = SR, music: bool = True, mood: str = "tension",
                music_db: float = -18.0, sfx_db: float = -10.0,
                sfx_peak_ratio: float = 0.7) -> dict:
    n = int(round(total_duration * sr))
    voice = np.zeros(n)
    sfx_track = np.zeros(n)
    cues_placed = 0
    narration_placed = 0

    for scene in scenes:
        start = float(scene.get("start", 0.0))
        sid = scene.get("id")
        path = narration_files.get(sid)
        if path and os.path.exists(path):
            data = read_wav(path, sr)
            i0 = int(start * sr)
            i1 = min(n, i0 + len(data))
            if i0 < n:
                voice[i0:i1] += data[: i1 - i0]
                narration_placed += 1
        for cue in scene.get("sfx") or []:
            name = cue.get("name") if isinstance(cue, dict) else str(cue)
            at = float(cue.get("at", 0.3)) if isinstance(cue, dict) else 0.3
            gain = float(cue.get("gain", 0.7)) if isinstance(cue, dict) else 0.7
            clip = sound_effects.get(name, sr) * gain
            i0 = int((start + at) * sr)
            i1 = min(n, i0 + len(clip))
            if i0 < n:
                sfx_track[i0:i1] += clip[: i1 - i0]
                cues_placed += 1

    voice_peak = float(np.max(np.abs(voice))) if n else 0.0
    if voice_peak <= 1e-6:
        voice_peak = 0.3  # silent-video mode: still give music/sfx a sane reference

    mix = voice.copy()

    if music:
        bed = music_bed(total_duration, sr, mood)
        bed = _fit(bed, n) * (voice_peak * (10 ** (music_db / 20.0)))
        # duck the bed under narration
        if np.any(voice):
            window = int(0.25 * sr)
            env = np.abs(voice)
            if window > 1:
                env = np.convolve(env, np.ones(window) / window, mode="same")
            env = env / (np.max(env) or 1.0)
            bed *= (1.0 - 0.55 * np.clip(env * 3.0, 0, 1))
        mix += bed

    if np.any(sfx_track):
        sfx_track *= (voice_peak * (10 ** (sfx_db / 20.0))) / (np.max(np.abs(sfx_track)) or 1.0)
        cap = voice_peak * sfx_peak_ratio
        peak = float(np.max(np.abs(sfx_track)))
        if peak > cap > 0:
            sfx_track *= cap / peak
        mix += sfx_track

    peak = float(np.max(np.abs(mix))) if n else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak

    write_wav(out_path, mix, sr)
    summary = {
        "duration": round(n / sr, 3),
        "voice_peak_db": round(_db(voice_peak), 2),
        "mix_peak_db": round(_db(float(np.max(np.abs(mix))) if n else 0.0), 2),
        "music": music,
        "music_db": music_db,
        "sfx_db": sfx_db,
        "sfx_peak_ratio": sfx_peak_ratio,
        "narration_clips": narration_placed,
        "sfx_cues": cues_placed,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(out_path)), "audio_report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
