"""Final assembly: concatenate scene segments, mux the audio, verify the result."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List

from render_engine.scene_renderer import ffmpeg_binary


def _run(cmd: List[str], desc: str) -> str:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{desc} failed: {proc.stderr.decode('utf8', 'ignore')[:800]}")
    return proc.stderr.decode("utf8", "ignore")


def concat_segments(segments: List[str], out_path: str) -> str:
    """Concatenate identically-encoded segments without re-encoding."""
    if not segments:
        raise RuntimeError("no segments to concatenate")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for seg in segments:
            f.write(f"file '{os.path.abspath(seg)}'\n")
        list_path = f.name
    try:
        _run([ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y",
              "-f", "concat", "-safe", "0", "-i", list_path,
              "-c", "copy", out_path], "concat")
    finally:
        os.unlink(list_path)
    return out_path


def mux_audio(video_path: str, audio_path: str, out_path: str, fps: int = 30) -> str:
    """Attach the mixed track and normalise to the locked output spec."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    args = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path]
    if audio_path and os.path.exists(audio_path):
        args += ["-i", audio_path, "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                 "-shortest"]
    args += ["-c:v", "copy", "-movflags", "+faststart", out_path]
    _run(args, "mux")
    return out_path


def probe(path: str) -> Dict:
    """Read duration / resolution / fps / codecs out of `ffmpeg -i` (no ffprobe needed)."""
    proc = subprocess.run([ffmpeg_binary(), "-hide_banner", "-i", path],
                          capture_output=True)
    text = proc.stderr.decode("utf8", "ignore")
    info: Dict = {"path": path, "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0}

    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", text)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info["duration"] = round(h * 3600 + mi * 60 + s, 3)

    m = re.search(r"Stream #0:0.*?Video:\s*(\w+)", text)
    if m:
        info["video_codec"] = m.group(1)
    m = re.search(r"(\d{2,5})x(\d{2,5})", text)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Stream #0:1.*?Audio:\s*(\w+)", text)
    if m:
        info["audio_codec"] = m.group(1)
    info["has_audio"] = "Audio:" in text

    problems = []
    if not info.get("duration"):
        problems.append("duration unreadable")
    if info.get("width") != 1080 or info.get("height") != 1920:
        problems.append(f"resolution is {info.get('width')}x{info.get('height')}, expected 1080x1920")
    if abs(float(info.get("fps", 0)) - 30.0) > 0.05:
        problems.append(f"fps is {info.get('fps')}, expected 30")
    if info.get("video_codec") != "h264":
        problems.append(f"video codec is {info.get('video_codec')}, expected h264")
    if not info.get("has_audio"):
        problems.append("no audio stream")
    info["problems"] = problems
    return info


def export(segments: List[str], audio_path: str, out_path: str, work_dir: str,
           fps: int = 30) -> Dict:
    joined = os.path.join(work_dir, "_joined.mp4")
    concat_segments(segments, joined)
    mux_audio(joined, audio_path, out_path, fps=fps)
    try:
        os.unlink(joined)
    except OSError:
        pass
    return probe(out_path)


def write_report(info: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return path
