"""Renders one scene to an MP4 segment by streaming raw frames into FFmpeg.

Frames are never held in memory and never written to disk, which is what keeps
the pipeline usable on a small CPU runner. Each scene is its own segment, so a
scene is also the checkpoint: an interrupted run re-renders only what is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Callable, Optional

from PIL import Image

from render_engine.frame_renderer import FrameRenderer


def ffmpeg_binary() -> str:
    exe = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not exe:
        try:  # pip-installed static build, used by CI and by tests
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    if not exe or not os.path.exists(exe):
        raise RuntimeError(
            "ffmpeg not found. Install it (apt install ffmpeg / brew install ffmpeg) "
            "or `pip install imageio-ffmpeg`, or set FFMPEG_BINARY."
        )
    return exe


class SceneRenderer:
    def __init__(self, renderer: FrameRenderer, fps: int = 30, width: int = 1080,
                 height: int = 1920, crf: int = 20, preset: str = "veryfast",
                 preview_scale: float = 1.0):
        self.renderer = renderer
        self.fps = fps
        self.width = max(2, int(width * preview_scale) // 2 * 2)
        self.height = max(2, int(height * preview_scale) // 2 * 2)
        self.crf = crf
        self.preset = preset

    # ------------------------------------------------------------------ #
    def render_scene(self, scene: dict, out_path: str, t_offset: float,
                     resume: bool = True, progress: Optional[Callable[[dict], None]] = None) -> dict:
        frames = max(1, int(round(float(scene.get("duration", 1.0)) * self.fps)))
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        if resume and os.path.exists(out_path) and os.path.getsize(out_path) > 2048:
            return {"scene": scene.get("id"), "frames": frames, "skipped": True, "path": out_path}

        # write to a temp file and rename at the end: a killed run must never
        # leave a half-written segment that looks finished
        tmp_path = (out_path[:-4] if out_path.endswith(".mp4") else out_path) + ".part.mp4"

        cmd = [
            ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
            "-pix_fmt", "yuv420p", "-g", str(self.fps * 2),
            tmp_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)

        started = time.time()
        prev_frame: Optional[Image.Image] = None
        try:
            for i in range(frames):
                t_local = i / self.fps
                frame = self.renderer.render(scene, t_local, t_offset + t_local, prev_frame)
                if (frame.width, frame.height) != (self.width, self.height):
                    frame = frame.resize((self.width, self.height), Image.LANCZOS)
                if frame.mode != "RGB":
                    frame = frame.convert("RGB")
                try:
                    proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    stderr = proc.stderr.read() if proc.stderr else b""
                    raise RuntimeError(
                        f"ffmpeg stopped accepting frames for scene {scene.get('id')}: "
                        f"{stderr.decode('utf8', 'ignore')[:600]}")
                prev_frame = frame

                if progress and (i % max(1, self.fps) == 0 or i == frames - 1):
                    elapsed = time.time() - started
                    per = elapsed / (i + 1)
                    progress({
                        "scene": scene.get("id"),
                        "frame": i + 1,
                        "frames": frames,
                        "percent": round((i + 1) / frames * 100, 1),
                        "fps": round(1 / per, 2) if per else 0,
                        "eta": round(per * (frames - i - 1), 1),
                    })
        finally:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            stderr = b""
            if proc.stderr:
                stderr = proc.stderr.read()
            code = proc.wait()

        if code != 0:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise RuntimeError(f"ffmpeg failed for scene {scene.get('id')}: "
                               f"{stderr.decode('utf8', 'ignore')[:600]}")
        os.replace(tmp_path, out_path)

        return {
            "scene": scene.get("id"),
            "frames": frames,
            "path": out_path,
            "seconds": round(time.time() - started, 2),
            "skipped": False,
        }
