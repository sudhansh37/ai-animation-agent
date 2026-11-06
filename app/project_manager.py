"""Project layout, checkpoints and cleanup.

A project directory holds everything for one video:

    projects/<name>/
      story.txt            input
      characters.json      character specs (generated or hand-edited)
      analysis.json        story analysis
      script.json          narration lines
      storyboard.json      scenes, camera, poses, sfx
      captions.srt         subtitles
      audio/               narration_XX.wav, mix.wav
      segments/            scene_XX.mp4 checkpoints
      state.json           what is done, so a run can resume
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional


class Project:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.dirs = {
            "audio": os.path.join(self.root, "audio"),
            "segments": os.path.join(self.root, "segments"),
            "frames": os.path.join(self.root, "frames"),
        }

    # ------------------------------------------------------------------ #
    def ensure(self) -> "Project":
        os.makedirs(self.root, exist_ok=True)
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)
        return self

    def path(self, name: str) -> str:
        return os.path.join(self.root, name)

    # ------------------------------------------------------------------ #
    def read_text(self, name: str, default: str = "") -> str:
        p = self.path(name)
        if not os.path.exists(p):
            return default
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    def write_text(self, name: str, text: str) -> str:
        p = self.path(name)
        os.makedirs(os.path.dirname(p) or self.root, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return p

    def read_json(self, name: str, default=None):
        p = self.path(name)
        if not os.path.exists(p):
            return default
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def write_json(self, name: str, data) -> str:
        p = self.path(name)
        os.makedirs(os.path.dirname(p) or self.root, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return p

    # ------------------------------------------------------------------ #
    def load_state(self) -> Dict:
        return self.read_json("state.json", {}) or {}

    def save_state(self, state: Dict) -> str:
        return self.write_json("state.json", state)

    def mark_done(self, key: str, value=True) -> Dict:
        state = self.load_state()
        state[key] = value
        self.save_state(state)
        return state

    # ------------------------------------------------------------------ #
    def segment_path(self, scene_id: str, frames: Optional[int] = None) -> str:
        """Segments are keyed by their frame count: if a scene's timing changes,
        the old file is simply not found and the scene is re-rendered."""
        name = f"{scene_id}_{int(frames)}f.mp4" if frames else f"{scene_id}.mp4"
        return os.path.join(self.dirs["segments"], name)

    def narration_path(self, scene_id: str) -> str:
        return os.path.join(self.dirs["audio"], f"narration_{scene_id}.wav")

    def mix_path(self) -> str:
        return os.path.join(self.dirs["audio"], "mix.wav")

    # ------------------------------------------------------------------ #
    def clean(self, what: str = "all") -> List[str]:
        """Remove generated intermediates. Returns the paths removed."""
        removed: List[str] = []
        targets = []
        if what in ("all", "segments"):
            targets.append(self.dirs["segments"])
        if what in ("all", "frames"):
            targets.append(self.dirs["frames"])
        if what in ("all", "audio"):
            targets.append(self.dirs["audio"])
        for path in targets:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path)
            os.makedirs(path, exist_ok=True)
        if what in ("all", "state"):
            state = self.path("state.json")
            if os.path.exists(state):
                os.unlink(state)
                removed.append(state)
        return removed

    def segments(self, scene_ids: Optional[List[str]] = None) -> List[str]:
        if scene_ids:
            found = []
            for sid in scene_ids:
                found += [os.path.join(self.dirs["segments"], f) for f in
                          (os.listdir(self.dirs["segments"]) if os.path.isdir(self.dirs["segments"]) else [])
                          if f.startswith(sid + "_") and f.endswith(".mp4")]
            return found
        if not os.path.isdir(self.dirs["segments"]):
            return []
        files = [os.path.join(self.dirs["segments"], f) for f in sorted(os.listdir(self.dirs["segments"]))
                 if f.endswith(".mp4")]
        return files


def create(name: str, base: str = "projects") -> Project:
    return Project(os.path.join(base, name)).ensure()
