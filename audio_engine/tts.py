"""Text-to-speech with honest fallbacks.

Backend order for `auto`:
  1. **Piper** — offline neural TTS (best quality, no network). Needs
     `pip install piper-tts` and a voice model: set PIPER_MODEL=/path/voice.onnx
     or drop a .onnx voice into assets/voices/.
  2. **Gemini TTS** — used when GEMINI_API_KEY is set (this is what CI uses).
  3. **espeak-ng** — robotic but offline and always available on Linux.
  4. Nothing available → raises `TTSUnavailable` with installation instructions.

If narration fails the pipeline says so; it never pretends audio was generated.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

GEMINI_TTS_MODELS = [
    os.environ.get("TTS_MODEL") or "gemini-3.8-flash-lite-tts",
    "gemini-2.5-flash-preview-tts",
]
VOICE_DIRS = ["assets/voices", os.path.expanduser("~/.local/share/piper")]


class TTSUnavailable(RuntimeError):
    pass


@dataclass
class TTSResult:
    path: str
    duration: float
    backend: str
    text: str


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #

def _wav_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        # not a wav — ask ffmpeg
        try:
            from render_engine.scene_renderer import ffmpeg_binary
            proc = subprocess.run([ffmpeg_binary(), "-hide_banner", "-i", path], capture_output=True)
            text = proc.stderr.decode("utf8", "ignore")
            import re
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", text)
            if m:
                return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        except Exception:
            pass
        return 0.0


def _find_piper_model() -> Optional[str]:
    env = os.environ.get("PIPER_MODEL")
    if env and os.path.exists(env):
        return env
    for d in VOICE_DIRS:
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                if name.endswith(".onnx"):
                    return os.path.join(d, name)
    return None


def synthesize_piper(text: str, out_path: str, speed: float = 1.0) -> TTSResult:
    exe = shutil.which("piper") or shutil.which("piper-tts")
    if not exe:
        raise TTSUnavailable("piper is not installed")
    model = _find_piper_model()
    if not model:
        raise TTSUnavailable("piper is installed but no .onnx voice model was found")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cmd = [exe, "--model", model, "--output_file", out_path]
    if speed != 1.0:
        cmd += ["--length_scale", str(1.0 / max(0.25, speed))]
    proc = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True)
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise TTSUnavailable(f"piper failed: {proc.stderr.decode('utf8', 'ignore')[:300]}")
    return TTSResult(out_path, _wav_duration(out_path), "piper", text)


def synthesize_gemini(text: str, out_path: str, voice: str = "Kore",
                      style: str = "Speak warmly and clearly, like a documentary narrator.") -> TTSResult:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise TTSUnavailable("GEMINI_API_KEY is not set")
    if requests is None:
        raise TTSUnavailable("the 'requests' package is not installed")

    body = {
        "contents": [{"role": "user", "parts": [{"text": f"{style}\n\nNarration:\n{text.strip()}"}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    last = "unknown error"
    for model in GEMINI_TTS_MODELS:
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": key}, json=body, timeout=180,
                headers={"Content-Type": "application/json"},
            )
        except Exception as exc:
            last = str(exc)[:200]
            continue
        if resp.status_code == 404:
            last = f"{model} not available for this key"
            continue
        if resp.status_code != 200:
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            continue
        data = resp.json()
        parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
        payload = None
        mime = ""
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                payload = base64.b64decode(inline["data"])
                mime = (inline.get("mimeType") or inline.get("mime_type") or "").lower()
                break
        if not payload:
            last = "no audio in response"
            continue
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        if "wav" in mime or payload[:4] == b"RIFF":
            with open(out_path, "wb") as f:
                f.write(payload)
        else:  # raw PCM
            sr = 24000
            with wave.open(out_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(payload)
        return TTSResult(out_path, _wav_duration(out_path), f"gemini:{model}", text)
    raise TTSUnavailable(f"Gemini TTS failed: {last}")


def synthesize_espeak(text: str, out_path: str, speed: float = 1.0, voice: str = "en") -> TTSResult:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise TTSUnavailable("espeak-ng is not installed")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    wpm = int(165 * max(0.5, min(2.0, speed)))
    proc = subprocess.run([exe, "-v", voice, "-s", str(wpm), "-w", out_path, text],
                          capture_output=True)
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise TTSUnavailable(f"espeak failed: {proc.stderr.decode('utf8', 'ignore')[:300]}")
    return TTSResult(out_path, _wav_duration(out_path), "espeak", text)


INSTRUCTIONS = """\
No text-to-speech backend is available. Install one of:

  Piper (offline, recommended)
      pip install piper-tts
      # download a voice, e.g. en_US-lessac-medium.onnx + .json into assets/voices/
      export PIPER_MODEL=assets/voices/en_US-lessac-medium.onnx

  Gemini TTS (needs a key)
      export GEMINI_API_KEY=...

  espeak-ng (offline, robotic but always works)
      sudo apt-get install espeak-ng

You can also render a silent video with --no-tts and add narration later.
"""


def synthesize(text: str, out_path: str, backend: str = "auto", voice: Optional[str] = None,
               speed: float = 1.0) -> TTSResult:
    text = (text or "").strip()
    if not text:
        raise TTSUnavailable("nothing to say (empty narration text)")

    backend = (backend or "auto").lower()
    errors = []

    order = {
        "auto": ["piper", "gemini", "espeak"],
        "piper": ["piper"],
        "gemini": ["gemini"],
        "espeak": ["espeak"],
    }.get(backend, ["piper", "gemini", "espeak"])

    for name in order:
        try:
            if name == "piper":
                return synthesize_piper(text, out_path, speed)
            if name == "gemini":
                return synthesize_gemini(text, out_path, voice or "Kore")
            if name == "espeak":
                return synthesize_espeak(text, out_path, speed, voice or "en")
        except TTSUnavailable as exc:
            errors.append(f"{name}: {exc}")
        except Exception as exc:  # never let one backend kill the run
            errors.append(f"{name}: {exc}")

    raise TTSUnavailable("all TTS backends failed ->\n  " + "\n  ".join(errors) + "\n\n" + INSTRUCTIONS)
