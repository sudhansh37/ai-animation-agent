# AI Animation Agent

Story in → **vertical 9:16 animated video** out. Runs on a CPU only: no GPU, no
CUDA, no Colab, no cloud rendering, no paid rendering service.

```
STORY ─► ANALYSIS ─► NARRATION ─► CHARACTERS ─► STORYBOARD
      ─► VOICE ─► SOUND ─► SKELETAL ANIMATION ─► CAMERA ─► SUBTITLES
      ─► CPU RENDER (FFmpeg) ─► 1080×1920 · 30 fps · H.264 + AAC
```

Output is always **1080 × 1920, 30 fps, MP4, H.264 video, AAC audio** — YouTube
Shorts / Reels / TikTok geometry. There is no landscape mode and no code path
that changes the aspect ratio.

---

## Quick start (2 commands)

```bash
pip install -r requirements.txt          # numpy + Pillow (+ cairosvg optional)
python -m app.main --project projects/pink_paint --preview
```

That renders the bundled sample story (the pink-paint experiment) to
`output/pink_paint_preview.mp4`. Drop `--preview` for the full-resolution render.

Needs **Python 3.9+** and **FFmpeg** on your PATH:

```bash
sudo apt-get install ffmpeg      # Debian/Ubuntu
brew install ffmpeg              # macOS
```

No FFmpeg? `pip install imageio-ffmpeg` gives you a bundled static build, which
the pipeline picks up automatically.

### Run it from your phone (GitHub Actions)

1. **Settings → Secrets and variables → Actions** → add `GEMINI_API_KEY`
   (optional; only used for nicer narration/story parsing and Gemini TTS).
2. **Actions → “1 · Generate video” → Run workflow.**
   Paste a story, tick *Fast half-resolution preview* the first time, run it.
3. The run summary ends with **⬇️ Download the video** — a direct MP4 link
   (release asset) plus the `animated-short` artifact.

Everything is rendered on the Actions CPU runner. No video is committed to the
repository.

---

## What the pipeline actually does

| stage | module | what happens |
|---|---|---|
| Analysis | `story_engine/analyzer.py` | characters, locations, objects, one *beat* per sentence, emotion + action per beat. Gemini when a key exists, otherwise a deterministic rule-based analyser. |
| Narration | `story_engine/script_writer.py` | one narration line per beat, timed at 2.6 words/second. Falls back to the story text verbatim — it never silently rewrites your story. |
| Characters | `character_engine/extractor.py`, `storyboard.py` | cast from `characters.json`; optional reference images are colour/proportion sampled. |
| Storyboard | `story_engine/storyboard.py` | scenes with duration, background, camera move, poses, expressions, props, SFX, captions, transitions → `storyboard.json` |
| Voice | `audio_engine/tts.py` | Piper (offline) → Gemini TTS → espeak-ng. If all fail it says so and continues without narration. |
| Audio | `audio_engine/sound_effects.py`, `audio_mixer.py` | SFX synthesised with NumPy; music bed at −18 dB, ducked under the voice; limiter on the summed mix only. |
| Animation | `character_engine/character_rig.py`, `animation.py` | 2D skeleton: hips → spine → neck → head, arms with elbows, legs with knees. 20 actions, 9 expressions. |
| Camera | `scene_engine/camera.py` | the scene is drawn on a larger world canvas; the camera crops a 9:16 window out of it, so zoom/pan/tilt are real. |
| Subtitles | `scene_engine/subtitles.py` | burned-in captions in the safe area + `captions.srt` |
| Render | `render_engine/*` | frame-by-frame into FFmpeg over a pipe; one segment per scene; concat + mux; verify. |

### The character rig

A character is a small bone hierarchy, not a sprite. Poses are joint angles, so
limbs rotate at the joints:

```
hip ──► spine ──► neck ──► head
 │                 └──► shoulder ──► elbow ──► hand
 └──► knee ──► ankle ──► foot
```

20 actions: stand, idle, walk, run, jump, sit, talk, point, wave, fight, fall,
cry, laugh, look_around, pick_up, drop, open_door, close_door, push, pull.
9 expressions: neutral, happy, sad, angry, surprised, scared, confused, laughing,
crying. Blinking, breathing and head turns run on top of every action, so nobody
is ever completely frozen.

### Character references — what is real

`character_engine/extractor.py` does two honest things:

* **Palette + proportion extraction** — samples a reference PNG and derives skin,
  hair, outfit and accent colours and the body build, so the character keeps the
  reference look.
* **Manual part sheets** — drop `head.png`, `torso.png`, `arm_upper.png`,
  `arm_lower.png`, `hand.png`, `leg_upper.png`, `leg_lower.png`, `foot.png` into
  `assets/characters/<id>/` and those parts are used for the rig.

It is **not** a semantic segmenter: there is no model that cuts a photo into body
parts on CPU. If you need exact art, use the part sheets (that is what they are
for). Anything the extractor could not determine is listed in the spec's `notes`
instead of being invented.

### Backgrounds

Procedural: `lab`, `prison_cell`, `prison_corridor`, `prison_yard`, `office`,
`city`, `forest`, `room`, `night_sky`, `plain`, `wall`.

Or use your own image — it is used as-is, characters composited on top, nothing
added:

```json
"background": { "kind": "lab", "image": "assets/backgrounds/prison_hall.png" }
```

---

## Command line

```bash
python -m app.main --project projects/pink_paint             # full render
python -m app.main --project projects/pink_paint --preview    # half res, fast
python -m app.main --project projects/pink_paint --resume     # keep finished scenes
python -m app.main --project projects/pink_paint --plan-only  # stop after the storyboard
python -m app.main --story "A robot learns to bake." --project projects/robot
python -m app.main --project projects/pink_paint --no-ai      # no API calls at all
python -m app.main --project projects/pink_paint --no-tts     # silent video
python -m app.main --project projects/pink_paint --clean      # wipe intermediates
```

Useful flags: `--voice`, `--voice-speed`, `--tts {auto,piper,gemini,espeak}`,
`--require-tts`, `--crf`, `--preset`, `--world-scale`, `--music-db`, `--sfx-db`.

### Cloning a reference video

`--storyboard` renders a hand-written shot list and skips analysis/planning — that
is how you match an existing edit:

```bash
python -m app.main --project projects/pink_cell \
    --storyboard projects/pink_cell/storyboard.json
```

`projects/pink_cell/` is a worked example: a 20-second clone of a stick-figure
prison short, with the shot list timed to the reference's cuts, a `caption` field
per scene (short keyword captions, like the reference), and the `stick` body style
(`"bodyStyle": "stick"` on a character spec gives the oversized head, thin limbs
and slim torso).

### Desktop interface

```bash
python -m app.interface
```

Story box, style/voice/length pickers, character and background uploads,
storyboard preview, live progress, **Pause / Resume / Cancel**, and a download
button. It shells out to the same CLI, so it can never drift from the pipeline.

---

## Project layout

```
app/              main.py (CLI) · interface.py (GUI) · project_manager.py
story_engine/     analyzer.py · script_writer.py · storyboard.py
character_engine/ extractor.py · character_rig.py · expressions.py · animation.py
scene_engine/     backgrounds.py · objects.py · camera.py · transitions.py · subtitles.py
audio_engine/     tts.py · sound_effects.py · audio_mixer.py
render_engine/    frame_renderer.py · scene_renderer.py · video_exporter.py · draw_utils.py
assets/           characters/ backgrounds/ sounds/ fonts/ (DejaVu, bundled)
projects/         one folder per video: story, characters, storyboard, checkpoints
output/           finished MP4s
tests/            test_pipeline.py — renders a real MP4 and asserts the spec
```

A project folder after a run:

```
projects/pink_paint/
  story.txt  characters.json  analysis.json  script.json  storyboard.json
  captions.srt  state.json  render_report.json
  audio/    narration_s01.wav … mix.wav  audio_report.json
  segments/ s01_405f.mp4 …        ← scene checkpoints (resume works from here)
```

---

## Performance (measured on a 2-core CPU container)

| setting | speed | 25 s video (≈750 frames) |
|---|---|---|
| full 1080×1920 | ~3.2 fps | ~4 minutes |
| half-res preview | ~7 fps | ~2 minutes |

Rendering is frame-by-frame and streamed straight into FFmpeg, so RAM stays flat
(~300 MB) regardless of video length. Backgrounds, shadows and the vignette are
cached; that alone was a 4× speed-up.

---

## Checkpoints, resume, errors

* Every scene is its own segment, named with its frame count
  (`s01_405f.mp4`), written to a `.part.mp4` and renamed only when complete — a
  killed run can never leave a half-finished scene that looks done.
* `--resume` re-renders only what is missing. Scene timing changes invalidate
  their segments automatically.
* A failed scene raises with FFmpeg's message and does not delete the others.
* Missing FFmpeg, missing fonts, unreadable references and TTS failures all
  produce a clear message and a fallback where one exists.

---

## Troubleshooting

| symptom | fix |
|---|---|
| `ffmpeg not found` | `apt-get install ffmpeg`, or `pip install imageio-ffmpeg`, or set `FFMPEG_BINARY` |
| `No usable TTF font found` | keep `assets/fonts/DejaVuSans-Bold.ttf`, or `apt-get install fonts-dejavu` |
| `TTSUnavailable` | `pip install piper-tts` + a voice in `assets/voices/`, or set `GEMINI_API_KEY`, or `apt-get install espeak-ng` |
| narration missing in the video | read the log: the pipeline names the backend that failed instead of pretending |
| characters overlap | widen the `x` values in `storyboard.json`, or edit `_placement()` |
| video is 540×960 | you ran `--preview`; the full run is 1080×1920 |
| render killed mid-way | re-run with `--resume` |

---

## Honest limitations

* The style is **2D vector animation that reads as a stylised 3D-ish scene** —
  it is not true 3D. Nothing here claims otherwise.
* Automatic character extraction is colour/proportion sampling, not image
  segmentation; exact art needs part sheets.
* Gemini analysis and Gemini TTS need a key and network. Without them the
  rule-based analyser and espeak-ng keep the pipeline working, at lower quality.
* Rendering is CPU-bound: ~3 fps at 1080p. Long videos take minutes, not seconds.

---

## License

MIT — see `LICENSE`. Bundled DejaVu fonts are under their own permissive licence.
