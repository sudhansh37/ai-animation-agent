"""Command line entry point — the whole autonomous pipeline.

    python -m app.main --project projects/pink_paint            # uses story.txt
    python -m app.main --story "A scientist discovers..."       # ad-hoc story
    python -m app.main --project projects/pink_paint --preview  # half-res, fast
    python -m app.main --project projects/pink_paint --resume   # keep finished scenes

Stages: analyse → narration → characters → storyboard → voice → audio mix →
scene render (checkpointed) → export → verify.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

# make the repo importable when run from anywhere
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.project_manager import Project, create  # noqa: E402
from audio_engine import audio_mixer, tts  # noqa: E402
from character_engine import extractor  # noqa: E402
from render_engine import video_exporter  # noqa: E402
from render_engine.frame_renderer import FrameRenderer  # noqa: E402
from render_engine.scene_renderer import SceneRenderer  # noqa: E402
from scene_engine import subtitles  # noqa: E402
from scene_engine.camera import Camera  # noqa: E402
from story_engine import analyzer, script_writer, storyboard as sb  # noqa: E402


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #

def prepare_story(project: Project, story: Optional[str], story_file: Optional[str]) -> str:
    if story:
        text = story.strip()
        project.write_text("story.txt", text + "\n")
        return text
    if story_file:
        with open(story_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        project.write_text("story.txt", text + "\n")
        return text
    text = project.read_text("story.txt").strip()
    if not text:
        raise SystemExit("No story found. Pass --story, --story-file, or put story.txt in the project.")
    return text


def analyse_and_plan(project: Project, text: str, args) -> Dict:
    step("1/8  Story analysis")
    analysis = analyzer.analyze(text, use_ai=not args.no_ai)
    project.write_json("analysis.json", analysis)
    log(f"source: {analysis.get('source')} | characters: "
        f"{', '.join(c['id'] for c in analysis['characters'])} | beats: {len(analysis['beats'])}")

    step("2/8  Narration")
    narration = script_writer.write_narration(text, analysis["beats"], use_ai=not args.no_ai)
    project.write_json("script.json", narration)
    log(f"{len(narration['lines'])} lines, {narration['words']} words "
        f"(~{narration['total_seconds']:.1f}s), source: {narration['source']}")

    step("3/8  Characters")
    project_cast = (project.read_json("characters.json", {}) or {}).get("characters") or []
    characters = sb.build_characters(analysis, args.style or {})
    if project_cast:
        # a hand-written cast (or one produced by an earlier run) wins: it is the
        # only way to keep a character that the story text never names
        characters = []
        for i, spec in enumerate(project_cast):
            merged = dict(sb.ROLE_TEMPLATES["default"])
            role_key = next((r for r in sb.ROLE_TEMPLATES if r != "default" and r in str(spec.get("id", ""))), None)
            if role_key:
                merged.update(sb.ROLE_TEMPLATES[role_key])
            merged.update({k: v for k, v in spec.items() if v is not None})
            merged.setdefault("id", f"char{i}")
            merged.setdefault("name", merged["id"].title())
            merged.setdefault("role", "protagonist" if i == 0 else "support")
            characters.append(merged)
        log(f"cast from characters.json ({len(characters)} characters)")
    # optional reference images: assets/characters/<id>.png
    for spec in characters:
        ref = os.path.join("assets", "characters", f"{spec['id']}.png")
        if os.path.exists(ref):
            extracted = extractor.spec_from_reference(spec["id"], ref, spec.get("name"))
            merged = {**spec, **{k: v for k, v in extracted.items()
                                 if k in ("skin", "hair", "outfit", "accent", "build", "reference")}}
            spec.update(merged)
            log(f"  {spec['id']}: palette taken from {ref}")
    if not project_cast:  # never clobber a hand-written cast
        project.write_json("characters.json", {"characters": characters})
    log(f"cast: {', '.join(c['id'] for c in characters)}")

    step("4/8  Storyboard")
    board = sb.build_storyboard(analysis, narration, {
        "fps": args.fps, "width": args.width, "height": args.height,
        "style": args.style or {},
        "mood": args.mood,
        "cast": characters,
    })
    project.write_json("storyboard.json", board)
    print(sb.summarise(board), flush=True)
    return board


def build_audio(project: Project, board: Dict, args) -> Dict:
    step("5/8  Voice generation")
    narration_files: Dict[str, str] = {}
    if args.no_tts:
        log("--no-tts: rendering a silent video (no narration will be claimed)")
    else:
        for scene in board["scenes"]:
            out = project.narration_path(scene["id"])
            if args.resume and os.path.exists(out) and os.path.getsize(out) > 2048:
                narration_files[scene["id"]] = out
                continue
            try:
                result = tts.synthesize(scene["narration"], out, backend=args.tts,
                                        voice=args.voice, speed=args.voice_speed)
                raw_len = result.duration
                result.duration = tts.trim_silence(out)      # TTS pads silence; trim it
                result.duration = tts.compress_pauses(out)   # and squeeze long gaps
                narration_files[scene["id"]] = result.path
                log(f"  {scene['id']}: {result.duration:.2f}s via {result.backend}"
                    + (f" (trimmed from {raw_len:.2f}s)" if raw_len - result.duration > 0.2 else ""))
            except tts.TTSUnavailable as exc:
                log(f"  {scene['id']}: TTS failed -> {exc}")
                if args.require_tts:
                    raise
                log("  continuing without narration for this scene")
        if not narration_files:
            log("WARNING: no narration was generated — the video will have music and SFX only")

    # re-time scenes so the visuals follow the real audio
    cursor = 0.0
    retimed = False
    for scene in board["scenes"]:
        path = narration_files.get(scene["id"])
        if path:
            audio_len = tts._wav_duration(path)
            wanted = max(float(scene["duration"]), audio_len + 0.45)
            if abs(wanted - scene["duration"]) > 0.05:
                scene["duration"] = round(wanted, 3)
                retimed = True
        scene["start"] = round(cursor, 3)
        cursor += float(scene["duration"])
    board["duration"] = round(cursor, 3)
    if retimed:
        project.write_json("storyboard.json", board)
        log(f"scene timings adjusted to the narration (total {board['duration']:.2f}s)")

    step("6/8  Audio mix")
    mix = project.mix_path()
    if args.no_tts:
        for scene in board["scenes"]:
            narration_files.pop(scene["id"], None)
    summary = audio_mixer.build_track(
        board["scenes"], narration_files, board["duration"], mix,
        music=not args.no_music, mood=board.get("mood", "tension"),
        music_db=args.music_db, sfx_db=args.sfx_db,
    )
    log(f"mix: {summary['duration']:.2f}s, voice {summary['voice_peak_db']} dBFS, "
        f"music {summary['music_db']} dB, sfx {summary['sfx_db']} dB, "
        f"{summary['sfx_cues']} cues")

    # captions
    cues = subtitles.cues_from_scenes(board["scenes"])
    srt = subtitles.write_srt(cues, project.path("captions.srt"))
    project.write_json("captions.json", cues)
    log(f"captions: {len(cues)} cues -> {srt}")
    return {"narration": narration_files, "mix": mix, "cues": cues, "summary": summary}


def render_video(project: Project, board: Dict, audio: Dict, args) -> Dict:
    step("7/8  Rendering scenes")
    camera = Camera(board["width"], board["height"], world_scale=args.world_scale)
    characters = {c["id"]: c for c in board["characters"]}
    renderer = FrameRenderer(characters, camera, cues=audio["cues"], fps=board["fps"])
    scale = 0.5 if args.preview else 1.0
    scene_renderer = SceneRenderer(
        renderer, fps=board["fps"], width=board["width"], height=board["height"],
        crf=28 if args.preview else args.crf,
        preset="veryfast" if args.preview else args.preset,
        preview_scale=scale,
    )

    results: List[dict] = []
    started = time.time()
    for scene in board["scenes"]:
        frames_expected = max(1, int(round(float(scene["duration"]) * board["fps"])))
        out = project.segment_path(scene["id"], frames_expected)
        last = {"t": 0.0}

        def progress(info, _last=last):
            now = time.time()
            if now - _last["t"] > 2.0 or info["frame"] == info["frames"]:
                _last["t"] = now
                log(f"  {info['scene']}: {info['frame']}/{info['frames']} "
                    f"({info['percent']}%) {info['fps']} fps, eta {info['eta']}s")

        result = scene_renderer.render_scene(scene, out, float(scene["start"]),
                                             resume=args.resume, progress=progress)
        results.append(result)
        if result.get("skipped"):
            log(f"  {scene['id']}: already rendered (checkpoint) — skipped")
        else:
            log(f"  {scene['id']}: {result['frames']} frames in {result['seconds']}s")

    step("8/8  Export")
    segments = [project.segment_path(s["id"], max(1, int(round(float(s["duration"]) * board["fps"]))))
                for s in board["scenes"]]
    missing = [p for p in segments if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"missing segments: {missing}")

    out_path = args.out or os.path.join("output", f"{os.path.basename(project.root)}.mp4")
    if args.preview:
        out_path = out_path.replace(".mp4", "_preview.mp4")
    info = video_exporter.export(segments, audio["mix"], out_path, project.dirs["segments"],
                                 fps=board["fps"])
    video_exporter.write_report(info, project.path("render_report.json"))
    log(f"wrote {out_path}  ({info.get('size_bytes', 0) / 1e6:.1f} MB, "
        f"{info.get('duration')}s, {info.get('width')}x{info.get('height')} @ {info.get('fps')} fps)")
    if info["problems"]:
        for problem in info["problems"]:
            log(f"WARNING: {problem}")
    else:
        log("output verified: 1080x1920, 30 fps, h264 + aac")
    log(f"total render time {time.time() - started:.1f}s")
    return info


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Autonomous CPU-only animated explainer video generator")
    p.add_argument("--project", default="projects/pink_paint", help="project directory")
    p.add_argument("--story", default=None, help="story text (creates/overwrites story.txt)")
    p.add_argument("--story-file", default=None)
    p.add_argument("--out", default=None, help="output mp4 path")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1920)
    p.add_argument("--crf", type=int, default=20)
    p.add_argument("--preset", default="medium")
    p.add_argument("--world-scale", type=float, default=1.35,
                   help="how much larger than the output the world canvas is (camera room)")
    p.add_argument("--tts", default="auto", choices=["auto", "piper", "gemini", "espeak"])
    p.add_argument("--voice", default=None, help="voice name (gemini voice or espeak language)")
    p.add_argument("--voice-speed", type=float, default=1.0)
    p.add_argument("--require-tts", action="store_true", help="fail instead of continuing without narration")
    p.add_argument("--no-tts", action="store_true", help="silent video: skip narration entirely")
    p.add_argument("--no-music", action="store_true")
    p.add_argument("--no-ai", action="store_true", help="rule-based analysis/narration (no API calls)")
    p.add_argument("--music-db", type=float, default=-18.0)
    p.add_argument("--sfx-db", type=float, default=-10.0)
    p.add_argument("--mood", default=None, help="music mood: tension, calm, hopeful, dark")
    p.add_argument("--preview", action="store_true", help="half resolution, faster encode")
    p.add_argument("--resume", action="store_true", help="skip scenes already rendered")
    p.add_argument("--clean", action="store_true", help="delete intermediates before starting")
    p.add_argument("--style", default=None, help="path to a style/character override JSON")
    p.add_argument("--plan-only", action="store_true", help="stop after the storyboard")
    p.add_argument("--storyboard", default=None,
                   help="use this storyboard.json as-is (skips analysis and planning) — "
                        "the way to clone a reference edit")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.style:
        with open(args.style, "r", encoding="utf-8") as f:
            args.style = json.load(f)
    else:
        args.style = {}

    project = create(os.path.basename(os.path.normpath(args.project)),
                     base=os.path.dirname(os.path.normpath(args.project)) or "projects")
    if args.clean:
        removed = project.clean("all")
        log(f"cleaned: {', '.join(removed) if removed else 'nothing'}")

    log(f"project: {project.root}")

    if args.storyboard:
        with open(args.storyboard, "r", encoding="utf-8") as f:
            board = json.load(f)
        # never overwrite the source shot list — retimed values go to a working copy
        project.write_json("storyboard.used.json", board)
        project.write_text("story.txt", project.read_text("story.txt", "") or board.get("title", ""))
        log(f"storyboard: {args.storyboard} ({len(board['scenes'])} scenes, "
            f"{board['duration']:.2f}s) — analysis and planning skipped")
        print(sb.summarise(board), flush=True)
    else:
        text = prepare_story(project, args.story, args.story_file)
        board = analyse_and_plan(project, text, args)

    if args.plan_only:
        log("--plan-only: stopping after the storyboard")
        return 0

    audio = build_audio(project, board, args)
    info = render_video(project, board, audio, args)
    return 0 if not info["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
