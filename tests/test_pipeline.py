"""Pipeline tests. Run with:  python -m tests.test_pipeline   (or pytest)

These are real checks, not placeholders: they render actual frames, synthesise
actual audio and encode an actual MP4, then assert the output spec.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from character_engine.animation import ACTIONS, pose_for  # noqa: E402
from character_engine.character_rig import CharacterRig  # noqa: E402
from render_engine.draw_utils import font, wrap_text  # noqa: E402
from scene_engine.camera import Camera  # noqa: E402
from scene_engine.subtitles import cues_from_scenes, write_srt  # noqa: E402
from story_engine import analyzer, script_writer, storyboard as sb  # noqa: E402
from audio_engine import sound_effects  # noqa: E402

STORY = """A scientist discovers a pink paint that temporarily calms aggressive prisoners.

He tests it inside a prison.

A guard paints the walls pink.

The prisoners gradually become calm.

However, after prolonged exposure, the effect reverses and the prisoners become aggressive again."""


def test_analysis_and_storyboard():
    analysis = analyzer.analyze(STORY, use_ai=False)
    assert analysis["beats"], "no beats extracted"
    assert len(analysis["characters"]) >= 3, analysis["characters"]

    narration = script_writer.write_narration(STORY, analysis["beats"], use_ai=False)
    assert len(narration["lines"]) == len(analysis["beats"])

    cast = [
        {"id": "scientist", "role": "protagonist"},
        {"id": "guard"}, {"id": "prisoner"}, {"id": "inmate"},
    ]
    board = sb.build_storyboard(analysis, narration, {"cast": cast})
    assert len(board["scenes"]) == len(analysis["beats"])
    # every scene must be stageable
    for scene in board["scenes"]:
        assert scene["narration"], scene
        assert scene["characters"], scene
        assert scene["camera"]["move"]
        assert scene["duration"] > 0
    # timeline must be contiguous
    cursor = 0.0
    for scene in board["scenes"]:
        assert abs(scene["start"] - cursor) < 1e-2, (scene["start"], cursor)
        cursor += scene["duration"]
    print(f"  storyboard: {len(board['scenes'])} scenes, {board['duration']:.1f}s")


def test_all_actions_produce_finite_poses():
    for name in ACTIONS:
        for t in (0.0, 0.5, 2.0):
            pose = pose_for(name, t, 3.0)
            for key, value in pose.items():
                assert isinstance(value, (int, float)), (name, key, value)
                assert value == value and abs(value) < 10, (name, key, value)
    print(f"  actions: {len(ACTIONS)} poses finite")


def test_rig_draws_everything():
    spec = {"id": "t", "skin": "#e8b48a", "hair": "#3a3038", "outfit": "#f4f1ea",
            "outfitStyle": "coat", "accessory": "glasses", "headgear": "cap"}
    rig = CharacterRig(spec)
    img = Image.new("RGBA", (400, 700), (30, 30, 40, 255))
    joints = rig.draw(img, (200, 620), 520, {
        "pose": pose_for("walk", 1.1, 3.0), "facing": 1,
        "expression": "happy", "mouth": 0.5, "blink": 0.0,
    })
    # the head must sit above the hips, and the feet below them
    assert joints["head"][1] < joints["hip"][1] < joints["ankle_r"][1], joints
    # something was actually drawn
    assert img.getbbox() is not None
    print("  rig: joints ordered, pixels drawn")


def test_camera_frames_stay_vertical():
    cam = Camera(1080, 1920, 1.35)
    for move in ("static", "push_in", "pan_left", "low_angle", "dolly"):
        for p in (0.0, 0.5, 1.0):
            state = cam.state(move, p)
            box = cam.crop_box(state)
            w, h = box[2] - box[0], box[3] - box[1]
            assert abs((w / h) - (1080 / 1920)) < 0.01, (move, p, w, h)
    print("  camera: every window is 9:16")


def test_captions_and_srt():
    scenes = [{"narration": "A guard paints the walls pink and the prisoners calm down slowly.",
               "start": 0.0, "duration": 4.0}]
    cues = cues_from_scenes(scenes)
    assert cues and cues[0]["text"]
    fnt = font(40)
    lines = wrap_text("x " * 200, fnt, 900)
    assert len(lines) > 1
    with tempfile.TemporaryDirectory() as tmp:
        path = write_srt(cues, os.path.join(tmp, "a.srt"))
        body = open(path, encoding="utf-8").read()
        assert "-->" in body
    print(f"  captions: {len(cues)} cues, wrapped and exported")


def test_sound_effects_are_audible():
    for name in ("footsteps", "door_open", "punch", "ambience"):
        data = sound_effects.get(name)
        assert data.size > 1000, name
        peak = float(abs(data).max())
        assert peak > 0.05, (name, peak)
    print("  sfx: synthesised and non-silent")


def test_end_to_end_render():
    from app.project_manager import Project
    from audio_engine import audio_mixer
    from render_engine import video_exporter
    from render_engine.frame_renderer import FrameRenderer
    from render_engine.scene_renderer import SceneRenderer, ffmpeg_binary

    ffmpeg_binary()  # raises a clear error if ffmpeg is missing

    with tempfile.TemporaryDirectory() as tmp:
        project = Project(os.path.join(tmp, "proj")).ensure()
        analysis = analyzer.analyze(STORY, use_ai=False)
        narration = script_writer.write_narration(STORY, analysis["beats"], use_ai=False)
        board = sb.build_storyboard(analysis, narration, {"fps": 30, "cast": [
            {"id": "scientist", "role": "protagonist"}, {"id": "guard"}, {"id": "prisoner"},
        ]})
        # keep the test quick: 3 short scenes
        board["scenes"] = board["scenes"][:2]
        for scene in board["scenes"]:
            scene["duration"] = 0.6
        board["duration"] = sum(s["duration"] for s in board["scenes"])

        cam = Camera(1080, 1920, 1.35)
        renderer = FrameRenderer({c["id"]: c for c in board["characters"]}, cam,
                                 cues_from_scenes(board["scenes"]))
        scene_renderer = SceneRenderer(renderer, fps=30, width=1080, height=1920,
                                       crf=30, preset="veryfast", preview_scale=1.0)
        segments = []
        for scene in board["scenes"]:
            out = project.segment_path(scene["id"], 18)
            scene_renderer.render_scene(scene, out, 0.0, resume=False)
            assert os.path.exists(out) and os.path.getsize(out) > 1000
            segments.append(out)

        # silent audio track of matching length
        n = int(board["duration"] * 44100)
        audio_mixer.write_wav(project.mix_path(), __import__("numpy").zeros(n))

        info = video_exporter.export(segments, project.mix_path(),
                                     os.path.join(tmp, "out.mp4"), project.dirs["segments"])
        assert info["width"] == 1080 and info["height"] == 1920, info
        assert abs(float(info["fps"]) - 30.0) < 0.1, info
        assert info["video_codec"] == "h264", info
        assert info["duration"] > 1.0, info
        print(f"  end-to-end: {info['duration']}s {info['width']}x{info['height']} "
              f"{info['video_codec']}/{info.get('audio_codec')}")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        name = test.__name__
        try:
            print(f"[ RUN  ] {name}")
            test()
            print(f"[  OK  ] {name}")
        except Exception as exc:
            failures += 1
            import traceback
            print(f"[ FAIL ] {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
