"""Character reference extraction.

Honest scope: this is *not* a semantic image segmenter. It does two things that
actually work on CPU with no models:

1. `spec_from_reference()` samples a reference image and derives the palette
   (skin, hair, outfit, accent) and rough proportions, so a character keeps the
   look of the reference.
2. `load_part_sheets()` loads hand-cut part images (head.png, torso.png,
   arm_upper.png, ...) when the automatic pass is not good enough. That is the
   documented manual fallback the spec asks for.

Anything the extractor cannot determine is left at the default and reported in
`notes`, so nothing is silently invented.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

from PIL import Image

from render_engine.draw_utils import luminance, rgb_to_hex, shade

PART_FILES = {
    "head": ["head.png", "head.webp"],
    "torso": ["torso.png", "body.png"],
    "arm_upper": ["arm_upper.png", "upper_arm.png"],
    "arm_lower": ["arm_lower.png", "forearm.png"],
    "hand": ["hand.png"],
    "leg_upper": ["leg_upper.png", "thigh.png"],
    "leg_lower": ["leg_lower.png", "shin.png"],
    "foot": ["foot.png"],
}


def _quantised_colours(img: Image.Image, top: int = 8) -> List[Tuple[str, float]]:
    small = img.convert("RGB").resize((96, 96), Image.BILINEAR)
    counts: Counter = Counter()
    for r, g, b in small.getdata():
        # ignore near-white / near-transparent background
        if r > 244 and g > 244 and b > 244:
            continue
        counts[(r // 16 * 16, g // 16 * 16, b // 16 * 16)] += 1
    total = sum(counts.values()) or 1
    return [(rgb_to_hex(c), n / total) for c, n in counts.most_common(top)]


def spec_from_reference(char_id: str, reference: str, name: Optional[str] = None) -> dict:
    """Derive a character spec from a reference image (colour + proportion pass)."""
    notes: List[str] = []
    spec = {
        "id": char_id,
        "name": name or char_id.title(),
        "skin": "#e8b48a",
        "hair": "#2b2430",
        "hairStyle": "short",
        "outfit": "#3f6fa8",
        "outfitStyle": "plain",
        "accent": "#f0e9dc",
        "pants": "#33384a",
        "shoe": "#221c26",
        "headgear": "none",
        "accessory": "none",
        "build": 1.0,
        "height": 1.0,
        "reference": reference,
        "notes": notes,
    }

    if not reference or not os.path.exists(reference):
        notes.append("no reference image found; using default palette")
        return spec

    try:
        img = Image.open(reference)
    except Exception as exc:  # corrupted reference must not kill the run
        notes.append(f"could not read reference ({exc}); using default palette")
        return spec

    img = img.convert("RGBA")
    colours = _quantised_colours(img)
    if not colours:
        notes.append("reference had no usable pixels")
        return spec

    # skin: the most common warm, mid-light tone in the top half
    top_half = img.crop((0, 0, img.width, max(1, img.height // 2)))
    head_colours = _quantised_colours(top_half)
    skin = None
    for colour, _weight in head_colours:
        r, g, b = int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)
        if r > g > b and 0.35 < luminance(colour) < 0.92:
            skin = colour
            break
    if skin:
        spec["skin"] = skin
    else:
        notes.append("could not isolate a skin tone; kept the default")

    # hair: darkest common colour in the top quarter
    top_quarter = img.crop((0, 0, img.width, max(1, img.height // 4)))
    hair_candidates = [c for c, _ in _quantised_colours(top_quarter) if luminance(c) < 0.35]
    if hair_candidates:
        spec["hair"] = hair_candidates[0]
    else:
        notes.append("no dark hair tone found; kept the default")

    # outfit: most common colour in the middle band, ignoring skin
    mid = img.crop((0, int(img.height * 0.35), img.width, int(img.height * 0.75)))
    for colour, _weight in _quantised_colours(mid):
        if colour != spec["skin"] and luminance(colour) > 0.08:
            spec["outfit"] = colour
            spec["accent"] = shade(colour, 0.45)
            break

    # proportions from the alpha bounding box
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if h > 0:
            ratio = w / h
            if ratio < 0.34:
                spec["build"] = 0.92
            elif ratio > 0.52:
                spec["build"] = 1.15
            else:
                spec["build"] = 1.0
            notes.append(f"body aspect ratio {ratio:.2f} -> build {spec['build']}")
    else:
        notes.append("reference has no transparent background; proportions defaulted")

    notes.append("expression/pose parts are drawn by the rig; supply part PNGs for exact art")
    return spec


def load_part_sheets(char_id: str, parts_dir: str) -> Dict[str, str]:
    """Find hand-cut part images for the manual rigging path."""
    found: Dict[str, str] = {}
    if not os.path.isdir(parts_dir):
        return found
    for part, names in PART_FILES.items():
        for name in names:
            path = os.path.join(parts_dir, name)
            if os.path.exists(path):
                found[part] = path
                break
    return found


def write_specs(specs: List[dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"characters": specs}, f, indent=2, ensure_ascii=False)
    return path
