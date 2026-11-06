"""Story analysis: characters, locations, objects, beats, emotions.

Gemini is used when a key is available. Without a key a deterministic rule-based
analyzer does the same job, so the pipeline is never dependent on a paid API.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
MODELS = [os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash-lite",
          "gemini-2.5-flash-lite", "gemini-flash-lite-latest"]

# --------------------------------------------------------------------------- #
# rule-based vocabulary
# --------------------------------------------------------------------------- #

ROLE_WORDS = {
    "scientist": ("scientist", "researcher", "doctor", "chemist", "professor"),
    "guard": ("guard", "officer", "warden", "cop", "sentry"),
    "prisoner": ("prisoner", "inmate", "convict", "detainee"),
    "inmate": ("big inmate", "inmate", "thug", "bruiser"),
    "teacher": ("teacher", "tutor"),
    "student": ("student", "pupil", "kid", "child", "boy", "girl"),
    "king": ("king", "queen", "monarch", "prince", "princess"),
    "robot": ("robot", "android", "machine"),
    "captain": ("captain", "sailor", "pilot"),
    "nurse": ("nurse",),
    "engineer": ("engineer", "mechanic"),
    "farmer": ("farmer",),
}

LOCATION_WORDS = {
    "lab": ("lab", "laboratory", "experiment", "beaker", "test tube", "chemistry", "scientist", "discovers"),
    "prison_cell": ("prison", "prisoners", "prisoner", "inmate", "cell", "barred", "bunk"),
    "prison_corridor": ("corridor", "hallway", "block", "wing"),
    "prison_yard": ("yard", "courtyard", "exercise"),
    "office": ("office", "desk", "meeting"),
    "city": ("city", "street", "town", "downtown"),
    "forest": ("forest", "woods", "jungle"),
    "room": ("room", "house", "home", "kitchen", "bedroom"),
    "night_sky": ("night", "stars", "moon", "sky"),
    "plain": ("field", "plain", "desert", "meadow"),
    "wall": ("wall", "walls", "corridor wall"),
}

OBJECT_WORDS = {
    "paint_can": ("paint", "bucket", "can"),
    "brush": ("brush", "roller"),
    "vial": ("vial", "tube", "serum", "sample"),
    "clipboard": ("clipboard", "notes", "chart", "file"),
    "baton": ("baton", "stick", "club"),
    "door": ("door", "gate"),
    "box": ("box", "crate", "package"),
    "phone": ("phone", "mobile"),
    "key": ("key", "keys"),
    "cup": ("cup", "mug", "coffee"),
    "machine": ("machine", "device", "equipment"),
    "chair": ("chair", "stool"),
    "table": ("table", "desk"),
    "sign": ("sign", "notice", "poster"),
}

ACTION_WORDS = {
    "walk": ("walk", "walked", "walks", "stroll", "step"),
    "run": ("run", "ran", "rushed", "sprinted", "fled"),
    "point": ("point", "pointed", "gestured", "showed", "paints", "painted", "painting", "brushes", "spray"),
    "pick_up": ("pick", "picked", "grabbed", "lifted"),
    "drop": ("drop", "dropped", "released"),
    "open_door": ("opened", "open the door", "unlocked"),
    "close_door": ("closed", "slammed", "shut"),
    "push": ("push", "pushed", "shoved"),
    "pull": ("pull", "pulled", "dragged"),
    "fight": ("fight", "fought", "punched", "hit", "attacked", "struggled"),
    "fall": ("fell", "fall", "collapsed", "dropped to"),
    "cry": ("cry", "cried", "wept", "tears"),
    "laugh": ("laugh", "laughed", "giggle", "smiled"),
    "sit": ("sit", "sat", "seated"),
    "jump": ("jump", "jumped", "leapt"),
    "talk": ("said", "says", "explained", "told", "spoke", "asked", "tested", "tests", "test", "measures", "checks", "reports"),
    "look_around": ("looked", "look", "watched", "stared", "observed"),
}

EMOTION_WORDS = {
    "angry": ("angry", "furious", "rage", "aggressive", "hostile", "shouted", "yelled"),
    "scared": ("afraid", "scared", "terrified", "panic", "fear"),
    "happy": ("happy", "calm", "smiled", "relieved", "peaceful", "gentle"),
    "sad": ("sad", "grief", "hopeless", "quiet"),
    "surprised": ("suddenly", "surprised", "shock", "unexpected", "realised", "realized"),
    "confused": ("confused", "puzzled", "unsure", "wondered"),
    "crying": ("crying", "wept", "tears"),
    "laughing": ("laughed", "giggle"),
}


def _first_match(text: str, table: Dict[str, tuple]) -> Optional[str]:
    low = text.lower()
    best: Optional[str] = None
    best_pos = 10 ** 9
    for key, words in table.items():
        for word in words:
            pos = low.find(word)
            if pos >= 0 and pos < best_pos:
                best, best_pos = key, pos
    return best


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and len(p.split()) < 4:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# rule-based analysis
# --------------------------------------------------------------------------- #

def rule_based_analysis(text: str) -> Dict:
    sentences = split_sentences(text)
    joined = " ".join(sentences)

    # characters
    found: Dict[str, int] = {}
    for key, words in ROLE_WORDS.items():
        for word in words:
            if re.search(rf"\b{re.escape(word)}s?\b", joined.lower()):
                found[key] = found.get(key, 0) + joined.lower().count(word)

    characters: List[dict] = []
    for key in sorted(found, key=lambda k: -found[k]):
        characters.append({
            "id": key,
            "name": key.replace("_", " ").title(),
            "role": "protagonist" if not characters else "support",
            "description": f"The {key} in the story.",
        })
    if not characters:
        characters = [{"id": "narrator_character", "name": "Main Character",
                       "role": "protagonist", "description": "The main character."}]

    # locations (ordered by first mention)
    locs: List[dict] = []
    for key, words in LOCATION_WORDS.items():
        for word in words:
            pos = joined.lower().find(word)
            if pos >= 0:
                locs.append({"kind": key, "first_mention": pos})
                break
    locs.sort(key=lambda x: x["first_mention"])
    locations = [{"kind": l["kind"], "name": l["kind"].replace("_", " ").title()} for l in locs] or \
                [{"kind": "plain", "name": "Plain"}]

    # objects
    objects: List[str] = []
    for key, words in OBJECT_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}", joined.lower()) for w in words):
            objects.append(key)

    # beats
    beats: List[dict] = []
    for i, sentence in enumerate(sentences):
        action = _first_match(sentence, ACTION_WORDS) or ("talk" if i == 0 else "stand")
        emotion = _first_match(sentence, EMOTION_WORDS) or "neutral"
        location = _first_match(sentence, LOCATION_WORDS)
        present = [c["id"] for c in characters
                   if re.search(rf"\b{re.escape(c['id'])}", sentence.lower())]
        if not present:
            present = [characters[0]["id"]]
        beats.append({
            "index": i,
            "text": sentence,
            "action": action,
            "emotion": emotion,
            "location": location,
            "characters": present[:3],
            "objects": [o for o in objects if o in sentence.lower()][:2],
        })

    title = sentences[0].rstrip(".")[:70] if sentences else "Untitled Story"
    return {
        "title": title,
        "summary": " ".join(sentences[:2]),
        "characters": characters,
        "locations": locations,
        "objects": objects,
        "beats": beats,
        "source": "rule-based",
    }


# --------------------------------------------------------------------------- #
# gemini analysis
# --------------------------------------------------------------------------- #

PROMPT = """Analyse this story for a short animated vertical explainer video.

STORY:
\"\"\"{story}\"\"\"

Return ONLY JSON:
{{
  "title": "short title",
  "summary": "one sentence",
  "characters": [{{"id":"scientist","name":"Dr. Rao","role":"protagonist","description":"..."}}],
  "locations": [{{"kind":"lab","name":"Laboratory"}}],
  "objects": ["vial","clipboard"],
  "beats": [{{"text":"one sentence of the story","action":"talk","emotion":"calm",
              "location":"lab","characters":["scientist"],"objects":["vial"]}}]
}}

Allowed locations: lab, prison_cell, prison_corridor, prison_yard, office, city,
forest, room, night_sky, plain, wall.
Allowed actions: stand, walk, run, jump, sit, talk, point, wave, fight, fall, cry,
laugh, look_around, pick_up, drop, open_door, close_door, push, pull.
Allowed emotions: neutral, happy, sad, angry, surprised, scared, confused, laughing, crying.

Rules:
- one beat per story sentence, in order
- character ids are lowercase, no spaces
- never invent characters or events that are not in the story
"""


def gemini_analysis(text: str) -> Optional[Dict]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key or requests is None:
        return None
    body = {
        "contents": [{"role": "user", "parts": [{"text": PROMPT.format(story=text[:6000])}]}],
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 3000,
                             "responseMimeType": "application/json"},
    }
    for model in MODELS:
        try:
            resp = requests.post(f"{API_ROOT}/{model}:generateContent", params={"key": key},
                                 json=body, timeout=120)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        try:
            data = resp.json()
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
            parsed = json.loads(raw)
            parsed["source"] = f"gemini:{model}"
            if parsed.get("beats") and parsed.get("characters"):
                return parsed
        except Exception:
            continue
    return None


def analyze(text: str, use_ai: bool = True) -> Dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("story text is empty")
    if use_ai:
        result = gemini_analysis(text)
        if result:
            return result
    return rule_based_analysis(text)


def save(analysis: Dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    return path
