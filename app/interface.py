"""Beginner-friendly desktop interface (Tkinter — ships with Python).

    python -m app.interface

Everything it does is a wrapper around `python -m app.main`, so the CLI and the
GUI can never drift apart. Rendering runs in a subprocess, which is also what
makes Pause / Resume / Cancel safe: pause stops feeding frames, resume reuses the
scene checkpoints, cancel kills the child process.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from typing import Optional

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover - headless machines
    raise SystemExit(
        "Tkinter is not available in this Python build.\n"
        "On Debian/Ubuntu: sudo apt-get install python3-tk\n"
        f"({exc})"
    )

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STYLES = ["stylized vector (default)", "flat", "storybook"]
VOICES = ["Kore", "Puck", "Charon", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Animation Agent — story to vertical video")
        self.geometry("980x760")
        self.minsize(900, 700)

        self.process: Optional[subprocess.Popen] = None
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.paused = False
        self.project = "projects/pink_paint"

        self._build()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}

        head = ttk.Frame(self)
        head.pack(fill="x", **pad)
        ttk.Label(head, text="Story", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self.story = tk.Text(head, height=8, wrap="word")
        self.story.pack(fill="x", pady=4)
        self.story.insert("1.0", self._default_story())

        row = ttk.Frame(self)
        row.pack(fill="x", **pad)
        ttk.Button(row, text="Load story file…", command=self.load_story).pack(side="left")
        ttk.Button(row, text="Load project…", command=self.load_project).pack(side="left", padx=6)
        ttk.Button(row, text="Upload characters…", command=lambda: self.upload("characters")).pack(side="left", padx=6)
        ttk.Button(row, text="Upload background…", command=lambda: self.upload("backgrounds")).pack(side="left", padx=6)

        opts = ttk.LabelFrame(self, text="Settings")
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="Visual style").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.style = ttk.Combobox(opts, values=STYLES, state="readonly", width=24)
        self.style.current(0)
        self.style.grid(row=0, column=1, sticky="w")

        ttk.Label(opts, text="Voice").grid(row=0, column=2, sticky="w", padx=6)
        self.voice = ttk.Combobox(opts, values=VOICES, state="readonly", width=14)
        self.voice.current(0)
        self.voice.grid(row=0, column=3, sticky="w")

        ttk.Label(opts, text="Target length (s)").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.duration = tk.Spinbox(opts, from_=10, to=120, increment=5, width=6)
        self.duration.delete(0, "end")
        self.duration.insert(0, "30")
        self.duration.grid(row=1, column=1, sticky="w")

        self.preview = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Fast preview (half resolution)", variable=self.preview).grid(
            row=1, column=2, columnspan=2, sticky="w", padx=6)

        actions = ttk.Frame(self)
        actions.pack(fill="x", **pad)
        self.btn_generate = ttk.Button(actions, text="Generate video", command=self.generate)
        self.btn_generate.pack(side="left")
        self.btn_pause = ttk.Button(actions, text="Pause", command=self.toggle_pause, state="disabled")
        self.btn_pause.pack(side="left", padx=6)
        self.btn_cancel = ttk.Button(actions, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left")
        self.btn_open = ttk.Button(actions, text="Open output folder", command=self.open_output)
        self.btn_open.pack(side="right")

        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill="x", **pad)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=10)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, **pad)
        ttk.Label(body, text="Storyboard preview").pack(anchor="w")
        self.board = tk.Text(body, height=10, wrap="none")
        self.board.pack(fill="both", expand=True)

        self.after(200, self.drain)

    # ------------------------------------------------------------------ #
    def _default_story(self) -> str:
        path = os.path.join(ROOT, "projects", "pink_paint", "story.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "A scientist discovers a pink paint that temporarily calms aggressive prisoners."

    def load_story(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if path:
            with open(path, "r", encoding="utf-8") as f:
                self.story.delete("1.0", "end")
                self.story.insert("1.0", f.read())

    def load_project(self) -> None:
        path = filedialog.askdirectory(initialdir=os.path.join(ROOT, "projects"))
        if path:
            self.project = os.path.relpath(path, ROOT)
            self.status.set(f"Project: {self.project}")

    def upload(self, kind: str) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if not path:
            return
        target_dir = os.path.join(ROOT, "assets", kind)
        os.makedirs(target_dir, exist_ok=True)
        name = os.path.basename(path)
        target = os.path.join(target_dir, name)
        with open(path, "rb") as src, open(target, "wb") as dst:
            dst.write(src.read())
        self.status.set(f"Copied {name} → assets/{kind}/")

    # ------------------------------------------------------------------ #
    def generate(self) -> None:
        if self.process:
            return
        story = self.story.get("1.0", "end").strip()
        if not story:
            messagebox.showwarning("No story", "Write or load a story first.")
            return
        story_path = os.path.join(ROOT, self.project, "story.txt")
        os.makedirs(os.path.dirname(story_path), exist_ok=True)
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(story + "\n")

        cmd = [sys.executable, "-m", "app.main", "--project", self.project,
               "--resume", "--voice", self.voice.get()]
        if self.preview.get():
            cmd.append("--preview")

        self.board.delete("1.0", "end")
        self.progress["value"] = 0
        self.status.set("Starting…")
        self.btn_generate.configure(state="disabled")
        self.btn_pause.configure(state="normal")
        self.btn_cancel.configure(state="normal")

        env = dict(os.environ)
        self.process = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self.process
        for line in self.process.stdout:  # type: ignore[union-attr]
            self.queue.put(line.rstrip())
        self.process.wait()
        self.queue.put(f"__EXIT__{self.process.returncode}")

    def drain(self) -> None:
        try:
            while True:
                line = self.queue.get_nowait()
                if line.startswith("__EXIT__"):
                    code = int(line.replace("__EXIT__", ""))
                    self.process = None
                    self.btn_generate.configure(state="normal")
                    self.btn_pause.configure(state="disabled")
                    self.btn_cancel.configure(state="disabled")
                    self.progress["value"] = 100 if code == 0 else self.progress["value"]
                    self.status.set("Done." if code == 0 else f"Finished with errors (code {code}).")
                else:
                    self.board.insert("end", line + "\n")
                    self.board.see("end")
                    if "%" in line and "(" in line:
                        try:
                            pct = float(line.split("(")[1].split("%")[0])
                            self.progress["value"] = pct
                        except Exception:
                            pass
                    if line.startswith("[") and "s0" in line:
                        self.status.set(line[:120])
        except queue.Empty:
            pass
        self.after(200, self.drain)

    # ------------------------------------------------------------------ #
    def toggle_pause(self) -> None:
        if not self.process:
            return
        # SIGSTOP/SIGCONT pause the child without losing checkpoints
        try:
            if not self.paused:
                self.process.send_signal(19)  # SIGSTOP
                self.paused = True
                self.btn_pause.configure(text="Resume")
                self.status.set("Paused.")
            else:
                self.process.send_signal(18)  # SIGCONT
                self.paused = False
                self.btn_pause.configure(text="Pause")
                self.status.set("Rendering…")
        except Exception as exc:
            self.status.set(f"Cannot pause on this platform: {exc}")

    def cancel(self) -> None:
        if self.process:
            self.process.terminate()
            self.status.set("Cancelling… (finished scenes are kept as checkpoints)")

    def open_output(self) -> None:
        path = os.path.join(ROOT, "output")
        os.makedirs(path, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])


def main() -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
