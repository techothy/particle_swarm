"""
PSO-TF-IDF desktop app — run the full benchmark and view results.

Launch (dev):
    python gui/app.py

Package as .exe (Windows):
    .\\scripts\\build_exe.ps1
"""

from __future__ import annotations

import queue
import subprocess
import sys
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR.parent))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import customtkinter as ctk

    USE_CTK = True
except ImportError:
    USE_CTK = False

from PIL import Image, ImageTk

from gui.layman_report import (
    CHART_CAPTIONS,
    build_layman_figures,
    format_at_a_glance,
    format_compare_methods,
    format_log_friendly,
    format_what_it_means,
)
from pso_tfidf.benchmark import run_benchmark
from pso_tfidf.config import load_config
from pso_tfidf.types import BenchmarkResult

CONFIGS = {
    "Quick demo (~1–3 min)": ROOT / "configs" / "fast.yaml",
    "Full benchmark (~15–45 min)": ROOT / "configs" / "default.yaml",
}

FONT_UI = ("Segoe UI", 13) if sys.platform == "win32" else ("Helvetica", 13)
FONT_UI_SM = ("Segoe UI", 12) if sys.platform == "win32" else ("Helvetica", 12)
FONT_MONO = ("Consolas", 11) if sys.platform == "win32" else ("Courier", 11)


class PsoTfidfApp:
    def __init__(self) -> None:
        if USE_CTK:
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
            self.root = ctk.CTk()
        else:
            self.root = tk.Tk()

        self.root.title("PSO-TF-IDF — Text tuning benchmark")
        self.root.geometry("1040x780")
        self.root.minsize(900, 640)

        self.msg_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self._image_refs: list = []
        self._technical_log: list[str] = []

        self._build_ui()
        self._poll_queue()

    def _build_ui(self) -> None:
        if USE_CTK:
            self._build_ctk()
        else:
            self._build_ttk()

    def _make_text(self, parent, mono: bool = False):
        font = ctk.CTkFont(family="Consolas" if mono else "Segoe UI", size=12 if mono else 13)
        return ctk.CTkTextbox(parent, font=font, wrap="word")

    def _build_ctk(self) -> None:
        ctk.CTkLabel(
            self.root,
            text="Benchmark toolset for PSO driven TF-IDF optimization approach",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(pady=(14, 2))
        ctk.CTkLabel(
            self.root,
            text="Finding salient word-classification approach for Natural Language Processing",
            font=ctk.CTkFont(size=13),
            text_color="gray70",
        ).pack(pady=(0, 10))

        bar = ctk.CTkFrame(self.root)
        bar.pack(fill="x", padx=18, pady=6)
        ctk.CTkLabel(bar, text="Profile:").pack(side="left", padx=(12, 6), pady=10)
        self.profile_var = tk.StringVar(value=list(CONFIGS.keys())[0])
        ctk.CTkOptionMenu(bar, variable=self.profile_var, values=list(CONFIGS.keys()), width=260).pack(
            side="left", padx=6
        )
        self.run_btn = ctk.CTkButton(bar, text="Run benchmark", command=self._start_run, width=140)
        self.run_btn.pack(side="left", padx=10)
        self.open_btn = ctk.CTkButton(
            bar, text="Open results folder", command=self._open_results, width=150, state="disabled"
        )
        self.open_btn.pack(side="left", padx=6)

        self.progress = ctk.CTkProgressBar(self.root, mode="indeterminate")
        self.progress.pack(fill="x", padx=18, pady=4)
        self.status = ctk.CTkLabel(self.root, text="Ready — choose a profile and press Run.", anchor="w")
        self.status.pack(fill="x", padx=20, pady=2)

        body = ctk.CTkTabview(self.root)
        body.pack(fill="both", expand=True, padx=14, pady=10)
        self.tab_glance = body.add("At a glance")
        self.tab_compare = body.add("Compare methods")
        self.tab_means = body.add("What it means")
        self.tab_charts = body.add("Charts")
        self.tab_log = body.add("Activity log")

        self.glance_text = self._make_text(self.tab_glance)
        self.glance_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.compare_text = self._make_text(self.tab_compare, mono=True)
        self.compare_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.means_text = self._make_text(self.tab_means)
        self.means_text.pack(fill="both", expand=True, padx=8, pady=8)
        self.chart_frame = ctk.CTkScrollableFrame(self.tab_charts)
        self.chart_frame.pack(fill="both", expand=True)

        log_split = ctk.CTkFrame(self.tab_log)
        log_split.pack(fill="both", expand=True, padx=6, pady=6)
        ctk.CTkLabel(log_split, text="Plain-language progress", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(4, 0)
        )
        self.log_friendly = self._make_text(log_split)
        self.log_friendly.pack(fill="both", expand=True, padx=8, pady=4)
        ctk.CTkLabel(log_split, text="Technical details", font=ctk.CTkFont(size=11), text_color="gray60").pack(
            anchor="w", padx=8
        )
        self.log_technical = self._make_text(log_split, mono=True)
        self.log_technical.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_technical.configure(height=120)

    def _build_ttk(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        ttk.Label(self.root, text="PSO-TF-IDF Benchmark", font=("Segoe UI", 16, "bold")).pack(pady=(12, 4))
        ttk.Label(
            self.root,
            text="Results explained in plain language.",
            font=("Segoe UI", 10),
        ).pack()

        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=14, pady=6)
        ttk.Label(bar, text="Profile:").pack(side="left", padx=6)
        self.profile_var = tk.StringVar(value=list(CONFIGS.keys())[0])
        ttk.Combobox(bar, textvariable=self.profile_var, values=list(CONFIGS.keys()), state="readonly", width=34).pack(
            side="left"
        )
        self.run_btn = ttk.Button(bar, text="Run benchmark", command=self._start_run)
        self.run_btn.pack(side="left", padx=8)
        self.open_btn = ttk.Button(bar, text="Open results folder", command=self._open_results, state="disabled")
        self.open_btn.pack(side="left")

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill="x", padx=14, pady=4)
        self.status = ttk.Label(self.root, text="Ready.")
        self.status.pack(fill="x", padx=16)

        body = ttk.Notebook(self.root)
        body.pack(fill="both", expand=True, padx=10, pady=8)
        self.tab_glance = ttk.Frame(body)
        self.tab_compare = ttk.Frame(body)
        self.tab_means = ttk.Frame(body)
        self.tab_charts = ttk.Frame(body)
        self.tab_log = ttk.Frame(body)
        body.add(self.tab_glance, text="At a glance")
        body.add(self.tab_compare, text="Compare methods")
        body.add(self.tab_means, text="What it means")
        body.add(self.tab_charts, text="Charts")
        body.add(self.tab_log, text="Activity log")

        def _text(parent):
            t = tk.Text(parent, font=FONT_UI, wrap="word", padx=8, pady=8)
            t.pack(fill="both", expand=True)
            return t

        self.glance_text = _text(self.tab_glance)
        self.compare_text = tk.Text(self.tab_compare, font=FONT_MONO, wrap="word", padx=8, pady=8)
        self.compare_text.pack(fill="both", expand=True)
        self.means_text = _text(self.tab_means)
        self.chart_inner = ttk.Frame(self.tab_charts)
        self.chart_inner.pack(fill="both", expand=True)
        self.log_friendly = _text(self.tab_log)
        self.log_technical = tk.Text(self.tab_log, font=FONT_MONO, height=8, wrap="word")
        self.log_technical.pack(fill="x", padx=8, pady=4)

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        if USE_CTK:
            self.run_btn.configure(state=state)
            if running:
                self.progress.start()
            else:
                self.progress.stop()
        else:
            self.run_btn.configure(state=state)
            if running:
                self.progress.start(12)
            else:
                self.progress.stop()

    def _write(self, widget, text: str) -> None:
        if USE_CTK:
            widget.delete("1.0", "end")
            widget.insert("end", text)
        else:
            widget.delete("1.0", "end")
            widget.insert("end", text)

    def _append_log(self, line: str) -> None:
        self._technical_log.append(line)
        if USE_CTK:
            self.log_technical.insert("end", line + "\n")
            self.log_technical.see("end")
            self.status.configure(text=line[:100])
        else:
            self.log_technical.insert("end", line + "\n")
            self.log_technical.see("end")
            self.status.config(text=line[:100])

    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self._set_running(True)
        self._technical_log.clear()
        for w in (
            self.glance_text,
            self.compare_text,
            self.means_text,
            self.log_friendly,
            self.log_technical,
        ):
            self._write(w, "")
        self._clear_charts()
        self._append_log("Starting benchmark…")

        config_path = CONFIGS[self.profile_var.get()]

        def task() -> None:
            try:
                import os

                os.chdir(ROOT)
                cfg = load_config(config_path)
                cfg.results_dir = ROOT / "results"

                def on_progress(msg: str) -> None:
                    self.msg_queue.put(("log", msg))

                result = run_benchmark(config=cfg, on_progress=on_progress)
                self.msg_queue.put(("done", result))
            except Exception as exc:
                self.msg_queue.put(("error", exc))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    self._show_results(payload)
                    self._set_running(False)
                elif kind == "error":
                    self._set_running(False)
                    messagebox.showerror("Benchmark failed", str(payload))
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)

    def _show_results(self, result: BenchmarkResult) -> None:
        figures_dir = result.results_dir / "figures"
        layman_figs = build_layman_figures(result, figures_dir)
        all_figures = {**layman_figs, **result.figures}

        self._write(self.glance_text, format_at_a_glance(result))
        self._write(self.compare_text, format_compare_methods(result))
        self._write(self.means_text, format_what_it_means(result))
        self._write(self.log_friendly, format_log_friendly(self._technical_log))
        self._render_charts(all_figures, layman_first=True)
        self._results_dir = result.results_dir
        self.open_btn.configure(state="normal")
        self._append_log("Done.")

    def _clear_charts(self) -> None:
        self._image_refs.clear()
        parent = self.chart_frame if USE_CTK else self.chart_inner
        for w in parent.winfo_children():
            w.destroy()

    def _render_charts(self, figures: dict[str, Path], layman_first: bool = True) -> None:
        order = [
            "comparison_layman",
            "improvement",
            "convergence_layman",
            "comparison",
            "convergence",
        ]
        keys = [k for k in order if k in figures]
        if not layman_first:
            keys = list(figures.keys())
        parent = self.chart_frame if USE_CTK else self.chart_inner

        for key in keys:
            path = figures.get(key)
            if not path or not path.exists():
                continue
            caps = CHART_CAPTIONS.get(key, (key.replace("_", " ").title(), ""))
            title, caption = caps if isinstance(caps, tuple) else (str(caps), "")

            if USE_CTK:
                ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(size=15, weight="bold")).pack(
                    anchor="w", padx=12, pady=(14, 2)
                )
                if caption:
                    ctk.CTkLabel(
                        parent, text=caption, font=ctk.CTkFont(size=12), text_color="gray75", wraplength=900
                    ).pack(anchor="w", padx=12, pady=(0, 6))
            else:
                ttk.Label(parent, text=title, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(12, 0))
                if caption:
                    ttk.Label(parent, text=caption, font=("Segoe UI", 10), wraplength=880).pack(
                        anchor="w", padx=10, pady=(0, 4)
                    )

            img = Image.open(path)
            w, h = img.size
            scale = min(900 / w, 380 / h, 1.0)
            size = (int(w * scale), int(h * scale))
            img = img.resize(size, Image.Resampling.LANCZOS)

            if USE_CTK:
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
                self._image_refs.append(ctk_img)
                ctk.CTkLabel(parent, image=ctk_img, text="").pack(pady=6)
            else:
                photo = ImageTk.PhotoImage(img)
                self._image_refs.append(photo)
                ttk.Label(parent, image=photo).pack(pady=6)

    def _open_results(self) -> None:
        folder = getattr(self, "_results_dir", ROOT / "results")
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(folder.resolve())])

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    PsoTfidfApp().run()


if __name__ == "__main__":
    main()
