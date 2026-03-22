"""
Result viewer: renders the animated logs and Gantt chart for a scenario.
"""
from __future__ import annotations

import base64
import io
import tkinter as tk
from typing import TYPE_CHECKING, Optional

from ..visualization import build_gantt_figure

if TYPE_CHECKING:
    from ..gui import SchedulerGUI
    from .models import Scenario

# Tag names used in the log Text widget
_HEADER_TAG = "header"
_PHASE_TAG = "phase"
_OK_TAG = "ok"
_WARN_TAG = "warn"
_ERR_TAG = "err"
_MONO_TAG = "mono"


def configure_log_tags(text_widget: tk.Text) -> None:
    """Apply colour/font tags to a log Text widget."""
    text_widget.tag_configure(_HEADER_TAG, foreground="#2563eb", font=("TkDefaultFont", 12, "bold"))
    text_widget.tag_configure(_PHASE_TAG, foreground="#7c3aed", font=("TkDefaultFont", 12, "bold"))
    text_widget.tag_configure(_OK_TAG, foreground="#16a34a")
    text_widget.tag_configure(_WARN_TAG, foreground="#b45309")
    text_widget.tag_configure(_ERR_TAG, foreground="#dc2626")
    text_widget.tag_configure(_MONO_TAG, font=("TkFixedFont", 12))


def _pick_tag(line: str) -> Optional[str]:
    """Return the appropriate tag name for a log line, or None."""
    if line.startswith("SCENARIO ") or line.startswith("SCENARIO:"):
        return _HEADER_TAG
    if line.startswith("PHASE ") or line.startswith("Processing "):
        return _PHASE_TAG
    if "✓" in line:
        return _OK_TAG
    if "❌" in line:
        return _ERR_TAG
    if "Warning" in line:
        return _WARN_TAG
    if line.startswith("Resource") or line.startswith("-") or line.startswith("="):
        return _MONO_TAG
    return None


def render_logs_animated(app: "SchedulerGUI", text: str) -> None:
    """Render log text into app.log_text with a gentle line-by-line animation."""
    if getattr(app, "_log_after_id", None) is not None:
        try:
            app.after_cancel(app._log_after_id)
        except Exception:
            pass
        app._log_after_id = None

    lines = text.splitlines()
    app.log_text.configure(state="normal")
    app.log_text.delete("1.0", tk.END)
    app.log_text.configure(state="disabled")

    def insert_line(i: int):
        if i >= len(lines):
            app._log_after_id = None
            return

        line = lines[i]
        tag = _pick_tag(line)

        app.log_text.configure(state="normal")
        if tag:
            app.log_text.insert(tk.END, line + "\n", (tag,))
        else:
            app.log_text.insert(tk.END, line + "\n")
        app.log_text.see(tk.END)
        app.log_text.configure(state="disabled")

        delay_ms = 10
        if line.startswith("PHASE ") or line.startswith("Processing ") or line.startswith("FINAL MAKESPAN"):
            delay_ms = 200
        app._log_after_id = app.after(delay_ms, lambda: insert_line(i + 1))

    insert_line(0)


def render_diagram(app: "SchedulerGUI", scenario: "Scenario") -> None:
    """Render the Gantt chart for *scenario* into the diagram frame."""
    fig = build_gantt_figure(
        scenario.schedule or [],
        title=f"{scenario.name} - Makespan={scenario.optimal_makespan}",
    )
    buf = io.BytesIO()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    buf.seek(0)
    photo = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
    app.diagram_image = photo

    if app.diagram_label is None:
        app.diagram_label = tk.Label(app.diagram_frame, image=photo)
        app.diagram_label.grid(row=0, column=0, sticky="nsew")
    else:
        app.diagram_label.configure(image=photo)
