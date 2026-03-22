"""
Stats viewer: computes KPIs and renders the graphical utilisation chart + text summary.
"""
from __future__ import annotations

import base64
import io
import tkinter as tk
from typing import TYPE_CHECKING, List

from ...core.io_utils import infer_dimensions_from_matrix
from ..visualization import build_stats_figure

if TYPE_CHECKING:
    from ..gui import SchedulerGUI
    from .models import Scenario


def render_stats(app: "SchedulerGUI", scenario: "Scenario") -> None:
    """Populate the stats tab for the given scenario."""
    if not scenario.schedule or scenario.optimal_makespan is None:
        app.stats_text.configure(state="normal")
        app.stats_text.delete("1.0", tk.END)
        app.stats_text.insert("1.0", "Run the scenario to see KPIs.")
        app.stats_text.configure(state="disabled")
        if app.stats_graphics_label:
            app.stats_graphics_label.configure(image="")
        return

    makespan = float(scenario.optimal_makespan)
    num_jobs, num_resources = infer_dimensions_from_matrix(app.matrix)

    # ── Aggregate per-resource busy times and per-job total processing times ──
    job_runtime = {j: 0.0 for j in range(num_jobs)}
    resource_busy = {r: 0.0 for r in range(num_resources)}
    for r, j, start, duration in scenario.schedule:
        job_runtime[j] += float(duration)
        resource_busy[r] += float(duration)

    # ── Idle time: capacity − busy time ──
    resource_idle = {}
    for r in range(num_resources):
        units = app.availability_vector[r] if r < len(app.availability_vector) else 1
        capacity = makespan * float(units)
        resource_idle[r] = max(0.0, capacity - resource_busy.get(r, 0.0))

    # ── Render utilisation chart ──
    _render_chart(app, num_resources, resource_busy, resource_idle, makespan)

    # ── Time matrix: demand × resource_time ──
    time_matrix: List[List[float]] = [
        [float(app.matrix[i][r]) * float(app.resource_times[r]) for r in range(num_resources)]
        for i in range(num_jobs)
    ]

    # ── Build text KPIs ──
    out: List[str] = []
    out.append("─" * 50)
    out.append(f"  MAKESPAN: {makespan}")
    out.append("─" * 50)
    out.append("")
    out.append("Job total processing time")
    for j in range(num_jobs):
        out.append(f"  J{j + 1}:  {job_runtime[j]:.2f}")
    out.append("")
    out.append("Resource utilization")
    for r in range(num_resources):
        units = app.availability_vector[r]
        cap = makespan * float(units)
        busy = resource_busy.get(r, 0.0)
        util = (busy / cap) if cap > 0 else 0.0
        out.append(
            f"  R{r + 1}:  busy={busy:.2f}  idle={resource_idle[r]:.2f}  "
            f"capacity={cap:.2f}  util={util:.1%}"
        )
    out.append("")
    out.append("Time matrix  (demand × exec_time)")
    header = "      " + "  ".join(f"R{r + 1:>2}" for r in range(num_resources))
    out.append(header)
    for i in range(num_jobs):
        out.append("J{:<3} {}".format(i + 1, "  ".join(f"{v:>4.1f}" for v in time_matrix[i])))

    app.stats_text.configure(state="normal")
    app.stats_text.delete("1.0", tk.END)
    app.stats_text.insert("1.0", "\n".join(out))
    app.stats_text.configure(state="disabled")


def _render_chart(
    app: "SchedulerGUI",
    num_resources: int,
    resource_busy: dict,
    resource_idle: dict,
    makespan: float,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = build_stats_figure(
        num_resources=num_resources,
        resource_busy=resource_busy,
        resource_idle=resource_idle,
        title=f"Resource Utilization  —  Makespan = {makespan}",
    )
    buf = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buf)
    buf.seek(0)
    photo = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
    app._stats_image_ref = photo  # keep a reference to avoid GC

    if app.stats_graphics_label is None:
        app.stats_graphics_label = tk.Label(app.stats_graphics_frame, image=photo)
        app.stats_graphics_label.grid(row=0, column=0, sticky="nsew")
    else:
        app.stats_graphics_label.configure(image=photo)
