"""
Matrix editor dialogs: create a new blank matrix or edit the current one.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, List, Optional

from ...core.io_utils import SchedulerConfig, infer_dimensions_from_matrix

if TYPE_CHECKING:
    from ..gui import SchedulerGUI


def open_new_matrix_dialog(app: "SchedulerGUI") -> None:
    """Open a dialog to choose matrix dimensions, then open the value editor."""
    dialog = tk.Toplevel(app)
    dialog.title("Configure matrix dimensions")

    ttk.Label(dialog, text="Jobs:").grid(row=0, column=0, padx=4, pady=4)
    ttk.Label(dialog, text="Resources:").grid(row=1, column=0, padx=4, pady=4)

    jobs_var = tk.IntVar(value=3)
    res_var = tk.IntVar(value=3)

    ttk.Spinbox(dialog, from_=1, to=20, textvariable=jobs_var, width=5).grid(
        row=0, column=1, padx=4, pady=4
    )
    ttk.Spinbox(dialog, from_=1, to=20, textvariable=res_var, width=5).grid(
        row=1, column=1, padx=4, pady=4
    )

    def confirm():
        dialog.destroy()
        _open_value_editor(app, jobs_var.get(), res_var.get())

    ttk.Button(dialog, text="Next", command=confirm).grid(
        row=2, column=0, columnspan=2, pady=8
    )


def open_edit_matrix_dialog(app: "SchedulerGUI") -> None:
    """Edit the currently loaded matrix in-place."""
    if not app.matrix:
        messagebox.showinfo("No configuration", "Load or create a configuration first.")
        return
    num_jobs, num_resources = infer_dimensions_from_matrix(app.matrix)
    _open_value_editor(app, num_jobs, num_resources, prefill=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_value_editor(
    app: "SchedulerGUI",
    num_jobs: int,
    num_resources: int,
    *,
    prefill: bool = False,
) -> None:
    """Generic editor that renders a grid of entries for the matrix + vectors."""
    editor = tk.Toplevel(app)
    editor.title("Edit matrix and vectors")

    entries: List[List[ttk.Entry]] = []
    for i in range(num_jobs):
        row_entries: List[ttk.Entry] = []
        for j in range(num_resources):
            e = ttk.Entry(editor, width=5)
            e.grid(row=i, column=j, padx=2, pady=2)
            default = str(app.matrix[i][j]) if prefill and i < len(app.matrix) and j < len(app.matrix[i]) else "1"
            e.insert(0, default)
            row_entries.append(e)
        entries.append(row_entries)

    ttk.Label(editor, text="Resource times (space-separated):").grid(
        row=num_jobs, column=0, columnspan=num_resources, sticky="w", pady=(8, 2)
    )
    times_entry = ttk.Entry(editor, width=40)
    times_entry.grid(row=num_jobs + 1, column=0, columnspan=num_resources, pady=2)
    if prefill and app.resource_times:
        times_entry.insert(0, " ".join(str(x) for x in app.resource_times))
    else:
        times_entry.insert(0, "3 5 4" if num_resources == 3 else " ".join(["1"] * num_resources))

    ttk.Label(editor, text="Availability vector (space-separated):").grid(
        row=num_jobs + 2, column=0, columnspan=num_resources, sticky="w", pady=(8, 2)
    )
    avail_entry = ttk.Entry(editor, width=40)
    avail_entry.grid(row=num_jobs + 3, column=0, columnspan=num_resources, pady=2)
    if prefill and app.availability_vector:
        avail_entry.insert(0, " ".join(str(x) for x in app.availability_vector))
    else:
        avail_entry.insert(0, "1 " * num_resources)

    def confirm():
        try:
            matrix = [
                [int(entries[i][j].get()) for j in range(num_resources)]
                for i in range(num_jobs)
            ]
            resource_times = [float(x) for x in times_entry.get().split()]
            availability_vector = [int(x) for x in avail_entry.get().split()]
        except ValueError:
            messagebox.showerror("Error", "Invalid integer/float value. Check all fields.")
            return

        if len(resource_times) != num_resources:
            messagebox.showerror(
                "Error",
                f"Resource times must have {num_resources} values, got {len(resource_times)}.",
            )
            return
        if any(t <= 0 for t in resource_times):
            messagebox.showerror("Error", "All resource execution times must be > 0.")
            return
        if len(availability_vector) != num_resources:
            messagebox.showerror(
                "Error",
                f"Availability vector must have {num_resources} values, got {len(availability_vector)}.",
            )
            return

        config = SchedulerConfig(
            matrix=matrix,
            resource_times=resource_times,
            availability_vector=availability_vector,
        )
        app._apply_config(config, rerun_scenarios=True)
        editor.destroy()

    ttk.Button(editor, text="Save", command=confirm).grid(
        row=num_jobs + 4, column=0, columnspan=num_resources, pady=8
    )
