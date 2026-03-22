"""
Scenario creation and editing dialogs.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Dict, List, Optional

from ...core.scheduler import generate_random_job_paths
from ...core.io_utils import infer_dimensions_from_matrix
from .models import Scenario

if TYPE_CHECKING:
    from ..gui import SchedulerGUI


def open_create_scenarios_dialog(app: "SchedulerGUI") -> None:
    """Dialog to create N new scenarios (random or custom paths)."""
    if not app.scheduler:
        messagebox.showwarning("No config", "Please configure matrix and vectors first.")
        return

    num_jobs, num_resources = infer_dimensions_from_matrix(app.matrix)
    count = app.num_scenarios_var.get()
    existing_count = len(app.scenarios)
    created_count = 0

    dialog = tk.Toplevel(app)
    dialog.title("Scenario types")

    ttk.Label(dialog, text="For each scenario choose random or custom paths.").grid(
        row=0, column=0, columnspan=3, padx=4, pady=4
    )

    def add_scenario(kind: str, index: int):
        nonlocal created_count
        logical_index = existing_count + index
        name = f"Scenario {logical_index + 1}"
        if kind == "random":
            job_paths = generate_random_job_paths(num_jobs, num_resources, app.matrix)
            scenario = Scenario(name + " (random)", job_paths)
            app.scenarios.append(scenario)
            app._insert_scenario_in_list(scenario)
        else:
            open_custom_path_editor(app, logical_index)
        created_count += 1
        if created_count >= count:
            dialog.destroy()

    for i in range(count):
        ttk.Label(dialog, text=f"Scenario {i + 1}:").grid(row=i + 1, column=0, padx=4, pady=2)
        ttk.Button(
            dialog,
            text="Random",
            command=lambda idx=i: add_scenario("random", idx),
        ).grid(row=i + 1, column=1, padx=4, pady=2)
        ttk.Button(
            dialog,
            text="Custom (tree)",
            command=lambda idx=i: add_scenario("custom", idx),
        ).grid(row=i + 1, column=2, padx=4, pady=2)

    ttk.Button(dialog, text="Close", command=dialog.destroy).grid(
        row=count + 2, column=0, columnspan=3, pady=8
    )


def open_custom_path_editor(
    app: "SchedulerGUI",
    scenario_index: int,
    existing_scenario: Optional[Scenario] = None,
) -> None:
    """Tree-based editor for defining per-job resource routing paths."""
    if not app.scheduler:
        return
    num_jobs, num_resources = infer_dimensions_from_matrix(app.matrix)

    editor = tk.Toplevel(app)
    editor.title(f"Custom job ordering - Scenario {scenario_index + 1}")
    editor.geometry("600x400")

    tree = ttk.Treeview(editor, columns=("path",), show="tree headings")
    tree.heading("#0", text="Job")
    tree.heading("path", text="Resource sequence")
    tree.column("path", width=300)
    tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    job_nodes: Dict[int, str] = {}
    for j in range(num_jobs):
        if existing_scenario:
            existing_path = existing_scenario.job_paths.get(j, [])
            path_str = " → ".join(f"R{r + 1}" for r in existing_path) if existing_path else "No path"
        else:
            path_str = "No path"
        node = tree.insert("", "end", text=f"J{j + 1}", values=(path_str,))
        job_nodes[j] = node

    def edit_job_path(job_idx: int):
        path_dialog = tk.Toplevel(editor)
        path_dialog.title(f"Path for J{job_idx + 1}")
        ttk.Label(
            path_dialog,
            text=f"Enter sequence of resource indices (1..{num_resources}), e.g. '3 1 2'",
        ).grid(row=0, column=0, padx=4, pady=4)
        entry = ttk.Entry(path_dialog, width=30)
        entry.grid(row=1, column=0, padx=4, pady=4)
        if existing_scenario:
            existing_path = existing_scenario.job_paths.get(job_idx, [])
            if existing_path:
                entry.insert(0, " ".join(str(r + 1) for r in existing_path))

        def confirm():
            try:
                path = [int(x) - 1 for x in entry.get().split()]
            except ValueError:
                messagebox.showerror("Error", "Invalid path format.")
                return
            if not all(0 <= r < num_resources for r in path):
                messagebox.showerror(
                    "Error", f"Resources must be between 1 and {num_resources}."
                )
                return
            if len(path) != len(set(path)):
                messagebox.showerror("Error", "Duplicate resources in path.")
                return
            tree.item(
                job_nodes[job_idx],
                values=(" → ".join(f"R{r + 1}" for r in path),),
            )
            path_dialog.destroy()

        ttk.Button(path_dialog, text="OK", command=confirm).grid(row=2, column=0, pady=6)

    def on_double_click(event):
        item = tree.selection()
        if not item:
            return
        node = item[0]
        for j, n in job_nodes.items():
            if n == node:
                edit_job_path(j)
                break

    tree.bind("<Double-1>", on_double_click)

    def confirm_scenario():
        job_paths: Dict[int, List[int]] = {}
        for j in range(num_jobs):
            values = tree.item(job_nodes[j])["values"]
            if not values or values[0] == "No path":
                messagebox.showerror("Error", f"Please define a path for job J{j + 1}.")
                return
            tokens = values[0].replace("R", "").split("→")
            path = [int(t.strip()) - 1 for t in tokens if t.strip()]
            job_paths[j] = path

        if existing_scenario is None:
            scenario = Scenario(f"Scenario {scenario_index + 1} (custom)", job_paths)
            app.scenarios.append(scenario)
            app._insert_scenario_in_list(scenario)
        else:
            existing_scenario.job_paths = job_paths
            existing_scenario.optimal_vector = None
            existing_scenario.optimal_makespan = None
            existing_scenario.schedule = None
            app.scenario_list.item(
                str(id(existing_scenario)), values=(existing_scenario.name, "", "")
            )
        editor.destroy()

    ttk.Button(editor, text="OK", command=confirm_scenario).pack(pady=6)


def open_edit_scenario_dialog(app: "SchedulerGUI", scenario: Scenario) -> None:
    """Open custom path editor pre-filled with an existing scenario's paths."""
    if not app.scheduler:
        messagebox.showwarning("No config", "Please configure matrix and vectors first.")
        return
    scenario_index = app.scenarios.index(scenario) if scenario in app.scenarios else 0
    open_custom_path_editor(app, scenario_index, existing_scenario=scenario)
