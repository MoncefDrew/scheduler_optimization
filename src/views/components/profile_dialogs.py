
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import TYPE_CHECKING

from ...controllers.profile_controller import (
    list_profiles,
    load_profile,
    profile_exists,
    save_profile,
    update_profile_by_name,
)
from ...core.io_utils import SchedulerConfig
from .models import Scenario

if TYPE_CHECKING:
    from ..gui import SchedulerGUI


def open_save_profile_dialog(app: "SchedulerGUI") -> None:

    if not app.matrix or not app.resource_times or not app.availability_vector:
        messagebox.showerror(
            "Missing configuration",
            "Configure matrix, resource times, and availability vector before saving.",
        )
        return

    name = simpledialog.askstring("Save profile", "Profile name:")
    if not name:
        return
    if profile_exists(app.profile_db_path, name):
        messagebox.showerror("Input error", "A profile with this name already exists.")
        return

    config = SchedulerConfig(
        matrix=app.matrix,
        resource_times=app.resource_times,
        availability_vector=app.availability_vector,
    )
    scenarios_payload = _build_scenarios_payload(app.scenarios)
    try:
        save_profile(app.profile_db_path, name, config, scenarios_payload)
    except Exception as e:
        messagebox.showerror("Input error", f"Could not save profile:\n{e}")
        return

    app.current_profile_name = name
    messagebox.showinfo("Saved", "Profile saved.")


def open_save_current_profile_dialog(app: "SchedulerGUI") -> None:

    if not app.current_profile_name:
        messagebox.showinfo(
            "No current profile",
            "Load a profile first (or save a new one) to enable 'Save current profile'.",
        )
        return
    if not app.matrix or not app.resource_times or not app.availability_vector:
        messagebox.showerror(
            "Missing configuration",
            "Configure matrix, resource times, and availability vector before saving.",
        )
        return

    config = SchedulerConfig(
        matrix=app.matrix,
        resource_times=app.resource_times,
        availability_vector=app.availability_vector,
    )
    scenarios_payload = _build_scenarios_payload(app.scenarios)
    try:
        update_profile_by_name(app.profile_db_path, app.current_profile_name, config, scenarios_payload)
    except Exception as e:
        messagebox.showerror("Input error", f"Could not update profile:\n{e}")
        return

    messagebox.showinfo("Saved", f"Profile updated: {app.current_profile_name}")


def open_load_profile_dialog(app: "SchedulerGUI") -> None:

    try:
        profiles = list_profiles(app.profile_db_path)
    except Exception as e:
        messagebox.showerror("Input error", f"Could not read profiles:\n{e}")
        return
    if not profiles:
        messagebox.showinfo("No profiles", "No saved profiles found.")
        return

    dlg = tk.Toplevel(app)
    dlg.title("Load profile")
    dlg.geometry("500x300")
    dlg.columnconfigure(0, weight=1)
    dlg.rowconfigure(0, weight=1)

    lst = tk.Listbox(dlg)
    lst.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    for p in profiles:
        lst.insert(tk.END, f"[{p.id}] {p.name} - {p.created_at}")

    def do_load():
        sel = lst.curselection()
        if not sel:
            return
        line = lst.get(sel[0])
        pid = int(line.split("]")[0].lstrip("["))
        try:
            config, scenarios, meta = load_profile(app.profile_db_path, pid)
        except Exception as e:
            messagebox.showerror("Input error", f"Could not load profile:\n{e}")
            return

        # Clear existing scenarios
        app.scenarios = []
        for row in app.scenario_list.get_children():
            app.scenario_list.delete(row)

        app._apply_config(config, rerun_scenarios=False)

        # Restore scenarios with stored results
        for s in scenarios:
            sc = Scenario(s["name"], s["job_paths"])
            sc.optimal_vector = s.get("optimal_vector")
            sc.optimal_makespan = s.get("optimal_makespan")
            sc.schedule = s.get("schedule")
            sc.logs = s.get("logs", "")
            app.scenarios.append(sc)
            app._insert_scenario_in_list(sc)
            if sc.optimal_vector is not None or sc.optimal_makespan is not None:
                app.scenario_list.item(
                    str(id(sc)),
                    values=(sc.name, sc.optimal_makespan, sc.optimal_vector),
                )

        app.current_profile_name = meta.split(" (created ")[0]
        app.info_label.config(text=f"Loaded profile: {meta}")
        dlg.destroy()

    ttk.Button(dlg, text="Load", command=do_load).grid(row=1, column=0, pady=6)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_scenarios_payload(scenarios):
    return [
        {
            "name": s.name,
            "job_paths": s.job_paths,
            "optimal_vector": s.optimal_vector,
            "optimal_makespan": s.optimal_makespan,
            "schedule": s.schedule,
            "logs": s.logs,
        }
        for s in scenarios
    ]
