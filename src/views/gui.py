"""
Main application window for the Flexible Job Shop Scheduler GUI.

This module is intentionally thin: it wires together the domain models, the
scheduling core, and the reusable UI components that live in
``src/views/components/``.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from ..controllers.csv_controller import load_csv_file, parse_csv_text, save_csv_file
from ..core.io_utils import SchedulerConfig, infer_dimensions_from_matrix
from ..core.scheduler import ResourceScheduler

from .components.models import Scenario
from .components.matrix_editor import open_new_matrix_dialog, open_edit_matrix_dialog
from .components.scenario_editor import (
    open_create_scenarios_dialog,
    open_edit_scenario_dialog,
)
from .components.result_viewer import configure_log_tags, render_logs_animated, render_diagram
from .components.stats_viewer import render_stats
from .components.profile_dialogs import (
    open_save_profile_dialog,
    open_save_current_profile_dialog,
    open_load_profile_dialog,
)


class SchedulerGUI(tk.Tk):
    """
    Root window for the Flexible Job Shop Scheduler.

    State management
    ----------------
    The window owns the current ``matrix``, ``resource_times``,
    ``availability_vector``, ``scheduler``, and ``scenarios`` list.
    UI components receive a reference to *self* and call back into the methods
    defined here (e.g. ``_apply_config``, ``_insert_scenario_in_list``).
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.title("Flexible Job Shop Scheduler")
        self.geometry("1100x700")
        self.profile_db_path = "profiles.db"
        self.current_profile_name: Optional[str] = None

        # ── Scheduling state ──
        self.matrix: List[List[int]] = []
        self.resource_times: List[float] = []
        self.availability_vector: List[int] = []
        self.scheduler: Optional[ResourceScheduler] = None

        # ── Scenarios ──
        self.scenarios: List[Scenario] = []

        # ── Internal refs kept alive to prevent GC ──
        self.diagram_image = None
        self._stats_image_ref = None
        self._log_after_id = None

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_top_bar()

        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.grid(row=1, column=0, sticky="nsew")

        self._build_left_panel(main_pane)
        self._build_right_panel(main_pane)

    def _build_top_bar(self):
        top_frame = ttk.Frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=4)

        # ── New menu ──
        new_btn = ttk.Menubutton(top_frame, text="New")
        new_menu = tk.Menu(new_btn, tearoff=0)
        new_menu.add_command(label="Load from CSV", command=self.load_from_csv)
        new_menu.add_command(label="New Matrix (manual)", command=lambda: open_new_matrix_dialog(self))
        new_btn["menu"] = new_menu
        new_btn.pack(side=tk.LEFT, padx=4)

        # ── Edit menu ──
        edit_btn = ttk.Menubutton(top_frame, text="Edit")
        edit_menu = tk.Menu(edit_btn, tearoff=0)
        edit_menu.add_command(label="Edit configuration", command=lambda: open_edit_matrix_dialog(self))
        edit_btn["menu"] = edit_menu
        edit_btn.pack(side=tk.LEFT, padx=4)

        # ── Profile menu ──
        profile_btn = ttk.Menubutton(top_frame, text="Profile")
        profile_menu = tk.Menu(profile_btn, tearoff=0)
        profile_menu.add_command(label="Save profile", command=lambda: open_save_profile_dialog(self))
        profile_menu.add_command(label="Save current profile", command=lambda: open_save_current_profile_dialog(self))
        profile_menu.add_command(label="Load profile", command=lambda: open_load_profile_dialog(self))
        profile_btn["menu"] = profile_menu
        profile_btn.pack(side=tk.LEFT, padx=4)

        self.info_label = ttk.Label(top_frame, text="No configuration loaded")
        self.info_label.pack(side=tk.LEFT, padx=12)

    def _build_left_panel(self, parent: ttk.PanedWindow):
        left_frame = ttk.Frame(parent)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        parent.add(left_frame, weight=2)

        # Matrix Treeview
        self.matrix_table = ttk.Treeview(left_frame, columns=[], show="headings", height=10)
        self.matrix_table.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Vector labels
        vec_frame = ttk.Frame(left_frame)
        vec_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        self.resource_times_var = tk.StringVar()
        self.availability_var = tk.StringVar()
        ttk.Label(vec_frame, text="Resource times:").grid(row=0, column=0, sticky="w", padx=2)
        ttk.Label(vec_frame, textvariable=self.resource_times_var).grid(row=0, column=1, sticky="w")
        ttk.Label(vec_frame, text="Availability vector:").grid(row=1, column=0, sticky="w", padx=2)
        ttk.Label(vec_frame, textvariable=self.availability_var).grid(row=1, column=1, sticky="w")

    def _build_right_panel(self, parent: ttk.PanedWindow):
        right_notebook = ttk.Notebook(parent)
        parent.add(right_notebook, weight=5)
        self.right_notebook = right_notebook

        self._build_scenarios_tab(right_notebook)
        self._build_result_tab(right_notebook)
        self._build_stats_tab(right_notebook)

    def _build_scenarios_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        notebook.add(frame, text="Scenarios")

        # Controls row
        top = ttk.Frame(frame)
        top.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        ttk.Label(top, text="Number of scenarios:").grid(row=0, column=0, sticky="w")
        self.num_scenarios_var = tk.IntVar(value=1)
        ttk.Spinbox(top, from_=1, to=10, textvariable=self.num_scenarios_var, width=5).grid(
            row=0, column=1, sticky="w", padx=4
        )
        ttk.Button(top, text="Create Scenarios", command=lambda: open_create_scenarios_dialog(self)).grid(
            row=0, column=2, padx=8
        )

        # Scenario list
        self.scenario_list = ttk.Treeview(
            frame,
            columns=("name", "makespan", "vector"),
            show="headings",
            height=8,
        )
        for col, label, width in [("name", "Scenario", 120), ("makespan", "Optimal makespan", 120), ("vector", "Optimal vector", 200)]:
            self.scenario_list.heading(col, text=label)
            self.scenario_list.column(col, width=width)
        self.scenario_list.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.scenario_list.bind("<Double-1>", self._on_scenario_double_click)

        # Action buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(btn_frame, text="Run selected", command=self.run_selected_scenario).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Show Result", command=self.show_result_for_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Edit selected", command=self.edit_selected_scenario).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Delete selected", command=self.delete_selected_scenario).pack(side=tk.LEFT, padx=4)

    def _build_result_tab(self, notebook: ttk.Notebook):
        result_frame = ttk.Frame(notebook)
        notebook.add(result_frame, text="Result")

        result_pane = ttk.PanedWindow(result_frame, orient=tk.VERTICAL)
        result_pane.pack(fill=tk.BOTH, expand=True)

        # Diagram area
        diagram_container = ttk.Frame(result_pane)
        diagram_container.columnconfigure(0, weight=1)
        diagram_container.rowconfigure(0, weight=1)
        self.diagram_frame = diagram_container
        self.diagram_label: Optional[tk.Label] = None

        # Log area
        log_frame = ttk.Frame(result_pane)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, wrap="word", height=18, font=("TkDefaultFont", 12))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        configure_log_tags(self.log_text)

        result_pane.add(diagram_container, weight=3)
        result_pane.add(log_frame, weight=2)

    def _build_stats_tab(self, notebook: ttk.Notebook):
        stats_frame = ttk.Frame(notebook)
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
        notebook.add(stats_frame, text="Stats")

        stats_pane = ttk.PanedWindow(stats_frame, orient=tk.VERTICAL)
        stats_pane.grid(row=0, column=0, sticky="nsew")

        # Chart area
        self.stats_graphics_frame = ttk.Frame(stats_pane)
        self.stats_graphics_frame.columnconfigure(0, weight=1)
        self.stats_graphics_frame.rowconfigure(0, weight=1)
        self.stats_graphics_label: Optional[tk.Label] = None

        # KPI text area
        stats_text_frame = ttk.Frame(stats_pane)
        stats_text_frame.columnconfigure(0, weight=1)
        stats_text_frame.rowconfigure(0, weight=1)
        self.stats_text = tk.Text(stats_text_frame, wrap="word", font=("TkDefaultFont", 12))
        self.stats_text.grid(row=0, column=0, sticky="nsew")
        self.stats_text.configure(state="disabled")

        stats_pane.add(self.stats_graphics_frame, weight=3)
        stats_pane.add(stats_text_frame, weight=2)

    # ------------------------------------------------------------------
    # CSV I/O
    # ------------------------------------------------------------------

    def load_from_csv(self):
        """Prompt for a CSV file, open an inline editor, then apply the config."""
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            content = load_csv_file(path)
        except Exception as e:
            messagebox.showerror("Input error", f"Could not read CSV file:\n{e}")
            return

        editor = tk.Toplevel(self)
        editor.title("Edit CSV input")
        editor.geometry("700x500")

        text = tk.Text(editor, wrap="none")
        text.insert("1.0", content)
        text.pack(fill=tk.BOTH, expand=True)

        def apply_and_save():
            csv_text = text.get("1.0", tk.END)
            try:
                config: SchedulerConfig = parse_csv_text(csv_text)
            except Exception as e:
                messagebox.showerror("Input error", f"Invalid CSV format:\n{e}")
                return
            try:
                save_csv_file(path, csv_text)
            except Exception as e:
                messagebox.showerror("Input error", f"Could not save CSV file:\n{e}")
                return
            self._apply_config(config, rerun_scenarios=True)
            editor.destroy()

        ttk.Button(editor, text="Save and apply", command=apply_and_save).pack(
            side=tk.BOTTOM, pady=6
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _apply_config(self, config: SchedulerConfig, *, rerun_scenarios: bool = True):
        """Update internal state from a SchedulerConfig and refresh the UI."""
        self.matrix = config.matrix
        self.resource_times = config.resource_times
        self.availability_vector = config.availability_vector
        self.scheduler = ResourceScheduler(self.matrix, self.resource_times)

        num_jobs, num_resources = infer_dimensions_from_matrix(self.matrix)
        self.info_label.config(text=f"Jobs: {num_jobs}, Resources: {num_resources}")
        self.resource_times_var.set(str(self.resource_times))
        self.availability_var.set(str(self.availability_vector))
        self._refresh_matrix_table()

        if not rerun_scenarios:
            return

        for s in self.scenarios:
            ok, reason = self._scenario_compatible(s, num_jobs, num_resources)
            if not ok:
                s.optimal_vector = None
                s.optimal_makespan = None
                s.schedule = None
                s.logs = reason or "Scenario is not compatible with the current configuration."
                self.scenario_list.item(str(id(s)), values=(s.name, "", ""))
                continue
            self._run_scenario_object(s, silent=True)
            self.scenario_list.item(
                str(id(s)),
                values=(s.name, s.optimal_makespan or "", s.optimal_vector or ""),
            )

    def _scenario_compatible(self, scenario: Scenario, num_jobs: int, num_resources: int):
        """Return (True, '') or (False, reason_str) for a scenario vs current dims."""
        for j in range(num_jobs):
            if j not in scenario.job_paths:
                return False, f"Scenario missing path for J{j + 1} under current configuration."
            path = scenario.job_paths.get(j, [])
            if any(r < 0 or r >= num_resources for r in path):
                return False, f"Scenario has invalid resource index in job path for J{j + 1}."
        return True, ""

    def _refresh_matrix_table(self):
        self.matrix_table.delete(*self.matrix_table.get_children())
        num_jobs, num_resources = infer_dimensions_from_matrix(self.matrix)
        cols = [f"R{j + 1}" for j in range(num_resources)]
        self.matrix_table["columns"] = cols
        for col in cols:
            self.matrix_table.heading(col, text=col)
            self.matrix_table.column(col, width=60, anchor="center")
        for i in range(num_jobs):
            self.matrix_table.insert("", "end", values=self.matrix[i])

    # ------------------------------------------------------------------
    # Scenario execution
    # ------------------------------------------------------------------

    def _run_scenario_object(self, scenario: Scenario, *, silent: bool = False):
        """Run optimisation for *scenario* and store results + logs."""
        if not self.scheduler:
            return

        scenario_index = (self.scenarios.index(scenario) + 1) if scenario in self.scenarios else 0
        log_lines: List[str] = []
        log_lines.append("=" * 70)
        log_lines.append(f"SCENARIO {scenario_index}: {scenario.name}" if scenario_index else f"SCENARIO: {scenario.name}")
        log_lines.append("=" * 70)
        log_lines.append("")

        optimal_vector, opt_schedule, opt_makespan, _ = self.scheduler.optimize_vector(
            self.availability_vector,
            scenario.job_paths,
            logger=log_lines.append,
            scenario_name=scenario.name,
            scenario_index=scenario_index if scenario_index else None,
        )
        scenario.optimal_vector = optimal_vector
        scenario.optimal_makespan = opt_makespan
        scenario.schedule = opt_schedule

        # ── Job execution process ──
        log_lines.append("")
        log_lines.append("=" * 70)
        log_lines.append("JOB EXECUTION PROCESS")
        log_lines.append("=" * 70)
        if opt_schedule:
            for r_col, j_col, start_t, dur in sorted(opt_schedule, key=lambda x: (x[2], x[1])):
                log_lines.append(
                    f"  Job {j_col + 1} on Resource {r_col + 1}:"
                    f"  Start={start_t:<6.1f} Finish={start_t + dur:<6.1f} Duration={dur:<6.1f}"
                )
        else:
            log_lines.append("  No schedule generated.")

        # ── Summary ──
        log_lines.append("")
        log_lines.append("=" * 70)
        log_lines.append(f"SCENARIO {scenario_index} SUMMARY" if scenario_index else "SCENARIO SUMMARY")
        log_lines.append("=" * 70)
        log_lines.append(f"  Initial: {self.availability_vector} → Makespan {opt_makespan}")
        log_lines.append(f"  Optimal: {optimal_vector} → Makespan {opt_makespan}")

        scenario.logs = "\n".join(log_lines)

        if not silent:
            self.scenario_list.item(
                str(id(scenario)),
                values=(scenario.name, scenario.optimal_makespan, scenario.optimal_vector),
            )

    def run_selected_scenario(self):
        if not self.scheduler:
            messagebox.showwarning("No config", "Please configure matrix and vectors first.")
            return
        if not self.matrix or not self.resource_times or not self.availability_vector:
            messagebox.showerror("Missing configuration", "Matrix, resource times, and availability vector must be defined.")
            return
        sel = self.scenario_list.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a scenario in the list.")
            return
        scenario = self._find_scenario_by_iid(sel[0])
        if not scenario:
            return
        self._run_scenario_object(scenario)
        messagebox.showinfo(
            "Scenario executed",
            f"{scenario.name}\nOptimal makespan: {scenario.optimal_makespan}\nOptimal vector: {scenario.optimal_vector}",
        )

    def show_result_for_selected(self):
        sel = self.scenario_list.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a scenario first.")
            return
        scenario = self._find_scenario_by_iid(sel[0])
        if not scenario:
            return
        if not scenario.schedule:
            messagebox.showinfo("No schedule", "Run the scenario first.")
            return
        render_diagram(self, scenario)
        render_logs_animated(self, scenario.logs or "No log available for this scenario.")
        render_stats(self, scenario)
        self.right_notebook.select(1)  # Switch to Result tab

    def edit_selected_scenario(self):
        if not self.scheduler:
            messagebox.showwarning("No config", "Please configure matrix and vectors first.")
            return
        sel = self.scenario_list.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a scenario in the list.")
            return
        scenario = self._find_scenario_by_iid(sel[0])
        if not scenario:
            return
        open_edit_scenario_dialog(self, scenario)

    def delete_selected_scenario(self):
        sel = self.scenario_list.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a scenario to delete.")
            return
        iid = sel[0]
        scenario = self._find_scenario_by_iid(iid)
        if not scenario:
            return
        self.scenarios = [s for s in self.scenarios if s is not scenario]
        self.scenario_list.delete(iid)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _on_scenario_double_click(self, event):
        sel = self.scenario_list.selection()
        if not sel:
            return
        scenario = self._find_scenario_by_iid(sel[0])
        if not scenario:
            return
        if scenario.schedule is None:
            self.run_selected_scenario()
        self.show_result_for_selected()

    def _insert_scenario_in_list(self, scenario: Scenario):
        self.scenario_list.insert(
            "",
            "end",
            iid=str(id(scenario)),
            values=(scenario.name, scenario.optimal_makespan or "", scenario.optimal_vector or ""),
        )

    def _find_scenario_by_iid(self, iid) -> Optional[Scenario]:
        for s in self.scenarios:
            if str(id(s)) == str(iid):
                return s
        return None


def run_gui():
    app = SchedulerGUI()
    app.mainloop()
