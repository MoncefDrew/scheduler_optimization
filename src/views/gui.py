
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from ..core.io_utils import (
    SchedulerConfig,
    infer_dimensions_from_matrix,
    load_csv_file,
    parse_csv_text,
    save_csv_file,
)
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
from .utils.config_helpers import apply_config, scenario_compatible, refresh_matrix_table
from .utils.ui_helpers import on_scenario_double_click, insert_scenario_in_list, find_scenario_by_iid


class SchedulerGUI(tk.Tk):


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
    # Configuration helpers  (delegate to views/utils/config_helpers.py)
    # ------------------------------------------------------------------

    def _apply_config(self, config: SchedulerConfig, *, rerun_scenarios: bool = True):
        apply_config(self, config, rerun_scenarios=rerun_scenarios)

    def _scenario_compatible(self, scenario: Scenario, num_jobs: int, num_resources: int):
        return scenario_compatible(self, scenario, num_jobs, num_resources)

    def _refresh_matrix_table(self):
        refresh_matrix_table(self)

    # ------------------------------------------------------------------
    # Scenario execution
    # ------------------------------------------------------------------

    def _run_scenario_object(self, scenario: Scenario, *, silent: bool = False):

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
    # Utility helpers  (delegate to views/utils/ui_helpers.py)
    # ------------------------------------------------------------------

    def _on_scenario_double_click(self, event):
        on_scenario_double_click(self, event)

    def _insert_scenario_in_list(self, scenario: Scenario):
        insert_scenario_in_list(self, scenario)

    def _find_scenario_by_iid(self, iid) -> Optional[Scenario]:
        return find_scenario_by_iid(self, iid)


def run_gui():
    app = SchedulerGUI()
    app.mainloop()
