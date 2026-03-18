import base64
import io
import tkinter as tk
from itertools import permutations, product
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from matplotlib.backends.backend_agg import FigureCanvasAgg

from .controllers.csv_controller import (
    load_csv_file,
    parse_csv_text,
    save_csv_file,
)
from .io_utils import SchedulerConfig, infer_dimensions_from_matrix
from .scheduler import ResourceScheduler, generate_random_job_paths
from .visualization import build_gantt_figure, show_gantt_by_resource


class Scenario:
    def __init__(self, name: str, job_paths: Dict[int, List[int]]):
        self.name = name
        self.job_paths = job_paths
        self.optimal_vector: Optional[List[int]] = None
        self.optimal_makespan: Optional[float] = None
        self.schedule = None


class SchedulerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flexible Job Shop Scheduler")
        self.geometry("1100x700")

        self.matrix: List[List[int]] = []
        self.resource_times: List[float] = []
        self.availability_vector: List[int] = []
        self.scheduler: Optional[ResourceScheduler] = None

        self.scenarios: List[Scenario] = []

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top_frame = ttk.Frame(self)
        top_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=4)

        ttk.Button(top_frame, text="Load from CSV", command=self.load_from_csv).pack(
            side=tk.LEFT, padx=4
        )

        ttk.Button(
            top_frame, text="New Matrix (manual)", command=self.configure_matrix_dialog
        ).pack(side=tk.LEFT, padx=4)

        self.info_label = ttk.Label(top_frame, text="No configuration loaded")
        self.info_label.pack(side=tk.LEFT, padx=12)

        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.grid(row=1, column=0, sticky="nsew")

        # Left side: matrix and vectors
        left_frame = ttk.Frame(main_pane)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        main_pane.add(left_frame, weight=3)

        self.matrix_table = ttk.Treeview(
            left_frame, columns=[], show="headings", height=10
        )
        self.matrix_table.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Vectors display
        vec_frame = ttk.Frame(left_frame)
        vec_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        self.resource_times_var = tk.StringVar()
        self.availability_var = tk.StringVar()
        ttk.Label(vec_frame, text="Resource times:").grid(
            row=0, column=0, sticky="w", padx=2
        )
        ttk.Label(vec_frame, textvariable=self.resource_times_var).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(vec_frame, text="Availability vector:").grid(
            row=1, column=0, sticky="w", padx=2
        )
        ttk.Label(vec_frame, textvariable=self.availability_var).grid(
            row=1, column=1, sticky="w"
        )

        # Right side: notebook with scenarios and diagram tabs
        right_notebook = ttk.Notebook(main_pane)
        main_pane.add(right_notebook, weight=2)

        # --- Scenarios tab ---
        right_frame = ttk.Frame(right_notebook)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        right_notebook.add(right_frame, text="Scenarios")
        self.right_notebook = right_notebook

        # Scenario config
        scenario_top = ttk.Frame(right_frame)
        scenario_top.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        ttk.Label(scenario_top, text="Number of scenarios:").grid(
            row=0, column=0, sticky="w"
        )
        self.num_scenarios_var = tk.IntVar(value=1)
        ttk.Spinbox(
            scenario_top,
            from_=1,
            to=10,
            textvariable=self.num_scenarios_var,
            width=5,
        ).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Button(
            scenario_top, text="Create Scenarios", command=self.create_scenarios_dialog
        ).grid(row=0, column=2, padx=8)

        # Scenario list
        self.scenario_list = ttk.Treeview(
            right_frame,
            columns=("name", "makespan", "vector"),
            show="headings",
            height=8,
        )
        self.scenario_list.heading("name", text="Scenario")
        self.scenario_list.heading("makespan", text="Optimal makespan")
        self.scenario_list.heading("vector", text="Optimal vector")
        self.scenario_list.column("name", width=120)
        self.scenario_list.column("makespan", width=120)
        self.scenario_list.column("vector", width=200)
        self.scenario_list.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # Scenario buttons
        btn_frame = ttk.Frame(right_frame)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=4)

        ttk.Button(
            btn_frame, text="Run selected", command=self.run_selected_scenario
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            btn_frame, text="Show Gantt (selected)", command=self.show_selected_gantt
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            btn_frame, text="Compare scenarios", command=self.compare_scenarios
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            btn_frame, text="Edit selected", command=self.edit_selected_scenario
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            btn_frame, text="Delete selected", command=self.delete_selected_scenario
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            btn_frame, text="Generate all scenarios", command=self.generate_all_scenarios
        ).pack(side=tk.LEFT, padx=4)

        # --- Diagram tab ---
        diagram_frame = ttk.Frame(right_notebook)
        diagram_frame.columnconfigure(0, weight=1)
        diagram_frame.rowconfigure(0, weight=1)
        right_notebook.add(diagram_frame, text="Diagram")
        self.diagram_frame = diagram_frame
        self.diagram_image = None
        self.diagram_label: Optional[tk.Label] = None

    # ------------------------------------------------------------------
    # Configuration
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
            self._apply_config(config)
            editor.destroy()

        ttk.Button(editor, text="Save and apply", command=apply_and_save).pack(
            side=tk.BOTTOM, pady=6
        )

    def configure_matrix_dialog(self):
        dialog = tk.Toplevel(self)
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
            self._edit_matrix_values(jobs_var.get(), res_var.get())

        ttk.Button(dialog, text="Next", command=confirm).grid(
            row=2, column=0, columnspan=2, pady=8
        )

    def _edit_matrix_values(self, num_jobs: int, num_resources: int):
        editor = tk.Toplevel(self)
        editor.title("Edit matrix and vectors")

        entries = []
        for i in range(num_jobs):
            row_entries = []
            for j in range(num_resources):
                e = ttk.Entry(editor, width=5)
                e.grid(row=i, column=j, padx=2, pady=2)
                e.insert(0, "1")
                row_entries.append(e)
            entries.append(row_entries)

        # Resource times
        ttk.Label(editor, text="Resource times (space-separated):").grid(
            row=num_jobs, column=0, columnspan=num_resources, sticky="w", pady=(8, 2)
        )
        times_entry = ttk.Entry(editor, width=40)
        times_entry.grid(row=num_jobs + 1, column=0, columnspan=num_resources, pady=2)
        times_entry.insert(0, "3 5 4" if num_resources == 3 else " ".join(["1"] * num_resources))

        # Availability vector
        ttk.Label(editor, text="Availability vector (space-separated):").grid(
            row=num_jobs + 2, column=0, columnspan=num_resources, sticky="w", pady=(8, 2)
        )
        avail_entry = ttk.Entry(editor, width=40)
        avail_entry.grid(row=num_jobs + 3, column=0, columnspan=num_resources, pady=2)
        avail_entry.insert(0, "1 " * num_resources)

        def confirm():
            try:
                matrix = []
                for i in range(num_jobs):
                    row = [int(entries[i][j].get()) for j in range(num_resources)]
                    matrix.append(row)
                resource_times = [float(x) for x in times_entry.get().split()]
                availability_vector = [int(x) for x in avail_entry.get().split()]
            except ValueError:
                messagebox.showerror("Error", "Invalid integer/float value.")
                return

            if len(resource_times) != num_resources or len(availability_vector) != num_resources:
                messagebox.showerror(
                    "Error", "Resource times and availability vector must match resource count."
                )
                return

            config = SchedulerConfig(
                matrix=matrix,
                resource_times=resource_times,
                availability_vector=availability_vector,
            )
            self._apply_config(config)
            editor.destroy()

        ttk.Button(editor, text="OK", command=confirm).grid(
            row=num_jobs + 4, column=0, columnspan=num_resources, pady=8
        )

    def _apply_config(self, config: SchedulerConfig):
        self.matrix = config.matrix
        self.resource_times = config.resource_times
        self.availability_vector = config.availability_vector
        self.scheduler = ResourceScheduler(self.matrix, self.resource_times)

        num_jobs, num_resources = infer_dimensions_from_matrix(self.matrix)
        self.info_label.config(
            text=f"Jobs: {num_jobs}, Resources: {num_resources}"
        )
        self.resource_times_var.set(str(self.resource_times))
        self.availability_var.set(str(self.availability_vector))
        self._refresh_matrix_table()
        # Keep existing scenarios but clear their results since configuration changed
        self.scenarios = []
        for row in self.scenario_list.get_children():
            self.scenario_list.delete(row)

    def _refresh_matrix_table(self):
        self.matrix_table.delete(*self.matrix_table.get_children())
        num_jobs, num_resources = infer_dimensions_from_matrix(self.matrix)
        cols = [f"R{j+1}" for j in range(num_resources)]
        self.matrix_table["columns"] = cols
        for col in cols:
            self.matrix_table.heading(col, text=col)
            self.matrix_table.column(col, width=60, anchor="center")

        for i in range(num_jobs):
            self.matrix_table.insert("", "end", values=self.matrix[i])

    # ------------------------------------------------------------------
    # Scenario creation and job path editing
    # ------------------------------------------------------------------
    def create_scenarios_dialog(self):
        if not self.scheduler:
            messagebox.showwarning("No config", "Please configure matrix and vectors first.")
            return

        num_jobs, num_resources = infer_dimensions_from_matrix(self.matrix)
        count = self.num_scenarios_var.get()

        # Do not clear old scenarios; append new ones after existing
        existing_count = len(self.scenarios)
        created_count = 0

        dialog = tk.Toplevel(self)
        dialog.title("Scenario types")

        ttk.Label(dialog, text="For each scenario choose random or custom paths.").grid(
            row=0, column=0, columnspan=3, padx=4, pady=4
        )

        def add_scenario(kind: str, index: int):
            nonlocal created_count
            logical_index = existing_count + index
            name = f"Scenario {logical_index+1}"
            if kind == "random":
                job_paths = generate_random_job_paths(num_jobs, num_resources, self.matrix)
                scenario = Scenario(name + " (random)", job_paths)
                self.scenarios.append(scenario)
                self._insert_scenario_in_list(scenario)
            else:
                self._open_custom_path_editor(logical_index)
            created_count += 1
            if created_count >= count:
                dialog.destroy()

        for i in range(count):
            ttk.Label(dialog, text=f"Scenario {i+1}:").grid(row=i + 1, column=0, padx=4, pady=2)
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

    def _open_custom_path_editor(self, scenario_index: int):
        if not self.scheduler:
            return
        num_jobs, num_resources = infer_dimensions_from_matrix(self.matrix)

        editor = tk.Toplevel(self)
        editor.title(f"Custom job ordering - Scenario {scenario_index+1}")
        editor.geometry("600x400")

        tree = ttk.Treeview(editor, columns=("path",), show="tree headings")
        tree.heading("#0", text="Job")
        tree.heading("path", text="Resource sequence")
        tree.column("path", width=300)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        job_nodes = {}
        for j in range(num_jobs):
            node = tree.insert("", "end", text=f"J{j+1}", values=("No path",))
            job_nodes[j] = node

        def edit_job_path(job_idx: int):
            path_dialog = tk.Toplevel(editor)
            path_dialog.title(f"Path for J{job_idx+1}")
            ttk.Label(
                path_dialog,
                text=f"Enter sequence of resource indices (1..{num_resources}), e.g. '3 1 2'",
            ).grid(row=0, column=0, padx=4, pady=4)
            entry = ttk.Entry(path_dialog, width=30)
            entry.grid(row=1, column=0, padx=4, pady=4)

            def confirm():
                try:
                    path = [int(x) - 1 for x in entry.get().split()]
                except ValueError:
                    messagebox.showerror("Error", "Invalid path format.")
                    return
                if not all(0 <= r < num_resources for r in path):
                    messagebox.showerror(
                        "Error",
                        f"Resources must be between 1 and {num_resources}.",
                    )
                    return
                if len(path) != len(set(path)):
                    messagebox.showerror("Error", "Duplicate resources in path.")
                    return
                tree.item(
                    job_nodes[job_idx],
                    values=(" → ".join([f"R{r+1}" for r in path]),),
                )
                path_dialog.destroy()

            ttk.Button(path_dialog, text="OK", command=confirm).grid(
                row=2, column=0, pady=6
            )

        def on_tree_double_click(event):
            item = tree.selection()
            if not item:
                return
            node = item[0]
            for j, n in job_nodes.items():
                if n == node:
                    edit_job_path(j)
                    break

        tree.bind("<Double-1>", on_tree_double_click)

        def confirm_scenario(existing_scenario: Optional[Scenario] = None):
            job_paths: Dict[int, List[int]] = {}
            for j in range(num_jobs):
                values = tree.item(job_nodes[j])["values"]
                if not values or values[0] == "No path":
                    messagebox.showerror(
                        "Error", f"Please define a path for job J{j+1}."
                    )
                    return
                tokens = values[0].replace("R", "").split("→")
                path = []
                for t in tokens:
                    t = t.strip()
                    if not t:
                        continue
                    idx = int(t)
                    path.append(idx - 1)
                job_paths[j] = path

            if existing_scenario is None:
                scenario = Scenario(f"Scenario {scenario_index+1} (custom)", job_paths)
                self.scenarios.append(scenario)
                self._insert_scenario_in_list(scenario)
            else:
                # Update existing scenario
                existing_scenario.job_paths = job_paths
                existing_scenario.optimal_vector = None
                existing_scenario.optimal_makespan = None
                existing_scenario.schedule = None
                # Refresh row in list
                self.scenario_list.item(
                    str(id(existing_scenario)),
                    values=(existing_scenario.name, "", ""),
                )
            editor.destroy()

        ttk.Button(editor, text="OK", command=lambda: confirm_scenario()).pack(pady=6)

    def _insert_scenario_in_list(self, scenario: Scenario):
        self.scenario_list.insert(
            "",
            "end",
            iid=str(id(scenario)),
            values=(scenario.name, scenario.optimal_makespan or "", scenario.optimal_vector or ""),
        )

    # ------------------------------------------------------------------
    # Running and comparing scenarios
    # ------------------------------------------------------------------
    def _find_scenario_by_iid(self, iid) -> Optional[Scenario]:
        for s in self.scenarios:
            if str(id(s)) == str(iid):
                return s
        return None

    def run_selected_scenario(self):
        if not self.scheduler:
            messagebox.showwarning("No config", "Please configure matrix and vectors first.")
            return
        if not self.matrix or not self.resource_times or not self.availability_vector:
            messagebox.showerror(
                "Missing configuration",
                "Matrix, resource times, and availability vector must be defined before running a scenario.",
            )
            return
        sel = self.scenario_list.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a scenario in the list.")
            return
        scenario = self._find_scenario_by_iid(sel[0])
        if not scenario:
            return

        optimal_vector, opt_schedule, opt_makespan, _ = self.scheduler.optimize_vector(
            self.availability_vector, scenario.job_paths
        )
        scenario.optimal_vector = optimal_vector
        scenario.optimal_makespan = opt_makespan
        scenario.schedule = opt_schedule

        self.scenario_list.item(
            sel[0],
            values=(scenario.name, scenario.optimal_makespan, scenario.optimal_vector),
        )

        messagebox.showinfo(
            "Scenario executed",
            f"{scenario.name}\nOptimal makespan: {opt_makespan}\nOptimal vector: {optimal_vector}",
        )

    def show_selected_gantt(self):
        sel = self.scenario_list.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a scenario first.")
            return
        scenario = self._find_scenario_by_iid(sel[0])
        if not scenario or not scenario.schedule:
            messagebox.showinfo("No schedule", "Run the scenario first.")
            return

        # Build figure and render it to a PNG image using Agg (no Pillow required)
        fig = build_gantt_figure(
            scenario.schedule,
            title=f"{scenario.name} - Makespan={scenario.optimal_makespan}",
        )
        buf = io.BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        buf.seek(0)
        png_data = buf.getvalue()
        b64_data = base64.b64encode(png_data)

        # Display inside Diagram tab via Tkinter PhotoImage
        photo = tk.PhotoImage(data=b64_data)
        self.diagram_image = photo  # keep reference to avoid GC

        if self.diagram_label is None:
            self.diagram_label = tk.Label(self.diagram_frame, image=photo)
            self.diagram_label.grid(row=0, column=0, sticky="nsew")
        else:
            self.diagram_label.configure(image=photo)

        # Switch to Diagram tab (index 1, Scenarios is 0)
        self.right_notebook.select(1)

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

        # Open the custom path editor pre-filled with current paths
        num_jobs, num_resources = infer_dimensions_from_matrix(self.matrix)
        editor = tk.Toplevel(self)
        editor.title(f"Edit job ordering - {scenario.name}")
        editor.geometry("600x400")

        tree = ttk.Treeview(editor, columns=("path",), show="tree headings")
        tree.heading("#0", text="Job")
        tree.heading("path", text="Resource sequence")
        tree.column("path", width=300)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        job_nodes: Dict[int, str] = {}
        for j in range(num_jobs):
            existing_path = scenario.job_paths.get(j, [])
            if existing_path:
                path_str = " → ".join([f"R{r+1}" for r in existing_path])
            else:
                path_str = "No path"
            node = tree.insert("", "end", text=f"J{j+1}", values=(path_str,))
            job_nodes[j] = node

        def edit_job_path(job_idx: int):
            path_dialog = tk.Toplevel(editor)
            path_dialog.title(f"Path for J{job_idx+1}")
            ttk.Label(
                path_dialog,
                text=f"Enter sequence of resource indices (1..{num_resources}), e.g. '3 1 2'",
            ).grid(row=0, column=0, padx=4, pady=4)
            entry = ttk.Entry(path_dialog, width=30)
            entry.grid(row=1, column=0, padx=4, pady=4)

            current = scenario.job_paths.get(job_idx, [])
            if current:
                entry.insert(0, " ".join(str(r + 1) for r in current))

            def confirm():
                try:
                    path = [int(x) - 1 for x in entry.get().split()]
                except ValueError:
                    messagebox.showerror("Error", "Invalid path format.")
                    return
                if not all(0 <= r < num_resources for r in path):
                    messagebox.showerror(
                        "Error",
                        f"Resources must be between 1 and {num_resources}.",
                    )
                    return
                if len(path) != len(set(path)):
                    messagebox.showerror("Error", "Duplicate resources in path.")
                    return
                tree.item(
                    job_nodes[job_idx],
                    values=(" → ".join([f"R{r+1}" for r in path]),),
                )
                path_dialog.destroy()

            ttk.Button(path_dialog, text="OK", command=confirm).grid(
                row=2, column=0, pady=6
            )

        def on_tree_double_click(event):
            item = tree.selection()
            if not item:
                return
            node = item[0]
            for j, n in job_nodes.items():
                if n == node:
                    edit_job_path(j)
                    break

        tree.bind("<Double-1>", on_tree_double_click)

        def confirm_changes():
            job_paths: Dict[int, List[int]] = {}
            for j in range(num_jobs):
                values = tree.item(job_nodes[j])["values"]
                if not values or values[0] == "No path":
                    messagebox.showerror(
                        "Error", f"Please define a path for job J{j+1}."
                    )
                    return
                tokens = values[0].replace("R", "").split("→")
                path: List[int] = []
                for t in tokens:
                    t = t.strip()
                    if not t:
                        continue
                    idx = int(t)
                    path.append(idx - 1)
                job_paths[j] = path

            scenario.job_paths = job_paths
            scenario.optimal_vector = None
            scenario.optimal_makespan = None
            scenario.schedule = None
            self.scenario_list.item(
                str(id(scenario)),
                values=(scenario.name, "", ""),
            )
            editor.destroy()

        ttk.Button(editor, text="OK", command=confirm_changes).pack(pady=6)

    def delete_selected_scenario(self):
        sel = self.scenario_list.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a scenario to delete.")
            return
        iid = sel[0]
        scenario = self._find_scenario_by_iid(iid)
        if not scenario:
            return
        # Remove from list and tree
        self.scenarios = [s for s in self.scenarios if s is not scenario]
        self.scenario_list.delete(iid)

    def compare_scenarios(self):
        executed = [s for s in self.scenarios if s.optimal_makespan is not None]
        if len(executed) < 2:
            messagebox.showinfo(
                "Not enough scenarios", "Run at least two scenarios to compare."
            )
            return

        # Choose scenario with minimal makespan; if tie, choose one with max vector
        def key_fn(s: Scenario):
            makespan = s.optimal_makespan
            vec = s.optimal_vector or []
            # negative of vector components for max vector with same makespan
            return (makespan, [-x for x in vec])

        best = min(executed, key=key_fn)

        lines = []
        for s in executed:
            lines.append(
                f"{s.name}: makespan={s.optimal_makespan}, vector={s.optimal_vector}"
            )
        lines.append("")
        lines.append(
            f"Best: {best.name} (makespan={best.optimal_makespan}, vector={best.optimal_vector})"
        )

        messagebox.showinfo("Scenario comparison", "\n".join(lines))

    def generate_all_scenarios(self):
        """
        Generate all possible scenarios given the matrix:
        for each job, consider all permutations of the resources it needs,
        and take the cartesian product over jobs.
        """
        if not self.scheduler:
            messagebox.showwarning("No config", "Please configure matrix and vectors first.")
            return
        if not self.matrix or not self.resource_times or not self.availability_vector:
            messagebox.showerror(
                "Missing configuration",
                "Matrix, resource times, and availability vector must be defined before generating scenarios.",
            )
            return

        num_jobs, num_resources = infer_dimensions_from_matrix(self.matrix)

        job_perms: List[List[List[int]]] = []
        for j in range(num_jobs):
            needed = [r for r in range(num_resources) if self.matrix[j][r] > 0]
            if not needed:
                job_perms.append([[]])
            else:
                perms = [list(p) for p in permutations(needed)]
                job_perms.append(perms)

        total_scenarios = 1
        for perms in job_perms:
            total_scenarios *= len(perms)

        if total_scenarios <= 0:
            messagebox.showinfo("No scenarios", "No valid job paths can be generated from this matrix.")
            return

        if total_scenarios > 500:
            proceed = messagebox.askyesno(
                "Many scenarios",
                f"This will generate and run {total_scenarios} scenarios.\n\n"
                f"This may take some time. Continue?",
            )
            if not proceed:
                return

        best_scenario: Optional[Scenario] = None

        def key_fn(s: Scenario):
            makespan = s.optimal_makespan
            vec = s.optimal_vector or []
            return (makespan, [-x for x in vec])

        index = 0
        for combo in product(*job_perms):
            job_paths: Dict[int, List[int]] = {}
            for j in range(num_jobs):
                job_paths[j] = list(combo[j])

            name = f"Auto Scenario {index + 1}"
            scenario = Scenario(name, job_paths)

            optimal_vector, opt_schedule, opt_makespan, _ = self.scheduler.optimize_vector(
                self.availability_vector, job_paths
            )
            scenario.optimal_vector = optimal_vector
            scenario.optimal_makespan = opt_makespan
            scenario.schedule = opt_schedule

            self.scenarios.append(scenario)
            self._insert_scenario_in_list(scenario)

            if best_scenario is None or key_fn(scenario) < key_fn(best_scenario):
                best_scenario = scenario

            index += 1

        if best_scenario is not None:
            best_iid = str(id(best_scenario))
            self.scenario_list.selection_set(best_iid)
            self.scenario_list.see(best_iid)


def run_gui():
    app = SchedulerGUI()
    app.mainloop()

