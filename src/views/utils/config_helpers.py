
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ...core.io_utils import SchedulerConfig, infer_dimensions_from_matrix
from ...core.scheduler import ResourceScheduler

if TYPE_CHECKING:
    from ..components.models import Scenario
    from ..gui import SchedulerGUI


def apply_config(
    app: "SchedulerGUI",
    config: SchedulerConfig,
    *,
    rerun_scenarios: bool = True,
) -> None:

    app.matrix = config.matrix
    app.resource_times = config.resource_times
    app.availability_vector = config.availability_vector
    app.scheduler = ResourceScheduler(app.matrix, app.resource_times)

    num_jobs, num_resources = infer_dimensions_from_matrix(app.matrix)
    app.info_label.config(text=f"Jobs: {num_jobs}, Resources: {num_resources}")
    app.resource_times_var.set(str(app.resource_times))
    app.availability_var.set(str(app.availability_vector))
    refresh_matrix_table(app)

    if not rerun_scenarios:
        return

    # Clear scenarios – a new matrix invalidates all existing routing paths.
    app.scenarios = []
    for row in app.scenario_list.get_children():
        app.scenario_list.delete(row)


def scenario_compatible(
    app: "SchedulerGUI",
    scenario: "Scenario",
    num_jobs: int,
    num_resources: int,
) -> Tuple[bool, str]:

    for j in range(num_jobs):
        if j not in scenario.job_paths:
            return False, f"Scenario missing path for J{j + 1} under current configuration."
        path = scenario.job_paths.get(j, [])
        if any(r < 0 or r >= num_resources for r in path):
            return False, f"Scenario has invalid resource index in job path for J{j + 1}."
    return True, ""


def refresh_matrix_table(app: "SchedulerGUI") -> None:

    app.matrix_table.delete(*app.matrix_table.get_children())
    num_jobs, num_resources = infer_dimensions_from_matrix(app.matrix)
    cols = [f"R{j + 1}" for j in range(num_resources)]
    app.matrix_table["columns"] = cols
    for col in cols:
        app.matrix_table.heading(col, text=col)
        app.matrix_table.column(col, width=60, anchor="center")
    for i in range(num_jobs):
        app.matrix_table.insert("", "end", values=app.matrix[i])
