
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..components.models import Scenario
    from ..gui import SchedulerGUI


def on_scenario_double_click(app: "SchedulerGUI", event) -> None:

    sel = app.scenario_list.selection()
    if not sel:
        return
    scenario = find_scenario_by_iid(app, sel[0])
    if not scenario:
        return
    if scenario.schedule is None:
        app.run_selected_scenario()
    app.show_result_for_selected()


def insert_scenario_in_list(app: "SchedulerGUI", scenario: "Scenario") -> None:

    app.scenario_list.insert(
        "",
        "end",
        iid=str(id(scenario)),
        values=(scenario.name, scenario.optimal_makespan or "", scenario.optimal_vector or ""),
    )


def find_scenario_by_iid(app: "SchedulerGUI", iid) -> Optional["Scenario"]:

    for s in app.scenarios:
        if str(id(s)) == str(iid):
            return s
    return None
