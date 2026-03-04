from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def build_resource_tasks(
    schedule: List[Tuple[int, int, float, float]]
) -> Dict[int, List[Tuple[int, float, float]]]:
    resource_tasks: Dict[int, List[Tuple[int, float, float]]] = {}
    for resource, job, start, duration in schedule:
        if resource not in resource_tasks:
            resource_tasks[resource] = []
        resource_tasks[resource].append((job, start, duration))
    return resource_tasks


def build_gantt_figure(
    schedule: List[Tuple[int, int, float, float]],
    title: str = "Gantt Chart - Optimal Schedule",
) -> Figure:
    """
    Build and return a matplotlib Figure for the given schedule.
    This is used by the GUI to embed the chart inside a Tkinter tab.
    """
    if not schedule:
        return Figure()

    resource_tasks = build_resource_tasks(schedule)

    y_labels: List[str] = []
    y_positions: Dict[str, float] = {}
    y_pos = 0.0
    resource_boundaries: List[float] = []

    for resource in sorted(resource_tasks.keys()):
        tasks = resource_tasks[resource]
        tasks.sort(key=lambda x: x[1])

        for job, start, duration in tasks:
            label = f"R{resource + 1} - J{job + 1}"
            y_labels.append(label)
            y_positions[label] = y_pos
            y_pos += 1.0

        resource_boundaries.append(y_pos - 0.5)
        y_pos += 0.8

    fig = Figure(figsize=(14, max(6, len(y_labels) * 0.6)))
    ax = fig.add_subplot(111)

    colors = [
        "#3498db",
        "#e74c3c",
        "#2ecc71",
        "#f39c12",
        "#9b59b6",
        "#e67e22",
        "#1abc9c",
    ]

    for resource, job, start, duration in schedule:
        label = f"R{resource + 1} - J{job + 1}"
        y_position = y_positions[label]
        color = colors[job % len(colors)]

        ax.barh(
            y_position,
            duration,
            left=start,
            height=0.6,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.85,
        )

        ax.text(
            start + duration / 2,
            y_position,
            f"J{job + 1}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
            fontsize=10,
        )

    for boundary in resource_boundaries[:-1]:
        ax.axhline(y=boundary, color="gray", linestyle="--", linewidth=2, alpha=0.5)

    y_tick_positions = [y_positions[label] for label in y_labels]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Resources (with Jobs)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle=":", linewidth=0.8)

    fig.tight_layout()
    return fig


def show_gantt_by_resource(
    schedule: List[Tuple[int, int, float, float]],
    title: str = "Gantt Chart - Optimal Schedule",
):
    """
    Convenience helper to show the Gantt chart in a standalone window
    (for non-GUI usage). The GUI uses build_gantt_figure instead.
    """
    if not schedule:
        return

    fig = build_gantt_figure(schedule, title)
    # Use pyplot to show the figure in its own window
    plt.figure(fig.number)
    plt.show()


