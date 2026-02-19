"""
Main Entry Point with Visualization
Run this file to execute the scheduler
"""

import matplotlib.pyplot as plt
from scheduler import compute_schedule_detailed, iterative_vector_reduction


# =============================================================================
# VISUALIZATION
# =============================================================================


def print_gantt_chart(schedule):
    """Generate Gantt chart showing resource allocation over time"""

    if schedule is None or len(schedule) == 0:
        print("Cannot generate Gantt chart - no valid schedule")
        return

    # Group tasks by resource
    resource_tasks = {}
    for resource, job, start, duration in schedule:
        if resource not in resource_tasks:
            resource_tasks[resource] = []
        resource_tasks[resource].append((job, start, duration))

    # Create y-axis labels
    y_labels = []
    y_positions = {}
    y_pos = 0
    resource_boundaries = []

    for resource in sorted(resource_tasks.keys()):
        tasks = resource_tasks[resource]
        tasks.sort(key=lambda x: x[1])  # Sort by start time

        for job, start, duration in tasks:
            label = f"R{resource + 1} - J{job + 1}"
            y_labels.append(label)
            y_positions[label] = y_pos
            y_pos += 1

        resource_boundaries.append(y_pos - 0.5)
        y_pos += 0.8

    # Create plot
    plt.figure(figsize=(12, max(6, len(y_labels) * 0.5 + len(resource_tasks) * 0.4)))

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

        plt.barh(
            y_position,
            duration,
            left=start,
            height=0.5,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

        plt.text(
            start + duration / 2,
            y_position,
            f"J{job + 1}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
            fontsize=9,
        )

    # Separator lines between resources
    for boundary in resource_boundaries[:-1]:
        plt.axhline(y=boundary, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    y_tick_positions = [y_positions[label] for label in y_labels]
    plt.yticks(y_tick_positions, y_labels)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Resources (with Jobs)", fontsize=12)
    plt.title("Gantt Chart - Optimal Schedule", fontsize=14)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# INPUT FUNCTIONS
# =============================================================================


def input_matrix_dimensions():
    """Get matrix dimensions from user"""
    print("\n" + "=" * 70)
    print("MATRIX DIMENSIONS")
    print("=" * 70)

    while True:
        try:
            num_jobs = int(input("\nEnter number of JOBS: "))
            if num_jobs <= 0:
                print("Error: Number of jobs must be positive")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer")

    while True:
        try:
            num_resources = int(input("Enter number of RESOURCES: "))
            if num_resources <= 0:
                print("Error: Number of resources must be positive")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer")

    return num_jobs, num_resources


def input_matrix(num_jobs, num_resources):
    """Get matrix values from user"""
    print("\n" + "=" * 70)
    print("JOBS × RESOURCES MATRIX")
    print("=" * 70)
    print(f"\nEnter the demand matrix ({num_jobs} jobs × {num_resources} resources)")
    print("Each row represents a job, each value is tokens needed from that resource")

    matrix = []

    for i in range(num_jobs):
        while True:
            try:
                print(f"\nJ{i + 1}: Enter {num_resources} values separated by spaces")
                row_input = input(f"J{i + 1}: ")
                row = list(map(int, row_input.split()))

                if len(row) != num_resources:
                    print(f"Error: Expected {num_resources} values, got {len(row)}")
                    continue

                if any(val < 0 for val in row):
                    print("Error: Values cannot be negative")
                    continue

                matrix.append(row)
                break
            except ValueError:
                print("Error: Please enter valid integers separated by spaces")

    return matrix


def input_resource_times(num_resources):
    """Get resource execution times from user"""
    print("\n" + "=" * 70)
    print("RESOURCE EXECUTION TIMES")
    print("=" * 70)
    print(f"\nEnter execution time per token for each resource")
    print(f"Enter {num_resources} values separated by spaces (e.g., '3 5 4')")

    while True:
        try:
            times_input = input(f"Execution times: ")
            times = list(map(float, times_input.split()))

            if len(times) != num_resources:
                print(f"Error: Expected {num_resources} values, got {len(times)}")
                continue

            if any(t <= 0 for t in times):
                print("Error: Execution times must be positive")
                continue

            return times
        except ValueError:
            print("Error: Please enter valid numbers separated by spaces")


def input_availability_vector(num_resources):
    """Get availability vector from user"""
    print("\n" + "=" * 70)
    print("AVAILABILITY VECTOR")
    print("=" * 70)
    print(f"\nEnter number of available units for each resource")
    print(f"Enter {num_resources} values separated by spaces (e.g., '2 3 2')")

    while True:
        try:
            vector_input = input(f"Availability vector: ")
            vector = list(map(int, vector_input.split()))

            if len(vector) != num_resources:
                print(f"Error: Expected {num_resources} values, got {len(vector)}")
                continue

            if any(v < 0 for v in vector):
                print("Error: Values cannot be negative")
                continue

            return vector
        except ValueError:
            print("Error: Please enter valid integers separated by spaces")


def use_predefined_example():
    """Ask if user wants to use predefined example"""
    print("\n" + "=" * 70)
    print("RESOURCE SCHEDULING SYSTEM")
    print("=" * 70)

    choice = (
        input("\nDo you want to use the predefined example? (y/n): ").strip().lower()
    )

    if choice == "y":
        from config import JOBS_RESOURCES_MATRIX, RESOURCE_EXECUTION_TIMES

        print("\nUsing predefined example:")
        print(f"  Jobs: {len(JOBS_RESOURCES_MATRIX)}")
        print(f"  Resources: {len(JOBS_RESOURCES_MATRIX[0])}")
        print(f"  Matrix:")
        for i, row in enumerate(JOBS_RESOURCES_MATRIX):
            print(f"    J{i + 1}: {row}")
        print(f"  Resource times: {RESOURCE_EXECUTION_TIMES}")

        availability_vector = input_availability_vector(len(JOBS_RESOURCES_MATRIX[0]))

        return JOBS_RESOURCES_MATRIX, RESOURCE_EXECUTION_TIMES, availability_vector

    return None, None, None


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main execution function"""

    # Ask if user wants predefined example
    matrix, resource_times, availability_vector = use_predefined_example()

    if matrix is None:
        # Custom input mode
        print("\n" + "=" * 70)
        print("CUSTOM INPUT MODE")
        print("=" * 70)

        # Get dimensions
        num_jobs, num_resources = input_matrix_dimensions()

        # Get matrix
        matrix = input_matrix(num_jobs, num_resources)

        # Get resource execution times
        resource_times = input_resource_times(num_resources)

        # Get availability vector
        availability_vector = input_availability_vector(num_resources)

    # Display final configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\nJobs: {len(matrix)}")
    print(f"Resources: {len(matrix[0])}")
    print(f"\nDemand Matrix:")
    for i, row in enumerate(matrix):
        print(f"  J{i + 1}: {row}")
    print(f"\nResource execution times: {resource_times}")
    print(f"Availability vector: {availability_vector}")

    # Confirm before proceeding
    confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
    if confirm != "y":
        print("Configuration cancelled.")
        return

    # Choose mode
    choice = input("\nDo you want iterative vector reduction? (y/n): ").strip().lower()

    if choice == "y":
        # OPTIMIZATION MODE
        optimal_vector, schedule, makespan, history = iterative_vector_reduction(
            matrix, resource_times, availability_vector
        )

        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"  Starting vector: {availability_vector}")
        print(f"  Optimal vector:  {optimal_vector}")
        print(f"  Final MAKESPAN:  {makespan}")
        print("=" * 70)

        print_gantt_chart(schedule)
    else:
        # DETAILED MODE
        schedule, makespan = compute_schedule_detailed(
            matrix, resource_times, availability_vector
        )

        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"  MAKESPAN = {makespan}")
        print("=" * 70)

        print_gantt_chart(schedule)


if __name__ == "__main__":
    main()
