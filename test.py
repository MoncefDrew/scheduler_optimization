import matplotlib.pyplot as plt

class ResourceScheduler:
    def __init__(self):
        # Jobs × Resources matrix
        self.matrix = [
            [2, 1, 0],  # J1
            [0, 2, 3],  # J2
            [1, 1, 1]   # J3
        ]
        # Execution time per single token
        self.resource_times = [3, 5, 4]
        self.num_jobs = len(self.matrix)
        self.num_resources = len(self.matrix[0])

    def get_execution_order(self):
        column_usage = []
        for col in range(self.num_resources):
            usage = sum(self.matrix[row][col] for row in range(self.num_jobs))
            column_usage.append((col, usage))
        column_usage.sort(key=lambda x: (-x[1], x[0]))
        return [col for col, _ in column_usage]

    def compute_schedule(self, available_resources):
        execution_order = self.get_execution_order()
        job_finish_time = [0] * self.num_jobs
        schedule = []  # (resource, job, start, duration)

        for col in execution_order:
            available_tokens = available_resources[col]
            base_time = self.resource_times[col]

            jobs = [j for j in range(self.num_jobs)
                    if self.matrix[j][col] > 0]

            required_tokens = len(jobs)
            if required_tokens > available_tokens:
                raise Exception(
                    f"Not enough resources for R{col+1}. "
                    f"Required {required_tokens}, Available {available_tokens}"
                )

            # Parallel execution
            for j in jobs:
                utilization = self.matrix[j][col]
                duration = utilization * base_time
                start_time = job_finish_time[j]
                schedule.append((col, j, start_time, duration))
                job_finish_time[j] += duration

        makespan = max(job_finish_time)
        return schedule, makespan

    def print_gantt_by_resource(self, schedule):
        # Group tasks by resource
        resource_tasks = {}
        for resource, job, start, duration in schedule:
            if resource not in resource_tasks:
                resource_tasks[resource] = []
            resource_tasks[resource].append((job, start, duration))

        # Create y-axis labels - one line per job per resource
        # Add spacing between resources
        y_labels = []
        y_positions = {}
        y_pos = 0
        resource_boundaries = []  # Track where each resource section ends

        # Sort resources by their index
        for resource in sorted(resource_tasks.keys()):
            tasks = resource_tasks[resource]
            # Sort tasks by job number for consistent ordering
            tasks.sort(key=lambda x: x[0])

            for job, start, duration in tasks:
                label = f"R{resource+1} - J{job+1}"
                y_labels.append(label)
                y_positions[label] = y_pos
                y_pos += 1

            # Add spacing between resources (0.8 units)
            resource_boundaries.append(y_pos - 0.5)
            y_pos += 0.8

        # Create the plot
        plt.figure(figsize=(10, max(6, len(y_labels) * 0.5 + len(resource_tasks) * 0.4)))

        # Plot each task
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for resource, job, start, duration in schedule:
            label = f"R{resource+1} - J{job+1}"
            y_position = y_positions[label]

            # Use different color for each job
            color = colors[job % len(colors)]

            plt.barh(
                y_position,
                duration,
                left=start,
                height=0.5,  # Decreased from 0.8 to 0.5
                color=color,
                edgecolor='black',
                linewidth=0.5
            )

            # Add text label inside bar
            plt.text(
                start + duration / 2,
                y_position,
                f"J{job+1}",
                ha='center',
                va='center',
                fontweight='bold',
                color='white',
                fontsize=9
            )

        # Draw horizontal lines to separate resources
        for boundary in resource_boundaries[:-1]:  # Don't draw after last resource
            plt.axhline(y=boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        # Get y-tick positions (only for actual labels, not spacing)
        y_tick_positions = [y_positions[label] for label in y_labels]

        plt.yticks(y_tick_positions, y_labels)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Resources (with Jobs)", fontsize=12)
        plt.title("Gantt Chart (Grouped by Resources)", fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    scheduler = ResourceScheduler()
    user_input = input("Enter availability vector (R1 R2 R3): ")
    available_resources = list(map(int, user_input.split()))

    schedule, makespan = scheduler.compute_schedule(available_resources)
    print("Makespan:", makespan)
    scheduler.print_gantt_by_resource(schedule)
