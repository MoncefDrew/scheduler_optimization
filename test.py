import matplotlib.pyplot as plt
from itertools import permutations

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
        # Automatic ordering: sort by (sum of tokens consumed × execution time)
        column_priority = []
        for col in range(self.num_resources):
            # Sum of tokens consumed by all jobs for this resource
            token_sum = sum(self.matrix[row][col] for row in range(self.num_jobs))
            # Multiply by execution time unit
            priority = token_sum * self.resource_times[col]
            column_priority.append((col, priority, token_sum))

        # Sort by priority (descending), then by column index (ascending) for ties
        column_priority.sort(key=lambda x: (-x[1], x[0]))
        return [col for col, _, _ in column_priority]

    def find_optimal_job_order(self, resource_col, jobs_with_data):
        """
        Find the optimal order of jobs within a resource to minimize makespan.

        Args:
            resource_col: The resource column index
            jobs_with_data: List of tuples (job_index, computing_time, release_time)

        Returns:
            Best job order and its makespan
        """
        if len(jobs_with_data) <= 1:
            return [j[0] for j in jobs_with_data], 0 if len(jobs_with_data) == 0 else jobs_with_data[0][1] + jobs_with_data[0][2]

        # Calculate number of permutations
        num_jobs = len(jobs_with_data)
        num_permutations = 1
        for i in range(1, num_jobs + 1):
            num_permutations *= i

        print(f"\n  Analyzing R{resource_col+1}: {num_jobs} jobs → {num_permutations} possible orderings")
        print(f"  {'─' * 70}")

        # Display job data (Pi and rj are same for all permutations)
        print(f"  Job data for R{resource_col+1}:")
        for job_idx, computing_time, release_time in jobs_with_data:
            print(f"    J{job_idx+1}: Pi={computing_time}, rj={release_time}")
        print()

        best_order = None
        best_makespan = float('inf')
        best_details = None
        all_results = []

        # Try all permutations
        for perm_idx, perm in enumerate(permutations(jobs_with_data), 1):
            # Calculate finish time for this permutation
            current_time = 0
            finish_times = []
            schedule_details = []

            for job_idx, computing_time, release_time in perm:
                # Job can start at max(current_time, release_time)
                start_time = max(current_time, release_time)
                finish_time = start_time + computing_time
                finish_times.append((job_idx, start_time, finish_time))
                schedule_details.append(f"J{job_idx+1}(start={start_time}, Cj={finish_time})")
                current_time = finish_time

            # Makespan is the finish time of the last job
            makespan = finish_times[-1][2]

            order_str = ' → '.join([f'J{j[0]+1}' for j in perm])
            result_str = f"  #{perm_idx}: {order_str:20} | {' | '.join(schedule_details)} | Makespan: {makespan}"
            all_results.append((makespan, result_str))

            if makespan < best_makespan:
                best_makespan = makespan
                best_order = [j[0] for j in perm]
                best_details = finish_times

        # Display all results
        for makespan, result_str in all_results:
            print(result_str)

        print(f"\n  ✓ BEST ORDER: {' → '.join([f'J{j+1}' for j in best_order])} with Makespan = {best_makespan}")
        print(f"  {'─' * 70}")

        return best_order, best_makespan, best_details

    def compute_schedule(self, available_resources):
        execution_order = self.get_execution_order()

        print("\nUsing automatic ordering (by tokens × execution time)...")
        print("\nPriority calculation:")
        for col in range(self.num_resources):
            token_sum = sum(self.matrix[row][col] for row in range(self.num_jobs))
            priority = token_sum * self.resource_times[col]
            print(f"  R{col+1}: {token_sum} tokens × {self.resource_times[col]} time = {priority}")

        print(f"\nResource Execution Order: {' → '.join([f'R{col+1}' for col in execution_order])}")

        # Track when each job finishes (initially all at time 0)
        job_finish_time = [0] * self.num_jobs
        schedule = []  # (resource, job, start, duration)

        print("\n" + "="*60)
        print("OPTIMAL JOB ORDERING WITHIN EACH RESOURCE")
        print("="*60)

        for col in execution_order:
            available_tokens = available_resources[col]
            base_time = self.resource_times[col]

            # Get jobs that use this resource
            jobs = [j for j in range(self.num_jobs) if self.matrix[j][col] > 0]

            required_tokens = len(jobs)
            if required_tokens > available_tokens:
                raise Exception(
                    f"Not enough resources for R{col+1}. "
                    f"Required {required_tokens}, Available {available_tokens}"
                )

            if len(jobs) == 0:
                continue

            # Prepare job data: (job_index, computing_time, release_time)
            jobs_with_data = []
            for j in jobs:
                utilization = self.matrix[j][col]
                computing_time = utilization * base_time  # Pi
                release_time = job_finish_time[j]  # rj (when job is ready)
                jobs_with_data.append((j, computing_time, release_time))

            # Find optimal job order for this resource
            optimal_order, resource_makespan, job_details = self.find_optimal_job_order(col, jobs_with_data)

            # Update schedule with optimal ordering
            for job_idx, start_time, finish_time in job_details:
                duration = finish_time - start_time
                schedule.append((col, job_idx, start_time, duration))
                job_finish_time[job_idx] = finish_time

        makespan = max(job_finish_time)
        return schedule, makespan, execution_order

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

    # Get availability vector
    user_input = input("Enter availability vector (R1 R2 R3): ")
    available_resources = list(map(int, user_input.split()))

    schedule, makespan, execution_order = scheduler.compute_schedule(available_resources)

    print("\n" + "="*60)
    print(f"FINAL RESULT: Makespan = {makespan}")
    print("="*60)

    scheduler.print_gantt_by_resource(schedule)
