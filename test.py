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

    # ========================================================================
    # PHASE 1: SEQUENCING AND DISPATCHING
    # ========================================================================

    def phase1_find_best_order_and_minimum_vector(self):
        """
        PHASE 1 - Step 2: Find best dispatching order and minimum vector
        """
        print("\n" + "="*70)
        print("PHASE 1: SEQUENCING AND DISPATCHING")
        print("="*70)

        # Step 2a: Find minimum vector (based on jobs needing each resource)
        print("\n[Step 2a] Finding MINIMUM VECTOR:")
        min_vector = []
        for col in range(self.num_resources):
            jobs_needing = sum(1 for row in range(self.num_jobs) if self.matrix[row][col] > 0)
            min_vector.append(jobs_needing)
            print(f"  R{col+1}: {jobs_needing} jobs need it → minimum = {jobs_needing}")

        print(f"\n  Minimum vector: {min_vector}")

        # Step 2b: Find best dispatching order (by priority)
        print("\n[Step 2b] Finding BEST DISPATCHING ORDER:")
        column_priority = []
        for col in range(self.num_resources):
            token_sum = sum(self.matrix[row][col] for row in range(self.num_jobs))
            priority = token_sum * self.resource_times[col]
            column_priority.append((col, priority, token_sum))
            print(f"  R{col+1}: {token_sum} tokens × {self.resource_times[col]} time = {priority}")

        column_priority.sort(key=lambda x: (-x[1], x[0]))
        best_order = [col for col, _, _ in column_priority]

        print(f"\n  Best dispatching order: {' → '.join([f'R{r+1}' for r in best_order])}")

        return best_order, min_vector

    # ========================================================================
    # PHASE 2: VECTOR REDUCTION AND CONFLICT RESOLUTION
    # ========================================================================

    def phase2_check_conflicts(self, available_vector):
        """
        PHASE 2 - Step 4: Check for conflicts when vector is lowered
        """
        print("\n" + "="*70)
        print("PHASE 2: VECTOR REDUCTION AND CONFLICT RESOLUTION")
        print("="*70)

        print("\n[Step 4] CHECKING FOR CONFLICTS:")
        conflicts = {}

        for col in range(self.num_resources):
            jobs_needing = [j for j in range(self.num_jobs) if self.matrix[j][col] > 0]
            available = available_vector[col]

            print(f"  R{col+1}: {len(jobs_needing)} jobs need it, {available} unit(s) available", end="")

            if len(jobs_needing) > available:
                conflicts[col] = jobs_needing
                print(f" → ⚠️ CONFLICT! Jobs competing: {[f'J{j+1}' for j in jobs_needing]}")
            else:
                print(f" → ✓ No conflict")

        if not conflicts:
            print("\n  No conflicts found! All jobs can run in parallel.")
        else:
            print(f"\n  Total conflicts: {len(conflicts)}")

        return conflicts

    def phase2_resolve_conflict_max_workload(self, resource_col, jobs, job_finish_times):
        """
        PHASE 2 - Step 5-6: Resolve conflict using max workload processing
        Find the least makespan and decide which job starts first
        """
        base_time = self.resource_times[resource_col]

        print(f"\n  {'='*68}")
        print(f"  RESOLVING CONFLICT IN R{resource_col+1}")
        print(f"  {'='*68}")
        print(f"  Question: Which job starts first in R{resource_col+1}?")
        print(f"  Jobs competing: {[f'J{j+1}' for j in jobs]}")

        # Prepare job data
        jobs_with_data = []
        for j in jobs:
            utilization = self.matrix[j][resource_col]
            computing_time = utilization * base_time  # Pi
            release_time = job_finish_times[j]  # rj
            jobs_with_data.append((j, computing_time, release_time))

        print(f"\n  [Step 5] MAX WORKLOAD PROCESSING:")
        print(f"  Job parameters:")
        for job_idx, computing_time, release_time in jobs_with_data:
            print(f"    J{job_idx+1}: Pi (computing time) = {computing_time}, rj (release time) = {release_time}")

        num_permutations = 1
        for i in range(1, len(jobs) + 1):
            num_permutations *= i

        print(f"\n  Testing all {num_permutations} possible orderings:")
        print(f"  {'-'*68}")

        best_order = None
        best_makespan = float('inf')
        best_details = None

        # Try all permutations
        for perm_idx, perm in enumerate(permutations(jobs_with_data), 1):
            current_time = 0
            finish_times = []
            schedule_details = []

            for job_idx, computing_time, release_time in perm:
                start_time = max(current_time, release_time)
                finish_time = start_time + computing_time
                finish_times.append((job_idx, start_time, finish_time))
                schedule_details.append(f"J{job_idx+1}(Cj={finish_time})")
                current_time = finish_time

            makespan = finish_times[-1][2]
            order_str = ' → '.join([f'J{j[0]+1}' for j in perm])

            marker = ""
            if makespan < best_makespan:
                best_makespan = makespan
                best_order = [j[0] for j in perm]
                best_details = finish_times
                marker = " ⭐"

            print(f"  #{perm_idx}: {order_str:20} | {' | '.join(schedule_details):40} | Makespan: {makespan}{marker}")

        print(f"  {'-'*68}")
        print(f"\n  [Step 6] DECISION:")
        print(f"    ✓ Least makespan: {best_makespan}")
        print(f"    ✓ Best ordering: {' → '.join([f'J{j+1}' for j in best_order])}")
        print(f"    ✓ DECISION: J{best_order[0]+1} starts FIRST in R{resource_col+1}")
        print(f"  {'='*68}")

        return best_order, best_details

    def compute_schedule(self, available_vector):
        """
        Main algorithm: Execute Phase 1 and Phase 2
        """
        print("\n" + "="*70)
        print("RESOURCE SCHEDULING ALGORITHM")
        print("="*70)

        # Display input
        print("\n[Step 1] INPUT DATA:")
        print(f"  Matrix (Jobs × Resources):")
        for i, row in enumerate(self.matrix):
            print(f"    J{i+1}: {row}")
        print(f"\n  Resource execution times: {self.resource_times}")
        print(f"  Available vector: {available_vector}")

        # PHASE 1: Sequencing and Dispatching
        best_order, min_vector = self.phase1_find_best_order_and_minimum_vector()

        # Step 3: Check if vector can be lowered
        print("\n[Step 3] LOWERING VECTOR VALUES:")
        print(f"  Current vector:  {available_vector}")
        print(f"  Minimum vector:  {min_vector}")

        can_schedule = all(available_vector[i] >= min_vector[i] for i in range(self.num_resources))

        if not can_schedule:
            print("\n  ❌ ERROR: Vector is BELOW minimum!")
            print("  Cannot schedule - insufficient resources.")
            return None, None
        else:
            print("\n  ✓ Vector is valid (at or above minimum)")

        # PHASE 2: Conflict resolution
        conflicts = self.phase2_check_conflicts(available_vector)

        # Solve schedule
        print("\n[Step 5-6] SCHEDULING WITH MAX WORKLOAD PROCESSING:")

        job_finish_times = [0] * self.num_jobs
        schedule = []

        for col in best_order:
            jobs = [j for j in range(self.num_jobs) if self.matrix[j][col] > 0]

            if len(jobs) == 0:
                continue

            if col in conflicts:
                # Conflict exists: resolve it
                optimal_order, job_details = self.phase2_resolve_conflict_max_workload(col, jobs, job_finish_times)

                # Update schedule
                for job_idx, start_time, finish_time in job_details:
                    duration = finish_time - start_time
                    schedule.append((col, job_idx, start_time, duration))
                    job_finish_times[job_idx] = finish_time
            else:
                # No conflict: parallel execution
                print(f"\n  R{col+1}: No conflict - parallel execution")
                for j in jobs:
                    utilization = self.matrix[j][col]
                    duration = utilization * self.resource_times[col]
                    start_time = job_finish_times[j]
                    schedule.append((col, j, start_time, duration))
                    job_finish_times[j] = start_time + duration
                    print(f"    J{j+1}: starts at {start_time}, finishes at {job_finish_times[j]}")

        makespan = max(job_finish_times)
        return schedule, makespan

    def print_gantt_by_resource(self, schedule):
        """Generate Gantt chart"""
        if schedule is None:
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
                label = f"R{resource+1} - J{job+1}"
                y_labels.append(label)
                y_positions[label] = y_pos
                y_pos += 1

            resource_boundaries.append(y_pos - 0.5)
            y_pos += 0.8

        # Create plot
        plt.figure(figsize=(12, max(6, len(y_labels) * 0.5 + len(resource_tasks) * 0.4)))

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for resource, job, start, duration in schedule:
            label = f"R{resource+1} - J{job+1}"
            y_position = y_positions[label]
            color = colors[job % len(colors)]

            plt.barh(y_position, duration, left=start, height=0.5,
                    color=color, edgecolor='black', linewidth=0.5)

            plt.text(start + duration / 2, y_position, f"J{job+1}",
                    ha='center', va='center', fontweight='bold',
                    color='white', fontsize=9)

        # Separator lines
        for boundary in resource_boundaries[:-1]:
            plt.axhline(y=boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        y_tick_positions = [y_positions[label] for label in y_labels]
        plt.yticks(y_tick_positions, y_labels)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Resources (with Jobs)", fontsize=12)
        plt.title("Gantt Chart - Optimal Schedule", fontsize=14)
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
    available_vector = list(map(int, user_input.split()))

    schedule, makespan = scheduler.compute_schedule(available_vector)

    if schedule is not None:
        print("\n" + "="*70)
        print("FINAL RESULT")
        print("="*70)
        print(f"  MAKESPAN = {makespan}")
        print("="*70)

        scheduler.print_gantt_by_resource(schedule)
