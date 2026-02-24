import matplotlib.pyplot as plt
from itertools import permutations
import random


class ResourceScheduler:
    def __init__(self, matrix=None, resource_times=None):
        # Jobs × Resources matrix
        if matrix is None:
            self.matrix = [
                [1, 2, 1],  # J1
                [1, 1, 1],  # J2
                [1, 1, 2],  # J3
            ]
        else:
            self.matrix = matrix

        # Execution time per single token
        if resource_times is None:
            self.resource_times = [3, 5, 4]
        else:
            self.resource_times = resource_times

        self.num_jobs = len(self.matrix)
        self.num_resources = len(self.matrix[0])

    # ========================================================================
    # PHASE 1: RESOURCE WORKLOAD CALCULATION
    # ========================================================================

    def calculate_resource_workload(self):
        """
        Calculate workload for each resource: Σ(tokens) × execution_time
        Returns resources sorted by workload (descending)
        """
        print("\n" + "=" * 70)
        print("PHASE 1: RESOURCE WORKLOAD CALCULATION")
        print("=" * 70)

        workloads = []
        for col in range(self.num_resources):
            token_sum = sum(self.matrix[row][col] for row in range(self.num_jobs))
            workload = token_sum * self.resource_times[col]
            workloads.append((col, workload, token_sum))

        # Sort by workload (descending)
        workloads.sort(key=lambda x: (-x[1], x[0]))

        print("\nWorkload = Σ(tokens) × execution_time")
        print(
            f"\n{'Resource':<12} {'Tokens':<10} {'Exec Time':<12} {'Workload':<12} {'Rank'}"
        )
        print("-" * 70)

        for rank, (col, workload, tokens) in enumerate(workloads, 1):
            print(
                f"R{col + 1:<11} {tokens:<10} {self.resource_times[col]:<12} {workload:<12} #{rank}"
            )

        resource_order = [col for col, _, _ in workloads]
        print(
            f"\nResource execution order (by workload): {' → '.join([f'R{r + 1}' for r in resource_order])}"
        )

        return resource_order

    # ========================================================================
    # PHASE 2: MATRIX METHOD FOR JOB ORDERING WITHIN RESOURCES
    # ========================================================================

    def apply_matrix_method_for_resource(
        self, resource_col, jobs, job_finish_times, available_units, verbose=True
    ):
        """
        Apply matrix method to determine optimal job ordering within a resource
        Tests all permutations to find the one with minimum makespan
        """
        base_time = self.resource_times[resource_col]

        if verbose:
            print(f"\n  {'=' * 68}")
            print(f"  APPLYING MATRIX METHOD - R{resource_col + 1}")
            print(f"  {'=' * 68}")
            print(f"  Jobs at this resource: {[f'J{j + 1}' for j in jobs]}")
            print(f"  Available units: {available_units}")

        # Prepare job data
        jobs_with_data = []
        for j in jobs:
            utilization = self.matrix[j][resource_col]
            computing_time = utilization * base_time  # Pi
            release_time = job_finish_times[j]  # rj (when job arrives at this resource)
            jobs_with_data.append((j, computing_time, release_time))

        if verbose:
            print(f"\n  Job parameters:")
            for job_idx, computing_time, release_time in jobs_with_data:
                print(
                    f"    J{job_idx + 1}: Pi={computing_time}, rj (arrival)={release_time}"
                )

            num_permutations = 1
            for i in range(1, len(jobs) + 1):
                num_permutations *= i

            print(f"\n  Testing {num_permutations} orderings to find optimal:")
            print(f"  {'-' * 68}")

        best_order = None
        best_makespan = float("inf")
        best_details = None

        # Try all permutations
        for perm_idx, perm in enumerate(permutations(jobs_with_data), 1):
            unit_free_times = [0] * available_units
            finish_times = []

            for job_idx, computing_time, release_time in perm:
                # Find earliest available unit
                earliest_unit = min(
                    range(available_units), key=lambda u: unit_free_times[u]
                )
                earliest_free_time = unit_free_times[earliest_unit]

                # Job starts when: unit is free AND job has arrived
                start_time = max(earliest_free_time, release_time)
                finish_time = start_time + computing_time

                finish_times.append((job_idx, start_time, finish_time))
                unit_free_times[earliest_unit] = finish_time

            makespan = max(ft[2] for ft in finish_times)
            order_str = " → ".join([f"J{j[0] + 1}" for j in perm])

            marker = ""
            if makespan < best_makespan:
                best_makespan = makespan
                best_order = [j[0] for j in perm]
                best_details = finish_times
                marker = " ⭐"

            if verbose:
                print(f"  #{perm_idx}: {order_str:20} | Makespan: {makespan}{marker}")

        if verbose:
            print(f"  {'-' * 68}")
            print(f"  ✓ OPTIMAL ORDER: {' → '.join([f'J{j + 1}' for j in best_order])}")
            print(f"  ✓ Makespan for this resource: {best_makespan}")

        return best_order, best_details

    # ========================================================================
    # SCHEDULING WITH JOB PATHS
    # ========================================================================

    def compute_schedule_with_job_paths(
        self, available_vector, job_paths, scenario_name="", verbose=True
    ):
        """
        Compute schedule where each job follows its predefined path through resources
        Resources are processed in order of workload (already calculated)
        Matrix method is applied within each resource to find optimal job ordering
        """
        if verbose:
            print("\n" + "=" * 70)
            print(f"PHASE 2: SCHEDULING - {scenario_name}")
            print("=" * 70)
            print("\nJob routing paths:")
            for job_idx, path in job_paths.items():
                path_str = " → ".join([f"R{r + 1}" for r in path])
                print(f"  J{job_idx + 1}: {path_str}")

        # Calculate resource execution order by workload
        resource_order = (
            self.calculate_resource_workload()
            if verbose
            else self._get_resource_order_silent()
        )

        job_finish_times = [0] * self.num_jobs
        schedule = []

        # Process resources in workload order
        for resource_col in resource_order:
            # Find which jobs are at this resource based on their paths
            jobs_at_resource = []
            for job_idx, path in job_paths.items():
                if resource_col in path:
                    jobs_at_resource.append(job_idx)

            if len(jobs_at_resource) == 0:
                continue

            available_units = available_vector[resource_col]

            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Processing R{resource_col + 1}")
                print(f"{'=' * 70}")
                print(
                    f"Jobs visiting this resource: {[f'J{j + 1}' for j in jobs_at_resource]}"
                )

            # Apply matrix method to find optimal job ordering
            if len(jobs_at_resource) > 1:
                optimal_order, job_details = self.apply_matrix_method_for_resource(
                    resource_col,
                    jobs_at_resource,
                    job_finish_times,
                    available_units,
                    verbose=verbose,
                )

                for job_idx, start_time, finish_time in job_details:
                    duration = finish_time - start_time
                    schedule.append((resource_col, job_idx, start_time, duration))
                    job_finish_times[job_idx] = finish_time
            else:
                # Single job at this resource
                j = jobs_at_resource[0]
                utilization = self.matrix[j][resource_col]
                duration = utilization * self.resource_times[resource_col]
                start_time = job_finish_times[j]

                schedule.append((resource_col, j, start_time, duration))
                job_finish_times[j] = start_time + duration

                if verbose:
                    print(f"\n  Only J{j + 1} at this resource")
                    print(
                        f"  Starts at {start_time}, finishes at {job_finish_times[j]}"
                    )

        makespan = max(job_finish_times) if job_finish_times else 0

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"FINAL MAKESPAN: {makespan}")
            print(f"{'=' * 70}")

        return schedule, makespan

    def _get_resource_order_silent(self):
        """Get resource order without printing"""
        workloads = []
        for col in range(self.num_resources):
            token_sum = sum(self.matrix[row][col] for row in range(self.num_jobs))
            workload = token_sum * self.resource_times[col]
            workloads.append((col, workload))
        workloads.sort(key=lambda x: (-x[1], x[0]))
        return [col for col, _ in workloads]

    # ========================================================================
    # PHASE 3: VECTOR OPTIMIZATION
    # ========================================================================

    def calculate_minimum_vector(self, job_paths):
        """
        Calculate minimum vector based on maximum concurrent jobs at each resource
        """
        min_vector = []

        for col in range(self.num_resources):
            # Count how many jobs use this resource
            jobs_using = sum(1 for job_idx, path in job_paths.items() if col in path)

            if jobs_using == 0:
                min_vector.append(0)
            else:
                # Find max concurrent based on position in paths
                position_groups = {}
                for job_idx, path in job_paths.items():
                    if col in path:
                        pos = path.index(col)
                        if pos not in position_groups:
                            position_groups[pos] = []
                        position_groups[pos].append(job_idx)

                max_concurrent = (
                    max(len(group) for group in position_groups.values())
                    if position_groups
                    else 1
                )
                min_vector.append(max_concurrent)

        return min_vector

    def optimize_vector(self, initial_vector, job_paths, scenario_name=""):
        """
        PHASE 3: Optimize vector by iteratively reducing while maintaining or improving makespan
        """
        print("\n" + "=" * 70)
        print(f"PHASE 3: VECTOR OPTIMIZATION - {scenario_name}")
        print("=" * 70)

        # Calculate minimum vector
        min_vector = self.calculate_minimum_vector(job_paths)

        print(f"\nMinimum vector calculation:")
        for col in range(self.num_resources):
            jobs_using = sum(1 for job_idx, path in job_paths.items() if col in path)
            if jobs_using > 0:
                print(
                    f"  R{col + 1}: {jobs_using} jobs use it → min = {min_vector[col]}"
                )
            else:
                print(f"  R{col + 1}: No jobs use it → min = 0")

        print(f"\nMinimum vector: {min_vector}")

        current_vector = initial_vector.copy()
        schedule, current_makespan = self.compute_schedule_with_job_paths(
            current_vector, job_paths, verbose=False
        )

        print(f"Starting vector: {current_vector}")
        print(f"Starting makespan: {current_makespan}")

        history = [(current_vector.copy(), current_makespan)]

        print("\nOptimizing vector...")
        iteration = 0
        improved = True

        while improved:
            improved = False

            for col in range(self.num_resources):
                if current_vector[col] > min_vector[col]:
                    iteration += 1
                    test_vector = current_vector.copy()
                    test_vector[col] -= 1

                    test_schedule, test_makespan = self.compute_schedule_with_job_paths(
                        test_vector, job_paths, verbose=False
                    )

                    if test_makespan > current_makespan:
                        print(
                            f"  Iter {iteration}: R{col + 1} {current_vector[col]}→{test_vector[col]} "
                            f"❌ Makespan {current_makespan}→{test_makespan}"
                        )
                    elif test_makespan == current_makespan:
                        print(
                            f"  Iter {iteration}: R{col + 1} {current_vector[col]}→{test_vector[col]} "
                            f"➡️  Same ({test_makespan})"
                        )
                        current_vector = test_vector
                        history.append((current_vector.copy(), current_makespan))
                        improved = True
                        break
                    else:
                        print(
                            f"  Iter {iteration}: R{col + 1} {current_vector[col]}→{test_vector[col]} "
                            f"✓ Makespan {current_makespan}→{test_makespan}"
                        )
                        current_vector = test_vector
                        current_makespan = test_makespan
                        history.append((current_vector.copy(), current_makespan))
                        improved = True
                        break

        print(f"\nOptimal vector: {current_vector}")
        print(f"Optimal makespan: {current_makespan}")

        if current_vector == min_vector:
            print("✓ Reached minimum vector")

        final_schedule, _ = self.compute_schedule_with_job_paths(
            current_vector, job_paths, verbose=False
        )

        return current_vector, final_schedule, current_makespan, history

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def print_gantt_by_resource(self, schedule, title="Gantt Chart - Optimal Schedule"):
        """Generate Gantt chart"""
        if schedule is None or len(schedule) == 0:
            print("Cannot generate Gantt chart - no valid schedule")
            return

        resource_tasks = {}
        for resource, job, start, duration in schedule:
            if resource not in resource_tasks:
                resource_tasks[resource] = []
            resource_tasks[resource].append((job, start, duration))

        y_labels = []
        y_positions = {}
        y_pos = 0
        resource_boundaries = []

        for resource in sorted(resource_tasks.keys()):
            tasks = resource_tasks[resource]
            tasks.sort(key=lambda x: x[1])

            for job, start, duration in tasks:
                label = f"R{resource + 1} - J{job + 1}"
                y_labels.append(label)
                y_positions[label] = y_pos
                y_pos += 1

            resource_boundaries.append(y_pos - 0.5)
            y_pos += 0.8

        plt.figure(figsize=(14, max(6, len(y_labels) * 0.6)))

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
                height=0.6,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.85,
            )

            plt.text(
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
            plt.axhline(
                y=boundary, color="gray", linestyle="--", linewidth=2, alpha=0.5
            )

        y_tick_positions = [y_positions[label] for label in y_labels]
        plt.yticks(y_tick_positions, y_labels, fontsize=10)
        plt.xlabel("Time", fontsize=12, fontweight="bold")
        plt.ylabel("Resources (with Jobs)", fontsize=12, fontweight="bold")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.grid(axis="x", alpha=0.3, linestyle=":", linewidth=0.8)
        plt.tight_layout()
        plt.show()


# ============================================================================
# INPUT FUNCTIONS
# ============================================================================


def input_matrix():
    """Input matrix from user"""
    print("\n" + "=" * 70)
    print("INPUT MATRIX")
    print("=" * 70)

    num_jobs = int(input("Number of jobs: "))
    num_resources = int(input("Number of resources: "))

    print(f"\nEnter {num_jobs}×{num_resources} matrix (rows separated by semicolons)")
    print("Example: '1 2 1; 1 1 1; 1 1 2'")

    matrix_input = input("Matrix: ")
    rows = matrix_input.split(";")

    matrix = []
    for row_str in rows:
        row = list(map(int, row_str.strip().split()))
        matrix.append(row)

    return matrix, num_jobs, num_resources


def input_resource_times(num_resources):
    """Input resource execution times"""
    print(f"\nEnter execution times for {num_resources} resources (space-separated):")
    times = list(map(float, input("Times: ").split()))
    return times


def input_job_paths(num_jobs, num_resources, matrix):
    """
    Input job routing paths - which resources each job visits in order
    """
    print("\n" + "=" * 70)
    print("JOB ROUTING PATHS (Job Sequencing)")
    print("=" * 70)
    print("Define the path each job takes through resources")
    print("Format: Resource numbers (1-based) in the order the job visits them")
    print("Example: '3 1 2' means job visits R3, then R1, then R2")

    job_paths = {}

    for job_idx in range(num_jobs):
        print(f"\nJ{job_idx + 1}:")
        # Show which resources this job needs (from matrix)
        needed = [r + 1 for r in range(num_resources) if matrix[job_idx][r] > 0]
        print(f"  Resources needed (from matrix): {needed}")

        while True:
            try:
                path_input = input(f"  Path for J{job_idx + 1}: ").strip()
                path = [int(x) - 1 for x in path_input.split()]

                # Validate
                if not all(0 <= r < num_resources for r in path):
                    print(f"  Error: Resources must be between 1 and {num_resources}")
                    continue

                if len(path) != len(set(path)):
                    print("  Error: Duplicate resources in path")
                    continue

                # Check if path includes all needed resources
                needed_set = set([r - 1 for r in needed])
                path_set = set(path)
                if needed_set != path_set:
                    print(f"  Warning: Path should include exactly {needed}")
                    confirm = input("  Continue anyway? (y/n): ").strip().lower()
                    if confirm != "y":
                        continue

                job_paths[job_idx] = path
                print(f"  ✓ J{job_idx + 1}: {' → '.join([f'R{r + 1}' for r in path])}")
                break
            except ValueError:
                print("  Error: Invalid format")

    return job_paths


def generate_random_job_paths(num_jobs, num_resources, matrix):
    """Generate random job paths based on matrix"""
    job_paths = {}
    for job_idx in range(num_jobs):
        # Get resources this job needs
        needed = [r for r in range(num_resources) if matrix[job_idx][r] > 0]
        # Random order
        random.shuffle(needed)
        job_paths[job_idx] = needed
    return job_paths


def input_availability_vector(num_resources):
    """Input availability vector"""
    print("\n" + "=" * 70)
    print("AVAILABILITY VECTOR")
    print("=" * 70)
    print(f"Enter {num_resources} values (space-separated):")
    vector = list(map(int, input("Vector: ").split()))
    return vector


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("FLEXIBLE JOB SHOP SCHEDULER WITH JOB ROUTING")
    print("=" * 70)

    # Step 1: Get matrix
    use_default = input("\nUse default 3×3 matrix? (y/n): ").strip().lower()

    if use_default == "y":
        matrix = [[1, 2, 1], [1, 1, 1], [1, 1, 2]]
        num_jobs, num_resources = 3, 3
        resource_times = [3, 5, 4]
    else:
        matrix, num_jobs, num_resources = input_matrix()
        resource_times = input_resource_times(num_resources)

    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print("Matrix:")
    for i, row in enumerate(matrix):
        print(f"  J{i + 1}: {row}")
    print(f"Resource times: {resource_times}")

    # Create scheduler
    scheduler = ResourceScheduler(matrix, resource_times)

    # Step 2: Get initial availability vector
    initial_vector = input_availability_vector(num_resources)

    # Step 3: Create scenarios with different job paths
    print("\n" + "=" * 70)
    print("SCENARIO CREATION")
    print("=" * 70)
    print("Each scenario has different job routing paths")

    scenarios = []

    # Scenario 1: User-defined
    print("\nSCENARIO 1 (Base scenario):")
    job_paths_1 = input_job_paths(num_jobs, num_resources, matrix)
    scenarios.append({"name": "Base Scenario", "job_paths": job_paths_1})

    # Additional scenarios
    num_scenarios = int(input("\nHow many additional scenarios? (0-5): "))

    for i in range(num_scenarios):
        scenario_num = i + 2
        print(f"\nSCENARIO {scenario_num}:")
        print("1. Manual input")
        print("2. Random generation")

        choice = input("Choice (1-2): ").strip()

        if choice == "1":
            job_paths = input_job_paths(num_jobs, num_resources, matrix)
            scenarios.append(
                {"name": f"Scenario {scenario_num}", "job_paths": job_paths}
            )
        else:
            job_paths = generate_random_job_paths(num_jobs, num_resources, matrix)
            print("\n✓ Random paths generated:")
            for j, p in job_paths.items():
                print(f"  J{j + 1}: {' → '.join([f'R{r + 1}' for r in p])}")
            scenarios.append(
                {"name": f"Random Scenario {scenario_num}", "job_paths": job_paths}
            )

    # Step 4: Run all scenarios
    print("\n" + "=" * 70)
    print("RUNNING ALL SCENARIOS")
    print("=" * 70)

    results = []

    for idx, scenario in enumerate(scenarios, 1):
        print("\n" + "=" * 70)
        print(f"SCENARIO {idx}: {scenario['name']}")
        print("=" * 70)

        # Run initial schedule (shows workload calculation and matrix method)
        schedule, makespan = scheduler.compute_schedule_with_job_paths(
            initial_vector, scenario["job_paths"], scenario["name"], verbose=True
        )

        # Optimize vector
        optimal_vector, opt_schedule, opt_makespan, history = scheduler.optimize_vector(
            initial_vector.copy(), scenario["job_paths"], scenario["name"]
        )

        result = {
            "num": idx,
            "name": scenario["name"],
            "job_paths": scenario["job_paths"],
            "initial_vector": initial_vector.copy(),
            "optimal_vector": optimal_vector,
            "initial_makespan": makespan,
            "optimal_makespan": opt_makespan,
            "schedule": opt_schedule,
        }
        results.append(result)

        print(f"\n{'=' * 70}")
        print(f"SCENARIO {idx} SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Initial: {initial_vector} → Makespan {makespan}")
        print(f"  Optimal: {optimal_vector} → Makespan {opt_makespan}")
        print(f"  Improvement: {makespan - opt_makespan} time units")

        show_chart = input("\nShow Gantt chart? (y/n): ").strip().lower()
        if show_chart == "y":
            scheduler.print_gantt_by_resource(
                opt_schedule,
                f"Scenario {idx}: {scenario['name']} - Makespan={opt_makespan}",
            )

    # Final comparison
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("FINAL SCENARIO COMPARISON")
        print("=" * 70)
        print(f"\n{'Scenario':<20} {'Optimal Vector':<20} {'Makespan':<12}")
        print("-" * 60)

        for r in results:
            print(
                f"{r['name']:<20} {str(r['optimal_vector']):<20} {r['optimal_makespan']:<12}"
            )
            print(f"  Job paths:")
            for j, p in r["job_paths"].items():
                print(f"    J{j + 1}: {' → '.join([f'R{rr + 1}' for rr in p])}")

        best = min(results, key=lambda x: x["optimal_makespan"])
        print("\n" + "=" * 70)
        print(
            f"✓ BEST SCENARIO: {best['name']} (Makespan = {best['optimal_makespan']})"
        )
        print(f"  Optimal vector: {best['optimal_vector']}")
        print("=" * 70)


if __name__ == "__main__":
    main()
