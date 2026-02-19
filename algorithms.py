"""
Core Scheduling Algorithms
Contains Phase 1 (Sequencing) and Phase 2 (Conflict Resolution)
"""

from itertools import permutations


# =============================================================================
# PHASE 1: SEQUENCING AND DISPATCHING
# =============================================================================


def find_best_order_and_minimum_vector(matrix, resource_times, verbose=True):
    """
    PHASE 1 - Step 2: Find best dispatching order and minimum vector

    Returns:
        tuple: (best_order, min_vector)
    """
    num_jobs = len(matrix)
    num_resources = len(matrix[0])

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1: SEQUENCING AND DISPATCHING")
        print("=" * 70)

    # Step 2a: Find minimum vector
    if verbose:
        print("\n[Step 2a] Finding MINIMUM VECTOR:")

    min_vector = []
    for col in range(num_resources):
        jobs_needing = sum(1 for row in range(num_jobs) if matrix[row][col] > 0)
        min_vector.append(jobs_needing)
        if verbose:
            print(
                f"  R{col + 1}: {jobs_needing} jobs need it → minimum = {jobs_needing}"
            )

    if verbose:
        print(f"\n  Minimum vector: {min_vector}")

    # Step 2b: Find best dispatching order
    if verbose:
        print("\n[Step 2b] Finding BEST DISPATCHING ORDER:")

    column_priority = []
    for col in range(num_resources):
        token_sum = sum(matrix[row][col] for row in range(num_jobs))
        priority = token_sum * resource_times[col]
        column_priority.append((col, priority, token_sum))
        if verbose:
            print(
                f"  R{col + 1}: {token_sum} tokens × {resource_times[col]} time = {priority}"
            )

    column_priority.sort(key=lambda x: (-x[1], x[0]))
    best_order = [col for col, _, _ in column_priority]

    if verbose:
        print(
            f"\n  Best dispatching order: {' → '.join([f'R{r + 1}' for r in best_order])}"
        )

    return best_order, min_vector


# =============================================================================
# PHASE 2: CONFLICT RESOLUTION
# =============================================================================


def check_conflicts(matrix, available_vector, verbose=True):
    """
    PHASE 2 - Step 4: Check for conflicts when vector is lowered

    Returns:
        dict: Conflicts {resource_col: [list of job indices]}
    """
    num_jobs = len(matrix)
    num_resources = len(matrix[0])

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2: VECTOR REDUCTION AND CONFLICT RESOLUTION")
        print("=" * 70)
        print("\n[Step 4] CHECKING FOR CONFLICTS:")

    conflicts = {}

    for col in range(num_resources):
        jobs_needing = [j for j in range(num_jobs) if matrix[j][col] > 0]
        available = available_vector[col]

        if verbose:
            print(
                f"  R{col + 1}: {len(jobs_needing)} jobs need it, {available} unit(s) available",
                end="",
            )

        if len(jobs_needing) > available:
            conflicts[col] = jobs_needing
            if verbose:
                print(
                    f" → ⚠️ CONFLICT! Jobs competing: {[f'J{j + 1}' for j in jobs_needing]}"
                )
        else:
            if verbose:
                print(f" → ✓ No conflict")

    if verbose:
        if not conflicts:
            print("\n  No conflicts found! All jobs can run in parallel.")
        else:
            print(f"\n  Total conflicts: {len(conflicts)}")

    return conflicts


def resolve_conflict_parallel(
    resource_col,
    jobs,
    job_finish_times,
    available_units,
    matrix,
    resource_times,
    verbose=True,
):
    """
    PHASE 2 - Step 5-6: Resolve conflict using max workload processing
    Supports parallel execution when multiple units are available

    Returns:
        tuple: (best_order, best_details)
    """
    base_time = resource_times[resource_col]

    if verbose:
        print(f"\n  {'=' * 68}")
        print(f"  RESOLVING CONFLICT IN R{resource_col + 1}")
        print(f"  {'=' * 68}")
        print(f"  Question: Which job order is optimal for R{resource_col + 1}?")
        print(f"  Jobs: {[f'J{j + 1}' for j in jobs]}")
        print(f"  Available units: {available_units} (parallel execution enabled)")

    # Prepare job data
    jobs_with_data = []
    for j in jobs:
        utilization = matrix[j][resource_col]
        computing_time = utilization * base_time  # Pi
        release_time = job_finish_times[j]  # rj
        jobs_with_data.append((j, computing_time, release_time))

    if verbose:
        print(f"\n  [Step 5] MAX WORKLOAD PROCESSING:")
        print(f"  Job parameters:")
        for job_idx, computing_time, release_time in jobs_with_data:
            print(f"    J{job_idx + 1}: Pi = {computing_time}, rj = {release_time}")

        num_permutations = 1
        for i in range(1, len(jobs) + 1):
            num_permutations *= i

        print(f"\n  Testing all {num_permutations} possible orderings:")
        print(f"  {'-' * 68}")

    best_order = None
    best_makespan = float("inf")
    best_details = None

    # Try all permutations
    for perm_idx, perm in enumerate(permutations(jobs_with_data), 1):
        # Simulate parallel execution with available units
        unit_free_times = [0] * available_units
        finish_times = []
        schedule_details = []

        for job_idx, computing_time, release_time in perm:
            # Find the earliest available unit
            earliest_unit = min(
                range(available_units), key=lambda u: unit_free_times[u]
            )
            earliest_free_time = unit_free_times[earliest_unit]

            # Job starts when both: unit is free AND job is released
            start_time = max(earliest_free_time, release_time)
            finish_time = start_time + computing_time

            finish_times.append((job_idx, start_time, finish_time))
            schedule_details.append(f"J{job_idx + 1}(s={start_time},C={finish_time})")
            unit_free_times[earliest_unit] = finish_time

        makespan = max(ft[2] for ft in finish_times)

        marker = ""
        if makespan < best_makespan:
            best_makespan = makespan
            best_order = [j[0] for j in perm]
            best_details = finish_times
            marker = " ⭐"

        if verbose:
            order_str = " → ".join([f"J{j[0] + 1}" for j in perm])
            print(
                f"  #{perm_idx}: {order_str:20} | {' | '.join(schedule_details):45} | Makespan: {makespan}{marker}"
            )

    if verbose:
        print(f"  {'-' * 68}")
        print(f"\n  [Step 6] DECISION:")
        print(f"    ✓ Least makespan: {best_makespan}")
        print(f"    ✓ Best ordering: {' → '.join([f'J{j + 1}' for j in best_order])}")
        print(
            f"    ✓ DECISION: J{best_order[0] + 1} starts FIRST in R{resource_col + 1}"
        )
        print(f"  {'=' * 68}")

    return best_order, best_details
