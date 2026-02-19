"""
Scheduler and Optimizer
Contains main scheduling logic and iterative vector reduction
"""

from algorithms import (
    find_best_order_and_minimum_vector,
    check_conflicts,
    resolve_conflict_parallel,
)


# =============================================================================
# SCHEDULER
# =============================================================================


def compute_schedule(matrix, resource_times, available_vector, verbose=False):
    """
    Compute schedule for a given vector with parallel execution support.

    Returns:
        tuple: (schedule, makespan)
            schedule: List of tuples (resource, job, start, duration)
            makespan: Total completion time
    """
    num_jobs = len(matrix)
    num_resources = len(matrix[0])

    # Get best resource execution order
    best_order, _ = find_best_order_and_minimum_vector(
        matrix, resource_times, verbose=False
    )

    # Identify conflicts
    conflicts = {}
    for col in range(num_resources):
        jobs_needing = [j for j in range(num_jobs) if matrix[j][col] > 0]
        if len(jobs_needing) > available_vector[col]:
            conflicts[col] = jobs_needing

    # Solve schedule with parallel execution
    job_finish_times = [0] * num_jobs
    schedule = []

    for col in best_order:
        jobs = [j for j in range(num_jobs) if matrix[j][col] > 0]

        if len(jobs) == 0:
            continue

        available_units = available_vector[col]

        if col in conflicts or len(jobs) > 1:
            # Use optimal ordering with parallel execution
            optimal_order, job_details = resolve_conflict_parallel(
                col,
                jobs,
                job_finish_times,
                available_units,
                matrix,
                resource_times,
                verbose=False,
            )
            for job_idx, start_time, finish_time in job_details:
                duration = finish_time - start_time
                schedule.append((col, job_idx, start_time, duration))
                job_finish_times[job_idx] = finish_time
        else:
            # Single job
            for j in jobs:
                utilization = matrix[j][col]
                duration = utilization * resource_times[col]
                start_time = job_finish_times[j]
                schedule.append((col, j, start_time, duration))
                job_finish_times[j] = start_time + duration

    makespan = max(job_finish_times) if job_finish_times else 0
    return schedule, makespan


def compute_schedule_detailed(matrix, resource_times, available_vector):
    """
    Compute schedule with DETAILED OUTPUT (Phase 1 and Phase 2 printed).

    Returns:
        tuple: (schedule, makespan)
    """
    num_jobs = len(matrix)
    num_resources = len(matrix[0])

    print("\n" + "=" * 70)
    print("RESOURCE SCHEDULING ALGORITHM")
    print("=" * 70)

    # Display input
    print("\n[Step 1] INPUT DATA:")
    print(f"  Matrix (Jobs × Resources):")
    for i, row in enumerate(matrix):
        print(f"    J{i + 1}: {row}")
    print(f"\n  Resource execution times: {resource_times}")
    print(f"  Available vector: {available_vector}")

    # PHASE 1
    best_order, min_vector = find_best_order_and_minimum_vector(
        matrix, resource_times, verbose=True
    )

    # Step 3
    print("\n[Step 3] VECTOR ANALYSIS:")
    print(f"  Current vector:  {available_vector}")
    print(f"  Minimum vector:  {min_vector}")

    has_conflicts = any(
        available_vector[i] < min_vector[i] for i in range(num_resources)
    )

    if has_conflicts:
        print(
            "\n  ⚠️  Vector is below minimum - conflicts will be resolved using max workload"
        )
    else:
        print("\n  ✓ Vector is at or above minimum")

    # PHASE 2
    conflicts = check_conflicts(matrix, available_vector, verbose=True)

    # Solve schedule
    print("\n[Step 5-6] SCHEDULING WITH MAX WORKLOAD PROCESSING:")

    job_finish_times = [0] * num_jobs
    schedule = []

    for col in best_order:
        jobs = [j for j in range(num_jobs) if matrix[j][col] > 0]

        if len(jobs) == 0:
            continue

        available_units = available_vector[col]

        if col in conflicts or len(jobs) > 1:
            # Use optimal ordering with parallel execution
            optimal_order, job_details = resolve_conflict_parallel(
                col,
                jobs,
                job_finish_times,
                available_units,
                matrix,
                resource_times,
                verbose=True,
            )

            # Update schedule
            for job_idx, start_time, finish_time in job_details:
                duration = finish_time - start_time
                schedule.append((col, job_idx, start_time, duration))
                job_finish_times[job_idx] = finish_time
        else:
            # Single job
            print(f"\n  R{col + 1}: Single job - direct execution")
            for j in jobs:
                utilization = matrix[j][col]
                duration = utilization * resource_times[col]
                start_time = job_finish_times[j]
                schedule.append((col, j, start_time, duration))
                job_finish_times[j] = start_time + duration
                print(
                    f"    J{j + 1}: starts at {start_time}, finishes at {job_finish_times[j]}"
                )

    makespan = max(job_finish_times)
    return schedule, makespan


# =============================================================================
# OPTIMIZER
# =============================================================================


def iterative_vector_reduction(matrix, resource_times, initial_vector):
    """
    Iteratively reduce vector values to find optimal allocation.

    Returns:
        tuple: (optimal_vector, final_schedule, final_makespan, history)
    """
    num_jobs = len(matrix)
    num_resources = len(matrix[0])

    print("\n" + "=" * 70)
    print("ITERATIVE VECTOR REDUCTION")
    print("=" * 70)

    current_vector = initial_vector.copy()
    schedule, current_makespan = compute_schedule(
        matrix, resource_times, current_vector
    )

    print(f"\nStarting vector: {current_vector}")
    print(f"Starting makespan: {current_makespan}")

    history = [(current_vector.copy(), current_makespan)]

    print("\n" + "-" * 70)
    print("TRYING TO REDUCE VECTOR...")
    print("-" * 70)

    iteration = 0
    improved = True

    while improved:
        improved = False

        for col in range(num_resources):
            # Check minimum value for this resource
            jobs_needing = sum(1 for row in range(num_jobs) if matrix[row][col] > 0)
            min_value = 1 if jobs_needing > 0 else 0

            if current_vector[col] > min_value:
                iteration += 1

                # Try lowering this column
                test_vector = current_vector.copy()
                test_vector[col] -= 1

                print(
                    f"\nIteration {iteration}: Testing R{col + 1} reduction: {current_vector} → {test_vector}"
                )

                # Count conflicts
                conflicts_count = 0
                for c in range(num_resources):
                    jobs_c = sum(1 for row in range(num_jobs) if matrix[row][c] > 0)
                    if jobs_c > test_vector[c]:
                        conflicts_count += 1

                if conflicts_count > 0:
                    print(f"  ⚠️  Conflicts detected: {conflicts_count} - resolving...")

                # Solve with new vector
                test_schedule, test_makespan = compute_schedule(
                    matrix, resource_times, test_vector
                )

                if test_makespan > current_makespan:
                    print(
                        f"  📈 Makespan INCREASED: {current_makespan} → {test_makespan}"
                    )
                    print(f"  → Reverting to {current_vector}")
                elif test_makespan == current_makespan:
                    print(f"  ➡️  Makespan SAME: {test_makespan}")
                    print(f"  → Keeping reduction: {test_vector}")
                    current_vector = test_vector
                    current_makespan = test_makespan
                    history.append((current_vector.copy(), current_makespan))
                    improved = True
                    break
                else:
                    print(
                        f"  📉 Makespan DECREASED: {current_makespan} → {test_makespan} ✓"
                    )
                    print(f"  → Accepting reduction: {test_vector}")
                    current_vector = test_vector
                    current_makespan = test_makespan
                    history.append((current_vector.copy(), current_makespan))
                    improved = True
                    break

    print("\n" + "-" * 70)
    print("REDUCTION COMPLETE")
    print("-" * 70)
    print(f"\nOptimal vector: {current_vector}")
    print(f"Optimal makespan: {current_makespan}")

    # Show minimum possible
    min_possible = [
        1 if sum(1 for row in range(num_jobs) if matrix[row][c] > 0) > 0 else 0
        for c in range(num_resources)
    ]
    print(f"Minimum possible: {min_possible}")

    print("\n" + "-" * 70)
    print("HISTORY:")
    print("-" * 70)
    for i, (vec, ms) in enumerate(history):
        print(f"  Step {i}: {vec} → Makespan = {ms}")

    final_schedule, final_makespan = compute_schedule(
        matrix, resource_times, current_vector
    )

    return current_vector, final_schedule, final_makespan, history
