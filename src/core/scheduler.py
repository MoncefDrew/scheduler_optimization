import random
from itertools import permutations
from typing import Callable, Dict, List, Optional, Tuple


class ResourceScheduler:
    """
    Core scheduling and optimization logic (Model).
    """

    def __init__(self, matrix=None, resource_times=None):
        if matrix is None:
            self.matrix = [
                [1, 2, 1],
                [1, 1, 1],
                [1, 1, 2],
            ]
        else:
            self.matrix = matrix

        if resource_times is None:
            self.resource_times = [3, 5, 4]
        else:
            self.resource_times = resource_times

        self.num_jobs = len(self.matrix)
        self.num_resources = len(self.matrix[0])

    def calculate_resource_workload(
        self, logger: Optional[Callable[[str], None]] = None
    ):
        if logger is not None:
            logger("=" * 70)
            logger("PHASE 1: RESOURCE WORKLOAD CALCULATION")
            logger("=" * 70)
            logger("resource, tokens, exec_time, workload")

        workloads = []
        for col in range(self.num_resources):
            token_sum = sum(self.matrix[row][col] for row in range(self.num_jobs))
            workload = token_sum * self.resource_times[col]
            workloads.append((col, workload, token_sum))

        workloads.sort(key=lambda x: (-x[1], x[0]))
        resource_order = [col for col, _, _ in workloads]

        if logger is not None:
            for rank, (col, workload, tokens) in enumerate(workloads, start=1):
                logger(
                    f"R{col + 1}: tokens={tokens}, exec_time={self.resource_times[col]}, "
                    f"workload={workload}, rank={rank}"
                )
            order_str = " -> ".join(f"R{c + 1}" for c in resource_order)
            logger(f"Resource execution order (by workload): {order_str}")

        return resource_order, workloads

    def _get_resource_order_silent(self):
        workloads = []
        for col in range(self.num_resources):
            token_sum = sum(self.matrix[row][col] for row in range(self.num_jobs))
            workload = token_sum * self.resource_times[col]
            workloads.append((col, workload))
        workloads.sort(key=lambda x: (-x[1], x[0]))
        return [col for col, _ in workloads]

    def apply_matrix_method_for_resource(
        self,
        resource_col: int,
        jobs: List[int],
        job_finish_times: List[float],
        available_units: int,
        logger: Optional[Callable[[str], None]] = None,
    ):
        base_time = self.resource_times[resource_col]

        jobs_with_data = []
        for j in jobs:
            utilization = self.matrix[j][resource_col]
            computing_time = utilization * base_time
            release_time = job_finish_times[j]
            jobs_with_data.append((j, computing_time, release_time))

        if logger is not None:
            logger("")
            logger("  " + "=" * 68)
            logger(f"  APPLYING MATRIX METHOD - R{resource_col + 1}")
            logger("  " + "=" * 68)
            logger(f"  Jobs at this resource: {[f'J{j + 1}' for j in jobs]}")
            logger(f"  Available units: {available_units}")
            logger("")
            logger("  Job parameters:")
            for job_idx, computing_time, release_time in jobs_with_data:
                logger(
                    f"    J{job_idx + 1}: Pi={computing_time}, rj (arrival)={release_time}"
                )

        best_order = None
        best_makespan = float("inf")
        best_details = None

        perms = list(permutations(jobs_with_data))
        if logger is not None:
            logger("")
            logger(f"  Testing {len(perms)} orderings to find optimal:")
            logger(f"  {'-' * 68}")

        for perm_index, perm in enumerate(perms, start=1):
            unit_free_times = [0] * available_units
            finish_times = []

            for job_idx, computing_time, release_time in perm:
                earliest_unit = min(
                    range(available_units), key=lambda u: unit_free_times[u]
                )
                earliest_free_time = unit_free_times[earliest_unit]
                start_time = max(earliest_free_time, release_time)
                finish_time = start_time + computing_time

                finish_times.append((job_idx, start_time, finish_time))
                unit_free_times[earliest_unit] = finish_time

            makespan = max(ft[2] for ft in finish_times)

            marker = ""
            if makespan < best_makespan:
                best_makespan = makespan
                best_order = [j[0] for j in perm]
                best_details = finish_times
                marker = " ⭐"

            if logger is not None:
                order_str = " → ".join(f"J{j[0] + 1}" for j in perm)
                logger(f"  #{perm_index}: {order_str:20} | Makespan: {makespan}{marker}")

        if logger is not None:
            logger(f"  {'-' * 68}")
            logger(f"  ✓ OPTIMAL ORDER: {' → '.join([f'J{j + 1}' for j in best_order])}")
            logger(f"  ✓ Makespan for this resource: {best_makespan}")

        return best_order, best_details

    def compute_schedule_with_job_paths(
        self,
        available_vector: List[int],
        job_paths: Dict[int, List[int]],
        logger: Optional[Callable[[str], None]] = None,
    ):
        resource_order = self._get_resource_order_silent()
        job_finish_times = [0] * self.num_jobs
        schedule = []

        for resource_col in resource_order:
            jobs_at_resource = []
            for job_idx, path in job_paths.items():
                if resource_col in path:
                    jobs_at_resource.append(job_idx)

            if len(jobs_at_resource) == 0:
                continue

            available_units = available_vector[resource_col]

            if len(jobs_at_resource) > 1:
                if logger is not None:
                    logger("")
                    logger("=" * 70)
                    logger(f"Processing R{resource_col + 1}")
                    logger("=" * 70)
                    logger(
                        f"Jobs visiting this resource: {[f'J{j + 1}' for j in jobs_at_resource]}"
                    )
                _, job_details = self.apply_matrix_method_for_resource(
                    resource_col,
                    jobs_at_resource,
                    job_finish_times,
                    available_units,
                    logger=logger,
                )

                for job_idx, start_time, finish_time in job_details:
                    duration = finish_time - start_time
                    schedule.append((resource_col, job_idx, start_time, duration))
                    job_finish_times[job_idx] = finish_time
            else:
                j = jobs_at_resource[0]
                utilization = self.matrix[j][resource_col]
                duration = utilization * self.resource_times[resource_col]
                start_time = job_finish_times[j]

                schedule.append((resource_col, j, start_time, duration))
                job_finish_times[j] = start_time + duration

                if logger is not None:
                    logger("")
                    logger("=" * 70)
                    logger(f"Processing R{resource_col + 1}")
                    logger("=" * 70)
                    logger(f"Jobs visiting this resource: ['J{j + 1}']")
                    logger("")
                    logger(f"  Only J{j + 1} at this resource")
                    logger(f"  Starts at {start_time}, finishes at {job_finish_times[j]}")

        makespan = max(job_finish_times) if job_finish_times else 0
        if logger is not None:
            logger("")
            logger("=" * 70)
            logger(f"FINAL MAKESPAN: {makespan}")
            logger("=" * 70)
        return schedule, makespan

    def calculate_minimum_vector(self, job_paths: Dict[int, List[int]]) -> List[int]:
        min_vector: List[int] = []

        for col in range(self.num_resources):
            jobs_using = sum(1 for _, path in job_paths.items() if col in path)
            if jobs_using == 0:
                min_vector.append(0)
            else:
                position_groups: Dict[int, List[int]] = {}
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

    def optimize_vector(
        self,
        initial_vector: List[int],
        job_paths: Dict[int, List[int]],
        logger: Optional[Callable[[str], None]] = None,
        scenario_name: str = "",
        scenario_index: Optional[int] = None,
    ):
        def log(msg: str) -> None:
            if logger is not None:
                logger(msg)

        title = scenario_name or "Scenario"

        # PHASE 1 (workload table, like the old script)
        log("")
        log("=" * 70)
        log("PHASE 1: RESOURCE WORKLOAD CALCULATION")
        log("=" * 70)
        log("")
        log("Workload = Σ(tokens) × execution_time")
        log("")
        log(f"{'Resource':<12} {'Tokens':<10} {'Exec Time':<12} {'Workload':<12} {'Rank'}")
        log("-" * 70)
        resource_order, workloads = self.calculate_resource_workload(logger=None)
        for rank, (col, workload, tokens) in enumerate(workloads, start=1):
            log(
                f"R{col + 1:<11} {tokens:<10} {self.resource_times[col]:<12} {workload:<12} #{rank}"
            )
        log("")
        order_str = " → ".join(f"R{c + 1}" for c in resource_order)
        log(f"Resource execution order (by workload): {order_str}")

        # PHASE 2 (scheduling with matrix method) must come after Phase 1
        log("")
        log("=" * 70)
        log(f"PHASE 2: MATRIX METHOD SCHEDULING - {title}")
        log("=" * 70)
        log("")
        log("Job routing paths:")
        for job_idx, path in job_paths.items():
            path_str = " → ".join(f"R{r + 1}" for r in path)
            log(f"  J{job_idx + 1}: {path_str}")

        # Detailed scheduling blocks ("Processing Rk")
        _ = self.compute_schedule_with_job_paths(initial_vector, job_paths, logger=logger)

        log("")
        log("=" * 70)
        log(f"PHASE 3: AVAILABILITY VECTOR OPTIMIZATION - {title}")
        log("=" * 70)

        min_vector = self.calculate_minimum_vector(job_paths)
        log("")
        log("Minimum vector calculation:")
        for col in range(self.num_resources):
            jobs_using = sum(1 for _, path in job_paths.items() if col in path)
            if jobs_using > 0:
                log(f"  R{col + 1}: {jobs_using} jobs use it → min = {min_vector[col]}")
            else:
                log(f"  R{col + 1}: No jobs use it → min = 0")

        log("")
        log(f"Minimum vector: {min_vector}")

        current_vector = initial_vector.copy()
        _, current_makespan = self.compute_schedule_with_job_paths(
            current_vector, job_paths, logger=None
        )
        log(f"Starting vector: {current_vector}")
        log(f"Starting makespan: {current_makespan}")

        history: List[Tuple[List[int], float]] = [(current_vector.copy(), current_makespan)]
        improved = True
        iteration = 0

        log("")
        log("Optimizing vector...")
        while improved:
            improved = False

            for col in range(self.num_resources):
                if current_vector[col] > min_vector[col]:
                    iteration += 1
                    test_vector = current_vector.copy()
                    test_vector[col] -= 1

                    _, test_makespan = self.compute_schedule_with_job_paths(
                        test_vector, job_paths, logger=None
                    )

                    if test_makespan <= current_makespan:
                        log(
                            f"  Iter {iteration}: R{col + 1} {current_vector[col]}→{test_vector[col]} "
                            f"✓ Makespan {current_makespan}→{test_makespan}"
                        )
                        current_vector = test_vector
                        current_makespan = test_makespan
                        history.append((current_vector.copy(), current_makespan))
                        improved = True
                        break
                    else:
                        log(
                            f"  Iter {iteration}: R{col + 1} {current_vector[col]}→{test_vector[col]} "
                            f"❌ Makespan {current_makespan}→{test_makespan}"
                        )

        log("")
        log(f"Optimal vector: {current_vector}")
        log(f"Optimal makespan: {current_makespan}")

        final_schedule, _ = self.compute_schedule_with_job_paths(
            current_vector, job_paths, logger=None
        )

        return current_vector, final_schedule, current_makespan, history


def generate_random_job_paths(num_jobs: int, num_resources: int, matrix):
    job_paths: Dict[int, List[int]] = {}
    for job_idx in range(num_jobs):
        needed = [r for r in range(num_resources) if matrix[job_idx][r] > 0]
        random.shuffle(needed)
        job_paths[job_idx] = needed
    return job_paths

