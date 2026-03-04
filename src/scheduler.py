import random
from itertools import permutations
from typing import Dict, List, Tuple


class ResourceScheduler:
    """
    Core scheduling and optimization logic, extracted from the original script.
    """

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
        workloads = []
        for col in range(self.num_resources):
            token_sum = sum(self.matrix[row][col] for row in range(self.num_jobs))
            workload = token_sum * self.resource_times[col]
            workloads.append((col, workload, token_sum))

        workloads.sort(key=lambda x: (-x[1], x[0]))
        resource_order = [col for col, _, _ in workloads]
        return resource_order, workloads

    def _get_resource_order_silent(self):
        workloads = []
        for col in range(self.num_resources):
            token_sum = sum(self.matrix[row][col] for row in range(self.num_jobs))
            workload = token_sum * self.resource_times[col]
            workloads.append((col, workload))
        workloads.sort(key=lambda x: (-x[1], x[0]))
        return [col for col, _ in workloads]

    # ========================================================================
    # PHASE 2: MATRIX METHOD FOR JOB ORDERING WITHIN RESOURCES
    # ========================================================================

    def apply_matrix_method_for_resource(
        self,
        resource_col: int,
        jobs: List[int],
        job_finish_times: List[float],
        available_units: int,
    ):
        base_time = self.resource_times[resource_col]

        jobs_with_data = []
        for j in jobs:
            utilization = self.matrix[j][resource_col]
            computing_time = utilization * base_time
            release_time = job_finish_times[j]
            jobs_with_data.append((j, computing_time, release_time))

        best_order = None
        best_makespan = float("inf")
        best_details = None

        for perm in permutations(jobs_with_data):
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

            if makespan < best_makespan:
                best_makespan = makespan
                best_order = [j[0] for j in perm]
                best_details = finish_times

        return best_order, best_details

    # ========================================================================
    # SCHEDULING WITH JOB PATHS
    # ========================================================================

    def compute_schedule_with_job_paths(
        self,
        available_vector: List[int],
        job_paths: Dict[int, List[int]],
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
                _, job_details = self.apply_matrix_method_for_resource(
                    resource_col,
                    jobs_at_resource,
                    job_finish_times,
                    available_units,
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

        makespan = max(job_finish_times) if job_finish_times else 0
        return schedule, makespan

    # ========================================================================
    # PHASE 3: VECTOR OPTIMIZATION
    # ========================================================================

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
    ):
        min_vector = self.calculate_minimum_vector(job_paths)

        current_vector = initial_vector.copy()
        _, current_makespan = self.compute_schedule_with_job_paths(
            current_vector, job_paths
        )

        history: List[Tuple[List[int], float]] = [(current_vector.copy(), current_makespan)]
        improved = True

        while improved:
            improved = False

            for col in range(self.num_resources):
                if current_vector[col] > min_vector[col]:
                    test_vector = current_vector.copy()
                    test_vector[col] -= 1

                    _, test_makespan = self.compute_schedule_with_job_paths(
                        test_vector, job_paths
                    )

                    if test_makespan <= current_makespan:
                        current_vector = test_vector
                        current_makespan = test_makespan
                        history.append((current_vector.copy(), current_makespan))
                        improved = True
                        break

        final_schedule, _ = self.compute_schedule_with_job_paths(
            current_vector, job_paths
        )

        return current_vector, final_schedule, current_makespan, history


def generate_random_job_paths(num_jobs: int, num_resources: int, matrix):
    job_paths: Dict[int, List[int]] = {}
    for job_idx in range(num_jobs):
        needed = [r for r in range(num_resources) if matrix[job_idx][r] > 0]
        random.shuffle(needed)
        job_paths[job_idx] = needed
    return job_paths

