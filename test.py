from typing import List

class ResourceScheduler:
    def __init__(self):
        # Jobs × Resources matrix
        self.matrix = [
            [2, 1, 0],  # J1
            [0, 2, 3],  # J2
            [1, 1, 1]   # J3
        ]

        # Execution time per single token of resource
        self.resource_times = [3, 5, 4]

        self.num_jobs = len(self.matrix)
        self.num_resources = len(self.matrix[0])

    def get_execution_order(self):
        """
        Select columns in order of maximum total resources used.
        If tie, smaller index first.
        """
        column_usage = []

        for col in range(self.num_resources):
            # Sum of tokens used by all jobs for this resource
            usage = sum(self.matrix[row][col] for row in range(self.num_jobs))
            column_usage.append((col, usage))

        # Sort descending by total resource usage
        column_usage.sort(key=lambda x: (-x[1], x[0]))

        print("Total column resource usage:", column_usage)

        return [col for col, _ in column_usage]

    def compute_makespan(self, available_resources: List[int]):
        """
        Compute makespan based on the rules:
        - Execute jobs in parallel if number of jobs ≤ available tokens
        - Execution time = utilization × resource_time
        - Error if parallel execution is impossible
        """
        execution_order = self.get_execution_order()
        job_finish_time = [0] * self.num_jobs

        for col in execution_order:
            available_tokens = available_resources[col]
            base_time = self.resource_times[col]

            # Jobs that need this resource
            jobs = [j for j in range(self.num_jobs) if self.matrix[j][col] > 0]

            print(f"\nExecuting R{col+1}")
            print("Jobs:", [f"J{j+1}" for j in jobs])

            required_tokens = len(jobs)  # minimum 1 per job

            if required_tokens <= available_tokens:
                print("Parallel execution allowed.")
                for j in jobs:
                    utilization = self.matrix[j][col]
                    real_time = utilization * base_time
                    job_finish_time[j] += real_time
            else:
                print("❌ ERROR: Not enough resources to execute in parallel.")
                print(f"Required jobs: {required_tokens}, Available: {available_tokens}")
                return None

            print("Job finish times:",
                  {f"J{i+1}": job_finish_time[i] for i in range(self.num_jobs)})

        makespan = max(job_finish_time)
        return makespan


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    scheduler = ResourceScheduler()

    # Input: available tokens for each resource
    user_input = input("Enter your availability vector (R1 R2 R3): ")
    available_resources = list(map(int, user_input.split()))

    makespan = scheduler.compute_makespan(available_resources)

    if makespan is not None:
        print("\nFinal Makespan:", makespan)

