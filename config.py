"""
Configuration file for the Resource Scheduler
Contains matrix data and resource execution times
"""

# Jobs × Resources matrix
# Each row represents a job, each column represents a resource
# Values represent the number of tokens that job needs from that resource
JOBS_RESOURCES_MATRIX = [
    [2, 1, 0],  # J1
    [0, 2, 3],  # J2
    [1, 1, 1],  # J3
]

# Execution time per single token for each resource
# Index corresponds to resource: [R1, R2, R3]
RESOURCE_EXECUTION_TIMES = [3, 5, 4]

# Number of jobs and resources (auto-calculated)
NUM_JOBS = len(JOBS_RESOURCES_MATRIX)
NUM_RESOURCES = len(JOBS_RESOURCES_MATRIX[0])
