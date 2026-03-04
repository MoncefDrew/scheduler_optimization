### Flexible Job Shop Scheduling System

A Python implementation of a **Flexible Job Shop Scheduler** that optimizes the execution of multiple jobs across shared resources. Each job follows a routing path through resources, and the system finds schedules that minimize total completion time (makespan) while using resources efficiently.

---

## Overview

The scheduler models a setting such as a factory with several machines (resources) and several products (jobs). Each job:

- Requires some amount (tokens) of each resource.
- Follows a specific route through resources (job path).
- Can run in parallel with other jobs, subject to resource capacity (availability vector).

The core ideas are summarized below:

| Aspect                 | Description                                                                                  |
|------------------------|----------------------------------------------------------------------------------------------|
| **Resource workloads** | Identify bottleneck resources based on total tokens × execution time.                       |
| **Matrix method**      | Exhaustive permutation testing to find the best job order on each resource.                |
| **Vector optimization**| Iteratively reduce resource units while keeping makespan the same or better.               |
| **Scenarios**          | Support multiple routing configurations (manual, random, or all possible) with comparison. |

---

## Features

| Area              | Capabilities                                                                                          |
|-------------------|-------------------------------------------------------------------------------------------------------|
| **Model**         | Flexible job shop; each job can have its own routing path.                                           |
| **Scheduling**    | Exact per-resource ordering using permutation-based matrix method.                                   |
| **Optimization**  | Availability vector reduction without degrading makespan.                                             |
| **Scenarios**     | Manual, random, and “generate all” routing scenarios with per-scenario optimization and comparison.  |
| **GUI**           | CSV import, manual data entry, scenario configuration, Gantt charts, and best-scenario highlighting. |

---

## Problem Model

### Jobs, Resources, Times, Paths, and Capacity

| Concept                    | Example (Python-style)                                                                                       | Meaning                                                  |
|---------------------------|--------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| **Jobs × Resources matrix** | `matrix = [[1, 2, 1], [1, 1, 1], [1, 1, 2]]`                                                                | Tokens each job needs from each resource.               |
| **Resource times**        | `resource_times = [3, 5, 4]`                                                                                 | Time per token for R1, R2, R3.                          |
| **Job routing paths**     | `job_paths = {0: [1, 0, 2], 1: [0, 1, 2], 2: [2, 1, 0]}`                                                     | Order each job visits resources (0-based indices).      |
| **Availability vector**   | `availability = [2, 3, 2]`                                                                                   | Parallel units per resource (R1, R2, R3).               |

---

## Algorithm Phases

### Phase 1 – Resource Workload Calculation

For each resource \(R_k\):

\[
\text{Workload}_k = \left(\sum_{\text{jobs } j} \text{tokens}_{j,k}\right) \times \text{exec\_time}_k
\]

Example workloads:

| Resource | Tokens sum | Exec time | Workload | Rank (example) |
|----------|------------|-----------|----------|----------------|
| R1       | 3          | 3         | 9        | #3             |
| R2       | 4          | 5         | 20       | #1             |
| R3       | 4          | 4         | 16       | #2             |

Execution order by workload: **R2 → R3 → R1**.

---

### Phase 2 – Scheduling with Matrix Method

For each resource, the scheduler finds the job order that minimizes makespan, given:

| Quantity            | Definition                                              |
|---------------------|---------------------------------------------------------|
| \(P_i\)             | Processing time = tokens × execution time.             |
| \(r_j\)             | Arrival time at the resource (release from prior steps). |
| Parallel capacity   | Number of units for the resource (from availability).  |

High-level procedure per resource:

| Step | Action                                                                                       |
|------|----------------------------------------------------------------------------------------------|
| 1    | Collect all jobs that visit this resource.                                                   |
| 2    | Compute \(P_i\) and \(r_j\) for these jobs.                                                  |
| 3    | Enumerate all permutations of job orders.                                                    |
| 4    | For each permutation, simulate using per-unit free times and compute its makespan.          |
| 5    | Select the permutation with the smallest makespan for that resource.                         |

Parallelism is handled by maintaining `unit_free_times` per unit and always assigning jobs to the earliest available unit, respecting their arrival times.

---

### Phase 3 – Availability Vector Optimization

**Goal**: reduce the availability vector while preserving or improving makespan.

Key steps:

| Step | Description                                                                                                              |
|------|--------------------------------------------------------------------------------------------------------------------------|
| 1    | Compute a minimum vector from job paths (max concurrent usage per resource).                                            |
| 2    | Start from a user-provided availability vector.                                                                         |
| 3    | Try decreasing each component by 1 (never below the minimum vector), re-running scheduling each time.                  |
| 4    | Accept reductions that do not worsen makespan; revert those that do.                                                   |
| 5    | Stop when no further reduction is accepted; the resulting vector is “optimal” under this local search.                 |

---

## Scenarios and Comparison

A **scenario** is a complete set of routing paths for all jobs. The system supports:

| Scenario type  | Description                                                                                                 |
|----------------|-------------------------------------------------------------------------------------------------------------|
| Manual         | User defines job paths explicitly via the GUI.                                                             |
| Random         | Job paths built by shuffling the resources each job requires (from the matrix).                           |
| All-scenarios  | Cartesian product of all permutations of required resources per job; every possible routing combination.   |

For each scenario, the scheduler:

| Output                         | Description                                              |
|--------------------------------|----------------------------------------------------------|
| Job paths                      | The chosen routing paths per job.                       |
| Initial vs optimal vector      | Availability vector before and after optimization.      |
| Initial vs optimal makespan    | Makespan before and after vector optimization.          |
| Final schedule                 | Task schedule per resource, used to build Gantt charts. |

### Best Scenario Selection

Scenario quality is measured by:

| Criterion           | Role                                              |
|---------------------|---------------------------------------------------|
| Minimal makespan    | Primary objective (lower is better).             |
| Maximal vector      | Tie-breaker (lexicographically maximal vector).  |

The best scenario according to these criteria is automatically selected and highlighted in the GUI.

---

## GUI Overview

The Tkinter GUI exposes the scheduler via a few main areas:

| Area                 | Capabilities                                                                                       |
|----------------------|----------------------------------------------------------------------------------------------------|
| Configuration input  | Load from CSV or define a new matrix, resource times, and availability vector.                    |
| Scenario creation    | Set number of scenarios; for each, choose random or custom paths via a tree-based editor.         |
| Scenario execution   | Run selected scenario, compute optimal vector and makespan, and view results in a list.           |
| Visualization        | Show Gantt charts for the selected scenario in the Diagram tab.                                   |
| Scenario comparison  | Compare executed scenarios and summarize their vectors and makespans.                             |
| All-scenarios run    | Generate and evaluate all possible routing scenarios (with a safeguard for very large counts).    |

---

## CSV Input Format

The CSV loader expects three logical blocks separated by blank lines:

| Block | Content          | Example                                                                 |
|-------|------------------|-------------------------------------------------------------------------|
| 1     | Matrix rows      | `1,2,1` / `1,1,1` / `1,1,2`                                             |
| 2     | Resource times   | `3,5,4`                                                                 |
| 3     | Availability     | `2,3,2`                                                                 |

Example complete file:

```csv
1,2,1
1,1,1
1,1,2

3,5,4

2,3,2
```

---

## Running the Application

### Requirements

| Dependency  | Notes              |
|------------|--------------------|
| Python 3   | Recommended 3.8+   |
| matplotlib | For Gantt plotting |

Install dependencies:

```bash
pip install matplotlib
```

### Start the GUI

From the project root:

```bash
python test.py
```

This opens the GUI for configuring data, defining scenarios, running optimizations, and visualizing schedules.

---

## Repository Structure

| Path              | Description                                   |
|-------------------|-----------------------------------------------|
| `src/scheduler.py`    | Core scheduling and vector optimization logic.  |
| `src/io_utils.py`     | CSV parsing and basic data helpers.            |
| `src/visualization.py`| Gantt chart construction and plotting.         |
| `src/gui.py`          | Tkinter GUI: configuration, scenarios, charts. |
| `test.py`             | GUI entry point (`run_gui`).                   |
| `README.md`           | Project documentation.                          |

---

## Notes and Limitations

| Topic              | Note                                                                                                  |
|--------------------|-------------------------------------------------------------------------------------------------------|
| Complexity         | Exhaustive permutation testing can be expensive when many jobs share a resource.                     |
| All-scenarios run  | Generating all routing scenarios can explode combinatorially; a confirmation is used for large runs. |
| Intended scale     | Best suited to small and medium instances where exact optimality is valuable.                        |

---

## Use Cases

| Domain                    | Example application                                      |
|---------------------------|----------------------------------------------------------|
| Manufacturing             | Scheduling products across machines and workstations.    |
| Compute clusters          | Assigning tasks to servers with limited capacities.      |
| Project management        | Resource-constrained task scheduling.                    |
| General operations research | Any flexible job shop with routing, capacities, and makespan objectives. |


