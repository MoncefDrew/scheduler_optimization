
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SchedulerConfig:
    matrix: List[List[int]]
    resource_times: List[float]
    availability_vector: List[int]


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------

def parse_matrix_from_csv_rows(rows: List[List[str]]) -> List[List[int]]:
    return [[int(cell) for cell in row if cell != ""] for row in rows]


def parse_vector_from_row(row: List[str], as_float: bool = False):
    if as_float:
        return [float(x) for x in row if x != ""]
    return [int(x) for x in row if x != ""]


def infer_dimensions_from_matrix(matrix: List[List[int]]) -> Tuple[int, int]:
    num_jobs = len(matrix)
    num_resources = len(matrix[0]) if num_jobs > 0 else 0
    return num_jobs, num_resources


# ---------------------------------------------------------------------------
# CSV I/O  (moved here from controllers/csv_controller.py)
# ---------------------------------------------------------------------------

def _split_blocks(rows: List[List[str]]) -> List[List[List[str]]]:

    blocks: List[List[List[str]]] = []
    current: List[List[str]] = []
    for row in rows:
        if not any(cell.strip() for cell in row):
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(row)
    if current:
        blocks.append(current)
    return blocks


def parse_csv_text(text: str) -> SchedulerConfig:

    reader = csv.reader(text.splitlines())
    rows = list(reader)

    blocks = _split_blocks(rows)
    if len(blocks) < 3:
        raise ValueError(
            "CSV must contain three blocks separated by blank lines: "
            "matrix, resource_times row, availability_vector row."
        )

    matrix_rows = blocks[0]
    resource_times_row = blocks[1][0]
    availability_row = blocks[2][0]

    matrix = parse_matrix_from_csv_rows(matrix_rows)
    if not matrix:
        raise ValueError("Matrix block is empty.")

    num_resources = len(matrix[0])
    for row in matrix:
        if len(row) != num_resources:
            raise ValueError("All matrix rows must have the same number of columns.")

    resource_times = parse_vector_from_row(resource_times_row, as_float=True)
    availability_vector = parse_vector_from_row(availability_row, as_float=False)

    if len(resource_times) != num_resources:
        raise ValueError(
            f"Resource times length must be {num_resources}, got {len(resource_times)}."
        )

    if any(t <= 0 for t in resource_times):
        raise ValueError("All resource execution times must be > 0.")

    if len(availability_vector) != num_resources:
        raise ValueError(
            f"Availability vector length must be {num_resources}, got {len(availability_vector)}."
        )

    return SchedulerConfig(
        matrix=matrix,
        resource_times=resource_times,
        availability_vector=availability_vector,
    )


def load_csv_file(path: str) -> str:

    with open(path, "r", newline="") as f:
        return f.read()


def save_csv_file(path: str, content: str) -> None:

    with open(path, "w", newline="") as f:
        f.write(content)
