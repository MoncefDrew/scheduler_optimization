from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SchedulerConfig:
    matrix: List[List[int]]
    resource_times: List[float]
    availability_vector: List[int]


def parse_matrix_from_csv_rows(rows: List[List[str]]) -> List[List[int]]:
    return [[int(cell) for cell in row if cell != ""] for row in rows]


def parse_vector_from_row(row: List[str], as_float: bool = False):
    if as_float:
        return [float(x) for x in row if x != ""]
    return [int(x) for x in row if x != ""]


def load_config_from_csv(path: str) -> SchedulerConfig:
    """
    Very simple CSV format:
      - First block: matrix rows until an empty line
      - Second block: resource_times on one row
      - Third block: availability_vector on one row
    """
    import csv

    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

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

    if len(blocks) < 3:
        raise ValueError(
            "CSV must contain three blocks: matrix, resource_times row, availability_vector row"
        )

    matrix_rows = blocks[0]
    resource_times_row = blocks[1][0]
    availability_row = blocks[2][0]

    matrix = parse_matrix_from_csv_rows(matrix_rows)
    resource_times = parse_vector_from_row(resource_times_row, as_float=True)
    availability_vector = parse_vector_from_row(availability_row, as_float=False)

    return SchedulerConfig(
        matrix=matrix,
        resource_times=resource_times,
        availability_vector=availability_vector,
    )


def infer_dimensions_from_matrix(matrix: List[List[int]]) -> Tuple[int, int]:
    num_jobs = len(matrix)
    num_resources = len(matrix[0]) if num_jobs > 0 else 0
    return num_jobs, num_resources

