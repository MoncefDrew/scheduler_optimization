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


def infer_dimensions_from_matrix(matrix: List[List[int]]) -> Tuple[int, int]:
    num_jobs = len(matrix)
    num_resources = len(matrix[0]) if num_jobs > 0 else 0
    return num_jobs, num_resources

