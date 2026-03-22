from __future__ import annotations
import csv
from typing import List
from ..core.io_utils import (
    SchedulerConfig,
    parse_matrix_from_csv_rows,
    parse_vector_from_row,
)


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
    """
    Parse and validate CSV text into a SchedulerConfig.

    Expected format:
      - Block 1: matrix rows (integers), one row per job
      - Blank line
      - Block 2: resource_times row (floats/ints)
      - Blank line
      - Block 3: availability_vector row (ints)

    Additional constraints:
      - Matrix must be square (n × n)
      - Resource_times length must be n
      - Availability_vector length must be n
    """
    reader = csv.reader(text.splitlines())
    rows = [row for row in reader]

    blocks = _split_blocks(rows)
    if len(blocks) < 3:
        raise ValueError(
            "CSV must contain three blocks: matrix, resource_times row, availability_vector row."
        )

    matrix_rows = blocks[0]
    resource_times_row = blocks[1][0]
    availability_row = blocks[2][0]

    matrix = parse_matrix_from_csv_rows(matrix_rows)
    if not matrix:
        raise ValueError("Matrix block is empty.")

    num_jobs = len(matrix)
    num_resources = len(matrix[0])
    if num_jobs != num_resources:
        raise ValueError(f"Matrix must be square (n×n). Found {num_jobs}×{num_resources}.")
    for row in matrix:
        if len(row) != num_resources:
            raise ValueError("All matrix rows must have the same length.")

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
