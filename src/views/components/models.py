
from __future__ import annotations

from typing import Dict, List, Optional


class Scenario:


    def __init__(self, name: str, job_paths: Dict[int, List[int]]):
        self.name = name
        self.job_paths = job_paths
        self.optimal_vector: Optional[List[int]] = None
        self.optimal_makespan: Optional[float] = None
        self.schedule = None
        self.logs: str = ""
