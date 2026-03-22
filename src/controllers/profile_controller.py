from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..core.io_utils import SchedulerConfig


@dataclass
class ProfileSummary:
    id: int
    name: str
    created_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              created_at TEXT NOT NULL,
              config_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scenarios (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              profile_id INTEGER NOT NULL,
              name TEXT NOT NULL,
              job_paths_json TEXT NOT NULL,
              optimal_vector_json TEXT,
              optimal_makespan REAL,
              schedule_json TEXT,
              logs TEXT,
              FOREIGN KEY(profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )
            """
        )
        # Enforce unique profile names for *future writes*.
        #
        # If the DB already contains duplicate names (from before we enforced uniqueness),
        # creating this UNIQUE index would raise an IntegrityError and break loading.
        # We therefore treat it as a best-effort migration.
        try:
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_name_unique ON profiles(name)"
            )
        except sqlite3.IntegrityError:
            # Duplicates already exist in DB; uniqueness will still be enforced by
            # save/update checks, but we can't add the UNIQUE constraint retroactively.
            pass
        con.commit()
    finally:
        con.close()


def profile_exists(db_path: str, name: str) -> bool:
    init_db(db_path)
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT 1 FROM profiles WHERE name=? LIMIT 1", (name,))
        return cur.fetchone() is not None
    finally:
        con.close()


def update_profile_by_name(
    db_path: str,
    name: str,
    config: SchedulerConfig,
    scenarios: List[Dict[str, Any]],
) -> int:
    """
    Update an existing profile (matched by name), overwriting config + scenarios.
    """
    init_db(db_path)
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT id FROM profiles WHERE name=?", (name,))
        row = cur.fetchone()
        if not row:
            raise ValueError("Profile not found.")
        profile_id = int(row[0])

        config_json = json.dumps(
            {
                "matrix": config.matrix,
                "resource_times": config.resource_times,
                "availability_vector": config.availability_vector,
            }
        )
        cur.execute("UPDATE profiles SET config_json=? WHERE id=?", (config_json, profile_id))

        cur.execute("DELETE FROM scenarios WHERE profile_id=?", (profile_id,))
        for s in scenarios:
            job_paths_json = json.dumps(s["job_paths"])
            optimal_vector_json = (
                json.dumps(s["optimal_vector"]) if s.get("optimal_vector") is not None else None
            )
            schedule_json = (
                json.dumps(s["schedule"]) if s.get("schedule") is not None else None
            )
            cur.execute(
                """
                INSERT INTO scenarios(
                  profile_id, name, job_paths_json,
                  optimal_vector_json, optimal_makespan,
                  schedule_json, logs
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    profile_id,
                    s["name"],
                    job_paths_json,
                    optimal_vector_json,
                    s.get("optimal_makespan"),
                    schedule_json,
                    s.get("logs", ""),
                ),
            )

        con.commit()
        return profile_id
    finally:
        con.close()


def list_profiles(db_path: str) -> List[ProfileSummary]:
    init_db(db_path)
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT id, name, created_at FROM profiles ORDER BY id DESC"
        )
        return [ProfileSummary(id=r[0], name=r[1], created_at=r[2]) for r in cur.fetchall()]
    finally:
        con.close()


def save_profile(
    db_path: str,
    name: str,
    config: SchedulerConfig,
    scenarios: List[Dict[str, Any]],
) -> int:
    """
    scenarios: list of dicts with keys:
      - name (str)
      - job_paths (Dict[int, List[int]])
      - optimal_vector (Optional[List[int]])
      - optimal_makespan (Optional[float])
      - schedule (Optional[List[Tuple[int,int,float,float]]])
      - logs (str)
    """
    init_db(db_path)
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        # Prevent duplicate profile names
        cur.execute("SELECT 1 FROM profiles WHERE name=? LIMIT 1", (name,))
        if cur.fetchone() is not None:
            raise ValueError("A profile with this name already exists.")

        config_json = json.dumps(
            {
                "matrix": config.matrix,
                "resource_times": config.resource_times,
                "availability_vector": config.availability_vector,
            }
        )
        cur.execute(
            "INSERT INTO profiles(name, created_at, config_json) VALUES(?,?,?)",
            (name, _utc_now_iso(), config_json),
        )
        profile_id = int(cur.lastrowid)

        for s in scenarios:
            job_paths_json = json.dumps(s["job_paths"])
            optimal_vector_json = (
                json.dumps(s["optimal_vector"]) if s.get("optimal_vector") is not None else None
            )
            schedule_json = (
                json.dumps(s["schedule"]) if s.get("schedule") is not None else None
            )
            cur.execute(
                """
                INSERT INTO scenarios(
                  profile_id, name, job_paths_json,
                  optimal_vector_json, optimal_makespan,
                  schedule_json, logs
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (
                    profile_id,
                    s["name"],
                    job_paths_json,
                    optimal_vector_json,
                    s.get("optimal_makespan"),
                    schedule_json,
                    s.get("logs", ""),
                ),
            )

        con.commit()
        return profile_id
    finally:
        con.close()


def load_profile(
    db_path: str, profile_id: int
) -> Tuple[SchedulerConfig, List[Dict[str, Any]], str]:
    init_db(db_path)
    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name, created_at, config_json FROM profiles WHERE id=?",
            (profile_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError("Profile not found.")
        profile_name, created_at, config_json = row
        cfg = json.loads(config_json)
        config = SchedulerConfig(
            matrix=cfg["matrix"],
            resource_times=cfg["resource_times"],
            availability_vector=cfg["availability_vector"],
        )

        cur.execute(
            """
            SELECT name, job_paths_json, optimal_vector_json, optimal_makespan, schedule_json, logs
            FROM scenarios
            WHERE profile_id=?
            ORDER BY id ASC
            """,
            (profile_id,),
        )
        scenarios: List[Dict[str, Any]] = []
        for srow in cur.fetchall():
            s_name, job_paths_json, opt_vec_json, opt_ms, schedule_json, logs = srow
            job_paths = {int(k): v for k, v in json.loads(job_paths_json).items()}
            optimal_vector = json.loads(opt_vec_json) if opt_vec_json else None
            schedule = json.loads(schedule_json) if schedule_json else None
            scenarios.append(
                {
                    "name": s_name,
                    "job_paths": job_paths,
                    "optimal_vector": optimal_vector,
                    "optimal_makespan": opt_ms,
                    "schedule": schedule,
                    "logs": logs or "",
                }
            )

        meta = f"{profile_name} (created {created_at})"
        return config, scenarios, meta
    finally:
        con.close()

