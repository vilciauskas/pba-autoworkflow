# SPDX-License-Identifier: GPL-3.0-or-later
"""Append-only provenance store.

A self-driving lab is only as good as its record.  Every suggestion, every device
call, every raw trace and every derived number is written here as it happens, so
that a campaign can be audited, resumed after a crash, or re-analysed with a
better analysis routine months later without re-running the chemistry.

Design decisions:

* **SQLite for metadata, files for arrays.**  Diffractograms and spectra go to
  disk as compressed ``.npz`` next to the database and are referenced by relative
  path.  Storing multi-thousand-point arrays as BLOBs makes the database
  unqueryable in practice.
* **Append-only event log.**  Rows in ``events`` are never updated.  The current
  state of an experiment is a projection over its events, which means a crash can
  never leave a half-written state -- the last consistent event wins.
* **Resumability.**  :meth:`ProvenanceStore.load_campaign` reconstructs the full
  experiment history, so restarting the orchestrator continues a campaign rather
  than starting a new one.
* **Analysis versioning.**  Every descriptor row records the code version that
  produced it, so re-reduction is distinguishable from the original pass.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .clock import timestamp
from .schema import (
    Experiment,
    ExperimentStatus,
    Objectives,
    SampleDescriptors,
    SynthesisParameters,
)

import importlib.metadata
try:
    ANALYSIS_VERSION = importlib.metadata.version("pba-autoworkflow")
except importlib.metadata.PackageNotFoundError:
    ANALYSIS_VERSION = "unknown"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id   TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    config_json   TEXT NOT NULL,
    design_json   TEXT NOT NULL,
    notes         TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    campaign_id   TEXT NOT NULL REFERENCES campaigns(campaign_id),
    batch_index   INTEGER NOT NULL,
    origin        TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    experiment_id     TEXT PRIMARY KEY REFERENCES experiments(experiment_id),
    status            TEXT NOT NULL,
    descriptors_json  TEXT,
    objectives_json   TEXT,
    qc_flags_json     TEXT,
    error             TEXT,
    analysis_version  TEXT NOT NULL,
    started_at        REAL,
    finished_at       REAL
);
CREATE TABLE IF NOT EXISTS raw_traces (
    trace_id      TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    kind          TEXT NOT NULL,
    device_id     TEXT NOT NULL,
    path          TEXT NOT NULL,
    meta_json     TEXT NOT NULL,
    recorded_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id   TEXT,
    experiment_id TEXT,
    kind          TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    at            REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS device_calls (
    call_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    device_id     TEXT NOT NULL,
    action        TEXT NOT NULL,
    started_at    REAL NOT NULL,   -- wall clock: when
    duration_s    REAL,             -- monotonic: how long (never a wall-clock delta)
    ok            INTEGER,
    detail        TEXT
);
CREATE INDEX IF NOT EXISTS ix_exp_campaign ON experiments(campaign_id, batch_index);
CREATE INDEX IF NOT EXISTS ix_events_campaign ON events(campaign_id, at);
CREATE INDEX IF NOT EXISTS ix_traces_exp ON raw_traces(experiment_id);
"""


class ProvenanceStore:
    """Durable record of a campaign."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.traces_dir = self.root / "traces"
        self.traces_dir.mkdir(exist_ok=True)
        self.db_path = self.root / "campaign.sqlite"
        self.conn = sqlite3.connect(self.db_path, isolation_level=None,
                                    check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ #
    # Campaign lifecycle
    # ------------------------------------------------------------------ #

    def create_campaign(self, campaign_id: str, config: dict[str, Any],
                        design_space_json: str, notes: str = "") -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO campaigns VALUES (?,?,?,?,?)",
            (campaign_id, timestamp(), json.dumps(config, default=str),
             design_space_json, notes),
        )
        self.log_event("campaign_created", campaign_id=campaign_id, payload=config)

    def campaign_exists(self, campaign_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM campaigns WHERE campaign_id=?", (campaign_id,)
        ).fetchone()
        return row is not None

    def campaign_config(self, campaign_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT config_json FROM campaigns WHERE campaign_id=?", (campaign_id,)
        ).fetchone()
        return json.loads(row["config_json"]) if row else None

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def log_event(self, kind: str, payload: Any = None, campaign_id: str | None = None,
                  experiment_id: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events (campaign_id, experiment_id, kind, payload_json, at)"
            " VALUES (?,?,?,?,?)",
            (campaign_id, experiment_id, kind,
             json.dumps(payload, default=str) if payload is not None else "{}",
             timestamp()),
        )

    def register_experiment(self, exp: Experiment) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?)",
            (exp.experiment_id, exp.campaign_id, exp.batch_index, exp.origin,
             exp.parameters.model_dump_json(), timestamp()),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO results (experiment_id, status, analysis_version)"
            " VALUES (?,?,?)",
            (exp.experiment_id, exp.status.value, ANALYSIS_VERSION),
        )
        self.log_event("experiment_queued", payload={"origin": exp.origin,
                                                     "batch": exp.batch_index},
                       campaign_id=exp.campaign_id, experiment_id=exp.experiment_id)

    def save_trace(self, experiment_id: str, kind: str, device_id: str,
                   arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> str:
        trace_id = f"{kind}-{uuid.uuid4().hex[:12]}"
        rel = Path("traces") / f"{experiment_id}_{trace_id}.npz"
        np.savez_compressed(self.root / rel, **arrays)
        self.conn.execute(
            "INSERT INTO raw_traces VALUES (?,?,?,?,?,?,?)",
            (trace_id, experiment_id, kind, device_id, str(rel),
             json.dumps(meta, default=str), timestamp()),
        )
        return str(rel)

    def load_trace(self, relative_path: str) -> dict[str, np.ndarray]:
        with np.load(self.root / relative_path) as z:
            return {k: z[k] for k in z.files}

    def record_device_call(self, experiment_id: str | None, device_id: str,
                           action: str, started_at: float, duration_s: float,
                           ok: bool, detail: str = "") -> None:
        """Record one device invocation.

        ``started_at`` is wall-clock (so the row is meaningful to a human) while
        ``duration_s`` must be measured monotonically by the caller.  Storing the
        duration rather than an end timestamp means utilization figures cannot be
        corrupted by a clock adjustment or a host suspend mid-call.
        """
        self.conn.execute(
            "INSERT INTO device_calls (experiment_id, device_id, action, started_at,"
            " duration_s, ok, detail) VALUES (?,?,?,?,?,?,?)",
            (experiment_id, device_id, action, started_at, float(duration_s),
             1 if ok else 0, detail),
        )

    def finalize_experiment(self, exp: Experiment, qc_flags: Iterable[str] = ()) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO results (experiment_id, status, descriptors_json,"
            " objectives_json, qc_flags_json, error, analysis_version, started_at,"
            " finished_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                exp.experiment_id,
                exp.status.value,
                exp.descriptors.model_dump_json(),
                exp.objectives.model_dump_json() if exp.objectives else None,
                json.dumps(list(qc_flags)),
                exp.error,
                ANALYSIS_VERSION,
                exp.started_at,
                exp.finished_at,
            ),
        )
        self.log_event("experiment_finalized",
                       payload={"status": exp.status.value, "qc": list(qc_flags)},
                       campaign_id=exp.campaign_id, experiment_id=exp.experiment_id)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def load_campaign(self, campaign_id: str) -> list[Experiment]:
        """Rebuild the experiment history, for resumption or analysis."""
        rows = self.conn.execute(
            "SELECT e.*, r.status, r.descriptors_json, r.objectives_json, r.error,"
            " r.started_at, r.finished_at FROM experiments e"
            " LEFT JOIN results r USING (experiment_id)"
            " WHERE e.campaign_id=? ORDER BY e.batch_index, e.created_at",
            (campaign_id,),
        ).fetchall()
        out: list[Experiment] = []
        for row in rows:
            params = SynthesisParameters.model_validate_json(row["params_json"])
            desc = (SampleDescriptors.model_validate_json(row["descriptors_json"])
                    if row["descriptors_json"] else SampleDescriptors())
            obj = (Objectives.model_validate_json(row["objectives_json"])
                   if row["objectives_json"] else None)
            traces = {
                t["kind"]: t["path"] for t in self.conn.execute(
                    "SELECT kind, path FROM raw_traces WHERE experiment_id=?",
                    (row["experiment_id"],)
                ).fetchall()
            }
            out.append(Experiment(
                experiment_id=row["experiment_id"],
                campaign_id=row["campaign_id"],
                batch_index=row["batch_index"],
                parameters=params,
                origin=row["origin"],
                status=ExperimentStatus(row["status"] or "queued"),
                descriptors=desc,
                objectives=obj,
                raw_refs=traces,
                error=row["error"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
            ))
        return out

    def next_batch_index(self, campaign_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(batch_index), -1) AS m FROM experiments"
            " WHERE campaign_id=?", (campaign_id,)
        ).fetchone()
        return int(row["m"]) + 1

    def device_utilization(self, campaign_id: str | None = None
                           ) -> list[dict[str, Any]]:
        q = ("SELECT d.device_id AS device_id, COUNT(*) AS n_calls,"
             " SUM(d.duration_s) AS busy_s,"
             " SUM(CASE WHEN d.ok=0 THEN 1 ELSE 0 END) AS n_failed"
             " FROM device_calls d")
        args: list[Any] = []
        if campaign_id is not None:
            q += (" JOIN experiments e ON e.experiment_id = d.experiment_id"
                  " WHERE e.campaign_id = ?")
            args.append(campaign_id)
        q += " GROUP BY d.device_id ORDER BY busy_s DESC"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def event_log(self, campaign_id: str, kinds: Iterable[str] | None = None
                  ) -> list[dict[str, Any]]:
        q = "SELECT kind, experiment_id, payload_json, at FROM events WHERE campaign_id=?"
        args: list[Any] = [campaign_id]
        if kinds:
            ks = list(kinds)
            q += f" AND kind IN ({','.join('?' * len(ks))})"
            args += ks
        q += " ORDER BY at"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def to_dataframe(self, campaign_id: str):
        """Flat table of the campaign, for plotting and reporting."""
        import pandas as pd

        rows = [e.row() for e in self.load_campaign(campaign_id)]
        return pd.DataFrame(rows)
