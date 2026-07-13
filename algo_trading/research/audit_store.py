"""
algo_trading/research/audit_store.py — Append-only experiment audit log.

Every backtest run is logged here with its git commit hash, config hash,
data hash, and full results.  Satisfies FO_SYSTEM_DESIGN.md Section 9.3:

    "Every experiment must be logged with parameters, data hash, results."

Design rules
------------
- Append-only: records are never modified or deleted after being written.
- Schema-versioned: the ``schema_version`` field allows forward-compatible
  additions without breaking existing records.
- Determinism check: two runs with the same config_hash + data_hash must
  produce identical results; ``check_determinism`` surfaces any violation.

Default storage is a JSON Lines file (one record per line). In production
this layer can be swapped for QuestDB or PostgreSQL while keeping the same
public API (``log_experiment``, ``load_all``).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_git_commit_hash() -> str | None:
    """Return the current HEAD commit hash, or None if not in a git repo.

    Uses a short timeout so it never blocks the calling thread.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _hash_data_config(config: dict) -> str:
    """Return a 16-character SHA-256 hex digest of *config* (sorted keys).

    Using ``default=str`` ensures non-JSON-serialisable values (e.g. dates,
    Decimals) are converted rather than raising.
    """
    serialised = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    """Immutable record of one backtest / optimisation run.

    All fields required to reproduce the exact same result at any future
    point in time.  The ``schema_version`` field allows safe additions: new
    optional fields can be appended to the dataclass without invalidating
    records written by earlier versions.
    """

    run_id: str                    # UUID4 string
    created_at: datetime           # UTC wall-clock time of the run
    git_commit: str | None         # HEAD commit hash when the run was executed
    config_hash: str               # SHA-256[:16] of strategy_params + data_params
    data_hash: str                 # SHA-256[:16] of data_params only
    strategy_name: str
    strategy_version: str          # semver string, e.g. "v1.0.0"
    strategy_params: dict          # every tuneable parameter used
    data_params: dict              # symbol, date range, quality threshold, …
    results: dict                  # Sharpe, PF, max_dd, win_rate, trade_count, …
    metadata: dict = field(default_factory=dict)  # any additional context
    schema_version: int = 1

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json_line(self) -> str:
        """Serialise to a single JSON Lines string (no trailing newline).

        ``datetime`` is stored as an ISO-8601 string via ``default=str``.
        """
        d = {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "strategy_params": self.strategy_params,
            "data_params": self.data_params,
            "results": self.results,
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }
        return json.dumps(d, default=str)

    @classmethod
    def from_json_line(cls, line: str) -> "ExperimentRecord":
        """Deserialise from a JSON Lines string.

        Unknown keys (from newer schema versions) are silently dropped so
        that this version of the code can read records written by a later
        version.
        """
        d = json.loads(line)
        d["created_at"] = datetime.fromisoformat(d["created_at"])

        # Forward-compat: only pass fields this version knows about.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class AuditStore:
    """Append-only store of experiment records.

    Every backtest run is logged here with:

    - **Git commit hash** — what version of the code produced this result.
    - **Config hash** — a deterministic fingerprint of all parameters used.
    - **Data hash** — a fingerprint of the data range and quality filters.
    - **Full results** — Sharpe, PF, max_dd, win_rate, trade_count, etc.

    Rules (FO_SYSTEM_DESIGN.md Section 9.3)
    ----------------------------------------
    1. **Append-only**: records are never modified or deleted.
    2. **Schema-versioned**: field additions do not break existing records.
    3. **Determinism check**: two runs with the same ``config_hash`` +
       ``data_hash`` must produce identical results.

    Default storage is a JSON Lines file (one record per line).
    Production deployments can swap this for QuestDB or PostgreSQL.

    Parameters
    ----------
    store_path:
        Path to the JSONL file.  The parent directory is created
        automatically if it does not exist.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        store_path: Path | str = "audit/experiments.jsonl",
    ) -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("audit_store.init", path=str(self.store_path))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        strategy_name: str,
        strategy_version: str,
        strategy_params: dict,
        data_params: dict,
        results: dict,
        metadata: dict | None = None,
    ) -> ExperimentRecord:
        """Log one experiment and return the created :class:`ExperimentRecord`.

        The write uses ``"a"`` (append) mode, which is safe for concurrent
        writers on most OS/filesystem combinations.  For strict
        multi-process safety use a lock file or a proper database.

        Parameters
        ----------
        strategy_name:
            Human-readable name of the strategy, e.g. ``"ema_cross_fo"``.
        strategy_version:
            Semver tag, e.g. ``"v1.0.0"``.
        strategy_params:
            All tuneable parameters (fast_period, slow_period, etc.).
        data_params:
            Data configuration: symbol, start/end date, quality threshold.
        results:
            Performance metrics dict (at minimum ``sharpe``, ``profit_factor``,
            ``total_trades``).
        metadata:
            Any additional context (notes, environment, feature_set name, …).

        Returns
        -------
        ExperimentRecord
            The newly created (and already persisted) record.
        """
        import uuid

        run_id = str(uuid.uuid4())
        git_commit = _get_git_commit_hash()
        config_hash = _hash_data_config({**strategy_params, **data_params})
        data_hash = _hash_data_config(data_params)

        record = ExperimentRecord(
            run_id=run_id,
            created_at=datetime.now(timezone.utc),
            git_commit=git_commit,
            config_hash=config_hash,
            data_hash=data_hash,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            strategy_params=strategy_params,
            data_params=data_params,
            results=results,
            metadata=metadata or {},
            schema_version=self.SCHEMA_VERSION,
        )

        # Append-only write — never truncate or seek.
        with open(self.store_path, "a", encoding="utf-8") as fh:
            fh.write(record.to_json_line() + "\n")

        log.info(
            "audit_store.logged",
            run_id=run_id,
            strategy=strategy_name,
            config_hash=config_hash,
            git=git_commit,
        )
        return record

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_all(self) -> list[ExperimentRecord]:
        """Return every record in the store.

        Malformed lines (truncated writes, encoding errors) are logged and
        skipped rather than raising, so a single bad record never blocks
        all downstream analysis.

        Returns an empty list if the store file does not yet exist.
        """
        if not self.store_path.exists():
            return []

        records: list[ExperimentRecord] = []
        with open(self.store_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(ExperimentRecord.from_json_line(line))
                except Exception as exc:
                    log.warning(
                        "audit_store.parse_error",
                        line_preview=line[:80],
                        error=str(exc),
                    )
        return records

    def find_by_config_hash(self, config_hash: str) -> list[ExperimentRecord]:
        """Return all records whose ``config_hash`` matches *config_hash*.

        Useful for grouping repeated runs of the same parameter set to
        verify determinism.
        """
        return [r for r in self.load_all() if r.config_hash == config_hash]

    def find_by_strategy(self, strategy_name: str) -> list[ExperimentRecord]:
        """Return all records for *strategy_name*, newest first."""
        records = [r for r in self.load_all() if r.strategy_name == strategy_name]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def check_determinism(self, config_hash: str) -> bool:
        """Return ``True`` if all runs with *config_hash* produced identical results.

        Determinism is defined as ``json.dumps(results, sort_keys=True)``
        being the same string for every run in the group.  Returns ``True``
        when fewer than two runs exist (nothing to compare).

        A ``False`` return indicates a non-determinism bug — typically
        uncontrolled randomness (unseeded RNG), data ordering instability,
        or a code change that was not reflected in a version bump.
        """
        runs = self.find_by_config_hash(config_hash)
        if len(runs) < 2:
            return True

        first = json.dumps(runs[0].results, sort_keys=True)
        for run in runs[1:]:
            if json.dumps(run.results, sort_keys=True) != first:
                log.error(
                    "audit_store.determinism_failure",
                    config_hash=config_hash,
                    run_count=len(runs),
                )
                return False
        return True

    def summarize(self, strategy_name: str | None = None) -> dict:
        """Return summary statistics across all logged experiments.

        Parameters
        ----------
        strategy_name:
            If provided, filter to records with this strategy name.

        Returns
        -------
        dict
            Keys: ``count``, ``avg_sharpe``, ``best_run_id``,
            ``strategy_name`` (if filtered).  Returns ``{"count": 0}`` when
            no matching records exist.
        """
        records = self.load_all()
        if strategy_name:
            records = [r for r in records if r.strategy_name == strategy_name]

        if not records:
            return {"count": 0}

        sharpes = [
            r.results["sharpe"]
            for r in records
            if "sharpe" in r.results and r.results["sharpe"] == r.results["sharpe"]  # NaN guard
        ]

        best = max(records, key=lambda r: r.results.get("sharpe", float("-inf")))

        return {
            "count": len(records),
            "strategy_name": strategy_name,
            "avg_sharpe": sum(sharpes) / len(sharpes) if sharpes else 0.0,
            "best_run_id": best.run_id,
        }
