"""
trade_logger.py — Async QuestDB trade logger with full P&L attribution.

Implements spec [SURV-11].  Every trade — including aborted ones — is written
to QuestDB's ``trades`` table via the HTTP ILP (InfluxDB Line Protocol).
If QuestDB is unavailable the record is appended to a local JSONL fallback
file so no trade is ever lost.

Architecture
------------
* A single background :class:`threading.Thread` drains a
  :class:`queue.Queue` of :class:`TradeRecord` objects.
* Records are batched: flushed every ``BATCH_SIZE`` (50) records *or*
  every ``FLUSH_INTERVAL_SEC`` (5 s), whichever comes first.
* The public ``log_entry``, ``log_exit``, ``log_abort``, ``flush``, and
  ``close`` methods are all non-blocking (they put items on the queue and
  return immediately).

Usage::

    from algo_trading.execution.trade_logger import TradeLogger, TradeRecord
    from decimal import Decimal

    logger = TradeLogger()

    # Log an aborted signal (no order placed)
    logger.log_abort(
        symbol="RELIANCE",
        reason="stale_signal",
        signal_price=Decimal("2450.50"),
    )

    logger.close()   # blocks until queue is drained
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Optional

import structlog

from algo_trading.config import settings

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 50
FLUSH_INTERVAL_SEC: float = 5.0
_FALLBACK_LOG_DIR: Path = Path(settings.log_dir)
_FALLBACK_LOG_FILE: Path = _FALLBACK_LOG_DIR / "trade_log_fallback.jsonl"

# Sentinel value placed on the queue to signal the worker to stop
_STOP_SENTINEL: object = object()


# ---------------------------------------------------------------------------
# TradeRecord
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """Full P&L-attributed record for one trade event.

    A record is created at entry and updated (or superseded) at exit.  For
    trades that were never placed, ``direction`` is ``"ABORTED"`` and
    ``abort_reason`` is set.

    Attributes:
        symbol: NSE symbol, e.g. ``"RELIANCE"``.
        direction: ``"BUY"``, ``"SELL"``, or ``"ABORTED"``.
        quantity: Number of shares ordered.
        entry_price: Limit/market price at which the order was placed.
        exit_price: Price at which the position was closed.  ``Decimal("0")``
            until the trade is closed.
        entry_time_ns: ``time.time_ns()`` at order placement.
        exit_time_ns: ``time.time_ns()`` at position close.  ``0`` until set.
        gross_pnl: ``(exit_price - entry_price) × quantity`` (before costs).
        net_pnl: ``gross_pnl`` minus all transaction costs.
        slippage_bps: Actual fill vs expected price in basis points.
        spread_cost_bps: Half bid-ask spread at entry in basis points.
        latency_ms: Tick-to-order latency in milliseconds.
        partial_fill_ratio: ``filled_qty / ordered_qty`` in ``[0.0, 1.0]``.
        abort_reason: Why the trade was not placed.  One of
            ``"stale_signal"``, ``"price_moved"``, ``"risk_veto"``, or ``""``.
        label: Triple-barrier label from the training set: ``+1``, ``0``,
            or ``-1``.  Used for online model-performance tracking.
        model_version: The ``FrozenModelSpec`` version string.
        regime_at_entry: HMM regime at the time of entry.
        signal_confidence: ML ensemble confidence score at signal time.
        trade_id: UUID assigned at construction; used to correlate entry and
            exit records.
    """

    symbol: str
    direction: str  # "BUY" | "SELL" | "ABORTED"
    quantity: int
    entry_price: Decimal
    exit_price: Decimal = field(default_factory=lambda: Decimal("0"))
    entry_time_ns: int = 0
    exit_time_ns: int = 0
    gross_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    net_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    slippage_bps: Decimal = field(default_factory=lambda: Decimal("0"))
    spread_cost_bps: Decimal = field(default_factory=lambda: Decimal("0"))
    latency_ms: float = 0.0
    partial_fill_ratio: float = 1.0
    abort_reason: str = ""
    label: int = 0          # +1 / 0 / -1
    model_version: str = ""
    regime_at_entry: str = ""
    signal_confidence: float = 0.0
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict, converting Decimal fields to strings.

        Returns:
            Dictionary with all fields; ``Decimal`` values are rendered as
            strings to preserve precision.
        """
        raw = asdict(self)
        for key in (
            "entry_price",
            "exit_price",
            "gross_pnl",
            "net_pnl",
            "slippage_bps",
            "spread_cost_bps",
        ):
            raw[key] = str(raw[key])
        return raw

    def to_ilp_columns(self) -> dict:
        """Return ILP ``columns`` dict with QuestDB-compatible types.

        Decimal monetary values are cast to float (QuestDB ILP does not
        support arbitrary-precision; precision beyond 10 decimal places is
        not required for INR amounts).

        Returns:
            Dict mapping column name to Python value compatible with the
            ``questdb.ingress`` sender.
        """
        return {
            "trade_id": self.trade_id,
            "direction": self.direction,
            "quantity": self.quantity,
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "entry_time_ns": self.entry_time_ns,
            "exit_time_ns": self.exit_time_ns,
            "gross_pnl": float(self.gross_pnl),
            "net_pnl": float(self.net_pnl),
            "slippage_bps": float(self.slippage_bps),
            "spread_cost_bps": float(self.spread_cost_bps),
            "latency_ms": self.latency_ms,
            "partial_fill_ratio": self.partial_fill_ratio,
            "abort_reason": self.abort_reason,
            "label": self.label,
            "model_version": self.model_version,
            "regime_at_entry": self.regime_at_entry,
            "signal_confidence": self.signal_confidence,
        }


# ---------------------------------------------------------------------------
# Internal command types
# ---------------------------------------------------------------------------


@dataclass
class _FlushCommand:
    """Internal sentinel: drain queue and flush pending batch immediately."""
    pass


@dataclass
class _CloseCommand:
    """Internal sentinel: flush then stop the worker thread."""
    pass


# ---------------------------------------------------------------------------
# TradeLogger
# ---------------------------------------------------------------------------


class TradeLogger:
    """Async trade logger — never blocks the execution thread.

    Writes are queued immediately and dispatched by a background worker
    thread.  Batches are flushed to QuestDB every ``BATCH_SIZE`` records
    or every ``FLUSH_INTERVAL_SEC`` seconds, whichever comes first.

    If QuestDB is unavailable the entire batch is appended to a local JSONL
    fallback file.  No trade record is ever discarded.

    Attributes:
        questdb_host: Hostname of the QuestDB instance.
        questdb_ilp_port: HTTP ILP port (default 9000 per ``config.py``).
    """

    def __init__(
        self,
        questdb_host: str = settings.questdb_host,
        questdb_ilp_port: int = settings.questdb_ilp_port,
    ) -> None:
        """Initialise the logger and start the background writer thread.

        Args:
            questdb_host: QuestDB hostname.  Defaults to ``settings.questdb_host``.
            questdb_ilp_port: QuestDB HTTP ILP port.  Defaults to
                ``settings.questdb_ilp_port``.
        """
        self.questdb_host = questdb_host
        self.questdb_ilp_port = questdb_ilp_port

        self._queue: queue.Queue[object] = queue.Queue()
        self._closed = False

        # Ensure the fallback directory exists
        _FALLBACK_LOG_DIR.mkdir(parents=True, exist_ok=True)

        self._worker = threading.Thread(
            target=self._run,
            name="trade-logger-worker",
            daemon=True,
        )
        self._worker.start()
        log.info(
            "trade_logger_started",
            questdb_host=questdb_host,
            questdb_ilp_port=questdb_ilp_port,
            fallback_file=str(_FALLBACK_LOG_FILE),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_entry(
        self,
        record: TradeRecord,
    ) -> None:
        """Queue a trade entry record for async writing.

        The call returns immediately; the record is written by the background
        thread.

        Args:
            record: Fully populated :class:`TradeRecord` for the entry event.

        Raises:
            RuntimeError: If the logger has already been closed.
        """
        self._check_open()
        self._queue.put(record)
        log.debug(
            "trade_entry_queued",
            symbol=record.symbol,
            direction=record.direction,
            trade_id=record.trade_id,
            entry_price=str(record.entry_price),
        )

    def log_exit(
        self,
        entry_id: str,
        exit_price: Decimal,
        exit_time_ns: int,
        gross_pnl: Decimal,
        net_pnl: Decimal,
        slippage_bps: Decimal,
        partial_fill_ratio: float = 1.0,
        model_version: str = "",
    ) -> None:
        """Queue a synthetic exit record referencing the original entry.

        A minimal :class:`TradeRecord` is constructed with ``direction``
        set to ``"EXIT"`` and all P&L fields populated.  The ``entry_id``
        links it to the original entry record via ``trade_id``.

        Args:
            entry_id: ``trade_id`` of the original entry record.
            exit_price: Fill price at exit.
            exit_time_ns: ``time.time_ns()`` at position close.
            gross_pnl: Gross P&L (before costs).
            net_pnl: Net P&L after all transaction costs.
            slippage_bps: Exit-leg slippage in basis points.
            partial_fill_ratio: Fraction of ordered qty actually filled.
            model_version: Model version string for performance tracking.

        Raises:
            RuntimeError: If the logger has already been closed.
        """
        self._check_open()
        exit_record = TradeRecord(
            symbol="",          # filled by caller if needed; omitted here
            direction="EXIT",
            quantity=0,
            entry_price=Decimal("0"),
            exit_price=exit_price,
            exit_time_ns=exit_time_ns,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            slippage_bps=slippage_bps,
            partial_fill_ratio=partial_fill_ratio,
            model_version=model_version,
            trade_id=entry_id,  # re-use entry_id so records join in QuestDB
        )
        self._queue.put(exit_record)
        log.debug(
            "trade_exit_queued",
            entry_id=entry_id,
            exit_price=str(exit_price),
            net_pnl=str(net_pnl),
        )

    def log_abort(
        self,
        symbol: str,
        reason: str,
        signal_price: Decimal,
        model_version: str = "",
        regime_at_entry: str = "",
        signal_confidence: float = 0.0,
    ) -> None:
        """Queue an aborted-signal record (no order was placed).

        Args:
            symbol: NSE symbol.
            reason: One of ``"stale_signal"``, ``"price_moved"``,
                ``"risk_veto"``, or a custom string.
            signal_price: Mid-price at the time the signal was generated.
            model_version: Model version string.
            regime_at_entry: HMM regime at abort time.
            signal_confidence: ML ensemble confidence at abort time.

        Raises:
            RuntimeError: If the logger has already been closed.
        """
        self._check_open()
        record = TradeRecord(
            symbol=symbol,
            direction="ABORTED",
            quantity=0,
            entry_price=signal_price,
            entry_time_ns=time.time_ns(),
            abort_reason=reason,
            model_version=model_version,
            regime_at_entry=regime_at_entry,
            signal_confidence=signal_confidence,
        )
        self._queue.put(record)
        log.info(
            "trade_abort_queued",
            symbol=symbol,
            reason=reason,
            signal_price=str(signal_price),
        )

    def flush(self) -> None:
        """Request an immediate flush of any pending batch.

        This call is non-blocking: a flush command is placed on the queue.
        The background thread will process it in order.

        Raises:
            RuntimeError: If the logger has already been closed.
        """
        self._check_open()
        self._queue.put(_FlushCommand())

    def close(self) -> None:
        """Flush all pending records and stop the background thread.

        Blocks until the worker thread has processed all queued items and
        exited.  Safe to call multiple times (subsequent calls are no-ops).
        """
        if self._closed:
            return
        self._closed = True
        self._queue.put(_CloseCommand())
        self._worker.join(timeout=30)
        if self._worker.is_alive():
            log.error("trade_logger_worker_did_not_stop", timeout_sec=30)
        else:
            log.info("trade_logger_closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("TradeLogger has been closed; cannot accept new records.")

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background worker: batches records and writes to QuestDB or JSONL.

        The worker maintains a pending batch and flushes it when the batch
        reaches ``BATCH_SIZE`` or ``FLUSH_INTERVAL_SEC`` elapses since the
        last flush.
        """
        batch: list[TradeRecord] = []
        last_flush_time = time.monotonic()

        while True:
            timeout = max(
                0.0,
                FLUSH_INTERVAL_SEC - (time.monotonic() - last_flush_time),
            )
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                # Periodic flush
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush_time = time.monotonic()
                continue

            if isinstance(item, _CloseCommand):
                if batch:
                    self._flush_batch(batch)
                self._queue.task_done()
                break

            if isinstance(item, _FlushCommand):
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush_time = time.monotonic()
                self._queue.task_done()
                continue

            if isinstance(item, TradeRecord):
                batch.append(item)
                self._queue.task_done()
                if len(batch) >= BATCH_SIZE:
                    self._flush_batch(batch)
                    batch = []
                    last_flush_time = time.monotonic()
            else:
                # Unknown item type — log and discard
                log.warning("trade_logger_unknown_queue_item", item_type=type(item).__name__)
                self._queue.task_done()

    def _flush_batch(self, batch: list[TradeRecord]) -> None:
        """Attempt to write ``batch`` to QuestDB; fall back to JSONL on failure.

        Args:
            batch: List of :class:`TradeRecord` objects to write.
        """
        if not batch:
            return

        try:
            self._write_to_questdb(batch)
        except Exception as exc:
            log.warning(
                "questdb_write_failed_falling_back",
                error=str(exc),
                batch_size=len(batch),
                fallback_file=str(_FALLBACK_LOG_FILE),
            )
            self._write_to_jsonl(batch)

    def _write_to_questdb(self, batch: list[TradeRecord]) -> None:
        """Write a batch to QuestDB via the HTTP ILP sender.

        Imports ``questdb.ingress`` lazily so the module can be imported even
        when the package is not installed (fallback mode).

        Args:
            batch: Records to write.

        Raises:
            ImportError: If ``questdb-connect`` / ``questdb.ingress`` is not
                installed.
            Exception: Any network or ILP error from the sender.
        """
        from questdb.ingress import Sender, TimestampNanos  # type: ignore[import]

        conf = f"http::addr={self.questdb_host}:{self.questdb_ilp_port};"
        with Sender.from_conf(conf) as sender:
            for record in batch:
                ts = record.entry_time_ns if record.entry_time_ns else time.time_ns()
                sender.row(
                    "trades",
                    symbols={"symbol": record.symbol},
                    columns=record.to_ilp_columns(),
                    at=TimestampNanos(ts),
                )
            sender.flush()

        log.debug(
            "questdb_batch_written",
            batch_size=len(batch),
            host=self.questdb_host,
            port=self.questdb_ilp_port,
        )

    def _write_to_jsonl(self, batch: list[TradeRecord]) -> None:
        """Append a batch to the local JSONL fallback file.

        Each record is serialised via :meth:`TradeRecord.to_dict` and written
        as a single JSON line.  This method never raises; if writing fails the
        error is logged and the records are dropped (last resort).

        Args:
            batch: Records to append.
        """
        try:
            with _FALLBACK_LOG_FILE.open("a", encoding="utf-8") as fh:
                for record in batch:
                    fh.write(json.dumps(record.to_dict()) + "\n")
            log.info(
                "trade_fallback_written",
                batch_size=len(batch),
                fallback_file=str(_FALLBACK_LOG_FILE),
            )
        except Exception as exc:
            # Absolute last resort: emit to structlog ERROR
            log.error(
                "trade_fallback_write_failed",
                error=str(exc),
                batch_size=len(batch),
            )
