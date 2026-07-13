"""
algo_trading/research — Phase 1 research infrastructure for NSE F&O backtesting.

Core components
---------------
ChainReplayEngine    — Deterministic replay of historical options chain snapshots.
FillSimulator        — Models fill quality against historical chain data (9 modes).
FillMode             — Enum of all supported fill simulation modes.
AuditStore           — Append-only experiment log (git hash + params + data hash + results).
ExperimentRecord     — Immutable record of one backtest run.
TradeRecord          — Single completed trade for metrics computation.
MetricsSuite         — Complete performance metrics with confidence intervals.
compute_metrics      — Compute MetricsSuite from a list of TradeRecord objects.
per_fold_stability   — Cross-fold stability statistics.
WalkForwardRunner    — Walk-forward test harness with purge + embargo.
WalkForwardConfig    — Configuration for walk-forward backtesting.
WalkForwardResult    — Aggregated results across all walk-forward folds.
FoldResult           — Results for one train-test fold.

Design contract
---------------
Replay is deterministic: same input -> same output, always.
No wall-clock dependency, no randomness beyond seeded RNG, no network calls
during replay.  All state comes from the data provider and config.
"""
from algo_trading.research.chain_replay_engine import ChainReplayEngine, ReplayConfig, ReplayEvent
from algo_trading.research.fill_simulator import FillSimulator, FillMode, SimulatedFill, FillSimulatorConfig
from algo_trading.research.audit_store import AuditStore, ExperimentRecord
from algo_trading.research.metrics_engine import (
    TradeRecord,
    MetricsSuite,
    compute_metrics,
    per_fold_stability,
)
from algo_trading.research.walk_forward_runner import (
    WalkForwardRunner,
    WalkForwardConfig,
    WalkForwardResult,
    FoldResult,
)

__all__ = [
    # Chain replay
    "ChainReplayEngine",
    "ReplayConfig",
    "ReplayEvent",
    # Fill simulation
    "FillSimulator",
    "FillMode",
    "SimulatedFill",
    "FillSimulatorConfig",
    # Audit store
    "AuditStore",
    "ExperimentRecord",
    # Metrics engine
    "TradeRecord",
    "MetricsSuite",
    "compute_metrics",
    "per_fold_stability",
    # Walk-forward runner
    "WalkForwardRunner",
    "WalkForwardConfig",
    "WalkForwardResult",
    "FoldResult",
]
