"""
algo_trading/market_data — Options chain snapshot infrastructure.

Phase 1 research infrastructure: stores, validates, and normalises
options chain snapshots for NSE F&O instruments.

Public API
----------
ChainSnapshot       : Point-in-time options chain data structure.
SnapshotArchiver    : Writes snapshots to a SnapshotStore at configurable frequency.
StalenessDetector   : Alerts when live quotes exceed age thresholds.
ChainNormalizer     : Normalises raw chain data and logs every correction.
"""
from __future__ import annotations

from algo_trading.market_data.chain_snapshot import ChainSnapshot, OptionQuote
from algo_trading.market_data.snapshot_archiver import SnapshotArchiver
from algo_trading.market_data.staleness_detector import StalenessDetector, StalenessConfig, StalenessAlert
from algo_trading.market_data.chain_normalizer import ChainNormalizer, NormalizationCorrection

__all__ = [
    "ChainSnapshot",
    "OptionQuote",
    "SnapshotArchiver",
    "StalenessDetector",
    "StalenessConfig",
    "StalenessAlert",
    "ChainNormalizer",
    "NormalizationCorrection",
]
