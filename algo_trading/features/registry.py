"""
algo_trading/features/registry.py — Feature metadata registry.

Maps each feature name to its metadata: formula, data source, compute latency,
lookback bars required, and leakage risk classification.  This registry is the
single source of truth for feature documentation and is used by:
  - Training pipeline to flag potentially leaky features
  - Live system to verify feature freshness budgets
  - Research to select features by latency or lookback constraints
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# LeakageRisk
# ---------------------------------------------------------------------------

class LeakageRisk(str, Enum):
    """Classification of forward-looking data leakage risk for a feature.

    Attributes:
        NONE: Definitely safe; computed purely from past data.
        LOW: Minor risk, e.g. VWAP uses intrabar cumulative volume.
        MEDIUM: Moderate risk, e.g. features using same-bar high/low extremes.
        HIGH: Should be reviewed before using in ML labels.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    """Immutable metadata descriptor for a single feature.

    Attributes:
        name: Canonical name; must match an entry in FEATURE_NAMES in engine.py.
        formula: Human-readable formula or description of the computation.
        source: Primary data source: "price", "volume", "order_book", or "macro".
        latency_ms: Expected compute latency on a mid-range i5 CPU (milliseconds).
        lookback_bars: Minimum number of historical bars needed to produce a
            non-NaN value.
        leakage_risk: Leakage risk classification from :class:`LeakageRisk`.
        notes: Optional free-text caveats or implementation references.
    """

    name: str
    formula: str
    source: str
    latency_ms: float
    lookback_bars: int
    leakage_risk: LeakageRisk
    notes: str = ""


# ---------------------------------------------------------------------------
# FeatureRegistry
# ---------------------------------------------------------------------------

class FeatureRegistry:
    """In-memory store of :class:`FeatureSpec` objects.

    Features are keyed by their canonical name.  The registry is typically
    built once at module import using the :data:`DEFAULT_REGISTRY` singleton.

    Example:
        >>> spec = DEFAULT_REGISTRY.get("rsi_14")
        >>> spec.lookback_bars
        14
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._specs: dict[str, FeatureSpec] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, spec: FeatureSpec) -> None:
        """Add or overwrite a feature spec in the registry.

        Args:
            spec: The :class:`FeatureSpec` to register.

        Raises:
            ValueError: If ``spec.name`` is empty.
        """
        if not spec.name:
            raise ValueError("FeatureSpec.name must not be empty.")
        if spec.name in self._specs:
            log.debug(
                "feature_registry.overwrite",
                name=spec.name,
            )
        self._specs[spec.name] = spec
        log.debug("feature_registry.registered", name=spec.name, source=spec.source)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> FeatureSpec:
        """Retrieve a feature spec by name.

        Args:
            name: Canonical feature name.

        Returns:
            The corresponding :class:`FeatureSpec`.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(
                f"Feature '{name}' is not registered. "
                f"Available: {sorted(self._specs)}"
            )

    def all_specs(self) -> list[FeatureSpec]:
        """Return all registered specs sorted by name.

        Returns:
            List of all :class:`FeatureSpec` instances.
        """
        return sorted(self._specs.values(), key=lambda s: s.name)

    def by_source(self, source: str) -> list[FeatureSpec]:
        """Return all specs whose source matches ``source`` (case-sensitive).

        Args:
            source: One of "price", "volume", "order_book", "macro".

        Returns:
            Filtered list of :class:`FeatureSpec` instances.
        """
        return [s for s in self._specs.values() if s.source == source]

    def high_leakage_risk(self) -> list[FeatureSpec]:
        """Return specs with leakage risk >= MEDIUM.

        Returns:
            List of specs with :attr:`LeakageRisk.MEDIUM` or
            :attr:`LeakageRisk.HIGH` classification.
        """
        risky = {LeakageRisk.MEDIUM, LeakageRisk.HIGH}
        return [s for s in self._specs.values() if s.leakage_risk in risky]

    def max_lookback(self) -> int:
        """Return the maximum lookback_bars across all registered specs.

        Returns:
            Integer maximum, or 0 if registry is empty.
        """
        if not self._specs:
            return 0
        return max(s.lookback_bars for s in self._specs.values())

    def total_latency_ms(self) -> float:
        """Return the sum of latency_ms across all registered specs.

        This represents a worst-case sequential compute budget.

        Returns:
            Float sum in milliseconds, or 0.0 if registry is empty.
        """
        return sum(s.latency_ms for s in self._specs.values())

    def validate_against_engine(self, engine_feature_names: list[str]) -> list[str]:
        """Find features present in engine but missing from this registry.

        Args:
            engine_feature_names: The ordered FEATURE_NAMES list from
                ``algo_trading.features.engine``.

        Returns:
            List of feature names that are in the engine list but have no
            corresponding registration in this registry.  An empty list
            means the registry is complete.
        """
        missing = [n for n in engine_feature_names if n not in self._specs]
        if missing:
            log.warning(
                "feature_registry.validation_gap",
                missing_count=len(missing),
                missing=missing,
            )
        else:
            log.info(
                "feature_registry.validation_ok",
                n_features=len(engine_feature_names),
            )
        return missing

    def __len__(self) -> int:
        """Return the number of registered specs."""
        return len(self._specs)

    def __repr__(self) -> str:
        return f"FeatureRegistry(n={len(self._specs)})"


# ---------------------------------------------------------------------------
# DEFAULT_REGISTRY — pre-populated with all 25 engine features
# Note: tod_sin and tod_cos share one conceptual feature (time-of-day) but
# are stored as two separate specs because they occupy two vector slots.
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY = FeatureRegistry()

_SPECS: list[FeatureSpec] = [
    # -- Technical features -------------------------------------------------
    FeatureSpec(
        name="rsi_14",
        formula="RSI(14) = 100 - 100 / (1 + avg_gain_14 / avg_loss_14)",
        source="price",
        latency_ms=0.10,
        lookback_bars=14,
        leakage_risk=LeakageRisk.NONE,
        notes="Wilder smoothing; uses prior bar close only.",
    ),
    FeatureSpec(
        name="macd_hist",
        formula="MACD_hist = EMA(12) - EMA(26) - EMA9(EMA12-EMA26)",
        source="price",
        latency_ms=0.08,
        lookback_bars=26,
        leakage_risk=LeakageRisk.NONE,
        notes="Signal line uses 9-bar EMA of the MACD line.",
    ),
    FeatureSpec(
        name="bb_pct_b",
        formula="BB %B = (close - lower_band) / (upper_band - lower_band); bands = SMA20 ± 2σ",
        source="price",
        latency_ms=0.08,
        lookback_bars=20,
        leakage_risk=LeakageRisk.NONE,
        notes="Returns NaN when upper == lower (zero-vol bar).",
    ),
    FeatureSpec(
        name="bb_width",
        formula="BB Width = (upper_band - lower_band) / SMA20",
        source="price",
        latency_ms=0.07,
        lookback_bars=20,
        leakage_risk=LeakageRisk.NONE,
        notes="Dimensionless volatility proxy.",
    ),
    FeatureSpec(
        name="atr_14",
        formula="ATR(14) = Wilder_EMA14(max(H-L, |H-C_prev|, |L-C_prev|))",
        source="price",
        latency_ms=0.08,
        lookback_bars=14,
        leakage_risk=LeakageRisk.MEDIUM,
        notes="Uses same-bar high and low — mild look-inside-bar risk.",
    ),
    FeatureSpec(
        name="adx_14",
        formula="ADX(14) = Wilder_EMA14(|+DI14 - -DI14| / (+DI14 + -DI14)) × 100",
        source="price",
        latency_ms=0.12,
        lookback_bars=14,
        leakage_risk=LeakageRisk.MEDIUM,
        notes="Uses same-bar H/L for directional movement calculation.",
    ),
    FeatureSpec(
        name="stoch_k",
        formula="%K = 100 × (close - lowest_low_14) / (highest_high_14 - lowest_low_14)",
        source="price",
        latency_ms=0.07,
        lookback_bars=14,
        leakage_risk=LeakageRisk.MEDIUM,
        notes="Uses same-bar close against 14-bar H/L range.",
    ),
    FeatureSpec(
        name="stoch_d",
        formula="%D = SMA3(%K)",
        source="price",
        latency_ms=0.05,
        lookback_bars=16,
        leakage_risk=LeakageRisk.MEDIUM,
        notes="3-bar smoothing of %K; lookback = 14 + 3 - 1.",
    ),
    FeatureSpec(
        name="obv_normalized",
        formula="OBV_norm = OBV / rolling_max_abs(OBV, 50); OBV += sign(Δclose) × volume",
        source="volume",
        latency_ms=0.09,
        lookback_bars=50,
        leakage_risk=LeakageRisk.LOW,
        notes="Volume accumulated intra-bar; normalised to [-1, 1] window.",
    ),
    FeatureSpec(
        name="supertrend_dir",
        formula="Supertrend(10, 3): direction = +1 (bullish) or -1 (bearish) vs trailing band",
        source="price",
        latency_ms=0.15,
        lookback_bars=10,
        leakage_risk=LeakageRisk.MEDIUM,
        notes="Uses same-bar ATR; direction is ±1 integer.",
    ),
    FeatureSpec(
        name="ema_cross_ratio",
        formula="ema_cross_ratio = EMA9 / EMA21 - 1.0",
        source="price",
        latency_ms=0.06,
        lookback_bars=21,
        leakage_risk=LeakageRisk.NONE,
        notes="Positive → bullish cross; negative → bearish. Ratio form is scale-invariant.",
    ),
    FeatureSpec(
        name="ichimoku_position",
        formula="ichimoku_position = (close - cloud_mid) / cloud_mid; cloud_mid = (senkou_a + senkou_b) / 2",
        source="price",
        latency_ms=0.20,
        lookback_bars=52,
        leakage_risk=LeakageRisk.NONE,
        notes="Requires 52-bar lookback for Senkou B. Chikou span projects 26 bars back (past data only).",
    ),
    # -- Statistical features -----------------------------------------------
    FeatureSpec(
        name="kalman_price",
        formula="Kalman filter residual: z_t - x_hat_t|t-1 (normalised by measurement noise sqrt)",
        source="price",
        latency_ms=0.25,
        lookback_bars=30,
        leakage_risk=LeakageRisk.NONE,
        notes="Online update; no look-ahead. Residual indicates trend deviation.",
    ),
    FeatureSpec(
        name="hmm_regime",
        formula="HMM regime = argmax P(S_t | O_1..t) via Viterbi on Gaussian HMM(3 states)",
        source="price",
        latency_ms=1.50,
        lookback_bars=60,
        leakage_risk=LeakageRisk.NONE,
        notes="Fitted offline; online decoding only. States: 0=bear, 1=sideways, 2=bull.",
    ),
    FeatureSpec(
        name="garch_vol",
        formula="GARCH(1,1) conditional vol forecast: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}",
        source="price",
        latency_ms=2.00,
        lookback_bars=60,
        leakage_risk=LeakageRisk.NONE,
        notes="Fitted offline; online one-step forecast only. Expressed as daily vol in bps.",
    ),
    # -- Custom / microstructure features -----------------------------------
    FeatureSpec(
        name="obi",
        formula="OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol) at top-of-book",
        source="order_book",
        latency_ms=0.05,
        lookback_bars=1,
        leakage_risk=LeakageRisk.LOW,
        notes="Snapshot at bar close; intrabar book pressure is not captured.",
    ),
    FeatureSpec(
        name="price_vol_divergence",
        formula="corr(price_ret_10, volume_change_10) × -1 (divergence = negative correlation)",
        source="volume",
        latency_ms=0.12,
        lookback_bars=10,
        leakage_risk=LeakageRisk.NONE,
        notes="Negative value flags distribution/accumulation divergence.",
    ),
    FeatureSpec(
        name="ha_direction",
        formula="Heikin-Ashi: ha_close=(O+H+L+C)/4; direction = sign(ha_close - ha_open)",
        source="price",
        latency_ms=0.06,
        lookback_bars=2,
        leakage_risk=LeakageRisk.MEDIUM,
        notes="Uses same-bar high/low for HA candle construction. Result is ±1.",
    ),
    FeatureSpec(
        name="pivot_dist_bps",
        formula="pivot_dist_bps = (close - pivot) / pivot × 10000; pivot = (H + L + C) / 3 (prior day)",
        source="price",
        latency_ms=0.04,
        lookback_bars=1,
        leakage_risk=LeakageRisk.NONE,
        notes="Prior-day pivot; no same-day leakage.",
    ),
    FeatureSpec(
        name="autocorr_lag1",
        formula="autocorr_lag1 = Pearson r(ret_t, ret_{t-1}) over rolling 20-bar window",
        source="price",
        latency_ms=0.10,
        lookback_bars=20,
        leakage_risk=LeakageRisk.NONE,
        notes="Positive → momentum; negative → mean-reversion.",
    ),
    FeatureSpec(
        name="relative_strength",
        formula="relative_strength = close_t / close_{t-20} - nifty_t / nifty_{t-20}",
        source="price",
        latency_ms=0.05,
        lookback_bars=20,
        leakage_risk=LeakageRisk.NONE,
        notes="Requires Nifty50 index bar in parallel. Uses prior close only.",
    ),
    # -- Time-of-day (two vector slots) ------------------------------------
    FeatureSpec(
        name="tod_sin",
        formula="tod_sin = sin(2π × minutes_since_open / 375)",
        source="macro",
        latency_ms=0.01,
        lookback_bars=1,
        leakage_risk=LeakageRisk.NONE,
        notes="375 = NSE session length in minutes. Deterministic — no leakage possible.",
    ),
    FeatureSpec(
        name="tod_cos",
        formula="tod_cos = cos(2π × minutes_since_open / 375)",
        source="macro",
        latency_ms=0.01,
        lookback_bars=1,
        leakage_risk=LeakageRisk.NONE,
        notes="Paired with tod_sin to encode cyclic time without discontinuity at session end.",
    ),
    # -- Microstructure: bid-ask spread -----------------------------------
    FeatureSpec(
        name="spread_percentile",
        formula="spread_percentile = percentile_rank(bid_ask_spread, rolling_100_bar_window)",
        source="order_book",
        latency_ms=0.08,
        lookback_bars=100,
        leakage_risk=LeakageRisk.LOW,
        notes="Tracks regime of market impact cost. Low spread → tighter execution cost.",
    ),
    FeatureSpec(
        name="vwap_dev_bps",
        formula="vwap_dev_bps = (close - vwap) / vwap × 10000; vwap uses cumulative intraday volume",
        source="volume",
        latency_ms=0.06,
        lookback_bars=1,
        leakage_risk=LeakageRisk.LOW,
        notes="Intraday VWAP uses all volume up to bar close including current bar.",
    ),
]

for _spec in _SPECS:
    DEFAULT_REGISTRY.register(_spec)

log.debug(
    "feature_registry.default_loaded",
    n_specs=len(DEFAULT_REGISTRY),
    max_lookback=DEFAULT_REGISTRY.max_lookback(),
    total_latency_ms=round(DEFAULT_REGISTRY.total_latency_ms(), 2),
)
