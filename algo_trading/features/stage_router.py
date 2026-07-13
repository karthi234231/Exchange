"""
Stage-aware feature router.

Selects which feature modules run based on DATA_FEED and STAGE config.

Stage 1:  technical + statistical + custom (15-min, Shoonya feed)
Stage 2A: Stage 1 + ensemble ML inference
Stage 2B: Stage 2A + VPIN + Avellaneda-Stoikov (TrueData required)
Stage 3:  Stage 2B + Hawkes + TFI (sub-minute tick data required)
Stage 4:  Stage 3 + F&O features + sentiment

Feature vector grows with stage:
  Stage 1-2: 25 features (FEATURE_NAMES from engine.py)
  Stage 2B:  25 + 2 = 27 features (+ vpin_signal_enc, reservation_price_offset)
  Stage 3:   27 + 4 = 31 features (+ hawkes_ratio, hawkes_signal_enc, tfi, tfi_zscore)
  Stage 4:   31 + 7 = 38 features (+ pcr_oi, max_pain_dist_pct, iv_rank, iv_skew, atm_iv,
                                      sentiment_score, oi_buildup_enc)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import structlog

from algo_trading.features.engine import FEATURE_NAMES

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

_HAWKES_SIGNAL_ENC: dict[str, float] = {
    "normal": 0.0,
    "elevated": 0.5,
    "burst": 1.0,
}

_OI_BUILDUP_ENC: dict[str, float] = {
    "long_buildup": 1.0,
    "short_buildup": -1.0,
    "long_unwinding": -0.5,
    "short_unwinding": 0.5,
}

# ---------------------------------------------------------------------------
# STAGE_FEATURE_NAMES
# ---------------------------------------------------------------------------

_BASE_NAMES: list[str] = list(FEATURE_NAMES)  # 25 features from engine.py

_STAGE_2B_EXTRA: list[str] = [
    "vpin_signal_enc",
    "reservation_price_offset",
]

_STAGE_3_EXTRA: list[str] = [
    "hawkes_ratio",
    "hawkes_signal_enc",
    "tfi",
    "tfi_zscore",
]

_STAGE_4_EXTRA: list[str] = [
    "pcr_oi",
    "max_pain_dist_pct",
    "iv_rank",
    "iv_skew",
    "atm_iv",
    "sentiment_score",
    "oi_buildup_enc",
]

STAGE_FEATURE_NAMES: dict[str, list[str]] = {
    "1":  _BASE_NAMES,
    "2A": _BASE_NAMES,
    "2B": _BASE_NAMES + _STAGE_2B_EXTRA,
    "3":  _BASE_NAMES + _STAGE_2B_EXTRA + _STAGE_3_EXTRA,
    "4":  _BASE_NAMES + _STAGE_2B_EXTRA + _STAGE_3_EXTRA + _STAGE_4_EXTRA,
}

_VALID_STAGES: frozenset[str] = frozenset(STAGE_FEATURE_NAMES.keys())

# Features whose NaN indicates the statistical model is not yet warmed up.
# Substituting 0.0 for these creates a train/live distribution mismatch because
# 0.0 is a valid (bear-regime / near-zero-vol) value in training data. [P0-FIX-12]
_CRITICAL_FEATURES: frozenset[str] = frozenset({
    "feat_hmm_regime",
    "feat_hmm_prob",
    "feat_garch_vol",
})


# ---------------------------------------------------------------------------
# StageConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    """Configuration that controls which feature modules are active.

    Attributes:
        stage: Deployment stage — one of "1", "2A", "2B", "3", "4".
        data_feed: Data source — "shoonya" or "truedata".
        enable_hawkes: Activate the Hawkes point-process engine (Stage 3+).
        enable_tfi: Activate Trade Flow Imbalance computation (Stage 3+).
        enable_fo_features: Activate F&O chain feature engine (Stage 4).
        enable_sentiment: Activate the FinBERT sentiment aggregator (Stage 4).
        enable_microstructure: Activate VPIN + Avellaneda-Stoikov (Stage 2B+).
    """

    stage: str
    data_feed: str
    enable_hawkes: bool = False
    enable_tfi: bool = False
    enable_fo_features: bool = False
    enable_sentiment: bool = False
    enable_microstructure: bool = False

    def __post_init__(self) -> None:
        if self.stage not in _VALID_STAGES:
            raise ValueError(
                f"Invalid stage {self.stage!r}. Must be one of {sorted(_VALID_STAGES)}"
            )

        # Stage 2B+ requires TrueData
        if self.stage >= "2B" and self.data_feed != "truedata":
            raise ValueError(
                f"Stage {self.stage} requires data_feed='truedata'. "
                f"Got data_feed={self.data_feed!r}. "
                "Shoonya's throttled feed (1-3 snaps/sec) is insufficient for "
                "VPIN / Avellaneda-Stoikov computation. See ARCH-FIX-1."
            )

        # Stage 4: auto-enable F&O and sentiment
        if self.stage == "4":
            self.enable_fo_features = True
            self.enable_sentiment = True

    @classmethod
    def from_env(cls) -> "StageConfig":
        """Construct a StageConfig by reading environment variables.

        Environment variables:
            STAGE          — deployment stage (default: "1")
            DATA_FEED      — "shoonya" or "truedata" (default: "shoonya")
            ENABLE_HAWKES  — "true" / "1" to activate (default: false)
            ENABLE_TFI     — "true" / "1" to activate (default: false)
            ENABLE_FO      — "true" / "1" to activate (default: false)
            ENABLE_SENTIMENT — "true" / "1" to activate (default: false)
            ENABLE_MICROSTRUCTURE — "true" / "1" to activate (default: false)

        Returns:
            StageConfig populated from the current process environment.
        """
        def _bool(key: str) -> bool:
            return os.environ.get(key, "").lower() in ("true", "1", "yes")

        stage = os.environ.get("STAGE", "1").strip()
        data_feed = os.environ.get("DATA_FEED", "shoonya").strip().lower()

        return cls(
            stage=stage,
            data_feed=data_feed,
            enable_hawkes=_bool("ENABLE_HAWKES"),
            enable_tfi=_bool("ENABLE_TFI"),
            enable_fo_features=_bool("ENABLE_FO"),
            enable_sentiment=_bool("ENABLE_SENTIMENT"),
            enable_microstructure=_bool("ENABLE_MICROSTRUCTURE"),
        )


# ---------------------------------------------------------------------------
# FeatureRouter
# ---------------------------------------------------------------------------

class FeatureRouter:
    """Stage-aware feature computation router.

    Instantiates only the feature engines required by the active stage and
    dispatches ``compute()`` calls to them, returning a flat feature dict
    that the downstream inference engine can consume.

    Args:
        config: A :class:`StageConfig` that controls which engines are active.
    """

    def __init__(self, config: StageConfig) -> None:
        self._config = config

        # Always present — Stage 1-2A feature computers (lazy import pattern
        # mirrors engine.py to avoid circular imports).
        self._tech: Optional[Any] = None
        self._stat: Optional[Any] = None
        self._cust: Optional[Any] = None

        try:
            from algo_trading.features.technical import TechnicalFeatures  # type: ignore[import]
            self._tech = TechnicalFeatures()
        except ImportError:
            log.debug("stage_router.technical_unavailable")

        try:
            from algo_trading.features.statistical import StatisticalFeatures  # type: ignore[import]
            self._stat = StatisticalFeatures()
        except ImportError:
            log.debug("stage_router.statistical_unavailable")

        try:
            from algo_trading.features.custom import CustomFeatures  # type: ignore[import]
            self._cust = CustomFeatures()
        except ImportError:
            log.debug("stage_router.custom_unavailable")

        # Stage 2B+: microstructure (VPIN / Avellaneda-Stoikov)
        self._micro: Optional[Any] = None
        if config.enable_microstructure:
            try:
                from algo_trading.features.microstructure import MicrostructureFeatures  # type: ignore[import]
                self._micro = MicrostructureFeatures(adv=500_000, stage=config.stage)
                log.info("stage_router.microstructure_enabled")
            except ImportError:
                log.warning("stage_router.microstructure_unavailable")

        # Stage 3+: Hawkes process engine
        self._hawkes_engine: Optional[Any] = None
        if config.enable_hawkes:
            try:
                from algo_trading.features.hawkes import HawkesFeatureEngine  # type: ignore[import]
                self._hawkes_engine = HawkesFeatureEngine()
                log.info("stage_router.hawkes_enabled")
            except ImportError:
                log.warning("stage_router.hawkes_unavailable")

        # Stage 3+: TFI calculator + rolling window
        self._tfi_calc: Optional[Any] = None
        self._tfi_window: Optional[Any] = None
        if config.enable_tfi:
            try:
                from algo_trading.features.tfi import TFICalculator, TFIRollingWindow  # type: ignore[import]
                self._tfi_calc = TFICalculator()
                self._tfi_window = TFIRollingWindow()
                log.info("stage_router.tfi_enabled")
            except ImportError:
                log.warning("stage_router.tfi_unavailable")

        # Stage 4: F&O feature engine
        self._fo_engine: Optional[Any] = None
        if config.enable_fo_features:
            try:
                from algo_trading.features.fo_features import FOFeatureEngine  # type: ignore[import]
                self._fo_engine = FOFeatureEngine()
                log.info("stage_router.fo_features_enabled")
            except ImportError:
                log.warning("stage_router.fo_features_unavailable")

        # Stage 4: Sentiment aggregator
        self._sentiment: Optional[Any] = None
        if config.enable_sentiment:
            try:
                from algo_trading.models.sentiment import SentimentAggregator  # type: ignore[import]
                self._sentiment = SentimentAggregator()
                log.info("stage_router.sentiment_enabled")
            except ImportError:
                log.warning("stage_router.sentiment_unavailable")

        # P0-FIX-12: Track warmup state so the inference path can block until
        # HMM/GARCH are fitted and returning real values.
        self._inference_ready: bool = False
        self._warmup_nan_count: int = 0  # consecutive bars with critical NaNs

        log.info(
            "stage_router.init",
            stage=config.stage,
            data_feed=config.data_feed,
            hawkes=config.enable_hawkes,
            tfi=config.enable_tfi,
            fo_features=config.enable_fo_features,
            sentiment=config.enable_sentiment,
            microstructure=config.enable_microstructure,
            n_features=len(STAGE_FEATURE_NAMES[config.stage]),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        bar_df: Any,  # polars.DataFrame at runtime
        tick_timestamps_ns: Optional[np.ndarray] = None,
        ob_snapshot: Optional[Any] = None,
        fo_chain: Optional[Any] = None,
        symbol: str = "",
        inventory: int = 0,
        minutes_remaining: int = 375,
        price_change_pct: float = 0.0,
        oi_change_pct: float = 0.0,
    ) -> dict[str, float]:
        """Compute all active feature modules and return a flat feature dict.

        The feature dict always contains every name from
        ``STAGE_FEATURE_NAMES[config.stage]``.  Missing or erroneous values
        default to 0.0 — the router never raises.

        Args:
            bar_df: Polars DataFrame of recent OHLCV bars (Stage 1-2 features).
            tick_timestamps_ns: Sorted nanosecond tick timestamps for the
                Hawkes engine (Stage 3).  None disables Hawkes computation.
            ob_snapshot: :class:`~algo_trading.features.tfi.OrderBookSnapshot`
                for TFI computation (Stage 3).  None disables TFI.
            fo_chain: :class:`~algo_trading.data.fo_chain.OptionChain` for
                F&O feature computation (Stage 4).  None disables F&O.
            symbol: NSE instrument identifier (used for sentiment lookup).
            inventory: Current position in shares (for reservation price).
            minutes_remaining: Minutes left in the trading session.
            price_change_pct: % price change vs prior session (for OI buildup).
            oi_change_pct: % OI change vs prior session (for OI buildup).

        Returns:
            Dict mapping feature name → float32-compatible float.
            All values are finite (NaN/None coerced to 0.0).
            Call :meth:`is_inference_ready` before running inference — if
            critical statistical features (HMM, GARCH) are still NaN-padded,
            inference will receive out-of-distribution zeros. [P0-FIX-12]
        """
        feat: dict[str, float] = {}

        # ----------------------------------------------------------------
        # Stage 1-2: core technical / statistical / custom features
        # ----------------------------------------------------------------
        if self._tech is not None:
            if len(bar_df) >= 60:
                try:
                    feat.update(self._tech.compute_single_bar(bar_df))
                except Exception as exc:
                    log.warning("stage_router.technical_error", error=str(exc))
            else:
                log.debug(
                    "stage_router.technical_insufficient_bars",
                    bars=len(bar_df),
                    required=60,
                )

        if self._stat is not None:
            if len(bar_df) >= 60:
                try:
                    feat.update(self._stat.compute_single_bar(bar_df, current_ofi=0.0))
                except Exception as exc:
                    log.warning("stage_router.statistical_error", error=str(exc))
            else:
                log.debug(
                    "stage_router.statistical_insufficient_bars",
                    bars=len(bar_df),
                    required=60,
                )

        if self._cust is not None:
            if len(bar_df) >= 2:
                try:
                    feat.update(self._cust.compute_single_bar(bar_df))
                except Exception as exc:
                    log.warning("stage_router.custom_error", error=str(exc))

        # ----------------------------------------------------------------
        # Stage 2B+: microstructure (VPIN + Avellaneda-Stoikov)
        # ----------------------------------------------------------------
        if self._config.enable_microstructure and self._micro is not None:
            try:
                ms = self._micro.compute_all(
                    bars=bar_df,
                    inventory=inventory,
                    minutes_remaining=minutes_remaining,
                )
                # VPIN → encode into signal bucket (0=normal, 0.5=elevated, 1=toxic)
                vpin_val = ms.get("vpin", float("nan"))
                if vpin_val != vpin_val:  # NaN check
                    feat["vpin_signal_enc"] = 0.5  # treat NaN as elevated
                elif vpin_val >= 0.70:
                    feat["vpin_signal_enc"] = 1.0
                elif vpin_val >= 0.35:
                    feat["vpin_signal_enc"] = 0.5
                else:
                    feat["vpin_signal_enc"] = 0.0

                # Reservation price as an offset from the last close
                res_price = ms.get("reservation_price", 0.0)
                try:
                    last_close = float(bar_df["close"][-1])
                    feat["reservation_price_offset"] = res_price - last_close
                except Exception:
                    feat["reservation_price_offset"] = 0.0

            except Exception as exc:
                log.warning("stage_router.microstructure_error", error=str(exc))
                feat.setdefault("vpin_signal_enc", 0.0)
                feat.setdefault("reservation_price_offset", 0.0)

        # ----------------------------------------------------------------
        # Stage 3+: Hawkes process
        # ----------------------------------------------------------------
        if (
            self._config.enable_hawkes
            and self._hawkes_engine is not None
            and tick_timestamps_ns is not None
            and len(tick_timestamps_ns) > 0
        ):
            try:
                last_ts = int(tick_timestamps_ns[-1])
                last_price = 0.0
                try:
                    last_price = float(bar_df["close"][-1])
                except Exception:
                    pass
                hawkes_out = self._hawkes_engine.update(
                    timestamp_ns=last_ts,
                    price=last_price,
                    qty=1,
                )
                feat["hawkes_ratio"] = float(hawkes_out.get("hawkes_ratio", 1.0))
                raw_signal = str(hawkes_out.get("hawkes_signal", "normal"))
                feat["hawkes_signal_enc"] = _HAWKES_SIGNAL_ENC.get(raw_signal, 0.0)
            except Exception as exc:
                log.warning("stage_router.hawkes_error", error=str(exc))
                feat.setdefault("hawkes_ratio", 1.0)
                feat.setdefault("hawkes_signal_enc", 0.0)

        # ----------------------------------------------------------------
        # Stage 3+: TFI
        # ----------------------------------------------------------------
        if (
            self._config.enable_tfi
            and self._tfi_calc is not None
            and self._tfi_window is not None
            and ob_snapshot is not None
        ):
            try:
                tfi_result = self._tfi_calc.compute_tfi(ob_snapshot)
                if tfi_result is not None:
                    stats = self._tfi_window.add(tfi_result)
                    feat["tfi"] = float(tfi_result.tfi)
                    feat["tfi_zscore"] = float(stats.get("tfi_zscore", 0.0))
                else:
                    feat.setdefault("tfi", 0.0)
                    feat.setdefault("tfi_zscore", 0.0)
            except Exception as exc:
                log.warning("stage_router.tfi_error", error=str(exc))
                feat.setdefault("tfi", 0.0)
                feat.setdefault("tfi_zscore", 0.0)

        # ----------------------------------------------------------------
        # Stage 4: F&O features
        # ----------------------------------------------------------------
        if (
            self._config.enable_fo_features
            and self._fo_engine is not None
            and fo_chain is not None
        ):
            try:
                fo_result = self._fo_engine.compute(
                    chain=fo_chain,
                    price_change_pct=price_change_pct,
                    oi_change_pct=oi_change_pct,
                )
                feat["pcr_oi"] = float(fo_result.pcr_oi)
                feat["max_pain_dist_pct"] = float(fo_result.price_vs_max_pain_pct)
                feat["iv_rank"] = float(fo_result.iv_rank)
                feat["iv_skew"] = float(fo_result.iv_skew)
                feat["atm_iv"] = float(fo_result.atm_iv)
                feat["oi_buildup_enc"] = _OI_BUILDUP_ENC.get(
                    fo_result.oi_buildup_signal, 0.0
                )
            except Exception as exc:
                log.warning("stage_router.fo_features_error", error=str(exc))
                for name in ("pcr_oi", "max_pain_dist_pct", "iv_rank", "iv_skew", "atm_iv", "oi_buildup_enc"):
                    feat.setdefault(name, 0.0)

        # ----------------------------------------------------------------
        # Stage 4: Sentiment
        # ----------------------------------------------------------------
        if self._config.enable_sentiment and self._sentiment is not None:
            try:
                self._sentiment.poll()  # drain any pending results
                feat["sentiment_score"] = float(
                    self._sentiment.get_current_score(symbol)
                )
            except Exception as exc:
                log.warning("stage_router.sentiment_error", error=str(exc))
                feat.setdefault("sentiment_score", 0.0)

        # ----------------------------------------------------------------
        # NaN / None guard — coerce all values to 0.0 if non-finite.
        # P0-FIX-12: detect critical-feature NaN BEFORE substitution so we
        # can warn and block inference rather than silently pass 0.0.
        # ----------------------------------------------------------------
        critical_nans: list[str] = []
        for key in list(feat):
            v = feat[key]
            is_nan = False
            try:
                is_nan = (v != v)  # IEEE 754: NaN != NaN
            except Exception:
                is_nan = True
            if not is_nan and not _is_finite(feat.get(key, float("nan"))):
                is_nan = True

            if is_nan:
                if key in _CRITICAL_FEATURES:
                    critical_nans.append(key)
                feat[key] = 0.0

        if critical_nans:
            self._warmup_nan_count += 1
            if self._warmup_nan_count == 1 or self._warmup_nan_count % 50 == 0:
                log.warning(
                    "stage_router.critical_features_not_ready",
                    features=critical_nans,
                    consecutive_bars=self._warmup_nan_count,
                    note="Inference blocked — statistical models not yet fitted/warmed up",
                )
        else:
            if not self._inference_ready:
                self._inference_ready = True
                log.info(
                    "stage_router.inference_ready",
                    after_warmup_bars=self._warmup_nan_count,
                )
            self._warmup_nan_count = 0

        return feat

    def is_inference_ready(self) -> bool:
        """Return True once all critical features have been non-NaN at least once.

        Critical features (feat_hmm_regime, feat_hmm_prob, feat_garch_vol) require
        the HMM and GARCH models to be fitted and warmed up.  Callers should
        check this before running inference to avoid feeding zero-padded
        out-of-distribution feature vectors to the model. [P0-FIX-12]
        """
        return self._inference_ready

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of feature names for the active stage.

        Returns:
            List of feature name strings in canonical order.
        """
        return STAGE_FEATURE_NAMES[self._config.stage]

    def get_feature_vector(self, feature_dict: dict[str, float]) -> np.ndarray:
        """Convert a feature dict to a float32 array in canonical order.

        Missing keys default to 0.0.  Non-finite values are coerced to 0.0.

        Args:
            feature_dict: Dict produced by :meth:`compute`.

        Returns:
            numpy float32 array of length ``len(get_feature_names())``.
        """
        names = self.get_feature_names()
        vector = np.array(
            [_safe_float(feature_dict.get(name, 0.0)) for name in names],
            dtype=np.float32,
        )
        return vector


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_finite(value: float) -> bool:
    """Return True if *value* is a finite float (not NaN, not inf)."""
    try:
        import math
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any) -> float:
    """Convert *value* to a finite float, returning 0.0 on any failure."""
    try:
        v = float(value)
        import math
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0
