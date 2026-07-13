"""
features/microstructure.py — Microstructure features for Stage 2B+.

STAGE 2B+ ONLY. Do NOT activate for Stage 1 or Stage 2A.
Requires DATA_FEED=truedata and USE_MICROSTRUCTURE=true in .env (ARCH-FIX-1).

Implements (per SURV-9):
  - VPIN (Volume-Synchronised Probability of Informed Trading)
    Valid at 15-min volume bars. Bucket size = ADV / 50.
  - Avellaneda-Stoikov reservation price
    Adjusts LIMIT order placement based on inventory and residual session time.
  - Kyle Lambda (price impact coefficient)
    Higher = less liquid = wider effective spread.

Excluded at 15-min (see SURV-9):
  - Hawkes Process  — inter-arrival times meaningless at 15-min aggregation
  - Trade Flow Imbalance (TFI) — signal decays in seconds, not minutes

At Stage 2A with Shoonya throttled feed only Kyle Lambda is valid.
VPIN requires unthrottled tick-by-tick volume bars from TrueData.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import polars as pl
import structlog
from scipy.stats import norm  # type: ignore[import]

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VPIN_WINDOW: int = 50                  # Rolling window: 50 consecutive buckets
_VPIN_DEFAULT_BUCKETS: int = 50         # ADV / 50 = default bucket size
_VPIN_ELEVATED_THRESHOLD: float = 0.35  # Reduce position 50%
_VPIN_TOXIC_THRESHOLD: float = 0.70     # Suppress all new entries

_KYLE_LAMBDA_WINDOW: int = 20           # 20 bars for covariance estimation
_KYLE_MIN_BARS: int = 5                 # Minimum bars for a non-NaN estimate

# NSE full session = 09:15 → 15:30 = 375 minutes
_SESSION_TOTAL_MINUTES: int = 375

# Avellaneda-Stoikov default risk aversion for NSE retail
_GAMMA_DEFAULT: float = 0.1


# ---------------------------------------------------------------------------
# MicrostructureFeatures
# ---------------------------------------------------------------------------

class MicrostructureFeatures:
    """Microstructure feature computation for Stage 2B+ only.

    Provides VPIN, Avellaneda-Stoikov reservation price, and Kyle Lambda at
    the 15-minute timeframe.

    Stage guard: if the instance was created with ``stage="2A"`` (or lower)
    and ``compute_vpin`` is called with ``use_vpin=True``, a ``RuntimeError``
    is raised immediately. Kyle Lambda remains available at Stage 2A because
    it does not require unthrottled L2 data.

    Args:
        adv: Average Daily Volume (shares) for the traded symbol.
            Used to compute the default VPIN bucket size (ADV / 50).
        stage: Current deployment stage string, e.g. ``"2A"`` or ``"2B"``.
            Defaults to ``"2A"`` to be safe — must be set to ``"2B"`` or
            higher to enable VPIN.

    Example::

        ms = MicrostructureFeatures(adv=500_000, stage="2B")
        vpin = ms.compute_vpin(volume_bars_df)
        mult, label = ms.get_vpin_signal(vpin)
        r = ms.compute_reservation_price(
            mid=1500.0, inventory=100, sigma_daily=0.015,
            minutes_remaining=90,
        )
        lam = ms.compute_kyle_lambda(bars_df)
    """

    def __init__(self, adv: float, stage: str = "2A") -> None:
        self._adv = adv
        self._stage = stage
        log.info(
            "microstructure_features.init",
            adv=adv,
            stage=stage,
        )

    # ------------------------------------------------------------------
    # Stage guard helper
    # ------------------------------------------------------------------

    def _require_stage_2b(self, feature: str) -> None:
        """Raise RuntimeError if the current stage is below 2B.

        Args:
            feature: Name of the feature being checked (for error message).

        Raises:
            RuntimeError: If stage is not 2B or higher.
        """
        # Stages are compared lexicographically: "2A" < "2B" < "3"
        if self._stage < "2B":
            raise RuntimeError(
                f"{feature} requires Stage 2B+ and DATA_FEED=truedata. "
                f"Current stage is {self._stage!r}. "
                "Shoonya's throttled feed (1-3 snaps/sec) produces ghost liquidity "
                "readings in VPIN bucket BVC classification. "
                "See ARCH-FIX-1 and SURV-9 for details."
            )

    # ------------------------------------------------------------------
    # VPIN
    # ------------------------------------------------------------------

    def compute_vpin(
        self,
        volume_bars: pl.DataFrame,
        bucket_size: Optional[int] = None,
    ) -> float:
        """Compute VPIN over the last 50 volume buckets using BVC classification.

        Uses the Bulk Volume Classification (BVC) method to split each bar's
        volume into buy- and sell-initiated fractions::

            tau_b = (close_b - close_{b-1}) / sigma_daily
            V_buy  = V_b × CDF(tau_b)
            V_sell = V_b × (1 - CDF(tau_b))

        Then VPIN is the average absolute order imbalance across the window::

            VPIN = (1/n) × Σ_b |V_buy(b) - V_sell(b)| / V_bucket

        Requires Stage 2B+. Raises RuntimeError if called at Stage 2A.

        Args:
            volume_bars: Polars DataFrame with at minimum ``close`` and
                ``volume`` columns. Rows are sorted oldest → newest.
                Should contain at least ``_VPIN_WINDOW + 1`` rows.
            bucket_size: Volume bucket size in shares. Defaults to
                ``int(ADV / 50)`` if not supplied.

        Returns:
            VPIN as a float in ``[0.0, 1.0]``. Returns ``float("nan")`` if
            there are fewer than 2 bars (cannot compute price change).

        Raises:
            RuntimeError: If stage < "2B".
            ValueError: If required columns are missing from ``volume_bars``.
        """
        self._require_stage_2b("VPIN")
        _require_columns(volume_bars, ["close", "volume"], "compute_vpin")

        effective_bucket: int = bucket_size if bucket_size is not None else max(
            1, int(self._adv / _VPIN_DEFAULT_BUCKETS)
        )

        closes = volume_bars["close"].to_numpy().astype("float64")
        volumes = volume_bars["volume"].to_numpy().astype("float64")

        n_bars = len(closes)
        if n_bars < 2:
            log.warning("compute_vpin.insufficient_bars", n_bars=n_bars)
            return float("nan")

        # Price changes (skip first bar — no prior close)
        delta_p = np.diff(closes)  # shape: (n_bars - 1,)
        vols = volumes[1:]          # align with delta_p

        # Estimate sigma_daily from the price change series
        sigma = float(np.std(delta_p)) if len(delta_p) > 1 else 1.0
        if sigma < 1e-12:
            sigma = 1.0  # Avoid division by zero on flat data

        # BVC: compute tau, then buy/sell fractions
        tau = delta_p / sigma                    # normalised price change
        cdf_tau = norm.cdf(tau)                  # Φ(tau)
        v_buy = vols * cdf_tau
        v_sell = vols * (1.0 - cdf_tau)

        # Take only the last _VPIN_WINDOW buckets
        window_start = max(0, len(v_buy) - _VPIN_WINDOW)
        v_buy_w = v_buy[window_start:]
        v_sell_w = v_sell[window_start:]

        if len(v_buy_w) == 0:
            return float("nan")

        imbalances = np.abs(v_buy_w - v_sell_w)
        vpin = float(np.mean(imbalances) / effective_bucket) if effective_bucket > 0 else float("nan")

        # Clamp to [0, 1] — BVC can occasionally exceed 1.0 on extreme moves
        vpin = min(max(vpin, 0.0), 1.0)

        log.debug(
            "compute_vpin.result",
            vpin=round(vpin, 4),
            n_buckets=len(v_buy_w),
            bucket_size=effective_bucket,
        )
        return vpin

    # ------------------------------------------------------------------
    def get_vpin_signal(self, vpin: float) -> tuple[float, str]:
        """Map VPIN value to a position-size multiplier and regime label.

        Thresholds (SURV-9)::

            vpin < 0.35          → (1.0, "normal")    — full position size
            0.35 ≤ vpin < 0.70   → (0.5, "elevated")  — halve position size
            vpin ≥ 0.70          → (0.0, "toxic")      — suppress all entries

        Args:
            vpin: VPIN value in ``[0.0, 1.0]``.

        Returns:
            Tuple of ``(size_multiplier, regime_label)`` where
            ``size_multiplier`` is 1.0, 0.5, or 0.0 and ``regime_label`` is
            one of ``"normal"``, ``"elevated"``, or ``"toxic"``.
        """
        if math.isnan(vpin):
            # Treat NaN conservatively as elevated
            log.warning("get_vpin_signal.nan_vpin_treated_as_elevated")
            return (0.5, "elevated")

        if vpin >= _VPIN_TOXIC_THRESHOLD:
            log.info("get_vpin_signal.toxic", vpin=round(vpin, 4))
            return (0.0, "toxic")
        elif vpin >= _VPIN_ELEVATED_THRESHOLD:
            log.info("get_vpin_signal.elevated", vpin=round(vpin, 4))
            return (0.5, "elevated")
        else:
            return (1.0, "normal")

    # ------------------------------------------------------------------
    # Avellaneda-Stoikov
    # ------------------------------------------------------------------

    def compute_reservation_price(
        self,
        mid: float,
        inventory: int,
        sigma_daily: float,
        minutes_remaining: int,
        gamma: float = _GAMMA_DEFAULT,
    ) -> float:
        """Compute the Avellaneda-Stoikov reservation (indifference) price.

        The reservation price is the mid-price adjusted for the cost of
        holding current inventory over the remaining session time::

            r(s, t) = s - q × γ × σ² × (T - t)

        Where:
            s = current mid-price
            q = current inventory (positive = long, negative = short)
            γ = risk aversion parameter (default 0.1 for NSE retail)
            σ = realised daily volatility (as a fraction, e.g. 0.015 = 1.5%)
            T - t = remaining session time as a fraction of the full session

        Use the reservation price to offset LIMIT order placement::

            buy_limit  = reservation_price - spread / 2
            sell_limit = reservation_price + spread / 2

        Valid at any timeframe — no Stage guard needed for this method.

        Args:
            mid: Current mid-price (e.g. (best_bid + best_ask) / 2).
            inventory: Current inventory in shares. Positive = long position.
            sigma_daily: Realised daily volatility as a fraction
                (e.g. ``0.015`` for 1.5%). Typically from GARCH forecast.
            minutes_remaining: Minutes left in the trading session. Must be
                in ``[0, _SESSION_TOTAL_MINUTES]``. Values outside are clamped.
            gamma: Risk aversion parameter (default 0.1 for NSE retail).
                Higher gamma → greater penalty for holding inventory.

        Returns:
            Reservation price as a float. Equal to ``mid`` when inventory
            is zero or when ``sigma_daily`` is zero.
        """
        # Clamp minutes_remaining to valid session range
        t_remaining = max(0, min(minutes_remaining, _SESSION_TOTAL_MINUTES))
        # Normalise to [0, 1] fraction of full session
        time_frac: float = t_remaining / _SESSION_TOTAL_MINUTES

        sigma_sq: float = sigma_daily ** 2

        # r(s,t) = s - q × γ × σ² × (T-t)
        reservation: float = mid - inventory * gamma * sigma_sq * time_frac

        log.debug(
            "compute_reservation_price",
            mid=mid,
            inventory=inventory,
            sigma_daily=sigma_daily,
            minutes_remaining=t_remaining,
            gamma=gamma,
            reservation=round(reservation, 4),
        )
        return reservation

    # ------------------------------------------------------------------
    # Kyle Lambda
    # ------------------------------------------------------------------

    def compute_kyle_lambda(self, bars: pl.DataFrame) -> float:
        """Compute Kyle's lambda (price impact coefficient) over the last 20 bars.

        Lambda measures the price impact per unit of order flow::

            λ = cov(Δp, order_flow) / var(order_flow)

        Order flow proxy uses signed volume (BVC approximation at 15-min)::

            signed_volume_b = volume_b × sign(close_b - open_b)

        Higher lambda indicates lower liquidity and a wider effective spread.
        Valid at 15-min timeframe — no Stage guard. However, accuracy is
        materially improved with TrueData L2 (unthrottled) versus Shoonya.

        Args:
            bars: Polars DataFrame with ``open``, ``close``, and ``volume``
                columns. Rows sorted oldest → newest. Should contain at least
                ``_KYLE_MIN_BARS`` rows; returns ``float("nan")`` otherwise.

        Returns:
            Kyle lambda as a float. Returns ``float("nan")`` if there are
            fewer than ``_KYLE_MIN_BARS`` rows, or if order-flow variance
            is zero (perfectly balanced flow or flat price).

        Raises:
            ValueError: If required columns are missing from ``bars``.
        """
        _require_columns(bars, ["open", "close", "volume"], "compute_kyle_lambda")

        n = len(bars)
        if n < _KYLE_MIN_BARS:
            log.warning(
                "compute_kyle_lambda.insufficient_bars",
                n_bars=n,
                required=_KYLE_MIN_BARS,
            )
            return float("nan")

        # Use last _KYLE_LAMBDA_WINDOW bars
        window_start = max(0, n - _KYLE_LAMBDA_WINDOW)
        opens = bars["open"].slice(window_start).to_numpy().astype("float64")
        closes = bars["close"].slice(window_start).to_numpy().astype("float64")
        volumes = bars["volume"].slice(window_start).to_numpy().astype("float64")

        # Signed volume proxy: volume × sign(close - open)
        direction = np.sign(closes - opens)
        signed_volume = volumes * direction  # shape: (window,)

        # Price changes — need at least 2 observations
        if len(closes) < 2:
            return float("nan")
        delta_p = np.diff(closes)       # shape: (window - 1,)
        order_flow = signed_volume[1:]  # align to same length as delta_p

        var_of = float(np.var(order_flow, ddof=1))
        if var_of < 1e-16:
            # Perfectly balanced flow — lambda is undefined, return NaN
            log.debug("compute_kyle_lambda.zero_variance")
            return float("nan")

        cov_matrix = np.cov(delta_p, order_flow)  # 2×2 matrix
        cov_dp_of: float = float(cov_matrix[0, 1])
        kyle_lambda: float = cov_dp_of / var_of

        log.debug(
            "compute_kyle_lambda.result",
            lambda_=round(kyle_lambda, 6),
            n_bars=len(delta_p),
        )
        return kyle_lambda

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def compute_all(
        self,
        bars: pl.DataFrame,
        inventory: int,
        minutes_remaining: int,
    ) -> dict[str, float]:
        """Compute all microstructure metrics for the most recent bar.

        Convenience method that calls all feature computations in sequence and
        returns a flat dict for downstream consumption (e.g. SignalRouter).

        VPIN requires Stage 2B+. If the stage check fails, ``"vpin"`` and
        ``"vpin_size_mult"`` are set to ``float("nan")`` rather than raising,
        so the SignalRouter can still use Kyle Lambda and reservation price at
        Stage 2A.

        Args:
            bars: Polars DataFrame with ``open``, ``high``, ``low``, ``close``,
                and ``volume`` columns sorted oldest → newest. At minimum 20
                rows recommended.
            inventory: Current inventory in shares (positive = long).
            minutes_remaining: Minutes remaining in the trading session.

        Returns:
            Dictionary with keys:

            - ``"vpin"`` (float): VPIN value ``[0.0, 1.0]`` or NaN at Stage 2A.
            - ``"vpin_size_mult"`` (float): Position size multiplier
              (1.0 / 0.5 / 0.0) based on VPIN regime; NaN if VPIN unavailable.
            - ``"reservation_price"`` (float): A-S reservation price.
            - ``"kyle_lambda"`` (float): Kyle lambda price impact coefficient.
        """
        result: dict[str, float] = {
            "vpin": float("nan"),
            "vpin_size_mult": float("nan"),
            "reservation_price": float("nan"),
            "kyle_lambda": float("nan"),
        }

        # --- VPIN (Stage 2B+ only) ---
        try:
            vpin = self.compute_vpin(bars)
            size_mult, _label = self.get_vpin_signal(vpin)
            result["vpin"] = vpin
            result["vpin_size_mult"] = size_mult
        except RuntimeError as exc:
            # Stage < 2B — expected; Kyle Lambda + reservation price still work
            log.info("compute_all.vpin_skipped", reason=str(exc))
        except Exception as exc:  # noqa: BLE001
            log.warning("compute_all.vpin_error", error=str(exc))

        # --- Kyle Lambda (valid at any stage) ---
        try:
            result["kyle_lambda"] = self.compute_kyle_lambda(bars)
        except Exception as exc:  # noqa: BLE001
            log.warning("compute_all.kyle_lambda_error", error=str(exc))

        # --- Reservation price (valid at any stage) ---
        try:
            if "close" in bars.columns and len(bars) >= 1:
                # Mid-price proxy: use last close (L2 mid preferred if available)
                last_close = float(bars["close"][-1])

                # Estimate sigma_daily from rolling standard deviation of returns
                if len(bars) >= 2:
                    closes_np = bars["close"].to_numpy().astype("float64")
                    log_rets = np.diff(np.log(np.where(closes_np > 0, closes_np, np.nan)))
                    valid_rets = log_rets[~np.isnan(log_rets)]
                    # Scale to daily: 15-min bars → ~25 bars per session
                    sigma_daily = float(np.std(valid_rets) * math.sqrt(25)) if len(valid_rets) > 1 else 0.015
                else:
                    sigma_daily = 0.015  # Sensible default for NSE large-cap

                result["reservation_price"] = self.compute_reservation_price(
                    mid=last_close,
                    inventory=inventory,
                    sigma_daily=sigma_daily,
                    minutes_remaining=minutes_remaining,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("compute_all.reservation_price_error", error=str(exc))

        log.debug("compute_all.result", **{k: round(v, 4) for k, v in result.items() if not math.isnan(v)})
        return result


# ---------------------------------------------------------------------------
# Stage 3+ stubs — Hawkes / TFI (NOT for use at 15-min)
# ---------------------------------------------------------------------------

def compute_hawkes_branching_ratio(*args: object, **kwargs: object) -> float:
    """Stub for Hawkes process branching ratio — Stage 3+ only.

    This function intentionally always raises ``NotImplementedError``.

    At the 15-minute timeframe, inter-arrival times between trades aggregate
    over hundreds of individual events, making the self-excitation structure
    of the Hawkes process unobservable. The branching ratio would be
    systematically underestimated and the confidence intervals would be
    enormous. See SURV-9 for the full explanation.

    Raises:
        NotImplementedError: Always. This feature requires Stage 3+ with
            sub-minute tick data.
    """
    raise NotImplementedError(
        "Hawkes process is Stage 3+ only (requires sub-minute tick data). "
        "At 15-min timeframe, inter-arrival times are meaningless. "
        "See SURV-9 for explanation."
    )


def compute_trade_flow_imbalance(*args: object, **kwargs: object) -> float:
    """Stub for Trade Flow Imbalance (TFI) — Stage 3+ only.

    This function intentionally always raises ``NotImplementedError``.

    TFI is a high-frequency signal with a half-life measured in seconds.
    At 15-minute aggregation the signal has fully decayed and what remains
    is noise. See SURV-9 for the full explanation.

    Raises:
        NotImplementedError: Always. This feature requires Stage 3+ with
            sub-minute tick data.
    """
    raise NotImplementedError(
        "Trade Flow Imbalance (TFI) is Stage 3+ only. "
        "The signal decays in seconds — at 15-min aggregation it is noise. "
        "See SURV-9 for explanation."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_columns(
    df: pl.DataFrame,
    required: list[str],
    caller: str,
) -> None:
    """Raise ValueError if any required column is absent from the DataFrame.

    Args:
        df: Polars DataFrame to validate.
        required: List of column names that must be present.
        caller: Name of the calling method (for error message context).

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{caller}: DataFrame missing required column(s): {missing}. "
            f"Available columns: {df.columns}"
        )
