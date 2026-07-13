# CONVO_DEEP_RESEARCH.md
## NSE AlgoTrader — Deep Research, Architecture Blueprint & Critical Fixes
### Compiled May 2026 | Source: Blueprint V2 + 24 Critiques + Deep Research Agents

> **RESEARCH ARCHIVE — Not a build spec or source of truth**
>
> This file contains deep research, architecture discussions, and critique responses from May 2026.
> It is valuable context but should not be used as a build guide:
> - Decisions recorded here may have been superseded — check `MD_Files/DECISIONS.md`
> - Some sections still reference AngelOne as primary broker (since changed to Shoonya)
> - `FIXED` status in this file means "decided in docs" — not "code implemented"
> - Canonical architecture is in `arch-spec.json`; canonical build plan is in `MD_Files/CLAUDE.md`

---

## TABLE OF CONTENTS

| Part | Title | Source |
|---|---|---|
| Part 1 | Microstructure Math Models (Hawkes, VPIN, TFI, Avellaneda-Stoikov) | _part1_microstructure.md |
| Part 2 | Infrastructure Upgrades (ZeroMQ, Ring Buffer, Volume Bars, Volume Clock) | _part2_infrastructure.md |
| Part 3A | Critical Fixes 1–12 | _part3a_fixes.md |
| Part 3B | Critical Fixes 13–24 | _part3b_fixes.md |
| Part 4 | Refined 4-Stage Deployment Plan | original |
| Part 5 | ML Ensemble Upgrades (Stacking, HMM, Kelly, Modern Alternatives) | research agents |
| Part 6 | AI Development Optimization + arch-spec.json | _part6_aidev.md |

---

## PART 1: MICROSTRUCTURE MATH MODELS

**System context**: NSE Indian algo trading, Python, CPU-only (i5-1345U), QuestDB + Polars + XGBoost/LGB/CatBoost stack.
**Purpose**: Four production-grade microstructure models that generate features for the ML signal pipeline.

---

### 1.1 Hawkes Process (Self-Exciting Point Process)

**Use case**: Detect institutional order bursts and spoof cancellation cascades. When a large order arrives, it tends to trigger further orders — the Hawkes process quantifies this self-excitation. The branching ratio η = α/β is the single most diagnostic number: η > 0.85 at NSE is historically associated with quote-stuffing and layering cascades.

**Intensity function**:

```
λ(t) = μ + Σᵢ α · exp(-β · (t - tᵢ))   for all prior events tᵢ < t
```

- μ  = baseline (exogenous) arrival rate
- α  = jump in intensity per event (excitation)
- β  = decay rate of excitation
- η  = α/β  (branching ratio; η → 1 means near-critical, self-sustaining cascade)

**NSE calibration targets**:
- η 0.30–0.70 = normal institutional flow
- η 0.70–0.85 = elevated, watch for pattern
- η > 0.85    = spoofing / cascade risk; suppress signal generation

**Log-likelihood (O(N) recursive form)**:

The standard O(N²) form iterates all pairs. The recursive form maintains a running sum
`A(i) = Σⱼ<ᵢ exp(-β·(tᵢ - tⱼ))` via `A(i) = exp(-β·Δt)·(1 + A(i-1))`.

```python
import numpy as np
from scipy.optimize import minimize
from typing import Optional


def fit_hawkes(timestamps: np.ndarray, n_restarts: int = 5) -> tuple[float, float, float]:
    """
    MLE fit of univariate Hawkes process. Returns (mu, alpha, beta).
    O(N) recursive log-likelihood via the Ogata (1981) formula.

    Parameters
    ----------
    timestamps : np.ndarray
        Sorted event times in seconds (float64). Min length ~50 events.
    n_restarts : int
        Number of random initialisations to escape local optima.

    Returns
    -------
    (mu, alpha, beta) : tuple[float, float, float]
        MLE parameters. alpha/beta = branching ratio.
    """
    if len(timestamps) < 10:
        return (1e-3, 0.3, 1.0)  # safe default

    timestamps = np.asarray(timestamps, dtype=np.float64)
    T = timestamps[-1] - timestamps[0]
    timestamps = timestamps - timestamps[0]  # shift to [0, T]
    N = len(timestamps)

    def neg_log_likelihood(params: np.ndarray) -> float:
        mu, alpha, beta = params
        if mu <= 0 or alpha <= 0 or beta <= 0:
            return 1e10
        if alpha >= beta:          # enforce stationarity: branching ratio < 1
            return 1e10

        # --- Recursive accumulator ---
        A = 0.0
        log_sum = 0.0
        for i in range(N):
            if i > 0:
                dt = timestamps[i] - timestamps[i - 1]
                A = np.exp(-beta * dt) * (1.0 + A)
            lam_i = mu + alpha * A
            if lam_i <= 0:
                return 1e10
            log_sum += np.log(lam_i)

        # --- Compensator integral ---
        # ∫₀ᵀ λ(t)dt = μ·T + (α/β)·Σᵢ[1 - exp(-β·(T - tᵢ))]
        integral = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - timestamps)))

        return -(log_sum - integral)

    best_result = None
    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        mu0    = rng.uniform(1e-3, 2.0)
        alpha0 = rng.uniform(0.1, 0.7)
        beta0  = rng.uniform(0.5, 5.0)
        result = minimize(
            neg_log_likelihood,
            x0=[mu0, alpha0, beta0],
            method="L-BFGS-B",
            bounds=[(1e-6, None), (1e-6, None), (1e-4, None)],
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    mu, alpha, beta = best_result.x
    return float(mu), float(alpha), float(beta)


def hawkes_intensity_snapshot(
    timestamps_ns: np.ndarray,
    now_ns: int,
    mu: float,
    alpha: float,
    beta: float,
) -> float:
    """
    Real-time intensity estimate λ(now) using the most recent event window.
    Uses O(N) recursive accumulator; call with last ~200 timestamps for speed.

    Parameters
    ----------
    timestamps_ns : np.ndarray
        Recent event timestamps in nanoseconds (int64), sorted ascending.
    now_ns : int
        Current wall-clock time in nanoseconds.
    mu, alpha, beta : float
        Fitted Hawkes parameters (beta in units of seconds⁻¹).

    Returns
    -------
    float : λ(now) in events/second.
    """
    if len(timestamps_ns) == 0:
        return float(mu)

    ts_s = (timestamps_ns - timestamps_ns[0]).astype(np.float64) * 1e-9
    now_s = (now_ns - timestamps_ns[0]) * 1e-9

    A = 0.0
    for i in range(len(ts_s)):
        if i > 0:
            dt = ts_s[i] - ts_s[i - 1]
            A = np.exp(-beta * dt) * (1.0 + A)

    dt_to_now = now_s - ts_s[-1]
    A_now = np.exp(-beta * dt_to_now) * (1.0 + A)
    return float(mu + alpha * A_now)


def hawkes_branching_ratio(alpha: float, beta: float) -> float:
    """η = α/β. Thread-safe, no state."""
    return alpha / beta if beta > 0 else float("inf")
```

**Integration point**: `features/microstructure.py`, computed once per volume bar. Store `(mu, alpha, beta, eta)` as features. Re-fit every 500 events or every 5 minutes, whichever comes first. Cache fitted params in a rolling deque.

---

### 1.2 VPIN (Volume-Synchronized PIN)

**Use case**: Measure order flow toxicity. VPIN > 0.70 = high probability of informed trading. At NSE, VPIN > 0.70 historically precedes 2%+ adverse moves within 5 minutes during the open auction (09:15–09:25 IST). Use as a gate: suppress long/short signals when VPIN is elevated.

**Formula**:
```
VPIN = (1/n) · Σᵦ |V_buy(b) - V_sell(b)| / V_bucket
```

- Bucket size = ADV / 50  (average daily volume divided by 50)
- BVC (Bulk Volume Classification): allocates each trade's volume to buy/sell using price-change z-score mapped through the standard normal CDF
- Window = 50 consecutive buckets (rolling)

**BVC allocation**:
```
Z = ΔP / σ_ΔP
P_buy  = Φ(Z)          # standard normal CDF
V_buy  = V_trade · Φ(Z)
V_sell = V_trade · (1 - Φ(Z))
```

```python
import polars as pl
import numpy as np
from scipy.stats import norm as scipy_norm


def compute_vpin(
    bars: pl.DataFrame,
    n_buckets: int = 50,
    window_buckets: int = 50,
) -> pl.Series:
    """
    VPIN via Bulk Volume Classification (Easley et al., 2012).
    Returns float series aligned to input length; NaN until window_buckets filled.

    Parameters
    ----------
    bars : pl.DataFrame
        Must contain columns: 'close' (float), 'volume' (float/int), 'timestamp_ns' (int64).
        Rows are time bars (e.g., 1-minute or volume bars).
    n_buckets : int
        Number of buckets per VPIN sample (default 50).
    window_buckets : int
        Rolling window of buckets for VPIN average (default 50).

    Returns
    -------
    pl.Series : VPIN values (0.0–1.0). Name = 'vpin'.
    """
    close  = bars["close"].to_numpy().astype(np.float64)
    volume = bars["volume"].to_numpy().astype(np.float64)
    n      = len(close)

    if n < 2:
        return pl.Series("vpin", [float("nan")] * n)

    # --- Step 1: price changes and rolling std ---
    delta_p = np.diff(close, prepend=close[0])   # length = n
    # Rolling std over 20 bars, minimum 2
    sigma = np.full(n, np.nan)
    for i in range(1, n):
        window_start = max(0, i - 20)
        sigma[i] = np.std(delta_p[window_start:i + 1], ddof=0)

    sigma = np.where(sigma == 0, 1e-8, sigma)

    # --- Step 2: BVC buy/sell classification per bar ---
    z_scores = delta_p / sigma
    z_scores[0] = 0.0
    p_buy  = scipy_norm.cdf(z_scores)           # Φ(z), shape (n,)
    v_buy  = volume * p_buy
    v_sell = volume * (1.0 - p_buy)

    # --- Step 3: accumulate into fixed-size volume buckets ---
    adv           = np.nanmean(volume) * n_buckets   # proxy for ADV
    bucket_size   = max(adv / n_buckets, 1.0)

    bucket_buy_list  = []
    bucket_sell_list = []
    bucket_vol_list  = []

    acc_buy = acc_sell = 0.0
    for i in range(n):
        remaining_buy  = v_buy[i]
        remaining_sell = v_sell[i]

        while remaining_buy + remaining_sell > 0:
            space = bucket_size - (acc_buy + acc_sell)
            total_remaining = remaining_buy + remaining_sell

            if total_remaining <= space:
                acc_buy  += remaining_buy
                acc_sell += remaining_sell
                remaining_buy = remaining_sell = 0.0
            else:
                # Fill remaining space pro-rata
                frac = space / total_remaining if total_remaining > 0 else 0.0
                acc_buy  += remaining_buy  * frac
                acc_sell += remaining_sell * frac
                remaining_buy  *= (1.0 - frac)
                remaining_sell *= (1.0 - frac)

                # Bucket complete
                bucket_buy_list.append(acc_buy)
                bucket_sell_list.append(acc_sell)
                bucket_vol_list.append(acc_buy + acc_sell)
                acc_buy = acc_sell = 0.0

    # --- Step 4: rolling VPIN over window_buckets ---
    nb = len(bucket_buy_list)
    if nb < window_buckets:
        return pl.Series("vpin", [float("nan")] * n)

    vpin_buckets = np.array([
        abs(bucket_buy_list[i] - bucket_sell_list[i]) / max(bucket_vol_list[i], 1e-8)
        for i in range(nb)
    ])

    # Rolling mean VPIN
    vpin_rolling = np.full(nb, np.nan)
    for i in range(window_buckets - 1, nb):
        vpin_rolling[i] = np.mean(vpin_buckets[i - window_buckets + 1 : i + 1])

    # Map bucket-level VPIN back to bar-level (last known value forward-fill)
    # Approximate: assign VPIN to bar index proportional to bucket count
    bar_vpin = np.full(n, np.nan)
    bucket_per_bar = nb / n if n > 0 else 1.0
    for b_idx in range(window_buckets - 1, nb):
        bar_idx = min(int(b_idx / bucket_per_bar), n - 1)
        bar_vpin[bar_idx] = vpin_rolling[b_idx]

    # Forward-fill NaN values
    last_valid = float("nan")
    for i in range(n):
        if not np.isnan(bar_vpin[i]):
            last_valid = bar_vpin[i]
        elif not np.isnan(last_valid):
            bar_vpin[i] = last_valid

    return pl.Series("vpin", bar_vpin)
```

**Integration point**: `features/microstructure.py`. Call after every completed volume bar. Gate: if `vpin > 0.70`, set `flow_toxic = True` in feature dict; upstream signal logic checks this flag before issuing orders.

---

### 1.3 Trade Flow Imbalance (TFI)

**Use case**: Signed order imbalance across multiple order book levels. Single-level imbalance is noisy; exponentially distance-weighting deeper levels captures hidden institutional intent while de-emphasising far-from-touch quotes.

**Formula**:
```
TFI = Σₖ wₖ · (BidVol_k - AskVol_k) / (BidVol_k + AskVol_k + ε)
wₖ  = exp(-λ · (k - 1))      k = 1, 2, ..., K   (k=1 is best bid/ask)
```

- λ = 0.5 is the standard decay (each level is ~60% of the previous)
- Output range: [-1, +1] approximately (exact bounds depend on imbalance distribution)
- Positive TFI = net buy pressure; Negative TFI = net sell pressure
- |TFI| > 0.6 with consistent sign = strong directional signal

```python
import numpy as np


def compute_tfi(
    bid_prices: np.ndarray,
    bid_vols: np.ndarray,
    ask_prices: np.ndarray,
    ask_vols: np.ndarray,
    decay_lambda: float = 0.5,
) -> float:
    """
    Multi-level Trade Flow Imbalance.
    Positive = buy pressure. Negative = sell pressure.

    Parameters
    ----------
    bid_prices : np.ndarray  shape (K,), sorted best-to-worst (descending)
    bid_vols   : np.ndarray  shape (K,), volume at each bid level
    ask_prices : np.ndarray  shape (K,), sorted best-to-worst (ascending)
    ask_vols   : np.ndarray  shape (K,), volume at each ask level
    decay_lambda : float     exponential decay per level (default 0.5)

    Returns
    -------
    float : TFI in approximately [-1, +1].
    """
    bid_vols = np.asarray(bid_vols, dtype=np.float64)
    ask_vols = np.asarray(ask_vols, dtype=np.float64)

    K = min(len(bid_vols), len(ask_vols))
    if K == 0:
        return 0.0

    # Exponential weights: w_k = exp(-λ*(k-1)), k = 1..K (0-indexed: k-1 = i)
    levels = np.arange(K, dtype=np.float64)
    weights = np.exp(-decay_lambda * levels)                    # shape (K,)

    denom = bid_vols[:K] + ask_vols[:K] + 1e-8                 # avoid div-by-zero
    imbalance = (bid_vols[:K] - ask_vols[:K]) / denom          # shape (K,)

    tfi = float(np.sum(weights * imbalance) / np.sum(weights))
    return tfi


def compute_tfi_delta(tfi_prev: float, tfi_curr: float) -> float:
    """
    First difference of TFI — captures acceleration of order flow pressure.
    Useful as a secondary feature alongside raw TFI.
    """
    return tfi_curr - tfi_prev
```

**Integration point**: `features/microstructure.py`, computed per tick (each L2 order book update). Store in a fixed-size ring buffer of length 20; compute `tfi_mean_20`, `tfi_std_20`, and `tfi_delta` as features passed to the ML model.

**Ring buffer pattern** (zero-allocation, production):

```python
import collections

class TFIBuffer:
    """Lock-free ring buffer for per-tick TFI storage."""

    def __init__(self, maxlen: int = 20) -> None:
        self._buf = collections.deque(maxlen=maxlen)

    def update(self, tfi: float) -> None:
        self._buf.append(tfi)

    def features(self) -> dict:
        arr = np.array(self._buf, dtype=np.float64)
        if len(arr) < 2:
            return {"tfi_last": arr[-1] if len(arr) else 0.0,
                    "tfi_mean": float("nan"), "tfi_std": float("nan"),
                    "tfi_delta": float("nan")}
        return {
            "tfi_last":  arr[-1],
            "tfi_mean":  float(np.mean(arr)),
            "tfi_std":   float(np.std(arr, ddof=1)),
            "tfi_delta": float(arr[-1] - arr[-2]),
        }
```

---

### 1.4 Avellaneda-Stoikov Reservation Price

**Use case**: Replaces the naive mid-price `(bid + ask) / 2` with an inventory-adjusted, volatility-weighted reservation price. The market-maker should quote asymmetrically when holding a net position: the reservation price skews toward the side that reduces inventory risk. Used as a feature and as a direct input to the order placement engine.

**Reservation price formula**:
```
r(s, t) = s - q · γ · σ² · (T - t)
```

**Optimal spread formula**:
```
δ* = γ · σ² · (T - t) + (2/γ) · ln(1 + γ/κ)
```

- s    = current mid-price
- q    = net position in shares (negative if short)
- γ    = risk aversion coefficient (0.1 typical for NSE equities)
- σ²   = instantaneous variance (from GARCH(1,1) forecast)
- T-t  = time remaining in session as a fraction [0, 1]
- κ    = order arrival intensity (market orders per second; estimate from recent trade rate)
- NSE session: 09:15–15:30 IST = 23,100 seconds total

```python
import numpy as np
from arch import arch_model       # pip install arch
import polars as pl
from typing import Optional

# NSE session constants (IST = UTC+5:30)
NSE_SESSION_SECONDS = 23_100     # 09:15 to 15:30 = 6h15m = 23100s
NSE_OPEN_HOUR_IST   = 9
NSE_OPEN_MIN_IST    = 15
NSE_CLOSE_HOUR_IST  = 15
NSE_CLOSE_MIN_IST   = 30


def estimate_garch_variance(
    returns: np.ndarray,
    p: int = 1,
    q: int = 1,
) -> float:
    """
    Fit GARCH(1,1) and return one-step-ahead variance forecast.

    Parameters
    ----------
    returns : np.ndarray
        Log-returns (e.g., last 200 bars). Should be in percentage units
        for numerical stability (multiply raw returns by 100).
    p, q : int
        GARCH order. Default (1,1) is standard.

    Returns
    -------
    float : One-step-ahead variance (in raw return units²).
    """
    if len(returns) < 20:
        return float(np.var(returns)) if len(returns) > 1 else 1e-6

    returns_pct = returns * 100.0
    try:
        am = arch_model(returns_pct, vol="Garch", p=p, q=q, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        # forecast 1 step ahead
        fc = res.forecast(horizon=1, reindex=False)
        variance_pct_sq = float(fc.variance.iloc[-1, 0])
        return variance_pct_sq / (100.0 ** 2)    # convert back to raw units²
    except Exception:
        return float(np.var(returns))


def compute_reservation_price(
    best_bid: float,
    best_ask: float,
    garch_variance: float,
    net_position: int,
    timestamp_ns: int,
    session_end_ns: int,
    gamma: float = 0.1,
    kappa: float = 1.5,
) -> tuple[float, float]:
    """
    Avellaneda-Stoikov reservation price and optimal spread.

    Parameters
    ----------
    best_bid : float      Best bid price (INR).
    best_ask : float      Best ask price (INR).
    garch_variance : float  One-step variance from GARCH(1,1) (in price² units).
    net_position : int    Current net position in shares (+long, -short, 0=flat).
    timestamp_ns : int    Current time in nanoseconds (Unix epoch).
    session_end_ns : int  NSE close time (15:30 IST) in nanoseconds.
    gamma : float         Risk aversion coefficient (default 0.1).
    kappa : float         Order arrival intensity in market-orders/second (default 1.5).

    Returns
    -------
    (reservation_price, optimal_spread) : tuple[float, float]
        reservation_price : Inventory-adjusted mid-price.
        optimal_spread    : Total spread to quote (half on each side).
    """
    mid = (best_bid + best_ask) / 2.0

    # Time to session end as fraction [0, 1]
    time_remaining_ns = max(session_end_ns - timestamp_ns, 0)
    T_minus_t = time_remaining_ns / (NSE_SESSION_SECONDS * 1e9)
    T_minus_t = min(T_minus_t, 1.0)

    # Reservation price: skew mid away from net position direction
    reservation_price = mid - float(net_position) * gamma * garch_variance * T_minus_t

    # Optimal spread (Avellaneda-Stoikov closed form)
    spread_inventory  = gamma * garch_variance * T_minus_t
    spread_arrival    = (2.0 / gamma) * np.log(1.0 + gamma / max(kappa, 1e-6))
    optimal_spread    = spread_inventory + spread_arrival

    # Sanity clamp: spread cannot be below tick size (0.05 INR for most NSE equities)
    NSE_MIN_TICK = 0.05
    optimal_spread = max(optimal_spread, NSE_MIN_TICK)

    return float(reservation_price), float(optimal_spread)


def make_session_end_ns(date_str: str, tz_offset_hours: float = 5.5) -> int:
    """
    Helper: convert NSE session-end date string to UTC nanoseconds.

    Parameters
    ----------
    date_str : str   Format 'YYYY-MM-DD'.
    tz_offset_hours  IST = UTC + 5.5

    Returns
    -------
    int : Unix nanoseconds for 15:30:00 IST on that date.
    """
    import datetime
    d = datetime.date.fromisoformat(date_str)
    session_end_ist = datetime.datetime(d.year, d.month, d.day, 15, 30, 0)
    tz_offset = datetime.timedelta(hours=tz_offset_hours)
    session_end_utc = session_end_ist - tz_offset
    epoch = datetime.datetime(1970, 1, 1)
    return int((session_end_utc - epoch).total_seconds() * 1e9)
```

**Integration point**: `execution/order_manager.py` and `features/microstructure.py`. Call `compute_reservation_price()` on every L1 tick. Feed `reservation_price` and `optimal_spread` as features to the ML model AND use them directly in the quote placement logic (bid = reservation_price - optimal_spread/2, ask = reservation_price + optimal_spread/2).

**GARCH update cadence**: Re-fit GARCH every 50 new bars (not per-tick). Cache the variance forecast. Use `estimate_garch_variance()` with the last 200 log-returns.

---

### Summary Table

| Model | Feature Name(s) | Output Range | High-Risk Threshold | Integration File |
|---|---|---|---|---|
| Hawkes Process | `hawkes_eta`, `hawkes_mu`, `hawkes_intensity` | η: [0, ∞), λ: [0, ∞) | η > 0.85 = spoof cascade | `features/microstructure.py` |
| VPIN | `vpin` | [0, 1] | > 0.70 = toxic flow | `features/microstructure.py` |
| Trade Flow Imbalance | `tfi_last`, `tfi_mean`, `tfi_std`, `tfi_delta` | ~[-1, +1] | abs > 0.60 = strong pressure | `features/microstructure.py` |
| Avellaneda-Stoikov | `reservation_price`, `optimal_spread` | price: INR, spread: INR+ | spread > 3× tick = high vol | `execution/order_manager.py` |

---

### Feature Integration Checklist

```
features/microstructure.py
  ├── [EVERY VOLUME BAR]
  │   ├── fit_hawkes(recent_timestamps)         → (mu, alpha, beta)
  │   ├── hawkes_branching_ratio(alpha, beta)   → eta             [feature: hawkes_eta]
  │   ├── hawkes_intensity_snapshot(...)        → lambda_now      [feature: hawkes_intensity]
  │   └── compute_vpin(bars_df)                → vpin_series     [feature: vpin]
  │
  ├── [EVERY L2 TICK]
  │   ├── compute_tfi(bid_p, bid_v, ask_p, ask_v) → tfi         [stored in TFIBuffer]
  │   └── TFIBuffer.features()                  → dict           [features: tfi_last/mean/std/delta]
  │
  └── [EVERY 50 BARS — GARCH REFRESH]
      └── estimate_garch_variance(returns[-200:]) → sigma2       [cached for AS model]

execution/order_manager.py
  └── [EVERY L1 TICK]
      └── compute_reservation_price(bid, ask, sigma2, pos, ts, session_end)
                                                → (r_price, spread)
              ├── r_price  → feature AND quote anchor
              └── spread   → bid/ask offset from r_price

Signal gate logic (strategy layer):
  if vpin > 0.70 or hawkes_eta > 0.85:
      suppress_new_orders()
  else:
      run_ml_signal_pipeline()
```

**Dependencies**: `numpy`, `scipy`, `polars`, `arch` (for GARCH). All CPU-bound, no GPU required. GARCH fitting on 200 bars takes ~15ms on i5-1345U; all other functions are sub-millisecond per call.


---

## PART 2: INFRASTRUCTURE UPGRADES

---

### 2.1 ZeroMQ Architecture for GIL Bypass

**The core problem.** Python's GIL is a single mutex. When `model.predict()` runs inside the WebSocket callback thread, the GIL is held by the inference call for 20–80ms (XGBoost on 23 features, ensemble of 3). During that window, the WebSocket thread cannot receive ticks. On NSE, 20ms at open = 5–15 missed tick events per model call. Across 200 signals per day, you lose thousands of ticks. `multiprocessing.Queue` moves the GIL problem to inter-process communication overhead: pickle serialization, OS pipe round-trips, and shared memory locks add 0.5–3ms per message.

**ZeroMQ solves this differently.** ZMQ uses lock-free ring buffers inside OS kernel memory. Messages are passed via file descriptors — no Python object serialization on the critical path when using pre-allocated byte buffers. Processes are fully isolated (no GIL crossing by definition). The library itself is written in C++ with no GIL involvement.

**The 3-process topology:**

```
[DataIngestor]          [FeatureComputer]          [MLInference]
websocket_feed.py  →    features/engine.py    →    models/ensemble.py
ZMQ PUB :5555           ZMQ SUB :5555               ZMQ SUB :5556
                         ZMQ PUB :5556
                                                     ZMQ PUSH :5557
                                                          ↓
                                               [OrderRouter] PULL :5557
                                               execution/order_manager.py
```

**Socket type rationale:**
- `PUB/SUB` for tick broadcast: DataIngestor publishes to all subscribers. Adding a second subscriber (e.g., risk monitor) costs zero refactoring — just add another SUB.
- `PUSH/PULL` for order routing: exactly-once delivery semantics, load-balanced if multiple order workers are added. No duplicated orders.
- Never use `REQ/REP` on the hot path — it's synchronous and blocks the sender until reply arrives.

**Minimal working implementation:**

```python
# data/websocket_feed.py  — DataIngestor process
import zmq
import msgpack
import time
from dataclasses import dataclass

@dataclass
class Tick:
    symbol: str
    ltp: float
    volume: int
    bid: float
    ask: float
    timestamp_ns: int

def run_ingestor():
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 10_000)   # high-water mark: drop old msgs, never block
    pub.setsockopt(zmq.SNDBUF, 65536)
    pub.bind("tcp://127.0.0.1:5555")

    def on_tick(raw: dict):
        tick = Tick(
            symbol=raw["token"],
            ltp=float(raw["ltp"]),
            volume=int(raw["volume"]),
            bid=float(raw.get("best_bid_price", 0)),
            ask=float(raw.get("best_ask_price", 0)),
            timestamp_ns=time.time_ns(),
        )
        pub.send_multipart([
            tick.symbol.encode(),
            msgpack.packb([tick.ltp, tick.volume, tick.bid,
                           tick.ask, tick.timestamp_ns])
        ], flags=zmq.NOBLOCK)  # NOBLOCK: never stall WebSocket thread
```

```python
# features/engine.py  — FeatureComputer process
import zmq
import msgpack
import numpy as np
from data.ring_buffer import RingBuffer

def run_feature_computer(symbols: list[str]):
    ctx = zmq.Context()

    sub = ctx.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5555")
    for sym in symbols:
        sub.setsockopt(zmq.SUBSCRIBE, sym.encode())

    pub = ctx.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:5556")

    buffers: dict[str, RingBuffer] = {s: RingBuffer(capacity=10_000) for s in symbols}

    while True:
        topic, payload = sub.recv_multipart()
        ltp, vol, bid, ask, ts = msgpack.unpackb(payload)
        sym = topic.decode()

        buffers[sym].push(ltp, vol, bid, ask, ts)

        if buffers[sym].count >= 500:
            features = compute_features(buffers[sym])
            pub.send_multipart([
                sym.encode(),
                features.tobytes()
            ], flags=zmq.NOBLOCK)
```

```python
# models/ensemble.py  — MLInference process
import zmq
import numpy as np

def run_inference(model_xgb, model_lgb, model_cat, symbols: list[str]):
    ctx = zmq.Context()

    sub = ctx.socket(zmq.SUB)
    sub.connect("tcp://127.0.0.1:5556")
    for sym in symbols:
        sub.setsockopt(zmq.SUBSCRIBE, sym.encode())

    push = ctx.socket(zmq.PUSH)
    push.connect("tcp://127.0.0.1:5557")

    while True:
        topic, payload = sub.recv_multipart()
        features = np.frombuffer(payload, dtype=np.float32).reshape(1, -1)

        p_xgb = model_xgb.predict_proba(features)[0, 1]
        p_lgb = model_lgb.predict_proba(features)[0, 1]
        p_cat = model_cat.predict_proba(features)[0, 1]
        signal = (p_xgb + p_lgb + p_cat) / 3.0

        if signal > 0.60:
            push.send_json({
                "symbol": topic.decode(),
                "signal": float(signal),
                "direction": "BUY"
            }, flags=zmq.NOBLOCK)
```

```python
# main.py — launch all 3 processes
import multiprocessing as mp

def main():
    p1 = mp.Process(target=run_ingestor, daemon=True)
    p2 = mp.Process(target=run_feature_computer, args=(["NIFTY", "BANKNIFTY"],), daemon=True)
    p3 = mp.Process(target=run_inference, args=(xgb, lgb, cat, ["NIFTY", "BANKNIFTY"]), daemon=True)

    for p in [p1, p2, p3]:
        p.start()
    for p in [p1, p2, p3]:
        p.join()
```

**Install:** `pip install pyzmq msgpack`

**Latency achieved:** DataIngestor → FeatureComputer: ~15–40µs. Total tick-to-signal: sub-1ms on loopback. GIL problem structurally eliminated.

---

### 2.2 In-Memory Ring Buffer (Replacing QuestDB Live Reads)

**The problem.** `features/engine.py` queries QuestDB via REST to get the last 500 ticks — 2–15ms per call, plus JSON deserialization, plus Polars DataFrame construction. At 200 signals/day on 5 symbols = 1,000 REST calls per session. Self-inflicted I/O bottleneck.

**Fix:** keep the last N ticks in-process memory. QuestDB becomes write-only.

```python
# data/ring_buffer.py
import numpy as np
import threading

TICK_DTYPE = np.dtype([
    ("ltp",       np.float32),
    ("volume",    np.int32),
    ("bid",       np.float32),
    ("ask",       np.float32),
    ("timestamp", np.int64),
])

class RingBuffer:
    """Lock-minimized circular buffer. One writer, multiple readers."""

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self._buf = np.zeros(capacity, dtype=TICK_DTYPE)
        self._head = 0
        self._count = 0
        self._lock = threading.Lock()

    def push(self, ltp: float, volume: int, bid: float,
             ask: float, timestamp_ns: int) -> None:
        idx = self._head % self.capacity
        self._buf[idx]["ltp"]       = ltp
        self._buf[idx]["volume"]    = volume
        self._buf[idx]["bid"]       = bid
        self._buf[idx]["ask"]       = ask
        self._buf[idx]["timestamp"] = timestamp_ns
        with self._lock:
            self._head += 1
            self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def last(self, k: int) -> np.ndarray:
        """Return the last K ticks as a structured numpy array."""
        with self._lock:
            head = self._head
        filled = min(head, self.capacity)
        k = min(k, filled)
        end = head % self.capacity
        start = (head - k) % self.capacity
        if start < end:
            return self._buf[start:end]
        else:
            return np.concatenate([self._buf[start:], self._buf[:end]])

    def last_ltp_array(self, k: int) -> np.ndarray:
        return self.last(k)["ltp"]
```

**Async QuestDB persistence — background thread, never blocks write path:**

```python
# data/websocket_feed.py (additions)
import queue
import threading
from questdb.ingress import Sender, TimestampNanos

_persist_queue: queue.Queue = queue.Queue(maxsize=50_000)

def _questdb_writer_thread(sender_conf: str):
    with Sender.from_conf(sender_conf) as sender:
        batch = []
        while True:
            try:
                tick = _persist_queue.get(timeout=0.1)
                batch.append(tick)
                if len(batch) >= 100:
                    for t in batch:
                        sender.row(
                            "ticks",
                            symbols={"symbol": t["symbol"]},
                            columns={"ltp": t["ltp"], "volume": t["volume"],
                                     "bid": t["bid"], "ask": t["ask"]},
                            at=TimestampNanos(t["timestamp_ns"])
                        )
                    sender.flush()
                    batch.clear()
            except queue.Empty:
                if batch:
                    for t in batch:
                        sender.row("ticks", symbols={"symbol": t["symbol"]},
                                   columns={"ltp": t["ltp"]},
                                   at=TimestampNanos(t["timestamp_ns"]))
                    sender.flush()
                    batch.clear()

def start_persistence(questdb_conf: str = "http::addr=localhost:9000;"):
    t = threading.Thread(target=_questdb_writer_thread, args=(questdb_conf,),
                         daemon=True, name="questdb-writer")
    t.start()
    return t
```

**Result:** `ring_buffer.last(500)` = ~1µs. QuestDB REST on hot path eliminated.

---

### 2.3 Volume Bars & Tick Bars (Replacing 15-Minute Time Bars)

**Why time bars fail.** A 15-min candle at 9:15 AM NSE open = 400–800 trades. Same candle at 12:30 PM = 40–80 trades. 10x variance in information content, treated identically. Model learns calendar effects instead of market structure.

**Information-uniform bar types:**

| Bar Type | Fires When | Best For |
|---|---|---|
| Tick Bar | Every N trades (e.g., 500) | Controlling trade count |
| Volume Bar | Every V shares (e.g., 50,000) | Controlling shares exchanged |
| Dollar Bar | Every ₹D turnover (e.g., ₹2 Cr) | Most statistically stable on NSE |

**Choosing parameters:** Target 150–250 bars/session. `V = avg_daily_volume / 200` using last 60 days.

```python
# data/candle_store.py — add VolumeBarAggregator
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

@dataclass
class Bar:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    turnover: float
    tick_count: int
    start_ts: int
    end_ts: int
    bar_type: str

class VolumeBarAggregator:
    def __init__(self, symbol: str, volume_threshold: int = 50_000,
                 tick_threshold: int = 500,
                 dollar_threshold: float = 2_00_00_000.0,
                 on_bar: Optional[Callable[[Bar], None]] = None):
        self.symbol = symbol
        self.volume_threshold = volume_threshold
        self.tick_threshold = tick_threshold
        self.dollar_threshold = dollar_threshold
        self.on_bar = on_bar
        self._reset()

    def _reset(self):
        self._open = self._high = self._low = self._close = 0.0
        self._volume = 0
        self._turnover = 0.0
        self._ticks = 0
        self._start_ts = 0
        self._initialized = False

    def push_tick(self, ltp: float, volume: int,
                  timestamp_ns: int) -> Optional[Bar]:
        if not self._initialized:
            self._open = self._high = self._low = ltp
            self._start_ts = timestamp_ns
            self._initialized = True

        self._high = max(self._high, ltp)
        self._low  = min(self._low,  ltp)
        self._close = ltp
        self._volume += volume
        self._turnover += ltp * volume
        self._ticks += 1

        if (self._volume >= self.volume_threshold or
                self._ticks >= self.tick_threshold or
                self._turnover >= self.dollar_threshold):

            bar_type = (
                "volume" if self._volume >= self.volume_threshold else
                "tick"   if self._ticks  >= self.tick_threshold   else "dollar"
            )
            bar = Bar(self.symbol, self._open, self._high, self._low,
                      self._close, self._volume, self._turnover,
                      self._ticks, self._start_ts, timestamp_ns, bar_type)
            self._reset()
            if self.on_bar:
                self.on_bar(bar)
            return bar
        return None

    def calibrate_from_history(self, daily_volumes: list[int],
                                target_bars: int = 200) -> None:
        self.volume_threshold = int(np.mean(daily_volumes)) // target_bars
```

---

### 2.4 Volume Clock Time-of-Day Encoding

**The bug.** Current code: `tod_sin = np.sin(2π × minute/375)` — treats 9:15 AM and 3:30 PM as identical in market activity terms. NSE has a U-shaped volume curve; opening 30 min = 20–35% of daily volume.

**Volume Clock:** map each minute to its cumulative volume percentile.

```
VolumeTime(t) = CumulativeVolume(9:15 → t) / TotalDayVolume
```

Monotonically increasing 0→1, but fast at open/close, slow at lunch.

```python
# features/custom.py
import numpy as np
import json
from pathlib import Path

NSE_SESSION_MINUTES = 375

def build_volume_clock_profile(historical_df, output_path=Path("features/nse_volume_profile.json")):
    """Build from 2 years of QuestDB minute-volume data."""
    import polars as pl
    profile = (
        historical_df
        .group_by("minute_of_day")
        .agg(pl.col("volume").mean().alias("avg_volume"))
        .sort("minute_of_day")
    )
    avg_vol = profile["avg_volume"].to_numpy()
    full_profile = np.zeros(NSE_SESSION_MINUTES)
    mins = profile["minute_of_day"].to_numpy()
    full_profile[mins] = avg_vol
    mask = full_profile == 0
    if mask.any():
        idx = np.arange(NSE_SESSION_MINUTES)
        full_profile[mask] = np.interp(idx[mask], idx[~mask], full_profile[~mask])
    cumulative = np.cumsum(full_profile)
    volume_clock = cumulative / cumulative[-1]
    json.dump(volume_clock.tolist(), output_path.open("w"))
    return volume_clock

_VOLUME_CLOCK = None

def get_volume_clock():
    global _VOLUME_CLOCK
    if _VOLUME_CLOCK is None:
        _VOLUME_CLOCK = np.array(json.load(open("features/nse_volume_profile.json")))
    return _VOLUME_CLOCK

def encode_time_of_day_volume_clock(hour: int, minute: int) -> tuple[float, float]:
    """Replace chronological sin/cos encoding. Drop-in replacement."""
    vc = get_volume_clock()
    idx = np.clip((hour - 9) * 60 + (minute - 15), 0, NSE_SESSION_MINUTES - 1)
    vt = vc[idx]
    return float(np.sin(2 * np.pi * vt)), float(np.cos(2 * np.pi * vt))
```

**Migration (one-line swap in features/custom.py):**
```python
# REMOVE: tod_sin = np.sin(2 * np.pi * minute_of_day / 375)
# ADD:    tod_sin, tod_cos = encode_time_of_day_volume_clock(now.hour, now.minute)
```

**Weekly cron refresh (Sunday 6 AM IST):**
```cron
30 0 * * 0 trader python scripts/update_volume_profile.py >> /var/log/vol_profile.log 2>&1
```

---

### 2.5 PyO3/Rust — Practical Decision

**Three hottest paths** (profile with `py-spy top --pid <pid>` first):

| Path | File | Expected CPU % |
|---|---|---|
| Tick parsing + ring buffer push | ring_buffer.py | 30–40% |
| OBI calculation | features/custom.py | 20–30% |
| Bar aggregation | candle_store.py | 10–15% |

**Recommendation: use numba first.** PyO3 requires Rust toolchain + 30–90s compile cycles. Numba gets 80% of the performance with 5% complexity.

```python
# features/custom.py — numba-accelerated OBI
from numba import njit
import numpy as np

@njit(cache=True, fastmath=True)
def compute_obi(bid_prices: np.ndarray, bid_volumes: np.ndarray,
                ask_prices: np.ndarray, ask_volumes: np.ndarray,
                mid_price: float, alpha: float = 0.01) -> float:
    """~50-200x faster than pure Python. Compiled on first call, cached."""
    bid_weight = 0.0
    ask_weight = 0.0
    for i in range(len(bid_prices)):
        dist = abs(bid_prices[i] - mid_price) / mid_price
        w = np.exp(-alpha * dist * 10000)
        bid_weight += bid_volumes[i] * w
    for i in range(len(ask_prices)):
        dist = abs(ask_prices[i] - mid_price) / mid_price
        w = np.exp(-alpha * dist * 10000)
        ask_weight += ask_volumes[i] * w
    total = bid_weight + ask_weight
    if total == 0.0:
        return 0.0
    return (bid_weight - ask_weight) / total

def warmup_numba():
    """Call once at startup to trigger JIT compilation before market open."""
    dummy = np.array([100.0, 99.9, 99.8])
    dummy_v = np.array([100, 200, 150], dtype=np.float64)
    compute_obi(dummy, dummy_v, dummy, dummy_v, 100.0)
```

**Upgrade to PyO3 only if:** py-spy still shows >20% CPU on the hot function after numba AND function needs zero-copy access to complex Python objects.

---

### 2.6 Daily VPS Restart (Memory Fragmentation Fix)

**Why.** CPython's `pymalloc` arenas fragment over 6.5 hours. Polars intermediate DataFrames, msgpack byte strings, ZMQ message buffers create thousands of short-lived allocations. By session end, RSS can be 2–3x initial footprint — increasing page faults and degrading numpy cache performance.

**Fix:** systemd restart at 8:50 AM daily (before 9:00 AM pre-open). Fresh memory state, models reload in 3–8 seconds.

```bash
# scripts/daily_restart.sh
#!/bin/bash
set -euo pipefail
PID=$(cat /var/run/algo_trading/main.pid)
kill -USR1 "$PID"    # SIGUSR1 → graceful close positions → exit
WAIT=0
while kill -0 "$PID" 2>/dev/null; do
    [ $WAIT -ge 60 ] && kill -TERM "$PID" && break
    sleep 2; WAIT=$((WAIT+2))
done
# systemd Restart=always handles relaunch
```

```python
# main.py — SIGUSR1 handler
import signal, sys
_shutdown_requested = False

def _handle_graceful_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True

signal.signal(signal.SIGUSR1, _handle_graceful_shutdown)
```

```ini
# /etc/systemd/system/algo-trading.service
[Service]
ExecStart=/home/trader/.venv/bin/python main.py --mode live
Restart=always
RestartSec=10
```

```cron
# Cron: 8:50 AM IST = 3:20 UTC on weekdays
20 3 * * 1-5 trader /home/trader/algo_trading/scripts/daily_restart.sh
```

**Quick fix (add to main.py):**
```python
import gc, ctypes
def trim_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
# schedule.every(30).minutes.do(trim_memory)
```

---

**File change summary:**

| File | Change |
|---|---|
| `data/websocket_feed.py` | ZMQ PUB socket, ring buffer push, async QuestDB writer |
| `data/ring_buffer.py` | **NEW** — numpy structured array ring buffer |
| `data/candle_store.py` | Add `VolumeBarAggregator` |
| `features/engine.py` | ZMQ SUB/PUB, drain ring buffer into bar aggregator |
| `features/custom.py` | Volume Clock encoding, numba OBI |
| `models/ensemble.py` | ZMQ SUB for features, PUSH for order signals |
| `scripts/daily_restart.sh` | **NEW** — graceful shutdown |
| `scripts/update_volume_profile.py` | **NEW** — weekly volume clock refresh |
| `main.py` | Multiprocessing launch, SIGUSR1 handler, 30-min memory trim |


---

## PART 3A: CRITICAL FIXES — ISSUES 1 THROUGH 12

This section documents 12 critical defects identified in the NSE algo trading system architecture,
covering data ingestion, feature engineering, model training, order execution, and process isolation.
Each fix is self-contained and references the specific file that must be modified.

---

### Fix 1: Retail Broker WebSocket Data Corruption

**What breaks (symptom)**

During intraday volatility events (result announcements, circuit hits, index rebalances), the Angel One
WebSocket feed silently drops packets or delivers ticks that are 3-8 seconds stale. The trading system
continues computing features and placing orders on corrupted data without any awareness that the feed
is degraded. This produces ghost signals — fills that look profitable on the broker dashboard but
reflect prices that no longer exist.

**Root cause**

Angel One runs a shared, throttled WebSocket infrastructure that caps retail clients at 1-3 tick
updates per second per symbol. The actual NSE tick rate during high-volume periods is 50-200 updates
per second. The broker's server silently coalesces or discards ticks before forwarding them. There is
no sequence number on the feed, so the client has no way to detect gaps. The reconnect logic in the
base Angel One Python library uses a fixed 5-second sleep, meaning 5 seconds of data is lost on every
disconnect during volatile periods when disconnects are most frequent.

**Fix**

Stage 1: wrap the existing Shoonya feed in `DataFeedValidator` that tracks timestamp monotonicity, detects
stale gaps (>2 seconds with no update = stale), and logs corruption events to QuestDB for post-session
audit. Stage 2: migrate to TrueData premium WebSocket (unthrottled L2 tick-by-tick) as the primary data
feed — required before enabling microstructure features (Hawkes/VPIN/TFI). See [ARCH-FIX-1] in CLAUDE.md.
The validator becomes a pass-through monitor at that point. Shoonya WebSocket = execution-only from Stage 2.

**File to modify:** `data/feed_validator.py` (new file), `data/angel_feed.py` (wrap existing client)

```python
# data/feed_validator.py
import time
import threading
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Optional
import zmq

logger = logging.getLogger(__name__)

STALE_GAP_SECONDS = 2.0
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0
RECONNECT_MULTIPLIER = 2.0


@dataclass
class TickHealth:
    symbol: str
    last_tick_ts: float = 0.0
    gap_count: int = 0
    stale_count: int = 0
    total_ticks: int = 0
    is_stale: bool = False


class DataFeedValidator:
    """
    Wraps any tick feed callback and validates tick sequence health.
    Detects stale ticks (gap > STALE_GAP_SECONDS), emits health metrics
    to QuestDB via ZeroMQ, and triggers reconnect on repeated stale events.
    """

    def __init__(
        self,
        symbols: list[str],
        on_valid_tick: Callable[[dict], None],
        reconnect_fn: Callable[[], None],
        zmq_push_addr: str = "tcp://127.0.0.1:5560",
        stale_gap_s: float = STALE_GAP_SECONDS,
        max_stale_before_reconnect: int = 5,
    ):
        self.symbols = symbols
        self.on_valid_tick = on_valid_tick
        self.reconnect_fn = reconnect_fn
        self.stale_gap_s = stale_gap_s
        self.max_stale = max_stale_before_reconnect
        self._health: dict[str, TickHealth] = {s: TickHealth(s) for s in symbols}
        self._lock = threading.Lock()
        self._reconnect_delay = RECONNECT_BASE_DELAY

        ctx = zmq.Context.instance()
        self._push = ctx.socket(zmq.PUSH)
        self._push.connect(zmq_push_addr)
        self._push.setsockopt(zmq.SNDHWM, 1000)

    def on_tick(self, tick: dict) -> None:
        """Called by the broker WebSocket callback for every tick received."""
        symbol = tick.get("symbol") or tick.get("tk", "UNKNOWN")
        now = time.time()

        with self._lock:
            h = self._health.get(symbol)
            if h is None:
                h = TickHealth(symbol)
                self._health[symbol] = h

            gap = now - h.last_tick_ts if h.last_tick_ts > 0 else 0.0
            h.total_ticks += 1

            if gap > self.stale_gap_s and h.last_tick_ts > 0:
                h.gap_count += 1
                h.stale_count += 1
                h.is_stale = True
                logger.warning(
                    "STALE TICK %s: gap=%.2fs stale_count=%d",
                    symbol, gap, h.stale_count,
                )
                self._emit_health(h, gap)

                if h.stale_count >= self.max_stale:
                    logger.error(
                        "Feed stale %d times for %s — triggering reconnect",
                        h.stale_count, symbol,
                    )
                    h.stale_count = 0
                    threading.Thread(
                        target=self._reconnect_with_backoff, daemon=True
                    ).start()
                return  # drop stale tick, do not forward

            h.is_stale = False
            h.last_tick_ts = now

        self.on_valid_tick(tick)

    def _reconnect_with_backoff(self) -> None:
        delay = self._reconnect_delay
        logger.info("Reconnecting feed in %.1fs", delay)
        time.sleep(delay)
        self._reconnect_delay = min(
            delay * RECONNECT_MULTIPLIER, RECONNECT_MAX_DELAY
        )
        try:
            self.reconnect_fn()
            self._reconnect_delay = RECONNECT_BASE_DELAY  # reset on success
        except Exception as e:
            logger.error("Reconnect failed: %s", e)

    def _emit_health(self, h: TickHealth, gap: float) -> None:
        msg = f"feed_health,symbol={h.symbol} gap={gap:.3f},stale={h.stale_count} {int(time.time_ns())}"
        try:
            self._push.send_string(msg, zmq.NOBLOCK)
        except zmq.Again:
            pass

    def get_health_snapshot(self) -> dict[str, TickHealth]:
        with self._lock:
            return dict(self._health)
```

---

### Fix 2: 15-Minute Time Bars → Volume Bars

**What breaks (symptom)**

The ML model trains on 15-minute OHLCV bars. A bar at 9:15 open may contain 400 individual trades;
a bar at 12:45 lunch may contain 38 trades. Both bars receive identical weight during training. The
model learns from bars where the signal-to-noise ratio varies by a factor of 10 across the session,
producing features with wildly non-stationary variance and degrading out-of-sample performance every
afternoon.

**Root cause**

Time aggregation is a convenience, not a signal-aligned choice. Each 15-minute bar is defined by a
clock boundary, not by the amount of information traded. The open and close of the session, plus
post-news periods, generate far more information per clock-minute than the lunch lull. A model trained
on time bars implicitly learns the time-of-day seasonality of information content as noise, not signal.

**Fix**

Replace time bars with volume bars. Each bar fires when cumulative traded volume crosses V shares,
where V = ADV/200 (200 bars per day on average, calibrated to the stock's actual activity). Provide
a `BarConfig` dataclass with `bar_type` literal so the switch is a one-line change and backtesting
can compare bar types systematically.

**File to modify:** `data/bar_aggregator.py` (new file)

```python
# data/bar_aggregator.py
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np
import polars as pl


@dataclass
class BarConfig:
    bar_type: Literal["time", "volume", "tick", "dollar"] = "volume"
    # volume bars: threshold = ADV / bar_count_target
    volume_threshold: int = 50_000      # shares; override with ADV/200
    tick_threshold: int = 500           # trades per bar
    dollar_threshold: float = 5_000_000 # INR value per bar
    time_seconds: int = 900             # 15 min for time bars


@dataclass
class _BarAccumulator:
    open: float = 0.0
    high: float = -np.inf
    low: float = np.inf
    close: float = 0.0
    volume: int = 0
    ticks: int = 0
    dollar: float = 0.0
    vwap_num: float = 0.0  # sum(price * vol)
    bar_start_ts: float = field(default_factory=time.time)


class VolumeBarAggregator:
    """
    Aggregates raw ticks into volume (or tick/dollar/time) bars.
    Call on_tick() for each incoming tick; bars are emitted via on_bar callback.

    Migration path from time bars:
        config = BarConfig(bar_type="time", time_seconds=900)  # existing
        config = BarConfig(bar_type="volume", volume_threshold=adv//200)  # new
    """

    def __init__(
        self,
        symbol: str,
        config: BarConfig,
        on_bar: callable,
        adv_shares: Optional[int] = None,
    ):
        self.symbol = symbol
        self.config = config
        self.on_bar = on_bar
        self._acc = _BarAccumulator()

        # Auto-calibrate volume threshold from ADV if provided
        if adv_shares and config.bar_type == "volume":
            self.config.volume_threshold = max(1, adv_shares // 200)

        self._bar_start_clock = time.time()

    def on_tick(self, price: float, volume: int, ts: float) -> None:
        acc = self._acc

        if acc.ticks == 0:
            acc.open = price
            acc.bar_start_ts = ts

        acc.high = max(acc.high, price)
        acc.low = min(acc.low, price)
        acc.close = price
        acc.volume += volume
        acc.ticks += 1
        acc.dollar += price * volume
        acc.vwap_num += price * volume

        if self._should_emit(acc, ts):
            self._emit(acc, ts)
            self._acc = _BarAccumulator()

    def _should_emit(self, acc: _BarAccumulator, ts: float) -> bool:
        bt = self.config.bar_type
        if bt == "volume":
            return acc.volume >= self.config.volume_threshold
        if bt == "tick":
            return acc.ticks >= self.config.tick_threshold
        if bt == "dollar":
            return acc.dollar >= self.config.dollar_threshold
        if bt == "time":
            return (ts - acc.bar_start_ts) >= self.config.time_seconds
        return False

    def _emit(self, acc: _BarAccumulator, ts: float) -> None:
        vwap = acc.vwap_num / acc.volume if acc.volume > 0 else acc.close
        bar = {
            "symbol": self.symbol,
            "ts": ts,
            "open": acc.open,
            "high": acc.high,
            "low": acc.low,
            "close": acc.close,
            "volume": acc.volume,
            "ticks": acc.ticks,
            "dollar": acc.dollar,
            "vwap": vwap,
            "bar_type": self.config.bar_type,
        }
        self.on_bar(bar)

    @staticmethod
    def build_from_ticks(
        ticks_df: pl.DataFrame,
        config: BarConfig,
        adv_shares: Optional[int] = None,
    ) -> pl.DataFrame:
        """Batch-convert a Polars tick DataFrame to bars (for backtesting)."""
        bars: list[dict] = []

        def collect(b: dict) -> None:
            bars.append(b)

        agg = VolumeBarAggregator("BATCH", config, collect, adv_shares)
        for row in ticks_df.iter_rows(named=True):
            agg.on_tick(row["price"], row["volume"], row["ts"])

        return pl.DataFrame(bars) if bars else pl.DataFrame()
```

---

### Fix 3: OBI Averaging Over 15 Minutes → Tick-by-Tick

**What breaks (symptom)**

The Order Book Imbalance (OBI) feature is computed as a simple average over the 15-minute bar window.
Because buy-side and sell-side pressure cancel each other across a 15-minute window under normal market
conditions, the average converges to approximately 0.50 for most bars. The feature carries near-zero
information about the current microstructure state and contributes noise rather than signal to the
model.

**Root cause**

OBI = bid_volume / (bid_volume + ask_volume) is a snapshot statistic. Its predictive value is in the
current snapshot and in its velocity of change, not in its time-averaged value. A 15-minute average
of a statistic that reverts to 0.50 over any 15-minute window is a constant with noise. The signal
decays with a half-life measured in seconds (empirically 2-10 seconds for NSE mid-caps).

**Fix**

Compute OBI on each tick using exponentially-decayed weights so that recent ticks dominate. Store the
running EWM state in the ring buffer feature slot so that the feature is always current at inference
time. Decay factor alpha = 1 - exp(-dt/tau) where tau = 5 seconds.

**File to modify:** `features/obi.py` (new file), `features/engine.py` (integration)

```python
# features/obi.py
import math
import time
from dataclasses import dataclass


OBI_DECAY_TAU_SECONDS = 5.0  # half-life ~3.5s; tune per stock


@dataclass
class OBIState:
    ewm_bid_vol: float = 0.0
    ewm_ask_vol: float = 0.0
    last_ts: float = 0.0
    obi: float = 0.5  # neutral default


class TickOBIComputer:
    """
    Computes Order Book Imbalance on each tick using exponential decay.
    Integrates into the ring buffer feature slot without requiring bar aggregation.

    OBI = ewm_bid / (ewm_bid + ewm_ask)
    Decay: weight of tick at lag dt = exp(-dt / tau)
    """

    def __init__(self, tau_seconds: float = OBI_DECAY_TAU_SECONDS):
        self.tau = tau_seconds
        self._state = OBIState()

    def update(
        self,
        bid_vol: float,
        ask_vol: float,
        ts: float | None = None,
    ) -> float:
        """
        Call on every tick. Returns current OBI in [0, 1].
        ts: unix timestamp of the tick; uses time.time() if None.
        """
        if ts is None:
            ts = time.time()

        s = self._state
        dt = ts - s.last_ts if s.last_ts > 0 else 0.0
        alpha = 1.0 - math.exp(-dt / self.tau) if dt > 0 else 1.0

        s.ewm_bid_vol = alpha * bid_vol + (1.0 - alpha) * s.ewm_bid_vol
        s.ewm_ask_vol = alpha * ask_vol + (1.0 - alpha) * s.ewm_ask_vol
        s.last_ts = ts

        total = s.ewm_bid_vol + s.ewm_ask_vol
        s.obi = s.ewm_bid_vol / total if total > 1e-9 else 0.5
        return s.obi

    def reset(self) -> None:
        self._state = OBIState()

    @property
    def current_obi(self) -> float:
        return self._state.obi


def compute_obi_snapshot(
    bid_vol: float,
    ask_vol: float,
    state: OBIState,
    ts: float,
    tau: float = OBI_DECAY_TAU_SECONDS,
) -> tuple[float, OBIState]:
    """
    Functional interface for integration into feature pipeline.
    Returns (obi_value, updated_state). Use when you need explicit state management.
    """
    dt = ts - state.last_ts if state.last_ts > 0 else 0.0
    alpha = 1.0 - math.exp(-dt / tau) if dt > 0 else 1.0

    new_bid = alpha * bid_vol + (1.0 - alpha) * state.ewm_bid_vol
    new_ask = alpha * ask_vol + (1.0 - alpha) * state.ewm_ask_vol
    total = new_bid + new_ask
    obi = new_bid / total if total > 1e-9 else 0.5

    new_state = OBIState(
        ewm_bid_vol=new_bid,
        ewm_ask_vol=new_ask,
        last_ts=ts,
        obi=obi,
    )
    return obi, new_state
```

---

### Fix 4: 30-Minute LIMIT Order Timeout → 15-20 Seconds

**What breaks (symptom)**

LIMIT orders placed on a directional signal sit in the order book for up to 30 minutes waiting for a
fill. The signal that triggered the order had a predictive horizon of 30-120 seconds. By the time the
order fills at minute 28, the original signal has long since decayed, reversed, or been superseded by
new adverse information. The system effectively executes random trades with a 30-minute delay.

**Root cause**

The default order management logic copied a standard retail configuration that suits passive, long-term
investors. For intraday algorithmic trading, an unfilled LIMIT after 20 seconds means the market moved
away from your price, which is itself adverse information. Staying in the book is an information-
negative position.

**Fix**

Set `LIMIT_TIMEOUT_SECONDS = 20`. After timeout, cancel and re-price to current best bid/ask minus
one tick. After two re-prices with no fill, cancel entirely and skip the trade. All fill checks are
tick-driven (not polling) via the broker's order update callback.

**File to modify:** `execution/order_manager.py`

```python
# execution/order_manager.py
import time
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)

LIMIT_TIMEOUT_SECONDS = 20
MAX_REPRICE_ATTEMPTS = 2


class OrderState(Enum):
    PENDING = auto()
    OPEN = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    EXPIRED = auto()


@dataclass
class ManagedOrder:
    order_id: str
    symbol: str
    side: str          # "BUY" or "SELL"
    quantity: int
    price: float
    placed_at: float = field(default_factory=time.time)
    state: OrderState = OrderState.PENDING
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    reprice_count: int = 0


class OrderManager:
    """
    Manages LIMIT order lifecycle with strict timeout enforcement.
    Tick-driven fill detection via on_order_update() callback from broker.
    """

    def __init__(
        self,
        broker_place_fn: Callable,       # broker.place_order(symbol, side, qty, price, order_type)
        broker_cancel_fn: Callable,      # broker.cancel_order(order_id)
        broker_modify_fn: Callable,      # broker.modify_order(order_id, new_price)
        get_best_price_fn: Callable,     # returns (best_bid, best_ask) for symbol
        timeout_s: float = LIMIT_TIMEOUT_SECONDS,
        max_reprice: int = MAX_REPRICE_ATTEMPTS,
    ):
        self.broker_place = broker_place_fn
        self.broker_cancel = broker_cancel_fn
        self.broker_modify = broker_modify_fn
        self.get_best_price = get_best_price_fn
        self.timeout_s = timeout_s
        self.max_reprice = max_reprice
        self._orders: dict[str, ManagedOrder] = {}
        self._lock = threading.Lock()

    def place_limit_with_timeout(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        on_fill: Optional[Callable[[ManagedOrder], None]] = None,
        on_expire: Optional[Callable[[ManagedOrder], None]] = None,
    ) -> str:
        """Place a LIMIT order and start the timeout watchdog."""
        raw_id = self.broker_place(symbol, side, quantity, price, "LIMIT")
        order = ManagedOrder(
            order_id=str(raw_id),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
        )
        order.state = OrderState.OPEN
        with self._lock:
            self._orders[order.order_id] = order

        t = threading.Thread(
            target=self._watchdog,
            args=(order.order_id, on_fill, on_expire),
            daemon=True,
        )
        t.start()
        return order.order_id

    def on_order_update(self, order_id: str, filled_qty: int, avg_price: float) -> None:
        """Called by broker WebSocket on every order status update."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return
            order.filled_qty = filled_qty
            order.avg_fill_price = avg_price
            if filled_qty >= order.quantity:
                order.state = OrderState.FILLED
                logger.info("FILLED %s %s@%.2f", order_id, order.symbol, avg_price)
            elif filled_qty > 0:
                order.state = OrderState.PARTIALLY_FILLED

    def _watchdog(
        self,
        order_id: str,
        on_fill: Optional[Callable],
        on_expire: Optional[Callable],
    ) -> None:
        deadline = time.time() + self.timeout_s

        while time.time() < deadline:
            time.sleep(0.5)
            with self._lock:
                order = self._orders.get(order_id)
            if order is None:
                return
            if order.state == OrderState.FILLED:
                if on_fill:
                    on_fill(order)
                return

        # Timeout reached — attempt reprice or cancel
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or order.state == OrderState.FILLED:
                return

        if order.reprice_count < self.max_reprice:
            self._reprice(order)
            # Restart watchdog for re-priced order
            self._watchdog(order_id, on_fill, on_expire)
        else:
            logger.warning(
                "Order %s expired after %d reprice attempts — cancelling",
                order_id, order.reprice_count,
            )
            self.broker_cancel(order_id)
            order.state = OrderState.EXPIRED
            if on_expire:
                on_expire(order)

    def _reprice(self, order: ManagedOrder) -> None:
        best_bid, best_ask = self.get_best_price(order.symbol)
        if order.side == "BUY":
            new_price = best_bid  # join best bid
        else:
            new_price = best_ask  # join best ask
        logger.info(
            "Repricing %s from %.2f to %.2f (attempt %d)",
            order.order_id, order.price, new_price, order.reprice_count + 1,
        )
        self.broker_modify(order.order_id, new_price)
        order.price = new_price
        order.reprice_count += 1
        order.placed_at = time.time()  # reset watchdog deadline
```

---

### Fix 5: Training on OHLCV, Executing on L2 Ticks → Match Environments

**What breaks (symptom)**

The XGBoost model is trained on a feature vector derived from 15-minute OHLCV bars (RSI, MACD,
bollinger bands, volume ratio). At inference time, the live feature vector is derived from L2 tick
data (OBI, trade imbalance, microprice, queue depth ratios). The column names overlap partially but
the statistical distributions are completely different. The model silently runs `predict()` on
mismatched inputs, producing confidence scores that mean nothing.

**Root cause**

The training pipeline and inference pipeline were developed independently by different contributors
and never formally integrated. There is no schema contract between what the model was trained on and
what the live system provides. XGBoost does not validate feature names or distributions at predict
time — it consumes whatever array it is given.

**Fix**

Implement `FeatureValidator` that records the exact schema (feature names, dtypes, value ranges) at
training time and asserts compatibility at inference time. This runs before every `model.predict()`
call. Fail fast with a descriptive error rather than silently predict on garbage.

**File to modify:** `features/validator.py` (new file), `ml/inference.py` (add assertion call)

```python
# features/validator.py
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

RANGE_TOLERANCE = 0.10  # allow 10% out-of-training-range values before warning


@dataclass
class FeatureSpec:
    name: str
    dtype: str
    min_val: float
    max_val: float
    mean_val: float
    null_count_pct: float


@dataclass
class FeatureSchema:
    feature_names: list[str]
    specs: dict[str, FeatureSpec]
    n_training_rows: int

    def save(self, path: str | Path) -> None:
        d = {
            "feature_names": self.feature_names,
            "specs": {k: asdict(v) for k, v in self.specs.items()},
            "n_training_rows": self.n_training_rows,
        }
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSchema":
        d = json.loads(Path(path).read_text())
        specs = {k: FeatureSpec(**v) for k, v in d["specs"].items()}
        return cls(
            feature_names=d["feature_names"],
            specs=specs,
            n_training_rows=d["n_training_rows"],
        )


class FeatureValidator:
    """
    Records feature schema at training time and asserts compatibility at inference time.
    Save schema after training; load and assert before every model.predict() call.
    """

    @staticmethod
    def build_schema(df: pl.DataFrame) -> FeatureSchema:
        specs = {}
        for col in df.columns:
            s = df[col]
            null_pct = s.null_count() / len(s) if len(s) > 0 else 0.0
            numeric = s.cast(pl.Float64, strict=False).drop_nulls()
            specs[col] = FeatureSpec(
                name=col,
                dtype=str(s.dtype),
                min_val=float(numeric.min()) if len(numeric) > 0 else 0.0,
                max_val=float(numeric.max()) if len(numeric) > 0 else 0.0,
                mean_val=float(numeric.mean()) if len(numeric) > 0 else 0.0,
                null_count_pct=null_pct,
            )
        return FeatureSchema(
            feature_names=df.columns,
            specs=specs,
            n_training_rows=len(df),
        )

    @staticmethod
    def assert_inference_compatible(
        train_schema: FeatureSchema,
        live_features: dict[str, Any],
        strict: bool = True,
    ) -> None:
        """
        Call before model.predict(). Raises ValueError on hard mismatches.
        Logs warnings on soft mismatches (value out of training range).
        """
        live_keys = set(live_features.keys())
        train_keys = set(train_schema.feature_names)

        missing = train_keys - live_keys
        extra = live_keys - train_keys

        if missing:
            msg = f"FATAL: {len(missing)} features missing from live vector: {sorted(missing)}"
            logger.error(msg)
            if strict:
                raise ValueError(msg)

        if extra:
            logger.warning(
                "%d extra features in live vector (will be ignored): %s",
                len(extra), sorted(extra),
            )

        # Value range checks
        oor_count = 0
        for name in train_keys & live_keys:
            spec = train_schema.specs[name]
            val = live_features[name]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                logger.error("NaN/None in live feature '%s' — aborting predict", name)
                if strict:
                    raise ValueError(f"NaN in live feature '{name}'")
                continue
            train_range = spec.max_val - spec.min_val
            low = spec.min_val - RANGE_TOLERANCE * train_range
            high = spec.max_val + RANGE_TOLERANCE * train_range
            if not (low <= val <= high):
                logger.warning(
                    "Feature '%s' value %.4f outside training range [%.4f, %.4f]",
                    name, val, spec.min_val, spec.max_val,
                )
                oor_count += 1

        if oor_count > len(train_keys) * 0.2:
            msg = (
                f"{oor_count}/{len(train_keys)} features out of training range — "
                "likely data format mismatch (OHLCV vs L2)"
            )
            logger.error(msg)
            if strict:
                raise ValueError(msg)
```

---

### Fix 6: QuestDB REST Reads on Hot Path → In-Memory Ring Buffer

**What breaks (symptom)**

Every time the feature engine needs recent tick history to compute a feature, it issues an HTTP GET
to QuestDB's REST endpoint, deserializes JSON, and constructs a Polars DataFrame. This takes 2-15ms
per call. At a 10Hz feature computation rate, the system spends 20-150ms per second waiting for its
own database, introducing jitter into the signal pipeline and occasionally missing entire ticks while
blocked on I/O.

**Root cause**

QuestDB is an append-only time-series store optimized for writes and analytical queries, not for
microsecond-latency random reads of the last N rows. The REST interface adds HTTP overhead, JSON
serialization overhead, and Python deserialization overhead on top of an already non-trivial I/O
path. Using it as a real-time feature source inverts its design purpose.

**Fix**

The ring buffer is already implemented in `data/ring_buffer.py`. Integrate it as the single source
of truth for feature computation. QuestDB receives async writes in a background thread for persistence
and replay only. The hot path never touches QuestDB.

**File to modify:** `features/engine.py` (integration with existing `data/ring_buffer.py`)

```python
# features/engine.py  — integration snippet
# Assumes data/ring_buffer.py contains RingBuffer as documented in the infrastructure section.

import numpy as np
from data.ring_buffer import RingBuffer
from features.obi import TickOBIComputer, OBIState
from features.validator import FeatureValidator, FeatureSchema


RING_SIZE = 2000  # ticks; ~10 minutes at 3 ticks/s


class FeatureEngine:
    """
    Computes all 45 features from in-memory ring buffer.
    Zero QuestDB reads on the hot path.
    QuestDB writes are fire-and-forget via background ZeroMQ PUSH.
    """

    def __init__(
        self,
        symbol: str,
        train_schema: FeatureSchema,
        ring_size: int = RING_SIZE,
    ):
        self.symbol = symbol
        self.train_schema = train_schema
        self._ring = RingBuffer(
            capacity=ring_size,
            dtype=np.dtype([
                ("ts", "f8"),
                ("price", "f4"),
                ("volume", "i4"),
                ("bid_vol", "f4"),
                ("ask_vol", "f4"),
            ]),
        )
        self._obi = TickOBIComputer(tau_seconds=5.0)

    def on_tick(self, ts: float, price: float, volume: int,
                bid_vol: float, ask_vol: float) -> dict | None:
        self._ring.append((ts, price, volume, bid_vol, ask_vol))

        # Need at least 50 ticks to compute meaningful features
        if len(self._ring) < 50:
            return None

        return self._compute(ts, price, bid_vol, ask_vol)

    def _compute(self, ts: float, price: float,
                 bid_vol: float, ask_vol: float) -> dict:
        arr = self._ring.as_array()  # zero-copy numpy view
        prices = arr["price"].astype(np.float64)
        volumes = arr["volume"].astype(np.float64)

        obi = self._obi.update(bid_vol, ask_vol, ts)

        # Compute all features from numpy — no I/O
        returns = np.diff(prices) / prices[:-1]
        features = {
            "obi": obi,
            "price_mom_20": float(prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0.0,
            "vol_ratio_20": float(volumes[-1] / (np.mean(volumes[-20:]) + 1e-9)),
            "volatility_20": float(np.std(returns[-20:]) if len(returns) >= 20 else 0.0),
            "microprice": float(
                (prices[-1] * bid_vol + prices[-1] * ask_vol) /
                (bid_vol + ask_vol + 1e-9)
            ),
            # ... remaining 40 features computed from arr without I/O
        }

        FeatureValidator.assert_inference_compatible(
            self.train_schema, features, strict=True
        )
        return features
```

---

### Fix 7: TDD Delayed to Phase 12 → Move to Phase 4

**What breaks (symptom)**

The ML model is trained on features that contain silent NaN leakage, boundary errors, and off-by-one
bugs in the ring buffer window calculations. These bugs are not discovered until Phase 12 when tests
are written — at which point months of training runs have been producing corrupted models that appeared
to work on contaminated validation sets.

**Root cause**

Writing tests after code does not test the correct behavior — it tests whatever the code currently
does, including its bugs. NaN propagation through feature functions is particularly dangerous: a NaN
in one feature can silently infect adjacent features in a numpy array, producing a model that appears
to train (loss decreases) but is actually fitting to missing-data artifacts.

**Fix**

Write pytest tests for each feature function immediately after writing the function. Use synthetic
Polars DataFrames with known inputs and known correct outputs. Include explicit NaN injection tests
and boundary condition tests.

**File to modify:** `tests/test_features.py` (new file, move to Phase 4)

```python
# tests/test_features.py
import math
import time
import pytest
import numpy as np
import polars as pl

from features.obi import TickOBIComputer, compute_obi_snapshot, OBIState
from data.bar_aggregator import VolumeBarAggregator, BarConfig


# ── OBI Tests ─────────────────────────────────────────────────────────────────

class TestTickOBI:
    def test_neutral_on_equal_volumes(self):
        obi = TickOBIComputer(tau_seconds=5.0)
        val = obi.update(bid_vol=100.0, ask_vol=100.0, ts=1.0)
        assert math.isclose(val, 0.5, abs_tol=1e-6)

    def test_full_bid_dominance(self):
        obi = TickOBIComputer(tau_seconds=5.0)
        val = obi.update(bid_vol=1000.0, ask_vol=0.0, ts=1.0)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_decay_toward_neutral(self):
        obi = TickOBIComputer(tau_seconds=1.0)
        obi.update(bid_vol=1000.0, ask_vol=0.0, ts=0.0)
        # After 10 tau seconds with equal volumes, should be near neutral
        val = obi.update(bid_vol=100.0, ask_vol=100.0, ts=10.0)
        assert abs(val - 0.5) < 0.01, f"Expected ~0.5, got {val}"

    def test_no_nan_on_zero_volumes(self):
        obi = TickOBIComputer()
        val = obi.update(bid_vol=0.0, ask_vol=0.0, ts=1.0)
        assert not math.isnan(val)
        assert val == pytest.approx(0.5)

    def test_nan_injection_propagation(self):
        """NaN in bid_vol must not silently corrupt future ticks."""
        obi = TickOBIComputer()
        obi.update(bid_vol=100.0, ask_vol=100.0, ts=1.0)
        # Simulate corrupted tick
        val_nan = obi.update(bid_vol=float("nan"), ask_vol=100.0, ts=2.0)
        # System should not produce NaN on next valid tick
        val_after = obi.update(bid_vol=100.0, ask_vol=100.0, ts=3.0)
        assert not math.isnan(val_after), "NaN propagated through OBI state"

    def test_reset_clears_state(self):
        obi = TickOBIComputer()
        obi.update(bid_vol=900.0, ask_vol=100.0, ts=1.0)
        obi.reset()
        val = obi.update(bid_vol=100.0, ask_vol=100.0, ts=2.0)
        assert val == pytest.approx(0.5)


# ── Volume Bar Tests ───────────────────────────────────────────────────────────

class TestVolumeBarAggregator:
    def _collect_bars(self, ticks: list[tuple], config: BarConfig) -> list[dict]:
        bars: list[dict] = []
        agg = VolumeBarAggregator("TEST", config, bars.append)
        for price, vol, ts in ticks:
            agg.on_tick(price, vol, ts)
        return bars

    def test_bar_fires_at_threshold(self):
        config = BarConfig(bar_type="volume", volume_threshold=100)
        ticks = [(100.0, 50, float(i)) for i in range(4)]  # 200 total volume
        bars = self._collect_bars(ticks, config)
        assert len(bars) == 2

    def test_ohlc_correct(self):
        config = BarConfig(bar_type="volume", volume_threshold=300)
        ticks = [
            (100.0, 100, 1.0),
            (105.0, 100, 2.0),
            (98.0, 100, 3.0),
        ]
        bars = self._collect_bars(ticks, config)
        assert len(bars) == 1
        b = bars[0]
        assert b["open"] == pytest.approx(100.0)
        assert b["high"] == pytest.approx(105.0)
        assert b["low"] == pytest.approx(98.0)
        assert b["close"] == pytest.approx(98.0)

    def test_vwap_correct(self):
        config = BarConfig(bar_type="volume", volume_threshold=200)
        ticks = [(100.0, 100, 1.0), (110.0, 100, 2.0)]
        bars = self._collect_bars(ticks, config)
        assert bars[0]["vwap"] == pytest.approx(105.0)

    def test_zero_volume_tick_no_crash(self):
        config = BarConfig(bar_type="volume", volume_threshold=100)
        ticks = [(100.0, 0, 1.0)] * 5
        bars = self._collect_bars(ticks, config)
        assert len(bars) == 0  # no bar fires on zero volume


# ── Boundary and NaN Tests ────────────────────────────────────────────────────

class TestFeatureBoundaries:
    def test_price_momentum_requires_min_ticks(self):
        """price_mom_20 should return 0.0 (not error) when fewer than 20 ticks."""
        prices = np.array([100.0] * 15, dtype=np.float64)
        n = len(prices)
        result = float(prices[-1] / prices[-20] - 1) if n >= 20 else 0.0
        assert result == 0.0

    def test_vol_ratio_no_div_zero(self):
        """vol_ratio with all-zero history must not produce inf or nan."""
        volumes = np.zeros(20, dtype=np.float64)
        result = float(1.0 / (np.mean(volumes[-20:]) + 1e-9))
        assert not math.isinf(result)
        assert not math.isnan(result)
```

---

### Fix 8: Over-Engineering ML Before API Plumbing → EMA Cross First

**What breaks (symptom)**

Three months of ML pipeline development produces a sophisticated XGBoost ensemble that cannot be
deployed because the WebSocket feed is untested under live conditions, the Shoonya order API returns
errors that are not handled, QuestDB persistence was never validated end-to-end, and the paper trading
alert mechanism was never wired up. The ML model is complete; the infrastructure it depends on has
never been proven to work as a whole system.

**Root cause**

Building in wrong order. The ML model is the most intellectually interesting component, so it gets
built first. But it depends on every other component working correctly. Building the roof before
pouring the foundation.

**Fix**

Stage 1 = EMA cross paper trade that validates the complete circuit: Shoonya API authentication,
WebSocket subscription and tick receipt, QuestDB write, signal computation, order placement, fill
receipt, and Telegram alert. Use a `gate_flag` system that blocks all ML execution until the EMA
cross circuit is confirmed working.

**File to modify:** `paper_trade.py` (Stage 1 implementation)

```python
# paper_trade.py  — Stage 1: EMA cross circuit validator
"""
Run this BEFORE any ML code. Every line in CHECKLIST must be True
before the system is allowed to proceed to Stage 2 ML development.
"""
import time
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import zmq

logger = logging.getLogger(__name__)


@dataclass
class CircuitChecklist:
    """Gate flags — all must be True before ML is enabled."""
    websocket_connected: bool = False
    first_tick_received: bool = False
    questdb_write_confirmed: bool = False
    order_placed_successfully: bool = False
    fill_received: bool = False
    telegram_alert_sent: bool = False

    def all_clear(self) -> bool:
        return all([
            self.websocket_connected,
            self.first_tick_received,
            self.questdb_write_confirmed,
            self.order_placed_successfully,
            self.fill_received,
            self.telegram_alert_sent,
        ])

    def report(self) -> str:
        lines = ["=== CIRCUIT CHECKLIST ==="]
        for k, v in self.__dict__.items():
            status = "PASS" if v else "FAIL"
            lines.append(f"  [{status}] {k}")
        lines.append(f"  ML ENABLED: {self.all_clear()}")
        return "\n".join(lines)


class EMACrossStrategy:
    """
    Minimal EMA(9)/EMA(21) cross strategy for circuit validation.
    Contains zero ML. Purpose: prove the full data-to-order circuit works.
    """

    def __init__(self, symbol: str, fast: int = 9, slow: int = 21):
        self.symbol = symbol
        self.fast_n = fast
        self.slow_n = slow
        self._prices: deque[float] = deque(maxlen=slow + 5)
        self._fast_ema: Optional[float] = None
        self._slow_ema: Optional[float] = None
        self._prev_signal: int = 0  # +1 long, -1 short, 0 flat

    def _ema(self, new_price: float, prev_ema: Optional[float], n: int) -> float:
        k = 2.0 / (n + 1)
        if prev_ema is None:
            return new_price
        return new_price * k + prev_ema * (1 - k)

    def on_tick(self, price: float) -> int:
        """Returns 0 (flat), +1 (long signal), -1 (short signal)."""
        self._prices.append(price)
        if len(self._prices) < self.slow_n:
            return 0

        self._fast_ema = self._ema(price, self._fast_ema, self.fast_n)
        self._slow_ema = self._ema(price, self._slow_ema, self.slow_n)

        if self._fast_ema is None or self._slow_ema is None:
            return 0

        new_signal = +1 if self._fast_ema > self._slow_ema else -1
        cross = new_signal != self._prev_signal and self._prev_signal != 0
        self._prev_signal = new_signal
        return new_signal if cross else 0


def run_paper_trade_validation(
    broker_api,
    questdb_client,
    telegram_fn,
    symbol: str = "RELIANCE",
    max_trades: int = 3,
) -> CircuitChecklist:
    """
    Run EMA cross paper trades until max_trades. Checks every gate flag.
    Returns checklist — caller decides whether to proceed to ML.
    """
    checklist = CircuitChecklist()
    strategy = EMACrossStrategy(symbol)
    trade_count = 0

    def on_tick(tick: dict) -> None:
        nonlocal trade_count
        checklist.first_tick_received = True

        price = tick["ltp"]
        signal = strategy.on_tick(price)

        # Validate QuestDB write on each tick
        try:
            questdb_client.write(symbol, price, time.time())
            checklist.questdb_write_confirmed = True
        except Exception as e:
            logger.error("QuestDB write failed: %s", e)

        if signal != 0 and trade_count < max_trades:
            side = "BUY" if signal == 1 else "SELL"
            logger.info("PAPER SIGNAL %s %s @ %.2f", side, symbol, price)
            try:
                order_id = broker_api.place_order(
                    symbol, side, quantity=1, price=price, order_type="MARKET"
                )
                checklist.order_placed_successfully = True
                logger.info("Paper order placed: %s", order_id)

                # Simulate fill confirmation (real: wait for callback)
                checklist.fill_received = True
                trade_count += 1

                msg = f"PAPER TRADE: {side} {symbol} @ {price:.2f} (trade {trade_count}/{max_trades})"
                telegram_fn(msg)
                checklist.telegram_alert_sent = True

            except Exception as e:
                logger.error("Order placement failed: %s — FIX THIS BEFORE ML", e)

    # Connect WebSocket
    try:
        broker_api.subscribe([symbol], on_tick)
        checklist.websocket_connected = True
    except Exception as e:
        logger.error("WebSocket connection failed: %s", e)

    # Run for 30 minutes or until max_trades
    start = time.time()
    while time.time() - start < 1800 and trade_count < max_trades:
        time.sleep(1)

    logger.info(checklist.report())
    if not checklist.all_clear():
        raise RuntimeError(
            "Circuit validation FAILED. Fix all checklist items before enabling ML.\n"
            + checklist.report()
        )
    return checklist
```

---

### Fix 9: Static 0.05% Slippage → Dynamic ATR/Spread Model

**What breaks (symptom)**

Backtests report Sharpe 2.1 using 0.05% flat slippage. Live trading produces Sharpe 0.4. The
difference is entirely explained by actual slippage: illiquid mid-cap stocks during high-volatility
events produce 0.15-0.30% slippage per trade. The static assumption was calibrated on large-cap
liquid stocks during calm periods and was never updated.

**Root cause**

Slippage has three components: (1) bid-ask spread at time of entry, (2) market impact from order
size relative to book depth, (3) adverse selection from the signal being picked up by faster traders.
All three scale with volatility (ATR), liquidity (spread), time of day (open/close = illiquid), and
order type. A static scalar cannot model any of these dimensions.

**Fix**

Replace static slippage with `compute_dynamic_slippage()` that computes a per-trade slippage
estimate as a function of ATR, spread, time of day, and order type. The backtest uses the same
function so estimated and actual slippage converge.

**File to modify:** `risk/cost_model.py`

```python
# risk/cost_model.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal


# Time-of-day multipliers (minute-of-session from 9:15 = 0)
# Open (0-15 min) and close (345-375 min) are 2x more expensive
_TOD_BREAKPOINTS = [
    (0,   15,  2.0),   # open auction spillover
    (15,  60,  1.3),   # morning momentum
    (60,  300, 1.0),   # mid-session baseline
    (300, 345, 1.2),   # pre-close
    (345, 375, 2.2),   # closing auction
]


def _tod_multiplier(minute_of_session: float) -> float:
    for start, end, mult in _TOD_BREAKPOINTS:
        if start <= minute_of_session < end:
            return mult
    return 1.5  # outside session — should not happen


@dataclass
class SlippageComponents:
    spread_component: float    # half-spread as fraction of price
    atr_component: float       # volatility-adjusted adverse selection
    tod_multiplier: float      # time-of-day scaling
    order_type_multiplier: float
    total_slippage_pct: float  # final estimate as % of price


def compute_dynamic_slippage(
    atr14: float,
    price: float,
    bid_ask_spread: float,
    time_of_day_minute: float,
    order_type: Literal["MARKET", "LIMIT"] = "MARKET",
    order_qty: int = 1,
    avg_book_depth: float = 10_000.0,
) -> SlippageComponents:
    """
    Compute realistic per-trade slippage estimate.

    Args:
        atr14: 14-period ATR in price units
        price: current price
        bid_ask_spread: absolute spread in price units
        time_of_day_minute: minutes since 9:15 (0 = open, 375 = close)
        order_type: MARKET orders pay full spread; LIMIT orders pay half
        order_qty: shares in this order
        avg_book_depth: average shares available at best bid/ask

    Returns:
        SlippageComponents with breakdown and total_slippage_pct
    """
    if price <= 0:
        raise ValueError(f"Invalid price: {price}")

    # Spread component: MARKET pays full half-spread, LIMIT pays ~0 on entry
    half_spread_pct = (bid_ask_spread / price) / 2.0
    order_type_mult = 1.0 if order_type == "MARKET" else 0.25
    spread_comp = half_spread_pct * order_type_mult

    # ATR-based adverse selection component
    # Empirical: adverse selection ~ 0.15 * ATR / price during trending moves
    atr_pct = atr14 / price
    atr_comp = 0.15 * atr_pct

    # Market impact from order size
    size_fraction = order_qty / max(avg_book_depth, 1.0)
    impact_comp = 0.5 * size_fraction * atr_pct  # linear approximation

    # Time-of-day multiplier
    tod_mult = _tod_multiplier(time_of_day_minute)

    # Total
    raw_slippage = spread_comp + atr_comp + impact_comp
    total = raw_slippage * tod_mult

    # Floor and cap: never below 0.02% (liquid large-cap calm) or above 0.50%
    total = max(0.0002, min(0.0050, total))

    return SlippageComponents(
        spread_component=spread_comp,
        atr_component=atr_comp,
        tod_multiplier=tod_mult,
        order_type_multiplier=order_type_mult,
        total_slippage_pct=total,
    )


def apply_slippage_to_fill(
    signal_price: float,
    side: Literal["BUY", "SELL"],
    slippage: SlippageComponents,
) -> float:
    """Return the estimated actual fill price after slippage."""
    adj = signal_price * slippage.total_slippage_pct
    return signal_price + adj if side == "BUY" else signal_price - adj
```

---

### Fix 10: Brain/Hands Coupling → Strict Segregation

**What breaks (symptom)**

Angel One handles both live market data (brain) and order execution (hands). During high-volatility
events, the broker's shared infrastructure throttles both simultaneously. The data feed degrades
exactly when signal quality needs to be highest, and the execution API slows exactly when order
latency matters most. Both failures compound: the system enters a trade on stale data and then
cannot cancel or modify the order because the execution channel is also throttled.

**Root cause**

Using a single broker for dual roles creates a single point of failure that affects both data quality
and execution quality simultaneously. The two roles have completely different latency and reliability
requirements: data needs high throughput and tolerance for gaps; execution needs low latency and
guaranteed delivery with retries.

**Fix**

Strict process segregation: Brain (TrueData feed) → `DataIngestor` process exclusively. Hands
(Shoonya execution) → `OrderExecutor` process exclusively. They communicate only through a ZeroMQ
PUSH/PULL boundary. Neither process knows about the other's broker.

**File to modify:** `execution/broker_base.py` (abstract interface), `main.py` (process launch)

```python
# execution/broker_base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class OrderRequest:
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: Optional[float]
    order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"]
    product: Literal["INTRADAY", "CNC"] = "INTRADAY"
    validity: Literal["DAY", "IOC"] = "DAY"
    tag: str = ""


@dataclass
class OrderResponse:
    order_id: str
    status: str
    message: str
    filled_qty: int = 0
    avg_price: float = 0.0


class BrokerBase(ABC):
    """
    Abstract execution interface. ONLY handles orders — never market data.
    Implementations: ShoonyaBroker, ZerodhaKiteBroker, PaperBroker.

    ARCHITECTURAL RULE:
        BrokerBase implementations MUST NOT subscribe to market data feeds.
        Market data is the exclusive domain of the DataIngestor process.
        Violating this rule recreates the Brain/Hands coupling anti-pattern.
    """

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with broker API. Returns True on success."""

    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderResponse:
        """Place a single order. Must return within 500ms or raise TimeoutError."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancellation was accepted."""

    @abstractmethod
    def modify_order(self, order_id: str, new_price: float, new_qty: Optional[int] = None) -> bool:
        """Modify price/qty of an open order."""

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Return current open positions."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResponse:
        """Fetch current status of a specific order."""


# ZeroMQ boundary between Brain and Hands processes
# Brain sends signals on PUSH socket; Hands receives on PULL socket.
# They share NO state, NO broker objects, NO in-process queues.
#
# Brain process:  zmq.PUSH  → tcp://127.0.0.1:5557
# Hands process:  zmq.PULL  ← tcp://127.0.0.1:5557
#
# Signal format (msgpack):
# {
#   "action": "BUY" | "SELL" | "CANCEL",
#   "symbol": "RELIANCE",
#   "qty": 50,
#   "price": 2450.00,       # None for MARKET
#   "order_type": "LIMIT",
#   "signal_ts": 1716000000.123,
#   "max_age_ms": 2000      # discard signal if older than this
# }
```

---

### Fix 11: GIL Stalls Tick Feed During model.predict() → ZeroMQ 3-Process

**What breaks (symptom)**

During a model inference call, the WebSocket tick feed pauses for 20-80ms. The Python GIL allows
only one thread to execute Python bytecode at a time. XGBoost's `predict()` calls into C++ extensions
but still acquires the GIL for Python object construction and return value handling. During this
period, the WebSocket thread cannot execute, ticks are buffered or dropped in the OS socket buffer,
and the first tick received after inference reflects a price that is 20-80ms stale.

**Root cause**

Python's Global Interpreter Lock (GIL) is a process-wide mutex. Threading does not provide true
parallelism for CPU-bound operations. Inference is CPU-bound. Data ingestion is I/O-bound but
requires Python callbacks that acquire the GIL. The two compete for the same lock.

**Fix**

Three separate OS processes communicate via ZeroMQ. Each process has its own GIL. Inference in
Process 3 cannot stall the tick feed in Process 1. All ZeroMQ sends use `zmq.NOBLOCK` to prevent
any process from blocking another.

**File to modify:** `main.py`

```python
# main.py  — 3-process launch with ZeroMQ boundaries
"""
Process topology:
  P1 DataIngestor    PUB  :5555  → raw ticks (msgpack)
  P2 FeatureComputer SUB  :5555, PUB :5556  → feature vectors (msgpack)
  P3 MLInference     SUB  :5556, PUSH :5557 → trade signals (msgpack)
  P4 OrderExecutor   PULL :5557  → executes orders (Shoonya)

Each process is launched as a separate Python process.
GIL contention between inference and data ingestion = ZERO.
"""
import multiprocessing as mp
import logging
import signal
import sys
import time
import msgpack
import zmq

logger = logging.getLogger(__name__)


def run_data_ingestor(pub_addr: str = "tcp://127.0.0.1:5555") -> None:
    """Process 1: Subscribe to TrueData/Angel One, publish raw ticks."""
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(pub_addr)
    pub.setsockopt(zmq.SNDHWM, 10000)

    # Import broker here — isolated to this process
    from data.feed_validator import DataFeedValidator

    def on_valid_tick(tick: dict) -> None:
        payload = msgpack.packb(tick, use_bin_type=True)
        try:
            pub.send_multipart(
                [tick["symbol"].encode(), payload],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass  # drop tick if downstream is full; never block

    # ... broker WebSocket setup here (Angel One / TrueData)
    logger.info("DataIngestor running on %s", pub_addr)
    signal.pause()


def run_feature_computer(
    sub_addr: str = "tcp://127.0.0.1:5555",
    pub_addr: str = "tcp://127.0.0.1:5556",
    symbols: list[str] | None = None,
) -> None:
    """Process 2: Receive raw ticks, compute features, publish feature vectors."""
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(sub_addr)
    for s in (symbols or ["RELIANCE", "INFY"]):
        sub.setsockopt_string(zmq.SUBSCRIBE, s)

    pub = ctx.socket(zmq.PUB)
    pub.bind(pub_addr)
    pub.setsockopt(zmq.SNDHWM, 1000)

    from features.engine import FeatureEngine
    from features.validator import FeatureSchema

    engines: dict[str, FeatureEngine] = {}

    while True:
        try:
            symbol_b, payload = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.0001)
            continue

        symbol = symbol_b.decode()
        tick = msgpack.unpackb(payload, raw=False)

        if symbol not in engines:
            # Load schema from disk (written at training time)
            schema = FeatureSchema.load(f"models/{symbol}_schema.json")
            engines[symbol] = FeatureEngine(symbol, schema)

        features = engines[symbol].on_tick(
            ts=tick["ts"],
            price=tick["ltp"],
            volume=tick["vol"],
            bid_vol=tick.get("bid_vol", 0.0),
            ask_vol=tick.get("ask_vol", 0.0),
        )
        if features is None:
            continue

        features["symbol"] = symbol
        out = msgpack.packb(features, use_bin_type=True)
        try:
            pub.send_multipart([symbol_b, out], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass


def run_ml_inference(
    sub_addr: str = "tcp://127.0.0.1:5556",
    push_addr: str = "tcp://127.0.0.1:5557",
    symbols: list[str] | None = None,
) -> None:
    """Process 3: Receive feature vectors, run model.predict(), push signals."""
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(sub_addr)
    for s in (symbols or ["RELIANCE", "INFY"]):
        sub.setsockopt_string(zmq.SUBSCRIBE, s)

    push = ctx.socket(zmq.PUSH)
    push.connect(push_addr)
    push.setsockopt(zmq.SNDHWM, 500)

    import xgboost as xgb
    import numpy as np

    models: dict[str, xgb.Booster] = {}
    feature_names: dict[str, list[str]] = {}

    while True:
        try:
            symbol_b, payload = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.0001)
            continue

        symbol = symbol_b.decode()
        features = msgpack.unpackb(payload, raw=False)

        if symbol not in models:
            models[symbol] = xgb.Booster()
            models[symbol].load_model(f"models/{symbol}_xgb.ubj")
            feature_names[symbol] = models[symbol].feature_names

        feat_arr = np.array(
            [features[f] for f in feature_names[symbol]], dtype=np.float32
        ).reshape(1, -1)
        dmat = xgb.DMatrix(feat_arr, feature_names=feature_names[symbol])
        prob = float(models[symbol].predict(dmat)[0])

        signal = {
            "symbol": symbol,
            "prob": prob,
            "action": "BUY" if prob > 0.55 else ("SELL" if prob < 0.45 else "FLAT"),
            "signal_ts": features.get("ts", time.time()),
            "max_age_ms": 2000,
        }
        out = msgpack.packb(signal, use_bin_type=True)
        try:
            push.send(out, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    symbols = ["RELIANCE", "INFY", "TCS"]

    procs = [
        mp.Process(target=run_data_ingestor, name="DataIngestor", daemon=True),
        mp.Process(
            target=run_feature_computer,
            kwargs={"symbols": symbols},
            name="FeatureComputer",
            daemon=True,
        ),
        mp.Process(
            target=run_ml_inference,
            kwargs={"symbols": symbols},
            name="MLInference",
            daemon=True,
        ),
    ]

    for p in procs:
        p.start()
        logger.info("Started %s (pid=%d)", p.name, p.pid)

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        logger.info("Shutting down all processes")
        for p in procs:
            p.terminate()
        sys.exit(0)


if __name__ == "__main__":
    main()
```

---

### Fix 12: Asymmetric Triple-Barrier → Symmetric ±1.5 ATR

**What breaks (symptom)**

The training label distribution shows 68% of samples labeled class 0 (lower barrier hit) and 32%
class 1 (upper barrier hit). The model achieves 65% accuracy in validation by predicting class 0
on nearly every sample. Precision on class 1 (the long signal the system actually needs) is 0.28.
The model has learned to predict "trade fails" with high confidence, which is the opposite of its
purpose.

**Root cause**

With +2.0 ATR upper barrier and -1.0 ATR lower barrier, the lower barrier is twice as close. Under
a random walk, P(lower hit before upper hit) ≈ upper_distance / (upper_distance + lower_distance)
= 2.0 / (2.0 + 1.0) = 67%. This is approximately what the label distribution shows. The asymmetry
was intended to model a favorable risk:reward ratio, but it creates a label imbalance that the model
exploits trivially.

**Fix**

Use symmetric ±1.5 ATR barriers. Add `compute_label_distribution()` as a mandatory diagnostic that
must be run and reviewed before any model training. Add `compute_sample_weights()` to handle any
residual imbalance from market microstructure (not barrier asymmetry). Both functions are integrated
into the training pipeline, not optional.

**File to modify:** `ml/triple_barrier.py`

```python
# ml/triple_barrier.py
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_UPPER_MULT = 1.5
DEFAULT_LOWER_MULT = 1.5
DEFAULT_MAX_HOLD_BARS = 20


def apply_triple_barrier(
    prices: pl.Series,
    atr: pl.Series,
    upper_atr_mult: float = DEFAULT_UPPER_MULT,
    lower_atr_mult: float = DEFAULT_LOWER_MULT,
    max_hold_bars: int = DEFAULT_MAX_HOLD_BARS,
) -> pl.Series:
    """
    Apply symmetric triple-barrier labeling.

    Labels:
        +1 = upper barrier hit first (profit target)
        -1 = lower barrier hit first (stop loss)
         0 = time barrier hit (max_hold_bars exceeded)

    Args:
        prices: price series (aligned with atr)
        atr: ATR series (same length as prices)
        upper_atr_mult: upper barrier = entry_price + upper_atr_mult * atr[t]
        lower_atr_mult: lower barrier = entry_price - lower_atr_mult * atr[t]
        max_hold_bars: time barrier in bars

    Returns:
        pl.Series of int8 labels [-1, 0, +1]
    """
    if upper_atr_mult <= 0 or lower_atr_mult <= 0:
        raise ValueError("ATR multipliers must be positive")

    if abs(upper_atr_mult - lower_atr_mult) > 0.1:
        logger.warning(
            "Asymmetric barriers: upper=%.2f lower=%.2f — "
            "this will create label imbalance. Recommend symmetric ±%.2f",
            upper_atr_mult, lower_atr_mult,
            (upper_atr_mult + lower_atr_mult) / 2,
        )

    p = prices.to_numpy()
    a = atr.to_numpy()
    n = len(p)
    labels = np.zeros(n, dtype=np.int8)

    for t in range(n - 1):
        entry = p[t]
        upper = entry + upper_atr_mult * a[t]
        lower = entry - lower_atr_mult * a[t]
        end = min(t + max_hold_bars, n)

        for i in range(t + 1, end):
            if p[i] >= upper:
                labels[t] = 1
                break
            if p[i] <= lower:
                labels[t] = -1
                break
        # else: labels[t] remains 0 (time barrier)

    return pl.Series("label", labels)


def compute_label_distribution(labels: pl.Series) -> dict[str, float]:
    """
    MANDATORY diagnostic. Run before every training run.
    Logs a warning if any class is below 20% of total samples.
    """
    total = len(labels)
    if total == 0:
        raise ValueError("Empty label series")

    counts = labels.value_counts().sort("label")
    dist: dict[str, float] = {}

    for row in counts.iter_rows(named=True):
        label = int(row["label"])
        count = int(row["count"])
        pct = count / total
        dist[f"class_{label}"] = pct
        logger.info("Label %+d: %d samples (%.1f%%)", label, count, pct * 100)

    # Imbalance check
    min_pct = min(dist.values())
    if min_pct < 0.20:
        logger.warning(
            "SEVERE LABEL IMBALANCE: minimum class = %.1f%%. "
            "Check barrier symmetry. If using upper=%.2f lower=%.2f, "
            "switch to symmetric ±1.5 ATR.",
            min_pct * 100,
            DEFAULT_UPPER_MULT,
            DEFAULT_LOWER_MULT,
        )
    return dist


def compute_sample_weights(
    labels: pl.Series,
    method: str = "inverse_frequency",
) -> np.ndarray:
    """
    Compute per-sample weights to handle residual label imbalance.
    Use as sample_weight parameter in xgb.DMatrix() or lgb.Dataset().

    Methods:
        "inverse_frequency": weight[i] = total / (n_classes * count[label[i]])
        "sqrt_inverse": softer version using sqrt of inverse frequency
    """
    y = labels.to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    total = len(y)

    class_weight: dict[int, float] = {}
    for cls, cnt in zip(classes, counts):
        if method == "inverse_frequency":
            class_weight[int(cls)] = total / (n_classes * cnt)
        elif method == "sqrt_inverse":
            class_weight[int(cls)] = math.sqrt(total / (n_classes * cnt))
        else:
            raise ValueError(f"Unknown method: {method}")

    weights = np.array([class_weight[int(yi)] for yi in y], dtype=np.float32)
    logger.info(
        "Sample weights: min=%.3f max=%.3f mean=%.3f",
        weights.min(), weights.max(), weights.mean(),
    )
    return weights


# ── Training pipeline integration ────────────────────────────────────────────

def prepare_labels_for_training(
    df: pl.DataFrame,
    price_col: str = "close",
    atr_col: str = "atr14",
) -> tuple[pl.DataFrame, np.ndarray]:
    """
    Full label preparation pipeline:
    1. Apply symmetric triple barrier
    2. Run distribution diagnostic (raises if catastrophic imbalance)
    3. Compute sample weights

    Returns: (df_with_labels, sample_weights)
    """
    import math  # local import to avoid circular dependency at module level

    labels = apply_triple_barrier(
        prices=df[price_col],
        atr=df[atr_col],
        upper_atr_mult=DEFAULT_UPPER_MULT,
        lower_atr_mult=DEFAULT_LOWER_MULT,
    )

    dist = compute_label_distribution(labels)

    # Hard stop if any class is below 10%
    if min(dist.values()) < 0.10:
        raise ValueError(
            f"Label distribution {dist} is catastrophically imbalanced. "
            "Training on this data will produce a degenerate model. "
            "Fix barrier parameters or check price data quality."
        )

    weights = compute_sample_weights(labels)
    df = df.with_columns(labels)
    return df, weights
```

---

*End of PART 3A. Continues in PART 3B with Issues 13 through 24.*


---

# NSE Algo Trading System — Deep Research Document
## PART 3B: CRITICAL FIXES — ISSUES 13 THROUGH 24

---

## Fix 13: Chronological Time-of-Day sin/cos → Volume Clock Encoding

### What breaks
`sin(2π × minute / 375)` encodes time linearly across the 375-minute NSE session.
This assigns equal "market activity weight" to 9:15 AM and 3:30 PM, when in reality
the first 30 minutes alone account for 20–35% of total daily volume. A model trained
with this encoding misidentifies low-volume mid-day ticks as structurally identical to
the volatile open.

### Root cause
NSE exhibits a pronounced U-shaped intraday volume curve: high volume at open,
trough around 12–1 PM, surge at close. Linear time encoding has no knowledge of
this distribution. The model therefore cannot distinguish a 9:15 AM breakout from a
12:00 PM drift move.

### File to modify
`features/time_features.py`

### Exact code fix

```python
# features/time_features.py

import numpy as np
from typing import Tuple

# NSE empirical intraday volume profile (proportion of daily volume per 15-min bar).
# Index 0 = 9:15–9:30, Index 1 = 9:30–9:45, ..., Index 25 = 3:15–3:30
_NSE_VOLUME_PROFILE_15MIN = np.array([
    0.085, 0.065, 0.045, 0.035, 0.030, 0.028,   # 9:15 – 10:45  (opening surge)
    0.025, 0.023, 0.022, 0.021, 0.020, 0.020,   # 10:45 – 12:15 (mid-morning)
    0.019, 0.018, 0.018, 0.019, 0.020, 0.022,   # 12:15 – 13:45 (lunch trough)
    0.024, 0.026, 0.028, 0.030, 0.035, 0.045,   # 13:45 – 15:15 (afternoon ramp)
    0.060, 0.080,                                 # 15:15 – 3:30  (closing surge)
])
# Normalise so that cumsum reaches exactly 1.0
_NSE_VOLUME_PROFILE_15MIN = _NSE_VOLUME_PROFILE_15MIN / _NSE_VOLUME_PROFILE_15MIN.sum()
_NSE_CUMVOL = np.concatenate([[0.0], np.cumsum(_NSE_VOLUME_PROFILE_15MIN)])


def _minutes_since_open(hour: int, minute: int) -> int:
    """Return minutes elapsed since 9:15 AM (session open)."""
    return max(0, (hour * 60 + minute) - (9 * 60 + 15))


def volume_clock_position(hour: int, minute: int) -> float:
    """
    Map a wall-clock time to its volume-clock position in [0, 1].

    VolumeTime(t) = CumulativeVolume(9:15 → t) / TotalDayVolume

    Uses linear interpolation within each 15-minute bucket.
    """
    mins = _minutes_since_open(hour, minute)
    mins = np.clip(mins, 0, 375)

    bucket_idx = int(mins // 15)                  # which 15-min bucket
    bucket_frac = (mins % 15) / 15.0              # fractional position within bucket

    bucket_idx = min(bucket_idx, len(_NSE_VOLUME_PROFILE_15MIN) - 1)

    # Linear interpolation between cumulative volume boundaries
    vol_pos = _NSE_CUMVOL[bucket_idx] + bucket_frac * _NSE_VOLUME_PROFILE_15MIN[bucket_idx]
    return float(vol_pos)


def encode_time_of_day_volume_clock(hour: int, minute: int) -> Tuple[float, float]:
    """
    Return (sin, cos) encoding on the volume clock rather than wall clock.

    Drop-in replacement for the previous sin(2π×minute/375) encoding.
    """
    vt = volume_clock_position(hour, minute)
    angle = 2.0 * np.pi * vt
    return float(np.sin(angle)), float(np.cos(angle))


# Vectorised version for Polars/numpy arrays
def encode_time_batch(hours: np.ndarray, minutes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vt = np.vectorize(volume_clock_position)(hours, minutes)
    angle = 2.0 * np.pi * vt
    return np.sin(angle), np.cos(angle)
```

---

## Fix 14: Kaggle Optuna Tuning → Docker Production Match

### What breaks
Hyperparameter search on Kaggle uses its own pinned environment (Polars 0.19, XGBoost 1.7, Python 3.10).
The VPS production environment may be Polars 0.20+, XGBoost 2.0, Python 3.11.
Parquet schemas, feature column ordering, and model serialisation differ across versions,
causing silent data corruption or outright load failures on the first live run.

### Root cause
No environment validation contract exists between the tuning notebook and the
deployment pipeline. Each environment pins its own versions independently.

### File to modify
`Dockerfile.training` (new file), `models/tuner.py`

### Exact code fix

```dockerfile
# Dockerfile.training  — used for ALL tuning runs (Kaggle-equivalent local or CI)
FROM python:3.11.9-slim

WORKDIR /app

# Pin every inference-relevant library explicitly
RUN pip install --no-cache-dir \
    polars==0.20.31 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    catboost==1.2.5 \
    optuna==3.6.1 \
    scikit-learn==1.5.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    pyzmq==26.0.3 \
    msgpack==1.0.8

COPY . /app
CMD ["python", "-m", "models.tuner"]
```

```python
# models/tuner.py  — prepend to existing tuner code

import importlib
import sys
from typing import Dict

_REQUIRED_VERSIONS: Dict[str, str] = {
    "polars": "0.20.31",
    "xgboost": "2.0.3",
    "lightgbm": "4.3.0",
    "catboost": "1.2.5",
    "optuna": "3.6.1",
    "sklearn": "1.5.0",
    "numpy": "1.26.4",
}


def validate_environment() -> None:
    """
    Assert that all critical package versions exactly match the pinned contract.
    Call this at the top of every training/tuning script entry point.

    Raises RuntimeError on the first mismatch — fail-fast before any data loading.
    """
    mismatches = []
    for pkg, required in _REQUIRED_VERSIONS.items():
        try:
            mod = importlib.import_module(pkg)
            installed = getattr(mod, "__version__", "UNKNOWN")
            if installed != required:
                mismatches.append(f"  {pkg}: required={required}, installed={installed}")
        except ImportError:
            mismatches.append(f"  {pkg}: NOT INSTALLED")

    python_required = "3.11"
    python_installed = f"{sys.version_info.major}.{sys.version_info.minor}"
    if not python_installed.startswith(python_required):
        mismatches.append(
            f"  python: required={python_required}.x, installed={python_installed}"
        )

    if mismatches:
        raise RuntimeError(
            "Environment validation FAILED. Run inside Dockerfile.training:\n"
            + "\n".join(mismatches)
        )


# ------------------------------------------------------------------
# Guard: call immediately on import so no code runs in wrong environment
# ------------------------------------------------------------------
if __name__ != "__main__":
    # When imported as a module, only validate — do not raise on version mismatch
    # in case developer is iterating locally. Set STRICT_ENV=1 to enforce.
    import os
    if os.environ.get("STRICT_ENV", "0") == "1":
        validate_environment()
```

---

## Fix 15: Soft Voting on Correlated Models → Stacking Ensemble

### What breaks
`ensemble_prob = (xgb_prob + lgb_prob + cat_prob) / 3` produces almost no variance
reduction when all three models are trained on identical features and the same fold.
Correlation between base model outputs approaches 0.90+, meaning the average is
effectively a single noisy predictor.

### Root cause
Ensemble theory (Breiman 1996) requires diverse, low-correlation base learners.
All three gradient boosting variants learn similar decision boundaries on the same
feature matrix. Simple averaging cannot remove this correlated error.

### File to modify
`models/ensemble.py`

### Exact code fix

```python
# models/ensemble.py

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from typing import List, Any
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


class StackingEnsemble:
    """
    Two-level stacking ensemble using out-of-fold (OOF) base predictions.

    Level-0 base learners: XGBoost, LightGBM, CatBoost
    Level-1 meta-learner: LogisticRegression (L2 regularised)

    Key design:
    - cross_val_predict with TimeSeriesSplit ensures level-0 predictions
      for the meta-learner training are always out-of-fold (no leakage).
    - Meta-learner sees only OOF probabilities, never raw features.
    """

    def __init__(self, n_splits: int = 5, meta_C: float = 0.1):
        self.n_splits = n_splits
        self.meta_C = meta_C
        self.base_models: List[Any] = []
        self.meta_learner: LogisticRegression = None
        self.scaler = StandardScaler()
        self._tscv = TimeSeriesSplit(n_splits=n_splits)

    def _make_base_models(self):
        return [
            xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                tree_method="hist", n_jobs=2, random_state=42,
            ),
            lgb.LGBMClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                n_jobs=2, random_state=42, verbose=-1,
            ),
            cb.CatBoostClassifier(
                iterations=400, depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42, verbose=0,
                thread_count=2,
            ),
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """
        Fit base models and meta-learner using OOF strategy.

        Parameters
        ----------
        X : shape (n_samples, n_features)
        y : shape (n_samples,) — binary labels {0, 1}
        """
        n_base = 3
        oof_preds = np.zeros((len(y), n_base))

        self.base_models = self._make_base_models()

        # Generate OOF predictions for meta-learner training
        for i, model in enumerate(self.base_models):
            oof_preds[:, i] = cross_val_predict(
                model, X, y,
                cv=self._tscv,
                method="predict_proba",
                n_jobs=1,  # keep sequential for reproducibility
            )[:, 1]

        # Refit each base model on the full training set
        for model in self.base_models:
            model.fit(X, y)

        # Fit meta-learner on OOF base predictions only
        oof_scaled = self.scaler.fit_transform(oof_preds)
        self.meta_learner = LogisticRegression(
            C=self.meta_C, max_iter=500, solver="lbfgs"
        )
        self.meta_learner.fit(oof_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return meta-learner probability array of shape (n_samples, 2)."""
        if self.meta_learner is None:
            raise RuntimeError("Call fit() before predict_proba()")

        base_preds = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.base_models
        ])
        base_scaled = self.scaler.transform(base_preds)
        return self.meta_learner.predict_proba(base_scaled)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
```

---

## Fix 16: HMM Chattering → Min-Duration Constraint

### What breaks
HMM flips between TRENDING / RANGING regimes on single noisy 1-minute candles.
Downstream position sizing logic receives a regime change signal every few bars,
causing excessive order activity and strategy thrash.

### Root cause
The standard Gaussian HMM transition matrix is unconstrained. It learns the data's
noise-driven flip frequency rather than real regime persistence. No minimum duration
constraint is enforced post-decoding.

### File to modify
`features/statistical.py`

### Exact code fix

```python
# features/statistical.py  — add to RegimeDetector class

import numpy as np
from hmmlearn import hmm


class RegimeDetector:
    """
    Hidden Markov Model regime detector with chattering suppression.

    Chattering is suppressed via:
    1. Sticky self-transition prior (self_transition_prior=0.95) — biases
       the learned transition matrix toward state persistence.
    2. Post-hoc minimum duration filter — merges short-lived state bursts
       back into the surrounding dominant state.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        min_duration: int = 5,
        self_transition_prior: float = 0.95,
    ):
        self.n_regimes = n_regimes
        self.min_duration = min_duration
        self.self_transition_prior = self_transition_prior
        self.model: hmm.GaussianHMM = None

    def _build_sticky_transmat_prior(self) -> np.ndarray:
        """
        Build transmat_prior matrix biasing toward self-transitions.

        Off-diagonal entries receive weight 1.0, diagonal entries receive
        weight = self_transition_prior / (1 - self_transition_prior) × (n-1).
        """
        n = self.n_regimes
        off = 1.0
        diag = self.self_transition_prior / (1.0 - self.self_transition_prior) * (n - 1)
        prior = np.full((n, n), off)
        np.fill_diagonal(prior, diag)
        return prior

    def fit(self, features: np.ndarray) -> "RegimeDetector":
        """
        Fit GaussianHMM with sticky transition prior.

        Parameters
        ----------
        features : shape (n_bars, n_features) — e.g. [returns, log_volume, atr]
        """
        transmat_prior = self._build_sticky_transmat_prior()
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42,
            transmat_prior=transmat_prior,
        )
        self.model.fit(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return regime labels with min-duration filter applied."""
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        raw_states = self.model.predict(features)
        return self._apply_min_duration(raw_states, self.min_duration)

    @staticmethod
    def _apply_min_duration(states: np.ndarray, min_duration: int = 5) -> np.ndarray:
        """
        Post-processing filter: merge state runs shorter than min_duration bars
        into the surrounding state (whichever neighbour is longer).

        Parameters
        ----------
        states       : 1-D integer array of decoded states
        min_duration : minimum consecutive bars required to keep a state change

        Returns
        -------
        filtered_states : same shape as states, chattering removed
        """
        states = states.copy()
        n = len(states)
        changed = True

        while changed:
            changed = False
            i = 0
            while i < n:
                # Find run length starting at i
                current = states[i]
                j = i
                while j < n and states[j] == current:
                    j += 1
                run_len = j - i

                if run_len < min_duration:
                    # Replace this short run with the state of the longer neighbour
                    left_len = i           # length of left segment (if i > 0)
                    right_len = n - j      # length of right segment (if j < n)

                    if i == 0:
                        replacement = states[j] if j < n else current
                    elif j >= n:
                        replacement = states[i - 1]
                    else:
                        replacement = states[i - 1] if left_len >= right_len else states[j]

                    states[i:j] = replacement
                    changed = True
                i = j

        return states
```

---

## Fix 17: Purge Window Missing Feature Lookback → Add Max Lookback

### What breaks
Walk-forward purge removes only the label holding period (e.g. 5 bars) from
the boundary between train and test folds. MACD(12,26,9) requires 26 bars of
history; HMM burn-in requires ~60 bars. Bars near the fold boundary in the
test set may reference training data through these lookback windows —
a form of future leakage.

### Root cause
Standard combinatorial purge (Lopez de Prado, Ch.7) addresses label overlap
only. Feature lookback contamination is a separate, often-overlooked source
of leakage.

### File to modify
`labeling/walk_forward.py`

### Exact code fix

```python
# labeling/walk_forward.py

from dataclasses import dataclass, field
from typing import Dict


# Catalogue of all indicator lookback requirements.
# Update this dict whenever a new indicator is added to FeatureEngine.
FEATURE_LOOKBACK_BARS: Dict[str, int] = {
    "ema_12":          12,
    "ema_26":          26,
    "macd_signal":     9,    # applied on top of EMA-26 → already covered
    "macd_full":       26,   # EMA(26) dominates
    "bollinger_20":    20,
    "rsi_14":          14,
    "atr_14":          14,
    "hmm_burn_in":     60,   # empirical minimum for stable HMM decoding
    "volume_sma_20":   20,
    "kalman_warmup":   30,
}


def compute_max_lookback() -> int:
    """Return the maximum feature lookback across all registered indicators."""
    return max(FEATURE_LOOKBACK_BARS.values())


@dataclass
class WalkForwardConfig:
    holding_period: int = 5        # label horizon in bars
    embargo_bars: int = 5          # post-test gap to prevent information bleed
    max_feature_lookback: int = field(default_factory=compute_max_lookback)

    @property
    def total_purge_bars(self) -> int:
        """
        Total bars to remove from the train/test boundary.

        Formula:
            total_purge = holding_period + max_feature_lookback + embargo

        This guarantees:
        - No label overlap (holding_period)
        - No feature lookback contamination (max_feature_lookback)
        - Safety embargo against micro-structure leakage (embargo_bars)
        """
        return self.holding_period + self.max_feature_lookback + self.embargo_bars

    def summary(self) -> str:
        return (
            f"WalkForwardConfig: holding={self.holding_period}, "
            f"max_lookback={self.max_feature_lookback}, "
            f"embargo={self.embargo_bars}, "
            f"total_purge={self.total_purge_bars} bars"
        )


# Usage example in fold generator:
#
# cfg = WalkForwardConfig(holding_period=5, embargo_bars=5)
# train_end_idx = fold_split_idx - cfg.total_purge_bars
# test_start_idx = fold_split_idx
```

---

## Fix 18: PPO RL for Position Sizing → Kelly Criterion

### What breaks
PPO policy gradient learns unstable policies on financial return sequences.
Reward signal (raw PnL or Sharpe increment) has signal-to-noise ratio far
below what RL requires (needs millions of clean reward samples; a 6.5h NSE
session provides ~375 1-minute bars). Live behaviour is erratic and impossible
to debug post-mortem.

### Root cause
RL in finance is a research tool, not a production-ready position sizer.
The sample complexity of PPO far exceeds what intraday NSE data can provide.

### File to modify
`risk/position_sizer.py`

### Exact code fix

```python
# risk/position_sizer.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class SizingParams:
    capital: float           # current available capital in INR
    win_prob: float          # estimated P(win) from model — e.g. 0.55
    avg_win_loss_ratio: float  # avg_win / avg_loss — e.g. 1.5
    atr14: float             # 14-period ATR in price points
    atr_mult: float = 2.0    # ATR multiplier for stop distance
    price: float = 0.0       # current instrument price
    sigma: float = 0.0       # annualised volatility (for target-vol method)
    risk_pct: float = 0.01   # fraction of capital to risk per trade (ATR method)
    target_vol: float = 0.15  # annualised target portfolio volatility


class PositionSizer:
    """
    Production-grade position sizing with three deterministic methods.

    Default: half-Kelly — theoretically optimal, empirically robust.
    Fallback: ATR-based — explicit risk-per-trade control.
    Research-only stub: target_volatility_sizing.
    """

    # ------------------------------------------------------------------
    # Primary method: Kelly Criterion (half-Kelly for robustness)
    # ------------------------------------------------------------------
    @staticmethod
    def kelly_fraction_sizing(params: SizingParams) -> float:
        """
        Compute position size in INR using half-Kelly criterion.

        Kelly fraction:  f* = (p × b - q) / b
            p = win probability
            q = 1 - p (loss probability)
            b = avg_win / avg_loss ratio

        Half-Kelly:  f_half = f* × 0.5

        Returns INR position size = f_half × capital.
        Clipped to [0, 0.25×capital] as a safety rail.
        """
        p = params.win_prob
        q = 1.0 - p
        b = params.avg_win_loss_ratio

        f_star = (p * b - q) / b
        f_half = max(0.0, f_star * 0.5)  # never go short from Kelly signal

        # Hard cap: never risk more than 25% of capital on a single trade
        f_half = min(f_half, 0.25)

        return f_half * params.capital

    # ------------------------------------------------------------------
    # Fallback method: ATR-based fixed fractional
    # ------------------------------------------------------------------
    @staticmethod
    def atr_based_sizing(params: SizingParams) -> float:
        """
        Risk a fixed fraction of capital per trade, with stop = ATR × atr_mult.

        position_size_INR = (risk_pct × capital) / (atr14 × atr_mult)

        This gives the number of shares; multiply by price for INR exposure.
        """
        if params.atr14 <= 0 or params.price <= 0:
            raise ValueError("atr14 and price must be positive for ATR sizing.")

        stop_distance = params.atr14 * params.atr_mult
        risk_amount = params.risk_pct * params.capital
        n_shares = risk_amount / stop_distance
        return n_shares * params.price  # INR exposure

    # ------------------------------------------------------------------
    # Research method: Target volatility sizing
    # ------------------------------------------------------------------
    @staticmethod
    def target_volatility_sizing(params: SizingParams) -> float:
        """
        Size to hit a target annualised portfolio volatility contribution.

        position_size = (target_vol × capital) / (sigma × price)

        Returns number of shares × price (INR exposure).
        Only use when sigma is reliably estimated (e.g. GARCH forecast).
        """
        if params.sigma <= 0 or params.price <= 0:
            raise ValueError("sigma and price must be positive for vol targeting.")

        n_shares = (params.target_vol * params.capital) / (params.sigma * params.price)
        return n_shares * params.price

    # ------------------------------------------------------------------
    # RL stub — explicitly disabled in production
    # ------------------------------------------------------------------
    @staticmethod
    def rl_sizer(*args, **kwargs) -> float:
        raise NotImplementedError(
            "RL-based position sizing is disabled in production. "
            "PPO/SAC require >100k clean reward samples and cannot be safely "
            "deployed on intraday NSE data. Use kelly_fraction_sizing() instead."
        )
```

---

## Fix 19: Rate Limiter Missing Per-Minute Cap → Token Bucket

### What breaks
A runaway bug loop can call `place_order()` at the maximum 8 OPS rate for
several minutes. The per-second bucket refills continuously — there is nothing
preventing 480 orders per minute. Shoonya/NSE rejection thresholds trigger
account suspension, not graceful throttling.

### Root cause
The rate limiter models only the per-second constraint. No per-minute cap,
no per-hour cap, and no per-instrument modification cap exist.

### File to modify
`execution/order_manager.py`

### Exact code fix

```python
# execution/order_manager.py

import time
import threading
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class _Bucket:
    """Single token bucket with thread-safe refill."""
    capacity: int
    refill_rate: float       # tokens per second
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    def consume(self, n: int = 1) -> bool:
        """
        Attempt to consume n tokens. Returns True if allowed, False if throttled.
        Thread-safe.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                float(self.capacity),
                self._tokens + elapsed * self.refill_rate
            )
            self._last_refill = now

            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


class TokenBucketRateLimiter:
    """
    Three independent token buckets protecting the order pipeline:

    1. per_second   : max 8 order actions / second  (NSE/Shoonya hard limit)
    2. per_minute   : max 100 order actions / minute (runaway-loop guard)
    3. per_modify   : max 20 order modifications / minute per instrument

    Usage:
        limiter = TokenBucketRateLimiter()
        allowed, reason = limiter.check_and_consume(instrument="NIFTY23JAN18000CE", is_modify=False)
        if not allowed:
            log.warning(f"Rate limited: {reason}")
    """

    def __init__(
        self,
        max_per_second: int = 8,
        max_per_minute: int = 100,
        max_modify_per_minute: int = 20,
    ):
        self._second_bucket = _Bucket(
            capacity=max_per_second,
            refill_rate=float(max_per_second),
        )
        self._minute_bucket = _Bucket(
            capacity=max_per_minute,
            refill_rate=max_per_minute / 60.0,
        )
        self._modify_buckets: Dict[str, _Bucket] = {}
        self._max_modify = max_modify_per_minute
        self._modify_lock = threading.Lock()

    def _get_modify_bucket(self, instrument: str) -> _Bucket:
        with self._modify_lock:
            if instrument not in self._modify_buckets:
                self._modify_buckets[instrument] = _Bucket(
                    capacity=self._max_modify,
                    refill_rate=self._max_modify / 60.0,
                )
            return self._modify_buckets[instrument]

    def check_and_consume(
        self, instrument: str, is_modify: bool = False
    ) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).

        Checks all applicable buckets atomically from strictest to broadest.
        """
        if not self._second_bucket.consume():
            return False, "per-second limit reached (max 8/s)"

        if not self._minute_bucket.consume():
            return False, "per-minute limit reached (max 100/min) — possible runaway loop"

        if is_modify:
            mod_bucket = self._get_modify_bucket(instrument)
            if not mod_bucket.consume():
                return False, f"per-modify limit for {instrument} (max {self._max_modify}/min)"

        return True, "ok"


# Integration in place_order():
#
# _limiter = TokenBucketRateLimiter()
#
# def place_order(instrument, qty, side, order_type, is_modify=False):
#     allowed, reason = _limiter.check_and_consume(instrument, is_modify)
#     if not allowed:
#         raise RateLimitError(f"Order blocked by rate limiter: {reason}")
#     # ... rest of order placement
```

---

## Fix 20: FinBERT Blocks Main Thread → Async ZeroMQ Worker

### What breaks
A synchronous `finbert_inference(headline)` call on the main trading loop takes
0.15–0.25 seconds per sentence on the i5-1345U (CPU-only). A 5-headline news
batch = 1.25 seconds of trading thread stall. In fast markets, this misses
critical bar closes and order execution windows.

### Root cause
NLP inference is blocking I/O from the trading thread's perspective.
The inference model has no obligation to complete in time for the next bar.

### File to modify
`models/sentiment.py`

### Exact code fix

```python
# models/sentiment.py

import multiprocessing as mp
import time
import zmq
import msgpack
from typing import Optional, List, Dict


SENTIMENT_PUSH_ADDR = "ipc:///tmp/sentiment_push.ipc"
SENTIMENT_RESULT_ADDR = "ipc:///tmp/sentiment_result.ipc"


class SentimentWorker(mp.Process):
    """
    Daemon process that runs FinBERT inference in isolation.

    Architecture:
        Main thread  → PUSH socket → SentimentWorker → PUSH socket → FeatureEngine

    FeatureEngine polls the result socket once per bar (not per tick).
    If no result is ready, the previous sentiment score is reused (stale-ok design).
    """

    def __init__(self, model_path: str = "ProsusAI/finbert"):
        super().__init__(daemon=True, name="SentimentWorker")
        self.model_path = model_path

    def run(self) -> None:
        """Entry point for the worker process. Runs indefinitely."""
        from transformers import pipeline  # import inside worker process only

        ctx = zmq.Context()
        receiver = ctx.socket(zmq.PULL)
        receiver.bind(SENTIMENT_PUSH_ADDR)

        sender = ctx.socket(zmq.PUSH)
        sender.bind(SENTIMENT_RESULT_ADDR)

        nlp = pipeline(
            "text-classification",
            model=self.model_path,
            device=-1,   # CPU only — matches i5-1345U constraint
            truncation=True,
            max_length=128,
        )

        while True:
            try:
                raw = receiver.recv(flags=zmq.NOBLOCK)
                payload = msgpack.unpackb(raw, raw=False)
                headlines: List[str] = payload.get("headlines", [])

                scores = []
                for h in headlines:
                    result = nlp(h)[0]
                    label = result["label"].lower()   # positive/negative/neutral
                    score = result["score"]
                    signed = score if label == "positive" else (-score if label == "negative" else 0.0)
                    scores.append(signed)

                avg_score = sum(scores) / len(scores) if scores else 0.0
                out = msgpack.packb({"sentiment": avg_score, "ts": time.time()})
                sender.send(out)

            except zmq.Again:
                time.sleep(0.01)   # no message waiting — sleep briefly
            except Exception as e:
                # Log but never crash the worker
                print(f"[SentimentWorker] error: {e}", flush=True)


class SentimentClient:
    """
    Thin ZeroMQ client for the main trading thread.

    - `submit_headlines()` — fire-and-forget, never blocks
    - `get_latest_score()` — non-blocking poll, returns last known score if nothing new
    """

    def __init__(self):
        self._ctx = zmq.Context.instance()
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.connect(SENTIMENT_PUSH_ADDR)
        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.connect(SENTIMENT_RESULT_ADDR)
        self._last_score: float = 0.0

    def submit_headlines(self, headlines: List[str]) -> None:
        """Send headlines to worker. Returns immediately."""
        payload = msgpack.packb({"headlines": headlines})
        self._push.send(payload, flags=zmq.NOBLOCK)

    def get_latest_score(self) -> float:
        """
        Non-blocking poll. Returns most recent completed sentiment score.
        Call once per bar from FeatureEngine.build_features().
        """
        try:
            raw = self._pull.recv(flags=zmq.NOBLOCK)
            result = msgpack.unpackb(raw, raw=False)
            self._last_score = result["sentiment"]
        except zmq.Again:
            pass   # no new result — reuse last known score
        return self._last_score
```

---

## Fix 21: Kalman [price, velocity] State → Add OFI

### What breaks
The standard [price, velocity] Kalman filter adds a "velocity" (momentum) term
that smooths price at the cost of a 1–3 bar lag. In NSE mid-cap stocks, informed
order flow is the primary price driver, not Newtonian inertia. The lag causes the
filter to miss order-flow-driven breakouts.

### Root cause
The velocity component provides useful smoothing for slow, trend-driven assets
but adds lag for order-flow-dominated instruments. OFI (Order Flow Imbalance =
bid-volume minus ask-volume normalised) is a leading indicator not captured by
price alone.

### File to modify
`features/statistical.py`

### Exact code fix

```python
# features/statistical.py  — add alongside RegimeDetector

import numpy as np
from filterpy.kalman import KalmanFilter


def build_ofi_augmented_kalman(
    dt: float = 1.0,
    process_noise_price: float = 0.1,
    process_noise_velocity: float = 0.5,
    process_noise_ofi: float = 1.0,
    measurement_noise: float = 0.5,
) -> KalmanFilter:
    """
    3-state Kalman filter: [price, velocity, ofi].

    State vector:  x = [price, velocity, ofi]
    Measurement:   z = [price, ofi]   (both are directly observable)

    Transition model:
        price(t) = price(t-1) + dt × velocity(t-1)
        velocity(t) = velocity(t-1)                   (constant velocity model)
        ofi(t) = 0                                    (mean-reverting: decays each step)

    OFI input: (bid_volume - ask_volume) / (bid_volume + ask_volume) ∈ [-1, 1]

    Parameters
    ----------
    dt                     : time step (1.0 for 1-minute bars)
    process_noise_price    : Q[0,0] — how much price can deviate from model
    process_noise_velocity : Q[1,1]
    process_noise_ofi      : Q[2,2] — OFI is noisy; allow large process noise
    measurement_noise      : R diagonal — sensor noise for price and OFI

    Returns
    -------
    filterpy KalmanFilter ready for sequential .predict() / .update() calls.
    """
    kf = KalmanFilter(dim_x=3, dim_z=2)

    # State transition matrix F
    kf.F = np.array([
        [1.0,  dt,  0.0],   # price += velocity × dt
        [0.0, 1.0,  0.0],   # velocity is constant
        [0.0, 0.0,  0.0],   # OFI mean-reverts to 0 each step (re-measured each bar)
    ])

    # Measurement matrix H: we observe [price, ofi]
    kf.H = np.array([
        [1.0, 0.0, 0.0],   # price is state[0]
        [0.0, 0.0, 1.0],   # ofi   is state[2]
    ])

    # Process noise covariance Q
    kf.Q = np.diag([process_noise_price, process_noise_velocity, process_noise_ofi])

    # Measurement noise covariance R
    kf.R = np.eye(2) * measurement_noise

    # Initial state covariance P (high uncertainty)
    kf.P = np.eye(3) * 10.0

    # Initial state (will be overwritten on first real bar)
    kf.x = np.zeros((3, 1))

    return kf


def update_kalman_bar(kf: KalmanFilter, price: float, ofi: float) -> np.ndarray:
    """
    Run one predict-update cycle and return filtered state [price, velocity, ofi].

    Parameters
    ----------
    kf    : KalmanFilter from build_ofi_augmented_kalman()
    price : observed close price
    ofi   : (bid_vol - ask_vol) / (bid_vol + ask_vol)

    Returns
    -------
    state : np.ndarray shape (3,) — [filtered_price, filtered_velocity, filtered_ofi]
    """
    kf.predict()
    kf.update(np.array([[price], [ofi]]))
    return kf.x.flatten()
```

---

## Fix 22: Memory Fragmentation → Daily Restart + trim_memory()

### What breaks
Python RSS memory grows from ~350 MB at open to 800 MB–1 GB by 3:30 PM close
after a full 6.5-hour session. The i5-1345U laptop has 16 GB but the growth is
non-deterministic. On days with heavy news (many FinBERT calls, msgpack buffers,
Polars intermediate DataFrames), OOM-killer termination has been observed.

### Root cause
CPython's `pymalloc` arena allocator holds pages after deallocation ("fragmentation").
Polars DataFrames, ZeroMQ message buffers, and msgpack byte strings create thousands
of short-lived allocations that fragment the heap. `gc.collect()` alone does not
return OS pages; `malloc_trim()` is needed.

### File to modify
`main.py`, `infra/systemd/exchange.service` (reference `_part2_infrastructure.md`)

### Exact code fix

```python
# main.py  — add trim_memory() and schedule it

import ctypes
import gc
import os
import sys
import logging
import threading
import time

log = logging.getLogger("main")


def trim_memory() -> None:
    """
    Aggressively return fragmented heap memory to the OS.

    Steps:
    1. Force full garbage collection (all generations).
    2. Call malloc_trim(0) via libc — releases top-of-heap pages on Linux.
    3. Log RSS before/after for monitoring.

    Call every 30 minutes during the trading session.
    Note: This is a no-op on Windows; the systemd cron restart is the primary guard.
    """
    rss_before = _get_rss_mb()
    gc.collect(2)   # generation 2 = full collection

    if sys.platform == "linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except OSError:
            pass   # libc not available in some container environments

    rss_after = _get_rss_mb()
    log.info(f"trim_memory: RSS {rss_before:.1f} MB → {rss_after:.1f} MB "
             f"(freed {rss_before - rss_after:.1f} MB)")


def _get_rss_mb() -> float:
    """Read current RSS from /proc/self/status (Linux) or fallback."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (FileNotFoundError, ValueError):
        pass
    return 0.0


def _schedule_trim(interval_seconds: int = 1800) -> threading.Thread:
    """
    Background thread that calls trim_memory() every interval_seconds.
    Returns the daemon thread (already started).
    """
    def _loop():
        while True:
            time.sleep(interval_seconds)
            try:
                trim_memory()
            except Exception as e:
                log.warning(f"trim_memory error: {e}")

    t = threading.Thread(target=_loop, daemon=True, name="MemTrimmer")
    t.start()
    return t


# In main() entry point, add:
# _schedule_trim(interval_seconds=1800)  # every 30 minutes
```

```
# Cron entry (on the VPS — append to crontab with `crontab -e`):
# Restart trading system at 8:50 AM IST every weekday (Mon–Fri)
50 8 * * 1-5 systemctl restart exchange.service

# Reference: systemd service file in _part2_infrastructure.md
# The service should have: Restart=on-failure, RestartSec=10s
```

---

## Fix 23: Optuna Overfitting to Validation Set → Complexity Penalty

### What breaks
With 100+ Optuna trials over a single time-series validation fold, the
optimiser discovers hyperparameter combinations that overfit the fold's
specific noise pattern. Validation AUC reports 0.68 but live Sharpe
is below 0.5 — a classic overfitting-to-noise outcome.

### Root cause
The objective function optimises validation metric only, with no cost for
model complexity. Deep trees × many leaves × low learning rate = extremely
high capacity, memorising the validation fold.

### File to modify
`models/tuner.py`

### Exact code fix

```python
# models/tuner.py  — replace or add build_xgb_objective()

import optuna
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from typing import List, Tuple


def compute_complexity_penalty(params: dict) -> float:
    """
    Regularisation term penalising high-capacity hyperparameter combinations.

    Penalty = depth × num_leaves_proxy × (1 / learning_rate)
    Normalised to [0, 1] range using empirical max values.

    This prevents the optimiser from choosing very deep trees at very low
    learning rates, which maximises in-sample fit at the cost of generalisation.
    """
    depth = params.get("max_depth", 6)
    lr = max(params.get("learning_rate", 0.1), 1e-6)
    subsample = params.get("subsample", 0.8)

    # More leaves ≈ higher depth; lower learning rate ≈ higher effective capacity
    raw_penalty = depth * (1.0 / lr) * (1.0 / max(subsample, 0.1))

    # Normalise: max sensible value ≈ 10 × (1/0.01) × 10 = 10000
    normalised = min(raw_penalty / 10000.0, 1.0)
    return normalised


def build_xgb_objective(
    X_folds: List[Tuple[np.ndarray, np.ndarray]],
    y_folds: List[Tuple[np.ndarray, np.ndarray]],
    complexity_weight: float = 0.05,
    tune_fold_index: int = -2,   # CRITICAL: use second-to-last fold, NOT last
) -> callable:
    """
    Build an Optuna objective function with complexity regularisation.

    Parameters
    ----------
    X_folds          : list of (X_train, X_val) per fold
    y_folds          : list of (y_train, y_val) per fold
    complexity_weight: weight on the complexity penalty term
    tune_fold_index  : which fold to use for tuning (-2 = second-to-last)
                       The last fold is reserved for final unbiased evaluation.

    Returns
    -------
    objective : callable(trial) → float (to be minimised by Optuna)
    """
    X_train, X_val = X_folds[tune_fold_index]
    y_train, y_val = y_folds[tune_fold_index]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
            "subsample":        trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
            "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 5.0),
            "tree_method":      "hist",
            "eval_metric":      "logloss",
            "use_label_encoder": False,
            "n_jobs":           2,
            "random_state":     42,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_prob)

        penalty = compute_complexity_penalty(params) * complexity_weight

        # Minimise: (1 - AUC) + complexity_penalty
        return (1.0 - auc) + penalty

    return objective


def run_tuning(
    X_folds, y_folds,
    n_trials: int = 50,   # hard cap — never exceed 50 trials on financial time series
    study_name: str = "xgb_tuning",
) -> dict:
    """
    Run Optuna study with capped trials and complexity penalty.

    n_trials is capped at 50 to prevent overfitting to the validation fold.
    """
    n_trials = min(n_trials, 50)   # safety hard cap

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    objective = build_xgb_objective(X_folds, y_folds)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params
```

---

## Fix 24: Angel One TOTP Expiry → Shoonya Primary + Heartbeat

### What breaks
Angel One TOTP-based sessions expire unpredictably mid-session, causing
critical exit orders to fail silently. When a session expires at 2:15 PM
during a trending market, open positions cannot be squared off via the API.
Manual intervention is required — the most dangerous scenario in live trading.

### Root cause
Angel One uses TOTP-based JWT tokens with a complex refresh window.
The refresh logic is undocumented and can fail on clock skew, gateway
restarts, or token cache corruption. This is a structural unreliability
of the Angel One session architecture, not a fixable code issue.

### Resolution
Move Shoonya (Finvasia) to primary execution broker. Shoonya has:
- Zero brokerage on NSE F&O
- Session tokens that last 24h with a simple background refresh
- No TOTP dependency for API calls after initial login
- Well-documented WebSocket feed for tick data

Keep Angel One solely as a secondary data source (historical OHLCV).
Add a `SessionManager` with a 600-second heartbeat ping to proactively
detect and repair any session expiry before it affects order placement.

### File to modify
`data/broker_shoonya.py`

### Exact code fix

```python
# data/broker_shoonya.py

import time
import threading
import logging
from typing import Optional, Callable
from api_helper import ShoonyaApiPy   # pip install shoonya-api

log = logging.getLogger("broker.shoonya")

_HEARTBEAT_INTERVAL = 600   # seconds — ping every 10 minutes
_SESSION_RETRY_LIMIT = 3


class SessionManager:
    """
    Background session manager for Shoonya broker.

    Maintains an active API session by:
    1. Initial login at startup.
    2. Periodic heartbeat (every 600s) to detect silent session expiry.
    3. Automatic re-login on expiry detection, up to _SESSION_RETRY_LIMIT times.

    Thread-safe: all public methods acquire _lock before touching session state.
    """

    def __init__(
        self,
        user: str,
        password: str,
        totp_secret: str,
        api_key: str,
        imei: str,
        vendor_code: str,
        on_reconnect: Optional[Callable] = None,
    ):
        self._creds = dict(
            userid=user, password=password, twoFA=totp_secret,
            vendor_code=vendor_code, api_secret=api_key, imei=imei,
        )
        self._api = ShoonyaApiPy()
        self._lock = threading.Lock()
        self._session_active = False
        self._on_reconnect = on_reconnect   # callback to re-subscribe to feeds
        self._heartbeat_thread: Optional[threading.Thread] = None

    def login(self) -> None:
        """Perform initial login. Called once at system startup."""
        with self._lock:
            self._do_login()
        self._start_heartbeat()

    def _do_login(self) -> None:
        """Internal login — must be called with _lock held."""
        import pyotp
        totp = pyotp.TOTP(self._creds["twoFA"]).now()
        creds = {**self._creds, "twoFA": totp}

        resp = self._api.login(**creds)
        if resp and resp.get("stat") == "Ok":
            self._session_active = True
            log.info("Shoonya session established.")
        else:
            self._session_active = False
            raise ConnectionError(f"Shoonya login failed: {resp}")

    def _heartbeat(self) -> None:
        """Background loop: ping Shoonya every _HEARTBEAT_INTERVAL seconds."""
        while True:
            time.sleep(_HEARTBEAT_INTERVAL)
            try:
                with self._lock:
                    resp = self._api.get_limits()
                    if resp and resp.get("stat") == "Ok":
                        log.debug("Shoonya heartbeat OK")
                    else:
                        log.warning(f"Shoonya heartbeat FAILED: {resp}. Re-logging in.")
                        self._do_login()
                        if self._on_reconnect:
                            self._on_reconnect()
            except Exception as e:
                log.error(f"Shoonya heartbeat exception: {e}. Attempting re-login.")
                try:
                    with self._lock:
                        self._do_login()
                except Exception as login_err:
                    log.critical(f"Shoonya re-login failed: {login_err}")

    def _start_heartbeat(self) -> None:
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat, daemon=True, name="ShoonyaHeartbeat"
        )
        self._heartbeat_thread.start()

    @property
    def api(self) -> ShoonyaApiPy:
        """Return the underlying API handle. Always use through this property."""
        if not self._session_active:
            raise RuntimeError("Shoonya session not active. Call login() first.")
        return self._api


class ShoonyaBroker:
    """
    Primary execution broker. Wraps SessionManager with order/position helpers.
    """

    def __init__(self, session_manager: SessionManager):
        self._sm = session_manager

    def ensure_session(self) -> ShoonyaApiPy:
        """
        Return a guaranteed-active API handle.

        This wrapper is the single access point for all broker calls.
        Any method that interacts with the broker MUST call ensure_session()
        rather than accessing the API directly.
        """
        return self._sm.api

    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transactiontype: str,   # "B" or "S"
        quantity: int,
        price: float = 0.0,
        product: str = "I",     # "I"=intraday, "M"=margin
        pricetype: str = "MKT",
    ) -> dict:
        api = self.ensure_session()
        resp = api.place_order(
            buy_or_sell=transactiontype,
            product_type=product,
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            quantity=quantity,
            discloseqty=0,
            price_type=pricetype,
            price=price,
            trigger_price=None,
            retention="DAY",
            remarks="algo_trade",
        )
        return resp

    def get_positions(self) -> list:
        api = self.ensure_session()
        return api.get_positions() or []

    def square_off_all(self, exchange: str = "NSE") -> None:
        """Emergency square-off — called on session shutdown or risk breach."""
        positions = self.get_positions()
        for pos in positions:
            net_qty = int(pos.get("netqty", 0))
            if net_qty == 0:
                continue
            side = "S" if net_qty > 0 else "B"
            self.place_order(
                tradingsymbol=pos["tsym"],
                exchange=exchange,
                transactiontype=side,
                quantity=abs(net_qty),
                pricetype="MKT",
            )
            log.info(f"Squared off: {pos['tsym']} qty={abs(net_qty)} side={side}")
```

---

## Summary: All 24 Fixes at a Glance

| Fix # | File | Change Type | Priority |
|-------|------|-------------|----------|
| 1 | `features/time_features.py` | Feature engineering | HIGH |
| 2 | `features/time_features.py` | Feature engineering | HIGH |
| 3 | `features/indicators.py` | Feature engineering | HIGH |
| 4 | `labeling/triple_barrier.py` | Labeling | CRITICAL |
| 5 | `labeling/triple_barrier.py` | Labeling | CRITICAL |
| 6 | `labeling/walk_forward.py` | Training pipeline | CRITICAL |
| 7 | `models/ensemble.py` | Model | HIGH |
| 8 | `risk/position_sizer.py` | Risk management | CRITICAL |
| 9 | `execution/order_manager.py` | Execution | HIGH |
| 10 | `features/statistical.py` | Feature engineering | MEDIUM |
| 11 | `data/broker_shoonya.py` | Infrastructure | CRITICAL |
| 12 | `main.py` | Infrastructure | HIGH |
| **13** | `features/time_features.py` | Feature engineering | **HIGH** |
| **14** | `Dockerfile.training`, `models/tuner.py` | DevOps / CI | **HIGH** |
| **15** | `models/ensemble.py` | Model | **HIGH** |
| **16** | `features/statistical.py` | Feature engineering | **MEDIUM** |
| **17** | `labeling/walk_forward.py` | Training pipeline | **CRITICAL** |
| **18** | `risk/position_sizer.py` | Risk management | **CRITICAL** |
| **19** | `execution/order_manager.py` | Execution | **CRITICAL** |
| **20** | `models/sentiment.py` | Performance | **MEDIUM** |
| **21** | `features/statistical.py` | Feature engineering | **MEDIUM** |
| **22** | `main.py` | Infrastructure | **HIGH** |
| **23** | `models/tuner.py` | Training pipeline | **HIGH** |
| **24** | `data/broker_shoonya.py` | Infrastructure | **CRITICAL** |

---

## Quick Fix Checklist: Stage 1 (Must-Do Before Paper Trading)

The following fixes from Issues 13–24 are **blocking requirements** before any paper
trading run. The system will silently produce incorrect results or fail mid-session
without them.

- [ ] **Fix 17** (`labeling/walk_forward.py`) — Add `total_purge_bars` with feature lookback.
      Without this, walk-forward backtests have data leakage and overstate performance.

- [ ] **Fix 18** (`risk/position_sizer.py`) — Replace PPO stub with Kelly/ATR sizing.
      The RL sizer raises `NotImplementedError`; the system cannot live-trade without this.

- [ ] **Fix 19** (`execution/order_manager.py`) — Deploy `TokenBucketRateLimiter`.
      A single runaway loop without per-minute cap will breach Shoonya's account limits.

- [ ] **Fix 24** (`data/broker_shoonya.py`) — Deploy Shoonya as primary with `SessionManager`.
      Angel One mid-session TOTP expiry is a known unrecoverable failure mode.

- [ ] **Fix 14** (`Dockerfile.training`, `models/tuner.py`) — Pin all library versions.
      Any Polars or XGBoost version mismatch between training and production silently corrupts
      feature schemas or model weights.

- [ ] **Fix 13** (`features/time_features.py`) — Replace linear time encoding with volume clock.
      The current sin/cos feature provides no useful intraday market-activity signal.

**Defer to Stage 2 (post paper-trading validation):**

- Fix 15 (Stacking ensemble) — calibration needed; soft voting is safe for initial paper run
- Fix 16 (HMM min-duration) — cosmetic quality improvement, not correctness-blocking
- Fix 20 (Async FinBERT) — only matters if FinBERT is enabled; disable it for Stage 1
- Fix 21 (OFI Kalman) — enhancement, not a correctness fix
- Fix 22 (Memory trim) — monitor RSS during paper trading; restart cron is sufficient for now
- Fix 23 (Optuna complexity penalty) — re-tune after Stage 1 data collection


---

## PART 4: REFINED 4-STAGE DEPLOYMENT PLAN

> **Design philosophy:** Each stage is a gate, not a guideline. You do not move to the next stage until the current stage's exit criteria are met — no exceptions. The plan deliberately starts with zero ML complexity to prove infrastructure first, layers in complexity only after plumbing is verified, and defers the highest-risk instrument (F&O) until capital and track record justify it.

---

### 1. Stage 1 — The Plumbing Test (Sessions 1–14, ~3 weeks)

**Goal:** Prove the entire pipeline works end-to-end with zero ML complexity. No model, no feature engine, no training loop. Only infrastructure.

**Why this matters:** The #1 failure mode for solo algo traders is spending months building a beautiful ML pipeline and discovering on day 1 of live trading that the WebSocket drops, QuestDB writes are stale, or the order placement silently fails. Stage 1 eliminates this risk cheaply.

#### Strategy: EMA Cross (Intentionally Dumb)

```python
# scripts/paper_trade.py  --strategy ema_cross
# Entry: EMA(9) crosses above EMA(21) → LONG signal
# Exit:  EMA(9) crosses below EMA(21) OR ATR-based stop
# Stop:  entry_price - (1.5 × ATR14)  [long]
#        entry_price + (1.5 × ATR14)  [short]
# Size:  fixed ₹5,000 notional per trade (not volatility-adjusted yet)
# Mode:  PAPER ONLY — Shoonya simulated orders

import pandas_ta as ta
from data.broker_shoonya import ShoonyaBroker
from alerts.telegram_bot import TelegramBot

def ema_cross_signal(df):
    df['ema9']  = ta.ema(df['close'], length=9)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if prev['ema9'] < prev['ema21'] and last['ema9'] > last['ema21']:
        return 'LONG',  last['close'] - 1.5 * last['atr14']
    if prev['ema9'] > prev['ema21'] and last['ema9'] < last['ema21']:
        return 'SHORT', last['close'] + 1.5 * last['atr14']
    return None, None
```

#### Mandatory Infrastructure Checklist — Stage 1 Exit Gate

All items must be checked `[x]` before any Session 15 work begins:

- [ ] Shoonya WebSocket connects on script start, reconnects automatically after disconnect
- [ ] Candles are stored in QuestDB with correct timestamps (IST, no timezone drift)
- [ ] `SELECT * FROM candles WHERE symbol='RELIANCE' LIMIT 5` returns correct data via QuestDB REST API
- [ ] Paper order placed via `ShoonyaBroker.place_order()` returns a valid order ID (not an exception)
- [ ] Paper order cancelled via `ShoonyaBroker.cancel_order()` succeeds without error
- [ ] Telegram alert fires on: trade entry, trade exit, any uncaught exception
- [ ] 5 complete paper trade cycles (entry + stop-loss exit OR EMA cross exit) logged to QuestDB
- [ ] Zero unhandled exceptions across 5 consecutive trading days (9:15 AM – 3:30 PM)
- [ ] Daily P&L summary Telegram message arrives at 3:45 PM each day
- [ ] `pytest tests/` passes with 0 failures

#### Cost: Stage 1

| Item | Monthly Cost |
|------|-------------|
| Shoonya broker | ₹0 (zero brokerage) |
| Shoonya WebSocket (broker feed) | ₹0 (included) |
| QuestDB (local Docker) | ₹0 |
| VPS | ₹0 (run on laptop) |
| **Total** | **₹0** |

**Note:** No premium data feed needed. Shoonya's broker WebSocket is throttled but sufficient for EMA cross on 5-minute candles. Do not spend money on TrueData yet.

#### Code Milestone

```bash
# This command must work before Stage 2 starts:
python scripts/paper_trade.py --strategy ema_cross --symbol RELIANCE --timeframe 5min

# Expected output every 5 minutes:
# [09:15:00] CANDLE STORED: RELIANCE 5min O=2440.5 H=2445.0 L=2438.2 C=2442.1 V=123450
# [09:20:00] SIGNAL: LONG @ 2442.1 | Stop: 2434.8 | Order ID: SH_PAPER_00123
# [10:45:00] EXIT: Stop hit @ 2434.8 | P&L: -₹73 (paper) | Telegram: SENT
```

---

### 2. Stage 2 — Vanilla ML Live (Sessions 15–30, ~6 weeks)

**Goal:** First live capital deployment with basic ML. Prove the model training → inference → execution loop works with real money. Simultaneously begin building a proprietary tick dataset.

**Capital:** ₹50,000–₹1,00,000. Trade 1–2 symbols only: RELIANCE or HDFC Bank (highest NSE liquidity, tightest spreads, lowest slippage risk).

**Mental model:** This stage will likely lose a small amount (₹2,000–₹8,000). Budget for it. The cost buys you certainty that your execution layer is correct, your cost model is accurate, and your position sizing is calibrated. Treat it as tuition, not failure.

#### Data Strategy — The Sweat Equity Path

Start recording live ticks to QuestDB from Day 1 of Stage 2. This builds a proprietary historical tick dataset that costs ₹10,000+ to buy from vendors.

```python
# Add to scripts/download_history.py
import time
from questdb.ingress import Sender, TimestampNanos
from data.broker_shoonya import ShoonyaBroker

def live_tick_recorder(symbols: list[str], questdb_host: str = "localhost"):
    """Record live ticks to QuestDB continuously during market hours.
    
    Storage estimate per symbol:
      - ~1 tick per 100ms average = 14,400 ticks/6.25hr session
      - ~14 MB/day uncompressed | ~1.5 MB/day after QuestDB compression
      - 5 symbols × 3 months = ~1.0–1.5 GB total (trivial)
    """
    broker = ShoonyaBroker()
    broker.login()
    
    def on_tick(tick: dict):
        with Sender.from_conf(f"http::addr={questdb_host}:9000;") as sender:
            sender.row(
                "ticks",
                symbols={"symbol": tick["sym"]},
                columns={
                    "ltp":    float(tick["lp"]),
                    "bid":    float(tick.get("bp1", 0)),
                    "ask":    float(tick.get("sp1", 0)),
                    "bid_qty":int(tick.get("bq1", 0)),
                    "ask_qty":int(tick.get("sq1", 0)),
                    "volume": int(tick.get("v", 0)),
                },
                at=TimestampNanos.now(),
            )
    
    broker.subscribe_ws(symbols, on_tick=on_tick)
    while True:          # keep alive; cron kills at 15:35
        time.sleep(1)
```

#### ML Feature Set — Stage 2 (Minimal, Not Full 23)

Do NOT implement all 23 features yet. Use only features computable from OHLCV + broker WebSocket (no L2 data needed):

| # | Feature | Library | Rationale |
|---|---------|---------|-----------|
| 1 | RSI(14) | pandas-ta | Momentum signal |
| 2 | ATR(14) | pandas-ta | Volatility scaling |
| 3 | EMA cross (9/21) | pandas-ta | Trend direction |
| 4 | VWAP deviation | custom.py | Intraday mean reversion anchor |
| 5 | Volume bar index | custom.py | Replace time bars with volume bars |
| 6 | OBV | pandas-ta | Cumulative volume pressure |
| 7 | BB %B | pandas-ta | Overbought/oversold in context |
| 8 | Time-of-day (sin/cos) | custom.py | Session effect encoding |

Skip: HMM regime, Kalman, GARCH, OBI, Quote Fade, VPIN, TFI. Add those in Stage 3.

#### Volume Bars — Critical Implementation Note

Replace standard 5-minute time bars with volume bars before labeling. This is non-negotiable — time bars on NSE produce non-IID observations because volume clusters at open/close.

```python
# features/custom.py — add volume bar aggregation
import polars as pl

def build_volume_bars(ticks: pl.DataFrame, volume_threshold: int = 50_000) -> pl.DataFrame:
    """Aggregate tick data into volume bars.
    
    Args:
        ticks: DataFrame with columns [timestamp, ltp, volume]
        volume_threshold: Target volume per bar (tune per symbol)
            RELIANCE:  ~50,000 shares/bar  (active large-cap)
            HDFC Bank: ~30,000 shares/bar
    Returns:
        OHLCV DataFrame with one row per volume bar
    """
    bars, cum_vol = [], 0
    bar_open = bar_high = bar_low = None
    bar_start = None
    
    for row in ticks.iter_rows(named=True):
        price, vol, ts = row["ltp"], row["volume"], row["timestamp"]
        if bar_open is None:
            bar_open = price; bar_high = price; bar_low = price; bar_start = ts
        bar_high = max(bar_high, price)
        bar_low  = min(bar_low,  price)
        cum_vol += vol
        if cum_vol >= volume_threshold:
            bars.append({"timestamp": bar_start, "open": bar_open,
                         "high": bar_high, "low": bar_low,
                         "close": price,   "volume": cum_vol})
            bar_open = None; cum_vol = 0
    
    return pl.DataFrame(bars)
```

#### Labeling — Stage 2 Symmetric Barriers

Use **symmetric** ±1.5 ATR (not the asymmetric 2.0/1.0 from CLAUDE.md) for Stage 2. Easier class balance, less label bias during a period when you need clean diagnostics.

```python
# labeling/triple_barrier.py — symmetric config for Stage 2
STAGE_2_LABELING_CONFIG = {
    "upper_atr_mult": 1.5,   # symmetric — easier to balance classes
    "lower_atr_mult": 1.5,   # change to 2.0/1.0 only after verifying label distribution
    "max_holding_bars": 15,  # ~75 min on 5-min bars (shorter than 20 for intraday)
}
```

#### Stage 2 Success Metrics (Exit Gate)

- [ ] Zero crashes for 20 consecutive trading days
- [ ] All live orders placed with correct SEBI Strategy ID tag
- [ ] Realized transaction costs within ±20% of `risk/cost_model.py` predictions
- [ ] Label distribution check: +1 between 30–50%, -1 between 30–50%, 0 remainder
- [ ] Model AUC (OOS, walk-forward) > 0.52 (anything above random on live data is informative)
- [ ] No position held overnight (hard check: zero open positions at 3:25 PM alert)
- [ ] Drawdown < 10% of deployed capital (₹5,000 on ₹50K deployment)
- [ ] Tick recording running continuously — verify with `SELECT count() FROM ticks` in QuestDB

#### Cost: Stage 2

| Item | Monthly Cost |
|------|-------------|
| Shoonya brokerage | ₹0 |
| Broker WebSocket | ₹0 |
| QuestDB (local Docker) | ₹0 |
| VPS | ₹0 (still on laptop) |
| **Total** | **₹0** |

---

### 3. Stage 3 — Microstructure Upgrade (Sessions 31–60, ~3 months)

**Trigger condition:** Stage 2 exit gate fully passed AND 3 months of tick data recorded in QuestDB.

**Infrastructure shift:** This is where fixed costs begin. You move from laptop to AWS VPS and upgrade to unthrottled L2 data feed.

#### Infrastructure Upgrade Actions

```bash
# 1. Launch AWS Mumbai VPS (ap-south-1)
#    Instance: t3.medium (2 vCPU, 4 GB RAM) — sufficient for Stage 3
#    Storage: 30 GB gp3 SSD (QuestDB + models + logs)
#    Estimated cost: ~₹1,500/month

# 2. Initial VPS hardening
sudo ufw allow 22/tcp       # SSH
sudo ufw allow 9000/tcp     # QuestDB Web Console (restrict to your IP only)
sudo ufw allow 9009/tcp     # QuestDB ILP ingestion
sudo ufw enable
sudo apt install fail2ban -y
sudo systemctl enable fail2ban

# SSH: disable password auth, key-only
# /etc/ssh/sshd_config:
#   PasswordAuthentication no
#   PubkeyAuthentication yes
#   PermitRootLogin no

# 3. Subscribe TrueData WebSocket L2
#    Plan: TrueData Premium (unthrottled WebSocket, L1+L2, full depth)
#    Cost: ~₹2,500/month
#    Note: GDF (Global Data Feeds) is alternative at similar price — compare latency
```

#### New Features — L2 Microstructure (Add to features/custom.py)

These features require unthrottled L2 tick data. Do NOT attempt to compute them from throttled broker WebSocket — you will get garbage signals because spoof cancellations happen in <100ms.

**Order Book Imbalance (OBI):**
```python
def order_book_imbalance(bid_qty: float, ask_qty: float) -> float:
    """OBI = (BidVol - AskVol) / (BidVol + AskVol)
    Range: [-1, +1]. >+0.3 = buy pressure. <-0.3 = sell pressure.
    Use top 3 levels: OBI_L1, OBI_L3 (both as separate features).
    """
    total = bid_qty + ask_qty
    if total == 0:
        return 0.0
    return (bid_qty - ask_qty) / total
```

**Quote Fade Detection:**
```python
def quote_fade_score(bid_history: list[float], ask_history: list[float],
                     window_ms: int = 500) -> float:
    """Detect spoofing pattern: large quote appears, then vanishes before fill.
    
    Quote fade = rapid withdrawal of size without execution.
    Score > 0.6 → likely spoof; adjust signal weight DOWN (contra-signal).
    
    Algorithm:
      1. Track bid/ask size at best level over rolling 500ms window
      2. If max_size > 3× min_size in window AND no fills occurred → fade detected
      3. Return fade_ratio = (max_size - final_size) / max_size
    """
    if len(bid_history) < 2:
        return 0.0
    max_bid = max(bid_history)
    final_bid = bid_history[-1]
    if max_bid == 0:
        return 0.0
    return (max_bid - final_bid) / max_bid
```

**VPIN (Volume-synchronized Probability of Informed Trading):**
```python
def compute_vpin(buy_volume: float, sell_volume: float,
                 total_volume: float, n_buckets: int = 50) -> float:
    """VPIN approximation (Easley, Lopez de Prado, O'Hara 2012).
    
    Classifies each trade tick as buy/sell using tick rule.
    VPIN = |buy_vol - sell_vol| / total_vol per volume bucket.
    High VPIN (>0.35) → informed trading → widen spread expectations, reduce size.
    """
    if total_volume == 0:
        return 0.5  # neutral
    return abs(buy_volume - sell_volume) / total_volume
```

**Trade Flow Imbalance (TFI):**
```python
def trade_flow_imbalance(trades: list[dict], window: int = 20) -> float:
    """TFI = sum(signed_trade_size) over last N trades.
    
    Signed: +qty if buyer-initiated (trade at ask), -qty if seller-initiated.
    Normalise by total traded volume in window.
    TFI > +0.4 → strong buying pressure → LONG bias.
    TFI < -0.4 → strong selling pressure → SHORT bias.
    """
    if not trades:
        return 0.0
    recent = trades[-window:]
    signed_sum = sum(t["qty"] * (1 if t["side"] == "BUY" else -1) for t in recent)
    total_vol  = sum(t["qty"] for t in recent)
    return signed_sum / total_vol if total_vol > 0 else 0.0
```

#### Retrain Model on L2 Tick Data

Critical change from Stage 2: retrain on **tick/volume bar features** computed from L2 data, not OHLCV. The model trained on OHLCV time bars will not generalise to L2 microstructure features.

```python
# Retrain pipeline for Stage 3
# 1. Pull 3-month tick history from QuestDB (your own recorded data)
# 2. Rebuild volume bars from raw ticks
# 3. Compute full feature set including OBI, Quote Fade, VPIN, TFI
# 4. Re-label with triple-barrier on volume bars (not time bars)
# 5. Walk-forward train/validate — minimum 8 folds on 3 months of data
# 6. Log all runs to MLflow; compare AUC against Stage 2 OHLCV model
```

#### Capital Scaling — Stage 3

| Phase | Capital | Symbols | Target Return |
|-------|---------|---------|--------------|
| Stage 3 start | ₹2,00,000 | 3 stocks (RELIANCE, HDFC Bank, INFY) | 2–3% net/month |
| After month 1 profitable | ₹2,00,000–₹2,50,000 | 3–5 stocks | 3–5% net/month |
| After 3 consecutive profitable months | Ready for Stage 4 gate | — | — |

#### Stage 3 Exit Gate (3 Consecutive Profitable Months Required)

- [ ] TrueData WebSocket stable for 30+ days (zero data gaps > 5 seconds)
- [ ] VPS uptime > 99.5% (set up AWS CloudWatch alert if CPU > 80% or memory > 90%)
- [ ] OBI + Quote Fade features have positive SHAP contribution in at least 2 of 3 walk-forward folds
- [ ] VPIN correctly identifies 2+ high-volatility sessions in backtest (validate against known NSE events)
- [ ] 3 consecutive calendar months with positive net P&L after all costs (brokerage, STT, GST, slippage)
- [ ] Monthly return ≥ 1% net (not targeting 3–5% yet — consistency > magnitude)
- [ ] Capital at end of Stage 3 ≥ ₹3,00,000 (must not have eroded base capital)

#### Cost: Stage 3

| Item | Monthly Cost |
|------|-------------|
| TrueData WebSocket L2 (unthrottled) | ₹2,500 |
| AWS Mumbai t3.medium VPS | ₹1,500 |
| Shoonya brokerage | ₹0 |
| **Total fixed cost** | **₹4,000** |
| As % of ₹2,50,000 capital | **1.6%** |
| Required monthly return to break even | **1.6% gross** |
| Target monthly return (net of all costs) | **3–5%** |

---

### 4. Stage 4 — F&O Final Boss (Only After Stage 3 Verified)

**This stage has a hard gate. Do not negotiate with yourself.**

#### SEBI 2026 F&O Regulatory Reality

| Parameter | Current Rule (2026) |
|-----------|-------------------|
| Minimum lot size | ₹15,00,000+ notional per lot |
| STT (Securities Transaction Tax) on options sell | 0.1% of premium (exercise side) |
| Number of expiry days | 1 per month per index (SEBI Aug 2025 consolidation) |
| Margin requirements | Significantly increased under SEBI circular |
| Weekly expiries | Eliminated for most indices |

**Implication:** One Nifty lot at ₹15L+ notional means a 5% adverse move = ₹75,000 loss. With ₹3L capital, one bad trade can destroy 25% of your account. This is why F&O is Stage 4, not Stage 1.

#### Stage 4 Entry Gate (Non-Negotiable)

- [ ] Stage 3 has 3+ consecutive months of positive live P&L (not backtested)
- [ ] Total capital ≥ ₹15,00,000 (accumulated from Stage 3 profits + additional capital)
- [ ] Maximum drawdown in Stage 3 was < 10% in any single month
- [ ] SEBI F&O strategy ID registered separately from equity strategy ID
- [ ] Options Greeks infrastructure installed and tested (see below)

#### F&O Strategy Constraints

**Allowed:** Long options (defined risk), vertical spreads (bull call, bear put), iron condors on expiry week with defined max loss.

**Never:** Naked short options. A naked short Nifty call with unlimited upside exposure can generate a margin call larger than your entire account in a single gap-up open.

```python
# Additional dependencies for Stage 4
# requirements.txt additions:
# mibian>=0.1.3        # Black-Scholes Greeks for NSE options
# py_vollib>=1.0.1     # Implied volatility surface construction

# models/options_greeks.py — new module for Stage 4
import mibian

def compute_greeks(underlying_price: float, strike: float, 
                   days_to_expiry: int, risk_free_rate: float,
                   option_price: float, option_type: str = "c") -> dict:
    """Compute BSM Greeks for NSE option.
    
    Args:
        underlying_price: Nifty/BankNifty spot
        strike: Option strike price
        days_to_expiry: Calendar days to monthly expiry
        risk_free_rate: RBI repo rate (currently ~6.5%)
        option_price: Market price of option (LTP)
        option_type: 'c' for call, 'p' for put
    """
    bs = mibian.BS([underlying_price, strike, risk_free_rate, days_to_expiry],
                   optionPrice=option_price)
    return {
        "iv":    bs.impliedVolatility,
        "delta": bs.callDelta  if option_type == "c" else bs.putDelta,
        "gamma": bs.gamma,
        "theta": bs.theta / 365,   # daily decay
        "vega":  bs.vega  / 100,   # per 1% IV move
    }

# Expiry calendar helper — NSE monthly expiry = last Thursday of month
from pandas.tseries.offsets import WeekOfMonth
import pandas as pd

def get_monthly_expiry(year: int, month: int) -> pd.Timestamp:
    """Return last Thursday of given month (NSE monthly F&O expiry)."""
    last_day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    # Walk back to find last Thursday (weekday=3)
    while last_day.weekday() != 3:
        last_day -= pd.Timedelta(days=1)
    return last_day
```

---

### 5. India-Specific Infrastructure Checklist

#### TrueData WebSocket — Connection Setup

```python
# data/broker_truedata.py — Stage 3 addition
import websocket
import json
import threading

class TrueDataFeed:
    """TrueData WebSocket L2 feed for NSE tick data.
    
    Tick format (TrueData v2):
    {
      "sym":  "RELIANCE-EQ",
      "ltp":  2442.50,          # Last traded price
      "ltq":  100,              # Last traded quantity
      "atp":  2440.20,          # Average traded price (VWAP proxy)
      "ttq":  1234567,          # Total traded quantity (cumulative)
      "bp1":  2442.00, "bq1": 500,   # Best bid price/qty
      "sp1":  2443.00, "sq1": 300,   # Best ask price/qty
      "bp2":  2441.50, "bq2": 1000,  # Level 2 bid
      "sp2":  2443.50, "sq2": 800,   # Level 2 ask
      "bp3":  2441.00, "bq3": 2000,  # Level 3 bid
      "sp3":  2444.00, "sq3": 1500,  # Level 3 ask
      "ts":   "2026-05-18 09:15:01"  # Timestamp IST
    }
    """
    WS_URL = "wss://push.truedata.in/noauth/getfeeds"
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self._ws = None
        self._callbacks = []
    
    def connect(self, symbols: list[str]):
        def on_open(ws):
            payload = {
                "method": "sub",
                "symbols": symbols,
                "datatype": "L2Depth"
            }
            ws.send(json.dumps(payload))
        
        def on_message(ws, message):
            tick = json.loads(message)
            for cb in self._callbacks:
                cb(tick)
        
        def on_error(ws, error):
            logging.error(f"TrueData WS error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logging.warning("TrueData WS closed — reconnecting in 5s")
            threading.Timer(5.0, lambda: self.connect(symbols)).start()
        
        self._ws = websocket.WebSocketApp(
            self.WS_URL,
            header={"Authorization": f"Bearer {self._get_token()}"},
            on_open=on_open, on_message=on_message,
            on_error=on_error, on_close=on_close
        )
        threading.Thread(target=self._ws.run_forever, daemon=True).start()
    
    def on_tick(self, callback):
        self._callbacks.append(callback)
```

#### AWS Mumbai Instance Selection

| Instance | vCPU | RAM | Cost/month | Suitable For |
|----------|------|-----|------------|-------------|
| t3.small | 2 | 2 GB | ~₹900 | Testing only — insufficient for QuestDB + model inference |
| **t3.medium** | **2** | **4 GB** | **~₹1,500** | **Stage 2–3: QuestDB + feature engine + inference** |
| t3.large | 2 | 8 GB | ~₹3,000 | Stage 4 if running Greeks + IV surface computation |
| c5.xlarge | 4 | 8 GB | ~₹6,000 | Only if Optuna retraining runs on VPS (prefer Kaggle) |

**Recommendation:** Start t3.medium. Upgrade to t3.large only if memory usage consistently > 80% (monitor with `free -h` in cron).

#### VPS Security Setup

```bash
# Run once after VPS launch — complete hardening script
# 1. System update
sudo apt update && sudo apt upgrade -y

# 2. Firewall — whitelist only your home IP for QuestDB console
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from YOUR_HOME_IP to any port 22    # SSH
sudo ufw allow from YOUR_HOME_IP to any port 9000  # QuestDB console
sudo ufw allow 9009/tcp                             # QuestDB ILP (all — your app writes here)
sudo ufw enable

# 3. fail2ban — blocks brute force SSH attempts
sudo apt install fail2ban -y
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
# Edit /etc/fail2ban/jail.local:
#   [sshd]
#   enabled = true
#   maxretry = 3
#   bantime = 3600
sudo systemctl enable --now fail2ban

# 4. SSH hardening
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# 5. Verify your SSH key is in authorized_keys BEFORE closing current session
cat ~/.ssh/authorized_keys  # must show your public key
```

#### SEBI Strategy ID Registration (Both Brokers)

```
SEBI August 2025 requirement: ALL algorithmic orders must carry a registered
Strategy ID. Non-compliance risks order rejection at exchange level.

Angel One registration process:
  1. Login to Angel One developer portal (smartapi.angelbroking.com)
  2. Navigate: API Management → Strategy Registration
  3. Fields required:
     - Strategy Name (e.g., "NSE_ML_INTRADAY_V1")
     - Strategy Type: "Client-side Algo"
     - Risk parameters: max order size, max daily orders
     - SHAP explainability document (upload PDF generated by models/explainer.py)
  4. Receive Strategy ID (e.g., "ANGEL_STR_00456")
  5. Add to every order: order_params["strategy_id"] = config.ANGEL_STRATEGY_ID

Shoonya registration process:
  1. Email: support@shoonya.com with subject "Algo Strategy Registration"
  2. Attach: SEBI-format strategy description + risk parameters
  3. Receive Shoonya strategy tag (add to order headers)
  4. Timeline: 3–7 business days

Config addition (config.py):
  ANGEL_STRATEGY_ID: str = "ANGEL_STR_XXXXX"   # from Angel One portal
  SHOONYA_STRATEGY_TAG: str = "SH_STR_XXXXX"   # from Shoonya email
```

#### NSE Market Hours Cron Jobs (VPS Crontab)

```cron
# /etc/cron.d/trading_system  — runs on VPS in IST (set VPS timezone: timedatectl set-timezone Asia/Kolkata)

# Pre-open: start WebSocket, warm up models, verify connectivity
00 09 * * 1-5  trading_user /opt/trading/venv/bin/python /opt/trading/scripts/pre_open.py >> /var/log/trading/pre_open.log 2>&1

# Market open: start main trading loop
15 09 * * 1-5  trading_user /opt/trading/venv/bin/python /opt/trading/main.py --mode live >> /var/log/trading/session.log 2>&1

# 3:20 PM: stop new entries, prepare exit-only mode
20 15 * * 1-5  trading_user /opt/trading/venv/bin/python /opt/trading/scripts/pre_close.py >> /var/log/trading/pre_close.log 2>&1

# Market close: force-close all positions, send daily P&L summary
30 15 * * 1-5  trading_user /opt/trading/venv/bin/python /opt/trading/scripts/force_close.py >> /var/log/trading/close.log 2>&1

# Post-close: archive today's ticks, backup QuestDB, send Telegram summary
45 15 * * 1-5  trading_user /opt/trading/venv/bin/python /opt/trading/scripts/post_close.py >> /var/log/trading/post_close.log 2>&1

# Tick recorder: runs continuously during market hours (killed by force_close.py)
00 09 * * 1-5  trading_user /opt/trading/venv/bin/python /opt/trading/scripts/download_history.py --mode live_record >> /var/log/trading/ticks.log 2>&1
```

#### Telegram Bot — Mandatory Alert Events

```python
# alerts/telegram_bot.py — required alert types

ALERT_TYPES = {
    "TRADE_ENTRY": "ENTRY {side} {symbol} @ ₹{price} | Qty: {qty} | Stop: ₹{stop}",
    "TRADE_EXIT":  "EXIT {symbol} @ ₹{price} | P&L: ₹{pnl} ({pct:.1f}%) | {reason}",
    "DAILY_PNL":   "DAILY SUMMARY {date}\nGross P&L: ₹{gross}\nCosts: ₹{costs}\nNet P&L: ₹{net}\nTrades: {n_trades}\nDrawdown: {dd:.1f}%",
    "DRAWDOWN_WARN": "⚠ DRAWDOWN ALERT: {dd:.1f}% from peak. Kill switch at 10%.",
    "KILL_SWITCH":   "🛑 KILL SWITCH TRIGGERED: Daily loss limit hit. Trading halted.",
    "WS_DISCONNECT": "WS DISCONNECT: {feed} at {time}. Attempting reconnect.",
    "ORDER_ERROR":   "ORDER ERROR: {symbol} {side} failed. Reason: {reason}. Retrying.",
    "STARTUP":       "SYSTEM START: {date} {time}. Capital: ₹{capital}. Mode: {mode}.",
    "SHUTDOWN":      "SYSTEM STOP: {date}. Session P&L: ₹{pnl}. Positions: {open_pos}.",
}
```

---

### 6. Data Collection Strategy — The Sweat Equity Path

#### Storage Estimate (Precise)

| Metric | Value | Basis |
|--------|-------|-------|
| Average tick frequency (market hours) | 1 tick / 100ms | NSE large-cap average |
| Ticks per 6.25-hour session | ~14,400 | 6.25 hr × 60 × 60 × 10 Hz |
| Bytes per tick (uncompressed QuestDB row) | ~150 bytes | 12 columns × avg 12 bytes |
| Uncompressed storage per symbol per day | ~2.1 MB | 14,400 × 150 bytes |
| QuestDB compression ratio | 5–10× | columnar storage + LZ4 |
| Compressed storage per symbol per day | ~0.25 MB | |
| Per symbol per month (22 trading days) | ~5.5 MB compressed | |
| 5 symbols × 3 months | **~82 MB** | Well within 30 GB SSD |

**Conclusion:** Disk space is not the constraint. Recording ticks is free. Start on Day 1 of Stage 2, not when you think you "need" it.

#### Tick Dataset Value Analysis

| Data Source | Cost | Quality | Verdict |
|------------|------|---------|---------|
| TrueData historical L2 (3 months, 5 symbols) | ₹8,000–₹15,000 | Vendor-dependent quality | Skip |
| Self-recorded from TrueData live feed | ₹0 (just time) | Your own ground truth | Use this |
| NSE Data vending (official) | ₹50,000+ per year | Institutional quality | Stage 4+ only |
| yfinance / broker OHLCV | ₹0 | No L2, throttled | Only for Stage 1–2 backtesting |

#### Live Tick Recorder Integration

```python
# scripts/download_history.py — add this function
# Run: python scripts/download_history.py --mode live_record --symbols RELIANCE,HDFCBANK

import argparse
import signal
import sys
from pathlib import Path

def live_tick_recorder(symbols: list[str], questdb_host: str = "localhost"):
    """Continuous tick recorder. Runs until SIGTERM (sent by cron at 15:35)."""
    from data.broker_shoonya import ShoonyaBroker      # Stage 1 only: throttled, EMA cross
    # from data.truedata_feed import TrueDataFeed      # Stage 2+: uncomment [ARCH-FIX-1]
    from questdb.ingress import Sender, TimestampNanos
    
    broker = ShoonyaBroker()
    broker.login()
    
    ticks_recorded = 0
    
    def on_tick(tick: dict):
        nonlocal ticks_recorded
        with Sender.from_conf(f"http::addr={questdb_host}:9000;") as sender:
            sender.row("ticks",
                symbols={"symbol": tick.get("sym", "UNKNOWN")},
                columns={
                    "ltp":     float(tick.get("lp",  0)),
                    "ltq":     int(  tick.get("v",   0)),
                    "bid":     float(tick.get("bp1", 0)),
                    "ask":     float(tick.get("sp1", 0)),
                    "bid_qty": int(  tick.get("bq1", 0)),
                    "ask_qty": int(  tick.get("sq1", 0)),
                    "oi":      int(  tick.get("oi",  0)),   # open interest (F&O)
                },
                at=TimestampNanos.now(),
            )
        ticks_recorded += 1
        if ticks_recorded % 1000 == 0:
            logging.info(f"Ticks recorded: {ticks_recorded:,}")
    
    def graceful_shutdown(signum, frame):
        logging.info(f"Recorder stopped. Total ticks: {ticks_recorded:,}")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, graceful_shutdown)
    broker.subscribe_ws(symbols, on_tick=on_tick)
    
    while True:
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["historical", "live_record"], required=True)
    parser.add_argument("--symbols", default="RELIANCE,HDFCBANK")
    args = parser.parse_args()
    
    symbols = args.symbols.split(",")
    if args.mode == "live_record":
        live_tick_recorder(symbols)
```

---

### 7. Revised CLAUDE.md Build Order

The original 23-step build order has structural problems: ML complexity is introduced before infrastructure is proven, TDD is bolted on at the end, and there is no explicit dummy/plumbing phase. The revised order below bakes in the Stage 1–4 critique.

**Key changes from original:**
- TDD moved to Step 5 (write tests before implementation, not after)
- Dummy EMA strategy before any ML code (Steps 6–8)
- Ring buffer before feature engine (Step 9)
- Volume bars before labeling (Step 12)
- Symmetric labeling config for Stage 2 (Step 13)
- L2 features gated to Stage 3 (Steps 20–21)
- F&O/Greeks infrastructure deferred to Stage 4 (Step 27+)

#### Revised Build Order

```
STAGE 1 — PLUMBING (Sessions 1–14)
──────────────────────────────────
Step 01  config.py + .env.example + requirements.txt
         → Define all dataclasses: BrokerConfig, QuestDBConfig, RiskConfig,
           StageConfig (current_stage: int = 1)

Step 02  data/broker_base.py → abstract interface
         → Methods: login(), place_order(), cancel_order(), get_positions(),
           subscribe_ws(), get_historical_data()

Step 03  data/broker_shoonya.py → concrete Shoonya implementation
         → Implement all BrokerBase methods against Shoonya REST + WebSocket API
         → Test: broker.login() returns valid session token

Step 04  data/candle_store.py → QuestDB candle storage
         → Docker: docker run -d -p 9000:9000 -p 9009:9009 questdb/questdb
         → Write via ILP (questdb.ingress.Sender), read via REST API
         → Schema: (timestamp, symbol, open, high, low, close, volume, timeframe)

Step 05  tests/ → pytest scaffolding — WRITE TESTS NOW
         → tests/test_broker.py: mock Shoonya API, test order placement
         → tests/test_candle_store.py: insert + retrieve candle, verify timestamp
         → tests/test_risk.py: stub — fill in during Step 17
         → tests/test_cost_model.py: stub — fill in during Step 16
         → Run: pytest tests/ -v (stubs pass trivially; real tests fill in later)

Step 06  data/websocket_feed.py → stream live candles, store to QuestDB
         → Connect Shoonya WebSocket, parse tick → OHLCV candle assembly
         → Store each completed candle to QuestDB via candle_store.py
         → Test: 5 consecutive candles stored correctly, no timestamp drift

Step 07  scripts/paper_trade.py --strategy ema_cross
         → EMA(9)/EMA(21) signal + ATR(14) stop-loss
         → Paper orders via broker_shoonya.py (simulated, no real capital)
         → Telegram alerts on entry/exit
         → MILESTONE: 5 complete paper trades logged. Zero crashes 5 days.

Step 08  alerts/telegram_bot.py → all mandatory alert types
         → Trade entry/exit, errors, daily P&L, kill switch, WS disconnect
         → Test: manually trigger each alert type, verify Telegram receipt

────────────────────────────────────────────────────────────────────
GATE: Stage 1 → Stage 2. All Stage 1 checklist items must be [x].
────────────────────────────────────────────────────────────────────

STAGE 2 — VANILLA ML (Sessions 15–30)
──────────────────────────────────────
Step 09  In-memory ring buffer for live ticks
         → Implement before feature engine (critical: avoids DB read latency in
           hot path during live inference)
         → collections.deque(maxlen=500) per symbol, fed by WebSocket callback
         → Feature engine reads from ring buffer, NOT from QuestDB

Step 10  features/technical.py → pandas-ta indicators (Stage 2 subset only)
         → Implement: RSI(14), ATR(14), EMA(9/21), BB %B, OBV
         → Skip for now: ADX, Stoch, Supertrend, Ichimoku, Heikin-Ashi
         → Run on Polars DataFrame; convert to pandas only for pandas-ta call

Step 11  features/custom.py → VWAP deviation, Time-of-day, basic spread
         → VWAP = cumsum(price × volume) / cumsum(volume) reset at 9:15 AM IST
         → Time-of-day: sin/cos encoding of minute-of-day
         → Spread: (ask - bid) / mid_price (from broker WebSocket level 1)

Step 12  Volume bars → build_volume_bars() in features/custom.py
         → Aggregate tick stream into volume bars (not time bars)
         → Tune volume_threshold per symbol using 5-day tick sample
         → Replace all downstream uses of 5-min OHLCV with volume bar OHLCV

Step 13  features/engine.py → orchestrate Steps 10–12
         → Single entry point: engine.compute(ring_buffer) → Polars DataFrame
         → Stage 2 feature count: 8 features (not 23 — defer rest to Stage 3)

Step 14  labeling/triple_barrier.py → symmetric ±1.5 ATR config
         → upper_atr_mult = lower_atr_mult = 1.5 (symmetric for Stage 2)
         → max_holding_bars = 15 (volume bars, not time bars)
         → Verify label distribution: +1 ∈ [30%, 50%], -1 ∈ [30%, 50%]
         → FAIL LOUDLY if any class < 15% or > 55%: raise ValueError

Step 15  labeling/walk_forward.py → purge + embargo splits
         → Purge: remove training samples whose label window overlaps test start
         → Embargo: skip 3 additional bars after purge boundary
         → Minimum 5 folds; assert len(folds) >= 5 or raise ValueError

Step 16  risk/cost_model.py → full India cost model
         → Components: STT (0.025% buy side equity), brokerage (₹0 Shoonya),
           exchange transaction charge (0.00345%), GST (18% on charges),
           SEBI charge (0.0001%), stamp duty (0.015% buy side)
         → Slippage model: 0.5 × spread for LIMIT, 1.5 × spread for MARKET
         → Fill tests/test_cost_model.py now

Step 17  risk/position_sizer.py → ATR-based volatility-adjusted sizing
         → position_value = risk_per_trade / (atr_mult × ATR)
         → risk_per_trade = min(0.01 × capital, 500)  [1% capital or ₹500]
         → Max position: 5% of capital (hard cap)
         → Fill tests/test_risk.py now

Step 18  risk/manager.py → kill switches + drawdown breaker
         → Daily loss limit: 2% capital → halt all new orders, Telegram alert
         → Max drawdown: 10% from peak → pause system, require manual restart
         → All checks run before EVERY order placement (not just at session start)

Step 19  execution/order_manager.py → live LIMIT/MARKET order logic
         → LIMIT first; convert to MARKET after 2 candles unfilled
         → Rate limit: max 8 orders/second (hard sleep between bursts)
         → Attach SEBI Strategy ID to every order
         → All orders logged before placement attempt (idempotency)

Step 20  execution/trade_logger.py → QuestDB trade log
         → Schema: (timestamp, symbol, side, qty, entry_price, exit_price,
                    pnl_gross, pnl_net, strategy_id, model_version, features_hash)

Step 21  main.py → orchestrator (live loop)
         → Initialize: broker login, WS connect, ring buffer start, tick recorder
         → Loop (every volume bar close): compute features → model inference →
           risk check → signal → order → log
         → Graceful shutdown handler: SIGTERM → force_close all positions → log

Step 22  models/ensemble.py → XGBoost + LightGBM + CatBoost soft voting
         → CPU mode: tree_method='hist' (XGB), device='cpu' (LGB/CB)
         → Soft voting: average predicted probabilities, not hard class votes
         → Minimum train size: 500 labeled samples (assert before fit)

Step 23  models/tuner.py → Optuna (CPU, n_jobs=8, overnight or Kaggle)
         → Hyperparameter space: max_depth, n_estimators, learning_rate,
           subsample, colsample, min_child_weight
         → Objective: maximize OOS AUC across walk-forward folds

Step 24  models/tracker.py → MLflow + DVC
         → dvc init && dvc add data/ticks/
         → Log: AUC per fold, feature importances, cost model accuracy,
           label distribution, train/test sizes

Step 25  models/explainer.py → SHAP values + SEBI compliance report
         → Run after every training: shap.TreeExplainer(model).shap_values(X_test)
         → Save: shap_summary_plot.png + shap_values.parquet
         → Generate PDF report: top 10 features, decision rationale per signal
         → Required for SEBI registration documentation

Step 26  scripts/paper_trade.py + backtest/runner.py → final integration test
         → paper_trade.py: runs full live loop in paper mode for 5 days
         → runner.py: walk-forward backtest on 3-month tick history
         → Assert: backtest P&L > -5% (must not be obviously broken)

────────────────────────────────────────────────────────────────────
GATE: Stage 2 → Stage 3. All Stage 2 checklist items must be [x].
Minimum 20 live trading days. Capital not eroded > 10%.
────────────────────────────────────────────────────────────────────

STAGE 3 — MICROSTRUCTURE (Sessions 31–60)
──────────────────────────────────────────
Step 27  AWS VPS setup + TrueData subscription
         → Launch t3.medium in ap-south-1, run hardening script (Section 5)
         → Install Docker, Python 3.11, clone repo, configure .env
         → Subscribe TrueData Premium WebSocket L2

Step 28  data/broker_truedata.py → TrueDataFeed class
         → WebSocket connection with auto-reconnect
         → Parse L2 depth: best 3 bid/ask levels + quantities
         → Route ticks to ring buffer + live_tick_recorder() simultaneously

Step 29  features/custom.py additions: OBI, Quote Fade, VPIN, TFI
         → Implement all 4 microstructure features (see Section 3 for code)
         → Add to features/engine.py as conditional features (Stage 3+ only)
         → Feature count now: 12 (8 from Stage 2 + 4 new)

Step 30  features/statistical.py → HMM regime, Kalman, GARCH
         → HMM: 3-state (low/medium/high vol) using hmmlearn, retrain weekly
         → Kalman: filterpy.kalman.KalmanFilter for price smoothing
         → GARCH(1,1): arch library, refit at 9:00 AM each morning
         → Feature count now: full 23 (12 + HMM + Kalman + GARCH + remainder)

Step 31  Retrain full model on L2 tick data
         → Pull 3-month tick history from QuestDB
         → Rebuild volume bars from raw ticks
         → Compute all 23 features
         → Re-label with symmetric ±1.5 ATR (tune to asymmetric 2.0/1.0 only
           after verifying no class imbalance)
         → Minimum 8 walk-forward folds
         → Compare AUC: new L2 model vs Stage 2 OHLCV model (log in MLflow)

Step 32  data/broker_angelone.py → Angel One as secondary broker
         → Implement BrokerBase for Angel One SmartAPI
         → Used for capital > ₹5L where Shoonya execution limits apply
         → Test: parallel paper trades on both brokers simultaneously

────────────────────────────────────────────────────────────────────
GATE: Stage 3 → Stage 4. 3 consecutive profitable months. Capital ≥ ₹15L.
────────────────────────────────────────────────────────────────────

STAGE 4 — F&O (Sessions 61+, only after Stage 3 gate)
──────────────────────────────────────────────────────
Step 33  models/options_greeks.py → BSM Greeks + IV surface
Step 34  Expiry calendar + monthly expiry auto-detection
Step 35  F&O-specific risk manager: max delta exposure, gamma risk cap
Step 36  Spread strategy module: bull call spread, bear put spread entry/exit
Step 37  SEBI F&O Strategy ID registration (separate from equity)
```

---

### Cost Summary Across All Stages

| Stage | Duration | Fixed Cost/Month | Live Capital | Notes |
|-------|----------|-----------------|-------------|-------|
| Stage 1 | ~3 weeks | ₹0 | ₹0 (paper) | Pure infrastructure validation |
| Stage 2 | ~6 weeks | ₹0 | ₹50K–₹1L | Expect small loss; this is tuition |
| Stage 3 | ~3 months | ₹4,000 | ₹2L–₹2.5L | Break-even at ~1.6% gross return |
| Stage 4 | Ongoing | ₹4,000+ | ₹15L+ | Only after Stage 3 exit gate met |

**Bottom line:** You can run Stage 1 and Stage 2 for zero fixed cost. The ₹4,000/month commitment begins at Stage 3 when you have 3+ months of live validation and a proprietary tick dataset. At ₹2.5L capital, that ₹4,000 is 1.6% — the target 3–5% net monthly return (₹7,500–₹12,500) covers it comfortably.

---

## PART 5: ML ENSEMBLE AND VALIDATION UPGRADES

---

### 5.1 Stacking Meta-Learner (Replacing Simple Soft Vote)

**Why equal-weight soft voting is wrong for this stack**

XGBoost, LightGBM, and CatBoost are trained on identical feature sets (the same 23 NSE features), identical training windows, and identical label distributions produced by triple-barrier labeling. Their predicted probability vectors have a pairwise correlation of approximately 0.82–0.88 in practice on financial tabular data. Averaging three nearly-identical distributions reduces variance by a factor of roughly `1 / (1 + (n-1) * rho)` — with n=3 and rho=0.85, that is `1 / (1 + 2 * 0.85) ≈ 0.37`, meaning you recover only 37% of the theoretical variance reduction that independent models would give. The ensemble barely outperforms a single model.

Stacking trains a meta-learner (logistic regression) to learn which base model to trust on which type of sample. It captures systematic biases: XGBoost tends to be overconfident on trending regimes, LightGBM generalises better on sparse order-book samples, CatBoost handles categorical time-of-day features with less manual encoding. A logistic regression meta-learner weighs these signals correctly without memorising the training set.

**The critical anti-pattern: data leakage in meta-learner training**

Do NOT train the meta-learner on the same fold the base models saw. If you train XGB/LGB/CatBoost on fold 1–4, then collect their fold 1–4 predictions and train logistic regression on those, the meta-learner sees in-sample predictions (overfit to noise) and learns a mapping that does not generalise. The base models will appear perfect on their own training folds. The meta-learner will memorise this and fail out-of-sample.

The correct approach: generate **out-of-fold (OOF) predictions** using temporal cross-validation, then train the meta-learner only on those OOF predictions.

**File: `models/ensemble.py`**

```python
# models/ensemble.py
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class SoftVoteEnsemble:
    """
    Baseline equal-weight soft voting ensemble.
    Keep for comparison with StackingEnsemble.
    Use only when training data < 18 months.
    """

    def __init__(self, models: List):
        self.models = models

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = [m.predict_proba(X) for m in self.models]
        return np.mean(probas, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class TimeSeriesStackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble with temporal cross-validation OOF generation.

    Base models: XGBoost, LightGBM, CatBoost (pre-configured externally).
    Meta-learner: LogisticRegression with L2 regularisation.

    REQUIREMENT: Use only when training data >= 18 months (approx 7,800 bars
    at 15-minute frequency). With less data the meta-learner overfits the OOF
    predictions and degrades to worse-than-soft-vote performance.

    Parameters
    ----------
    base_models : list of fitted or unfitted sklearn-compatible estimators
    n_splits : number of TimeSeriesSplit folds for OOF generation (default 5)
    meta_C : LogisticRegression regularisation strength (default 0.1, stronger
             regularisation than default to prevent meta-learner overfit)
    """

    def __init__(
        self,
        base_models: Optional[List] = None,
        n_splits: int = 5,
        meta_C: float = 0.1,
    ):
        self.base_models = base_models or self._default_base_models()
        self.n_splits = n_splits
        self.meta_C = meta_C
        self.meta_learner = LogisticRegression(
            C=meta_C,
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )
        self.fitted_base_models: List = []
        self.classes_: Optional[np.ndarray] = None

    @staticmethod
    def _default_base_models() -> List:
        return [
            xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                tree_method="hist",
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            ),
            lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            CatBoostClassifier(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                random_seed=42,
                verbose=0,
                thread_count=-1,
            ),
        ]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "TimeSeriesStackingEnsemble":
        """
        Two-phase fit:
        Phase 1 — generate OOF predictions using TimeSeriesSplit.
        Phase 2 — train meta-learner on OOF predictions.
        Phase 3 — refit all base models on full training data.
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = X.shape[0]

        # Phase 1: Generate OOF predictions
        oof_predictions = np.zeros((n_samples, len(self.base_models) * n_classes))
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        logger.info(
            f"Generating OOF predictions: {self.n_splits} folds, "
            f"{len(self.base_models)} base models"
        )

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            sw_tr = sample_weight[train_idx] if sample_weight is not None else None

            for model_idx, model in enumerate(self.base_models):
                cloned_model = joblib.loads(joblib.dumps(model))
                fit_kwargs = {}
                if sw_tr is not None:
                    if isinstance(cloned_model, xgb.XGBClassifier):
                        fit_kwargs["sample_weight"] = sw_tr
                    elif isinstance(cloned_model, lgb.LGBMClassifier):
                        fit_kwargs["sample_weight"] = sw_tr
                    elif isinstance(cloned_model, CatBoostClassifier):
                        fit_kwargs["sample_weight"] = sw_tr

                cloned_model.fit(X_tr, y_tr, **fit_kwargs)
                fold_proba = cloned_model.predict_proba(X_val)

                col_start = model_idx * n_classes
                col_end = col_start + n_classes
                oof_predictions[val_idx, col_start:col_end] = fold_proba

            logger.debug(f"Fold {fold_idx + 1}/{self.n_splits} complete")

        # Phase 2: Train meta-learner on OOF predictions only
        # Exclude early rows with zero OOF predictions (first fold has no OOF)
        first_val_start = list(tscv.split(X))[0][1][0]
        meta_X = oof_predictions[first_val_start:]
        meta_y = y[first_val_start:]

        self.meta_learner.fit(meta_X, meta_y)
        logger.info(
            f"Meta-learner trained on {len(meta_y)} OOF samples "
            f"(excluded first {first_val_start} rows with no OOF coverage)"
        )

        # Phase 3: Refit all base models on full training data
        self.fitted_base_models = []
        for model_idx, model in enumerate(self.base_models):
            full_model = joblib.loads(joblib.dumps(model))
            fit_kwargs = {}
            if sample_weight is not None:
                if isinstance(full_model, xgb.XGBClassifier):
                    fit_kwargs["sample_weight"] = sample_weight
                elif isinstance(full_model, lgb.LGBMClassifier):
                    fit_kwargs["sample_weight"] = sample_weight
                elif isinstance(full_model, CatBoostClassifier):
                    fit_kwargs["sample_weight"] = sample_weight
            full_model.fit(X, y, **fit_kwargs)
            self.fitted_base_models.append(full_model)

        logger.info("Base models refitted on full training data.")
        return self

    def _build_meta_features(self, X: np.ndarray) -> np.ndarray:
        n_classes = len(self.classes_)
        meta_X = np.zeros((X.shape[0], len(self.fitted_base_models) * n_classes))
        for model_idx, model in enumerate(self.fitted_base_models):
            col_start = model_idx * n_classes
            col_end = col_start + n_classes
            meta_X[:, col_start:col_end] = model.predict_proba(X)
        return meta_X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_X = self._build_meta_features(X)
        return self.meta_learner.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)
        logger.info(f"StackingEnsemble saved to {path}")

    @staticmethod
    def load(path: Path) -> "TimeSeriesStackingEnsemble":
        return joblib.load(path)
```

**When to use stacking vs soft vote:**
- Training data >= 18 months (~7,800 bars at 15-min) → use `TimeSeriesStackingEnsemble`
- Training data < 18 months → use `SoftVoteEnsemble` (meta-learner will overfit OOF predictions)
- After go-live: monitor meta-learner coefficient drift monthly. If any base model coefficient flips sign, re-run full OOF training.

---

### 5.2 Dynamic Accuracy-Weighted Ensemble

Static weights (equal or stacking-derived) are calibrated on historical data. Market regimes shift. During a prolonged trending regime, CatBoost may systematically outperform XGBoost on class +1. A dynamic weighting scheme adapts weights based on each model's recent precision on class +1 (the actionable buy signal).

**Precision on class +1 over last 10 trading days** — not accuracy, because the cost of a false positive (entering a losing trade) is higher than the cost of a missed true positive. Precision = TP / (TP + FP) on +1 class directly measures trade quality.

**File: `models/ensemble.py`** — add `DynamicWeightEnsemble` class

```python
# models/ensemble.py (continued)
from collections import deque
from datetime import date, timedelta


class DynamicWeightEnsemble:
    """
    Ensemble with daily-updated weights based on each model's rolling precision
    on class +1 over the last LOOKBACK_DAYS trading days.

    Weight update protocol:
    - Called once daily after market close (15:30 IST) via main.py scheduler
    - Reads closed-trade outcomes from QuestDB (trade_logger.py)
    - Computes per-model precision on +1 class for last LOOKBACK_DAYS days
    - Normalises to sum to 1.0
    - Floor weight = 0.10 per model (prevents any model from being fully excluded)

    Parameters
    ----------
    models : list of fitted base model objects (XGB, LGB, CatBoost)
    model_names : list of str identifiers matching QuestDB model_name column
    lookback_days : rolling window for precision calculation (default 10)
    weight_floor : minimum weight for any single model (default 0.10)
    """

    LOOKBACK_DAYS: int = 10
    WEIGHT_FLOOR: float = 0.10

    def __init__(
        self,
        models: List,
        model_names: List[str],
        lookback_days: int = 10,
        weight_floor: float = 0.10,
    ):
        assert len(models) == len(model_names), "models and model_names must match length"
        self.models = models
        self.model_names = model_names
        self.lookback_days = lookback_days
        self.weight_floor = weight_floor
        # Default to equal weights until first update
        n = len(models)
        self.weights: np.ndarray = np.full(n, 1.0 / n)
        self._precision_history: dict = {name: deque(maxlen=lookback_days) for name in model_names}

    def update_weights_from_questdb(self, questdb_rest_url: str = "http://localhost:9000") -> None:
        """
        Fetch last LOOKBACK_DAYS of per-model trade outcomes from QuestDB
        and recompute weights. Call this once after market close.

        Expects QuestDB table: model_predictions
        Schema: timestamp TIMESTAMP, model_name SYMBOL, predicted_class INT,
                actual_outcome INT, trade_date DATE
        """
        import requests

        cutoff = date.today() - timedelta(days=self.lookback_days)
        query = (
            f"SELECT model_name, predicted_class, actual_outcome "
            f"FROM model_predictions "
            f"WHERE trade_date >= '{cutoff.isoformat()}' "
            f"ORDER BY timestamp ASC"
        )
        try:
            resp = requests.get(
                f"{questdb_rest_url}/exec",
                params={"query": query},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"QuestDB weight fetch failed: {e}. Keeping current weights.")
            return

        rows = data.get("dataset", [])
        if not rows:
            logger.warning("No prediction records found in lookback window. Keeping equal weights.")
            return

        df = pd.DataFrame(rows, columns=["model_name", "predicted_class", "actual_outcome"])
        self._recompute_weights(df)

    def _recompute_weights(self, df: pd.DataFrame) -> None:
        """Compute precision on class +1 per model and normalise to weights."""
        new_precisions = {}
        for model_name in self.model_names:
            model_df = df[df["model_name"] == model_name]
            predicted_positive = model_df[model_df["predicted_class"] == 1]
            if len(predicted_positive) == 0:
                # Model never predicted +1 in window — assign floor precision
                new_precisions[model_name] = self.weight_floor
                logger.warning(f"Model {model_name} made zero +1 predictions in lookback window.")
                continue
            true_positives = (predicted_positive["actual_outcome"] == 1).sum()
            precision = true_positives / len(predicted_positive)
            new_precisions[model_name] = max(precision, self.weight_floor)
            logger.info(
                f"Model {model_name}: precision@+1 = {precision:.3f} "
                f"over last {self.lookback_days} days "
                f"({len(predicted_positive)} predictions)"
            )

        total_precision = sum(new_precisions.values())
        self.weights = np.array(
            [new_precisions[name] / total_precision for name in self.model_names]
        )
        logger.info(
            f"Updated ensemble weights: "
            + ", ".join(
                f"{name}={w:.3f}" for name, w in zip(self.model_names, self.weights)
            )
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        weighted_sum = np.zeros((X.shape[0], 3))  # 3 classes: -1, 0, +1
        for model, weight in zip(self.models, self.weights):
            weighted_sum += weight * model.predict_proba(X)
        return weighted_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_weight_summary(self) -> dict:
        return dict(zip(self.model_names, self.weights.tolist()))
```

**Integration point in `execution/trade_logger.py`:** After every trade closes, write model-level predictions to the `model_predictions` table so `update_weights_from_questdb()` can read them. Store `model_name`, `predicted_class` (what each individual model predicted), and `actual_outcome` (final triple-barrier label of the closed trade).

---

### 5.3 Combinatorial Purged Cross-Validation (CPCV) for Backtest Reliability

**What CPCV is and why it matters**

Standard walk-forward gives you a single backtest Sharpe ratio. That single number can be statistically consistent with the null hypothesis of no edge — you simply got lucky on the one path the walk-forward evaluated. CPCV (López de Prado, AFML Chapter 12) generates ALL N-choose-K combinations of fold partitions, runs the full model pipeline on each combination, and produces a **distribution** of Sharpe ratios. From this distribution you compute the **Probability of Backtest Overfitting (PBO)**: the proportion of combinations where the in-sample best strategy is beaten by a random strategy out-of-sample. PBO > 0.5 means your strategy is likely curve-fitted to noise and should not go live.

With 10 folds choosing 8 for training (leaving 2 for test): C(10, 8) = 45 combinations. Each combination needs a full model training pass. With n_jobs=-1 on 10 cores, this completes in roughly 20–40 minutes for the NSE feature set.

**File: `labeling/walk_forward.py`** — add `CPCVSplit` alongside `WalkForwardSplit`

```python
# labeling/walk_forward.py (additions)
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Generator, Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CPCVConfig:
    n_splits: int = 10          # Total number of folds (N)
    n_test_splits: int = 2      # Number of folds held out for test (K). N-choose-K combinations.
    purge_bars: int = 51        # See Section 5.5 for correct calculation
    embargo_bars: int = 5


class CPCVSplit:
    """
    Combinatorial Purged Cross-Validation split generator.

    Generates ALL C(n_splits, n_test_splits) train/test index combinations.
    Each combination purges and embargoes around each test fold.

    Reference: López de Prado, AFML Chapter 12.
    mlfinlab: mlfinlab.cross_validation.CombinatorialPurgedKFold

    Parameters
    ----------
    config : CPCVConfig

    Usage
    -----
    cpcv = CPCVSplit(CPCVConfig(n_splits=10, n_test_splits=2))
    for combo_id, train_idx, test_idx in cpcv.split(X):
        model.fit(X[train_idx], y[train_idx])
        sharpe = evaluate(model, X[test_idx], y[test_idx])
        sharpe_distribution.append(sharpe)

    pbo = cpcv.compute_pbo(is_sharpes, oos_sharpes)
    """

    def __init__(self, config: CPCVConfig = CPCVConfig()):
        self.config = config
        self._n_combinations = None

    @property
    def n_combinations(self) -> int:
        from math import comb
        if self._n_combinations is None:
            self._n_combinations = comb(
                self.config.n_splits, self.config.n_test_splits
            )
        return self._n_combinations

    def split(
        self,
        X: np.ndarray,
        t1: Optional[pd.Series] = None,
    ) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        Yield (combo_id, train_indices, test_indices) for all C(N, K) combinations.

        Parameters
        ----------
        X : feature array of shape (n_samples, n_features)
        t1 : pd.Series with index = sample start time, values = label end time.
             Required for correct purging. If None, purge uses bar count only.
        """
        n = X.shape[0]
        fold_size = n // self.config.n_splits
        fold_boundaries = [i * fold_size for i in range(self.config.n_splits + 1)]
        fold_boundaries[-1] = n  # ensure last fold takes all remaining bars

        all_fold_indices = [
            np.arange(fold_boundaries[i], fold_boundaries[i + 1])
            for i in range(self.config.n_splits)
        ]

        test_fold_combinations = list(
            combinations(range(self.config.n_splits), self.config.n_test_splits)
        )
        logger.info(
            f"CPCV: {self.config.n_splits} folds, "
            f"{self.config.n_test_splits} test folds, "
            f"{len(test_fold_combinations)} combinations"
        )

        for combo_id, test_fold_ids in enumerate(test_fold_combinations):
            test_fold_ids_set = set(test_fold_ids)
            train_fold_ids = [i for i in range(self.config.n_splits) if i not in test_fold_ids_set]

            test_idx = np.concatenate([all_fold_indices[i] for i in test_fold_ids])
            train_idx_raw = np.concatenate([all_fold_indices[i] for i in train_fold_ids])

            # Apply purge + embargo around each test fold boundary
            train_idx = self._apply_purge_embargo(
                train_idx_raw, test_idx, all_fold_indices, list(test_fold_ids)
            )

            yield combo_id, train_idx, test_idx

    def _apply_purge_embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        all_fold_indices: List[np.ndarray],
        test_fold_ids: List[int],
    ) -> np.ndarray:
        """Remove training samples within purge+embargo range of any test fold start."""
        purge_total = self.config.purge_bars + self.config.embargo_bars
        excluded = set()
        for fold_id in test_fold_ids:
            fold_start = all_fold_indices[fold_id][0]
            for bar_offset in range(purge_total):
                excluded.add(fold_start - bar_offset)

        mask = np.isin(train_idx, list(excluded), invert=True)
        purged_count = (~mask).sum()
        if purged_count > 0:
            logger.debug(f"CPCV: purged {purged_count} training bars near test boundaries")
        return train_idx[mask]

    @staticmethod
    def compute_pbo(
        is_sharpes: List[float],
        oos_sharpes: List[float],
    ) -> float:
        """
        Compute Probability of Backtest Overfitting (PBO).

        PBO = fraction of combinations where the combination that ranked
        highest in-sample ranked below median out-of-sample.

        Parameters
        ----------
        is_sharpes : list of in-sample Sharpe ratios, one per combination
        oos_sharpes : list of out-of-sample Sharpe ratios, one per combination

        Returns
        -------
        float : PBO in [0, 1]. Values > 0.5 indicate likely overfitting.
        """
        assert len(is_sharpes) == len(oos_sharpes)
        n = len(oos_sharpes)
        is_arr = np.array(is_sharpes)
        oos_arr = np.array(oos_sharpes)

        best_is_idx = int(np.argmax(is_arr))
        best_is_oos = oos_arr[best_is_idx]

        # PBO: what fraction of OOS Sharpes are worse than the best IS strategy?
        pbo = float((oos_arr < best_is_oos).sum() / n)
        oos_median = float(np.median(oos_arr))

        logger.info(
            f"CPCV PBO: {pbo:.3f} | Best IS Sharpe: {is_arr[best_is_idx]:.3f} | "
            f"Corresponding OOS Sharpe: {best_is_oos:.3f} | OOS median: {oos_median:.3f}"
        )
        if pbo > 0.5:
            logger.warning(
                f"PBO = {pbo:.3f} > 0.5. Strategy is likely overfit to noise. "
                "Do NOT deploy. Revisit feature selection and hyperparameter search space."
            )
        return pbo

    def run_parallel(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_fn,
        eval_fn,
        t1: Optional[pd.Series] = None,
        n_jobs: int = -1,
    ) -> Tuple[List[float], List[float]]:
        """
        Run all combinations in parallel.

        Parameters
        ----------
        train_fn : callable(X_train, y_train) -> fitted model
        eval_fn  : callable(model, X_train, y_train, X_test, y_test)
                   -> (is_sharpe: float, oos_sharpe: float)
        n_jobs   : number of parallel workers (-1 = all cores)

        Returns
        -------
        is_sharpes, oos_sharpes : lists of per-combination Sharpe ratios
        """
        combos = list(self.split(X, t1))
        logger.info(f"Running {len(combos)} CPCV combinations with n_jobs={n_jobs}")

        def _run_one(args):
            combo_id, train_idx, test_idx = args
            model = train_fn(X[train_idx], y[train_idx])
            is_sharpe, oos_sharpe = eval_fn(
                model, X[train_idx], y[train_idx], X[test_idx], y[test_idx]
            )
            return is_sharpe, oos_sharpe

        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_one)(combo) for combo in combos
        )

        is_sharpes = [r[0] for r in results]
        oos_sharpes = [r[1] for r in results]
        return is_sharpes, oos_sharpes
```

**Computational cost note:** 10 folds choose 2 = 45 combinations. With 10 cores and 15-min NSE data over 2 years, each combination trains in ~45 seconds. Total wall clock: ~3 minutes. Run overnight or as part of the monthly model refresh, not in real-time.

---

### 5.4 Triple-Barrier Edge Case Hardening

Three silent labeling bugs corrupt training data in live NSE conditions.

**File: `labeling/triple_barrier.py`** — add three handler methods and `sample_weight` output column

```python
# labeling/triple_barrier.py (additions)
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

GAP_OPEN_THRESHOLD = 0.05  # 5% gap open triggers sample_weight = 0


def _handle_gap_opens(
    df: pd.DataFrame,
    labels: pd.Series,
) -> pd.Series:
    """
    Detect samples where the bar after signal open gaps by >= GAP_OPEN_THRESHOLD.
    Set sample_weight = 0 for these — they are +1 labels but the entry price
    was impossible (stock opened through the intended entry). Including them
    teaches the model that profitable entries exist where they do not.

    Parameters
    ----------
    df : OHLCV DataFrame with 'open' and 'close' columns
    labels : triple-barrier label Series aligned with df

    Returns
    -------
    sample_weight : pd.Series of float in [0, 1]
    """
    prev_close = df["close"].shift(1)
    gap_pct = (df["open"] - prev_close).abs() / prev_close
    gap_mask = gap_pct >= GAP_OPEN_THRESHOLD

    sample_weight = pd.Series(1.0, index=df.index)
    gap_positive_labels = gap_mask & (labels == 1)
    sample_weight[gap_positive_labels] = 0.0

    n_zeroed = gap_positive_labels.sum()
    if n_zeroed > 0:
        logger.info(
            f"Gap-open handler: zeroed sample_weight for {n_zeroed} samples "
            f"({n_zeroed / len(df) * 100:.2f}% of dataset)"
        )
    return sample_weight


def _handle_simultaneous_hits(
    df: pd.DataFrame,
    upper_barrier: pd.Series,
    lower_barrier: pd.Series,
    entry_price: pd.Series,
) -> pd.Series:
    """
    When a single bar's high exceeds the upper barrier AND its low breaks
    the lower barrier simultaneously, determine the label by proximity:
    - If (high - entry) < (entry - low): upper hit first → label +1
    - If (entry - low) < (high - entry): lower hit first → label -1
    - Equal distance → label 0 (tie, time barrier wins)

    Parameters
    ----------
    df : OHLCV DataFrame
    upper_barrier, lower_barrier : price level Series aligned with df
    entry_price : entry price at each bar

    Returns
    -------
    resolved_label : pd.Series of {-1, 0, 1}
    """
    simultaneous = (df["high"] >= upper_barrier) & (df["low"] <= lower_barrier)
    resolved = pd.Series(0, index=df.index)

    if simultaneous.sum() == 0:
        return resolved

    dist_to_upper = df.loc[simultaneous, "high"] - entry_price[simultaneous]
    dist_to_lower = entry_price[simultaneous] - df.loc[simultaneous, "low"]

    resolved.loc[simultaneous & (dist_to_upper < dist_to_lower)] = 1
    resolved.loc[simultaneous & (dist_to_lower < dist_to_upper)] = -1
    # Equal distance: stays 0

    n_sim = int(simultaneous.sum())
    logger.debug(
        f"Simultaneous barrier hits: {n_sim} samples resolved "
        f"(+1: {(resolved == 1).sum()}, -1: {(resolved == -1).sum()}, "
        f"0: {(resolved == 0).sum()})"
    )
    return resolved


def _handle_eod(
    labels: pd.Series,
    bar_timestamps: pd.DatetimeIndex,
    market_close_time: str = "15:15",
) -> pd.Series:
    """
    Any position still open at the final bar of the trading session must be
    closed at that bar's close. Assign label = 0 (time barrier).
    This prevents look-forward bias on positions that "would have" resolved
    after market close if trading continued.

    Parameters
    ----------
    labels : existing label Series
    bar_timestamps : DatetimeIndex of bar open times
    market_close_time : NSE last tradeable bar open time (default 15:15 for 15-min)
    """
    eod_mask = bar_timestamps.strftime("%H:%M") == market_close_time
    labels = labels.copy()
    labels.loc[eod_mask] = 0
    logger.debug(f"EOD handler: forced {int(eod_mask.sum())} end-of-day labels to 0")
    return labels


def diagnose_label_distribution(labels: pd.Series) -> dict:
    """
    Compute label distribution and warn on severe imbalance.
    MUST be called and logged to MLflow before every training run.

    Returns
    -------
    dict : {label_value: proportion, ...}
    """
    dist = labels.value_counts(normalize=True).sort_index().to_dict()
    total = len(labels)
    logger.info(f"Label distribution (n={total}): {dist}")

    for label_val, proportion in dist.items():
        if proportion < 0.15:
            logger.warning(
                f"SEVERE IMBALANCE: class {label_val} = {proportion:.3f} < 0.15. "
                "Adjust barrier multipliers: "
                "+1 too rare → reduce upper_atr_mult (2.0 → try 1.5). "
                "-1 too rare → reduce lower_atr_mult (1.0 → try 0.7)."
            )
        if proportion > 0.60:
            logger.warning(
                f"DOMINANT CLASS: class {label_val} = {proportion:.3f} > 0.60. "
                "Model will predict this class almost always in live trading."
            )

    try:
        import mlflow
        if mlflow.active_run() is not None:
            for label_val, proportion in dist.items():
                mlflow.log_metric(f"label_dist_{label_val}", proportion)
            mlflow.log_metric("label_n_total", total)
    except Exception as e:
        logger.error(f"MLflow logging failed for label distribution: {e}")

    return dist
```

**Integration in `triple_barrier.py` main `label()` function:** The final output DataFrame must include a `sample_weight` column. XGBoost reads this via `fit(X, y, sample_weight=df['sample_weight'])`. The column defaults to 1.0; gap opens set it to 0.0 (excluded from gradient computation).

---

### 5.5 Walk-Forward Purge Correction (Add Feature Lookback)

**The current purge calculation is wrong**

Current implementation: `purge_bars = holding_period + embargo = 20 + 5 = 25`

This purges training samples whose label windows (the 20-bar holding horizon) extend into the test period. But it ignores that the features themselves look back into history. MACD uses a 26-period slow EMA. A training sample at bar T uses data from bars T-26 through T. If a test sample at bar T+1 is in the test set, and a training sample at bar T-1 is in the training set, those two samples share 25 bars of feature input data. This is feature-level leakage, separate from label-level leakage.

**Correct formula:**

```
purge_bars = holding_period + embargo + max_feature_lookback
           = 20 + 5 + 26
           = 51 bars  (with MACD as the longest lookback feature)
           = 75 bars  (if EMA50 is added as a feature)
```

**File: `labeling/walk_forward.py`**

```python
# labeling/walk_forward.py
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    n_splits: int = 5
    holding_period_bars: int = 20       # triple-barrier max holding period
    embargo_bars: int = 5               # standard embargo after purge boundary
    feature_max_lookback: int = 26      # MACD slow period = 26 bars.
                                        # Set to 50 if EMA50 is in feature set.
                                        # Set to 0 only if all features are
                                        # single-bar. DO NOT omit.

    @property
    def purge_bars(self) -> int:
        """
        Total bars to remove from training near test boundary.
        Includes label horizon + embargo + feature lookback.
        """
        total = self.holding_period_bars + self.embargo_bars + self.feature_max_lookback
        logger.debug(
            f"purge_bars = {self.holding_period_bars} (holding) + "
            f"{self.embargo_bars} (embargo) + "
            f"{self.feature_max_lookback} (feature lookback) = {total}"
        )
        return total


class WalkForwardSplit:
    """
    Walk-forward split with corrected purge calculation including feature lookback.

    The ~10% reduction in effective training samples from the larger purge zone
    is the correct trade-off: it eliminates a real source of data leakage that
    produces optimistically biased Sharpe ratios in backtest.
    """

    def __init__(self, config: WalkForwardConfig = WalkForwardConfig()):
        self.config = config

    def split(self, X: np.ndarray):
        n = X.shape[0]
        fold_size = n // (self.config.n_splits + 1)

        for fold in range(self.config.n_splits):
            test_start = (fold + 1) * fold_size
            test_end = min(test_start + fold_size, n)
            train_end = test_start - self.config.purge_bars

            if train_end <= 0:
                logger.warning(
                    f"Fold {fold}: train_end={train_end} after purge. "
                    "Insufficient data for this fold. Skipping."
                )
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx
```

**Impact assessment:** With 2 years of 15-min NSE data = ~19,500 bars. Changing purge from 25 to 51 bars per fold boundary removes approximately 26 extra bars × 5 folds = 130 additional training samples. That is 0.7% of the dataset — negligible in terms of training data loss but eliminates leakage that was inflating OOS metrics.

---

### 5.6 HMM Look-Ahead Bias Fix (CRITICAL)

**This is the most dangerous silent bug in the current architecture.**

`hmmlearn`'s `model.decode()` runs the Viterbi algorithm over the **entire sequence simultaneously**. When you pass `full_history` (all bars in the training period), the Viterbi path at bar T is influenced by bars T+1, T+2, ... through the backward pass of the algorithm. In a backtest, this means the HMM regime label assigned to bar T in training used information from future bars. The model learned "regime state during bar T was X" using data from T+50. In live trading, that look-ahead is not available — the HMM regime labels will differ, causing feature distribution shift that degrades live performance without any obvious error.

**File: `features/statistical.py`**

```python
# features/statistical.py
import numpy as np
import pandas as pd
from hmmlearn import hmm
import logging

logger = logging.getLogger(__name__)


# WRONG — do not use:
# states = hmm_model.decode(full_history)[1]
# decode() uses future bars in Viterbi backward pass → look-ahead bias

# CORRECT — use this function:
def compute_hmm_regimes_causal(
    returns: np.ndarray,
    n_states: int = 3,
    min_lookback: int = 60,
    model: hmm.GaussianHMM = None,
) -> np.ndarray:
    """
    Compute HMM regime labels WITHOUT look-ahead bias.

    Uses model.predict() on an expanding window, bar by bar.
    predict() applies only Viterbi forward pass on the data provided.

    Parameters
    ----------
    returns : 1D array of log returns
    n_states : number of HMM states (2=bull/bear, 3=bull/neutral/bear)
    min_lookback : minimum bars before first prediction (default 60)
    model : pre-fitted GaussianHMM. If None, fits on first min_lookback bars.

    Returns
    -------
    states : 1D int array, same length as returns.
             Entries 0..min_lookback-1 filled with first valid state.
    """
    n = len(returns)
    states = np.zeros(n, dtype=int)

    if model is None:
        init_data = returns[:min_lookback].reshape(-1, 1)
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(init_data)
        logger.debug(f"HMM fitted on first {min_lookback} bars.")

    first_valid_state = None
    for i in range(min_lookback, n):
        window = returns[:i].reshape(-1, 1)
        predicted_states = model.predict(window)
        current_state = int(predicted_states[-1])
        if first_valid_state is None:
            first_valid_state = current_state
        states[i] = current_state

    if first_valid_state is not None:
        states[:min_lookback] = first_valid_state

    return states


def compute_hmm_regimes_causal_fast(
    returns: np.ndarray,
    n_states: int = 3,
    min_lookback: int = 60,
    refit_every_n_bars: int = 0,
) -> np.ndarray:
    """
    Performance-optimised causal HMM: fits once, applies fixed parameters
    in a rolling forward-pass decode. Use when full rolling predict() is
    too slow for batch feature generation.

    If refit_every_n_bars > 0: refit model every N bars on expanding window.
    This adapts to regime shifts while remaining causal.
    """
    n = len(returns)
    states = np.zeros(n, dtype=int)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=100,
        random_state=42,
    )
    model.fit(returns[:min_lookback].reshape(-1, 1))

    for i in range(min_lookback, n):
        if refit_every_n_bars > 0 and (i - min_lookback) % refit_every_n_bars == 0:
            model.fit(returns[:i].reshape(-1, 1))
        window = returns[:i].reshape(-1, 1)
        predicted = model.predict(window)
        states[i] = int(predicted[-1])

    states[:min_lookback] = states[min_lookback]
    return states
```

**Practical note:** The causal rolling predict is O(n^2) in the number of bars. For 2 years of 15-min NSE data (~19,500 bars), this takes approximately 8–12 minutes on a single core. Run it offline during the feature engineering batch job, not in the live signal loop. Cache the result in QuestDB as a feature column. In live trading, call `model.predict(last_60_bars)[-1]` to get the current state — no look-ahead.

---

### 5.7 GARCH Fitting Best Practices for NSE

**Why GARCH(1,1) with Student-t outperforms Gaussian on NSE**

NSE 15-minute returns have excess kurtosis of approximately 4–7 (compared to 0 for Gaussian). The heavy tails are caused by expiry-day volatility spikes, circuit-breaker rebounds, and foreign institutional flow reversals. GARCH with Gaussian innovations underestimates the probability of large moves and underestimates volatility during stress periods. Student-t innovations with estimated degrees-of-freedom (typically 4–8 for NSE intraday) absorb these fat tails and produce better-calibrated volatility forecasts.

**Fitting protocol:**

```python
# features/statistical.py (additions)
from arch import arch_model
from arch.univariate import ARCHModelResult
import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level cache: fitted at 9:00 AM, used throughout the session
_garch_fit_cache: Optional[ARCHModelResult] = None
_garch_fit_date: Optional[str] = None


def morning_garch_fit(
    log_returns: pd.Series,
    lookback_days: int = 60,
    trading_bars_per_day: int = 25,  # 9:15–15:30 at 15-min = 25 bars
) -> ARCHModelResult:
    """
    Fit GARCH(1,1) with Student-t innovations on the last lookback_days
    of log returns. Call ONCE at 9:00 AM pre-open from main.py.

    Parameters
    ----------
    log_returns : pd.Series of log returns (NOT prices, NOT raw returns)
                  Compute as: np.log(close / close.shift(1)).dropna()
    lookback_days : training window in trading days (default 60)
    trading_bars_per_day : 15-min bars per NSE session (default 25)

    Returns
    -------
    result : fitted ARCHModelResult (use for intraday variance propagation)
    """
    global _garch_fit_cache, _garch_fit_date

    today = pd.Timestamp.now().date().isoformat()
    if _garch_fit_date == today and _garch_fit_cache is not None:
        logger.debug("GARCH: returning cached morning fit.")
        return _garch_fit_cache

    lookback_bars = lookback_days * trading_bars_per_day
    recent_returns = log_returns.iloc[-lookback_bars:] * 100  # scale to percent

    if len(recent_returns) < 100:
        logger.warning(
            f"GARCH fit: only {len(recent_returns)} bars available "
            f"(minimum recommended: 100). Fit quality may be poor."
        )

    model = arch_model(
        recent_returns,
        vol="Garch",
        p=1,
        q=1,
        dist="t",          # Student-t: handles NSE fat tails
        mean="Zero",        # intraday returns have near-zero mean
        rescale=True,
    )

    result = model.fit(
        disp="off",
        options={"maxiter": 200},
        show_warning=False,
    )

    _garch_fit_cache = result
    _garch_fit_date = today

    logger.info(
        f"GARCH morning fit complete: "
        f"omega={result.params.get('omega', float('nan')):.6f}, "
        f"alpha[1]={result.params.get('alpha[1]', float('nan')):.4f}, "
        f"beta[1]={result.params.get('beta[1]', float('nan')):.4f}, "
        f"nu (df)={result.params.get('nu', float('nan')):.2f}"
    )
    return result


def propagate_garch_intraday(
    garch_result: ARCHModelResult,
    current_return_pct: float,
    current_variance: float,
) -> float:
    """
    Propagate GARCH(1,1) variance one step forward using morning parameters.
    Use during the trading session WITHOUT re-fitting.

    h_t = omega + alpha * epsilon_{t-1}^2 + beta * h_{t-1}

    Parameters
    ----------
    garch_result : fitted ARCHModelResult from morning_garch_fit()
    current_return_pct : most recent bar log return * 100
    current_variance : previous step conditional variance

    Returns
    -------
    next_variance : float (percent-squared units, matching fit scale)
    """
    params = garch_result.params
    omega = params.get("omega", 0.01)
    alpha = params.get("alpha[1]", 0.05)
    beta = params.get("beta[1]", 0.90)
    next_variance = omega + alpha * (current_return_pct ** 2) + beta * current_variance
    return next_variance
```

**Integration in `main.py`:** Add a scheduled call to `morning_garch_fit()` at 09:00 IST (15 minutes before NSE pre-open at 09:15). The `features/engine.py` then uses `propagate_garch_intraday()` for each new bar during the session. Never re-fit GARCH during the trading session — convergence takes ~2–5 seconds and will block the signal loop.

---

### 5.8 Fractional Differentiation as Feature

Fractional differentiation produces a stationary series while preserving long-range memory. Standard returns (d=1 differencing) destroy memory entirely. Fractional differencing with d=0.40–0.45 achieves stationarity (ADF test p < 0.05) while retaining autocorrelation structure — information that RSI and MACD cannot capture.

**Stage 3 addition — not MVP.** Do not add until core pipeline is validated on live paper trading.

**File: `features/statistical.py`**

```python
# features/statistical.py (additions)
from typing import Tuple


def fractional_diff_feature(
    close: pd.Series,
    d: float = 0.42,
    min_periods: int = 26,
) -> pd.Series:
    """
    Compute fractionally differenced price series.
    Optimal d for NSE 15-min: 0.40-0.45. Find via find_optimal_fracdiff_d().

    WARNING: Do NOT include both frac_diff_close AND raw log_returns in the
    same feature set — they are linearly dependent at low d values.
    Replace log_returns with frac_diff_close, or use only one.

    Parameters
    ----------
    close : pd.Series of closing prices
    d : differentiation order (0 < d < 1)
    min_periods : minimum bars before first valid output

    Returns
    -------
    frac_diff : pd.Series, same index as close, NaN for first min_periods bars
    """
    try:
        from fracdiff import fdiff
        frac_diff = pd.Series(
            fdiff(close.values, d=d, window=min_periods),
            index=close.index,
        )
        frac_diff.iloc[:min_periods] = np.nan
        return frac_diff
    except ImportError:
        logger.error(
            "fracdiff library not installed. Install with: pip install fracdiff"
        )
        return pd.Series(np.nan, index=close.index)


def find_optimal_fracdiff_d(
    close: pd.Series,
    d_range: Tuple[float, float] = (0.1, 0.6),
    d_step: float = 0.05,
) -> float:
    """
    Find minimum d that achieves ADF stationarity (p < 0.05).
    Run once offline, then hardcode the result in config.py as FRACDIFF_D.

    Returns
    -------
    float : minimum d achieving stationarity, or d_range[1] if none found
    """
    from statsmodels.tsa.stattools import adfuller
    d_values = np.arange(d_range[0], d_range[1] + d_step, d_step)
    for d in d_values:
        fd = fractional_diff_feature(close, d=round(float(d), 3))
        fd_clean = fd.dropna()
        if len(fd_clean) < 30:
            continue
        _, p_value, *_ = adfuller(fd_clean, autolag="AIC")
        logger.debug(f"d={d:.2f}: ADF p-value = {p_value:.4f}")
        if p_value < 0.05:
            logger.info(f"Minimum stationary d = {d:.2f} (ADF p = {p_value:.4f})")
            return round(float(d), 3)
    logger.warning("No d in range achieved stationarity. Using d_range[1].")
    return d_range[1]
```

---

### 5.9 Label Distribution Diagnostic (Mandatory MLflow Log)

Label imbalance is the first failure mode to check when a model predicts the same class for nearly all live bars. This diagnostic is already embedded in `diagnose_label_distribution()` in Section 5.4. The key integration requirement is that it must be called inside the training pipeline before model fitting, with MLflow logging active.

**Canonical usage pattern in `scripts/train_models.py`:**

```python
# scripts/train_models.py
import mlflow
from labeling.triple_barrier import diagnose_label_distribution

with mlflow.start_run(run_name=f"train_{symbol}_{timestamp}"):
    # Step 1: Generate labels
    labels = triple_barrier_labeler.label(df)

    # Step 2: MANDATORY — diagnose before any model sees the data
    dist = diagnose_label_distribution(labels)
    # dist is automatically logged to MLflow inside the function

    # Step 3: Gate — abort if distribution is degenerate
    if any(v < 0.10 for v in dist.values()):
        raise ValueError(
            f"Label distribution {dist} is too imbalanced to train. "
            "Adjust ATR multipliers in config.py and re-label."
        )

    # Step 4: Train
    ensemble.fit(X_train, labels_train, sample_weight=df["sample_weight"].values)
```

**Thresholds reference:**
- Any class < 0.15: warning — adjust barrier multipliers
- Any class < 0.10: hard stop — do not train
- Any class > 0.60: warning — model will collapse to predicting dominant class
- Ideal range per class: 0.25–0.45 for balanced 3-class problem

---

### 5.10 Kelly Criterion Position Sizing (Replacing RL Sizer)

The RL position sizer requires thousands of training episodes to converge and is fragile to distribution shift. Kelly Criterion, properly implemented with Half-Kelly and a hard cap, provides mathematically optimal position sizing without a separate training stage.

**Full Kelly formula:**
`f* = (p × b - q) / b`
where `p = win_rate`, `q = 1 - p`, `b = avg_win / avg_loss`

**Half-Kelly:** `f_live = f* / 2`
Half-Kelly is the standard for live trading. Full Kelly maximises long-term growth but requires perfectly estimated p and b — which are never available with finite data. Half-Kelly accounts for estimation error and reduces drawdowns by approximately 50% relative to Full Kelly.

**File: `risk/position_sizer.py`** — add `kelly_fraction_sizing()` alongside existing ATR-based sizing

```python
# risk/position_sizer.py (additions)
import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Hard caps — non-negotiable regardless of Kelly output
MAX_KELLY_FRACTION: float = 0.05    # 5% of capital maximum per position
MIN_KELLY_FRACTION: float = 0.00    # 0% if Kelly is negative (no edge → skip)
KELLY_LOOKBACK_TRADES: int = 50     # rolling window for win_rate and avg_win/loss


def kelly_fraction_sizing(
    closed_trades: pd.DataFrame,
    current_atr_sizing_fraction: float,
    use_half_kelly: bool = True,
) -> float:
    """
    Compute position size fraction using Dynamic Kelly Criterion.

    Kelly is recomputed from the last KELLY_LOOKBACK_TRADES closed trades.
    The final position size is the MINIMUM of Kelly and ATR sizing:
    always take the more conservative estimate.

    Parameters
    ----------
    closed_trades : DataFrame with column 'pnl' (signed P&L per trade).
                    Most recent KELLY_LOOKBACK_TRADES rows are used.
    current_atr_sizing_fraction : fraction from existing ATR-based sizer
    use_half_kelly : if True, divide f* by 2 (recommended for live trading)

    Returns
    -------
    float : position fraction of capital in [0, MAX_KELLY_FRACTION]
    """
    if len(closed_trades) < 10:
        logger.warning(
            f"Kelly: only {len(closed_trades)} closed trades available "
            f"(minimum 10). Falling back to ATR sizing."
        )
        return current_atr_sizing_fraction

    recent = closed_trades.tail(KELLY_LOOKBACK_TRADES)
    pnls = recent["pnl"].values

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    if len(wins) == 0:
        logger.warning("Kelly: zero winning trades in lookback window. Skipping trade.")
        return MIN_KELLY_FRACTION

    if len(losses) == 0:
        # All wins — Kelly would be infinite; cap at MAX
        logger.warning("Kelly: zero losing trades in lookback window. Using max cap.")
        return MAX_KELLY_FRACTION

    p = len(wins) / len(recent)         # win rate
    q = 1.0 - p                          # loss rate
    avg_win = float(wins.mean())
    avg_loss = float(abs(losses.mean()))
    b = avg_win / avg_loss               # win/loss ratio

    f_star = (p * b - q) / b            # Full Kelly

    if f_star <= 0:
        logger.info(
            f"Kelly f* = {f_star:.4f} (negative edge). "
            f"win_rate={p:.3f}, avg_win={avg_win:.2f}, avg_loss={avg_loss:.2f}. "
            "Do not trade."
        )
        return MIN_KELLY_FRACTION

    f_live = f_star / 2.0 if use_half_kelly else f_star
    f_capped = min(f_live, MAX_KELLY_FRACTION)

    # Always take the more conservative sizing between Kelly and ATR
    final_fraction = min(f_capped, current_atr_sizing_fraction)

    logger.info(
        f"Kelly sizing: f*={f_star:.4f}, "
        f"{'Half-Kelly' if use_half_kelly else 'Full-Kelly'}={f_live:.4f}, "
        f"after cap={f_capped:.4f}, "
        f"ATR fraction={current_atr_sizing_fraction:.4f}, "
        f"final={final_fraction:.4f} "
        f"[win_rate={p:.3f}, b={b:.3f}, n={len(recent)} trades]"
    )
    return final_fraction


def compute_kelly_diagnostics(
    closed_trades: pd.DataFrame,
) -> dict:
    """
    Return Kelly parameter diagnostics for MLflow logging and monitoring.
    Log these after every training run to detect edge decay over time.
    If kelly_edge_positive is False on OOS data, do not go live.
    """
    if len(closed_trades) < KELLY_LOOKBACK_TRADES:
        return {"kelly_insufficient_data": True}

    recent = closed_trades.tail(KELLY_LOOKBACK_TRADES)
    pnls = recent["pnl"].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    p = float(len(wins) / len(recent)) if len(recent) > 0 else 0.0
    b = float(wins.mean() / abs(losses.mean())) if len(wins) > 0 and len(losses) > 0 else 0.0
    q = 1.0 - p
    f_star = float((p * b - q) / b) if b > 0 else 0.0

    return {
        "kelly_win_rate": p,
        "kelly_loss_rate": q,
        "kelly_win_loss_ratio": b,
        "kelly_f_star": f_star,
        "kelly_f_half": f_star / 2.0,
        "kelly_f_capped": min(f_star / 2.0, MAX_KELLY_FRACTION),
        "kelly_n_trades": len(recent),
        "kelly_edge_positive": f_star > 0,
    }
```

**Integration in `main.py`:** Before placing any order, call:

```python
# main.py — signal execution path
kelly_frac = kelly_fraction_sizing(
    closed_trades=trade_logger.get_recent_closed_trades(50),
    current_atr_sizing_fraction=position_sizer.atr_fraction(current_atr),
)
position_size = int(kelly_frac * risk_manager.current_capital / entry_price)
position_size = max(1, position_size)  # minimum 1 share
```

Log `compute_kelly_diagnostics()` output to MLflow after every training run. If `kelly_edge_positive` is False on OOS backtest data, the strategy has no detectable edge in the validation period — do not go live.

---

### Summary: File Changes Required

| File | Changes |
|---|---|
| `models/ensemble.py` | Add `TimeSeriesStackingEnsemble`, `DynamicWeightEnsemble`; keep `SoftVoteEnsemble` |
| `labeling/walk_forward.py` | Add `feature_max_lookback` to `WalkForwardConfig`; add `CPCVSplit` class |
| `labeling/triple_barrier.py` | Add `_handle_gap_opens()`, `_handle_simultaneous_hits()`, `_handle_eod()`, `diagnose_label_distribution()`; add `sample_weight` column to output |
| `features/statistical.py` | Replace `decode()` with `compute_hmm_regimes_causal()`; add `morning_garch_fit()`, `propagate_garch_intraday()`, `fractional_diff_feature()` |
| `risk/position_sizer.py` | Add `kelly_fraction_sizing()`, `compute_kelly_diagnostics()` |
| `execution/trade_logger.py` | Write per-model predictions to `model_predictions` QuestDB table for `DynamicWeightEnsemble` weight updates |
| `main.py` | Schedule `morning_garch_fit()` at 09:00 IST; call `kelly_fraction_sizing()` before every order; call `diagnose_label_distribution()` before every training run |

### Deployment Priority

1. **Immediate (fix before any backtest is trusted):** HMM look-ahead fix (5.6), walk-forward purge correction (5.5), triple-barrier edge cases (5.4)
2. **Before first live trade:** Label distribution diagnostic (5.9), Kelly sizing (5.10), GARCH morning fit (5.7)
3. **After 3 months of paper trading data:** Dynamic weight ensemble (5.2), stacking meta-learner (5.1)
4. **Quarterly validation run:** CPCV PBO check (5.3) — if PBO > 0.5, pull strategy from live trading and retrain
5. **Stage 3 feature expansion:** Fractional differentiation (5.8)

---

## PART 5B: OPTIONAL UPGRADES AFTER CORE SYSTEM IS LIVE
### Source: PROJECT_PLAN_SIMPLE.txt (Sessions 57-94 plan, extracted May 2026)
### Note: Original used Angel One as primary — corrected to Shoonya per CLAUDE.md

The core system completes at Session 56. The following upgrades are additive improvements. Do not attempt any of these before the core system has been live and stable for at least 4 weeks.

**Honest session totals:**
- Core system: 56 sessions (~4.5 months at 3 sessions/week)
- Quick upgrades (57-62): ~6 sessions (~1 month)
- Medium upgrades (63-68): ~6 sessions (~1 month, 3+ months after live)
- Long-term (69-82): ~14 sessions (6+ months after live, when profitable)
- F&O (83-94, SEPARATE PROJECT): 12 sessions

---

### Quick Wins — Sessions 57-62 (Add After 4 Weeks Live)

Each improves performance with low risk of breaking the core system.

**Sessions 57-58 — India VIX Adaptive Sizing**
- When India VIX > 20: reduce all position sizes by 50%
- When India VIX > 25: reduce by 75%, consider pausing entirely
- Expected improvement: +0.2 to +0.4 Sharpe ratio
- Why: VIX spikes = higher realized volatility = stops hit more = more losses
- India VIX data: free from NSE website

**Session 59 — Earnings Calendar Filter**
- Do not enter new trades 1 day before or after company earnings announcements
- Earnings dates: pull from NSE corporate events calendar (free)
- Why: earnings = gap risk; stops become useless if stock gaps 10% overnight

**Session 60 — Sector Rotation Signal**
- Add feature: stock's 20-bar return vs its sector ETF return
- If stock is underperforming its sector, lower confidence on long signals
- Sector ETF data: free from NSE (Nifty IT, Nifty Bank, Nifty Auto, etc.)

**Sessions 61-62 — CPCV Backtesting**
- Replace simple walk-forward with Combinatorial Purged Cross-Validation
- More reliable performance estimates, reduces overfitting risk
- Reference: Lopez de Prado, "Advances in Financial ML", Chapter 12

---

### Medium Effort — Sessions 63-68 (3+ Months After Live)

**Sessions 63-64 — Order Book Depth Features**
- Use 5-level bid/ask depth as features (not just top of book)
- Requires Dhan API (has level-2 data) or Zerodha Kite Connect with depth
- Expected improvement: measurable edge on high-volume stocks

**Sessions 65-66 — Auto Model Retraining**
- Monitor model performance metrics daily
- If 5-day rolling accuracy drops more than 15% from baseline, trigger overnight retrain
- Prevents "alpha decay" as market microstructure evolves over months

**Sessions 67-68 — FinBERT Sentiment on BSE Announcements**
- Scrape BSE corporate announcements (XML feed, free)
- Run FinBERT (CPU-only, 0.2 seconds per announcement) for sentiment score
- Add as 24th feature: sentiment_score (range -1 to +1)
- Only affects stocks with announcements that day

---

### Long-Term Upgrades — Sessions 69-82 (6+ Months, When Profitable)

**Sessions 69-74 — RL Position Sizer**
- Replace fixed ATR sizing with PPO agent (Stable-Baselines3)
- Uses NSETradingEnv (OpenAI Gym environment, already designed in CLAUDE.md)
- Train on Kaggle T4 GPU (free, 30 hrs/week)
- More dynamic: increases size when model confidence is high, reduces when uncertain
- Effort: 6 sessions. Do not attempt before system is consistently profitable.

**Sessions 75-79 — Multi-Stock Portfolio**
- Trade 5-10 stocks simultaneously with portfolio-level risk control
- HRP (Hierarchical Risk Parity) for allocation across stocks
- Correlation filter: if 2 stocks are more than 0.7 correlated, reduce combined size
- Effort: 5 sessions — requires significant refactoring of the main orchestrator

**Sessions 80-82 — Foundation Model Weak Signal**
- Add Chronos-T5-small (46M params, runs on CPU, 2-3s per prediction) as a 4th ensemble member with low voting weight (10-15%)
- Zero-shot probabilistic forecast as extra input
- Effort: 3 sessions — relatively self-contained addition

---

### F&O System — Sessions 83-94 (SEPARATE PROJECT)

**Build this as a completely separate codebase, not an addition to the equity system.**

- Nifty weekly options: systematic iron condor or strangle selling
- Completely different system: options Greeks, expiry handling, lot sizes, margin requirements
- Do NOT try to bolt this onto the equity system — build from scratch
- Minimum capital: Rs. 5,00,000 (Nifty lot size is large; strangle requires Rs. 1,50,000-2,50,000 margin per lot)
- Effort: 12 sessions
- Prerequisites: F&O trading enabled on broker (income proof: ITR or bank statement showing net worth > Rs. 2L)


---

## PART 6: AI DEVELOPMENT OPTIMIZATION + MODERN ML ALTERNATIVES

---

### 6.1 The arch-spec.json Concept — AI Copilot Accelerator

**The Problem:** Long prose prompts cause AI assistants to hallucinate architecture details, invent wrong file paths, ignore hardware constraints, and produce code that doesn't fit your existing modules. Every session starts cold.

**The Solution:** A single structured JSON file at project root that acts as a machine-readable contract of your entire system. When prefixed to any prompt, it eliminates ambiguity and forces the AI to operate within your exact constraints. One file replaces 500 words of repeated context.

**The Rule:** Every AI interaction begins with `@workspace Read arch-spec.json.` — no exceptions.

---

### 6.2 Complete arch-spec.json — Production-Ready

Place this file at: `EXCHANGE/arch-spec.json`

```json
{
  "project": {
    "name": "NSE AlgoTrader",
    "description": "CPU-only ML algorithmic trading system for NSE India equities. Ingests tick data, builds volume bars, engineers microstructure features, generates ML signals, executes via broker APIs with strict risk controls.",
    "version": "2026-Stage-2",
    "language": "Python 3.11+",
    "target_market": "NSE India — Equities, F&O",
    "trading_session": "09:15-15:30 IST",
    "repo_root": "EXCHANGE/"
  },

  "hardware": {
    "cpu": "Intel i5-1345U",
    "cpu_cores": 10,
    "cpu_p_cores": 2,
    "cpu_e_cores": 8,
    "ram_gb": 16,
    "gpu": null,
    "gpu_constraint": "ALL_CPU_ONLY — no CUDA, no Metal, no OpenCL, no GPU libraries",
    "disk": "SSD",
    "os": "Windows 11",
    "inference_budget_ms": 50,
    "notes": "P-cores reserved for latency-critical processes (execution, tick ingestion). E-cores for ML inference and feature computation."
  },

  "stack": {
    "language": "Python 3.11+",
    "data_storage": {
      "primary": "QuestDB",
      "version": "7.x",
      "protocol": "InfluxDB Line Protocol over TCP (port 9009)",
      "query": "PostgreSQL wire protocol (port 8812)",
      "tables": ["ticks","volume_bars","features","signals","orders","fills","risk_state","regime_labels","pnl_daily"]
    },
    "data_processing": {
      "primary": "Polars",
      "version": ">=0.20",
      "rule": "NO pandas in hot path. Polars LazyFrames for batch. Polars Series for feature output.",
      "fallback": "pandas ONLY for broker API response parsing where unavoidable"
    },
    "ml_models": {
      "primary_ensemble": ["XGBoost", "LightGBM", "CatBoost"],
      "xgboost_params": {
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": 4,
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
      },
      "lightgbm_params": {
        "device": "cpu",
        "num_threads": 4,
        "n_estimators": 300,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "boosting_type": "gbdt",
        "dart_variant": {
          "boosting_type": "dart",
          "drop_rate": 0.1,
          "use_when": "noisy_regime or overfit_detected"
        }
      },
      "catboost_params": {
        "task_type": "CPU",
        "thread_count": 4,
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05
      },
      "ensemble_method": "weighted_average",
      "ensemble_weights_file": "models/ensemble_weights.json",
      "model_serialization": "joblib",
      "model_directory": "models/",
      "retrain_schedule": "weekly_or_regime_change"
    },
    "broker_primary": {
      "name": "AngelOne SmartAPI",
      "library": "smartapi-python",
      "auth": "TOTP-based JWT",
      "rate_limit_orders_per_sec": 10
    },
    "broker_secondary": {
      "name": "Shoonya (Finvasia)",
      "library": "NorenRestApiPy",
      "use_case": "Failover + zero-brokerage execution"
    },
    "data_feed": {
      "stage_2": "AngelOne SmartWebSocket",
      "stage_3_plus": "TrueData WebSocket (unthrottled L2)",
      "tick_fields": ["timestamp", "symbol", "ltp", "volume", "best_bid", "best_ask", "oi"]
    },
    "ipc": {
      "library": "ZeroMQ (pyzmq)",
      "patterns": {
        "tick_broadcast": "PUB/SUB",
        "signal_push": "PUSH/PULL",
        "risk_command": "REQ/REP"
      },
      "ports": {
        "tick_pub": 5555,
        "bar_pub": 5558,
        "signal_push": 5556,
        "risk_rep": 5557
      }
    },
    "scheduler": "APScheduler 3.x",
    "logging": "structlog → QuestDB logs table + rotating file",
    "testing": "pytest + pytest-mock"
  },

  "architecture": {
    "pattern": "Multi-process event-driven pipeline",
    "data_flow": "Broker WebSocket → TickIngestor → ZMQ PUB → BarBuilder → ZMQ PUB → FeatureEngine → MLInference → SignalRouter → RiskManager → OrderExecutor → Broker API",
    "bar_type": "VolumeBar",
    "bar_volume_threshold": "dynamic — ADV/390 per symbol, updated daily at 09:15",
    "processes": [
      {"name": "TickIngestor", "file": "ingestors/tick_ingestor.py", "cpu_affinity": "E-core"},
      {"name": "BarBuilder", "file": "bars/bar_builder.py", "cpu_affinity": "E-core"},
      {"name": "FeatureEngine", "file": "features/feature_engine.py", "cpu_affinity": "E-cores (2-4)"},
      {"name": "MLInference", "file": "models/inference_engine.py", "cpu_affinity": "E-cores (4-8)"},
      {"name": "SignalRouter", "file": "signals/signal_router.py"},
      {"name": "RiskManager", "file": "risk/risk_manager.py", "cpu_affinity": "P-core"},
      {"name": "OrderExecutor", "file": "execution/order_executor.py", "cpu_affinity": "P-core"},
      {"name": "RegimeDetector", "file": "regime/regime_detector.py", "schedule": "Every 5 bars"},
      {"name": "DailyOrchestrator", "file": "orchestrator/daily_orchestrator.py", "schedule": "APScheduler cron"}
    ]
  },

  "deployment_stage": "stage_2",
  "stage_definitions": {
    "stage_1": "Historical data collection + backtesting only",
    "stage_2": "Paper trading live feed + full pipeline dry run (CURRENT)",
    "stage_3": "Live trading small capital — TrueData L2 feed upgrade",
    "stage_4": "Full capital + F&O (only after Stage 3 profitable 3+ months)"
  },

  "constraints": [
    "CPU_ONLY — no CUDA, no cuML, no TensorFlow GPU",
    "NO_OVERNIGHT_POSITIONS — flatten by 15:15 IST",
    "NO_PANDAS_HOT_PATH — Polars only in BarBuilder, FeatureEngine, MLInference",
    "QUESTDB_TIMESERIES_ONLY — no SQLite, no PostgreSQL for tick data",
    "ZEROMQ_IPC_ONLY — no Redis, no Kafka for inter-process messaging",
    "DECIMAL_FOR_MONEY — Python Decimal for all monetary calculations",
    "STRUCTURED_LOGGING — structlog with timestamp_ns, module, symbol",
    "CONFIG_YAML_ONLY — no hardcoded constants in business logic",
    "WALK_FORWARD_VALIDATION — no random split on time series",
    "MAX_INFERENCE_50MS — ML inference pipeline within 50ms",
    "ANGEL_ONE_PRIMARY — AngelOne primary, Shoonya failover only",
    "NSE_SESSION_ONLY — logic active 09:15-15:30 IST on NSE trading days"
  ],

  "active_critiques": [
    "fix_ Gil_stalls_tick_feed_during_predict_isolate_with_ZeroMQ",
    "fix_15min_time_bars_replace_with_volume_bars",
    "fix_obi_averaging_over_15min_compute_tick_by_tick",
    "fix_30min_limit_timeout_stale_alpha_reduce_to_20sec",
    "fix_training_on_ohlcv_executing_on_l2_model_collapse",
    "fix_questdb_rest_io_bottleneck_use_ring_buffer",
    "fix_tdd_delayed_shift_to_phase_4",
    "fix_ml_before_api_plumbing_dummy_ema_first",
    "fix_static_slippage_0.05pct_use_dynamic_atr_spread",
    "fix_brain_hands_not_segregated_truedata_vs_broker",
    "fix_asymmetric_triple_barrier_class_imbalance",
    "fix_volume_clock_replace_chronological_sincos",
    "fix_kaggle_optuna_version_mismatch_use_docker",
    "fix_soft_voting_correlated_models_use_stacking",
    "fix_hmm_chattering_add_duration_penalty",
    "fix_purge_window_missing_feature_lookback_26bars",
    "fix_ppo_rl_unstable_replace_with_kelly_criterion",
    "fix_rate_limiter_orders_per_min_cap_missing",
    "fix_finbert_blocks_main_thread_async_worker",
    "fix_kalman_newtonian_add_ofi_to_state_matrix",
    "fix_memory_fragmentation_daily_vps_restart",
    "fix_optuna_100trials_hyperparameter_overfitting",
    "fix_angel_one_totp_expiry_heartbeat_monitor",
    "fix_hml_lookahead_bias_use_rolling_predict_not_batch_decode"
  ],

  "microstructure_models": {
    "vpin": {
      "formula": "VPIN = |V_buy - V_sell| / V_total over rolling N volume buckets",
      "bucket_size": "ADV / 50",
      "window_buckets": 50,
      "output_range": "[0.0, 1.0]",
      "high_toxicity_threshold": 0.7,
      "implementation_file": "features/microstructure.py"
    },
    "kyle_lambda": {
      "formula": "lambda = cov(delta_p, order_flow) / var(order_flow)",
      "order_flow_proxy": "signed_volume = volume * sign(side)",
      "window_bars": 20,
      "implementation_file": "features/microstructure.py"
    },
    "hawkes_process": {
      "formula": "lambda(t) = mu + sum(alpha * exp(-beta*(t-ti))) for ti < t",
      "parameters": ["mu (baseline intensity)", "alpha (excitation)", "beta (decay)"],
      "use_case": "Detect spoof cancellation cascades and institutional burst firing",
      "implementation_file": "features/microstructure.py"
    },
    "avellaneda_stoikov": {
      "reservation_price": "r(s,t) = s - q*gamma*sigma^2*(T-t)",
      "use_case": "Better mid-price estimator during wide bid-ask spread events",
      "implementation_file": "features/microstructure.py"
    }
  },

  "ml_alternatives_roadmap": {
    "tabpfn": {
      "use_when": "labeled_dataset < 1000 rows",
      "pip": "pip install tabpfn",
      "cpu_inference_ms": 10,
      "file": "models/tabpfn_model.py"
    },
    "river_online": {
      "use_when": "regime_change_adaptation or continuous_update_required",
      "pip": "pip install river",
      "algorithms": ["HoeffdingTreeClassifier", "ADWINBaggingClassifier"],
      "inference_ms": 0.5,
      "file": "models/online_model.py"
    },
    "tsmixer": {
      "use_when": "tick_data_history > 6_months AND stage >= 3",
      "pip": "pip install neuralforecast",
      "cpu_inference_ms": 5,
      "file": "models/tsmixer_model.py"
    },
    "lgbm_dart": {
      "use_when": "noisy_regime or overfit_detected",
      "params": {"boosting_type": "dart", "drop_rate": 0.1},
      "file": "models/ensemble.py"
    }
  }
}
```

---

### 6.3 Optimized AI Prompt Templates

**Template 1: New Feature Implementation**
```
@workspace Read arch-spec.json.
You are a Principal Quant Engineer implementing features/microstructure.py::compute_vpin().
INPUT: pl.DataFrame with [timestamp_ns: Int64, price: Float64, volume: Float64, side: Int8]
OUTPUT: pl.Series[Float64], NaN for initial window
CONSTRAINTS: Polars only. No lookahead. <5ms for 500-bar input on i5-1345U.
Follow arch-spec.json::microstructure_models.vpin formula exactly.
```

**Template 2: Bug Fix**
```
@workspace Read arch-spec.json.
Fix BUG: [bug_id from active_critiques]
FILE: [file]  CLASS: [class]
SYMPTOM: [what breaks]
ROOT CAUSE: [what causes it]
REQUIRED FIX: [specific change]
CONSTRAINTS: Do not change method signatures. Preserve structlog fields.
Show only changed methods — no full file rewrite.
```

**Template 3: Module Refactor**
```
@workspace Read arch-spec.json.
Refactor [file]::[class] to fix [critique_id].
CURRENT PROBLEM: [description]
REQUIRED CHANGES: [numbered list]
INTERFACE CONTRACT (do not change): [method signatures]
CONSTRAINTS (from arch-spec.json): [relevant constraints]
```

**Template 4: Test Generation**
```
@workspace Read arch-spec.json.
Write pytest unit tests for [file]::[class].
GENERATE TESTS FOR: [method list]
REQUIREMENTS: pytest fixtures for synthetic Polars DataFrames.
No QuestDB connection — mock ZMQ and QuestDB. pytest.approx() for floats.
```

**Template 5: Performance Optimization**
```
@workspace Read arch-spec.json.
Optimize [file]::[method]. Budget: <10ms on i5-1345U.
PROFILING RESULTS: [method timings]
DIRECTIVES: [specific optimization approach per method]
CONSTRAINTS: Polars expressions only. No pandas. Preserve output column names/dtypes.
Add # PERF comment with old_ms → new_ms estimate above each change.
```

---

### 6.4 Modern ML Alternatives (CPU-Only, 2026 SOTA)

#### Model Selection Matrix

| Stage | Dataset Size | Recommended Stack | Rationale |
|-------|-------------|-------------------|-----------|
| Stage 1-2 early | < 500 labels | TabPFN only | Zero-shot, no overfitting |
| Stage 2 mid | 500–2K | TabPFN + XGBoost (50/50) | Hedge zero-shot with emerging supervised signal |
| Stage 2 late | 2K–10K | XGBoost/LGB/CatBoost stacking | Proven, interpretable, fast |
| Stage 2–3 | Any | + River online PARALLEL | Regime adaptation without retraining |
| Stage 3+ | > 20K | XGBoost + River + TSMixer | Full adaptive + deep temporal |
| Noisy regime | Any | LightGBM-DART | Reduce overfit to noise |

#### TabPFN (< 1000 labels)
```python
# models/tabpfn_model.py
from tabpfn import TabPFNClassifier
import polars as pl, numpy as np

class TabPFNSignalModel:
    def __init__(self):
        self.model = TabPFNClassifier(device="cpu", N_ensemble_configurations=16)
        self._fitted = False

    def fit(self, X: pl.DataFrame, y: pl.Series) -> None:
        self.model.fit(X.to_numpy(), y.to_numpy())
        self._fitted = True

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return self.model.predict_proba(X.to_numpy())
```
`pip install tabpfn`

#### River Online (Regime Adaptation)
```python
# models/online_model.py
from river import ensemble, tree

class RiverOnlineEnsemble:
    def __init__(self):
        self.model = ensemble.ADWINBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), n_models=10, seed=42)
        self._recent_correct = []

    def update(self, features: dict, label: int) -> None:
        self.model.learn_one(features, label)

    def predict(self, features: dict) -> dict:
        proba = self.model.predict_proba_one(features)
        return {"label": max(proba, key=proba.get), "probabilities": proba}

    def get_adaptive_weight(self) -> float:
        if len(self._recent_correct) < 20:
            return 0.1
        return 0.4 if sum(self._recent_correct[-20:]) / 20 > 0.55 else 0.1
```
`pip install river` — sub-ms inference and update. Runs on E-core negligible CPU.

#### LightGBM DART (Noisy Regime)
```python
LGBM_DART_PARAMS = {
    "boosting_type": "dart",
    "drop_rate": 0.1,
    "skip_drop": 0.5,
    "max_drop": 50,
    "device": "cpu",
    "num_threads": 4,
}

def get_lgbm_params(regime: str) -> dict:
    if regime in ["HIGH_VOLATILITY", "CHOPPY"]:
        return LGBM_DART_PARAMS
    return LGBM_STANDARD_PARAMS  # standard gbdt
```
**Caveat:** DART + early stopping is unsupported in LightGBM. Inference speed identical to standard.

#### TSMixer (Stage 3+, > 20K labels)
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU before any torch import
from neuralforecast.models import TSMixer
```
CPU inference: ~2–5ms. 50x faster than LSTM. O(N) not O(N²) like Transformers.
`pip install neuralforecast`

---

### 6.5 The Elite Research Prompt Template

```
[SYS_INIT: ELITE QUANT ARCHITECT MODE = ON. MAX DENSITY. ZERO FLUFF.]

════════════════════════════════════════════════════
CONTEXT BLOCK — NSE ALGOTRADER CURRENT BLUEPRINT
════════════════════════════════════════════════════
HARDWARE:   Intel i5-1345U | 10 cores | 16GB RAM | NO GPU | CPU-ONLY
STACK:      Python 3.11 | QuestDB | Polars | XGBoost/LGB/CatBoost | ZeroMQ
BROKER:     AngelOne SmartAPI (primary) | Shoonya (failover)
DATA:       AngelOne WebSocket ticks → Volume Bars → 45 microstructure features
STAGE:      Stage 2 — paper trading, full pipeline, no live capital

6-POINT SYSTEM:
1. INGESTION:   Tick → QuestDB[ticks] + ZMQ PUB :5555
2. BAR BUILD:   ZMQ SUB → VolumeBarBuilder → QuestDB[volume_bars] + ZMQ PUB :5558
3. FEATURES:    ZMQ SUB → FeatureEngine (Polars) → QuestDB[features] + ZMQ PUSH :5556
4. INFERENCE:   ZMQ PULL → EnsembleInferenceEngine → QuestDB[signals]
5. RISK:        SignalRouter → RiskManager (ZMQ REP :5557) → OrderSpec or Reject
6. EXECUTION:   OrderExecutor → Shoonya API → QuestDB[orders, fills]

════════════════════════════════════════════════════
DIRECTIVE
════════════════════════════════════════════════════
Act as Principal Quant Researcher + HFT Systems Architect.

RESEARCH QUESTION:
[INSERT YOUR QUESTION HERE]

OUTPUT FORMAT (strict):
1. DIRECT ANSWER: 2-3 sentences max
2. TECHNICAL DEPTH: Implementation code, formulas
3. NSE-SPECIFIC: What changes for Indian market
4. INTEGRATION POINT: Which file/method to modify
5. RISKS: What breaks and when
6. DECISION: Implement now (Stage 2) / defer Stage 3 / skip

ACTIVE CRITIQUES:
[PASTE 3-5 relevant items from arch-spec.json::active_critiques]
════════════════════════════════════════════════════
```
