"""
rl_env.py — OpenAI Gymnasium-compatible NSE intraday trading environment.

Stage 4 ONLY. Raises NotImplementedError on instantiation if config STAGE < 4.

Episode = one trading day (26 bars of 15-min data, 9:15 to 15:30 IST).
The RL agent controls position sizing (HOW MUCH); the ML ensemble controls
direction (WHAT direction).  Reward is Sharpe-style risk-adjusted P&L after
India costs via IndiaCostModel.

References:
    - FinRL: Liu et al. (2021) "FinRL: A Deep Reinforcement Learning Library
      for Automated Stock Trading in Quantitative Finance"
    - NSE-specific cost model: algo_trading.risk.cost_model.IndiaCostModel
"""
from __future__ import annotations

# ── Stage guard ────────────────────────────────────────────────────────────────
_STAGE = "4"  # This module is Stage 4 only — do NOT import in Stages 1-3

# ── Gymnasium import guard ─────────────────────────────────────────────────────
try:
    import gymnasium
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    raise ImportError(
        "gymnasium is required for Stage 4 RL environment. "
        "Install it with: pip install gymnasium"
    )

import math
from collections import deque
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import numpy as np
import structlog

from algo_trading.risk.cost_model import IndiaCostModel

log = structlog.get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
# P1-FIX-35: Derive feature dimension from the authoritative feature engine constant
# rather than hardcoding, so RL env stays aligned when feature set changes.
from algo_trading.features.engine import N_FEATURES as _N_FEATURES

_N_STATE_EXTRAS: int = 4       # position_fraction, unrealized_pnl_pct, time_remaining, cash_fraction
STATE_DIM: int = _N_FEATURES + _N_STATE_EXTRAS

# P1-FIX-35: Derive bars/day from the central market calendar rather than
# hardcoding 26. If exchange session changes (e.g. F&O 15:40 extension),
# this constant stays correct automatically.
from algo_trading.core.calendar import SESSION_START as _SESSION_START
from algo_trading.core.calendar import SESSION_END as _SESSION_END

_SESSION_MINUTES: int = (
    (_SESSION_END.hour - _SESSION_START.hour) * 60
    + (_SESSION_END.minute - _SESSION_START.minute)
)
_N_BARS_PER_DAY: int = _SESSION_MINUTES // 15   # one bar per 15-min interval
_FORCE_CLOSE_BAR: int = _N_BARS_PER_DAY - 1     # last bar = force-close bar (15:15 IST)

_DIRECTION_CHANGE_PENALTY: float = -0.01   # Penalise position thrashing
_TERMINAL_SHARPE_SCALE: float = 0.1        # Terminal reward = sharpe × this
_ROLLING_WINDOW: int = 20                  # Window for rolling std of returns


def _require_stage_4() -> None:
    """Raise NotImplementedError when not in Stage 4 configuration."""
    try:
        # Lazy import to avoid circular dependency; config may not have STAGE attr yet.
        import os
        stage_str = os.environ.get("ALGO_STAGE", _STAGE)
        stage_num = int(stage_str)
        if stage_num < 4:
            raise NotImplementedError(
                f"NSETradingEnv is Stage 4 only. "
                f"Current ALGO_STAGE={stage_str!r}. "
                "Set ALGO_STAGE=4 in your environment to enable RL position sizing."
            )
    except ValueError:
        # Non-numeric ALGO_STAGE — let it pass if it matches "4"
        stage_str = os.environ.get("ALGO_STAGE", _STAGE)
        if stage_str != "4":
            raise NotImplementedError(
                f"NSETradingEnv is Stage 4 only. Current ALGO_STAGE={stage_str!r}."
            )


# ══════════════════════════════════════════════════════════════════════════════
# NSETradingEnv
# ══════════════════════════════════════════════════════════════════════════════

class NSETradingEnv(gymnasium.Env):
    """
    Single-stock intraday trading environment for NSE equities.

    One episode = one trading day (26 bars of 15-min OHLCV data).

    Observation space (27-dim float32):
        [0:23]  — feature engine outputs (momentum, microstructure, etc.)
        [23]    — current_position_fraction ∈ [-1.0, 1.0]
        [24]    — unrealized_pnl_pct  (unrealised P&L as % of initial capital)
        [25]    — time_remaining_fraction ∈ [0.0, 1.0]
        [26]    — cash_fraction  (available cash / initial_capital)

    Action space (Box [-1, 1], shape (1,)):
        Target position fraction.  Actual position = action × max_position_fraction.
        -1.0 → maximum short,  0.0 → flat,  +1.0 → maximum long.

    Reward (Sharpe-style, per step):
        raw_reward  = (step_pnl - step_cost) / initial_capital
        shaped      = raw_reward / (rolling_std_returns + 1e-8)
        + thrashing penalty if direction changed
        + terminal bonus = cumulative Sharpe × 0.1

    Force-close:
        At bar 25 (15:15 IST), any open position is closed at market price and
        closing cost is charged.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,                    # shape (n_bars, 23)
        closes: np.ndarray,                      # shape (n_bars,)
        atrs: np.ndarray,                        # shape (n_bars,) — ATR14
        timestamps: list[datetime],              # length n_bars
        initial_capital: float = 500_000.0,
        max_position_fraction: float = 0.05,
        cost_model: Optional[IndiaCostModel] = None,
        symbol: str = "RELIANCE",
        render_mode: Optional[str] = None,
    ) -> None:
        _require_stage_4()

        super().__init__(render_mode=render_mode)

        # ── Validate inputs ────────────────────────────────────────────────
        n_bars = len(closes)
        if features.shape != (n_bars, _N_FEATURES):
            raise ValueError(
                f"features must have shape ({n_bars}, {_N_FEATURES}), "
                f"got {features.shape}"
            )
        if len(atrs) != n_bars:
            raise ValueError(f"atrs length {len(atrs)} must match n_bars {n_bars}")
        if len(timestamps) != n_bars:
            raise ValueError(f"timestamps length {len(timestamps)} must match n_bars {n_bars}")
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if not (0.0 < max_position_fraction <= 1.0):
            raise ValueError(f"max_position_fraction must be in (0, 1], got {max_position_fraction}")

        # ── Store episode data ─────────────────────────────────────────────
        self._features = features.astype(np.float32)
        self._closes = closes.astype(np.float32)
        self._atrs = atrs.astype(np.float32)
        self._timestamps = timestamps
        self._n_bars = n_bars
        self.symbol = symbol

        # ── Parameters ────────────────────────────────────────────────────
        self.initial_capital = float(initial_capital)
        self.max_position_fraction = float(max_position_fraction)
        self._cost_model = cost_model if cost_model is not None else IndiaCostModel()

        # ── Gymnasium spaces ───────────────────────────────────────────────
        obs_low = np.full(STATE_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(STATE_DIM, np.inf, dtype=np.float32)
        # Bound the structured extras
        obs_low[_N_FEATURES]     = -1.0   # position_fraction
        obs_high[_N_FEATURES]    =  1.0
        obs_low[_N_FEATURES + 2] =  0.0   # time_remaining_fraction
        obs_high[_N_FEATURES + 2] = 1.0
        obs_low[_N_FEATURES + 3] =  0.0   # cash_fraction
        obs_high[_N_FEATURES + 3] = 1.0

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ── Episode state (initialised by reset) ──────────────────────────
        self._bar_index: int = 0
        self._cash: float = self.initial_capital
        self._position_qty: int = 0           # shares held (negative = short)
        self._position_fraction: float = 0.0
        self._entry_price: float = 0.0        # weighted average entry price
        self._cum_pnl: float = 0.0            # cumulative episode P&L (INR)
        self._step_returns: deque[float] = deque(maxlen=_ROLLING_WINDOW)
        self._cum_sharpe_numerator: float = 0.0   # for terminal reward
        self._n_steps_done: int = 0

        log.debug(
            "rl_env.init",
            symbol=symbol,
            n_bars=n_bars,
            initial_capital=initial_capital,
            max_position_fraction=max_position_fraction,
        )

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to bar 0 for a new episode."""
        super().reset(seed=seed)

        self._bar_index = 0
        self._cash = self.initial_capital
        self._position_qty = 0
        self._position_fraction = 0.0
        self._entry_price = 0.0
        self._cum_pnl = 0.0
        self._step_returns.clear()
        self._cum_sharpe_numerator = 0.0
        self._n_steps_done = 0

        obs = self._get_observation()
        info: dict[str, Any] = {
            "bar_index": self._bar_index,
            "timestamp": self._timestamps[self._bar_index],
            "position_qty": 0,
            "position_fraction": 0.0,
            "cash": self._cash,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one bar step.

        Args:
            action: numpy array of shape (1,) with target position fraction in [-1, 1].

        Returns:
            obs, reward, terminated, truncated, info
        """
        if self._bar_index >= self._n_bars:
            raise RuntimeError("Episode is done — call reset() before stepping again.")

        bar = self._bar_index
        close = float(self._closes[bar])
        atr = float(self._atrs[bar])
        timestamp = self._timestamps[bar]

        prev_position_qty = self._position_qty
        prev_position_fraction = self._position_fraction
        prev_close = float(self._closes[bar - 1]) if bar > 0 else close

        # ── Force-close at bar 25 (15:15 IST) ─────────────────────────────
        force_close = (bar == _FORCE_CLOSE_BAR) and (self._position_qty != 0)
        if force_close:
            target_fraction = 0.0
        else:
            # Clip action to [-1, 1] and scale by max_position_fraction
            raw_action = float(np.clip(action[0], -1.0, 1.0))
            target_fraction = raw_action * self.max_position_fraction

        # ── Compute target quantity ────────────────────────────────────────
        if close > 0:
            capital_to_use = self.initial_capital * abs(target_fraction)
            target_qty_raw = int(capital_to_use / close)
            target_qty = target_qty_raw if target_fraction >= 0 else -target_qty_raw
        else:
            target_qty = 0

        # ── Compute cost if position changed ──────────────────────────────
        delta_qty = target_qty - prev_position_qty
        cost_inr = 0.0

        if delta_qty != 0:
            side = "BUY" if delta_qty > 0 else "SELL"
            trade_qty = abs(delta_qty)
            trade_price = Decimal(str(round(close, 2)))
            trade_atr = Decimal(str(round(atr, 2)))
            hour = timestamp.hour

            trade_cost = self._cost_model.compute(
                symbol=self.symbol,
                side=side,
                quantity=trade_qty,
                price=trade_price,
                is_delivery=False,
                atr=trade_atr,
                hour_of_day=hour,
            )
            cost_inr = float(trade_cost.total_inr)

        # ── Update position ────────────────────────────────────────────────
        self._position_qty = target_qty
        if target_qty != 0:
            self._position_fraction = (
                target_qty * close / self.initial_capital
            )
        else:
            self._position_fraction = 0.0
            self._entry_price = 0.0

        # Track entry price for unrealised P&L
        if delta_qty != 0 and target_qty != 0:
            if prev_position_qty == 0:
                self._entry_price = close
            else:
                # Weighted average entry price
                prev_value = abs(prev_position_qty) * self._entry_price
                new_value = abs(delta_qty) * close
                total_qty = abs(target_qty)
                self._entry_price = (prev_value + new_value) / total_qty if total_qty > 0 else close

        # ── Compute reward ─────────────────────────────────────────────────
        reward = self._compute_reward(
            prev_pos=prev_position_qty,
            curr_pos=self._position_qty,
            close=close,
            prev_close=prev_close,
            atr=atr,
            cost_inr=cost_inr,
        )

        # Accumulate P&L for terminal reward
        step_pnl_inr = reward * self.initial_capital  # approximate
        self._cum_pnl += step_pnl_inr
        self._n_steps_done += 1

        # ── Advance bar index ──────────────────────────────────────────────
        self._bar_index += 1

        # ── Terminal condition ─────────────────────────────────────────────
        terminated = self._bar_index >= self._n_bars or bar == _FORCE_CLOSE_BAR
        truncated = False

        # Terminal bonus: cumulative Sharpe × 0.1
        if terminated and len(self._step_returns) >= 2:
            mean_r = float(np.mean(self._step_returns))
            std_r = float(np.std(self._step_returns))
            if std_r > 1e-8:
                terminal_sharpe = mean_r / std_r * math.sqrt(self._n_steps_done)
            else:
                terminal_sharpe = 0.0
            reward += terminal_sharpe * _TERMINAL_SHARPE_SCALE

        # ── Observation ────────────────────────────────────────────────────
        if not terminated:
            obs = self._get_observation()
        else:
            # Return last valid observation (bar_index may now equal n_bars)
            obs = self._get_observation(bar_override=min(bar, self._n_bars - 1))

        # ── Info dict ─────────────────────────────────────────────────────
        info: dict[str, Any] = {
            "pnl_inr": float(step_pnl_inr),
            "cost_inr": float(cost_inr),
            "position_qty": int(self._position_qty),
            "position_fraction": float(self._position_fraction),
            "bar_index": int(bar),
            "timestamp": timestamp,
            "force_close": force_close,
        }

        log.debug(
            "rl_env.step",
            bar=bar,
            action=float(action[0]),
            target_qty=target_qty,
            cost_inr=round(cost_inr, 2),
            reward=round(reward, 6),
            terminated=terminated,
        )

        return obs, float(reward), terminated, truncated, info

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_observation(self, bar_override: Optional[int] = None) -> np.ndarray:
        """Assemble the 27-dimensional state vector as float32."""
        bar = bar_override if bar_override is not None else self._bar_index
        bar = min(bar, self._n_bars - 1)  # clamp to valid range

        obs = np.empty(STATE_DIM, dtype=np.float32)

        # Indices 0-22: feature engine output
        obs[:_N_FEATURES] = self._features[bar]

        # Index 23: current_position_fraction in [-1, 1]
        obs[_N_FEATURES] = np.clip(self._position_fraction / self.max_position_fraction, -1.0, 1.0)

        # Index 24: unrealized_pnl_pct
        if self._position_qty != 0 and self._entry_price > 0:
            current_price = float(self._closes[bar])
            direction = 1 if self._position_qty > 0 else -1
            unrealised = (
                (current_price - self._entry_price)
                * abs(self._position_qty)
                * direction
                / self.initial_capital
            )
        else:
            unrealised = 0.0
        obs[_N_FEATURES + 1] = np.float32(unrealised)

        # Index 25: time_remaining_fraction in [0, 1]
        bars_remaining = max(0, self._n_bars - 1 - bar)
        obs[_N_FEATURES + 2] = np.float32(bars_remaining / max(1, self._n_bars - 1))

        # Index 26: cash_fraction in [0, 1]
        position_value = abs(self._position_qty) * float(self._closes[bar])
        cash_available = max(0.0, self.initial_capital - position_value)
        obs[_N_FEATURES + 3] = np.float32(
            np.clip(cash_available / self.initial_capital, 0.0, 1.0)
        )

        return obs

    def _compute_reward(
        self,
        prev_pos: int,
        curr_pos: int,
        close: float,
        prev_close: float,
        atr: float,
        cost_inr: float,
    ) -> float:
        """
        Compute Sharpe-style shaped reward for one step.

        step_pnl = price_change × held_qty × direction_sign
        raw_reward = (step_pnl - cost_inr) / initial_capital
        shaped_reward = raw_reward / (rolling_std + 1e-8)
        + direction_change_penalty if sign flipped
        """
        # P&L on the position held through this bar (held = prev position)
        if prev_pos != 0:
            price_change = close - prev_close
            direction = 1 if prev_pos > 0 else -1
            step_pnl = price_change * abs(prev_pos) * direction
        else:
            step_pnl = 0.0

        raw_reward = (step_pnl - cost_inr) / self.initial_capital

        # Rolling std of returns for Sharpe shaping
        self._step_returns.append(raw_reward)
        if len(self._step_returns) >= 2:
            rolling_std = float(np.std(self._step_returns))
        else:
            rolling_std = 0.0

        shaped_reward = raw_reward / (rolling_std + 1e-8)

        # Penalise direction changes (prevents thrashing)
        prev_sign = 1 if prev_pos > 0 else (-1 if prev_pos < 0 else 0)
        curr_sign = 1 if curr_pos > 0 else (-1 if curr_pos < 0 else 0)
        if prev_sign != 0 and curr_sign != 0 and prev_sign != curr_sign:
            shaped_reward += _DIRECTION_CHANGE_PENALTY

        return float(shaped_reward)

    def render(self) -> None:
        """Print current bar info to stdout."""
        bar = min(self._bar_index, self._n_bars - 1)
        ts = self._timestamps[bar] if bar < len(self._timestamps) else "?"
        close = float(self._closes[bar])
        print(
            f"[NSETradingEnv] {self.symbol} | bar={bar:02d} | ts={ts} | "
            f"close={close:.2f} | pos={self._position_qty:+d} | "
            f"pos_frac={self._position_fraction:+.4f} | "
            f"cum_pnl={self._cum_pnl:+.2f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MultiStockTradingEnv
# ══════════════════════════════════════════════════════════════════════════════

class MultiStockTradingEnv(gymnasium.Env):
    """
    Portfolio-level training wrapper for multiple NSETradingEnv instances.

    Concatenates observations and actions across all constituent environments.
    Total reward = sum of per-stock rewards.

    Observation: np.concatenate([env.observation_space for env in envs])
    Action:      np.concatenate([env.action_space for env in envs])
    Reward:      sum of all individual step rewards
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, envs: list[NSETradingEnv]) -> None:
        _require_stage_4()

        if not envs:
            raise ValueError("MultiStockTradingEnv requires at least one NSETradingEnv.")

        super().__init__()
        self._envs = envs
        self._n_envs = len(envs)

        total_obs_dim = STATE_DIM * self._n_envs
        total_action_dim = self._n_envs

        # Concatenated observation space
        obs_low = np.full(total_obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(total_obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(total_obs_dim,), dtype=np.float32
        )

        # Concatenated action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_action_dim,), dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset all constituent environments."""
        super().reset(seed=seed)

        all_obs = []
        combined_info: dict[str, Any] = {"envs": []}

        for i, env in enumerate(self._envs):
            env_seed = (seed + i) if seed is not None else None
            obs, info = env.reset(seed=env_seed, options=options)
            all_obs.append(obs)
            combined_info["envs"].append(info)

        return np.concatenate(all_obs).astype(np.float32), combined_info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Step all constituent environments with their individual actions.

        Args:
            action: shape (n_envs,) — one target fraction per stock.
        """
        all_obs = []
        total_reward = 0.0
        all_terminated = []
        all_truncated = []
        combined_info: dict[str, Any] = {"envs": []}

        for i, env in enumerate(self._envs):
            env_action = action[i : i + 1]  # shape (1,)
            obs, reward, terminated, truncated, info = env.step(env_action)
            all_obs.append(obs)
            total_reward += reward
            all_terminated.append(terminated)
            all_truncated.append(truncated)
            combined_info["envs"].append(info)

        # Episode done when ALL sub-environments are done
        terminated = all(all_terminated)
        truncated = any(all_truncated)

        return (
            np.concatenate(all_obs).astype(np.float32),
            float(total_reward),
            terminated,
            truncated,
            combined_info,
        )

    def render(self) -> None:
        for env in self._envs:
            env.render()
