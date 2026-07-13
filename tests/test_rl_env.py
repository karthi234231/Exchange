"""
tests/test_rl_env.py — Unit tests for NSETradingEnv (Stage 4 RL environment).

Runs entirely on synthetic data — no file I/O, no real market data required.
All tests skip gracefully when gymnasium is not installed.

Tests:
    1.  test_env_reset_returns_correct_shape
    2.  test_env_step_returns_tuple
    3.  test_obs_shape_after_step
    4.  test_terminated_at_bar_25
    5.  test_reward_is_finite
    6.  test_position_fraction_bounded
    7.  test_flat_action_zero_cost
    8.  test_force_close_at_bar_25
    9.  test_positive_reward_on_uptrend
    10. test_negative_reward_on_wrong_direction
    11. test_reset_resets_state
    12. test_gymnasium_check_env
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

# Activate Stage 4 for all tests in this module
os.environ.setdefault("ALGO_STAGE", "4")

try:
    from algo_trading.models.rl_env import NSETradingEnv, STATE_DIM, _N_BARS_PER_DAY
except ImportError:
    pytest.skip("gymnasium not installed — skipping RL env tests", allow_module_level=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper factory
# ══════════════════════════════════════════════════════════════════════════════

def make_env(n_bars: int = 26, symbol: str = "RELIANCE") -> NSETradingEnv:
    """
    Create a deterministic NSETradingEnv with synthetic data for testing.

    Args:
        n_bars:  Number of bars in the episode (default 26 = one full day).
        symbol:  NSE symbol string (default "RELIANCE").

    Returns:
        A freshly constructed NSETradingEnv ready for reset().
    """
    np.random.seed(42)
    features = np.random.randn(n_bars, 25).astype(np.float32)
    closes = 2800.0 + np.cumsum(np.random.randn(n_bars) * 5)

    from datetime import datetime, timedelta
    base = datetime(2024, 1, 2, 9, 15)
    timestamps = [base + timedelta(minutes=15 * i) for i in range(n_bars)]
    atrs = np.full(n_bars, 15.0)

    return NSETradingEnv(
        features=features,
        closes=closes.astype(np.float32),
        atrs=atrs,
        timestamps=timestamps,
        symbol=symbol,
    )


def make_trending_env(direction: float = 1.0, n_bars: int = 26) -> NSETradingEnv:
    """
    Create an environment where closes move monotonically (up or down).

    Args:
        direction:  +1.0 for uptrend, -1.0 for downtrend.
        n_bars:     Episode length.
    """
    from datetime import datetime, timedelta

    np.random.seed(0)
    features = np.random.randn(n_bars, 25).astype(np.float32)

    # Strongly monotonic price series: +/- 10 per bar
    step = 10.0 * direction
    closes = np.array([2800.0 + i * step for i in range(n_bars)], dtype=np.float32)

    base = datetime(2024, 1, 2, 9, 15)
    timestamps = [base + timedelta(minutes=15 * i) for i in range(n_bars)]
    atrs = np.full(n_bars, 5.0, dtype=np.float32)

    return NSETradingEnv(
        features=features,
        closes=closes,
        atrs=atrs,
        timestamps=timestamps,
        symbol="TESTSTOCK",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEnvReset:
    """Tests covering reset() behaviour."""

    def test_env_reset_returns_correct_shape(self):
        """Observation shape after reset must be (27,) and info must be a dict."""
        env = make_env()
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray), "obs must be np.ndarray"
        assert obs.shape == (STATE_DIM,), f"Expected obs shape ({STATE_DIM},), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
        assert isinstance(info, dict), "info must be a dict"

    def test_reset_resets_state(self):
        """
        After running an episode, reset() must produce the same initial
        observation as the very first reset().
        """
        env = make_env()
        obs_first, _ = env.reset()

        # Run through a full episode
        done = False
        action = np.array([0.5], dtype=np.float32)
        while not done:
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        # Reset again and compare
        obs_second, _ = env.reset()

        np.testing.assert_array_equal(
            obs_first, obs_second,
            err_msg="reset() must return identical initial obs across episode boundaries"
        )


class TestEnvStep:
    """Tests covering step() behaviour."""

    def test_env_step_returns_tuple(self):
        """step() must return a 5-tuple (obs, reward, terminated, truncated, info)."""
        env = make_env()
        env.reset()
        result = env.step(np.array([0.5], dtype=np.float32))

        assert isinstance(result, tuple), "step() must return a tuple"
        assert len(result) == 5, f"Expected 5-tuple, got {len(result)}-tuple"

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, int, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_obs_shape_after_step(self):
        """Observation shape must remain (27,) after each step."""
        env = make_env()
        env.reset()

        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(np.array([0.3], dtype=np.float32))
            assert obs.shape == (STATE_DIM,), \
                f"Observation shape changed to {obs.shape} after step"
            if terminated or truncated:
                break

    def test_reward_is_finite(self):
        """Reward must be a finite float on every step."""
        env = make_env()
        env.reset()
        done = False
        step_count = 0

        while not done:
            action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_count += 1

            assert math.isfinite(reward), \
                f"Reward is not finite at step {step_count}: {reward}"

    def test_position_fraction_bounded(self):
        """position_fraction in info must stay within [-1, 1] on every step."""
        env = make_env()
        env.reset()
        done = False

        while not done:
            action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frac = info["position_fraction"]
            assert -1.0 <= frac <= 1.0, \
                f"position_fraction {frac} is out of [-1, 1] bounds"

    def test_flat_action_zero_cost(self):
        """
        Sending action=0.0 on the very first step (flat→flat) must incur zero cost,
        because no position change occurs.
        """
        env = make_env()
        env.reset()

        # Start flat; action=0.0 means stay flat
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

        assert info["cost_inr"] == pytest.approx(0.0, abs=1e-6), \
            f"Flat→flat should have zero cost, got {info['cost_inr']}"


class TestEpisodeTermination:
    """Tests covering termination logic."""

    def test_terminated_at_bar_25(self):
        """The episode must terminate at bar index 25 (15:15 IST)."""
        env = make_env(n_bars=26)
        env.reset()

        terminated = False
        steps = 0
        final_bar = None

        while not terminated:
            _, _, terminated, truncated, info = env.step(
                np.array([0.1], dtype=np.float32)
            )
            steps += 1
            final_bar = info["bar_index"]
            if truncated:
                break

        assert terminated, "Episode must terminate (terminated=True)"
        assert steps == 26, \
            f"Expected 26 steps for a 26-bar episode, got {steps}"
        assert final_bar == 25, \
            f"Final bar_index must be 25 (force-close bar), got {final_bar}"

    def test_force_close_at_bar_25(self):
        """
        If the agent holds a long position at bar 24, bar 25 must force-close it.
        After force close, position_qty must be 0.
        """
        env = make_env(n_bars=26)
        env.reset()

        # Take a long position on bars 0-24
        buy_action = np.array([1.0], dtype=np.float32)
        flat_action = np.array([0.0], dtype=np.float32)

        for step_idx in range(24):
            _, _, terminated, truncated, info = env.step(buy_action)
            assert not (terminated or truncated), \
                f"Episode terminated too early at step {step_idx}"

        # Bar 25 — force close should happen
        _, _, terminated, truncated, info = env.step(flat_action)

        assert terminated, "Episode must terminate at bar 25"
        assert info["position_qty"] == 0, \
            f"Force-close must zero position, got {info['position_qty']}"


class TestRewardSignals:
    """Tests verifying reward signals are economically sensible."""

    def test_positive_reward_on_uptrend(self):
        """
        Monotonically increasing closes + sustained long action should yield
        positive cumulative reward over a complete episode.
        """
        env = make_trending_env(direction=+1.0)
        env.reset()

        buy_action = np.array([1.0], dtype=np.float32)
        cumulative_reward = 0.0
        done = False

        while not done:
            _, reward, terminated, truncated, _ = env.step(buy_action)
            cumulative_reward += reward
            done = terminated or truncated

        assert cumulative_reward > 0.0, \
            f"Long position in uptrend should yield positive reward, got {cumulative_reward:.6f}"

    def test_negative_reward_on_wrong_direction(self):
        """
        Strongly decreasing closes + sustained long (buy) action should yield
        negative cumulative reward over a complete episode.
        """
        env = make_trending_env(direction=-1.0)
        env.reset()

        buy_action = np.array([1.0], dtype=np.float32)
        cumulative_reward = 0.0
        done = False

        while not done:
            _, reward, terminated, truncated, _ = env.step(buy_action)
            cumulative_reward += reward
            done = terminated or truncated

        assert cumulative_reward < 0.0, \
            f"Long position in downtrend should yield negative reward, got {cumulative_reward:.6f}"


class TestGymnasiumCompliance:
    """Tests verifying gymnasium API compliance."""

    def test_gymnasium_check_env(self):
        """
        gymnasium.utils.env_checker.check_env must pass without errors.

        Skips silently if gymnasium is not installed.
        """
        try:
            from gymnasium.utils.env_checker import check_env
        except ImportError:
            pytest.skip("gymnasium.utils.env_checker not available")

        env = make_env(n_bars=26)
        # check_env raises AssertionError on API violations
        try:
            check_env(env, warn=True, skip_render_check=True)
        except Exception as exc:
            pytest.fail(f"gymnasium.check_env raised an exception: {exc}")
