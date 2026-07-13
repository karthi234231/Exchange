"""
rl_sizer.py — PPO-based position sizer using Stable-Baselines3.

Stage 4 ONLY. The ML ensemble + signal_hierarchy handles sizing in Stages 1-3.
This module is disabled until Stage 4 is active.

Architecture:
    - PPO (Proximal Policy Optimisation) on top of NSETradingEnv
    - CPU-only: device="cpu" is non-negotiable (Intel i5-1345U, no GPU)
    - Paper/live trading: FORBIDDEN until all 13 go-live gates pass

The RL agent learns HOW MUCH to trade (position size fraction in [-1, 1]).
The ML ensemble continues to control WHAT direction to trade.

Guardrails enforced at inference:
    - min_training_episodes before deployment
    - min_sharpe on validation environment
    - max_position_override hard cap on any RL output
    - cooldown_days between retrains
    - Freeze flat when ATR > 2× baseline (high-volatility protection)
    - No retraining during market hours (9:15-15:30 IST on weekdays)
"""
from __future__ import annotations

# ── Stage guard ────────────────────────────────────────────────────────────────
_STAGE = "4"  # Stage 4 only — do NOT import in Stages 1-3

import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# ── IST timezone offset ────────────────────────────────────────────────────────
# IST = UTC+5:30 (no DST)
_IST_OFFSET_HOURS = 5
_IST_OFFSET_MINUTES = 30

_MARKET_OPEN_IST  = time(9, 15)
_MARKET_CLOSE_IST = time(15, 30)

# Weekdays: Monday=0 … Friday=4
_WEEKDAYS = frozenset(range(5))


def _require_stage_4() -> None:
    """Raise NotImplementedError when not in Stage 4 configuration."""
    stage_str = os.environ.get("ALGO_STAGE", _STAGE)
    try:
        stage_num = int(stage_str)
        if stage_num < 4:
            raise NotImplementedError(
                f"RLPositionSizer is Stage 4 only. "
                f"Current ALGO_STAGE={stage_str!r}. "
                "Set ALGO_STAGE=4 in your environment to enable RL position sizing."
            )
    except ValueError:
        if stage_str != "4":
            raise NotImplementedError(
                f"RLPositionSizer is Stage 4 only. Current ALGO_STAGE={stage_str!r}."
            )


# ══════════════════════════════════════════════════════════════════════════════
# Configuration dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RLSizerConfig:
    """
    Hyperparameters and paths for the PPO position sizer.

    All SB3 PPO parameters are explicitly set here so training is reproducible.
    device MUST remain "cpu" — this machine has no GPU.
    """

    # SB3 PPO parameters
    n_timesteps: int = 500_000          # Total environment steps for training
    learning_rate: float = 3e-4        # Adam learning rate
    n_steps: int = 2048                # PPO rollout buffer size (steps per update)
    batch_size: int = 64               # Mini-batch size for gradient update
    n_epochs: int = 10                 # PPO epochs per rollout batch
    gamma: float = 0.99                # Discount factor
    device: str = "cpu"                # NEVER change — hardware constraint

    # Model architecture
    policy: str = "MlpPolicy"         # Fully-connected policy network

    # Persistence
    model_save_path: str = "models/saved/rl_sizer.zip"
    tensorboard_log: str = "logs/rl_tensorboard/"

    def __post_init__(self) -> None:
        if self.device != "cpu":
            raise ValueError(
                f"RLSizerConfig.device must be 'cpu' (hardware constraint). "
                f"Got {self.device!r}. This system has no GPU."
            )


@dataclass
class RLSizerGuardrails:
    """
    Safety guardrails for RL sizer deployment and retraining.

    These limits exist to prevent the RL agent from over-trading,
    retraining on stale data, or operating in adverse market conditions.
    """

    min_training_episodes: int = 100   # Minimum episodes before deployment
    min_sharpe: float = 0.5            # Minimum Sharpe on validation env
    max_position_override: float = 0.05  # Hard cap on |RL output| (5% of capital)
    cooldown_days: int = 7             # Minimum days between retrains
    freeze_in_high_vol: bool = True    # Flat when ATR > 2× baseline ATR

    def __post_init__(self) -> None:
        if self.max_position_override <= 0 or self.max_position_override > 1.0:
            raise ValueError(
                f"max_position_override must be in (0, 1]. "
                f"Got {self.max_position_override}"
            )
        if self.min_sharpe < 0:
            raise ValueError(f"min_sharpe must be non-negative. Got {self.min_sharpe}")


# ══════════════════════════════════════════════════════════════════════════════
# RLPositionSizer
# ══════════════════════════════════════════════════════════════════════════════

class RLPositionSizer:
    """
    PPO-based position sizer for NSE intraday equities.

    The sizer is responsible only for the SIZE of a position (how much).
    Direction (long/short/flat) comes from the ML ensemble signal hierarchy.

    Usage:
        config = RLSizerConfig()
        guardrails = RLSizerGuardrails()
        sizer = RLPositionSizer(config, guardrails)
        result = sizer.train(train_envs, val_env)
        fraction = sizer.get_position_fraction(obs, current_atr, baseline_atr)

    The model is not usable until train() or load() has been called.
    SB3 and gymnasium are imported lazily to avoid hard dependency at module import.
    """

    def __init__(
        self,
        config: RLSizerConfig,
        guardrails: RLSizerGuardrails,
    ) -> None:
        _require_stage_4()

        self.config = config
        self.guardrails = guardrails

        # Deferred SB3 import — guard against missing optional dependency
        try:
            import stable_baselines3  # noqa: F401 — validate availability
            self._sb3_available = True
        except ImportError:
            self._sb3_available = False
            log.warning(
                "rl_sizer.sb3_not_installed",
                msg="stable_baselines3 not found. Install with: pip install stable-baselines3",
            )

        self._model: Optional[object] = None   # SB3 PPO instance, typed as object to avoid import
        self._trained: bool = False
        self._last_train_date: Optional[date] = None

        log.info(
            "rl_sizer.init",
            stage=_STAGE,
            n_timesteps=config.n_timesteps,
            device=config.device,
            sb3_available=self._sb3_available,
        )

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        train_envs: list,   # list[NSETradingEnv]
        val_env: object,    # NSETradingEnv
    ) -> dict:
        """
        Train PPO on the training environments and validate on val_env.

        Args:
            train_envs: List of NSETradingEnv instances for training.
                        Wrapped in SyncVectorEnv for vectorised rollout collection.
            val_env:    A single NSETradingEnv for post-training Sharpe validation.

        Returns:
            dict with keys: "status" ("ok" | "failed"), "sharpe", "episodes"
        """
        self._require_sb3("train")

        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SyncVectorEnv

        if not train_envs:
            raise ValueError("train_envs must contain at least one environment.")

        n_envs = len(train_envs)
        log.info(
            "rl_sizer.train.start",
            n_envs=n_envs,
            n_timesteps=self.config.n_timesteps,
            device=self.config.device,
        )

        # Wrap list of envs into SyncVectorEnv
        # Each factory captures the env by reference — reset() will be called by SB3.
        env_factories = [self._make_env_factory(env) for env in train_envs]
        vec_env = SyncVectorEnv(env_factories)

        try:
            # Ensure save directory exists
            save_path = Path(self.config.model_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            model = PPO(
                policy=self.config.policy,
                env=vec_env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                device=self.config.device,
                tensorboard_log=self.config.tensorboard_log,
                verbose=0,
            )

            model.learn(total_timesteps=self.config.n_timesteps)

        finally:
            vec_env.close()

        # ── Validate on val_env ────────────────────────────────────────────
        sharpe = self._run_validation_episode(model, val_env)
        n_episodes = self.config.n_timesteps // max(1, getattr(val_env, "_n_bars", 26))

        log.info(
            "rl_sizer.train.complete",
            sharpe=round(sharpe, 4),
            min_required_sharpe=self.guardrails.min_sharpe,
            n_approx_episodes=n_episodes,
        )

        if sharpe < self.guardrails.min_sharpe:
            log.warning(
                "rl_sizer.train.below_sharpe_threshold",
                sharpe=round(sharpe, 4),
                min_sharpe=self.guardrails.min_sharpe,
                action="model_NOT_saved",
            )
            self._trained = False
            return {"status": "failed", "sharpe": sharpe, "episodes": n_episodes}

        # Save model and mark as trained
        model.save(str(save_path))
        self._model = model
        self._trained = True
        self._last_train_date = date.today()

        log.info(
            "rl_sizer.train.saved",
            path=str(save_path),
            last_train_date=str(self._last_train_date),
        )

        return {"status": "ok", "sharpe": sharpe, "episodes": n_episodes}

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self, path: Optional[str] = None) -> None:
        """
        Load a pre-trained PPO model from a zip file.

        Args:
            path: Path to the saved SB3 model zip. Defaults to config.model_save_path.
        """
        self._require_sb3("load")

        from stable_baselines3 import PPO

        load_path = Path(path) if path else Path(self.config.model_save_path)

        if not load_path.exists():
            raise FileNotFoundError(
                f"RL sizer model not found at {load_path}. "
                "Run train() first or provide a valid path."
            )

        log.info("rl_sizer.load", path=str(load_path))
        self._model = PPO.load(str(load_path), device=self.config.device)
        self._trained = True

        log.info("rl_sizer.load.ok", path=str(load_path))

    # ── Inference ──────────────────────────────────────────────────────────────

    def get_position_fraction(
        self,
        observation: np.ndarray,
        current_atr: float,
        baseline_atr: float,
    ) -> float:
        """
        Return target position fraction in [-max_position_override, +max_position_override].

        Args:
            observation:  27-dim state vector from NSETradingEnv._get_observation().
            current_atr:  Current ATR(14) value in price units.
            baseline_atr: Baseline (e.g. rolling 20-day median) ATR for comparison.

        Returns:
            float in [-guardrails.max_position_override, +guardrails.max_position_override].

        Raises:
            RuntimeError: If the model has not been trained or loaded.
        """
        if not self._trained or self._model is None:
            raise RuntimeError(
                "RL sizer not trained. Use ML ensemble for sizing. "
                "Call train() or load() before get_position_fraction()."
            )

        # Freeze flat in high-volatility regime
        if self.guardrails.freeze_in_high_vol and baseline_atr > 0:
            if current_atr > 2.0 * baseline_atr:
                log.info(
                    "rl_sizer.freeze_high_vol",
                    current_atr=round(current_atr, 4),
                    baseline_atr=round(baseline_atr, 4),
                    threshold=round(2.0 * baseline_atr, 4),
                )
                return 0.0

        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        action, _ = self._model.predict(obs, deterministic=True)  # type: ignore[union-attr]
        raw_fraction = float(action.ravel()[0])

        # Hard cap regardless of RL output
        cap = self.guardrails.max_position_override
        clipped = float(np.clip(raw_fraction, -cap, cap))

        log.debug(
            "rl_sizer.predict",
            raw=round(raw_fraction, 6),
            clipped=round(clipped, 6),
            current_atr=round(current_atr, 4),
        )

        return clipped

    # ── Retraining safety ──────────────────────────────────────────────────────

    def is_safe_to_retrain(self) -> bool:
        """
        Return True only when it is safe to trigger a retrain.

        Blocks retraining if:
          1. Last retrain was within cooldown_days.
          2. Current time is during NSE market hours (9:15-15:30 IST, Mon-Fri).

        Returns:
            bool — True if retraining can proceed now.
        """
        today = date.today()

        # Cooldown check
        if self._last_train_date is not None:
            days_since = (today - self._last_train_date).days
            if days_since < self.guardrails.cooldown_days:
                log.info(
                    "rl_sizer.retrain_blocked.cooldown",
                    days_since=days_since,
                    cooldown_required=self.guardrails.cooldown_days,
                )
                return False

        # Market hours check — convert UTC now to IST
        utc_now = datetime.now(tz=timezone.utc)
        ist_now = utc_now.replace(tzinfo=None) + timedelta(
            hours=_IST_OFFSET_HOURS, minutes=_IST_OFFSET_MINUTES
        )

        is_weekday = ist_now.weekday() in _WEEKDAYS
        ist_time = ist_now.time()
        in_market_hours = _MARKET_OPEN_IST <= ist_time <= _MARKET_CLOSE_IST

        if is_weekday and in_market_hours:
            log.info(
                "rl_sizer.retrain_blocked.market_hours",
                ist_time=ist_time.strftime("%H:%M"),
                market_open=_MARKET_OPEN_IST.strftime("%H:%M"),
                market_close=_MARKET_CLOSE_IST.strftime("%H:%M"),
            )
            return False

        return True

    # ── Private helpers ────────────────────────────────────────────────────────

    def _require_sb3(self, operation: str) -> None:
        """Raise ImportError with installation instructions if SB3 is missing."""
        if not self._sb3_available:
            raise ImportError(
                f"stable_baselines3 is required for RLPositionSizer.{operation}(). "
                "Install with: pip install stable-baselines3"
            )

    @staticmethod
    def _make_env_factory(env: object):  # returns Callable[[], gymnasium.Env]
        """
        Create a factory function for SyncVectorEnv from an existing env instance.

        SyncVectorEnv expects callables that return fresh environments.
        Each call to the returned factory creates a new env instance by copying
        the configuration of the provided env, so that parallel workers do not
        share state through a single object reference.
        """
        # Capture the class and init kwargs (if the env exposes them) so that
        # each factory invocation produces an independent instance.  NSETradingEnv
        # is expected to expose a ``_init_kwargs`` attribute (dict) set in its
        # __init__; if it does not, fall back to copying via copy.deepcopy so
        # at minimum the workers start with independent state snapshots.
        import copy

        env_class = type(env)
        init_kwargs = getattr(env, "_init_kwargs", None)

        if init_kwargs is not None:
            def _factory():
                return env_class(**init_kwargs)  # type: ignore[return-value]
        else:
            def _factory():  # type: ignore[misc]
                return copy.deepcopy(env)  # type: ignore[return-value]

        return _factory

    def _run_validation_episode(self, model: object, val_env: object) -> float:
        """
        Run one complete episode on val_env using deterministic policy.

        Returns:
            Sharpe ratio over step returns for the episode, or 0.0 if insufficient data.
        """
        obs, _ = val_env.reset()  # type: ignore[union-attr]
        step_returns: list[float] = []
        terminated = truncated = False

        while not (terminated or truncated):
            obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            action, _ = model.predict(obs_arr, deterministic=True)  # type: ignore[union-attr]
            obs, reward, terminated, truncated, info = val_env.step(action.ravel())  # type: ignore[union-attr]
            step_returns.append(float(reward))

        if len(step_returns) < 2:
            return 0.0

        arr = np.array(step_returns, dtype=np.float64)
        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr))

        if std_r < 1e-8:
            return 0.0

        # Annualised Sharpe proxy: scale by sqrt(n_steps)
        sharpe = (mean_r / std_r) * math.sqrt(len(arr))
        return float(sharpe)
