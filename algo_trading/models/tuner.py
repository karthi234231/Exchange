"""Optuna hyperparameter tuner for XGBoost, LightGBM, and CatBoost.

Spec reference: CDR-3B-C23.

Critical constraints:
- Max 50 trials (100+ trials overfits hyperparameters to the validation fold).
- fold[-2] is used for optimization; fold[-1] is the untouched final holdout.
- Complexity penalty is added to the objective to discourage deep/large trees.
- Retraining only allowed on Sundays (enforced via model_registry.is_safe_to_retrain).
- n_jobs=-1 on study.optimize() to use all available cores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import structlog
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial logs

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Registry safety check import — imported lazily to avoid circular imports.
# ---------------------------------------------------------------------------


def _get_is_safe_to_retrain() -> bool:
    """Import and call is_safe_to_retrain() from model_registry.

    Returns:
        True if it is safe (Sunday) to begin a retraining run, False otherwise.
    """
    try:
        from algo_trading.models.model_registry import ModelRegistry

        return ModelRegistry().is_safe_to_retrain()
    except ImportError:
        logger.warning(
            "model_registry.ModelRegistry not found — defaulting to True",
            module="tuner",
        )
        return True


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class TunerConfig:
    """Configuration for HyperparamTuner.

    Attributes:
        n_trials: Hard cap on Optuna trials. Do NOT increase above 50 — more
            trials overfit the hyperparameters to the validation fold.
        timeout_seconds: Wall-clock budget per study (1 hour).
        n_jobs: Parallel trial workers. -1 uses all available CPU cores.
        metric: Walk-forward objective metric to maximise.
        use_fold_minus_2: When True (required), use fold[-2] for tuning.
            fold[-1] is reserved as the final holdout and must never be seen
            during optimisation.
    """

    n_trials: int = 50  # HARD CAP — do not increase
    timeout_seconds: int = 3600  # 1 hour max
    n_jobs: int = -1  # all cores
    metric: str = "sharpe"  # optimise for walk-forward Sharpe
    use_fold_minus_2: bool = True  # NEVER tune on fold[-1]
    random_seed: int = 42
    verbosity: int = 0  # 0 = silent inside models


# ---------------------------------------------------------------------------
# Complexity penalty
# ---------------------------------------------------------------------------


def _complexity_penalty(params: dict[str, Any]) -> float:
    """Compute a scalar penalty that discourages deep and large tree models.

    The penalty is combined into the objective as:
        ``return -sharpe + _complexity_penalty(params) * 0.1``

    This prevents Optuna from selecting models that barely improve validation
    Sharpe but are significantly more complex.

    Args:
        params: Hyperparameter dict from the current Optuna trial. Must
            contain at least a subset of the keys: ``max_depth``,
            ``num_leaves``, ``depth``, ``min_child_weight``,
            ``min_child_samples``, ``n_estimators``, ``iterations``.

    Returns:
        Non-negative scalar penalty value.
    """
    penalty = 0.0

    # Depth: normalise over the range [3, 8] so max penalty = 1.0
    depth = params.get("max_depth") or params.get("depth") or 0
    if depth:
        penalty += (depth - 3) / (8 - 3)  # 0.0 at depth=3, 1.0 at depth=8

    # Leaf count (LGB-specific): normalise over [15, 127]
    num_leaves = params.get("num_leaves", 0)
    if num_leaves:
        penalty += (num_leaves - 15) / (127 - 15)

    # min_child_weight: small values = less regularisation → penalise
    mcw = params.get("min_child_weight", None)
    if mcw is not None:
        penalty += (20 - mcw) / (20 - 1)  # 0.0 at mcw=20, 1.0 at mcw=1

    # Estimator count: normalise over [100, 500]
    n_est = (
        params.get("n_estimators")
        or params.get("iterations")
        or 0
    )
    if n_est:
        penalty += (n_est - 100) / (500 - 100)

    return penalty


# ---------------------------------------------------------------------------
# Main tuner class
# ---------------------------------------------------------------------------


class HyperparamTuner:
    """Optuna-based hyperparameter optimiser for the NSE AlgoTrader ensemble.

    Uses walk-forward fold[-2] for all optimisation.  fold[-1] is the final
    held-out test set and is NEVER accessed inside this class.

    Args:
        config: Tuner configuration. Defaults to ``TunerConfig()``.
    """

    def __init__(self, config: TunerConfig = TunerConfig()) -> None:
        self.config = config
        self._log = structlog.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_safety(self) -> None:
        """Assert it is safe (Sunday) to begin optimisation.

        Raises:
            RuntimeError: If is_safe_to_retrain() returns False.
        """
        if not _get_is_safe_to_retrain():
            raise RuntimeError(
                "HyperparamTuner: is_safe_to_retrain() returned False. "
                "Hyperparameter tuning is permitted on Sundays only."
            )

    def _make_study(self, study_name: str) -> optuna.Study:
        """Create a minimisation study (objective returns -sharpe + penalty).

        Args:
            study_name: Unique name used by Optuna storage.

        Returns:
            A new Optuna Study configured for minimisation.
        """
        sampler = optuna.samplers.TPESampler(seed=self.config.random_seed)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        return optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

    @staticmethod
    def _compute_sharpe(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Estimate a proxy Sharpe ratio from predicted probabilities.

        The proxy treats each sample as a trade: long when predicted bull
        (class 2) probability > bear (class 0) probability, short otherwise,
        scaled by the true label.

        Class encoding matches ensemble.py _CLASS_TO_SIGNAL: [P0-FIX-11]
          class 0 = bear  → signal -1
          class 1 = sideways → signal 0
          class 2 = bull  → signal +1

        Args:
            y_true: Integer class labels (0=bear, 1=sideways, 2=bull).
            y_pred_proba: Probability array of shape (n_samples, n_classes).

        Returns:
            Annualised proxy Sharpe ratio. Returns 0.0 if std is near zero.
        """
        # P0-FIX-11: class 0=bear, 1=sideways, 2=bull (matches ensemble._CLASS_TO_SIGNAL)
        direction_map = {0: -1.0, 1: 0.0, 2: 1.0}
        true_returns = np.array([direction_map.get(int(c), 0.0) for c in y_true])

        # Predicted signal: sign of (bull_prob - bear_prob)
        # bull = class 2, bear = class 0  [P0-FIX-11]
        bull_prob = y_pred_proba[:, 2] if y_pred_proba.shape[1] > 2 else y_pred_proba[:, -1]
        bear_prob = y_pred_proba[:, 0]
        signal = np.sign(bull_prob - bear_prob)

        trade_returns = signal * true_returns
        mean_ret = np.mean(trade_returns)
        std_ret = np.std(trade_returns, ddof=1)
        if std_ret < 1e-9:
            return 0.0
        # Annualise assuming 252 trading days
        return float(mean_ret / std_ret * np.sqrt(252))

    # ------------------------------------------------------------------
    # XGBoost tuning
    # ------------------------------------------------------------------

    def tune_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Tune XGBoost hyperparameters using Optuna.

        Search space (per CDR-3B-C23):
            - max_depth: [3, 8]
            - learning_rate: [0.01, 0.3] (log scale)
            - n_estimators: [100, 500]
            - subsample: [0.6, 1.0]
            - colsample_bytree: [0.6, 1.0]
            - min_child_weight: [1, 20]

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Validation feature matrix (must be fold[-2]).
            y_val: Validation labels.
            sample_weights: Optional per-sample weights for training.

        Returns:
            Dictionary of best XGBoost hyperparameters found within n_trials.
        """
        self._check_safety()
        self._log.info("tune_xgboost.start", n_trials=self.config.n_trials)
        n_classes = len(np.unique(y_train))

        def objective(trial: optuna.Trial) -> float:
            try:
                params: dict[str, Any] = {
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                    "objective": "multi:softprob",
                    "num_class": n_classes,
                    "tree_method": "hist",
                    "device": "cpu",
                    "verbosity": self.config.verbosity,
                    "random_state": self.config.random_seed,
                    "n_jobs": 1,  # avoid CPU oversubscription with Optuna parallel trials
                }
                model = XGBClassifier(**params)
                fit_kwargs: dict[str, Any] = {}
                if sample_weights is not None:
                    fit_kwargs["sample_weight"] = sample_weights
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, **fit_kwargs)
                proba = model.predict_proba(X_val)
                sharpe = self._compute_sharpe(y_val, proba)
                penalty = _complexity_penalty(params)
                return -sharpe + penalty * 0.1
            except Exception as exc:
                self._log.warning("tune_xgboost.trial_failed", error=str(exc))
                return 0.0

        study = self._make_study("xgboost_tuning")
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=self.config.n_jobs,
            show_progress_bar=False,
        )
        best_sharpe = -study.best_value  # objective returns -sharpe + penalty
        self._log.info(
            "tune_xgboost.done",
            best_value=study.best_value,
            best_sharpe=best_sharpe,
            best_params=study.best_params,
        )
        if best_sharpe < 0.3:
            self._log.error(
                "tune_xgboost.best_sharpe_below_threshold",
                best_sharpe=best_sharpe,
                threshold=0.3,
            )
            raise RuntimeError(
                f"Optuna best Sharpe {best_sharpe:.3f} < 0.3 — aborting param write"
            )
        best = study.best_params
        return best

    # ------------------------------------------------------------------
    # LightGBM tuning
    # ------------------------------------------------------------------

    def tune_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Tune LightGBM hyperparameters using Optuna.

        Search space (per CDR-3B-C23):
            - num_leaves: [15, 127]
            - learning_rate: [0.01, 0.3] (log scale)
            - n_estimators: [100, 500]
            - min_child_samples: [5, 50]
            - feature_fraction: [0.6, 1.0]

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Validation feature matrix (must be fold[-2]).
            y_val: Validation labels.
            sample_weights: Optional per-sample weights for training.

        Returns:
            Dictionary of best LightGBM hyperparameters found within n_trials.
        """
        self._check_safety()
        self._log.info("tune_lightgbm.start", n_trials=self.config.n_trials)

        def objective(trial: optuna.Trial) -> float:
            try:
                params: dict[str, Any] = {
                    "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                    "objective": "multiclass",
                    "verbose": -1,
                    "random_state": self.config.random_seed,
                    "device": "cpu",
                    "n_jobs": 1,  # avoid CPU oversubscription with Optuna parallel trials
                }
                model = LGBMClassifier(**params)
                fit_kwargs: dict[str, Any] = {
                    "eval_set": [(X_val, y_val)],
                    "callbacks": [
                        __import__("lightgbm").early_stopping(10, verbose=False),
                        __import__("lightgbm").log_evaluation(period=-1),
                    ],
                }
                if sample_weights is not None:
                    fit_kwargs["sample_weight"] = sample_weights
                model.fit(X_train, y_train, **fit_kwargs)
                proba = model.predict_proba(X_val)
                sharpe = self._compute_sharpe(y_val, proba)
                penalty = _complexity_penalty(params)
                return -sharpe + penalty * 0.1
            except Exception as exc:
                self._log.warning("tune_lightgbm.trial_failed", error=str(exc))
                return 0.0

        study = self._make_study("lightgbm_tuning")
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=self.config.n_jobs,
            show_progress_bar=False,
        )
        best_sharpe = -study.best_value  # objective returns -sharpe + penalty
        self._log.info(
            "tune_lightgbm.done",
            best_value=study.best_value,
            best_sharpe=best_sharpe,
            best_params=study.best_params,
        )
        if best_sharpe < 0.3:
            self._log.error(
                "tune_lightgbm.best_sharpe_below_threshold",
                best_sharpe=best_sharpe,
                threshold=0.3,
            )
            raise RuntimeError(
                f"Optuna best Sharpe {best_sharpe:.3f} < 0.3 — aborting param write"
            )
        best = study.best_params
        return best

    # ------------------------------------------------------------------
    # CatBoost tuning (private helper used by tune_ensemble)
    # ------------------------------------------------------------------

    def _tune_catboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Tune CatBoost hyperparameters using Optuna.

        Search space (per CDR-3B-C23):
            - depth: [3, 8]
            - learning_rate: [0.01, 0.3] (log scale)
            - iterations: [100, 500]
            - l2_leaf_reg: [1, 10]

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Validation feature matrix (must be fold[-2]).
            y_val: Validation labels.
            sample_weights: Optional per-sample weights for training.

        Returns:
            Dictionary of best CatBoost hyperparameters found within n_trials.
        """
        self._check_safety()
        self._log.info("_tune_catboost.start", n_trials=self.config.n_trials)

        def objective(trial: optuna.Trial) -> float:
            try:
                params: dict[str, Any] = {
                    "depth": trial.suggest_int("depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "iterations": trial.suggest_int("iterations", 100, 500),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                    "loss_function": "MultiClass",
                    "task_type": "CPU",
                    "verbose": False,
                    "random_seed": self.config.random_seed,
                }
                model = CatBoostClassifier(**params)
                fit_kwargs: dict[str, Any] = {
                    "eval_set": (X_val, y_val),
                    "early_stopping_rounds": 10,
                }
                if sample_weights is not None:
                    fit_kwargs["sample_weight"] = sample_weights
                model.fit(X_train, y_train, **fit_kwargs)
                proba = model.predict_proba(X_val)
                sharpe = self._compute_sharpe(y_val, proba)
                penalty = _complexity_penalty(params)
                return -sharpe + penalty * 0.1
            except Exception as exc:
                self._log.warning("_tune_catboost.trial_failed", error=str(exc))
                return 0.0

        study = self._make_study("catboost_tuning")
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=self.config.n_jobs,
            show_progress_bar=False,
        )
        best_sharpe = -study.best_value  # objective returns -sharpe + penalty
        self._log.info(
            "_tune_catboost.done",
            best_value=study.best_value,
            best_sharpe=best_sharpe,
            best_params=study.best_params,
        )
        if best_sharpe < 0.3:
            self._log.error(
                "_tune_catboost.best_sharpe_below_threshold",
                best_sharpe=best_sharpe,
                threshold=0.3,
            )
            raise RuntimeError(
                f"Optuna best Sharpe {best_sharpe:.3f} < 0.3 — aborting param write"
            )
        best = study.best_params
        return best

    # ------------------------------------------------------------------
    # Ensemble tuning (public entry point)
    # ------------------------------------------------------------------

    def tune_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
        sample_weights: np.ndarray | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Tune XGBoost, LightGBM, and CatBoost together using walk-forward folds.

        Uses fold[-2] for optimisation.  fold[-1] is the untouched final
        holdout and is NEVER accessed in this method.

        Args:
            X: Full feature matrix (all samples).
            y: Full label array (all samples).
            folds: List of (train_indices, val_indices) produced by
                WalkForwardSplitter.  Must have at least 2 elements so that
                fold[-2] can be used.
            sample_weights: Optional per-sample weights aligned to ``y``.

        Returns:
            Dictionary with keys ``"xgb"``, ``"lgb"``, ``"cat"`` each
            mapping to the respective best hyperparameter dict.

        Raises:
            RuntimeError: If is_safe_to_retrain() returns False (not Sunday).
            ValueError: If fewer than 2 folds are provided (fold[-2] undefined).
        """
        self._check_safety()

        if len(folds) < 2:
            raise ValueError(
                "tune_ensemble requires at least 2 folds so fold[-2] can be "
                "used for optimisation while fold[-1] remains untouched."
            )

        # Use fold[-2]: the second-to-last fold only.
        # fold[-1] is the final holdout — never seen here.
        train_idx, val_idx = folds[-2]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        sw_train = sample_weights[train_idx] if sample_weights is not None else None

        self._log.info(
            "tune_ensemble.start",
            fold_used="fold[-2]",
            n_train=len(train_idx),
            n_val=len(val_idx),
            n_folds_total=len(folds),
        )

        xgb_params = self.tune_xgboost(X_train, y_train, X_val, y_val, sw_train)
        lgb_params = self.tune_lightgbm(X_train, y_train, X_val, y_val, sw_train)
        cat_params = self._tune_catboost(X_train, y_train, X_val, y_val, sw_train)

        result: dict[str, dict[str, Any]] = {
            "xgb": xgb_params,
            "lgb": lgb_params,
            "cat": cat_params,
        }
        self._log.info("tune_ensemble.done", result=result)
        return result
