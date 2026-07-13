"""Full ML training pipeline for NSE AlgoTrader.

Spec reference: CDR-P5-STACK.
XGBoost + LightGBM + CatBoost base models, stacked with a LogisticRegression
meta-learner trained on out-of-fold (OOF) predictions from walk-forward folds.
After training, Treelite ``.so`` files are compiled automatically.

CPU-only constraints (non-negotiable):
- XGBoost: ``tree_method="hist"``, ``device="cpu"``
- LightGBM: ``device="cpu"``
- CatBoost: ``task_type="CPU"``
"""

from __future__ import annotations

import json
import pickle
import platform
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import importlib.metadata as importlib_metadata

import numpy as np
import structlog
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from algo_trading.models.model_registry import FrozenModelSpec

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Hard-coded training hyperparameters (arch-spec.json — do not modify)
# ---------------------------------------------------------------------------

XGB_PARAMS: dict[str, Any] = {
    "tree_method": "hist",       # REQUIRED — CPU hist method
    "device": "cpu",             # REQUIRED
    "n_jobs": 4,
    "n_estimators": 300,
    "max_depth": 8,              # deeper tree for diversity
    "min_child_weight": 3,       # diversity-critical
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,              # {-1, 0, +1} mapped to {0, 1, 2}
    "eval_metric": "mlogloss",
}

LGB_PARAMS: dict[str, Any] = {
    "device": "cpu",             # REQUIRED
    "num_threads": 4,
    "n_estimators": 300,
    "num_leaves": 127,           # more leaves for diversity
    "min_child_samples": 20,     # diversity-critical
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "num_class": 3,
    "verbose": -1,
}

CAT_PARAMS: dict[str, Any] = {
    "task_type": "CPU",          # REQUIRED
    "thread_count": 4,
    "iterations": 300,
    "depth": 5,                  # shallower tree for diversity
    "learning_rate": 0.03,       # lower learning rate for diversity
    "loss_function": "MultiClass",
    "classes_count": 3,
    "verbose": False,
}

# Signal mapping: model class index → trading signal
_CLASS_TO_SIGNAL: dict[int, int] = {0: -1, 1: 0, 2: 1}
_SIGNAL_TO_CLASS: dict[int, int] = {-1: 0, 0: 1, 1: 2}

# Number of classes
_N_CLASSES: int = 3


# ---------------------------------------------------------------------------
# Treelite compilation
# ---------------------------------------------------------------------------


def compile_for_production(
    xgb_model: xgb.XGBClassifier,
    lgb_model: lgb.LGBMClassifier,
    cat_model: CatBoostClassifier,
    output_dir: Path,
) -> tuple[str, str, str]:
    """Compile trained base models to Treelite ``.so`` shared libraries.

    The compiled artefacts enable low-latency inference without the Python
    overhead of the original model objects.  Compilation order:
    1. XGBoost  → ``xgb_compiled.so``
    2. LightGBM → ``lgb_compiled.so``
    3. CatBoost → ``cat_compiled.so`` (exported to JSON first)

    Args:
        xgb_model: Trained XGBoost classifier (CPU mode).
        lgb_model: Trained LightGBM classifier (CPU mode).
        cat_model: Trained CatBoost classifier (CPU mode).
        output_dir: Directory where ``.so`` files are written.

    Returns:
        Tuple of ``(xgb_so_path, lgb_so_path, cat_so_path)`` as strings.

    Raises:
        ImportError: If ``treelite`` is not installed.
        RuntimeError: If compilation of any model fails.
    """
    try:
        import treelite  # type: ignore[import]
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "treelite_unavailable",
            error=str(exc),
            note="Falling back to sklearn/xgboost native predict",
        )
        raise ImportError(
            "treelite is required for production compilation. "
            "Install with: pip install treelite treelite_runtime"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    # Shared library extension is platform-dependent
    so_ext = ".dll" if platform.system() == "Windows" else ".so"

    # --- XGBoost ---
    xgb_so_path = str(output_dir / f"xgb_compiled{so_ext}")
    try:
        # Treelite 3.x: use treelite.frontend.from_xgboost if available,
        # fall back to treelite.Model.from_xgboost for older 3.x builds.
        try:
            tl_xgb = treelite.frontend.from_xgboost(xgb_model.get_booster())
        except AttributeError:
            tl_xgb = treelite.Model.from_xgboost(xgb_model.get_booster())
        tl_xgb.export_lib(
            toolchain="gcc",
            libpath=xgb_so_path,
            params={"parallel_comp": 4},
            verbose=False,
        )
        logger.info("treelite_xgb_compiled", path=xgb_so_path)
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "treelite_unavailable",
            error=str(exc),
            note="Falling back to sklearn/xgboost native predict",
        )
        raise RuntimeError(f"XGBoost Treelite compilation failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"XGBoost Treelite compilation failed: {exc}") from exc

    # --- LightGBM ---
    lgb_so_path = str(output_dir / f"lgb_compiled{so_ext}")
    try:
        # Treelite 3.x: use treelite.frontend.from_lightgbm if available,
        # fall back to treelite.Model.from_lightgbm for older 3.x builds.
        try:
            tl_lgb = treelite.frontend.from_lightgbm(lgb_model.booster_)
        except AttributeError:
            tl_lgb = treelite.Model.from_lightgbm(lgb_model.booster_)
        tl_lgb.export_lib(
            toolchain="gcc",
            libpath=lgb_so_path,
            params={"parallel_comp": 4},
            verbose=False,
        )
        logger.info("treelite_lgb_compiled", path=lgb_so_path)
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "treelite_unavailable",
            error=str(exc),
            note="Falling back to sklearn/xgboost native predict",
        )
        raise RuntimeError(f"LightGBM Treelite compilation failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"LightGBM Treelite compilation failed: {exc}") from exc

    # --- CatBoost (export to JSON, then load via Treelite) ---
    cat_json_path = output_dir / "cat_model.json"
    cat_so_path = str(output_dir / f"cat_compiled{so_ext}")
    try:
        cat_model.save_model(str(cat_json_path), format="json")
        # Treelite 3.x: use treelite.frontend.load if available,
        # fall back to treelite.Model.load for older 3.x builds.
        try:
            tl_cat = treelite.frontend.load(str(cat_json_path), model_format="catboost")
        except AttributeError:
            tl_cat = treelite.Model.load(str(cat_json_path), model_format="catboost")
        tl_cat.export_lib(
            toolchain="gcc",
            libpath=cat_so_path,
            params={"parallel_comp": 4},
            verbose=False,
        )
        logger.info("treelite_cat_compiled", path=cat_so_path)
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "treelite_unavailable",
            error=str(exc),
            note="Falling back to sklearn/xgboost native predict",
        )
        raise RuntimeError(f"CatBoost Treelite compilation failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"CatBoost Treelite compilation failed: {exc}") from exc

    return xgb_so_path, lgb_so_path, cat_so_path


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters that control walk-forward training.

    Attributes:
        n_folds: Number of walk-forward folds for OOF generation.
        early_stopping_rounds: Early-stopping patience for XGB/LGB.
        eval_split_pct: Fraction of each training fold held out for
            early-stopping evaluation (applied chronologically).
    """

    n_folds: int = 5
    early_stopping_rounds: int = 50
    eval_split_pct: float = 0.1


# ---------------------------------------------------------------------------
# Regime-conditioned meta-learner
# ---------------------------------------------------------------------------


class RegimeConditionedMeta:
    """Per-regime meta-learner weights.

    Trains 3 separate LogisticRegression meta-learners (bull/bear/sideways)
    on OOF predictions filtered by HMM regime label.  Falls back to the
    global meta-learner if a regime bucket has fewer than 200 samples.

    Regime label convention: 0 = bear, 1 = sideways, 2 = bull.
    """

    _MIN_SAMPLES_PER_REGIME: int = 200

    def __init__(self) -> None:
        # regime_idx → fitted LR
        self._regime_models: dict[int, LogisticRegression] = {}
        self._global_model: LogisticRegression | None = None

    def fit(
        self,
        meta_features: np.ndarray,
        y_mapped: np.ndarray,
        regime_labels: np.ndarray,
    ) -> None:
        """Fit per-regime LR models plus one global fallback.

        Args:
            meta_features: Stacked OOF probabilities, shape ``(n, 9)``.
            y_mapped: Integer labels ``{0, 1, 2}``, shape ``(n,)``.
            regime_labels: HMM regime integer labels ``{0, 1, 2}``, shape ``(n,)``.
        """
        # Global fallback
        self._global_model = LogisticRegression(C=1.0, max_iter=1000, n_jobs=4)
        self._global_model.fit(meta_features, y_mapped)
        logger.info("regime_meta.global_fitted", n_samples=len(y_mapped))

        for regime in range(3):
            mask = regime_labels == regime
            n_regime = int(mask.sum())
            if n_regime < self._MIN_SAMPLES_PER_REGIME:
                logger.warning(
                    "regime_meta.insufficient_samples",
                    regime=regime,
                    n_samples=n_regime,
                    min_required=self._MIN_SAMPLES_PER_REGIME,
                    note="Falling back to global meta for this regime",
                )
                continue
            lr = LogisticRegression(C=1.0, max_iter=1000, n_jobs=4)
            lr.fit(meta_features[mask], y_mapped[mask])
            self._regime_models[regime] = lr
            logger.info(
                "regime_meta.regime_fitted",
                regime=regime,
                n_samples=n_regime,
            )

    def predict_proba(
        self,
        meta_features: np.ndarray,
        regime_label: int,
    ) -> np.ndarray:
        """Return class probabilities using the appropriate regime model.

        Args:
            meta_features: Feature matrix, shape ``(n, 9)``.
            regime_label: HMM regime integer label for the current bar.

        Returns:
            Probability matrix, shape ``(n, 3)``.
        """
        model = self._regime_models.get(regime_label, self._global_model)
        if model is None:
            raise RuntimeError("RegimeConditionedMeta has not been fitted.")
        return model.predict_proba(meta_features)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


class MLEnsemble:
    """XGBoost + LightGBM + CatBoost stacked with a LogisticRegression meta-learner.

    Walk-forward OOF predictions from each base model are used as features for
    the meta-learner (spec CDR-P5-STACK).  The last fold is reserved as an
    untouched holdout for final reported metrics.  Production artefacts are
    compiled to Treelite ``.so`` files automatically.

    Attributes:
        config: :class:`TrainingConfig` instance.
        MIN_CALIBRATION_SAMPLES: Minimum calibration-split samples required to
            fit isotonic calibration.  Calibration is skipped with a WARNING
            log if the split has fewer than this many samples.
    """

    MIN_CALIBRATION_SAMPLES: int = 10

    def __init__(self, config: TrainingConfig = TrainingConfig()) -> None:
        """Initialise the ensemble with training configuration.

        Args:
            config: Walk-forward training configuration.
        """
        self.config = config
        self._xgb_model: xgb.XGBClassifier | None = None
        self._lgb_model: lgb.LGBMClassifier | None = None
        self._cat_model: CatBoostClassifier | None = None
        self._meta_lr: LogisticRegression | None = None
        self._meta_lr_calibrated: CalibratedClassifierCV | None = None
        self._regime_meta: RegimeConditionedMeta | None = None
        self._feature_names: list[str] = []
        # Per-fold feature importances: model_type → list of importance arrays
        self._fold_importances: dict[str, list[np.ndarray]] = {
            "xgb": [], "lgb": [], "cat": []
        }
        logger.info("ensemble_initialised", config=config)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        feature_names: list[str],
        output_dir: Path,
        bar_type: str = "time_15min",
        bar_spec: dict[str, Any] | None = None,
        symbol_universe: tuple[str, ...] = (),
        version: str = "v1.0.0",
        regime_labels: np.ndarray | None = None,
    ) -> FrozenModelSpec:
        """Run the full walk-forward training pipeline.

        Steps:
        1. Generate OOF predictions for XGB, LGB, and CatBoost using
           :meth:`_walk_forward_oof` (temporal folds — no shuffle).
           The last fold is reserved as holdout; OOF is only generated on
           the first (n_folds - 1) folds.
        2. Train meta LogisticRegression on OOF predictions and apply
           isotonic calibration.
        3. Retrain base models on folds[0..n-2] data only (not holdout fold).
        4. Evaluate on holdout fold (fold[-1]) for final reported metrics.
        5. Call :func:`compile_for_production` to produce ``.so`` files.
        6. Persist meta LR as ``meta_lr.pkl``.
        7. Build and return a :class:`FrozenModelSpec` with all metadata.

        Labels must be in ``{-1, 0, +1}``; they are mapped internally to
        ``{0, 1, 2}`` for multi-class classifiers.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``,
                ordered chronologically. dtype should be ``float32`` or
                ``float64``.
            y: Label array of shape ``(n_samples,)`` with values in
                ``{-1, 0, +1}``.
            sample_weights: Per-sample weight array of shape ``(n_samples,)``.
            feature_names: List of feature column names matching ``X``
                column order.
            output_dir: Directory for compiled artefacts and meta model.
            bar_type: Bar type string, propagated to :class:`FrozenModelSpec`.
            bar_spec: Bar specification dict, propagated to spec.
            symbol_universe: Tuple of symbol strings the model covers.
            version: Semantic version string for the resulting spec.
            regime_labels: Optional HMM regime integer labels ``{0, 1, 2}``
                of shape ``(n_samples,)``.  When provided, a
                :class:`RegimeConditionedMeta` is also trained.

        Returns:
            A :class:`FrozenModelSpec` ready for registration and deployment.
        """
        if bar_spec is None:
            bar_spec = {}

        n_samples, n_features = X.shape
        if n_features != len(feature_names):
            raise ValueError(
                f"X has {n_features} columns but {len(feature_names)} feature names provided."
            )

        self._feature_names = feature_names
        # Map {-1, 0, +1} → {0, 1, 2}
        y_mapped = np.array([_SIGNAL_TO_CLASS[int(lbl)] for lbl in y], dtype=np.int32)

        logger.info(
            "ensemble_training_started",
            n_samples=n_samples,
            n_features=n_features,
            n_folds=self.config.n_folds,
            version=version,
            holdout_fold_reserved=True,
        )

        # Identify holdout fold boundaries before any OOF work.
        tscv = TimeSeriesSplit(n_splits=self.config.n_folds)
        all_splits = list(tscv.split(X))
        holdout_train_idx, holdout_test_idx = all_splits[-1]

        # ------------------------------------------------------------------
        # Step 1: Walk-forward OOF generation (folds[:-1] only)
        # ------------------------------------------------------------------
        self._fold_importances = {"xgb": [], "lgb": [], "cat": []}

        oof_xgb = self._walk_forward_oof(
            X, y_mapped, sample_weights, model_type="xgb", holdout_fold_reserve=True
        )
        oof_lgb = self._walk_forward_oof(
            X, y_mapped, sample_weights, model_type="lgb", holdout_fold_reserve=True
        )
        oof_cat = self._walk_forward_oof(
            X, y_mapped, sample_weights, model_type="cat", holdout_fold_reserve=True
        )

        # Stack OOF probabilities: shape (n_samples, 9)
        meta_features = np.hstack([oof_xgb, oof_lgb, oof_cat])

        # OOF rows that were actually predicted (folds[:-1] val sets)
        # Rows that belong to the holdout fold test set should be masked out
        # from meta-learner training.
        oof_mask = np.ones(n_samples, dtype=bool)
        oof_mask[holdout_test_idx] = False
        # Also mask rows that were never a val_idx in folds[:-1]
        # (the very first training portion of fold 0) — these keep uniform proba.
        # We identify them: a row is "predicted" if it appeared in any val_idx
        # across folds[:-1].
        predicted_mask = np.zeros(n_samples, dtype=bool)
        for fold_idx, (_, val_idx) in enumerate(all_splits[:-1]):
            predicted_mask[val_idx] = True
        # Meta-learner trains only on rows that were both predicted AND not holdout
        train_meta_mask = predicted_mask & oof_mask
        meta_train_features = meta_features[train_meta_mask]
        meta_train_y = y_mapped[train_meta_mask]
        meta_train_sw = sample_weights[train_meta_mask]

        # ------------------------------------------------------------------
        # Step 2: Fit meta-learner on OOF predictions (folds[:-1])
        # ------------------------------------------------------------------
        self._meta_lr = LogisticRegression(
            C=1.0, max_iter=1000, n_jobs=4
        )
        self._meta_lr.fit(meta_train_features, meta_train_y, sample_weight=meta_train_sw)
        logger.info("meta_lr_fitted", n_meta_train=int(train_meta_mask.sum()))

        # Isotonic calibration on the last 20% of OOF training rows
        cal_cut = int(len(meta_train_features) * 0.8)
        n_cal_samples = len(meta_train_features) - cal_cut
        if n_cal_samples >= self.MIN_CALIBRATION_SAMPLES:
            self._meta_lr_calibrated = CalibratedClassifierCV(
                self._meta_lr, method="isotonic", cv="prefit"
            )
            self._meta_lr_calibrated.fit(
                meta_train_features[cal_cut:], meta_train_y[cal_cut:]
            )
            logger.info(
                "meta_lr_calibrated",
                calibration_samples=n_cal_samples,
            )
        else:
            logger.warning(
                "ensemble.calibration_skipped",
                n_samples=n_cal_samples,
                threshold=self.MIN_CALIBRATION_SAMPLES,
            )
            self._meta_lr_calibrated = None

        # Regime-conditioned meta (optional)
        if regime_labels is not None:
            rl = regime_labels[train_meta_mask]
            self._regime_meta = RegimeConditionedMeta()
            self._regime_meta.fit(meta_train_features, meta_train_y, rl)
        else:
            self._regime_meta = None

        # ------------------------------------------------------------------
        # Step 3: Retrain base models on folds[0..n-2] data only
        # ------------------------------------------------------------------
        self._xgb_model = self._build_xgb()
        self._lgb_model = self._build_lgb()
        self._cat_model = self._build_cat()

        # Use all data up to (but not including) holdout_test_idx
        retrain_end = int(holdout_train_idx[-1]) + 1  # last bar of holdout train set
        X_retrain = X[:retrain_end]
        y_retrain = y_mapped[:retrain_end]
        sw_retrain = sample_weights[:retrain_end]

        retrain_eval_cut = int(retrain_end * (1.0 - self.config.eval_split_pct))
        X_tr, X_ev = X_retrain[:retrain_eval_cut], X_retrain[retrain_eval_cut:]
        y_tr, y_ev = y_retrain[:retrain_eval_cut], y_retrain[retrain_eval_cut:]
        sw_tr = sw_retrain[:retrain_eval_cut]

        self._xgb_model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_ev, y_ev)],
            verbose=False,
        )
        self._lgb_model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_ev, y_ev)],
            callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        self._cat_model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=(X_ev, y_ev),
        )
        logger.info(
            "base_models_retrained_on_non_holdout_data",
            retrain_end=retrain_end,
        )

        # ------------------------------------------------------------------
        # Step 4: Holdout evaluation metrics
        # ------------------------------------------------------------------
        X_holdout = X[holdout_test_idx]
        y_holdout = y_mapped[holdout_test_idx]

        xgb_h = self._xgb_model.predict_proba(X_holdout)
        lgb_h = self._lgb_model.predict_proba(X_holdout)
        cat_h = self._cat_model.predict_proba(X_holdout)
        holdout_meta = np.hstack([xgb_h, lgb_h, cat_h])
        holdout_proba = self._meta_lr.predict_proba(holdout_meta)
        holdout_preds = np.argmax(holdout_proba, axis=1)
        holdout_accuracy = float(np.mean(holdout_preds == y_holdout))

        # Sharpe on holdout: position = (pred_class - 1) mapped back to {-1, 0, +1}
        # We compute returns as sign(position) * bar_return; use ±1 for direction.
        # Since we have no actual bar returns here, we use label correctness proxy:
        # return_i = (pred_signal == true_signal) ? +1 : -1
        holdout_signals = np.array([_CLASS_TO_SIGNAL[int(p)] for p in holdout_preds])
        holdout_true_signals = np.array([_CLASS_TO_SIGNAL[int(t)] for t in y_holdout])
        # Directional return: +1 if correct direction, -1 if wrong, 0 if neutral pred
        directional_returns = np.where(
            holdout_signals == 0,
            0.0,
            np.where(holdout_signals == holdout_true_signals, 1.0, -1.0),
        )
        active_returns = directional_returns[holdout_signals != 0]
        if len(active_returns) > 1 and active_returns.std() > 0:
            holdout_sharpe = float(
                (active_returns.mean() / active_returns.std()) * np.sqrt(252)
            )
        else:
            holdout_sharpe = 0.0

        logger.info(
            "holdout_evaluation_complete",
            holdout_accuracy=holdout_accuracy,
            holdout_sharpe=holdout_sharpe,
            holdout_n=len(holdout_test_idx),
        )

        # ------------------------------------------------------------------
        # Step 5: Feature importance stability tracking
        # ------------------------------------------------------------------
        unstable_features_count, feature_importance_stability = (
            self._compute_feature_importance_stability(feature_names)
        )

        # ------------------------------------------------------------------
        # Step 6: Compile to Treelite .so
        # ------------------------------------------------------------------
        output_dir.mkdir(parents=True, exist_ok=True)
        xgb_so, lgb_so, cat_so = compile_for_production(
            self._xgb_model, self._lgb_model, self._cat_model, output_dir
        )

        # ------------------------------------------------------------------
        # Step 7: Persist meta LR (and calibrated version)
        # ------------------------------------------------------------------
        meta_lr_path = output_dir / "meta_lr.pkl"
        with meta_lr_path.open("wb") as fh:
            pickle.dump(self._meta_lr, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("meta_lr_saved", path=str(meta_lr_path))

        if self._meta_lr_calibrated is not None:
            cal_path = output_dir / "meta_lr_calibrated.pkl"
            with cal_path.open("wb") as fh:
                pickle.dump(self._meta_lr_calibrated, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("meta_lr_calibrated_saved", path=str(cal_path))

        if self._regime_meta is not None:
            regime_path = output_dir / "regime_meta.pkl"
            with regime_path.open("wb") as fh:
                pickle.dump(self._regime_meta, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("regime_meta_saved", path=str(regime_path))

        # Save OOF predictions for stability analysis tooling
        oof_save_path = output_dir / f"oof_{version}.npz"
        np.savez_compressed(
            str(oof_save_path),
            oof_xgb=oof_xgb,
            oof_lgb=oof_lgb,
            oof_cat=oof_cat,
            meta_features=meta_features,
            y_mapped=y_mapped,
            train_meta_mask=train_meta_mask,
            holdout_test_idx=holdout_test_idx,
        )
        logger.info("oof_predictions_saved", path=str(oof_save_path))

        # ------------------------------------------------------------------
        # Step 8: Build FrozenModelSpec
        # ------------------------------------------------------------------
        feature_tuple = tuple(feature_names)
        schema_hash = FrozenModelSpec.compute_feature_hash(feature_tuple)

        package_versions: dict[str, str] = {}
        for pkg in ("xgboost", "lightgbm", "catboost", "treelite", "scikit-learn", "numpy"):
            try:
                package_versions[pkg] = importlib_metadata.version(pkg)
            except importlib_metadata.PackageNotFoundError:
                package_versions[pkg] = "unknown"

        # Collect validation metrics
        oof_preds = np.argmax(self._meta_lr.predict_proba(meta_train_features), axis=1)
        oof_accuracy = float(np.mean(oof_preds == meta_train_y))
        oof_holdout_gap = abs(oof_accuracy - holdout_accuracy)

        validation_metrics: dict[str, float] = {
            "oof_accuracy": oof_accuracy,
            "holdout_accuracy": holdout_accuracy,
            "holdout_sharpe": holdout_sharpe,
            "oof_holdout_gap": oof_holdout_gap,
            "unstable_features_count": float(unstable_features_count),
        }

        spec = FrozenModelSpec(
            version=version,
            feature_names=feature_tuple,
            feature_schema_hash=schema_hash,
            bar_type=bar_type,
            bar_spec=bar_spec,
            trained_on=date.today(),
            symbol_universe=symbol_universe,
            validation_metrics=validation_metrics,
            treelite_xgb_path=xgb_so,
            treelite_lgb_path=lgb_so,
            treelite_cat_path=cat_so,
            meta_lr_path=str(meta_lr_path),
            package_versions=package_versions,
            approval_status="pending",
            holdout_sharpe=holdout_sharpe,
            holdout_accuracy=holdout_accuracy,
            unstable_features_count=unstable_features_count,
        )
        logger.info(
            "ensemble_training_complete",
            version=version,
            oof_accuracy=oof_accuracy,
            holdout_accuracy=holdout_accuracy,
            holdout_sharpe=holdout_sharpe,
            oof_holdout_gap=oof_holdout_gap,
            unstable_features_count=unstable_features_count,
            xgb_so=xgb_so,
            lgb_so=lgb_so,
            cat_so=cat_so,
        )
        return spec

    # ------------------------------------------------------------------
    # Inference (in-memory — research/training only)
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from the stacked ensemble.

        This method uses the in-memory Python model objects and is intended
        for training-time research only.  Use :class:`EnsembleInferenceEngine`
        for production inference.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Probability matrix of shape ``(n_samples, 3)`` where columns
            correspond to classes ``{-1, 0, +1}`` (mapped as ``{0, 1, 2}``).

        Raises:
            RuntimeError: If the ensemble has not been trained yet.
        """
        self._assert_trained()
        xgb_proba = self._xgb_model.predict_proba(X)   # type: ignore[union-attr]
        lgb_proba = self._lgb_model.predict_proba(X)   # type: ignore[union-attr]
        cat_proba = self._cat_model.predict_proba(X)   # type: ignore[union-attr]
        meta_features = np.hstack([xgb_proba, lgb_proba, cat_proba])
        # Prefer calibrated meta-learner when available
        predictor = self._meta_lr_calibrated or self._meta_lr
        return predictor.predict_proba(meta_features)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def compute_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Compute average normalised feature importance across the 3 base models.

        XGBoost and LightGBM expose ``feature_importances_`` (gain-based).
        CatBoost exposes ``get_feature_importance()``.  All three are
        normalised to sum to 1 before averaging.

        Args:
            feature_names: Feature names matching the columns of the training
                matrix.

        Returns:
            Dict mapping each feature name to its average normalised importance
            across the three base models.

        Raises:
            RuntimeError: If the ensemble has not been trained yet.
        """
        self._assert_trained()

        def _normalise(arr: np.ndarray) -> np.ndarray:
            total = arr.sum()
            return arr / total if total > 0 else arr

        xgb_imp = _normalise(
            np.array(self._xgb_model.feature_importances_, dtype=np.float64)  # type: ignore[union-attr]
        )
        lgb_imp = _normalise(
            np.array(self._lgb_model.feature_importances_, dtype=np.float64)  # type: ignore[union-attr]
        )
        cat_imp = _normalise(
            np.array(
                self._cat_model.get_feature_importance(), dtype=np.float64  # type: ignore[union-attr]
            )
        )
        avg_imp = (xgb_imp + lgb_imp + cat_imp) / 3.0

        importance_dict: dict[str, float] = {
            name: float(score) for name, score in zip(feature_names, avg_imp)
        }
        logger.info("feature_importance_computed", top_feature=max(importance_dict, key=importance_dict.get))  # type: ignore[arg-type]
        return importance_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _walk_forward_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        model_type: str,
        holdout_fold_reserve: bool = True,
    ) -> np.ndarray:
        """Generate OOF probability predictions using temporal walk-forward folds.

        Never shuffles; always preserves chronological ordering.  Uses
        :class:`sklearn.model_selection.TimeSeriesSplit` to guarantee that
        validation bars always come after training bars.

        When ``holdout_fold_reserve=True``, the last fold is skipped and its
        rows are never used as a validation set during OOF generation.  This
        prevents the meta-learner from training on predictions that overlap
        with the final holdout evaluation set.

        Per-fold feature importances are accumulated into
        ``self._fold_importances[model_type]`` for stability analysis.

        Args:
            X: Feature matrix (chronological order).
            y: Integer label array mapped to ``{0, 1, 2}``.
            sample_weights: Per-sample weights.
            model_type: One of ``"xgb"``, ``"lgb"``, ``"cat"``.
            holdout_fold_reserve: If True, skip the last fold.

        Returns:
            OOF probability matrix of shape ``(n_samples, 3)``.  Rows
            corresponding to the first fold's training set (which never
            appears in any validation split) and holdout fold test rows are
            filled with uniform probability ``1/3``.
        """
        n_samples = X.shape[0]
        oof = np.full((n_samples, _N_CLASSES), 1.0 / _N_CLASSES, dtype=np.float64)

        tscv = TimeSeriesSplit(n_splits=self.config.n_folds)
        all_splits = list(tscv.split(X))

        # Reserve last fold if requested
        active_splits = all_splits[:-1] if holdout_fold_reserve else all_splits
        n_active = len(active_splits)

        for fold_idx, (train_idx, val_idx) in enumerate(active_splits):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            sw_tr = sample_weights[train_idx]

            # Carve an eval split from the end of the training portion
            eval_cut = int(len(train_idx) * (1.0 - self.config.eval_split_pct))
            X_tr_fit, X_tr_ev = X_tr[:eval_cut], X_tr[eval_cut:]
            y_tr_fit, y_tr_ev = y_tr[:eval_cut], y_tr[eval_cut:]
            sw_fit = sw_tr[:eval_cut]

            model = self._build_and_fit(
                model_type, X_tr_fit, y_tr_fit, sw_fit, X_tr_ev, y_tr_ev, fold_idx
            )
            oof[val_idx] = self._predict_proba_raw(model, model_type, X_val)

            # Accumulate feature importances for this fold
            fold_imp = self._extract_feature_importance(model, model_type)
            if fold_imp is not None:
                self._fold_importances[model_type].append(fold_imp)

            logger.debug(
                "oof_fold_complete",
                model=model_type,
                fold=fold_idx,
                train_size=len(train_idx),
                val_size=len(val_idx),
                holdout_reserved=holdout_fold_reserve,
                active_folds=n_active,
            )

        return oof

    def _extract_feature_importance(
        self, model: Any, model_type: str
    ) -> np.ndarray | None:
        """Extract normalised feature importance array from a fitted model.

        Args:
            model: Fitted model object.
            model_type: ``"xgb"``, ``"lgb"``, or ``"cat"``.

        Returns:
            Normalised importance array or None on failure.
        """
        try:
            if model_type in ("xgb", "lgb"):
                raw = np.array(model.feature_importances_, dtype=np.float64)
            elif model_type == "cat":
                raw = np.array(model.get_feature_importance(), dtype=np.float64)
            else:
                return None
            total = raw.sum()
            return raw / total if total > 0 else raw
        except Exception as exc:
            logger.debug(
                "feature_importance_extraction_failed",
                model_type=model_type,
                error=str(exc),
            )
            return None

    def _compute_feature_importance_stability(
        self, feature_names: list[str]
    ) -> tuple[int, dict[str, float]]:
        """Compute cross-fold feature importance stability statistics.

        For each feature, computes the standard deviation of importance across
        folds (pooled across all three model types).  Features where
        ``std > 0.5 * mean`` are flagged as unstable.

        Args:
            feature_names: Ordered list of feature names.

        Returns:
            Tuple of (unstable_count, stability_dict) where stability_dict
            maps feature_name → std_deviation_across_folds.
        """
        # Pool importances across models for each fold
        all_fold_imps: list[np.ndarray] = []
        for model_type in ("xgb", "lgb", "cat"):
            all_fold_imps.extend(self._fold_importances.get(model_type, []))

        if len(all_fold_imps) < 2:
            logger.warning(
                "feature_importance_stability_skipped",
                reason="Fewer than 2 fold importance arrays available",
            )
            return 0, {}

        stacked = np.vstack(all_fold_imps)  # shape: (n_fold_records, n_features)
        means = stacked.mean(axis=0)
        stds = stacked.std(axis=0)

        stability_dict: dict[str, float] = {}
        unstable_count = 0
        for i, name in enumerate(feature_names):
            if i >= stds.shape[0]:
                break
            std_val = float(stds[i])
            mean_val = float(means[i])
            stability_dict[name] = std_val
            if mean_val > 0 and std_val > 0.5 * mean_val:
                unstable_count += 1
                logger.warning(
                    "unstable_feature_detected",
                    feature=name,
                    mean_importance=mean_val,
                    std_importance=std_val,
                    ratio=std_val / mean_val,
                )

        logger.info(
            "feature_importance_stability_computed",
            unstable_count=unstable_count,
            total_features=len(feature_names),
        )
        return unstable_count, stability_dict

    def _build_and_fit(
        self,
        model_type: str,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        sw_tr: np.ndarray,
        X_ev: np.ndarray,
        y_ev: np.ndarray,
        fold_idx: int,
    ) -> Any:
        """Construct and fit a single base model for one fold.

        Args:
            model_type: ``"xgb"``, ``"lgb"``, or ``"cat"``.
            X_tr: Training features.
            y_tr: Training labels.
            sw_tr: Training sample weights.
            X_ev: Early-stopping evaluation features.
            y_ev: Early-stopping evaluation labels.
            fold_idx: Fold index, used only for logging.

        Returns:
            Fitted model object.
        """
        if model_type == "xgb":
            m = self._build_xgb()
            m.fit(
                X_tr, y_tr,
                sample_weight=sw_tr,
                eval_set=[(X_ev, y_ev)],
                verbose=False,
            )
            return m
        if model_type == "lgb":
            m = self._build_lgb()
            m.fit(
                X_tr, y_tr,
                sample_weight=sw_tr,
                eval_set=[(X_ev, y_ev)],
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            return m
        if model_type == "cat":
            m = self._build_cat()
            m.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=(X_ev, y_ev))
            return m
        raise ValueError(f"Unknown model_type '{model_type}'.")

    @staticmethod
    def _predict_proba_raw(model: Any, model_type: str, X: np.ndarray) -> np.ndarray:
        """Return probability predictions from a fitted model.

        Args:
            model: Fitted model with a ``predict_proba``-compatible interface.
            model_type: ``"xgb"``, ``"lgb"``, or ``"cat"``.
            X: Feature matrix.

        Returns:
            Probability matrix of shape ``(n_samples, 3)``.
        """
        return model.predict_proba(X)

    @staticmethod
    def _build_xgb() -> xgb.XGBClassifier:
        """Instantiate an XGBClassifier with the canonical CPU params.

        Returns:
            Unfitted :class:`xgboost.XGBClassifier`.
        """
        return xgb.XGBClassifier(**XGB_PARAMS)

    @staticmethod
    def _build_lgb() -> lgb.LGBMClassifier:
        """Instantiate an LGBMClassifier with the canonical CPU params.

        Returns:
            Unfitted :class:`lightgbm.LGBMClassifier`.
        """
        return lgb.LGBMClassifier(**LGB_PARAMS)

    @staticmethod
    def _build_cat() -> CatBoostClassifier:
        """Instantiate a CatBoostClassifier with the canonical CPU params.

        Returns:
            Unfitted :class:`catboost.CatBoostClassifier`.
        """
        return CatBoostClassifier(**CAT_PARAMS)

    def _assert_trained(self) -> None:
        """Raise RuntimeError if any base model or meta-learner is not yet fitted.

        Raises:
            RuntimeError: If :meth:`train` has not been called.
        """
        if (
            self._xgb_model is None
            or self._lgb_model is None
            or self._cat_model is None
            or self._meta_lr is None
        ):
            raise RuntimeError(
                "MLEnsemble has not been trained. Call .train() first."
            )
