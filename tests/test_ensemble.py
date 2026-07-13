"""Stage 2 tests for the ML ensemble pipeline.

Covers:
- Module importability
- FrozenModelSpec immutability and schema hash consistency
- ModelRegistry retrain-safety logic
- CPU-only hyperparameter constraints for XGB / LGB / CatBoost
- Bar-type mismatch raises ModelMismatchError

Heavy training steps (XGBoost, LightGBM, CatBoost, Treelite) are either
skipped via pytest.importorskip or mocked with unittest.mock.patch.
"""
from __future__ import annotations

import importlib
from datetime import date, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_feature_matrix(
    n: int = 500,
    n_features: int = 25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, weights) with deterministic random data.

    Args:
        n: Number of samples.
        n_features: Number of feature columns.
        seed: NumPy random seed.

    Returns:
        Tuple of:
          - X: float64 array of shape (n, n_features).
          - y: int array of labels from {-1, 0, +1}.
          - weights: float64 array of positive sample weights.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float64)
    y = rng.choice([-1, 0, 1], size=n).astype(np.int32)
    weights = rng.uniform(0.5, 2.0, size=n).astype(np.float64)
    return X, y, weights


# ---------------------------------------------------------------------------
# Helper — build a minimal FrozenModelSpec without running training
# ---------------------------------------------------------------------------


def _make_frozen_spec(**overrides: Any):
    """Return a minimal FrozenModelSpec suitable for unit tests."""
    from algo_trading.models.model_registry import FrozenModelSpec

    feature_names: tuple[str, ...] = tuple(f"feat_{i}" for i in range(25))
    schema_hash = FrozenModelSpec.compute_feature_hash(feature_names)

    defaults: dict[str, Any] = dict(
        version="v0.0.1",
        feature_names=feature_names,
        feature_schema_hash=schema_hash,
        bar_type="time_15min",
        bar_spec={},
        trained_on=date(2026, 1, 1),
        symbol_universe=("RELIANCE",),
        validation_metrics={"oof_accuracy": 0.6},
        treelite_xgb_path="/tmp/xgb.so",
        treelite_lgb_path="/tmp/lgb.so",
        treelite_cat_path="/tmp/cat.so",
        meta_lr_path="/tmp/meta_lr.pkl",
        package_versions={"xgboost": "2.0.0"},
        approval_status="pending",
    )
    defaults.update(overrides)
    return FrozenModelSpec(**defaults)


# ===========================================================================
# 1. Module importability
# ===========================================================================


def test_ensemble_module_importable() -> None:
    """algo_trading.models.ensemble must import without raising."""
    # Heavy deps (xgb, lgb, catboost) are required at import time per the module.
    # Skip gracefully if any is absent.
    xgb = pytest.importorskip("xgboost", reason="xgboost not installed")
    lgb = pytest.importorskip("lightgbm", reason="lightgbm not installed")
    pytest.importorskip("catboost", reason="catboost not installed")

    module = importlib.import_module("algo_trading.models.ensemble")
    assert module is not None


# ===========================================================================
# 2. FrozenModelSpec immutability
# ===========================================================================


def test_frozen_model_spec_immutable() -> None:
    """Modifying any field of a FrozenModelSpec must raise FrozenInstanceError."""
    from dataclasses import FrozenInstanceError

    spec = _make_frozen_spec()
    with pytest.raises(FrozenInstanceError):
        spec.version = "v9.9.9"  # type: ignore[misc]


# ===========================================================================
# 3. Feature schema hash — consistency
# ===========================================================================


def test_feature_schema_hash_consistent() -> None:
    """The same ordered feature list must produce the same 8-char hash."""
    from algo_trading.models.model_registry import FrozenModelSpec

    features = tuple(f"feat_{i}" for i in range(25))
    h1 = FrozenModelSpec.compute_feature_hash(features)
    h2 = FrozenModelSpec.compute_feature_hash(features)
    assert h1 == h2
    assert len(h1) == 8


# ===========================================================================
# 4. Feature schema hash — different lists give different hashes
# ===========================================================================


def test_feature_schema_hash_different() -> None:
    """Different feature lists must produce different hashes."""
    from algo_trading.models.model_registry import FrozenModelSpec

    features_a = tuple(f"feat_{i}" for i in range(25))
    features_b = tuple(f"feature_{i}" for i in range(25))
    assert FrozenModelSpec.compute_feature_hash(features_a) != FrozenModelSpec.compute_feature_hash(
        features_b
    )


# ===========================================================================
# 5. model_registry importability
# ===========================================================================


def test_model_registry_importable() -> None:
    """FrozenModelSpec and ModelMismatchError must be importable from model_registry."""
    from algo_trading.models.model_registry import FrozenModelSpec, ModelMismatchError  # noqa: F401

    assert FrozenModelSpec is not None
    assert ModelMismatchError is not None


# ===========================================================================
# 6. FrozenModelSpec required fields
# ===========================================================================


def test_frozen_spec_has_required_fields() -> None:
    """FrozenModelSpec must expose the fields required by the spec contract."""
    spec = _make_frozen_spec()
    # Check every mandatory attribute exists and is not None / empty
    assert spec.version
    assert spec.feature_names
    assert spec.feature_schema_hash
    assert spec.trained_on is not None
    # treelite_so_path — spec uses treelite_xgb_path as the primary path
    assert spec.treelite_xgb_path
    assert spec.treelite_lgb_path
    assert spec.treelite_cat_path


# ===========================================================================
# 7. is_safe_to_retrain — market hours (weekday daytime) → False
# ===========================================================================


def test_is_safe_to_retrain_during_market_hours() -> None:
    """10:30 AM on a Wednesday is inside market hours → is_safe_to_retrain() == False."""
    from zoneinfo import ZoneInfo

    from algo_trading.models.model_registry import ModelRegistry

    IST = ZoneInfo("Asia/Kolkata")
    # Wednesday = weekday 2, 10:30 IST — market is open, not Sunday
    fake_now = datetime(2026, 6, 17, 10, 30, 0, tzinfo=IST)  # 2026-06-17 is a Wednesday

    registry = ModelRegistry.__new__(ModelRegistry)
    registry._dir = MagicMock()
    registry._dir.mkdir = MagicMock()

    with patch("algo_trading.models.model_registry.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        result = registry.is_safe_to_retrain()

    assert result is False


# ===========================================================================
# 8. is_safe_to_retrain — Sunday evening IST → True
# ===========================================================================


def test_is_safe_to_retrain_sunday_evening() -> None:
    """21:00 IST on Sunday → is_safe_to_retrain() == True."""
    from zoneinfo import ZoneInfo

    from algo_trading.models.model_registry import ModelRegistry

    IST = ZoneInfo("Asia/Kolkata")
    # 2026-06-21 is a Sunday
    fake_now = datetime(2026, 6, 21, 21, 0, 0, tzinfo=IST)

    registry = ModelRegistry.__new__(ModelRegistry)
    registry._dir = MagicMock()
    registry._dir.mkdir = MagicMock()

    with patch("algo_trading.models.model_registry.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        result = registry.is_safe_to_retrain()

    assert result is True


# ===========================================================================
# 9. validate_bar_type — mismatch raises ModelMismatchError
# ===========================================================================


def test_validate_bar_type_mismatch_raises() -> None:
    """TRAIN_BAR_TYPE != LIVE_BAR_TYPE must raise ModelMismatchError."""
    from algo_trading.models.model_registry import ModelMismatchError

    spec = _make_frozen_spec(bar_type="time_15min", bar_spec={})
    with pytest.raises(ModelMismatchError):
        spec.validate_bar_type("volume", {"volume_threshold": 50_000})


# ===========================================================================
# 10. XGB_PARAMS — CPU-only constraints
# ===========================================================================


def test_xgb_params_cpu_only() -> None:
    """XGB_PARAMS must specify tree_method='hist' and device='cpu'."""
    pytest.importorskip("xgboost", reason="xgboost not installed")
    pytest.importorskip("lightgbm", reason="lightgbm not installed")
    pytest.importorskip("catboost", reason="catboost not installed")

    from algo_trading.models.ensemble import XGB_PARAMS

    assert XGB_PARAMS["tree_method"] == "hist", "XGB must use hist (CPU)"
    assert XGB_PARAMS["device"] == "cpu", "XGB must be CPU-only"


# ===========================================================================
# 11. LGB_PARAMS — CPU-only constraints
# ===========================================================================


def test_lgb_params_cpu_only() -> None:
    """LGB_PARAMS must specify device='cpu'."""
    pytest.importorskip("xgboost", reason="xgboost not installed")
    pytest.importorskip("lightgbm", reason="lightgbm not installed")
    pytest.importorskip("catboost", reason="catboost not installed")

    from algo_trading.models.ensemble import LGB_PARAMS

    assert LGB_PARAMS["device"] == "cpu", "LGB must be CPU-only"


# ===========================================================================
# 12. CAT_PARAMS — CPU-only constraints
# ===========================================================================


def test_catboost_params_cpu_only() -> None:
    """CAT_PARAMS must specify task_type='CPU'."""
    pytest.importorskip("xgboost", reason="xgboost not installed")
    pytest.importorskip("lightgbm", reason="lightgbm not installed")
    pytest.importorskip("catboost", reason="catboost not installed")

    from algo_trading.models.ensemble import CAT_PARAMS

    assert CAT_PARAMS["task_type"] == "CPU", "CatBoost must be CPU-only"
